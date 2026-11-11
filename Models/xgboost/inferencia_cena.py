#!/usr/bin/env python3
"""XGBoost (ou Random Forest) sobre uma CENA INTEIRA, em faixas, retomável.

O vit_model/inferencia_cena.py só carrega modelos .pth (ViT ou CNN); para o
modelo.ubj do XGBoost e o modelo.joblib do rf_patches.py não havia caminho.
Este script fecha isso, reaproveitando as duas metades que já existiam:

  - o preparo das bandas é o MESMO do vit_model/inference.py: PAN esticada
    p2-p98 -> uint8 -> GLCM entropia (janela 7, 8 niveis), LocalSTD 7x7 e Sobel
    -> [PAN, GLCM, (z(STD)+z(Grad))/2] em z-score, com as estatisticas da cena
    lidas do receita.json do dataset do modelo. Trocar isso quebra o mapa SEM
    dar erro: o modelo recebe entrada noutra escala e classifica errado calado.
  - o laco de features na GPU + predicao por bloco e o mesmo do avaliar.py.

Diferenca em relacao ao caminho CNN: aqui nao ha janela deslizante nem soma de
probabilidades. O XGBoost decide pixel a pixel, entao cada pixel e previsto uma
vez so. O que exige sobreposicao e o CALCULO DAS FEATURES (janelas de ate 155
px), nao a predicao -- dai `--overlap`.

Saida em <run>/cena_<nome>/: faixa_<y>.tif por faixa, progresso.txt (retomada),
classes.tif (uniao por VRT) e info.json.

Uso (no WSL, a partir de `Models/`):
  ../Data/.venv/bin/python xgboost/inferencia_cena.py \\
      --modelo xgboost/runs/xgb_spot5_v18/modelo.ubj \\
      --imagem ../Data/dados/spot5/colinas_do_sul/original/Colinas_do_Sul_mosaico_v2.tif \\
      --janelas 5 15 31
  # teste parcial antes da cena toda:
  ... --ate 4096
  # so juntar as faixas ja feitas:
  ... --unir
"""

import argparse
import gc
import json
import os
import sys
import time

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch)
import numpy as np
import scipy.ndimage as ndi
import torch
from osgeo import gdal

import dados
import features

sys.path.insert(0, os.path.join(dados.DATA, "scripts", "wv1"))
from gerar_features_pan import features_block  # noqa: E402

gdal.UseExceptions()
BASE = os.path.dirname(os.path.abspath(__file__))
MARGEM = 8                 # px extras p/ as texturas (como no dataset)
N = dados.N_CLASSES


# ---------------------------------------------------------------- bandas
# Copias de vit_model/inference.py. O repositorio ja mantem ambiente.py e
# dados.py como copias por pasta de proposito; importar inference.py aqui
# arrastaria segmentacao.py e o timm junto, e faria `dados` resolver para a
# copia errada dependendo da ordem do sys.path.
def preencher_nodata(pan, valido):
    if valido.all():
        return pan
    idx = ndi.distance_transform_edt(~valido, return_distances=False, return_indices=True)
    return pan[tuple(idx)]


def texturas(pan, valido, p2, p98, janela, niveis):
    half = janela // 2
    pan = np.pad(preencher_nodata(pan, valido), MARGEM, mode="reflect")
    s = (np.clip((pan - p2) / (p98 - p2), 0, 1) * 255 + 0.5).astype(np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, std, ent, grad = features_block(s, s.shape[0] - 2 * half, janela, niveis)
    H, W = pan.shape[0] - 2 * MARGEM, pan.shape[1] - 2 * MARGEM
    corta = (slice(MARGEM - half, MARGEM - half + H), slice(MARGEM, MARGEM + W))
    return pan[MARGEM:MARGEM + H, MARGEM:MARGEM + W], ent[corta], std[corta], grad[corta]


def bandas(pan, valido, st, janela, niveis):
    pan, ent, std, grad = texturas(pan, valido, st["p2"], st["p98"], janela, niveis)
    z = lambda a, k: (a - st["media"][k]) / st["desvio"][k]  # noqa: E731
    rug = (z(std, "std") + z(grad, "grad")) / 2
    return np.stack([z(pan, "pan"), z(ent, "glcm"), z(rug, "rugosidade")]).astype(np.float32)


# ---------------------------------------------------------------- saida
def gravar_faixa(caminho, codigos, y0, gt, proj, W):
    h = codigos.shape[0]
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(caminho, W, h, 1, gdal.GDT_Byte,
                    ["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"])
    g = list(gt)
    g[3] = gt[3] + y0 * gt[5]
    ds.SetGeoTransform(g)
    ds.SetProjection(proj)
    b = ds.GetRasterBand(1)
    b.WriteArray(codigos)
    b.SetNoDataValue(0)
    ds = None


def unir_faixas(cls_caminho, pasta, W, H, gt, proj):
    faixas = sorted(f for f in os.listdir(pasta) if f.startswith("faixa_") and f.endswith(".tif"))
    if not faixas:
        return None
    vrt = os.path.join(pasta, "faixas.vrt")
    gdal.BuildVRT(vrt, [os.path.join(pasta, f) for f in faixas],
                  options=gdal.BuildVRTOptions(resolution="highest"))
    gdal.Translate(cls_caminho, vrt,
                   options=gdal.TranslateOptions(
                       creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",
                                        "BIGTIFF=IF_SAFER"]))
    return len(faixas)


# ---------------------------------------------------------------- predicao
@torch.no_grad()
def prever_faixa(predizer, usa_features, X, janelas, margem, bloco_col, dev):
    """-> (h, W) uint8 com o indice de classe de CADA pixel da faixa.

    A faixa inteira nao cabe na VRAM com 21 features (uma faixa de 1624 x 29068
    daria 3,9 GB so de features), entao ela e varrida em blocos de colunas com
    `margem` px de sobreposicao de cada lado -- o raio da maior janela, para a
    feature de cada pixel entregue ser calculada com a vizinhanca completa.
    """
    _, h, W = X.shape
    out = np.empty((h, W), np.uint8)
    for x0 in range(0, W, bloco_col):
        x1 = min(W, x0 + bloco_col)
        a = max(0, x0 - margem)
        b = min(W, x1 + margem)
        sub = torch.from_numpy(np.ascontiguousarray(X[:, :, a:b])).to(dev)[None]
        f = sub if not usa_features else features.calcular(sub, janelas)
        f = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).cpu().numpy()
        pred = predizer(f).reshape(sub.shape[2], sub.shape[3])
        out[:, x0:x1] = pred[:, x0 - a:x0 - a + (x1 - x0)]
        del sub, f
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True, help="modelo.ubj (XGBoost) ou modelo.joblib (RF)")
    ap.add_argument("--imagem", required=True, help="GeoTIFF PAN (1 banda) da cena inteira")
    ap.add_argument("--dataset", default=None,
                    help="dataset cujo receita.json tem as estatisticas (padrao: o do relatorio "
                         "do run)")
    ap.add_argument("--cena", default=None,
                    help="cena do receita.json cujas estatisticas usar (padrao: pelo caminho "
                         "da imagem)")
    ap.add_argument("--texturas", default=None,
                    help="sufixo dos GeoTIFFs de textura ao lado do PAN, p.ex. '_v18' usa "
                         "<pan>_glcm_v18.tif e <pan>_rugosidade_v18.tif em vez de recalcular "
                         "(74%% do tempo por faixa). ATENCAO: so sirvam se tiverem sido geradas "
                         "com as estatisticas DESTE dataset -- as de outra versao parecem certas "
                         "e nao sao (medido: GLCM do v15 contra patches do v18 da diferenca "
                         "maxima 2,11; com as do v18, 2e-6). Confira antes.")
    ap.add_argument("--janelas", type=int, nargs="+", default=None,
                    help="janelas das features (padrao: as do relatorio do run)")
    ap.add_argument("--saida", default=None, help="pasta de saida")
    ap.add_argument("--faixa", type=int, default=1024, help="linhas por faixa")
    ap.add_argument("--overlap", type=int, default=300,
                    help="px extras em cada lado da faixa (texturas + janelas das features)")
    ap.add_argument("--bloco-col", type=int, default=2048,
                    help="colunas por bloco dentro da faixa; e ele que define o pico de VRAM")
    ap.add_argument("--ate", type=int, default=0,
                    help="processa so ate esta linha (0 = cena inteira; util p/ teste parcial)")
    ap.add_argument("--unir", action="store_true",
                    help="so junta as faixas ja existentes em classes.tif (nao infere)")
    args = ap.parse_args()
    t_inicio = time.monotonic()

    run = os.path.dirname(os.path.abspath(args.modelo))
    nome_run = os.path.basename(run)
    rel = os.path.join(run, "relatorio.json")
    cfg = json.load(open(rel, encoding="utf-8")) if os.path.exists(rel) else {}
    dataset = args.dataset or cfg.get("dataset") or cfg.get("args", {}).get("dataset")
    if not dataset:
        raise SystemExit("sem --dataset e o relatorio do run nao diz qual e")
    if not os.path.isabs(dataset):
        dataset = os.path.normpath(os.path.join(ambiente.MODELS, dataset))

    global N
    dados.configurar(dataset)
    N = dados.N_CLASSES
    janelas = args.janelas or cfg.get("args", {}).get("janelas") or [5, 15, 31]
    margem = max(janelas) // 2
    print("[classes] %d: %s" % (N, ", ".join(dados.CLASS_NOMES)))

    dev = ambiente.exigir_gpu()
    if args.modelo.endswith(".ubj"):
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(args.modelo)
        bst.set_param({"device": "cuda"})
        usa_features = True
        def predizer(Z):
            return bst.inplace_predict(Z).argmax(1).astype(np.uint8)
        print("[modelo] %s | XGBoost | %d arvores | janelas %s | margem %d"
              % (nome_run, bst.num_boosted_rounds(), janelas, margem), flush=True)
    else:
        import joblib
        rf = joblib.load(args.modelo)
        usa_features = False
        def predizer(Z):
            return rf.predict(Z).astype(np.uint8)
        print("[modelo] %s | RandomForest | %d arvores | 3 bandas cruas"
              % (nome_run, len(rf.estimators_)), flush=True)

    receita = json.load(open(os.path.join(dataset, "receita.json"), encoding="utf-8"))
    par = receita["parametros"]
    ds = gdal.Open(args.imagem)
    W, H = ds.RasterXSize, ds.RasterYSize
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    banda = ds.GetRasterBand(1)
    nd = banda.GetNoDataValue()

    tex = None
    if args.texturas is not None:
        stem = os.path.splitext(args.imagem)[0]
        cams = [stem + "_glcm" + args.texturas + ".tif",
                stem + "_rugosidade" + args.texturas + ".tif"]
        falta = [c for c in cams if not os.path.exists(c)]
        if falta:
            raise SystemExit("--texturas %s: nao achei %s" % (args.texturas, ", ".join(falta)))
        tex_ds = [gdal.Open(c) for c in cams]
        for d, c in zip(tex_ds, cams):
            if (d.RasterXSize, d.RasterYSize) != (W, H):
                raise SystemExit("%s tem %dx%d, o PAN tem %dx%d"
                                 % (c, d.RasterXSize, d.RasterYSize, W, H))
        tex = [d.GetRasterBand(1) for d in tex_ds]
        # O gerar_features_cena.py --byte grava Byte com offset/escala no
        # proprio GeoTIFF (1..255 = valor, 0 = sem imagem); sem --byte grava
        # Float32 cru. Ler sem aplicar offset/escala daria numero errado SEM
        # erro nenhum, entao a conversao vem do metadado, nao de constante.
        tex_oe = [(b.GetOffset() or 0.0, b.GetScale() or 1.0) for b in tex]
        print("[texturas] lendo do disco: %s"
              % ", ".join("%s (%s%s)" % (os.path.basename(c), gdal.GetDataTypeName(b.DataType),
                                         "" if sc == 1.0 and off == 0.0
                                         else ", x%g%+g" % (sc, off))
                          for c, b, (off, sc) in zip(cams, tex, tex_oe)), flush=True)

    nome_cena = args.cena
    if nome_cena is None:
        cams = {c: v for c, v in receita["resultado"]["por_cena"].items()}
        for c in cams:
            if c in os.path.abspath(args.imagem).replace("\\", "/"):
                nome_cena = c
                break
    if nome_cena is None:
        raise SystemExit("nao identifiquei a cena pelo caminho; use --cena (opcoes: %s)"
                         % ", ".join(receita["resultado"]["por_cena"]))
    st = receita["resultado"]["por_cena"][nome_cena]["estatisticas"]
    print("[cena] %s | %dx%d = %.1f Mpx | estatisticas do receita (p2=%.1f p98=%.1f)"
          % (nome_cena, W, H, W * H / 1e6, st["p2"], st["p98"]), flush=True)

    saida = args.saida or os.path.join(run, "cena_%s" % nome_cena)
    os.makedirs(saida, exist_ok=True)
    cls_caminho = os.path.join(saida, "classes.tif")
    prog = os.path.join(saida, "progresso.txt")

    if args.unir:
        n = unir_faixas(cls_caminho, saida, W, H, gt, proj)
        print("unidas %s faixas -> %s" % (n, cls_caminho))
        return

    feitas = set()
    if os.path.exists(prog):
        feitas = {int(l) for l in open(prog) if l.strip()}
        print("[retomada] %d faixas ja feitas" % len(feitas), flush=True)

    H_alvo = min(H, args.ate) if args.ate else H
    n_faixas, t_pred, px_feitos = 0, 0.0, 0
    total = -(-H_alvo // args.faixa)
    for y0 in range(0, H_alvo, args.faixa):
        if y0 in feitas:
            continue
        y1 = min(H, y0 + args.faixa)
        ya = max(0, y0 - args.overlap)
        yb = min(H, y1 + args.overlap)
        pan = banda.ReadAsArray(0, ya, W, yb - ya).astype(np.float32)
        ini = y0 - ya
        valido = pan != (nd if nd is not None else 0) if (nd is not None or par.get("pan_zero_nodata")) \
            else np.ones_like(pan, bool)
        n_faixas += 1
        if not valido[ini:ini + (y1 - y0)].any():
            with open(prog, "a") as f:
                f.write("%d\n" % y0)
            print("  faixa %3d/%d  y=%d..%d | sem dado, pulada" % (n_faixas, total, y0, y1),
                  flush=True)
            del pan, valido
            continue
        if tex is None:
            X = bandas(pan, valido, st, par["janela"], par["niveis"])
        else:
            # Banda 0 do dataset e a PAN CRUA em z-score (nao a esticada); as
            # outras duas ja vem em z-score dos proprios GeoTIFFs.
            zp = (pan - st["media"]["pan"]) / st["desvio"]["pan"]
            X = np.stack([zp] + [b.ReadAsArray(0, ya, W, yb - ya).astype(np.float32) * sc + off
                                 for b, (off, sc) in zip(tex, tex_oe)])
        X[:, ~valido] = 0
        t = time.monotonic()
        cls = prever_faixa(predizer, usa_features, X, janelas, margem, args.bloco_col, dev)
        t_pred += time.monotonic() - t
        cls = cls[ini:ini + (y1 - y0)]
        codigos = np.asarray(dados.REPR_CODIGO, np.uint8)[cls]
        codigos[~valido[ini:ini + (y1 - y0)]] = 0
        gravar_faixa(os.path.join(saida, "faixa_%06d.tif" % y0), codigos, y0, gt, proj, W)
        with open(prog, "a") as f:
            f.write("%d\n" % y0)
        px_feitos += int(valido[ini:ini + (y1 - y0)].sum())
        print("  faixa %3d/%d  y=%d..%d | %.0fs acumulados | %.0f kpx/s"
              % (n_faixas, total, y0, y1, time.monotonic() - t_inicio,
                 px_feitos / max(t_pred, 1e-9) / 1e3), flush=True)
        del pan, valido, X, cls, codigos
        gc.collect()
        torch.cuda.empty_cache()
    ds = banda = None

    n_unidas = unir_faixas(cls_caminho, saida, W, H, gt, proj)
    if n_unidas is None:
        print("nenhuma faixa_*.tif para unir")
        return
    with open(os.path.join(saida, "info.json"), "w", encoding="utf-8") as f:
        json.dump({"run": nome_run, "modelo": os.path.abspath(args.modelo),
                   "imagem": os.path.abspath(args.imagem), "cena": nome_cena,
                   "dataset": dataset, "janelas": janelas, "margem": margem,
                   "classes": dados.CLASS_NOMES, "repr_codigo": dados.REPR_CODIGO,
                   "estatisticas": st, "faixas": n_unidas,
                   "px_classificados": px_feitos,
                   "tempo_predicao_s": round(t_pred, 1),
                   "tempo_total_s": round(time.monotonic() - t_inicio, 1)},
                  f, indent=2, ensure_ascii=False)
    print("\n-> %s (%d faixas, %d px, %.0f kpx/s)"
          % (cls_caminho, n_unidas, px_feitos, px_feitos / max(t_pred, 1e-9) / 1e3), flush=True)


if __name__ == "__main__":
    main()
