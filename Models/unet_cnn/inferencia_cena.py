#!/usr/bin/env python3
"""U-Net / U-Net++ sobre uma CENA INTEIRA, em faixas, retomável.

Por que existe, se o vit_model/inferencia_cena.py já roda modelos CNN: aquele
script nasceu para o ViT e foi estendido depois, então um modelo da familia CNN
é rodado por um arquivo que mora na pasta do ViT e importa as classes UNet de
volta daqui. Este é o mesmo trabalho sem esse desvio -- CNN apenas, na pasta da
CNN, como o xgboost/inferencia_cena.py é do XGBoost. O do vit_model continua
intacto para quando o ViT voltar a ser usado.

O preparo das bandas é o MESMO do vit_model/inference.py: PAN esticada p2-p98
-> uint8 -> GLCM entropia (janela 7, 8 niveis), LocalSTD 7x7 e Sobel ->
[PAN, GLCM, (z(STD)+z(Grad))/2] em z-score, com as estatisticas da cena lidas
do receita.json do dataset do modelo. Trocar isso quebra o mapa SEM dar erro: o
modelo recebe entrada noutra escala e classifica errado calado.

`--texturas <sufixo>` lê GLCM e rugosidade de GeoTIFFs prontos ao lado do PAN
em vez de recalcular, o que no XGBoost derrubou a cena de colinas de 2 h para
28 min (o recalculo na CPU era 74% do tempo). Só servem se tiverem sido geradas
com as estatisticas DESTE dataset -- ver gerar_features_cena.py --receita.

Predição: janelas de 256 px com passo `--passo`, ponderadas com peso maior no
centro (a borda do patch é onde a rede erra), somando probabilidades antes do
argmax. Com o padrão 128 cada pixel é previsto 4 vezes.

Saída em runs/<nome>/cena_<cena>/: faixa_<y>.tif, progresso.txt (retomada),
classes.tif (união por VRT) e info.json.

Uso (no WSL, a partir de `Models/`):
  ../Data/.venv/bin/python unet_cnn/inferencia_cena.py \\
      --modelo unet_cnn/runs/unetpp_v18/modelo_melhor.pth \\
      --imagem ../Data/dados/spot5/colinas_do_sul/original/Colinas_do_Sul_mosaico_v2.tif \\
      --texturas _v18
  ... --ate 4096     # teste parcial
  ... --unir         # só junta as faixas já feitas
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
from unet import UNet, UNetPP

sys.path.insert(0, os.path.join(dados.DATA, "scripts", "wv1"))
from gerar_features_pan import features_block  # noqa: E402

gdal.UseExceptions()
BASE = os.path.dirname(os.path.abspath(__file__))
MARGEM = 8                 # px extras p/ as texturas (como no dataset)
P = 256                    # lado da janela de predição


# ---------------------------------------------------------------- bandas
# Cópias de vit_model/inference.py. O repositório já mantém ambiente.py e
# dados.py como cópias por pasta de propósito; importar inference.py aqui
# arrastaria segmentacao.py e o timm junto, e faria `dados` resolver para a
# cópia errada dependendo da ordem do sys.path.
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


# ---------------------------------------------------------------- predição
def peso_janela(lado, margem):
    """1 no interior, rampa linear até 0,05 na borda (a borda do patch erra mais)."""
    r = np.minimum(np.arange(lado), np.arange(lado)[::-1]).astype(np.float32)
    r = np.clip((r + 1) / margem, 0.05, 1.0)
    return torch.from_numpy(np.outer(r, r))


def posicoes(L, lado, passo):
    if L <= lado:
        return [0]
    p = list(range(0, L - lado + 1, passo))
    return p if p[-1] == L - lado else p + [L - lado]


@torch.no_grad()
def prever(modelo, X, passo, lote, dev, margem, bloco_col=0):
    """X (C,H,W) -> probabilidades (n_cls,H,W) em janelas sobrepostas.

    `bloco_col` > 0 varre a faixa em blocos de colunas, com P px de margem de
    cada lado, e monta a saida no numpy. Sem isso o acumulador de
    probabilidades mora inteiro na GPU: numa cena de 274.176 colunas e faixa de
    1024+2*128 linhas sao 3*1280*274176*4 = 4,2 GB so dele, mais o peso -- nao
    cabe em 6 GB. Com bloco de 8192 colunas o pico cai para ~130 MB.
    """
    _, H, W = X.shape
    if bloco_col and W > bloco_col:
        n_cls = None
        out = None
        for x0 in range(0, W, bloco_col):
            x1 = min(W, x0 + bloco_col)
            a, b = max(0, x0 - P), min(W, x1 + P)
            sub_p, _ = prever(modelo, X[:, :, a:b], passo, lote, dev, margem)
            if out is None:
                n_cls = sub_p.shape[0]
                out = np.empty((n_cls, H, W), np.float32)
            out[:, :, x0:x1] = sub_p[:, :, x0 - a:x0 - a + (x1 - x0)]
            del sub_p
            torch.cuda.empty_cache()
        return out, 0
    ph, pw = max(0, P - H), max(0, P - W)
    if ph or pw:
        X = np.pad(X, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    Xt = torch.from_numpy(X)
    Hp, Wp = Xt.shape[1:]
    w = peso_janela(P, margem).to(dev)
    soma, peso = None, torch.zeros(Hp, Wp, device=dev)
    jan = [(y, x) for y in posicoes(Hp, P, passo) for x in posicoes(Wp, P, passo)]
    for i in range(0, len(jan), lote):
        bloco = jan[i:i + lote]
        xb = torch.stack([Xt[:, y:y + P, x:x + P] for y, x in bloco]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = modelo(xb).float().softmax(1)
        if soma is None:
            soma = torch.zeros(pr.shape[1], Hp, Wp, device=dev)
        for (y, x), p in zip(bloco, pr):
            soma[:, y:y + P, x:x + P] += p * w
            peso[y:y + P, x:x + P] += w
        del xb, pr
    out = (soma / peso).cpu().numpy()[:, :H, :W]
    del soma, peso
    return out, len(jan)


# ---------------------------------------------------------------- saída
def gravar_bloco(caminho, codigos, x0, y0, gt, proj):
    """Grava um retangulo em (x0,y0) com o geotransform deslocado."""
    h, w = codigos.shape
    ds = gdal.GetDriverByName("GTiff").Create(
        caminho, w, h, 1, gdal.GDT_Byte,
        ["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"])
    g = list(gt)
    g[0] = gt[0] + x0 * gt[1]
    g[3] = gt[3] + y0 * gt[5]
    ds.SetGeoTransform(g)
    ds.SetProjection(proj)
    b = ds.GetRasterBand(1)
    b.WriteArray(codigos)
    b.SetNoDataValue(0)
    ds = None


def gravar_faixa(caminho, codigos, y0, gt, proj, W):
    h = codigos.shape[0]
    ds = gdal.GetDriverByName("GTiff").Create(
        caminho, W, h, 1, gdal.GDT_Byte,
        ["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"])
    g = list(gt)
    g[3] = gt[3] + y0 * gt[5]
    ds.SetGeoTransform(g)
    ds.SetProjection(proj)
    b = ds.GetRasterBand(1)
    b.WriteArray(codigos)
    b.SetNoDataValue(0)
    ds = None


def unir_faixas(cls_caminho, pasta):
    faixas = sorted(f for f in os.listdir(pasta)
                    if (f.startswith("faixa_") or f.startswith("bloco_"))
                    and f.endswith(".tif"))
    if not faixas:
        return None
    vrt = os.path.join(pasta, "faixas.vrt")
    gdal.BuildVRT(vrt, [os.path.join(pasta, f) for f in faixas],
                  options=gdal.BuildVRTOptions(resolution="highest"))
    gdal.Translate(cls_caminho, vrt, options=gdal.TranslateOptions(
        creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",
                         "BIGTIFF=IF_SAFER"]))
    return len(faixas)


def carregar_modelo(caminho, dev, n_classes):
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    arq = a.get("arquitetura", "unetpp")
    base = a.get("base", 32)
    sd = ck["state_dict"]
    canais = next(v for k, v in sd.items() if k.endswith("weight") and v.ndim == 4).shape[1]
    modelo = (UNet if arq == "unet" else UNetPP)(
        in_chans=canais, n_classes=n_classes, base=base).to(dev)
    modelo.load_state_dict(sd)
    modelo.eval()
    return modelo, ck, arq, base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True, help="modelo_melhor.pth de um run unet*")
    ap.add_argument("--imagem", required=True, help="GeoTIFF PAN (1 banda) da cena inteira")
    ap.add_argument("--dataset", default=None,
                    help="dataset cujo receita.json tem as estatisticas (padrao: o gravado "
                         "no checkpoint)")
    ap.add_argument("--cena", default=None,
                    help="cena do receita.json cujas estatisticas usar (padrao: pelo caminho "
                         "da imagem)")
    ap.add_argument("--texturas", default=None,
                    help="sufixo dos GeoTIFFs de textura ao lado do PAN, p.ex. '_v18' usa "
                         "<pan>_glcm_v18.tif e <pan>_rugosidade_v18.tif em vez de recalcular. "
                         "ATENCAO: so servem se geradas com as estatisticas DESTE dataset; "
                         "as de outra versao parecem certas e nao sao")
    ap.add_argument("--saida", default=None, help="pasta de saida")
    ap.add_argument("--faixa", type=int, default=1024, help="linhas por faixa")
    ap.add_argument("--overlap", type=int, default=300,
                    help="px extras em cada lado da faixa (texturas + janelas)")
    ap.add_argument("--passo", type=int, default=128, help="px entre janelas de 256")
    ap.add_argument("--ladrilho", type=int, default=0,
                    help="lado do ladrilho 2D em px (0 = faixas de largura inteira). "
                         "Obrigatorio em cenas muito largas: uma faixa de 768 linhas numa "
                         "cena de 274.176 colunas carrega 5,8 GB (X + PAN + probabilidades) "
                         "e trava a maquina; com ladrilho 4096 sao ~400 MB. Tambem evita "
                         "reler as colunas na sobreposicao e descarta area sem imagem cedo")
    ap.add_argument("--bloco-col", type=int, default=0,
                    help="colunas por bloco dentro da faixa (0 = faixa inteira). Necessario "
                         "em cenas muito largas: o acumulador de probabilidades mora na GPU "
                         "e cresce com a largura")
    ap.add_argument("--lote", type=int, default=16, help="janelas por lote na GPU")
    ap.add_argument("--ate", type=int, default=0,
                    help="processa so ate esta linha (0 = cena inteira)")
    ap.add_argument("--unir", action="store_true",
                    help="so junta as faixas ja existentes em classes.tif (nao infere)")
    args = ap.parse_args()
    t_inicio = time.monotonic()

    run = os.path.dirname(os.path.abspath(args.modelo))
    nome_run = os.path.basename(run)
    ck_args = torch.load(args.modelo, map_location="cpu", weights_only=False).get("args", {})
    dataset = args.dataset or ck_args.get("dataset")
    if not dataset:
        raise SystemExit("sem --dataset e o checkpoint nao diz qual e")
    if not os.path.isabs(dataset):
        dataset = os.path.normpath(os.path.join(ambiente.MODELS, dataset))

    dados.configurar(dataset)          # REPR_CODIGO/CLASS_NOMES seguem o modelo
    N = dados.N_CLASSES
    print("[classes] %d: %s" % (N, ", ".join(dados.CLASS_NOMES)))

    dev = ambiente.exigir_gpu()
    modelo, ck, arq, base = carregar_modelo(args.modelo, dev, N)
    margem_peso = ck_args.get("margem", 15)
    print("[modelo] %s | %s base=%d | epoca %s (val mIoU=%.4f) | passo %d"
          % (nome_run, arq, base, ck.get("epoca", "?"),
             ck.get("val_miou", float("nan")), args.passo), flush=True)

    receita = json.load(open(os.path.join(dataset, "receita.json"), encoding="utf-8"))
    par = receita["parametros"]
    ds = gdal.Open(args.imagem)
    W, H = ds.RasterXSize, ds.RasterYSize
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    banda = ds.GetRasterBand(1)
    nd = banda.GetNoDataValue()

    nome_cena = args.cena
    if nome_cena is None:
        for c in receita["resultado"]["por_cena"]:
            if c in os.path.abspath(args.imagem).replace("\\", "/"):
                nome_cena = c
                break
    if nome_cena is None:
        raise SystemExit("nao identifiquei a cena pelo caminho; use --cena (opcoes: %s)"
                         % ", ".join(receita["resultado"]["por_cena"]))
    st = receita["resultado"]["por_cena"][nome_cena]["estatisticas"]
    print("[cena] %s | %dx%d = %.1f Mpx | estatisticas do receita (p2=%.1f p98=%.1f)"
          % (nome_cena, W, H, W * H / 1e6, st["p2"], st["p98"]), flush=True)

    tex, tex_oe = None, None
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
        # --byte grava Byte com offset/escala no metadado; sem --byte, Float32.
        tex_oe = [(b.GetOffset() or 0.0, b.GetScale() or 1.0) for b in tex]
        print("[texturas] lendo do disco: %s"
              % ", ".join("%s (%s)" % (os.path.basename(c), gdal.GetDataTypeName(b.DataType))
                          for c, b in zip(cams, tex)), flush=True)

    saida = args.saida or os.path.join(run, "cena_%s" % nome_cena)
    os.makedirs(saida, exist_ok=True)
    cls_caminho = os.path.join(saida, "classes.tif")
    prog = os.path.join(saida, "progresso.txt")

    if args.unir:
        n = unir_faixas(cls_caminho, saida)
        print("unidas %s faixas -> %s" % (n, cls_caminho))
        return

    feitas = set()
    if os.path.exists(prog):
        feitas = {int(l) for l in open(prog) if l.strip()}
        print("[retomada] %d faixas ja feitas" % len(feitas), flush=True)

    H_alvo = min(H, args.ate) if args.ate else H
    n_faixas, t_pred, px_feitos = 0, 0.0, 0

    def ler(banda_, x0, y0, w, h):
        return banda_.ReadAsArray(x0, y0, w, h)

    if args.ladrilho:
        L, OV = args.ladrilho, args.overlap
        xs = list(range(0, W, L))
        ys = list(range(0, H_alvo, L))
        total = len(xs) * len(ys)
        print("[ladrilhos] %d x %d = %d de %d px (+%d de margem)"
              % (len(xs), len(ys), total, L, OV), flush=True)
        for y0 in ys:
            for x0 in xs:
                chave = y0 * 10 ** 7 + x0
                if chave in feitas:
                    continue
                x1, y1 = min(W, x0 + L), min(H, y0 + L)
                xa, xb = max(0, x0 - OV), min(W, x1 + OV)
                ya, yb = max(0, y0 - OV), min(H, y1 + OV)
                pan = ler(banda, xa, ya, xb - xa, yb - ya).astype(np.float32)
                iy, ix = y0 - ya, x0 - xa
                valido = pan != (nd if nd is not None else 0)                     if (nd is not None or par.get("pan_zero_nodata")) else np.ones_like(pan, bool)
                n_faixas += 1
                miolo = valido[iy:iy + (y1 - y0), ix:ix + (x1 - x0)]
                if not miolo.any():
                    with open(prog, "a") as f:
                        f.write("%d\n" % chave)
                    del pan, valido
                    if n_faixas % 50 == 0:
                        print("  ladrilho %5d/%d  x=%d y=%d | sem dado"
                              % (n_faixas, total, x0, y0), flush=True)
                    continue
                if tex is None:
                    X = bandas(pan, valido, st, par["janela"], par["niveis"])
                else:
                    zp = (pan - st["media"]["pan"]) / st["desvio"]["pan"]
                    X = np.stack([zp] + [bb.ReadAsArray(xa, ya, xb - xa, yb - ya)
                                         .astype(np.float32) * sc + off
                                         for bb, (off, sc) in zip(tex, tex_oe)])
                X[:, ~valido] = 0
                t = time.monotonic()
                probs, _ = prever(modelo, X, args.passo, args.lote, dev, margem_peso)
                t_pred += time.monotonic() - t
                cls = probs.argmax(0)[iy:iy + (y1 - y0), ix:ix + (x1 - x0)]
                codigos = np.asarray(dados.REPR_CODIGO, np.uint8)[cls]
                codigos[~miolo] = 0
                gravar_bloco(os.path.join(saida, "bloco_%06d_%06d.tif" % (y0, x0)),
                             codigos, x0, y0, gt, proj)
                with open(prog, "a") as f:
                    f.write("%d\n" % chave)
                px_feitos += int(miolo.sum())
                print("  ladrilho %5d/%d  x=%d y=%d | %.0fs | %.0f kpx/s"
                      % (n_faixas, total, x0, y0, time.monotonic() - t_inicio,
                         px_feitos / max(t_pred, 1e-9) / 1e3), flush=True)
                del pan, valido, X, probs, cls, codigos, miolo
                gc.collect()
                torch.cuda.empty_cache()
        ds = banda = None
        n_unidas = unir_faixas(cls_caminho, saida)
        print("\n-> %s (%s blocos, %d px, %.0f kpx/s)"
              % (cls_caminho, n_unidas, px_feitos, px_feitos / max(t_pred, 1e-9) / 1e3),
              flush=True)
        return

    total = -(-H_alvo // args.faixa)
    for y0 in range(0, H_alvo, args.faixa):
        if y0 in feitas:
            continue
        y1 = min(H, y0 + args.faixa)
        ya, yb = max(0, y0 - args.overlap), min(H, y1 + args.overlap)
        pan = banda.ReadAsArray(0, ya, W, yb - ya).astype(np.float32)
        ini = y0 - ya
        valido = pan != (nd if nd is not None else 0) \
            if (nd is not None or par.get("pan_zero_nodata")) else np.ones_like(pan, bool)
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
            zp = (pan - st["media"]["pan"]) / st["desvio"]["pan"]
            X = np.stack([zp] + [b.ReadAsArray(0, ya, W, yb - ya).astype(np.float32) * sc + off
                                 for b, (off, sc) in zip(tex, tex_oe)])
        X[:, ~valido] = 0
        t = time.monotonic()
        probs, _ = prever(modelo, X, args.passo, args.lote, dev, margem_peso,
                          args.bloco_col)
        t_pred += time.monotonic() - t
        cls = probs.argmax(0)[ini:ini + (y1 - y0)]
        codigos = np.asarray(dados.REPR_CODIGO, np.uint8)[cls]
        codigos[~valido[ini:ini + (y1 - y0)]] = 0
        gravar_faixa(os.path.join(saida, "faixa_%06d.tif" % y0), codigos, y0, gt, proj, W)
        with open(prog, "a") as f:
            f.write("%d\n" % y0)
        px_feitos += int(valido[ini:ini + (y1 - y0)].sum())
        print("  faixa %3d/%d  y=%d..%d | %.0fs acumulados | %.0f kpx/s"
              % (n_faixas, total, y0, y1, time.monotonic() - t_inicio,
                 px_feitos / max(t_pred, 1e-9) / 1e3), flush=True)
        del pan, valido, X, probs, cls, codigos
        gc.collect()
        torch.cuda.empty_cache()
    ds = banda = None

    n_unidas = unir_faixas(cls_caminho, saida)
    if n_unidas is None:
        print("nenhuma faixa_*.tif para unir")
        return
    with open(os.path.join(saida, "info.json"), "w", encoding="utf-8") as f:
        json.dump({"run": nome_run, "modelo": os.path.abspath(args.modelo),
                   "arquitetura": arq, "base": base, "epoca": ck.get("epoca"),
                   "val_miou": ck.get("val_miou"),
                   "imagem": os.path.abspath(args.imagem), "cena": nome_cena,
                   "dataset": dataset, "passo": args.passo, "margem_peso": margem_peso,
                   "texturas": args.texturas, "classes": dados.CLASS_NOMES,
                   "repr_codigo": dados.REPR_CODIGO, "estatisticas": st,
                   "faixas": n_unidas, "px_classificados": px_feitos,
                   "tempo_predicao_s": round(t_pred, 1),
                   "tempo_total_s": round(time.monotonic() - t_inicio, 1)},
                  f, indent=2, ensure_ascii=False)
    print("\n-> %s (%d faixas, %d px, %.0f kpx/s)"
          % (cls_caminho, n_unidas, px_feitos, px_feitos / max(t_pred, 1e-9) / 1e3), flush=True)


if __name__ == "__main__":
    main()
