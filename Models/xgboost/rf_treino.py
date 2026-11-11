#!/usr/bin/env python3
"""Treina o Random Forest do RF_macro.py e mede a predição na CPU.

Serve de linha de base para `rf_gpu.py`: mesmos hiperparâmetros do script
original (500 árvores, profundidade livre), mesma leitura por blocos, mesma
regra de pixels válidos. A diferença é que aqui as amostras vêm de um raster
de rótulos já pronto (a máscara do shapefile) em vez de um GeoJSON, e dá para
limitar quantos pixels entram no treino com --max-amostras.

Esse limite não existe no RF_macro.py: lá todo pixel sob polígono vira uma
linha. Com polígonos generosos isso são dezenas de milhões de linhas, e 500
árvores de profundidade livre sobre isso não cabem em memória. O --max-amostras
sorteia com cota igual por classe, como faz o xgboost/train.py do projeto.

A divisão treino/validação é por BLOCOS espaciais, não por pixel: pixels
vizinhos de um mesmo polígono são quase idênticos e, sorteados ao acaso,
vazariam o treino para a validação e inflariam a acurácia.

Uso (no WSL, a partir de `Models/`):
  ../Data/.venv/bin/python xgboost/rf_treino.py --nome rf_colinas \\
      --imagem ../Data/dados/spot5/colinas_do_sul/original/colinas_3b.vrt \\
      --rotulos ../Data/dados/spot5/colinas_do_sul/mascara_amostra_v1.tif \\
      --max-amostras 300000
"""

import argparse
import json
import os
import time

import ambiente
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

NODATA_ENTRADA = -9999
BLOCO_SPLIT = 4096          # lado do bloco espacial treino/validação


def janelas(largura, altura, bloco):
    for y in range(0, altura, bloco):
        ny = min(bloco, altura - y)
        for x in range(0, largura, bloco):
            yield x, y, min(bloco, largura - x), ny


def ler_bloco(ds, x, y, nx, ny, mascara):
    dados = ds.ReadAsArray(x, y, nx, ny)
    if dados.ndim == 2:
        dados = dados[np.newaxis]
    dados = dados.astype(np.float32, copy=False)
    validos = np.all(np.isfinite(dados), axis=0)
    validos &= np.all(dados != NODATA_ENTRADA, axis=0)
    for i in range(ds.RasterCount):
        nd = ds.GetRasterBand(i + 1).GetNoDataValue()
        if nd is not None and not np.isnan(nd):
            validos &= dados[i] != nd
    if mascara is not None:
        validos &= mascara.ReadAsArray(x, y, nx, ny) != 0
    return np.moveaxis(dados, 0, -1), validos


def coletar(img, rot, bloco, semente):
    """Varre a cena e devolve (X, y, split) de todo pixel rotulado e válido."""
    ds = gdal.Open(img)
    rs = gdal.Open(rot)
    if (ds.RasterXSize, ds.RasterYSize) != (rs.RasterXSize, rs.RasterYSize):
        raise SystemExit("imagem e rótulos têm grades diferentes")
    masc = ds.GetRasterBand(1).GetMaskBand()
    rng = np.random.default_rng(semente)
    Xs, ys, sp = [], [], []
    for x, y, nx, ny in janelas(ds.RasterXSize, ds.RasterYSize, bloco):
        r = rs.GetRasterBand(1).ReadAsArray(x, y, nx, ny)
        if not (r > 0).any():
            continue
        d, v = ler_bloco(ds, x, y, nx, ny, masc)
        sel = (r > 0) & v
        if not sel.any():
            continue
        Xs.append(d[sel])
        ys.append(r[sel])
        # bloco espacial -> treino (70%) ou validação (30%), decidido pelo bloco
        bx, by = x // BLOCO_SPLIT, y // BLOCO_SPLIT
        h = abs(hash((int(bx), int(by), int(semente)))) % 100
        sp.append(np.full(int(sel.sum()), 0 if h < 70 else 1, np.uint8))
    if not ys:
        raise SystemExit("nenhum pixel rotulado sobre pixel válido")
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(sp), rng)


def cota(y, maximo, rng):
    """Índices com cota igual por classe, até `maximo` no total."""
    if not maximo or len(y) <= maximo:
        return np.arange(len(y))
    cls = np.unique(y)
    por = maximo // len(cls)
    out = []
    for c in cls:
        idx = np.where(y == c)[0]
        out.append(rng.choice(idx, min(por, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", required=True)
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--rotulos", required=True)
    ap.add_argument("--arvores", type=int, default=500)
    ap.add_argument("--max-profundidade", type=int, default=0, help="0 = livre")
    ap.add_argument("--max-amostras", type=int, default=0, help="0 = todas")
    ap.add_argument("--bloco", type=int, default=2048)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--medir-cpu", type=int, default=2000000,
                    help="pixels para cronometrar rf.predict na CPU")
    ap.add_argument("--verificar", type=int, default=500000,
                    help="pixels guardados com a resposta da CPU, p/ o rf_gpu conferir")
    ap.add_argument("--max-validacao", type=int, default=3000000,
                    help="teto de pixels de validação (RAM)")
    ap.add_argument("--salvar-joblib", action="store_true",
                    help="grava o modelo do sklearn (GBs; não recarrega em 7 GB)")
    args = ap.parse_args()

    import joblib
    from sklearn.ensemble import RandomForestClassifier

    saida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.nome)
    os.makedirs(saida, exist_ok=True)

    print("[coletando amostras]", flush=True)
    t0 = time.monotonic()
    X, y, sp, rng = coletar(args.imagem, args.rotulos, args.bloco, args.semente)
    print("  %d pixels rotulados | %.0fs" % (len(y), time.monotonic() - t0), flush=True)
    for c in np.unique(y):
        print("    classe %3d: %9d px (treino %d / val %d)"
              % (c, (y == c).sum(), ((y == c) & (sp == 0)).sum(),
                 ((y == c) & (sp == 1)).sum()))

    tr = np.where(sp == 0)[0]
    va = np.where(sp == 1)[0]
    tr = tr[cota(y[tr], args.max_amostras, rng)]
    print("  treino usado: %d px | validacao: %d px" % (len(tr), len(va)), flush=True)

    rf = RandomForestClassifier(
        n_estimators=args.arvores,
        max_depth=args.max_profundidade or None,
        oob_score=False,          # ligado no original; custa uma passada extra
        random_state=args.semente,
        n_jobs=-1,
    )
    print("[treinando] %d arvores, profundidade %s"
          % (args.arvores, args.max_profundidade or "livre"), flush=True)
    t0 = time.monotonic()
    rf.fit(X[tr], y[tr])
    t_treino = time.monotonic() - t0
    nos = sum(e.tree_.node_count for e in rf.estimators_)
    prof = max(e.tree_.max_depth for e in rf.estimators_)
    print("  %.0fs | %d nos no total | profundidade max %d" % (t_treino, nos, prof),
          flush=True)

    from rf_gpu import achatar
    pasta_fl = os.path.join(saida, "floresta")
    os.makedirs(pasta_fl, exist_ok=True)

    # --- cronometragem da predição na CPU + amostra de verificação
    n = min(args.medir_cpu, len(X))
    amostra = X[rng.choice(len(X), n, replace=False)]
    print("[medindo predict na cpu] %d pixels" % n, flush=True)
    t0 = time.monotonic()
    pred_cpu = rf.predict(amostra)
    t_cpu = time.monotonic() - t0
    print("  %.1fs -> %.3f Mpx/s" % (t_cpu, n / 1e6 / t_cpu), flush=True)
    nv = min(args.verificar, n)
    np.savez(os.path.join(pasta_fl, "verificacao.npz"),
             X=amostra[:nv], pred=pred_cpu[:nv])
    print("  verificacao.npz: %d pixels com a resposta da CPU" % nv, flush=True)

    # --- validação, em subamostra (o conjunto inteiro não cabe na RAM daqui)
    nva = min(len(va), args.max_validacao)
    va_s = rng.choice(va, nva, replace=False) if nva < len(va) else va
    pv = rf.predict(X[va_s])
    acerto = float((pv == y[va_s]).mean())
    print("[validacao] %d px | acuracia %.4f" % (nva, acerto), flush=True)

    if args.salvar_joblib:
        cam = os.path.join(saida, "modelo.joblib")
        joblib.dump(rf, cam, compress=0)
        print("  modelo.joblib: %.2f GB (nao recarregavel nesta maquina)"
              % (os.path.getsize(cam) / 1e9), flush=True)

    # --- achatar é a ÚLTIMA etapa: ela esvazia rf.estimators_ para liberar RAM,
    #     e depois disso o modelo do sklearn não prediz mais nada
    print("[achatando a floresta]", flush=True)
    t0 = time.monotonic()
    achatar(rf, pasta_fl)
    print("  %.0fs" % (time.monotonic() - t0), flush=True)

    info = {"nome": args.nome, "arvores": args.arvores,
            "max_profundidade": args.max_profundidade or None,
            "treino_px": int(len(tr)), "validacao_px": int(nva),
            "segundos_treino": t_treino, "nos": int(nos), "profundidade": int(prof),
            "cpu_mpx_s": n / 1e6 / t_cpu, "cpu_segundos": t_cpu, "cpu_pixels": int(n),
            "acuracia_validacao": acerto,
            "classes": rf.classes_.tolist()}
    with open(os.path.join(saida, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("OK -> %s" % saida)


if __name__ == "__main__":
    main()
