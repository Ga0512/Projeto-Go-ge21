#!/usr/bin/env python3
"""Avalia um modelo contra uma máscara de referência (GT) externa, na GPU.

Roda a inferência SÓ onde o GT tem rótulo (em vez da cena inteira) e confronta
pixel a pixel. As bandas são construídas pelo mesmo caminho do inference.py
(PAN esticada p2–p98 -> GLCM/LocalSTD/Sobel), com as estatísticas da cena
gravadas no receita.json do dataset do modelo, de modo que a entrada é a mesma
que o modelo viu no treino.

O GT é recortado em blocos contíguos de rótulo (os "tiles" do GT); cada bloco
vira uma inferência independente, com --margem px extras em volta para que a
textura na borda tenha contexto.

Além das métricas globais e por bloco, separa os pixels do GT pela EXPOSIÇÃO do
modelo àquele terreno, reconstruída do manifest.csv do dataset de treino:
  treino        pixel coberto por algum patch do split de treino
  val/teste     coberto só por patch de validação ou teste
  nunca visto   fora de qualquer patch do dataset
A terceira faixa é a única avaliação honesta de generalização.

Saída em <saida>/: relatorio.json, comparacao.png e tabela no terminal.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/avaliar_gt_raster.py \\
      --modelo unet_cnn/runs/unetpp_v15_ft/modelo_melhor.pth \\
      --gt ../Data/dados/spot5/colinas_do_sul/mascara_amostra_v1.tif \\
      --cena colinas_do_sul --passo 64
"""

import argparse
import json
import os
import sys
import time

import ambiente  # noqa: F401  (caches -> D:; antes de torch/matplotlib)
import numpy as np
import scipy.ndimage as ndi
import torch

import dados

sys.path.insert(0, os.path.join(ambiente.MODELS, "vit_model"))
from inference import bandas, carregar_modelo, prever  # noqa: E402

from osgeo import gdal  # noqa: E402

gdal.UseExceptions()
BASE = os.path.dirname(os.path.abspath(__file__))


def blocos_do_gt(gt_ds, reducao=16, min_px=500_000):
    """Acha os blocos contíguos de rótulo do GT -> [(x0, y0, w, h), ...]."""
    W, H = gt_ds.RasterXSize, gt_ds.RasterYSize
    pequeno = gt_ds.GetRasterBand(1).ReadAsArray(
        0, 0, W, H, buf_xsize=W // reducao, buf_ysize=H // reducao) > 0
    rot, n = ndi.label(pequeno)
    saida = []
    for i in range(1, n + 1):
        ys, xs = np.where(rot == i)
        x0, x1 = xs.min() * reducao, (xs.max() + 1) * reducao
        y0, y1 = ys.min() * reducao, (ys.max() + 1) * reducao
        # refina nos limites exatos lendo a janela em resolução cheia
        jan = gt_ds.GetRasterBand(1).ReadAsArray(x0, y0, min(x1 - x0, W - x0),
                                                 min(y1 - y0, H - y0)) > 0
        if jan.sum() < min_px:
            continue
        ys2, xs2 = np.where(jan.any(1))[0], np.where(jan.any(0))[0]
        saida.append((int(x0) + int(xs2[0]), int(y0) + int(ys2[0]),
                      int(xs2[-1] - xs2[0] + 1), int(ys2[-1] - ys2[0] + 1)))
    return sorted(saida, key=lambda b: (b[1], b[0]))


def pegada_do_treino(dataset, cena, W, H, patch=256):
    """Mapa uint8 da exposição do modelo: 0 nunca visto, 1 val/teste, 2 treino."""
    import csv
    cam = os.path.join(dataset, "manifest.csv")
    if not os.path.exists(cam):
        return None, {}
    mapa = np.zeros((H, W), np.uint8)
    cont = {"treino": 0, "validacao": 0, "teste": 0}
    ignorados = 0
    with open(cam, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["cena"] != cena:
                continue
            # patches fora dos 3 splits (ex.: 'descartado_balanceamento') nao
            # foram vistos pelo modelo: nao podem contar como exposicao
            if r["split"] not in cont:
                ignorados += 1
                continue
            cont[r["split"]] += 1
            x, y = int(float(r["x"])), int(float(r["y"]))
            v = 2 if r["split"] == "treino" else 1
            sl = (slice(max(0, y), min(H, y + patch)), slice(max(0, x), min(W, x + patch)))
            np.maximum(mapa[sl], v, out=mapa[sl])
    if ignorados:
        cont["fora_dos_splits"] = ignorados
    return mapa, cont


def metricas(cm, nomes):
    tp = np.diag(cm).astype(np.float64)
    ref, prd = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec, prec = tp / ref, tp / prd
        iou = tp / (ref + prd - tp)
    oa = tp.sum() / max(cm.sum(), 1)
    pe = (ref * prd).sum() / max(cm.sum() ** 2, 1)
    kappa = (oa - pe) / (1 - pe) if pe < 1 else float("nan")
    pres = ref > 0
    return {"oa": float(oa), "kappa": float(kappa),
            "miou": float(np.nanmean(iou[pres])) if pres.any() else float("nan"),
            "iou": [None if np.isnan(v) else float(v) for v in iou],
            "recall": [None if np.isnan(v) else float(v) for v in rec],
            "precisao": [None if np.isnan(v) else float(v) for v in prec],
            "ref_frac": [float(v) for v in ref / max(cm.sum(), 1)],
            "matriz_confusao": cm.tolist(), "n_pixels": int(cm.sum())}


def imprimir(tag, m, nomes):
    print("  %-26s OA=%.4f  kappa=%.4f  mIoU=%.4f  (%.2f Mpx)"
          % (tag, m["oa"], m["kappa"], m["miou"], m["n_pixels"] / 1e6))
    print("     %-18s %8s %8s %8s %8s" % ("classe", "IoU", "recall", "precis", "ref%"))
    for k, n in enumerate(nomes):
        f = lambda v: "   -  " if v is None else "%.4f" % v
        print("     %-18s %8s %8s %8s %7.1f%%"
              % (n, f(m["iou"][k]), f(m["recall"][k]), f(m["precisao"][k]),
                 100 * m["ref_frac"][k]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--gt", required=True, help="GeoTIFF de referência (códigos MapBiomas)")
    ap.add_argument("--pan", default=None, help="PAN da cena (padrão: pelo receita + --cena)")
    ap.add_argument("--cena", required=True, help="cena do receita.json cujas estatísticas usar")
    ap.add_argument("--passo", type=int, default=64, help="px entre janelas de 256")
    ap.add_argument("--margem", type=int, default=128, help="px extras em volta de cada bloco")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()

    t0 = time.monotonic()
    dev = ambiente.exigir_gpu()
    torch.backends.cuda.matmul.allow_tf32 = True

    cam = args.modelo if os.path.isabs(args.modelo) else os.path.join(ambiente.MODELS, args.modelo)
    modelo, ck = carregar_modelo(cam, dev)
    ta = ck["args"]
    dataset = ta["dataset"] if os.path.isabs(ta["dataset"]) \
        else os.path.normpath(os.path.join(ambiente.MODELS, ta["dataset"]))
    dados.configurar(dataset)
    N, IGN = dados.N_CLASSES, dados.IGNORE
    nomes = dados.CLASS_NOMES
    receita = json.load(open(os.path.join(dataset, "receita.json"), encoding="utf-8"))
    par = receita["parametros"]
    st = receita["resultado"]["por_cena"][args.cena]["estatisticas"]
    pan_rel = next(c["pan"] for c in receita["cenas"] if c["nome"] == args.cena)
    PAN = args.pan or os.path.join(dados.DATA, pan_rel)
    nome_run = os.path.basename(os.path.dirname(cam))
    print("[modelo] %s | epoca %s | val mIoU=%.4f | dataset %s | %d classes"
          % (nome_run, ck.get("epoca"), ck.get("val_miou", float("nan")),
             os.path.basename(dataset), N), flush=True)
    print("[cena] %s | p2=%.1f p98=%.1f" % (args.cena, st["p2"], st["p98"]), flush=True)

    pan_ds, gt_ds = gdal.Open(PAN), gdal.Open(args.gt)
    W, H = pan_ds.RasterXSize, pan_ds.RasterYSize
    if (gt_ds.RasterXSize, gt_ds.RasterYSize) != (W, H):
        raise SystemExit("GT %dx%d nao bate com a PAN %dx%d"
                         % (gt_ds.RasterXSize, gt_ds.RasterYSize, W, H))
    blocos = blocos_do_gt(gt_ds)
    print("[gt] %d blocos de rotulo | %s" % (len(blocos), args.gt), flush=True)

    expo, cont = pegada_do_treino(dataset, args.cena, W, H, par["patch"])
    if expo is not None:
        print("[pegada] patches de %s no dataset do modelo: %s" % (args.cena, cont), flush=True)

    P = par["patch"]
    cm_tot = np.zeros((N, N), np.int64)
    cm_exp = {k: np.zeros((N, N), np.int64) for k in ("treino", "val/teste", "nunca visto")}
    por_bloco, paineis = [], []
    for i, (bx, by, bw, bh) in enumerate(blocos, 1):
        m = args.margem
        rx0, ry0 = max(0, bx - m), max(0, by - m)
        rx1, ry1 = min(W, bx + bw + m), min(H, by + bh + m)
        pan = pan_ds.GetRasterBand(1).ReadAsArray(rx0, ry0, rx1 - rx0, ry1 - ry0).astype(np.float32)
        valido = pan > 0 if par.get("pan_zero_nodata") else np.ones(pan.shape, bool)
        X = bandas(pan, valido, st, par["janela"], par["niveis"])
        probs, njan = prever(modelo, X, P, args.passo, args.lote, dev, 15)
        pred = probs.argmax(0).astype(np.uint8)

        ox, oy = bx - rx0, by - ry0
        pred_c = pred[oy:oy + bh, ox:ox + bw]
        val_c = valido[oy:oy + bh, ox:ox + bw]
        gtc = gt_ds.GetRasterBand(1).ReadAsArray(bx, by, bw, bh)
        ref = dados.LUT[gtc]
        ok = (ref != IGN) & val_c
        cm = np.bincount(ref[ok].astype(np.int64) * N + pred_c[ok],
                         minlength=N * N).reshape(N, N)
        cm_tot += cm
        if expo is not None:
            e = expo[by:by + bh, bx:bx + bw]
            for v, k in ((2, "treino"), (1, "val/teste"), (0, "nunca visto")):
                s = ok & (e == v)
                if s.any():
                    cm_exp[k] += np.bincount(ref[s].astype(np.int64) * N + pred_c[s],
                                             minlength=N * N).reshape(N, N)
        mb = metricas(cm, nomes)
        por_bloco.append({"bloco": i, "janela_px": [bx, by, bw, bh], "n_janelas": njan, **mb})
        print("  bloco %d/%d  px %d,%d %dx%d | %d janelas | OA=%.4f mIoU=%.4f | %.0fs"
              % (i, len(blocos), bx, by, bw, bh, njan, mb["oa"], mb["miou"],
                 time.monotonic() - t0), flush=True)
        paineis.append((pan[oy:oy + bh, ox:ox + bw], ref, pred_c, ok))

    print("\n=== GLOBAL ===")
    g = metricas(cm_tot, nomes)
    imprimir("todos os blocos", g, nomes)
    res_exp = {}
    if expo is not None:
        print("\n=== POR EXPOSIÇÃO DO MODELO AO TERRENO ===")
        for k in ("treino", "val/teste", "nunca visto"):
            if cm_exp[k].sum():
                res_exp[k] = metricas(cm_exp[k], nomes)
                imprimir(k, res_exp[k], nomes)
    print("\n=== POR BLOCO ===")
    print("  %-6s %9s %8s %8s %8s  %s" % ("bloco", "Mpx", "OA", "kappa", "mIoU", "IoU por classe"))
    for b in por_bloco:
        print("  %-6d %9.2f %8.4f %8.4f %8.4f  %s"
              % (b["bloco"], b["n_pixels"] / 1e6, b["oa"], b["kappa"], b["miou"],
                 " ".join("   -  " if v is None else "%.3f" % v for v in b["iou"])))

    saida = args.saida or os.path.join(os.path.dirname(cam), "avaliacao_gt")
    os.makedirs(saida, exist_ok=True)
    with open(os.path.join(saida, "relatorio.json"), "w", encoding="utf-8") as f:
        json.dump({"modelo": cam, "run": nome_run, "epoca": ck.get("epoca"),
                   "dataset_modelo": dataset, "gt": args.gt, "cena": args.cena,
                   "passo": args.passo, "margem": args.margem, "classes": nomes,
                   "global": g, "por_exposicao": res_exp, "por_bloco": por_bloco,
                   "patches_no_dataset": cont}, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------ figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(dados.CORES_CLASSE)
    cmap.set_bad("#bdbdbd")
    cerro = ListedColormap(["#2ca02c", "#d62728"])
    cerro.set_bad("#bdbdbd")
    n = len(paineis)
    fig, axs = plt.subplots(n, 4, figsize=(9.5, 2.45 * n), squeeze=False)
    for i, (pan, ref, pred, ok) in enumerate(paineis):
        k = max(1, max(pan.shape) // 700)
        pan, ref, pred, ok = pan[::k, ::k], ref[::k, ::k], pred[::k, ::k], ok[::k, ::k]
        v0, v1 = st["p2"], st["p98"]
        a = axs[i]
        a[0].imshow(pan, cmap="gray", vmin=v0, vmax=v1)
        a[1].imshow(np.ma.masked_where(~ok, ref), cmap=cmap, vmin=0, vmax=N - 1,
                    interpolation="nearest")
        a[2].imshow(np.ma.masked_where(~ok, pred), cmap=cmap, vmin=0, vmax=N - 1,
                    interpolation="nearest")
        a[3].imshow(np.ma.masked_where(~ok, pred != ref), cmap=cerro, vmin=0, vmax=1,
                    interpolation="nearest")
        oa = (pred[ok] == ref[ok]).mean() if ok.any() else float("nan")
        for c, t in enumerate(["PAN", "GT", "predição", "erro (OA=%.3f)" % oa]):
            a[c].set_title(t if i == 0 else "", fontsize=10)
            a[c].set_xlabel(t if i else "", fontsize=8)
            a[c].set_xticks([])
            a[c].set_yticks([])
        a[0].set_ylabel("bloco %d" % (i + 1), fontsize=9)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in dados.CORES_CLASSE],
               labels=nomes, loc="lower center", ncol=N, fontsize=9)
    fig.tight_layout(rect=(0, 0.022, 1, 1))
    fig.savefig(os.path.join(saida, "comparacao.png"), dpi=110)
    plt.close(fig)
    print("\nOK -> %s | %.0fs" % (saida, time.monotonic() - t0))


if __name__ == "__main__":
    main()
