#!/usr/bin/env python3
"""Avaliação externa: modelos de segmentação × MapBiomas 2007, nos patches
sem rótulo reservados para avaliação (uso=aval em preparar_sem_rotulo.py).

Esses patches ficam em blocos da cena 013-014 FORA do tile refinado e fora do
pré-treino do MAE, e — ao contrário do teste — têm agropecuária. A referência
é o MapBiomas (30 m, na grade da PAN), um pseudo-rótulo grosseiro: a
comparação principal é por CÉLULA de 30 m (12x12 px), com a classe majoritária
da predição na célula. Só células inteiras no interior do patch (sem --margem
px da borda, mesmo recorte do XGBoost) e com classe válida entram.

MapBiomas -> classes: CLASS_MAP do projeto + lavouras 20/39/41/62 como
agropecuária (como no dataset refinado); 24 (urbano), 29 (afloramento
rochoso), 30 (mineração) e 0 ficam de fora.

Modelos: todos os vit_model/runs/seg_*/modelo_melhor.pth (ou --runs) e o
XGBoost de --xgb (features calculadas no próprio patch).

Saída em vit_model/runs/avaliacao_mapbiomas/: relatorio.json, tabela.txt,
exemplos.png.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python vit_model/avaliar_mapbiomas.py
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import time

import ambiente
import numpy as np
import torch

import dados
from segmentacao import SegViT, criar_encoder

BASE = os.path.dirname(os.path.abspath(__file__))
N = dados.N_CLASSES
IGN = dados.IGNORE
MB_PADRAO = os.path.join(dados.DATA, "dados", "spot5", "cena_013-014_2007-06-14",
                         "mapbiomas_2007_30m", "MASK_MAPBIOMAS_2007.tif")
LUT_MB = dados.LUT.copy()
for _c in (20, 39, 41, 62):
    LUT_MB[_c] = dados.CLASS_MAP[19]


def carregar_aval(pasta):
    with open(os.path.join(pasta, "indice.csv"), newline="", encoding="utf-8") as f:
        linhas = [r for r in csv.DictReader(f) if r["uso"] == "aval"]
    mm = np.load(os.path.join(pasta, "patches.npy"), mmap_mode="r")
    ids = np.array([int(r["i"]) for r in linhas])
    return np.array(mm[ids]), linhas


def ler_mapbiomas(caminho, linhas, lado):
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(caminho)
    b = ds.GetRasterBand(1)
    out = np.stack([b.ReadAsArray(int(r["x"]), int(r["y"]), lado, lado) for r in linhas])
    return LUT_MB[out]


def celulas(linhas, lado, margem, cel):
    """Para cada patch: ids de célula (grade de 30 m alinhada à origem da cena)
    por pixel, só p/ células inteiras no interior; -1 fora. -> (P, lado, lado)."""
    out = np.full((len(linhas), lado, lado), -1, dtype=np.int32)
    for k, r in enumerate(linhas):
        x0, y0 = int(r["x"]), int(r["y"])
        ini_x = -(-(x0 + margem) // cel) * cel - x0      # 1ª borda de célula dentro do interior
        ini_y = -(-(y0 + margem) // cel) * cel - y0
        nx = (lado - margem - ini_x) // cel
        ny = (lado - margem - ini_y) // cel
        if nx <= 0 or ny <= 0:
            continue
        gy, gx = np.meshgrid(np.arange(ny * cel) // cel, np.arange(nx * cel) // cel, indexing="ij")
        out[k, ini_y:ini_y + ny * cel, ini_x:ini_x + nx * cel] = gy * nx + gx
    return out


def por_celula(pred, ref, cel_ids):
    """Classe majoritária da predição e classe do MapBiomas por célula válida."""
    ps, rs = [], []
    for p, r, c in zip(pred, ref, cel_ids):
        v = c >= 0
        if not v.any():
            continue
        ids = c[v]
        n_cel = ids.max() + 1
        cont = np.zeros((n_cel, N), dtype=np.int64)
        np.add.at(cont, (ids, p[v].astype(np.int64)), 1)
        ref_cel = np.full(n_cel, IGN, dtype=np.int64)
        ref_cel[ids] = r[v]                      # MapBiomas é constante dentro da célula
        ok = ref_cel != IGN
        ps.append(cont.argmax(1)[ok])
        rs.append(ref_cel[ok])
    return np.concatenate(ps), np.concatenate(rs)


def metricas(pred, ref):
    ok = ref != IGN
    cm = np.bincount(ref[ok].astype(np.int64) * N + pred[ok].astype(np.int64),
                     minlength=N * N).reshape(N, N)
    tp = np.diag(cm).astype(np.float64)
    r, p = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec, prec, iou = tp / r, tp / p, tp / (r + p - tp)
    lst = lambda v: [None if np.isnan(a) else round(float(a), 4) for a in v]  # noqa: E731
    return {"oa": float(tp.sum() / max(cm.sum(), 1)), "miou": float(np.nanmean(iou)),
            "recall": lst(rec), "precisao": lst(prec), "iou": lst(iou),
            "ref_frac": lst(r / max(cm.sum(), 1)), "matriz_confusao": cm.tolist(), "n": int(cm.sum())}


@torch.no_grad()
def prever_seg(run, X, dev, lote=32):
    ck = torch.load(os.path.join(run, "modelo_melhor.pth"), map_location="cpu", weights_only=False)
    enc, _ = criar_encoder("aleatorio", in_chans=X.shape[1])
    modelo = SegViT(enc, N, in_chans=X.shape[1])
    modelo.load_state_dict(ck["state_dict"])
    modelo.to(dev).eval()
    out = []
    for i in range(0, len(X), lote):
        x = torch.from_numpy(X[i:i + lote]).to(dev).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(modelo(x).argmax(1).to(torch.uint8).cpu().numpy())
    del modelo
    torch.cuda.empty_cache()
    return np.concatenate(out)


def prever_xgb(run, X, dev, lote=8):
    import xgboost as xgb
    spec = importlib.util.spec_from_file_location(
        "xgb_features", os.path.join(ambiente.MODELS, "xgboost", "features.py"))
    feats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feats)
    rel = json.load(open(os.path.join(run, "relatorio.json"), encoding="utf-8"))
    janelas = sorted(set(rel["args"]["janelas"]))
    bst = xgb.Booster()
    bst.load_model(os.path.join(run, "modelo.ubj"))
    bst.set_param({"device": "cuda"})
    out = []
    for i in range(0, len(X), lote):
        x = torch.from_numpy(X[i:i + lote]).to(dev).float()
        f = feats.calcular(x, janelas).permute(0, 2, 3, 1)
        B, H, W, F = f.shape
        d = xgb.DMatrix(f.reshape(-1, F).cpu().numpy(), feature_names=bst.feature_names)
        out.append(bst.predict(d).argmax(1).astype(np.uint8).reshape(B, H, W))
    return np.concatenate(out)


def exemplos(X, ref, preds, linhas, caminho, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(dados.CORES_CLASSE)
    cmap.set_bad("#bdbdbd")
    # prioriza patches com agropecuária no MapBiomas (a classe que o teste não tem)
    frac_agro = (ref == dados.CLASS_MAP[15]).mean(axis=(1, 2))
    cand = np.argsort(-frac_agro)[: max(n * 4, n)]
    idx = np.sort(cand[np.linspace(0, len(cand) - 1, n).astype(int)])
    nomes = list(preds)
    fig, axs = plt.subplots(len(idx), 2 + len(nomes), figsize=(2.6 * (2 + len(nomes)), 2.7 * len(idx)),
                            squeeze=False)
    for lin, i in enumerate(idx):
        v0, v1 = np.percentile(X[i, 0], [2, 98])
        axs[lin, 0].imshow(X[i, 0], cmap="gray", vmin=v0, vmax=v1)
        axs[lin, 0].set_ylabel("x=%s y=%s" % (linhas[i]["x"], linhas[i]["y"]), fontsize=7)
        axs[lin, 1].imshow(np.ma.masked_where(ref[i] == IGN, ref[i]), cmap=cmap, vmin=0, vmax=N - 1,
                           interpolation="nearest")
        for c, nome in enumerate(nomes):
            axs[lin, 2 + c].imshow(preds[nome][i], cmap=cmap, vmin=0, vmax=N - 1, interpolation="nearest")
        for c, tit in enumerate(["PAN", "MapBiomas 2007 (30 m)"] + nomes):
            axs[lin, c].set_title(tit, fontsize=8)
            axs[lin, c].set_xticks([])
            axs[lin, c].set_yticks([])
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cor) for cor in dados.CORES_CLASSE],
               labels=dados.CLASS_NOMES, loc="lower center", ncol=N, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(caminho, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default="spot5_013-014_sem_rotulo",
                    help="pasta de preparar_sem_rotulo.py (nome em vit_model/dados/ ou caminho)")
    ap.add_argument("--mapbiomas", default=MB_PADRAO)
    ap.add_argument("--runs", nargs="*", default=None, help="runs seg_* (padrão: todos)")
    ap.add_argument("--xgb", default=os.path.join(ambiente.MODELS, "xgboost", "runs", "xgb_spot5_v1"))
    ap.add_argument("--margem", type=int, default=15)
    ap.add_argument("--celula", type=int, default=12, help="px por célula do MapBiomas (30 m / 2,5 m)")
    args = ap.parse_args()

    dev = ambiente.exigir_gpu()
    pasta = args.dados if os.path.isdir(args.dados) else os.path.join(BASE, "dados", args.dados)
    saida = os.path.join(BASE, "runs", "avaliacao_mapbiomas")
    os.makedirs(saida, exist_ok=True)

    t0 = time.monotonic()
    X, linhas = carregar_aval(pasta)
    lado = X.shape[-1]
    ref = ler_mapbiomas(args.mapbiomas, linhas, lado)
    cel_ids = celulas(linhas, lado, args.margem, args.celula)
    interior = np.zeros((lado, lado), bool)
    interior[args.margem:lado - args.margem, args.margem:lado - args.margem] = True
    print("[dados] %d patches de avaliação | %.0fs" % (len(X), time.monotonic() - t0))

    runs = args.runs or sorted(os.path.basename(p) for p in glob.glob(os.path.join(BASE, "runs", "seg_*"))
                               if os.path.exists(os.path.join(p, "modelo_melhor.pth")))
    preds = {}
    if args.xgb and os.path.exists(os.path.join(args.xgb, "modelo.ubj")):
        t = time.monotonic()
        preds["xgb:" + os.path.basename(args.xgb)] = prever_xgb(args.xgb, X, dev)
        print("  xgb %.0fs" % (time.monotonic() - t), flush=True)
    for r in runs:
        t = time.monotonic()
        preds[r] = prever_seg(os.path.join(BASE, "runs", r), X, dev)
        print("  %s %.0fs" % (r, time.monotonic() - t), flush=True)

    res = {}
    ref_px = np.where(interior[None], ref, IGN)
    for nome, p in preds.items():
        pc, rc = por_celula(p, ref, cel_ids)
        res[nome] = {"celula_30m": metricas(pc, rc), "pixel_2_5m": metricas(p.ravel(), ref_px.ravel())}

    ab = [c.split()[0][:4] for c in dados.CLASS_NOMES]
    primeiro = next(iter(res.values()))["celula_30m"]
    linhas_txt = [
        "Avaliação × MapBiomas 2007 — %d patches (blocos fora do tile refinado e do MAE), "
        "por célula de 30 m (majoritária), interior" % len(X),
        "referência: %s  (%d células)" % ("  ".join("%s=%.1f%%" % (a, 100 * f)
                                                     for a, f in zip(ab, primeiro["ref_frac"])),
                                         primeiro["n"]),
        "%-26s %6s %6s | %-23s | %-23s" % ("modelo", "OA", "mIoU", "IoU " + " ".join("%5s" % a for a in ab),
                                           "recall " + " ".join("%5s" % a for a in ab))]
    fmt = lambda v: "  -  " if v is None else "%.3f" % v  # noqa: E731
    for nome, r in res.items():
        m = r["celula_30m"]
        linhas_txt.append("%-26s %6.3f %6.3f | %s | %s"
                          % (nome[:26], m["oa"], m["miou"], "    " + " ".join(fmt(v) for v in m["iou"]),
                             "       " + " ".join(fmt(v) for v in m["recall"])))
    tabela = "\n".join(linhas_txt)
    print(tabela)
    with open(os.path.join(saida, "tabela.txt"), "w", encoding="utf-8") as f:
        f.write(tabela + "\n")
    with open(os.path.join(saida, "relatorio.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "patches": len(X), "classes": dados.CLASS_NOMES,
                   "modelos": res}, f, indent=2, ensure_ascii=False)
    exemplos(X, ref, preds, linhas, os.path.join(saida, "exemplos.png"))
    print("OK -> %s" % saida)


if __name__ == "__main__":
    main()
