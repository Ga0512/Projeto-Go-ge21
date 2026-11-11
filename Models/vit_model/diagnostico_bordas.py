#!/usr/bin/env python3
"""Onde o modelo erra: taxa de erro em função da distância (px) até a borda
mais próxima entre classes na referência.

Para cada pixel rotulado do split (interior, sem --margem px da borda do
patch), calcula a distância até o pixel mais próximo de OUTRA classe válida
(sem rótulo não conta como borda) e agrupa em faixas. Mostra, por faixa:
quanto da área está ali, a taxa de erro geral, e a confusão vegetação densa ->
não densa e não densa -> densa. Se o erro de densa × não densa se concentra
nas primeiras faixas, é borda (transição/rótulo incerto); se continua alto
longe da borda, é o modelo que não separa a textura.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python vit_model/diagnostico_bordas.py \\
      --modelo vit_model/runs/seg_mae_v3_ft_v6/modelo_melhor.pth
  -> tabela no terminal + diagnostico_bordas_<split>.json na pasta do run
"""

import argparse
import json
import os

import ambiente  # noqa: F401  (caches -> D:)
import numpy as np
import scipy.ndimage as ndi
import torch

import dados
from inference import carregar_modelo

FAIXAS = [(0, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 6)]


def distancia_borda(y, n_cls, ign):
    """px até o pixel mais próximo de outra classe válida (inf se não houver)."""
    d = np.full(y.shape, np.inf, np.float32)
    presentes = [k for k in range(n_cls) if (y == k).any()]
    for k in presentes:
        dk = ndi.distance_transform_edt(y != k).astype(np.float32)   # distância até a classe k
        outros = (y != k) & (y != ign)
        d[outros] = np.minimum(d[outros], dk[outros])
    return d


@torch.no_grad()
def prever(modelo, X, dev, lote=16):
    out = []
    for i in range(0, len(X), lote):
        xb = torch.from_numpy(X[i:i + lote]).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(modelo(xb).argmax(1).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--split", default="teste", choices=["teste", "validacao"])
    ap.add_argument("--margem", type=int, default=15)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, ck = carregar_modelo(args.modelo, dev)
    ta = ck["args"]
    dataset = ta["dataset"] if os.path.isabs(ta["dataset"]) else \
        os.path.normpath(os.path.join(ambiente.MODELS, ta["dataset"]))
    nomes = dados.ler_split(dataset, args.split)
    cena_de = dados.cenas(dataset)
    X, Y = dados.carregar(dataset, nomes)
    P = prever(modelo, X, dev)
    N, IGN = dados.N_CLASSES, dados.IGNORE
    DENSA, NDENSA = 0, 1
    m = args.margem

    acc = {}   # (cena, faixa) -> contadores
    for i, nome in enumerate(nomes):
        y, p = Y[i, m:-m, m:-m], P[i, m:-m, m:-m]
        d = distancia_borda(Y[i], N, IGN)[m:-m, m:-m]
        val = y != IGN
        for cena in ("TODAS", cena_de[nome]):
            for fi, (a, b) in enumerate(FAIXAS):
                sel = val & (d >= a) & (d <= b)
                c = acc.setdefault((cena, fi), np.zeros(7, np.int64))
                c += [sel.sum(), (sel & (p != y)).sum(),
                      (sel & (y == DENSA)).sum(), (sel & (y == DENSA) & (p == NDENSA)).sum(),
                      (sel & (y == NDENSA)).sum(), (sel & (y == NDENSA) & (p == DENSA)).sum(),
                      (sel & (((y == DENSA) & (p == NDENSA)) | ((y == NDENSA) & (p == DENSA)))).sum()]

    res = {}
    cenas = ["TODAS"] + sorted({c for c, _ in acc if c != "TODAS"})
    for cena in cenas:
        tot = sum(acc[(cena, fi)][0] for fi in range(len(FAIXAS)))
        conf_tot = max(1, sum(acc[(cena, fi)][6] for fi in range(len(FAIXAS))))
        print("\n%s — %s (%d px no interior)" % (args.split.upper(), cena, tot))
        print("%-12s %7s %8s | %16s %16s | %s" % ("dist. borda", "% área", "erro", "densa->n.densa",
                                                  "n.densa->densa", "% da confusão densa×n.densa"))
        linhas = []
        for fi, (a, b) in enumerate(FAIXAS):
            n, err, nd, d2n, nn, n2d, conf = (int(v) for v in acc[(cena, fi)])
            f = lambda num, den: None if den == 0 else num / den  # noqa: E731
            r = {"faixa_px": [a, b if b < 10 ** 6 else None], "fracao_area": f(n, tot),
                 "erro": f(err, n), "densa_para_nao_densa": f(d2n, nd), "px_densa": nd,
                 "nao_densa_para_densa": f(n2d, nn), "px_nao_densa": nn,
                 "fracao_da_confusao_densa_ndensa": conf / conf_tot}
            linhas.append(r)
            pct = lambda v: "     -" if v is None else "%5.1f%%" % (100 * v)  # noqa: E731
            print("%-12s %7s %8s | %8s (%6.2fM) %8s (%6.2fM) | %s" % (
                "%d-%s px" % (a, b if b < 10 ** 6 else "∞"), pct(r["fracao_area"]), pct(r["erro"]),
                pct(r["densa_para_nao_densa"]), nd / 1e6, pct(r["nao_densa_para_densa"]), nn / 1e6,
                pct(r["fracao_da_confusao_densa_ndensa"])))
        res[cena] = linhas

    saida = os.path.join(os.path.dirname(os.path.abspath(args.modelo)),
                         "diagnostico_bordas_%s.json" % args.split)
    with open(saida, "w", encoding="utf-8") as fh:
        json.dump({"modelo": os.path.abspath(args.modelo), "split": args.split,
                   "margem": m, "faixas": res}, fh, indent=2, ensure_ascii=False)
    print("\nOK -> %s" % saida)


if __name__ == "__main__":
    main()
