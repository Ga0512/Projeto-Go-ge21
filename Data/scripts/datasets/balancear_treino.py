#!/usr/bin/env python3
"""Reescolhe o split de TREINO de um dataset para equilibrar as classes.

O gerador sorteia as janelas uniformemente entre as candidatas, então a
distribuição de classes do treino acaba sendo a da cena. Numa cena dominada
por uma classe (ex.: 79% de vegetação), o modelo vê pouco das outras.

Este script não gera patch nenhum: ele escolhe, DENTRO do split de treino já
existente, o subconjunto cuja distribuição de pixels mais se aproxima do alvo.
Validação e teste não são tocados, e nenhum patch muda de split, então a
separação espacial do gerador continua valendo.

Seleção gulosa: parte vazio e, a cada passo, adiciona o patch que mais aproxima
a distribuição acumulada do alvo (distância L1). Para quando nenhum patch
melhora ou quando --max-patches é atingido.

Os patches de treino descartados saem de splits/treino.txt; com --apagar, os
.tif deles também são removidos do disco. O que foi feito fica em
balanceamento.json.

Uso (no WSL, a partir de Data/):
  .venv/bin/python scripts/datasets/balancear_treino.py \\
      --dataset dados/datasets/spot5_refinado_v17 --alvo 0.5 0.25 0.25
"""

import argparse
import csv
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alvo", type=float, nargs="+", default=None,
                    help="fração de pixels desejada por classe (default: uniforme)")
    ap.add_argument("--max-patches", type=int, default=0, help="0 = sem teto")
    ap.add_argument("--min-ganho", type=float, default=0.0,
                    help="para quando o melhor passo melhora menos que isto")
    ap.add_argument("--apagar", action="store_true",
                    help="apaga do disco os .tif de treino descartados")
    ap.add_argument("--simular", action="store_true", help="não grava nada")
    args = ap.parse_args()

    D = args.dataset
    man = os.path.join(D, "manifest.csv")
    with open(man, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
        campos = list(linhas[0].keys())
    cols = [c for c in campos if c.startswith("frac_")
            and c not in ("frac_rotulado", "frac_outros")]
    nomes = [c[5:] for c in cols]
    alvo = np.array(args.alvo if args.alvo else [1.0 / len(cols)] * len(cols), float)
    alvo = alvo / alvo.sum()
    if len(alvo) != len(cols):
        raise SystemExit("--alvo precisa de %d valores (%s)" % (len(cols), ", ".join(nomes)))

    tr = [r for r in linhas if r["split"] == "treino"]
    # pixels rotulados de cada classe em cada patch (patch 256x256)
    P = np.array([[float(r[c]) for c in cols] for r in tr]) \
        * np.array([[float(r["frac_rotulado"])] for r in tr]) * 256 * 256
    atual = P.sum(0)
    print("=== %s ===" % os.path.basename(D))
    print("  classes: %s" % nomes)
    print("  treino atual: %d patches | distribuicao %s"
          % (len(tr), np.round(atual / atual.sum(), 4)))
    print("  alvo: %s" % np.round(alvo, 4))

    # cota por classe: o maior total possivel com essa proporcao e limitado
    # pela classe mais escassa (quem define o teto e min_c A_c/alvo_c)
    A = P.sum(0)
    T = float(np.min(A / np.maximum(alvo, 1e-12)))
    cota = alvo * T
    print("  pixels disponiveis por classe: %s" % np.round(A / 1e6, 1))
    print("  teto com essa proporcao: %.1f Mpx | cotas %s"
          % (T / 1e6, np.round(cota / 1e6, 1)))

    n = len(tr)
    escolhidos = np.zeros(n, bool)
    acc = np.zeros(len(cols))
    teto = args.max_patches or n
    hist = []
    for passo in range(teto):
        cand = np.where(~escolhidos)[0]
        if not len(cand):
            break
        falta = np.maximum(cota - acc, 0.0)                  # (k,)
        util = np.minimum(P[cand], falta).sum(1)             # pixels que cabem na cota
        sobra = np.maximum(P[cand] - falta, 0.0).sum(1)      # pixels que estouram
        score = util - sobra
        j = int(np.argmax(score))
        if util[j] <= 0 or score[j] <= args.min_ganho:
            break
        i = int(cand[j])
        escolhidos[i] = True
        acc = acc + P[i]
        if (passo + 1) % 200 == 0:
            hist.append({"n": int(escolhidos.sum()),
                         "distribuicao": (acc / acc.sum()).round(4).tolist()})
    dist = float(np.abs(acc / acc.sum() - alvo).sum())
    hist.append({"n": int(escolhidos.sum()), "dist_l1": dist,
                 "distribuicao": (acc / acc.sum()).round(4).tolist()})
    sel = [tr[i]["arquivo"] for i in np.where(escolhidos)[0]]
    desc = [tr[i]["arquivo"] for i in np.where(~escolhidos)[0]]
    print("  -> selecionados %d de %d patches (descartados %d)" % (len(sel), n, len(desc)))
    print("  -> distribuicao final %s  (L1 ao alvo = %.4f)"
          % (np.round(acc / acc.sum(), 4), dist))
    for h in hist[-3:]:
        print("     %4d patches -> %s" % (h["n"], h["distribuicao"]))
    if args.simular:
        print("  (--simular: nada foi gravado)")
        return

    ordem = {r["arquivo"]: k for k, r in enumerate(linhas)}
    sel.sort(key=lambda a: ordem[a])
    with open(os.path.join(D, "splits", "treino.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sel) + "\n")
    fora = set(desc)
    for r in linhas:
        if r["arquivo"] in fora:
            r["split"] = "descartado_balanceamento"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    if args.apagar:
        for a in desc:
            for sub in ("images", "masks"):
                p = os.path.join(D, sub, a + ".tif")
                if os.path.exists(p):
                    os.remove(p)
        print("  -> %d pares de .tif apagados" % len(desc))
    with open(os.path.join(D, "balanceamento.json"), "w", encoding="utf-8") as f:
        json.dump({"alvo": alvo.tolist(), "classes": nomes,
                   "treino_antes": len(tr), "treino_depois": len(sel),
                   "distribuicao_antes": (P.sum(0) / P.sum()).round(6).tolist(),
                   "distribuicao_depois": (acc / acc.sum()).round(6).tolist(),
                   "dist_l1": float(dist), "apagados": bool(args.apagar),
                   "historico": hist}, f, indent=2, ensure_ascii=False)
    print("OK -> splits/treino.txt, manifest.csv e balanceamento.json atualizados")


if __name__ == "__main__":
    main()
