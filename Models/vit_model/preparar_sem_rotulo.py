#!/usr/bin/env python3
"""Empacota um dataset de patches sem rótulo (images/*.tif + manifest.csv) num
único .npy float16 (N, C, 256, 256) lido por memmap no pré-treino.

Separa uma fração dos patches por BLOCOS espaciais (--bloco px) para avaliação
externa (segmentação × MapBiomas): esses patches ficam fora do MAE, e blocos
inteiros evitam que um patch de avaliação tenha vizinhos colados no treino.

Saída em vit_model/dados/<nome>/:
  patches.npy   float16 (N, C, H, W), mesma ordem do indice.csv
  indice.csv    i, arquivo, x, y, bloco, uso (mae | aval)
  info.json     origem, bandas, contagens, estatísticas por banda
  divisao.png   mapa dos patches (mae × aval)

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python vit_model/preparar_sem_rotulo.py
"""

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import ambiente  # noqa: F401
import numpy as np

import dados

BASE = os.path.dirname(os.path.abspath(__file__))
PADRAO = os.path.join(dados.DATA, "dados", "datasets", "spot5_013-014_sem_rotulo")


def _ler(caminho):
    return dados._ler_tif(caminho).astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=PADRAO)
    ap.add_argument("--nome", default=None, help="pasta em vit_model/dados/ (padrão: nome do dataset)")
    ap.add_argument("--bloco", type=int, default=2048, help="lado (px) dos blocos da divisão")
    ap.add_argument("--frac-aval", type=float, default=0.10, help="fração de patches para avaliação")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    nome = args.nome or os.path.basename(os.path.normpath(args.dataset))
    saida = os.path.join(BASE, "dados", nome)
    os.makedirs(saida, exist_ok=True)
    with open(os.path.join(args.dataset, "manifest.csv"), newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    bandas = dados.bandas(args.dataset)
    N = len(linhas)

    # divisão por blocos: sorteia blocos até juntar frac_aval dos patches
    blocos = [(int(r["y"]) // args.bloco, int(r["x"]) // args.bloco) for r in linhas]
    unicos = sorted(set(blocos))
    cont = {b: 0 for b in unicos}
    for b in blocos:
        cont[b] += 1
    rng = np.random.default_rng(args.seed)
    aval, n_aval = set(), 0
    for k in rng.permutation(len(unicos)):
        if n_aval >= args.frac_aval * N:
            break
        aval.add(unicos[k])
        n_aval += cont[unicos[k]]
    uso = ["aval" if b in aval else "mae" for b in blocos]

    caminhos = [os.path.join(args.dataset, "images", r["arquivo"] + ".tif") for r in linhas]
    amostra = _ler(caminhos[0])
    destino = os.path.join(saida, "patches.npy")
    mm = np.lib.format.open_memmap(destino, mode="w+", dtype=np.float16, shape=(N, *amostra.shape))
    t0 = time.monotonic()
    with Pool(args.workers) as pool:
        for i, arr in enumerate(pool.imap(_ler, caminhos, chunksize=16)):
            if arr.shape != amostra.shape:
                raise ValueError("%s: shape %s != %s" % (caminhos[i], arr.shape, amostra.shape))
            mm[i] = arr
            if (i + 1) % 1000 == 0 or i + 1 == N:
                print("  %5d/%d  (%.0fs)" % (i + 1, N, time.monotonic() - t0), flush=True)
    mm.flush()

    idx = rng.choice(N, min(N, 500), replace=False)
    sub = mm[np.sort(idx)].astype(np.float32)
    nao_finitos = int((~np.isfinite(sub)).sum())
    stats = {b: {"media": float(sub[:, k].mean()), "desvio": float(sub[:, k].std()),
                 "min": float(sub[:, k].min()), "max": float(sub[:, k].max())}
             for k, b in enumerate(bandas)}
    del mm

    with open(os.path.join(saida, "indice.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["i", "arquivo", "x", "y", "bloco", "uso"])
        for i, (r, b, u) in enumerate(zip(linhas, blocos, uso)):
            wr.writerow([i, r["arquivo"], r["x"], r["y"], "%d_%d" % b, u])
    info = {"dataset": os.path.abspath(args.dataset), "bandas": bandas,
            "shape": [N, *amostra.shape], "dtype": "float16", "bloco": args.bloco,
            "patches": {"mae": uso.count("mae"), "aval": uso.count("aval")},
            "blocos": {"total": len(unicos), "aval": len(aval)},
            "estatisticas_amostra_500": stats, "nao_finitos_amostra": nao_finitos,
            "seed": args.seed, "tempo_s": round(time.monotonic() - t0, 1)}
    with open(os.path.join(saida, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.array([int(r["x"]) for r in linhas])
    ys = np.array([int(r["y"]) for r in linhas])
    av = np.array(uso) == "aval"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs[~av], ys[~av], s=2, c="#9e9e9e", marker="s", label="MAE (%d)" % (~av).sum())
    ax.scatter(xs[av], ys[av], s=2, c="#d62728", marker="s", label="avaliação (%d)" % av.sum())
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=6)
    ax.set_title("%s — divisão por blocos de %d px" % (nome, args.bloco))
    fig.savefig(os.path.join(saida, "divisao.png"), dpi=80)
    plt.close(fig)

    print("[ok] %s | %d patches (mae=%d, aval=%d em %d blocos) | %.0fs"
          % (saida, N, info["patches"]["mae"], info["patches"]["aval"], len(aval), info["tempo_s"]))
    print("  estatísticas (amostra 500): %s | não finitos=%d"
          % ("  ".join("%s: %.2f±%.2f" % (b, s["media"], s["desvio"]) for b, s in stats.items()),
             nao_finitos))


if __name__ == "__main__":
    main()
