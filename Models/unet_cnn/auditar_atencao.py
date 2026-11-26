#!/usr/bin/env python3
"""Exporta os pesos de atenção das bandas para auditar um run da UNet++.

Cada linha do CSV corresponde a um patch, com sete pesos sigmoid (0 a 1) e a
fração rotulada de cada classe. Os pesos são portas aprendidas, não medidas
causais de importância; a comparação causal requer oclusão de bandas.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/auditar_atencao.py \
      --nome unetpp_cbam_spot5_colinas_7b_p256_v28 \
      --dataset ../Data/dados/datasets/spot5_colinas_v28
"""

import argparse
import csv
import json
import os

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch)
import numpy as np
import torch

import dados
from unet import modelo_do_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nome", required=True, help="pasta em unet_cnn/runs/")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["validacao", "teste"])
    ap.add_argument("--lote", type=int, default=16)
    args = ap.parse_args()
    if args.lote < 1:
        ap.error("--lote precisa ser positivo")

    dados.configurar(args.dataset)
    bandas = dados.bandas(args.dataset)
    cenas = dados.cenas(args.dataset)
    dev = ambiente.exigir_gpu()
    saida = os.path.join(os.path.dirname(__file__), "runs", args.nome)
    ck = torch.load(os.path.join(saida, "modelo_melhor.pth"),
                    map_location="cpu", weights_only=False)
    if not ck.get("args", {}).get("atencao_entrada"):
        raise SystemExit("o checkpoint não tem --atencao-entrada")
    modelo = modelo_do_checkpoint(ck, dados.N_CLASSES).to(dev).eval()
    modelo.load_state_dict(ck["state_dict"])
    if ck["state_dict"]["e0.net.0.weight"].shape[1] != len(bandas):
        raise SystemExit("número de bandas da receita difere do checkpoint")

    for split in args.splits:
        nomes = dados.ler_split(args.dataset, split)
        caminho = os.path.join(saida, "atencao_%s.csv" % split)
        todos = []
        grupos = {c: [] for c in range(dados.N_CLASSES)}
        cabecalho = (["patch", "cena", "classe_predominante"]
                     + ["fracao_classe_%d" % c for c in range(dados.N_CLASSES)]
                     + ["peso_%d_%s" % (i + 1, nome)
                        for i, nome in enumerate(bandas)])
        with open(caminho, "w", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            escritor.writerow(cabecalho)
            for ini in range(0, len(nomes), args.lote):
                bloco = nomes[ini:ini + args.lote]
                imgs, rotulos = dados.carregar(args.dataset, bloco)
                with torch.inference_mode():
                    pesos = modelo.pesos_entrada(
                        torch.from_numpy(imgs).to(dev)).cpu().numpy()
                for nome, y, w in zip(bloco, rotulos, pesos):
                    cont = np.bincount(y[y != dados.IGNORE].ravel(),
                                       minlength=dados.N_CLASSES)
                    frac = cont / max(int(cont.sum()), 1)
                    predominante = int(np.argmax(frac)) if cont.sum() else -1
                    escritor.writerow([nome, cenas[nome], predominante]
                                     + [float(v) for v in frac]
                                     + [float(v) for v in w])
                    todos.append(w)
                    if predominante >= 0:
                        grupos[predominante].append(w)

        matriz = np.stack(todos)
        def resumir(valores):
            v = np.stack(valores)
            return {"n_patches": len(v), "media": v.mean(0).tolist(),
                    "desvio": v.std(0).tolist()}

        resumo = {"run": args.nome, "dataset": os.path.abspath(args.dataset),
                  "split": split, "bandas": bandas, "classes": dados.CLASS_NOMES,
                  "epoca": ck.get("epoca"), "todos": resumir(todos),
                  "por_classe_predominante": {
                      dados.CLASS_NOMES[c]: resumir(v) for c, v in grupos.items() if v}}
        with open(os.path.join(saida, "atencao_%s_resumo.json" % split),
                  "w", encoding="utf-8") as f:
            json.dump(resumo, f, ensure_ascii=False, indent=2)
        print("[atenção] %s | %d patches | %s" % (split, len(matriz), caminho),
              flush=True)


if __name__ == "__main__":
    main()
