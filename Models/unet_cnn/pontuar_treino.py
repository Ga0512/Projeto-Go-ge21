#!/usr/bin/env python3
"""Pontua exemplos dificeis do split de treino para selecao espacial.

Usa somente os patches de treino e um modelo ja treinado. Os escores sao
descritivos/in-sample; nao servem para estimar generalizacao. Nao muda dados.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/pontuar_treino.py \
      --nome unetpp_spot5_colinas_v28 \
      --dataset ../Data/dados/datasets/spot5_colinas_v28 --lote 4
"""

import argparse
import csv
import os
import time

import ambiente  # caches antes de torch
import numpy as np
import torch

import dados
from unet import model_from_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nome", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lote", type=int, default=4)
    args = ap.parse_args()
    dados.configurar(args.dataset)
    if dados.N_CLASSES != 3:
        raise SystemExit("pontuacao requer 3 classes")
    dev = ambiente.exigir_gpu()
    pasta = os.path.join(os.path.dirname(__file__), "runs", args.nome,
                          "diagnostico_erros")
    ck = torch.load(os.path.join(os.path.dirname(pasta), "modelo_melhor.pth"),
                    map_location="cpu", weights_only=False)
    modelo = model_from_checkpoint(ck, 3).to(dev).eval()
    nomes = dados.ler_split(args.dataset, "treino")
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, "escores_patches_treino.csv")
    cols = ["patch", "pixels_veg", "pixels_nao_veg", "pixels_agua",
            "fp_nao_veg", "fn_nao_veg", "prob_nao_veg_p95_em_veg",
            "prob_nao_veg_media_em_veg"]
    inicio = time.time()
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(0, len(nomes), args.lote):
            bloco = nomes[i:i + args.lote]
            imgs, ys = dados.carregar(args.dataset, bloco)
            x = torch.from_numpy(imgs).to(dev)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = modelo(x.float()).float()
                probs = logits.softmax(1)[:, 1].cpu().numpy()
                pred = logits.argmax(1).cpu().numpy()
            for k, nome in enumerate(bloco):
                y, z, p = ys[k], pred[k], probs[k]
                veg, nv, agua = y == 0, y == 1, y == 2
                veg_probs = p[veg]
                w.writerow({"patch": nome,
                            "pixels_veg": int(veg.sum()),
                            "pixels_nao_veg": int(nv.sum()),
                            "pixels_agua": int(agua.sum()),
                            "fp_nao_veg": int((veg & (z == 1)).sum() +
                                               (agua & (z == 1)).sum()),
                            "fn_nao_veg": int((nv & (z != 1)).sum()),
                            "prob_nao_veg_p95_em_veg":
                                float(np.percentile(veg_probs, 95)) if veg_probs.size else 0,
                            "prob_nao_veg_media_em_veg":
                                float(veg_probs.mean()) if veg_probs.size else 0})
            if (i // args.lote + 1) % 50 == 0 or i + args.lote >= len(nomes):
                print("[treino] %d/%d patches (%.0fs)" %
                      (min(i + args.lote, len(nomes)), len(nomes), time.time() - inicio),
                      flush=True)
    print("[feito] %s" % caminho, flush=True)


if __name__ == "__main__":
    main()
