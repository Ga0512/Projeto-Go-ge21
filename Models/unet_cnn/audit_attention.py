#!/usr/bin/env python3
"""Export per-patch input channel attention weights for a trained CNN.

Each CSV row records one patch, its class fractions and one gate weight per
input band. Gate weights are not causal measures of band importance.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/audit_attention.py \
      --name unetpp_cbam_spot5_colinas_7b_p256_v28 \
      --dataset ../Data/dados/datasets/spot5_colinas_v28
"""

import argparse
import csv
import json
import os

import ambiente  # noqa: F401  (configure caches before importing torch)
import numpy as np
import torch

import dados
from unet import model_from_checkpoint


SPLITS = {"train": "treino", "validation": "validacao", "test": "teste"}


def summarize(rows):
    values = np.stack(rows)
    return {"patch_count": len(values), "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="directory under unet_cnn/runs/")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS,
                        default=["validation", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    dados.configurar(args.dataset)
    bands = dados.bandas(args.dataset)
    scene_by_patch = dados.cenas(args.dataset)
    device = ambiente.exigir_gpu()
    run_dir = os.path.join(os.path.dirname(__file__), "runs", args.name)
    checkpoint = torch.load(os.path.join(run_dir, "modelo_melhor.pth"),
                            map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    if not model_args.get("attention_input", model_args.get("atencao_entrada", False)):
        raise SystemExit("checkpoint has no input channel attention")
    model = model_from_checkpoint(checkpoint, dados.N_CLASSES).to(device).eval()
    if checkpoint["state_dict"]["e0.net.0.weight"].shape[1] != len(bands):
        raise SystemExit("recipe band count differs from checkpoint")

    for split in args.splits:
        patch_names = dados.ler_split(args.dataset, SPLITS[split])
        output = os.path.join(run_dir, "attention_%s.csv" % split)
        rows = []
        by_majority = {c: [] for c in range(dados.N_CLASSES)}
        header = (["patch", "scene", "majority_class"]
                  + ["class_fraction_%d" % c for c in range(dados.N_CLASSES)]
                  + ["band_weight_%d_%s" % (i + 1, band)
                     for i, band in enumerate(bands)])
        with open(output, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for start in range(0, len(patch_names), args.batch_size):
                batch = patch_names[start:start + args.batch_size]
                images, labels = dados.carregar(args.dataset, batch)
                with torch.inference_mode():
                    weights = model.input_attention_weights(
                        torch.from_numpy(images).to(device)).cpu().numpy()
                for name, label, weight in zip(batch, labels, weights):
                    counts = np.bincount(label[label != dados.IGNORE].ravel(),
                                         minlength=dados.N_CLASSES)
                    fractions = counts / max(int(counts.sum()), 1)
                    majority = int(np.argmax(fractions)) if counts.sum() else -1
                    writer.writerow([name, scene_by_patch[name], majority]
                                    + [float(v) for v in fractions]
                                    + [float(v) for v in weight])
                    rows.append(weight)
                    if majority >= 0:
                        by_majority[majority].append(weight)

        summary = {
            "run": args.name, "dataset": os.path.abspath(args.dataset),
            "split": split, "bands": bands, "class_names": dados.CLASS_NOMES,
            "epoch": checkpoint.get("epoca"), "all": summarize(rows),
            "by_majority_class": {
                str(c): summarize(group) for c, group in by_majority.items() if group},
        }
        with open(os.path.join(run_dir, "attention_%s_summary.json" % split),
                  "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        print("[attention] %s | %d patches | %s" % (split, len(rows), output),
              flush=True)


if __name__ == "__main__":
    main()
