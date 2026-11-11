#!/usr/bin/env python3
"""Gera patches de treino com labels da classificação MANUAL.

Usa exatamente as MESMAS janelas (x, y) do treino_2007 (manifest.csv),
reaproveitando as imagens (img/ por symlink). Só a máscara muda:
mask_manual.tif (shapefile classificado à mão) no lugar do MapBiomas.

Saída: dados/wv1/patches/treino_manual/ (img -> symlink, mask/ nova, manifest,
splits/ copiado do treino_2007 para os mesmos splits).

Uso: .venv/bin/python scripts/wv1/gerar_patches_manual.py
"""

import csv
import os
import shutil
import sys

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

ORIG = caminhos.WV1_TREINO_2007
MASK = caminhos.WV1_MASK_MANUAL
OUT = caminhos.WV1_TREINO_MANUAL
PATCH = 512


def main():
    mask_ds = gdal.Open(MASK)
    mask_b = mask_ds.GetRasterBand(1)

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "mask"), exist_ok=True)
    img_link = os.path.join(OUT, "img")
    if not os.path.islink(img_link):
        # relativo: continua válido se a pasta do projeto mudar de nome/lugar
        os.symlink(os.path.relpath(os.path.join(ORIG, "img"), OUT), img_link)

    manifest_orig = os.path.join(ORIG, "manifest.csv")
    rows = []
    with open(manifest_orig) as f:
        next(f)
        for line in f:
            idx, x, y, aug, vf = line.strip().split(",")
            idx = int(idx)
            patch = mask_b.ReadAsArray(int(x), int(y), PATCH, PATCH)
            vf_manual = float((patch != 0).mean())
            np.savez_compressed(os.path.join(OUT, "mask", "%06d.npz" % idx),
                                patch.astype(np.uint8), allow_pickle=False)
            rows.append([idx, x, y, aug, round(vf_manual, 4)])

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["idx", "x", "y", "aug", "valid_frac"])
        cw.writerows(rows)

    shutil.copytree(os.path.join(ORIG, "splits"),
                    os.path.join(OUT, "splits"), dirs_exist_ok=True)
    print("OK: %d patches em %s (splits copiados do treino_2007)"
          % (len(rows), OUT))


if __name__ == "__main__":
    main()