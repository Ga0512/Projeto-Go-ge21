#!/usr/bin/env python3
"""Cria um dataset sintético WV de 7 bandas para testar o treino completo.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/testar_treino_sintetico.py /tmp/wv_smoke

O dataset criado é apenas diagnóstico e não representa precisão real.
"""

import ambiente  # noqa: F401; importar antes de GDAL e demais bibliotecas

import csv
import json
import os
import sys

import numpy as np
from osgeo import gdal


def main():
    if len(sys.argv) != 2:
        raise SystemExit("uso: testar_treino_sintetico.py <diretorio>")
    pasta = sys.argv[1]
    for sub in ("images", "masks", "splits"):
        os.makedirs(os.path.join(pasta, sub), exist_ok=True)

    altura = largura = 512
    yy, xx = np.mgrid[:altura, :largura]
    base = np.sin(xx / 40).astype(np.float32) + np.cos(yy / 35).astype(np.float32)
    driver = gdal.GetDriverByName("GTiff")
    splits = {"treino": [], "validacao": [], "teste": []}
    rng = np.random.default_rng(42)

    with open(os.path.join(pasta, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("arquivo", "cena", "split", "frac_veg", "frac_nao_veg", "frac_agua"))
        for i in range(28):
            split = "treino" if i < 24 else ("validacao" if i < 26 else "teste")
            nome = "sintetico_%03d" % i
            splits[split].append(nome)
            imagem = driver.Create(os.path.join(pasta, "images", nome + ".tif"),
                                    largura, altura, 7, gdal.GDT_Float32)
            for banda in range(7):
                valores = base * (0.15 + banda * 0.05) + rng.normal(
                    0, 0.2, (altura, largura)).astype(np.float32)
                imagem.GetRasterBand(banda + 1).WriteArray(valores)
            imagem = None

            classes = ((xx + 13 * i) % largura < 170).astype(np.uint8)
            classes += ((xx + 13 * i) % largura >= 340).astype(np.uint8) * 2
            codigos = np.array((3, 15, 33), np.uint8)[classes]
            mascara = driver.Create(os.path.join(pasta, "masks", nome + ".tif"),
                                     largura, altura, 1, gdal.GDT_Byte)
            mascara.GetRasterBand(1).WriteArray(codigos)
            mascara = None
            writer.writerow((nome, "jatai", split, 0.33, 0.33, 0.34))

    for split, nomes in splits.items():
        with open(os.path.join(pasta, "splits", split + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(nomes) + "\n")
    with open(os.path.join(pasta, "receita.json"), "w", encoding="utf-8") as f:
        json.dump({"bandas": ["b%d" % i for i in range(1, 8)],
                   "parametros": {"resolucao": 2.5}}, f)
    print("Dataset sintético criado:", pasta, "24/2/2 patches, 7 bandas, 3 classes")


if __name__ == "__main__":
    main()
