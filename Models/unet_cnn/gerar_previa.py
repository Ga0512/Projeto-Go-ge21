#!/usr/bin/env python3
"""Cria uma prévia PNG legível de um GeoTIFF de classes MapBiomas.

O GeoTIFF completo permanece intacto. A prévia amostra a grade para uma
largura pequena e pinta os códigos de classe com as cores de `dados.py`.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/gerar_previa.py \\
      --imagem unet_cnn/runs/MEU_RUN/cena_jatai/classes.tif \\
      --saida unet_cnn/runs/MEU_RUN/cena_jatai/classes.png \\
      --largura 4096
"""

import argparse
import os

import ambiente  # noqa: F401  (caches antes das bibliotecas cientificas)
import numpy as np
from osgeo import gdal


gdal.UseExceptions()
CORES = {
    0: (255, 255, 255, 0),
    3: (31, 141, 73, 255),
    4: (125, 201, 117, 255),
    15: (237, 222, 142, 255),
    33: (37, 50, 228, 255),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagem", required=True, help="classes.tif georreferenciado")
    ap.add_argument("--saida", required=True, help="classes.png de prévia")
    ap.add_argument("--largura", type=int, default=4096)
    args = ap.parse_args()
    if args.largura < 1:
        ap.error("--largura precisa ser positiva")

    fonte = gdal.Open(args.imagem)
    if fonte is None or fonte.RasterCount != 1:
        raise SystemExit("esperado GeoTIFF de classes com uma banda")
    largura = min(args.largura, fonte.RasterXSize)
    altura = max(1, round(fonte.RasterYSize * largura / fonte.RasterXSize))
    reduzido = gdal.Translate("", fonte, format="MEM", width=largura,
                              height=altura, resampleAlg="nearest")
    codigos = reduzido.GetRasterBand(1).ReadAsArray()

    rgba = gdal.GetDriverByName("MEM").Create("", largura, altura, 4,
                                              gdal.GDT_Byte)
    interpretacoes = (gdal.GCI_RedBand, gdal.GCI_GreenBand,
                      gdal.GCI_BlueBand, gdal.GCI_AlphaBand)
    for canal, interpretacao in enumerate(interpretacoes):
        tabela = np.zeros(256, dtype=np.uint8)
        for codigo, cor in CORES.items():
            tabela[codigo] = cor[canal]
        banda = rgba.GetRasterBand(canal + 1)
        banda.WriteArray(tabela[codigos])
        banda.SetColorInterpretation(interpretacao)

    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    gdal.Translate(args.saida, rgba, format="PNG")
    print("Previa: %s (%dx%d)" % (args.saida, largura, altura), flush=True)


if __name__ == "__main__":
    main()
