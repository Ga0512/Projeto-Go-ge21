#!/usr/bin/env python3
"""Visualização sandbox: PAN e máscara agregadas para 30m (bloco 60x60 px).

Gera PNGs separados em experimentos/sandbox_30m/:
  pan_30m.png          — banda PAN agregada por média (percentil 2-98 p/ contraste)
  mascara_30m.png      — máscara MapBiomas 30m original (blocos 60x60, nearest)
Uso: .venv/bin/python scripts/visualizacao/visualizar_30m.py
"""

import os
import sys

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

PAN = caminhos.WV1_FEATURES
MASK = caminhos.WV1_MASK_2007
OUT = os.path.join(caminhos.EXPERIMENTOS, "sandbox_30m")

LEGEND = {
    0: ("Nodata / Fora do mapa", "#ffffff"),
    3: ("Formação Florestal", "#1f8d49"),
    4: ("Formação Savânica", "#7dc975"),
    9: ("Silvicultura", "#7a5900"),
    11: ("Campo Alagado e Área Pantanosa", "#519799"),
    12: ("Formação Campestre", "#d6bc74"),
    15: ("Pastagem", "#edde8e"),
    19: ("Lavoura Temporária", "#c27ba0"),
    21: ("Mosaico de Usos", "#ffefc3"),
    25: ("Outras Áreas não Vegetadas", "#db4d4f"),
    33: ("Rio, Lago e Oceano", "#2532e4"),
    39: ("Soja", "#f5b3c8"),
    41: ("Outras Lavouras Temporárias", "#f54ca9"),
    62: ("Algodão", "#ff69b4"),
}


def agregar_30m(path, banda=1, modo="media"):
    src = gdal.Open(path)
    nx, ny = src.RasterXSize, src.RasterYSize
    b = 60  # 30m / 0.5m
    h, w = ny // b, nx // b
    xoff = (nx - w * b) // 2
    arr = np.empty((h, w), dtype=np.float32)
    for i in range(h):
        bloco = src.GetRasterBand(banda).ReadAsArray(xoff, i * b, w * b, b)
        if modo == "media":
            arr[i] = np.mean(bloco[:b].reshape(b, w, b), axis=(0, 2))
        else:
            arr[i] = bloco[b // 2].reshape(w, b)[:, b // 2]
    src = None
    return arr


def main():
    os.makedirs(OUT, exist_ok=True)
    pan = agregar_30m(PAN, banda=1, modo="media")
    mask = agregar_30m(MASK, banda=1, modo="nearest")

    lo, hi = np.percentile(pan[pan > 0], (2, 98))
    gray = np.clip((pan - lo) / (hi - lo), 0, 1)
    from PIL import Image
    pan_png = os.path.join(OUT, "pan_30m.png")
    Image.fromarray((gray * 255).astype(np.uint8)).save(pan_png)

    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for c, (_, hexc) in LEGEND.items():
        rgb[mask == c] = np.array(
            [int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    mask_png = os.path.join(OUT, "mascara_30m.png")
    Image.fromarray(rgb).save(mask_png)

    # lado a lado (PAN | MÁSCARA)
    pan_im = Image.open(pan_png)
    mask_im = Image.open(mask_png)
    lado = Image.new("RGB", (pan_im.width * 2, pan_im.height), "black")
    lado.paste(pan_im, (0, 0))
    lado.paste(mask_im, (pan_im.width, 0))
    lado.save(os.path.join(OUT, "lado_a_lado_30m.png"))

    # overlay: máscara semitransparente sobre a PAN
    import matplotlib.colors as mcolors
    overlay = np.stack([(gray * 255).astype(np.uint8)] * 3, axis=-1)
    for c, (_, hexc) in LEGEND.items():
        cor = mcolors.to_rgba(hexc)
        sel = mask == c
        overlay[sel] = (overlay[sel] * 0.55
                        + np.array(cor[:3]) * 255 * 0.45).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(OUT, "overlay_30m.png"))

    uniq, counts = np.unique(mask, return_counts=True)
    print("mascara 30m: %dx%d px, classes: %s"
          % (mask.shape[1], mask.shape[0],
             {int(c): int(n) for c, n in zip(uniq, counts)}))
    print("pan 30m: %dx%d px (media)" % (pan.shape[1], pan.shape[0]))
    print("OK -> %s" % OUT)


if __name__ == "__main__":
    main()