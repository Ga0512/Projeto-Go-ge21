#!/usr/bin/env python3
"""Teste isolado: boundary-snapping da máscara MapBiomas à PAN via SLIC superpixels.

Recorte de teste (x0, y0, size) com bordas ricas; a máscara de cada superpixel
recebe a classe majoritária. Gera mask_slic_test.tif + PNGs + relatório em
experimentos/slic_test/. Não altera a máscara de entrada.
"""

import argparse
import os
import sys
import time

import numpy as np
from osgeo import gdal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from skimage.segmentation import slic  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

DEFAULT_PAN = caminhos.WV1_FEATURES
DEFAULT_MASK = caminhos.WV1_MASK_2008
DEFAULT_OUT = os.path.join(caminhos.EXPERIMENTOS, "slic_test")

LEGEND = {
    0: ("Nodata / Fora do mapa", "#ffffff"),
    3: ("Formação Florestal (Forest Formation)", "#1f8d49"),
    4: ("Formação Savânica (Savanna Formation)", "#7dc975"),
    9: ("Silvicultura (Forest Plantation)", "#7a5900"),
    11: ("Campo Alagado e Área Pantanosa (Wetland)", "#519799"),
    12: ("Formação Campestre (Grassland)", "#d6bc74"),
    15: ("Pastagem (Pasture)", "#edde8e"),
    19: ("Lavoura Temporária (Temporary Crop)", "#c27ba0"),
    21: ("Mosaico de Usos (Mosaic of Uses)", "#ffefc3"),
    25: ("Outras Áreas não Vegetadas (Other non Vegetated Areas)", "#db4d4f"),
    33: ("Rio, Lago e Oceano (River, Lake and Ocean)", "#2532e4"),
}


def main():
    ap = argparse.ArgumentParser(description="SLIC boundary-snapping test")
    ap.add_argument("--pan", default=DEFAULT_PAN)
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--x0", type=int, default=3072)
    ap.add_argument("--y0", type=int, default=8192)
    ap.add_argument("--size", type=int, default=3072)
    ap.add_argument("--segments", type=int, default=16000)
    ap.add_argument("--compactness", type=float, default=30)
    ap.add_argument("--sigma", type=float, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gdal.UseExceptions()
    pan_ds = gdal.Open(args.pan)
    mask_ds = gdal.Open(args.mask)
    gt = pan_ds.GetGeoTransform()
    wkt = pan_ds.GetProjection()
    s = args.size
    pan = pan_ds.GetRasterBand(1).ReadAsArray(args.x0, args.y0, s, s).astype(np.float32)
    mask = mask_ds.GetRasterBand(1).ReadAsArray(args.x0, args.y0, s, s)

    t0 = time.time()
    sp = slic(pan, n_segments=args.segments, compactness=args.compactness,
              sigma=args.sigma, start_label=1, channel_axis=None)
    print("SLIC: %.1fs, %d segments" % (time.time() - t0, sp.max()))

    out_mask = np.zeros_like(mask)
    for lab in np.unique(sp):
        sel = sp == lab
        cls, counts = np.unique(mask[sel], return_counts=True)
        keep = cls != 0
        cls, counts = cls[keep], counts[keep]
        if len(cls) == 0:
            continue
        out_mask[sel] = cls[np.argmax(counts)]

    changed = int((out_mask != mask).sum())
    print("pixels alterados: %d (%.2f%%)" % (changed, changed / mask.size * 100))

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(os.path.join(args.out, "mask_slic_test.tif"), s, s, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((gt[0] + args.x0 * gt[1], gt[1], 0, gt[3] + args.y0 * gt[5], 0, gt[5]))
    ds.SetProjection(wkt)
    ds.GetRasterBand(1).WriteArray(out_mask)
    ds = None

    used = sorted({int(v) for v in np.unique(mask) if int(v) in LEGEND})
    cmap = ListedColormap([LEGEND[c][1] for c in used])
    norm = BoundaryNorm(used + [used[-1] + 1], cmap.N)

    a = pan[pan > 0]
    p1, p99 = np.percentile(a, (1, 99))
    pand = np.clip((pan - p1) / (p99 - p1), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    for ax, img, title in [
        (axes[0], pand, "PAN 0,5m"),
        (axes[1], np.ma.masked_where(mask == 0, mask), "Máscara 2008 original"),
        (axes[2], np.ma.masked_where(out_mask == 0, out_mask), "Máscara SLIC test"),
    ]:
        if title.startswith("PAN"):
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title("%s  (%d,%d +%dx%d)" % (title, args.x0, args.y0, s, s))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "comparacao.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(pand, cmap="gray")
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    for c in used:
        rgba[out_mask == c] = matplotlib.colors.to_rgba(LEGEND[c][1])
    rgba[out_mask == 0, 3] = 0
    rgba[out_mask != 0, 3] = 0.55
    ax.imshow(rgba)
    ax.set_title("SLIC test sobre PAN (%d px alterados = %.1f%%)" % (changed, changed / mask.size * 100))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "overlay.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    orig = dict(zip(*np.unique(mask, return_counts=True)))
    novo = dict(zip(*np.unique(out_mask, return_counts=True)))
    with open(os.path.join(args.out, "relatorio.txt"), "w") as f:
        f.write("Teste SLIC boundary-snapping (recorte %d,%d +%dx%d)\n" % (args.x0, args.y0, s, s))
        f.write("segmentos=%d compactness=%.0f sigma=%.0f\n" % (args.segments, args.compactness, args.sigma))
        f.write("pixels alterados: %d (%.2f%%)\n\n" % (changed, changed / mask.size * 100))
        f.write("classe | original(ha) | slic(ha)\n")
        for c in sorted(set(orig) | set(novo)):
            f.write("%5d | %10.2f | %10.2f\n" % (c, orig.get(c, 0) * 0.25 / 10000, novo.get(c, 0) * 0.25 / 10000))
    print("salvo em", args.out)


if __name__ == "__main__":
    main()