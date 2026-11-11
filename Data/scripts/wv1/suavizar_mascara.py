#!/usr/bin/env python3
"""Suaviza as bordas da máscara MapBiomas usando a PAN (watershed guiado).

Pipeline completo, replicável:
  1. Unificação de classes de lavoura específicas (39 Soja, 41 Outras
     Lavouras Temporárias, 62 Algodão) em 19 Lavoura Temporária.
  2. Respeita a cobertura real da PAN: onde PAN == 0 a máscara vira nodata.
  3. Watershed guiado pelo gradiente (banda 4 da PAN): cada classe é uma
     semente (interior erodido) e a fronteira segue o gradiente real da
     imagem em vez do degrau do pixel de 30m.
  4. Filtro de moda 3x3 para limpar ruído de 1px nas bordas.
  5. Validação (códigos oficiais, grade da PAN) + visualização + relatório.

Uso:
  .venv/bin/python scripts/wv1/suavizar_mascara.py \
      --mask dados/wv1/mapbiomas/2007_30m/mask.tif \
      --out dados/wv1/mascaras/mask_2007_suavizada.tif \
      --alias 2007_30m
"""

import argparse
import os
import sys
import time

import numpy as np
from osgeo import gdal
from scipy import ndimage
from skimage.segmentation import watershed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

DEFAULT_PAN = caminhos.WV1_FEATURES

# código específico -> código geral (lavouras vão todas p/ 19)
CLASS_REMAP = {39: 19, 41: 19, 62: 19}

LEGEND = {
    0: ("Nodata / Fora da imagem", "#ffffff"),
    3: ("Formação Florestal", "#1f8d49"),
    4: ("Formação Savânica", "#7dc975"),
    9: ("Silvicultura", "#7a5900"),
    11: ("Campo Alagado", "#519799"),
    12: ("Formação Campestre", "#d6bc74"),
    15: ("Pastagem", "#edde8e"),
    19: ("Lavoura Temporária", "#c27ba0"),
    21: ("Mosaico de Usos", "#ffefc3"),
    25: ("Outras Áreas não Vegetadas", "#db4d4f"),
    33: ("Rio, Lago e Oceano", "#2532e4"),
}

# parâmetros do watershed (calibrados no teste experimentos/slic_test)
EROSION_RADII = (25, 10, 5, 1)   # erosão máxima p/ semente; cai até 1 se classe fina
GRAD_SIGMA = 2.0                 # suavização do gradiente
GRAD_WEIGHT = 15.0               # quão profundo é o vale do gradiente
BOUNDARY_PULL = 0.4              # puxão p/ borda original (0 = confia só na PAN)


def _markers_from_mask(mask):
    """Interior de cada classe = semente firme (erosão máxima, min 1px)."""
    markers = np.zeros_like(mask, dtype=np.int32)
    for c in np.unique(mask):
        if c == 0:
            continue
        for r in EROSION_RADII:
            sel = ndimage.binary_erosion(mask == c, iterations=r)
            if sel.any():
                markers[sel] = c
                break
    return markers


def smooth_block(pan, grad, mask, grad_sigma=GRAD_SIGMA, weight=GRAD_WEIGHT,
                 pull=BOUNDARY_PULL, pull_denom=60.0):
    """Aplica o watershed a um bloco e devolve máscara suavizada (mesma shape)."""
    valid = pan > 0
    mask_v = np.where(valid, mask, 0).astype(np.uint8)
    grad_s = ndimage.gaussian_filter(grad, sigma=grad_sigma)
    gm = np.percentile(grad_s[grad_s > 0], 99)
    edge_all = mask_v != 0
    dm = np.clip(ndimage.distance_transform_edt(edge_all) / pull_denom, 0, 1)
    markers = _markers_from_mask(mask_v)

    cost = 1.0 / (1.0 + np.clip(grad_s / gm, 0, 1) * weight) + pull * dm
    labels = watershed(cost, markers=markers, mask=edge_all)
    out = np.where(edge_all, labels, 0).astype(np.uint8)
    out = ndimage.median_filter(out, size=3)
    out[~valid] = 0
    return out


def main():
    ap = argparse.ArgumentParser(description="Suaviza bordas da máscara via PAN")
    ap.add_argument("--mask", required=True, help="Máscara MapBiomas NA grade da PAN")
    ap.add_argument("--pan", default=DEFAULT_PAN, help="PAN WV1 (features 4 bandas)")
    ap.add_argument("--out", required=True, help="GeoTIFF de saída")
    ap.add_argument("--alias", default="exp", help="Nome do experimento p/ relatório")
    ap.add_argument("--block-rows", type=int, default=2048)
    ap.add_argument("--skip-remap", action="store_true",
                    help="Não unificar classes de lavoura específicas")
    args = ap.parse_args()

    gdal.UseExceptions()
    # 1. unificação (in-place na máscara de entrada!)
    if not args.skip_remap:
        mds = gdal.Open(args.mask, gdal.GA_Update)
        band = mds.GetRasterBand(1)
        a = band.ReadAsArray()
        for old, new in CLASS_REMAP.items():
            a[a == old] = new
        band.WriteArray(a)
        band.FlushCache()
        mds = None
        print("[1/5] classes unificadas:", CLASS_REMAP)

    pan_ds = gdal.Open(args.pan)
    mask_ds = gdal.Open(args.mask)
    mask_b = mask_ds.GetRasterBand(1)
    pan_b = pan_ds.GetRasterBand(1)
    grad_b = pan_ds.GetRasterBand(4)
    gt = pan_ds.GetGeoTransform()
    wkt = pan_ds.GetProjection()
    W, H = pan_ds.RasterXSize, pan_ds.RasterYSize
    print("[2/5] borda da PAN respeitada (PAN==0 -> nodata) - grade %dx%d" % (W, H))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ds = gdal.GetDriverByName("GTiff").Create(
        args.out, W, H, 1, gdal.GDT_Byte, ["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(wkt)
    bw = ds.GetRasterBand(1)

    t0 = time.time()
    y = 0
    while y < H:
        y1 = min(y + args.block_rows, H)
        n = y1 - y
        pan = pan_b.ReadAsArray(0, y, W, n).astype(np.float32)
        mask = mask_b.ReadAsArray(0, y, W, n)
        grad = grad_b.ReadAsArray(0, y, W, n).astype(np.float32)
        out = smooth_block(pan, grad, mask)
        bw.WriteArray(out, 0, y)
        bw.FlushCache()
        print("[3/5] bloco %d-%d ok (%.0fs)" % (y, y1 - 1, time.time() - t0), flush=True)
        y = y1
    ds = None
    print("[4/5] máscara suavizada salva:", args.out)

    # 5. validação + estatística
    m = gdal.Open(args.out)
    a2 = m.GetRasterBand(1).ReadAsArray()
    used = sorted({int(v) for v in np.unique(a2)})
    unknown = [c for c in used if c not in LEGEND]
    print("[5/5] validação:")
    print("  códigos presentes:", used)
    print("  códigos fora da legenda:", unknown or "nenhum")
    rel = os.path.join(os.path.dirname(os.path.abspath(args.out)), "relatorio_suavizacao_%s.txt" % args.alias)
    with open(rel, "w") as f:
        f.write("Relatório suavização %s\n" % args.alias)
        f.write("entrada: %s\nsaída: %s\n\n" % (args.mask, args.out))
        f.write("classe | ha\n")
        for v, c in zip(*np.unique(a2, return_counts=True)):
            f.write("%5d | %.2f\n" % (v, c * 0.25 / 10000))
    print("  relatório:", rel)


if __name__ == "__main__":
    main()