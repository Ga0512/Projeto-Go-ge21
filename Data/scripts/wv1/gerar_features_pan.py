#!/usr/bin/env python3
"""Gera as camadas de features da PAN WorldView-1 e salva GeoTIFF de 4 bandas:

  band1 = PAN           (informação radiométrica)   uint8
  band2 = Local STD     (variabilidade local)       float32
  band3 = GLCM Entropy  (textura)                   float32
  band4 = Gradient      (estrutura/bordas)          float32

Processa em blocos de linhas para caber em memória (imagem 22400x18336).
GLCM Entropy: janela W x W, offset (1,0), N níveis quantizados, acumulado por par.
"""

import argparse
import os
import sys

import numpy as np
import scipy.ndimage as ndi

from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

DEFAULT_PAN = caminhos.WV1_PAN
DEFAULT_OUT = caminhos.WV1_FEATURES

WINDOW = 7          # janela ímpar p/ Local STD e GLCM
LEVELS = 8          # níveis de quantização do GLCM
BLOCK_ROWS = 4096   # linhas por bloco


def features_block(pad_strip, block_rows, window, levels):
    """Recebe bloco uint8 já com margem vertical refletida (len = block_rows + 2*half)
    e devolve as 3 camadas centrais (linhas originais, largura total W)."""
    half = window // 2
    full_rows, W = pad_strip.shape
    rows = block_rows
    c0 = pad_strip[half:half + rows, :]  # PAN central (uint8)

    # --- Gradient (sobel, magnitude) ---
    im = pad_strip.astype(np.float32)
    gx = ndi.sobel(im, axis=1, mode="reflect")
    gy = ndi.sobel(im, axis=0, mode="reflect")
    grad = np.hypot(gx, gy)[half:half + rows, :]

    # --- Local STD ---
    mean = ndi.uniform_filter(im, size=window, mode="reflect")
    mean2 = ndi.uniform_filter(im * im, size=window, mode="reflect")
    std = np.sqrt(np.clip(mean2 - mean * mean, 0, None))[half:half + rows, :]

    # --- GLCM Entropy (offset (0,1): q[c] vs q[c+1]; reflect nas colunas) ---
    q = (pad_strip.astype(np.int16) * levels // 256).clip(0, levels - 1)
    qh = np.empty_like(q)
    qh[:, :-1] = q[:, 1:]
    qh[:, -1] = q[:, -1]
    ent = np.zeros((full_rows, W), dtype=np.float32)
    total = np.zeros((full_rows, W), dtype=np.float32)
    for i in range(levels):
        Ai = (q == i).astype(np.uint8)
        for j in range(levels):
            Bj = (qh == j).astype(np.uint8)
            m = (Ai * Bj).astype(np.float32)
            s = ndi.uniform_filter1d(m, window, axis=0, mode="reflect")
            s = ndi.uniform_filter1d(s, window, axis=1, mode="reflect")
            cnt = s * (window * window)
            total += cnt
            ent += np.where(cnt > 0, cnt * np.log(cnt), 0.0)
    ent = np.where(total > 0, np.log(total) - ent / total, 0.0)[half:half + rows, :]

    return c0, std, ent, grad


def main():
    ap = argparse.ArgumentParser(description="Features da PAN WV1")
    ap.add_argument("--pan", default=DEFAULT_PAN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--levels", type=int, default=LEVELS)
    ap.add_argument("--block-rows", type=int, default=BLOCK_ROWS)
    ap.add_argument("--threads", type=int, default=0,
                    help="GDAL threads (padrão 0=auto)")
    args = ap.parse_args()

    window = args.window
    if window % 2 == 0:
        raise SystemExit("--window deve ser ímpar")
    half = window // 2
    gdal.UseExceptions()
    if args.threads:
        gdal.SetConfigOption("GDAL_NUM_THREADS", str(args.threads))

    ds = gdal.Open(args.pan)
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    W, H = ds.RasterXSize, ds.RasterYSize
    src = ds.GetRasterBand(1)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    print("PAN: %s" % args.pan)
    print("dims=%dx%d window=%d levels=%d block_rows=%d" % (W, H, window, args.levels, args.block_rows))

    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(
        args.out, W, H, 4, gdal.GDT_Float32,
        ["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES"],
    )
    out.SetGeoTransform(gt)
    out.SetProjection(wkt)
    for i, name in enumerate(["PAN", "LocalSTD", "GLCM_Entropy", "Gradient"], start=1):
        out.GetRasterBand(i).SetDescription(name)

    nblocks = 0
    for y0 in range(0, H, args.block_rows):
        y1 = min(y0 + args.block_rows, H)
        rows = y1 - y0
        # margem vertical com reflect (limita nos extremos e copia se faltar)
        top = y0 - half
        bot = y1 + half
        t0 = max(0, top)
        t1 = min(H, bot)
        strip = src.ReadAsArray(0, t0, W, t1 - t0).astype(np.uint8)
        if t0 > top:                      # falta margem no topo (primeiro bloco)
            n = t0 - top
            strip = np.concatenate([strip[n:0:-1], strip], axis=0)
        if t1 < bot:                      # falta margem na base (último bloco)
            n = bot - t1
            strip = np.concatenate([strip, strip[-2:-2 - n:-1]], axis=0)
        assert strip.shape[0] == rows + 2 * half, (strip.shape[0], rows + 2 * half)

        c, std, ent, grad = features_block(strip, rows, window, args.levels)
        out.GetRasterBand(1).WriteArray(c, 0, y0)
        out.GetRasterBand(2).WriteArray(std, 0, y0)
        out.GetRasterBand(3).WriteArray(ent, 0, y0)
        out.GetRasterBand(4).WriteArray(grad, 0, y0)
        nblocks += 1
        for k in range(1, 5):
            out.GetRasterBand(k).FlushCache()
        print("bloco %d linhas %d-%d ok" % (nblocks, y0, y1 - 1))

    out = None
    ds = None
    print("GeoTIFF 4 camadas salvo:", args.out)
    print("  band1=PAN(uint8->float32), band2=LocalSTD, band3=GLCM_Entropy, band4=Gradient")


if __name__ == "__main__":
    main()