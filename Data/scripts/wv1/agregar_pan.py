#!/usr/bin/env python3
"""Agrega uma PAN por um fator inteiro, com média que IGNORA o fundo.

Serve para a WV1 de Jataí (274176x293760 a ~0,5 m, 67% de fundo 0): a 0,5 m as
duas texturas da cena dariam ~150 GB. Agregando 5x a grade fica em ~2,38 x 2,50 m
no chão — a escala das cenas SPOT5, onde a janela 7 px das texturas cobre 17,5 m
— e as texturas caem para ~6 GB.

Por que não `gdal.Translate(..., resampleAlg="average")`:
  1. com `width=W//fator` a escala sai 5,00002 (a largura não é múltiplo de 5) e
     as células saem desalinhadas — medido: até 30 DN de diferença no interior;
  2. a média do GDAL NÃO exclui o fundo (o `noData` do Translate só marca a
     saída), o que escurece uma orla de 1 pixel em toda a borda da imagem.
Aqui a saída é recortada para múltiplo exato do fator e a média usa só os
subpixels com imagem; célula 100% fundo sai 0 (= sem imagem, igual às outras
cenas do projeto).

Uso:
  .venv/bin/python scripts/wv1/agregar_pan.py --pan <PAN.tif> [--fator 5] [--saida <tif>]
"""
import argparse
import os
import time

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
FUNDO = 0


def agregar(pan_path, fator, saida, bloco_saida=512):
    t0 = time.time()
    gdal.SetCacheMax(256 * 1024 * 1024)
    ds = gdal.Open(pan_path)
    band = ds.GetRasterBand(1)
    W, H = ds.RasterXSize, ds.RasterYSize
    gt = list(ds.GetGeoTransform())
    Wo, Ho = W // fator, H // fator          # recorte p/ múltiplo exato do fator
    gt[1] *= fator
    gt[5] *= fator
    print("%s: %dx%d -> %dx%d (fator %d) | pixel %.3e -> %.3e"
          % (os.path.basename(pan_path), W, H, Wo, Ho, fator,
             ds.GetGeoTransform()[1], gt[1]), flush=True)

    d = gdal.GetDriverByName("GTiff").Create(
        saida, Wo, Ho, 1, gdal.GDT_Byte,
        ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"])
    d.SetGeoTransform(gt)
    d.SetProjection(ds.GetProjection())
    bo = d.GetRasterBand(1)
    bo.SetNoDataValue(FUNDO)
    bo.SetDescription("PAN agregada %dx (media ignorando fundo)" % fator)

    n_bloco = -(-Ho // bloco_saida)
    for n, y in enumerate(range(0, Ho, bloco_saida), 1):
        rows = min(bloco_saida, Ho - y)
        a = band.ReadAsArray(0, y * fator, Wo * fator, rows * fator)
        sub = (a.reshape(rows, fator, Wo, fator)
                .transpose(0, 2, 1, 3)
                .reshape(rows, Wo, fator * fator).astype(np.float32))
        ok = sub != FUNDO
        cnt = ok.sum(2)
        soma = np.where(ok, sub, 0).sum(2)
        out = np.where(cnt > 0, soma / np.maximum(cnt, 1) + 0.5, FUNDO).astype(np.uint8)
        bo.WriteArray(out, 0, y)
        d.FlushCache()
        print("   bloco %3d/%d  linhas %6d-%6d  (%.0fs)"
              % (n, n_bloco, y, y + rows - 1, time.time() - t0), flush=True)
    d = ds = None
    print("OK %s (%.1f MB) em %.0fs" % (saida, os.path.getsize(saida) / 1e6,
                                        time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("Uso:"):])
    ap.add_argument("--pan", required=True)
    ap.add_argument("--fator", type=int, default=5)
    ap.add_argument("--saida", help="default: <pan>_agg<fator>.tif")
    args = ap.parse_args()
    saida = args.saida or "%s_agg%d.tif" % (os.path.splitext(args.pan)[0], args.fator)
    agregar(args.pan, args.fator, saida)


if __name__ == "__main__":
    main()
