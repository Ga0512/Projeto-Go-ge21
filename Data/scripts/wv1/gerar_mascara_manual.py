#!/usr/bin/env python3
"""Rasteriza a classificação manual (shapefile) na grade da PAN 0,5m.

Classes do shapefile -> código de máscara (dentro do CLASS_MAP do config):
  Outros           -> 3   (vegetação natural)
  Área consolidada -> 15  (agropecuária / não vegetado)
  Corpos d'água    -> 33  (água)

Saída: dados/wv1/mascaras/mask_manual.tif (22400x18336, EPSG:32722, Byte).
Uso: .venv/bin/python scripts/wv1/gerar_mascara_manual.py
"""

import os
import sys

from osgeo import gdal, ogr

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, ".."))
sys.path.insert(0, os.path.join(BASE, "..", "modelos"))
import caminhos  # noqa: E402
import config  # noqa: E402

SHP = caminhos.WV1_SHP_MANUAL
REF = caminhos.WV1_FEATURES
OUT = caminhos.WV1_MASK_MANUAL

SHP_MAP = {"Outros": 3, "Área consolidada": 15, "Corpos d'água": 33}


def main():
    tmp = "/vsimem/manual_32722.shp"
    gdal.VectorTranslate(tmp, SHP, format="ESRI Shapefile",
                         dstSRS="EPSG:32722")
    shp = ogr.Open(tmp)
    lyr = shp.GetLayer(0)

    ref = gdal.Open(REF)
    nx, ny = ref.RasterXSize, ref.RasterYSize
    geo = ref.GetGeoTransform()
    proj = ref.GetProjection()
    ref = None
    print("grade: %dx%d (0,5m)" % (nx, ny))

    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(OUT, nx, ny, 1, gdal.GDT_Byte,
                     ["TILED=YES", "COMPRESS=DEFLATE"])
    dst.SetGeoTransform(geo)
    dst.SetProjection(proj)
    band = dst.GetRasterBand(1)
    band.Fill(0)
    for nome, val in SHP_MAP.items():
        lyr.SetAttributeFilter("Classe = '%s'" % nome.replace("'", "''"))
        n = lyr.GetFeatureCount()
        if n == 0:
            print("  classe %s: 0 features" % nome)
            continue
        gdal.RasterizeLayer(dst, [1], lyr, burn_values=[val])
        print("  %-16s -> %3d (%d poligonos)" % (nome, val, n))
    dst = None
    shp = None
    gdal.Unlink(tmp)
    print("OK -> %s" % OUT)


if __name__ == "__main__":
    main()
