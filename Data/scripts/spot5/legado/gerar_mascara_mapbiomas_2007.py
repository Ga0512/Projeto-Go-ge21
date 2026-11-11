#!/usr/bin/env python3
"""Gera a máscara MapBiomas 2007 (30m, Coleção 10.1) na grade da PAN SPOT5.

A imagem SPOT5 é de set/2007; o MapBiomas 10m só existe a partir de 2017, então
para 2007 usamos a Coleção 10.1 Landsat (30m), banda `classification` do asset
`projects/mapbiomas-public/assets/brazil/lulc/v1/collection_10_1_2007_v1`.

Pipeline (igual ao de referência):
  1. Lê IMAGEM_SPOT5_10092007.tif e extrai CRS/transform/dimensões/footprint.
  2. Baixa o MapBiomas 30m (ano 2007) sobre a ROI em EPSG:4326 (scale 30).
  3. Reamostra nearest-neighbor para a grade exata da PAN (GDT_Byte).
  4. Gera o par multibanda (Band1=PAN, Band2=MASK_2007).

Uso:
  .venv/bin/python scripts/spot5/legado/gerar_mascara_mapbiomas_2007.py

Legado: para cenas novas use scripts/spot5/gerar_mascara_mapbiomas_cena.py.
"""

import os
import sys
import urllib.request

import ee
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import caminhos  # noqa: E402

PROJ = "beaming-botany-373619"
MB_COLL_30M = "projects/mapbiomas-public/assets/brazil/lulc/v1"
YEAR = 2007
SCALE = 30
MB_CRS = "EPSG:4326"

PAN_PATH = caminhos.SPOT5_RECORTE_IMAGEM
OUT_DIR = caminhos.SPOT5_RECORTE_MB_2007
MASK_RAW = os.path.join(OUT_DIR, "_mapbiomas_2007_30m_raw.tif")
MASK_PATH = os.path.join(OUT_DIR, "MASK_MAPBIOMAS_2007.tif")
PAR_PATH = os.path.join(OUT_DIR, "PAR_SPOT5_PAN_MASK_2007.tif")
PAN_OUT = os.path.join(OUT_DIR, "IMAGEM_SPOT5_10092007.tif")


def pan_info():
    ds = gdal.Open(PAN_PATH)
    if ds is None:
        raise RuntimeError("Não foi possível abrir a PAN em %s" % PAN_PATH)
    gt = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    xmin, ymax = gt[0], gt[3]
    xmax = gt[0] + gt[1] * w
    ymin = gt[3] + gt[5] * h
    info = {
        "crs": ds.GetProjection(),
        "gt": gt,
        "width": w,
        "height": h,
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
    }
    ds = None
    return info


def fetch_mapbiomas(info):
    ee.Initialize(project=PROJ)
    roi = ee.Geometry.Polygon([
        [info["xmin"], info["ymin"]],
        [info["xmin"], info["ymax"]],
        [info["xmax"], info["ymax"]],
        [info["xmax"], info["ymin"]],
    ])
    src_image = "%s/collection_10_1_%d_v1" % (MB_COLL_30M, YEAR)
    img = ee.Image(src_image).select("classification")
    url = img.clip(roi).getDownloadURL({
        "name": "mapbiomas_2007",
        "scale": SCALE,
        "crs": MB_CRS,
        "format": "GEO_TIFF",
        "maxPixels": 1e10,
    })
    urllib.request.urlretrieve(url, MASK_RAW)
    print("MapBiomas 30m (%d) baixado:" % YEAR, MASK_RAW)


def resample_to_pan(info):
    gdal.UseExceptions()
    wkt = info["crs"]
    out_ds = gdal.GetDriverByName("GTiff").Create(
        MASK_PATH, info["width"], info["height"], 1, gdal.GDT_Byte)
    out_ds.SetProjection(wkt)
    out_ds.SetGeoTransform(info["gt"])
    bnd = out_ds.GetRasterBand(1)
    bnd.SetNoDataValue(0)
    src = gdal.Open(MASK_RAW)
    gdal.ReprojectImage(
        src, out_ds,
        src.GetProjection(), wkt,
        gdal.GRA_NearestNeighbour,
        options=["INIT_DEST=0"])
    bnd.FlushCache()
    src = None
    out_ds = None
    print("MASK 2007 reamostrada p/ grade PAN:", MASK_PATH)


def build_multiband(info):
    gdal.UseExceptions()
    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(PAR_PATH, info["width"], info["height"], 2, gdal.GDT_Byte)
    dst.SetProjection(info["crs"])
    dst.SetGeoTransform(info["gt"])
    pan = gdal.Open(PAN_PATH)
    mask = gdal.Open(MASK_PATH)
    dst.GetRasterBand(1).WriteArray(pan.GetRasterBand(1).ReadAsArray())
    dst.GetRasterBand(2).WriteArray(mask.GetRasterBand(1).ReadAsArray())
    dst.GetRasterBand(1).SetDescription("SPOT5_PAN")
    dst.GetRasterBand(2).SetDescription("MAPBIOMAS_2007")
    pan = mask = dst = None
    print("PAR multibanda criado:", PAR_PATH)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    info = pan_info()
    print("PAN info:", info)
    fetch_mapbiomas(info)
    resample_to_pan(info)
    import shutil
    shutil.copyfile(PAN_PATH, PAN_OUT)
    build_multiband(info)
    print("OK ->", OUT_DIR)


if __name__ == "__main__":
    main()