#!/usr/bin/env python3
"""Gera o par SPOT-5 PAN 2.5m <-> máscara MapBiomas 10m (Coleção 3 beta) na mesma grade.

Pipeline:
  1. Lê IMAGEM_SPOT5_10092007.tif e extrai CRS/transform/resolução/dimensões/footprint.
  2. Usa a área do GeoTIFF como ROI.
  3. No GEE (projeto beaming-botany-373619), acessa o MapBiomas Coverage 10m /
     Collection 3 beta e seleciona o ano 2024 (banda classification_2024).
  4. Baixa o recorte do MapBiomas em resolução nativa (10m) sobre a ROI.
  5. Reamostra o MapBiomas para a grade exata da PAN usando nearest-neighbor (GDAL),
     preservando os valores categóricos nível 3.
  6. Salva o par PAN + MASK e um GeoTIFF multibanda (Band1=PAN, Band2=MASK).

Uso (legado; saída em dados/spot5/recorte_2007-09-10/mapbiomas_2024_10m/):
  .venv/bin/python scripts/spot5/legado/gerar_mascara_mapbiomas.py
"""

import os
import subprocess
import sys

from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import caminhos  # noqa: E402

PROJ = "beaming-botany-373619"
MB_ASSET = (
    "projects/mapbiomas-public/assets/brazil/lulc_10m/collection3/"
    "mapbiomas_10m_collection3_integration_v1"
)
MB_BAND = "classification_2024"
YEAR = 2024

PAN_PATH = caminhos.SPOT5_RECORTE_IMAGEM
OUT_DIR = caminhos.SPOT5_RECORTE_MB_2024
MASK_RAW = os.path.join(OUT_DIR, "_mapbiomas_10m_raw.tif")
MASK_PATH = os.path.join(OUT_DIR, "MASK_MAPBIOMAS_2024.tif")
PAR_PATH = os.path.join(OUT_DIR, "PAR_SPOT5_PAN_MASK_2024.tif")
PAN_OUT = os.path.join(OUT_DIR, "IMAGEM_SPOT5_10092007.tif")


def pan_info():
    ds = gdal.Open(PAN_PATH)
    if ds is None:
        raise RuntimeError("Não foi possível abrir a PAN em %s" % PAN_PATH)
    gt = ds.GetGeoTransform()
    info = {
        "crs": ds.GetProjection(),
        "gt": gt,
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "xmin": gt[0],
        "ymax": gt[3],
        "xmax": gt[0] + gt[1] * ds.RasterXSize,
        "ymin": gt[3] + gt[5] * ds.RasterYSize,
        "resx": abs(gt[1]),
        "resy": abs(gt[5]),
    }
    ds = None
    return info


def fetch_mapbiomas(info):
    import ee

    ee.Initialize(project=PROJ)
    roi = ee.Geometry.Rectangle(
        [info["xmin"], info["ymin"], info["xmax"], info["ymax"]]
    )
    img = ee.Image(MB_ASSET).select(MB_BAND)
    url = img.clip(roi).getDownloadURL(
        {"name": "mapbiomas_10m", "scale": 10, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    import urllib.request

    urllib.request.urlretrieve(url, MASK_RAW)
    print("MapBiomas baixado:", MASK_RAW)


def resample_to_pan(info):
    wkt = info["crs"]
    gdal.UseExceptions()
    out_ds = gdal.GetDriverByName("GTiff").Create(
        MASK_PATH, info["width"], info["height"], 1, gdal.GDT_Byte
    )
    out_ds.SetProjection(wkt)
    out_ds.SetGeoTransform(info["gt"])
    bnd = out_ds.GetRasterBand(1)
    bnd.SetNoDataValue(0)
    src = gdal.Open(MASK_RAW)
    gdal.ReprojectImage(
        src,
        out_ds,
        src.GetProjection(),
        wkt,
        gdal.GRA_NearestNeighbour,
        options=["INIT_DEST=0"],
    )
    bnd.FlushCache()
    src = None
    out_ds = None
    print("MASK reamostrada p/ grade PAN:", MASK_PATH)


def build_multiband(info):
    gdal.UseExceptions()
    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(
        PAR_PATH, info["width"], info["height"], 2, gdal.GDT_Byte
    )
    dst.SetProjection(info["crs"])
    dst.SetGeoTransform(info["gt"])
    pan = gdal.Open(PAN_PATH)
    mask = gdal.Open(MASK_PATH)
    dst.GetRasterBand(1).WriteArray(pan.GetRasterBand(1).ReadAsArray())
    dst.GetRasterBand(2).WriteArray(mask.GetRasterBand(1).ReadAsArray())
    dst.GetRasterBand(1).SetDescription("SPOT5_PAN")
    dst.GetRasterBand(2).SetDescription("MAPBIOMAS_2024")
    pan = mask = dst = None
    print("PAR multibanda criado:", PAR_PATH)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    info = pan_info()
    print("PAN info:", info)
    fetch_mapbiomas(info)
    resample_to_pan(info)
    subprocess.run(
        ["cp", "-f", PAN_PATH, PAN_OUT], check=True
    )
    build_multiband(info)
    print("OK ->", OUT_DIR)


if __name__ == "__main__":
    main()
