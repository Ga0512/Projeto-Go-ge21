#!/usr/bin/env python3
"""Gera, por cena SPOT5, um shapefile com os polígonos das classes do MapBiomas
2007 (30 m) — organizado para abrir/editar no QGIS.

Fonte: dados/spot5/<cena>/mapbiomas_2007_30m/MASK_MAPBIOMAS_2007.tif (máscara
2007 30m na grade da cena, 0 = sem rótulo/nodata).

Nas cenas grandes (013-014, Colinas, Jataí) a máscara é a repetição 12x12 do
pixel 30m original; vetorizar na grade 2,5m geraria ~144x mais vértices com os
MESMOS limites de classe. O script reamostra a máscara de volta para ~30m
(nearest, alinhado à origem) antes do Polygonize — polígonos idênticos, muuuuito
mais leves. No recorte (pequeno) vetoriza direto.

Atributos por polígono: classe (int), nome (str, legenda oficial), area_ha.

Uso:
  .venv/bin/python scripts/spot5/gerar_shp_mapbiomas_cenas.py
  # ou uma cena só:
  .venv/bin/python scripts/spot5/gerar_shp_mapbiomas_cenas.py --cena jatai
Saída: dados/shp_mapbiomas_2007/<cena>/MAPBIOMAS_2007.shp
"""

import argparse
import math
import os
import sys
import time

from osgeo import gdal, ogr, osr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402
from gerar_mascara_mapbiomas_cena import LEGEND  # noqa: E402

gdal.UseExceptions()

# (nome_cena, caminho_máscara) — ordem igual ao editor
CENAS = [
    ("recorte_2007-09-10",
     os.path.join(caminhos.SPOT5_RECORTE_MB_2007, "MASK_MAPBIOMAS_2007.tif")),
    ("cena_013-014_2007-06-14",
     os.path.join(caminhos.SPOT5_CENA_013_MB_2007, "MASK_MAPBIOMAS_2007.tif")),
    ("colinas_do_sul",
     os.path.join(caminhos.SPOT5_COLINAS_MB_2007, "MASK_MAPBIOMAS_2007.tif")),
    ("jatai",
     os.path.join(caminhos.SPOT5_JATAI_MB_2007, "MASK_MAPBIOMAS_2007.tif")),
]

MAX_PX_DIRETO = 40_000_000   # acima disso reamostra p/ ~30m antes de vetorizar
MASK_30M = "/vsimem/mascara_30m.tif"


def abre_mascara(masc_path):
    """Máscara a vetorizar: a original se for pequena, senão reamostrada 30m."""
    ds = gdal.Open(masc_path)
    w, h = ds.RasterXSize, ds.RasterYSize
    if w * h > MAX_PX_DIRETO:
        gt = ds.GetGeoTransform()
        f = 12 if abs(gt[1]) < 3.0 else 6   # 2,5m -> 30m; 5m(recorte) n/ passa aqui
        x0, y0 = gt[0], gt[3]
        gw, gh = -(-w // f), -(-h // f)
        gdal.Warp(MASK_30M, ds, dstSRS=ds.GetProjection(), resampleAlg="near",
                  outputBounds=(x0, y0 + h * gt[5], x0 + w * gt[1], y0),
                  width=gw, height=gh, dstNodata=0, outputType=gdal.GDT_Byte,
                  creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
        ds = None
        ds = gdal.Open(MASK_30M)
        print("   reamostrada p/ ~30m: %dx%d" % (gw, gh))
    return ds


def gerar(cena, masc_path, out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    ds = abre_mascara(masc_path)
    band = ds.GetRasterBand(1)
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    gt = ds.GetGeoTransform()
    res_x, res_y = abs(gt[1]), abs(gt[5])
    # para área em m²: cenas UTM usam a geometria direto; geográficas são
    # reprojetadas para a UTM da cena (a geometria do polígono em graus²)
    geog = srs.IsGeographic()
    tx = None
    if geog:
        src_cx = gt[0] + res_x * ds.RasterXSize / 2
        src_cy = gt[3] + res_y * ds.RasterYSize / 2
        zona = int((src_cx + 180) // 6) + 1
        utm = (32700 if src_cy < 0 else 32600) + zona
        proj = osr.SpatialReference(); proj.ImportFromEPSG(int(utm))
        tx = osr.CoordinateTransformation(srs, proj)
    shp = os.path.join(out_dir, "MAPBIOMAS_2007.shp")

    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shp):
        drv.DeleteDataSource(shp)
    dst = drv.CreateDataSource(shp)
    lyr = dst.CreateLayer("mapbiomas_2007", srs, ogr.wkbMultiPolygon)
    lyr.CreateField(ogr.FieldDefn("classe", ogr.OFTInteger))
    fd = ogr.FieldDefn("nome", ogr.OFTString)
    fd.SetWidth(60)
    lyr.CreateField(fd)
    lyr.CreateField(ogr.FieldDefn("area_ha", ogr.OFTReal))

    mem = ogr.GetDriverByName("Memory").CreateDataSource("mem")
    mlyr = mem.CreateLayer("m", srs, ogr.wkbPolygon)
    mlyr.CreateField(ogr.FieldDefn("classe", ogr.OFTInteger))
    gdal.Polygonize(band, None, mlyr, 0, [], callback=None)
    band = ds = None

    n = 0
    for feat in mlyr:
        cl = int(feat.GetField("classe"))
        geom = feat.GetGeometryRef()
        if cl == 0 or geom is None:
            continue
        area = geom.GetArea()
        if tx is not None:              # geográfico: reprojeta p/ UTM p/ área em m²
            g2 = geom.Clone()
            g2.Transform(tx)
            area = g2.GetArea()
        of = ogr.Feature(lyr.GetLayerDefn())
        of.SetField("classe", cl)
        of.SetField("nome", LEGEND.get(cl, ("?",))[0])
        of.SetField("area_ha", round(area / 1e4, 3))
        of.SetGeometry(geom)
        lyr.CreateFeature(of)
        n += 1
    dst = lyr = mem = mlyr = None
    try:
        gdal.Unlink(MASK_30M)
    except Exception:
        pass
    print("OK %s -> %s (%d polígonos, %.0fs)"
          % (cena, shp, n, time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cena", default=None,
                    help="uma cena só (recorte_2007-09-10 | cena_013-014_2007-06-14 | "
                         "colinas_do_sul | jatai)")
    args = ap.parse_args()
    saidas = os.path.join(caminhos.DADOS, "shp_mapbiomas_2007")
    for cena, masc_path in CENAS:
        if args.cena and cena != args.cena:
            continue
        if not os.path.exists(masc_path):
            print("!! %s: máscara não encontrada: %s" % (cena, masc_path))
            continue
        gerar(cena, masc_path, os.path.join(saidas, cena))


if __name__ == "__main__":
    main()