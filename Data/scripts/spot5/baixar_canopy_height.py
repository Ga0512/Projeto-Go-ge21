#!/usr/bin/env python3
"""Baixa a altura de dossel do Meta/WRI (CanopyHeight, 1 m) para tiles
refinados, na grade 2,5 m de cada tile, e salva PNG (PAN × altura) + GeoTIFF.

Asset: projects/meta-forest-monitoring-okw37/assets/CanopyHeight (banda
"cover_code" = altura em metros). App:
https://meta-forest-monitoring-okw37.projects.earthengine.app/view/canopyheight

Uso:
  .venv/bin/python scripts/spot5/baixar_canopy_height.py --todos
  .venv/bin/python scripts/spot5/baixar_canopy_height.py --tile dados/spot5/<cena>/tiles/<rRR_cCC>
Saída: experimentos/canopy_height/canopy_<cena>_tiles_<tile>.png (+ .tif, _grade.tif)
"""
import argparse
import glob
import os
import urllib.request

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
ASSET = "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
PROJ = "beaming-botany-373619"
OUT = "experimentos/canopy_height"
CENAS = ("cena_013-014_2007-06-14", "colinas_do_sul", "jatai")


def footprint_lonlat(gt, w, h, wkt):
    src = osr.SpatialReference(wkt=wkt)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tx = osr.CoordinateTransformation(src, dst)
    x0, y1 = gt[0], gt[3] + h * gt[5]
    x1, y0 = gt[0] + w * gt[1], gt[3]
    pts = [tx.TransformPoint(x, y) for x, y in
           ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return pts


def tiles_refinados():
    out = []
    for cena in CENAS:
        base = os.path.join("dados", "spot5", cena, "tiles")
        for m in sorted(glob.glob(os.path.join(base, "*", "mascara_refinada.tif"))):
            out.append(os.path.join(base, os.path.basename(os.path.dirname(m))))
    return out


def processar(ee, tile):
    img_tif = os.path.join(tile, "imagem.tif")
    ds = gdal.Open(img_tif)
    gt, w, h, wkt = ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize, ds.GetProjection()
    pan = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None

    pts = footprint_lonlat(gt, w, h, wkt)
    roi = ee.Geometry.Polygon([[[p[0], p[1]] for p in pts]])
    canopy = ee.ImageCollection(ASSET).mosaic().select(0).rename("altura")

    x0, dx, _, y0, _, dy = gt
    url = canopy.clip(roi).getDownloadURL({
        "name": "canopy", "crs": "EPSG:32722",
        "crs_transform": [dx, 0, x0, 0, dy, y0],
        "dimensions": "%dx%d" % (w, h), "format": "GEO_TIFF"})
    os.makedirs(OUT, exist_ok=True)
    nome = "_".join(os.path.normpath(tile).split(os.sep)[-3:])
    tmp = os.path.join(OUT, "canopy_%s.tif" % nome)
    urllib.request.urlretrieve(url, tmp)
    d = gdal.Open(tmp)
    alt = d.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    d = None
    print("%-40s min/max/média = %d/%d/%.2f m"
          % (nome, int(alt.min()), int(alt.max()), float(alt.mean())))

    dd = gdal.GetDriverByName("GTiff").Create(
        os.path.join(OUT, "canopy_%s_grade.tif" % nome), w, h, 1, gdal.GDT_Byte,
        ["COMPRESS=DEFLATE", "TILED=YES"])
    dd.SetGeoTransform(gt); dd.SetProjection(wkt)
    b = dd.GetRasterBand(1); b.SetNoDataValue(0)
    b.WriteArray(alt); dd = None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p2, p98 = np.percentile(pan, [2, 98])
    pg = np.clip((pan - p2) / (p98 - p2), 0, 1)
    vmax = max(5, min(35, float(np.percentile(alt, 99))))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(pg, cmap="gray"); ax[0].set_title("PAN — %s" % nome)
    im = ax[1].imshow(alt, cmap="viridis", vmin=0, vmax=vmax)
    ax[1].set_title("Altura de dossel Meta/WRI (m)")
    fig.colorbar(im, ax=ax[1], shrink=0.7, label="m")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "canopy_%s.png" % nome), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", help="pasta de um tile (com imagem.tif)")
    ap.add_argument("--todos", action="store_true",
                    help="todos os tiles refinados das cenas SPOT5")
    args = ap.parse_args()
    if not args.tile and not args.todos:
        ap.error("informe --tile ou --todos")

    import ee
    ee.Initialize(project=PROJ)

    alvos = tiles_refinados() if args.todos else [args.tile]
    for t in alvos:
        try:
            processar(ee, t)
        except Exception as e:
            print("FALHOU %s: %s" % (t, str(e)[:150]))


if __name__ == "__main__":
    main()