#!/usr/bin/env python3
"""Reprojeção e normalização do NDVI temporal para treino e inferência.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python -c 'import sys; sys.path.insert(0, "unet_cnn"); import ndvi_temporal'
"""

import ambiente  # noqa: F401: configurar caches antes de bibliotecas numéricas
import numpy as np
from osgeo import gdal


def ler_extra(src, proj, gt, x0, y0, w, h):
    """Banda auxiliar reprojetada na grade exata da janela pedida."""
    bnd = (gt[0] + x0 * gt[1], gt[3] + (y0 + h) * gt[5],
           gt[0] + (x0 + w) * gt[1], gt[3] + y0 * gt[5])
    d = gdal.Warp("", src, format="VRT", dstSRS=proj, outputBounds=bnd,
                  width=w, height=h, resampleAlg="bilinear",
                  srcNodata=0, dstNodata=0, outputType=gdal.GDT_Float32)
    a = d.GetRasterBand(1).ReadAsArray().astype(np.float32)
    d = None
    return a


def preparar_ndvi_temporal(caminho):
    """Abre quatro bandas e calcula o z-score usado no treino e inferência."""
    src = gdal.Open(caminho)
    if src is None or src.RasterCount != 4:
        raise SystemExit("--ndvi-temporal precisa de GeoTIFF com quatro bandas")
    vrts, estatisticas = [], []
    for k in range(4):
        cru = src.GetRasterBand(k + 1).ReadAsArray().astype(np.float32)
        valido = cru != 0
        if not valido.any():
            raise SystemExit("NDVI temporal: banda %d sem cobertura" % (k + 1))
        valores = (cru[valido] / 10000.0 if k == 3 else
                   (cru[valido] - 10000.0) / 10000.0)
        estatisticas.append((float(valores.mean()), float(valores.std())))
        vrts.append(gdal.Translate("", src, format="VRT", bandList=[k + 1],
                                   noData=0))
        del cru, valido, valores
    return src, vrts, estatisticas


def ler_ndvi_temporal(vrts, estatisticas, proj, gt, x0, y0, w, h):
    """Reprojeta e padroniza quatro bandas; fora da cobertura resulta zero."""
    extras = []
    for k, vrt in enumerate(vrts):
        cru = ler_extra(vrt, proj, gt, x0, y0, w, h)
        valido = cru != 0
        z = np.zeros((h, w), np.float32)
        if valido.any():
            valores = (cru[valido] / 10000.0 if k == 3 else
                       (cru[valido] - 10000.0) / 10000.0)
            media, desvio = estatisticas[k]
            z[valido] = (valores - media) / max(desvio, 1e-9)
        extras.append(z)
    return np.stack(extras)
