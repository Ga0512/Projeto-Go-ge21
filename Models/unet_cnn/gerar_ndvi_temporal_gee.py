#!/usr/bin/env python3
"""Gera quatro estatísticas de NDVI Landsat 5 para uma PAN SPOT5.

O GeoTIFF de saída fica em 30 m, EPSG da PAN. Mediana/mínimo/máximo usam
(NDVI + 1) * 10000; desvio padrão usa NDVI * 10000. Zero significa sem dado.
Essa convenção é a esperada por montar_ndvi_temporal.py e inferencia_cena.py.
Blocos são mantidos ao lado da saída para retomar downloads interrompidos.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python -u unet_cnn/gerar_ndvi_temporal_gee.py \
    --pan ../Data/dados/spot5/colinas_do_sul/original/Colinas_do_Sul_mosaico_v2.tif \
    --saida ../Data/dados/spot5/colinas_do_sul/original/NDVI_Temporal_Landsat5_Colinas_2007_2008.tif \
    --bloco 512
  ... --limite-blocos 1  # confere o primeiro download sem montar a saída
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import urllib.request

import ambiente  # noqa: F401; caches em D: antes de bibliotecas pesadas
import ee
from osgeo import gdal, osr


NOMES = ["NDVI_median", "NDVI_min", "NDVI_max", "NDVI_stdDev"]
gdal.UseExceptions()


def imagem_ndvi(colecao):
    def por_cena(imagem):
        qa = imagem.select("QA_PIXEL")
        valido = qa.bitwiseAnd(63).eq(0).And(imagem.select("QA_RADSAT").eq(0))
        vermelho = imagem.select("SR_B3").multiply(0.0000275).add(-0.2)
        nir = imagem.select("SR_B4").multiply(0.0000275).add(-0.2)
        valido = valido.And(vermelho.gt(0)).And(nir.gt(0))
        ndvi = nir.subtract(vermelho).divide(nir.add(vermelho))
        return ndvi.clamp(-1, 1).rename("NDVI").updateMask(valido)

    r = (ee.Reducer.median()
         .combine(ee.Reducer.minMax(), sharedInputs=True)
         .combine(ee.Reducer.stdDev(), sharedInputs=True))
    est = colecao.map(por_cena).reduce(r)
    primeiras = [est.select(nome).add(1).multiply(10000).round()
                 for nome in NOMES[:3]]
    desvio = est.select(NOMES[3]).multiply(10000).round()
    return ee.Image.cat(primeiras + [desvio]).rename(NOMES).toUint16().unmask(0)


def baixar_bloco(imagem, epsg, transform, largura, altura, destino):
    temp = destino.with_suffix(".part.tif")
    for tentativa in range(1, 4):
        try:
            url = imagem.getDownloadURL({
                "crs": epsg, "crs_transform": transform,
                "dimensions": [largura, altura], "format": "GEO_TIFF",
                "filePerBand": False,
            })
            with urllib.request.urlopen(url, timeout=600) as origem, temp.open("wb") as saida:
                shutil.copyfileobj(origem, saida)
            ds = gdal.Open(str(temp))
            if ds is None or ds.RasterCount != 4 or (ds.RasterXSize, ds.RasterYSize) != (largura, altura):
                raise ValueError("download não tem 4 bandas ou dimensões esperadas")
            got = ds.GetGeoTransform()
            esperado = (transform[2], transform[0], transform[1],
                        transform[5], transform[3], transform[4])
            if max(abs(a - b) for a, b in zip(got, esperado)) > 0.01:
                raise ValueError("grade do bloco difere da requisitada: %s vs %s" % (got, esperado))
            ds = None
            os.replace(temp, destino)
            return
        except Exception as erro:
            temp.unlink(missing_ok=True)
            if tentativa == 3:
                raise
            print("[tentativa %d] %s: %s" % (tentativa, destino.name, erro), flush=True)
            time.sleep(5 * tentativa)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pan", required=True, type=Path)
    ap.add_argument("--saida", required=True, type=Path)
    ap.add_argument("--inicio", default="2007-01-01")
    ap.add_argument("--fim", default="2009-01-01", help="fim exclusivo")
    ap.add_argument("--projeto", default="beaming-botany-373619")
    ap.add_argument("--bloco", type=int, default=512)
    ap.add_argument("--deadline-s", type=int, default=180,
                    help="tempo máximo por pedido à API Earth Engine")
    ap.add_argument("--limite-blocos", type=int, default=0)
    a = ap.parse_args()
    if a.bloco < 128 or a.bloco > 1536:
        ap.error("--bloco deve ficar entre 128 e 1536 (limite de 32 MB do Earth Engine)")
    pan = gdal.Open(str(a.pan))
    if pan is None:
        ap.error("PAN não encontrada: " + str(a.pan))
    gt = pan.GetGeoTransform()
    if gt[1] <= 0 or gt[5] >= 0 or gt[2] != 0 or gt[4] != 0:
        ap.error("PAN precisa de grade norte-acima sem rotação")
    srs = osr.SpatialReference(wkt=pan.GetProjection())
    epsg = "EPSG:" + str(srs.GetAuthorityCode(None))
    x0, x1 = gt[0], gt[0] + pan.RasterXSize * gt[1]
    y1, y0 = gt[3], gt[3] + pan.RasterYSize * gt[5]
    resolucao = 30
    xmin, ymax = math.floor(x0 / resolucao) * resolucao, math.ceil(y1 / resolucao) * resolucao
    largura = math.ceil((x1 - xmin) / resolucao)
    altura = math.ceil((ymax - y0) / resolucao)
    print("[grade] %s %dx%d em %d m | blocos %d"
          % (epsg, largura, altura, resolucao, a.bloco), flush=True)
    ee.Initialize(project=a.projeto)
    ee.data.setDeadline(a.deadline_s * 1000)
    area = ee.Geometry.Rectangle([x0, y0, x1, y1], ee.Projection(epsg), False)
    colecao = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
               .filterBounds(area).filterDate(a.inicio, a.fim))
    n_cenas = colecao.size().getInfo()
    if n_cenas == 0:
        raise RuntimeError("sem cenas Landsat 5 para a área/período")
    print("[landsat] %d cenas entre %s e %s" % (n_cenas, a.inicio, a.fim), flush=True)
    imagem = imagem_ndvi(colecao)
    if imagem.bandNames().getInfo() != NOMES:
        raise RuntimeError("bandas do composto fora da ordem esperada")
    pasta = a.saida.parent / (a.saida.stem + "_blocos_%d" % a.bloco)
    pasta.mkdir(parents=True, exist_ok=True)
    blocos = []
    t0 = time.monotonic()
    for linha in range(0, altura, a.bloco):
        for coluna in range(0, largura, a.bloco):
            w, h = min(a.bloco, largura - coluna), min(a.bloco, altura - linha)
            nome = pasta / ("bloco_%04d_%04d.tif" % (linha, coluna))
            transform = [resolucao, 0, xmin + coluna * resolucao,
                         0, -resolucao, ymax - linha * resolucao]
            if not nome.is_file():
                # Aproveita um bloco piloto maior se ele cobre este recorte.
                antigo = a.saida.parent / (a.saida.stem + "_blocos") / (
                    "bloco_%04d_%04d.tif" % ((linha // 1024) * 1024,
                                              (coluna // 1024) * 1024))
                origem = gdal.Open(str(antigo)) if antigo.is_file() else None
                dx, dy = coluna % 1024, linha % 1024
                if origem is not None and dx + w <= origem.RasterXSize and dy + h <= origem.RasterYSize:
                    gdal.Translate(str(nome), origem, srcWin=[dx, dy, w, h],
                                   creationOptions=["TILED=YES", "COMPRESS=LZW"])
                    origem = None
                else:
                    origem = None
                    baixar_bloco(imagem, epsg, transform, w, h, nome)
            blocos.append((linha, coluna, w, h, nome))
            print("[bloco] %d/%d %s | %.0fs" % (len(blocos),
                  math.ceil(largura / a.bloco) * math.ceil(altura / a.bloco),
                  nome.name, time.monotonic() - t0), flush=True)
            if a.limite_blocos and len(blocos) >= a.limite_blocos:
                print("[piloto] bloco conferido; rode sem --limite-blocos", flush=True)
                return
    temp = a.saida.with_suffix(".tmp.tif")
    temp.unlink(missing_ok=True)
    out = gdal.GetDriverByName("GTiff").Create(str(temp), largura, altura, 4,
        gdal.GDT_UInt16, ["TILED=YES", "COMPRESS=LZW", "PREDICTOR=2", "BIGTIFF=IF_SAFER"])
    out.SetGeoTransform((xmin, resolucao, 0, ymax, 0, -resolucao))
    out.SetProjection(pan.GetProjection())
    for k, nome in enumerate(NOMES, 1):
        out.GetRasterBand(k).SetDescription(nome)
        out.GetRasterBand(k).SetNoDataValue(0)
    for linha, coluna, w, h, nome in blocos:
        ds = gdal.Open(str(nome))
        dados = ds.ReadAsArray()
        for k in range(4):
            out.GetRasterBand(k + 1).WriteArray(dados[k], coluna, linha)
        ds = None
    out = None
    os.replace(temp, a.saida)
    manifesto = {"fonte": "LANDSAT/LT05/C02/T1_L2", "inicio": a.inicio,
                 "fim_exclusivo": a.fim, "cenas": n_cenas, "resolucao_m": resolucao,
                 "epsg": epsg, "bandas": NOMES,
                 "escala": "(bruto-10000)/10000 nas três primeiras; bruto/10000 na stdDev",
                 "mascara_qa_pixel_bits": [0, 1, 2, 3, 4, 5],
                 "qa_radsat_zero": True, "pan": str(a.pan.resolve())}
    a.saida.with_suffix(".json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[pronto] %s | %dx%d | %.0fs" % (a.saida, largura, altura,
          time.monotonic() - t0), flush=True)


if __name__ == "__main__":
    main()
