#!/usr/bin/env python3
"""Cria seis atributos PAN+NDVI a partir do dataset de sete bandas.

As quatro bandas fundidas sao (PAN_z + NDVI_z) / sqrt(2), uma por estatistica
temporal. GLCM e rugosidade sao copiadas sem mudanca. Isso e fusao de atributos
para a CNN, nao reconstrucao fisica de NDVI de alta resolucao. Mascara, posicao
e splits sao identicos a origem. A saida nao pode existir antes da execucao.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/criar_dataset_fusao_pan_ndvi.py \
      --origem ../Data/dados/datasets/spot5_colinas_v28 \
      --saida ../Data/dados/datasets/spot5_colinas_v30
"""

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import ambiente  # noqa: F401  (convenção do projeto)
import numpy as np
from osgeo import gdal


ORIGINAIS = ["PAN", "GLCM_Entropia", "Rugosidade_STD_Gradiente",
             "NDVI_median", "NDVI_min", "NDVI_max", "NDVI_stdDev"]
DERIVADAS = ["PAN_NDVI_median", "PAN_NDVI_min", "PAN_NDVI_max",
             "PAN_NDVI_stdDev", "GLCM_Entropia", "Rugosidade_STD_Gradiente"]
ESCALA = np.float32(1 / math.sqrt(2))


def criar_imagem(origem, saida):
    fonte = gdal.Open(str(origem), gdal.GA_ReadOnly)
    if fonte is None or fonte.RasterCount != 7:
        raise ValueError("esperadas sete bandas em %s" % origem)
    arr = fonte.ReadAsArray().astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError("valores nao finitos em %s" % origem)
    h, w = arr.shape[1:]
    out = np.empty((6, h, w), np.float32)
    for j in range(4):
        out[j] = (arr[0] + arr[j + 3]) * ESCALA
    out[4] = arr[1]
    out[5] = arr[2]
    driver = gdal.GetDriverByName("GTiff")
    destino = driver.Create(str(saida), w, h, 6, gdal.GDT_Float32,
                            options=["TILED=YES", "COMPRESS=DEFLATE",
                                     "PREDICTOR=3", "BIGTIFF=IF_SAFER"])
    if destino is None:
        raise RuntimeError("nao foi possivel criar %s" % saida)
    destino.SetGeoTransform(fonte.GetGeoTransform())
    destino.SetProjection(fonte.GetProjection())
    for j, nome in enumerate(DERIVADAS):
        band = destino.GetRasterBand(j + 1)
        band.SetDescription(nome)
        band.WriteArray(out[j])
    destino.FlushCache()
    destino = None
    fonte = None
    if not saida.is_file() or saida.stat().st_size == 0:
        raise RuntimeError("saida vazia: %s" % saida)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origem", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--retomar-temp", default=None,
                    help="pasta .incompleto_* de uma execucao interrompida")
    args = ap.parse_args()
    gdal.UseExceptions()
    origem = Path(args.origem).resolve(strict=True)
    saida = Path(args.saida).absolute()
    if saida.exists():
        raise SystemExit("saida ja existe: %s" % saida)
    with open(origem / "receita.json", encoding="utf-8") as f:
        receita = json.load(f)
    if receita["bandas"] != ORIGINAIS:
        raise SystemExit("ordem das bandas nao bate com a formula: %s" %
                         receita["bandas"])
    with open(origem / "manifest.csv", newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    nomes = [r["arquivo"] for r in linhas]
    if len(nomes) != len(set(nomes)):
        raise SystemExit("manifest com nomes duplicados")
    receita["bandas"] = DERIVADAS
    receita["fusao_pan_ndvi"] = {
        "tipo": "atributos_aditivos_normalizados",
        "formula": "(PAN_z + NDVI_estatistica_z) / sqrt(2)",
        "ordem_fonte": ORIGINAIS,
        "ordem_saida": DERIVADAS,
        "origem": str(origem),
        "nota": "NDVI temporal Landsat permanece na resolucao informativa original; "
                "a grade de 2,5 m e apenas a grade dos atributos da CNN",
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    if args.retomar_temp:
        tmp = Path(args.retomar_temp).resolve(strict=True)
        if tmp.parent != saida.parent.resolve() or not tmp.name.startswith(saida.name + ".incompleto_"):
            raise SystemExit("--retomar-temp fora da pasta esperada")
        if not (tmp / "images").is_dir() or not (tmp / "masks").is_dir():
            raise SystemExit("pasta temporaria incompleta")
    else:
        tmp = Path(tempfile.mkdtemp(prefix=saida.name + ".incompleto_",
                                    dir=saida.parent))
        (tmp / "images").mkdir()
        (tmp / "masks").mkdir()
        shutil.copytree(origem / "splits", tmp / "splits")
        shutil.copy2(origem / "manifest.csv", tmp / "manifest.csv")
        (tmp / "receita.json").write_text(
            json.dumps(receita, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    inicio = time.time()
    for i, nome in enumerate(nomes, 1):
        source_im = origem / "images" / (nome + ".tif")
        source_mask = origem / "masks" / (nome + ".tif")
        if not source_im.is_file() or not source_mask.is_file():
            raise SystemExit("imagem ou mascara ausente: %s" % nome)
        dest_im = tmp / "images" / source_im.name
        dest_mask = tmp / "masks" / source_mask.name
        if dest_im.exists() and dest_mask.exists():
            if not os.path.samefile(source_mask, dest_mask):
                raise RuntimeError("mascara retomada nao corresponde: %s" % nome)
            d = gdal.Open(str(dest_im), gdal.GA_ReadOnly)
            if d is None or d.RasterCount != 6:
                raise RuntimeError("imagem retomada invalida: %s" % dest_im)
            d = None
        else:
            if dest_im.exists():
                dest_im.unlink()  # apenas arquivo parcialmente gravado nesta pasta validada
            criar_imagem(source_im, dest_im)
            if not dest_mask.exists():
                os.link(source_mask, dest_mask)
        if i % 250 == 0 or i == len(nomes):
            print("[fusao] %d/%d patches (%.0fs)" %
                  (i, len(nomes), time.time() - inicio), flush=True)
    if len(list((tmp / "images").iterdir())) != len(nomes) or \
            len(list((tmp / "masks").iterdir())) != len(nomes):
        raise RuntimeError("contagem final incorreta")
    os.rename(tmp, saida)
    print("[feito] %s: %d patches, %d bandas, splits identicos" %
          (saida, len(nomes), len(DERIVADAS)), flush=True)


if __name__ == "__main__":
    main()
