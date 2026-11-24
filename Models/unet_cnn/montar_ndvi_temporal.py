#!/usr/bin/env python3
"""Acrescenta NDVI temporal (4 bandas) a um dataset de patches de 3 bandas.

Reusa a reprojeção, escala e normalização da inferencia_cena.py. Mantém
exatamente as imagens PAN/texturas, máscaras e splits do dataset original.
É retomável: imagens completas de 7 bandas são puladas na próxima execução.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python -u unet_cnn/montar_ndvi_temporal.py \
    --origem ../Data/dados/datasets/spot5_jatai_v24 \
    --saida ../Data/dados/datasets/spot5_jatai_v27 \
    --ndvi ../Data/dados/spot5/jatai/original/NDVI_Temporal_Landsat5_Jatai_2007_2008.tif
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import ambiente  # noqa: F401: antes de torch, importado por inferencia_cena
import numpy as np
from osgeo import gdal

from inferencia_cena import ler_ndvi_temporal, preparar_ndvi_temporal


NOMES = ["NDVI_median", "NDVI_min", "NDVI_max", "NDVI_stdDev"]
gdal.UseExceptions()


def imagem_pronta(caminho, largura, altura):
    if not caminho.is_file():
        return False
    try:
        ds = gdal.Open(str(caminho))
        ok = ds is not None and ds.RasterCount == 7 and (
            ds.RasterXSize, ds.RasterYSize) == (largura, altura)
        ds = None
        return ok
    except RuntimeError:
        return False


def copiar_arvore(origem, destino):
    for arquivo in origem.rglob("*"):
        if not arquivo.is_file():
            continue
        alvo = destino / arquivo.relative_to(origem)
        alvo.parent.mkdir(parents=True, exist_ok=True)
        if not alvo.is_file() or alvo.stat().st_size != arquivo.stat().st_size:
            shutil.copy2(arquivo, alvo)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origem", required=True, type=Path)
    ap.add_argument("--saida", required=True, type=Path)
    ap.add_argument("--ndvi", required=True, type=Path)
    ap.add_argument("--limite", type=int, default=0, help="gera só N imagens para inspeção")
    a = ap.parse_args()
    origem, saida, ndvi = a.origem.resolve(), a.saida.resolve(), a.ndvi.resolve()
    if origem == saida:
        ap.error("origem e saída devem ser pastas diferentes")
    if not ndvi.is_file():
        ap.error("NDVI não encontrado: " + str(ndvi))
    receita = json.loads((origem / "receita.json").read_text(encoding="utf-8"))
    if receita["bandas"] != ["PAN", "GLCM_Entropia", "Rugosidade_STD_Gradiente"]:
        ap.error("a origem precisa ter as três bandas PAN/GLCM/rugosidade")

    nomes = sorted((origem / "images").glob("*.tif"))
    if not nomes:
        ap.error("origem sem imagens")
    alvo_imagens = saida / "images"
    alvo_imagens.mkdir(parents=True, exist_ok=True)
    src, vrts, estatisticas = preparar_ndvi_temporal(str(ndvi))
    print("[ndvi] " + ", ".join("%s %.5f/%.5f" % (nome, *est)
                               for nome, est in zip(NOMES, estatisticas)), flush=True)

    t0, vazios, novos = time.monotonic(), 0, 0
    processar = nomes[:a.limite] if a.limite else nomes
    for i, imagem in enumerate(processar, 1):
        entrada = gdal.Open(str(imagem))
        if entrada.RasterCount != 3:
            raise ValueError("patch sem 3 bandas: " + str(imagem))
        w, h = entrada.RasterXSize, entrada.RasterYSize
        destino = alvo_imagens / imagem.name
        if imagem_pronta(destino, w, h):
            pronto = gdal.Open(str(destino))
            if a.limite:
                esperado = ler_ndvi_temporal(vrts, estatisticas,
                                               entrada.GetProjection(),
                                               entrada.GetGeoTransform(), 0, 0, w, h)
                if (not np.array_equal(pronto.ReadAsArray()[:3], entrada.ReadAsArray())
                        or not np.array_equal(pronto.ReadAsArray()[3:], esperado)
                        or pronto.GetGeoTransform() != entrada.GetGeoTransform()
                        or pronto.GetProjection() != entrada.GetProjection()):
                    raise ValueError("piloto diferente da origem/inferência: " + str(destino))
            vazios += int(all(not np.any(pronto.GetRasterBand(k).ReadAsArray())
                              for k in range(4, 8)))
            pronto = None
            entrada = None
            continue
        dados = entrada.ReadAsArray().astype(np.float32)
        extras = ler_ndvi_temporal(vrts, estatisticas, entrada.GetProjection(),
                                   entrada.GetGeoTransform(), 0, 0, w, h)
        if not np.isfinite(extras).all():
            raise ValueError("NDVI não finito: " + str(imagem))
        vazios += int(not np.any(extras))
        temp = destino.with_suffix(".tmp.tif")
        temp.unlink(missing_ok=True)
        d = gdal.GetDriverByName("GTiff").Create(
            str(temp), w, h, 7, gdal.GDT_Float32,
            options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3",
                     "BIGTIFF=IF_SAFER", "NUM_THREADS=2"])
        d.SetGeoTransform(entrada.GetGeoTransform())
        d.SetProjection(entrada.GetProjection())
        for k, arr in enumerate(np.concatenate([dados, extras])):
            b = d.GetRasterBand(k + 1)
            b.WriteArray(arr)
            b.SetDescription((receita["bandas"] + NOMES)[k])
        d.FlushCache()
        d = None
        entrada = None
        os.replace(temp, destino)
        novos += 1
        if i % 100 == 0 or i == len(processar):
            print("%d/%d | novos %d | sem NDVI %d | %.0fs" %
                  (i, len(processar), novos, vazios, time.monotonic() - t0),
                  flush=True)
    src, vrts = None, None
    if a.limite:
        print("[piloto] imagens geradas; rode sem --limite para completar", flush=True)
        return

    if len(list(alvo_imagens.glob("*.tif"))) != len(nomes):
        raise RuntimeError("número de imagens diferente da origem")
    for pasta in ("masks", "splits", "visualizacao"):
        if (origem / pasta).is_dir():
            copiar_arvore(origem / pasta, saida / pasta)
    shutil.copy2(origem / "manifest.csv", saida / "manifest.csv")
    receita["bandas"] = receita["bandas"] + NOMES
    receita["derivado_de"] = {
        "dataset": origem.name,
        "bandas_extra": NOMES,
        "fonte_ndvi": os.path.relpath(ndvi, origem.parent.parent.parent),
        "normalizacao": "mesmas funções de unet_cnn/inferencia_cena.py",
        "media_desvio_ndvi": {nome: {"media": est[0], "desvio": est[1]}
                              for nome, est in zip(NOMES, estatisticas)},
        "patches_e_splits": "idênticos ao dataset de origem",
    }
    temp_receita = saida / "receita.json.tmp"
    temp_receita.write_text(json.dumps(receita, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    os.replace(temp_receita, saida / "receita.json")
    print("[pronto] %d patches, %d novos, %d sem NDVI; %.0fs" %
          (len(nomes), novos, vazios, time.monotonic() - t0), flush=True)


if __name__ == "__main__":
    main()
