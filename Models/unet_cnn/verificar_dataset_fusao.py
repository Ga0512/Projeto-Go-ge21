#!/usr/bin/env python3
"""Confere bandas, formula, georreferencia e splits do dataset fundido.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/verificar_dataset_fusao.py \
      --origem ../Data/dados/datasets/spot5_colinas_v28 \
      --fundido ../Data/dados/datasets/spot5_colinas_v30
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import ambiente  # noqa: F401  (convenção do projeto)
import numpy as np
from osgeo import gdal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origem", required=True)
    ap.add_argument("--fundido", required=True)
    ap.add_argument("--parcial", action="store_true",
                    help="confere apenas as imagens ja gravadas no diretorio temporario")
    args = ap.parse_args()
    gdal.UseExceptions()
    origem, fundido = Path(args.origem).resolve(strict=True), Path(args.fundido).resolve(strict=True)
    with open(fundido / "receita.json", encoding="utf-8") as f:
        receita = json.load(f)
    if len(receita["bandas"]) != 6:
        raise RuntimeError("esperadas seis bandas")
    for f in ("manifest.csv", "splits/treino.txt", "splits/validacao.txt",
              "splits/teste.txt"):
        if sha(origem / f) != sha(fundido / f):
            raise RuntimeError("arquivo difere da origem: %s" % f)
    entrada = {p.name for p in (origem / "images").glob("*.tif")}
    saida = {p.name for p in (fundido / "images").glob("*.tif")}
    masks = {p.name for p in (fundido / "masks").glob("*.tif")}
    if not saida.issubset(entrada):
        raise RuntimeError("imagem fundida sem correspondencia na origem")
    if args.parcial:
        if not masks.issubset(saida) or len(saida - masks) > 1:
            raise RuntimeError("mascaras parciais nao acompanham imagens")
    elif entrada != saida or saida != masks:
        raise RuntimeError("contagens ou nomes de imagens/mascaras diferem")
    nomes = sorted(entrada)
    if args.parcial:
        nomes = sorted(saida)[:-10]  # evita o arquivo mais recente, possivelmente aberto
    amostra = [nomes[i] for i in np.linspace(0, len(nomes) - 1,
                                            min(40, len(nomes)), dtype=int)]
    max_erro = 0.0
    soma = np.zeros(6, np.float64)
    soma2 = np.zeros(6, np.float64)
    n_pixels = 0
    for n in amostra:
        a = gdal.Open(str(origem / "images" / n))
        b = gdal.Open(str(fundido / "images" / n))
        if a.RasterCount != 7 or b.RasterCount != 6:
            raise RuntimeError("numero de bandas incorreto: %s" % n)
        if a.GetGeoTransform() != b.GetGeoTransform() or a.GetProjection() != b.GetProjection():
            raise RuntimeError("grade/georreferencia diferente: %s" % n)
        orig, fus = a.ReadAsArray(), b.ReadAsArray()
        if not np.isfinite(fus).all():
            raise RuntimeError("valores nao finitos na saida: %s" % n)
        esperado = np.concatenate(
            [((orig[0:1] + orig[3:7]) / math.sqrt(2)), orig[1:3]], axis=0)
        max_erro = max(max_erro, float(np.max(np.abs(fus - esperado))))
        soma += fus.sum(axis=(1, 2), dtype=np.float64)
        soma2 += np.square(fus, dtype=np.float64).sum(axis=(1, 2))
        n_pixels += fus.shape[1] * fus.shape[2]
        if not os.path.samefile(origem / "masks" / n, fundido / "masks" / n):
            raise RuntimeError("mascara nao e hard link: %s" % n)
        a = b = None
    if max_erro > 2e-6:
        raise RuntimeError("formula divergiu: erro maximo %g" % max_erro)
    rel = {"origem": str(origem), "fundido": str(fundido),
           "imagens": len(nomes), "mascaras": len(masks),
           "bandas": receita["bandas"], "amostras_verificadas": len(amostra),
           "erro_maximo_formula": max_erro,
           "media_bandas_amostra": (soma / n_pixels).tolist(),
           "desvio_bandas_amostra":
               np.sqrt(np.maximum(0, soma2 / n_pixels - (soma / n_pixels) ** 2)).tolist(),
           "splits_manifest_identicos": True, "georreferencia_identica": True,
           "parcial": args.parcial}
    (fundido / "verificacao.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[OK] %d imagens, %d bandas, formula erro maximo %.2g" %
          (len(nomes), len(receita["bandas"]), max_erro), flush=True)


if __name__ == "__main__":
    main()
