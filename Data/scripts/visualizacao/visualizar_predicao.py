#!/usr/bin/env python3
"""Visualização da máscara predita pelo XGBoost (cena inteira 0,5m).

Lê mascara_xgb.tif (códigos MapBiomas) + banda PAN + máscara MapBiomas
original + classificação manual, reduz para uma largura máxima e gera PNGs
em <modelo_dir>/visualizacao/:
  pan_pred.png              — PAN (contraste percentil 2-98)
  pred.png                  — máscara predita colorida (3 classes)
  orig.png                  — máscara MapBiomas remapeada (3 classes)
  manual.png                — classificação manual rasterizada
  lado_a_lado_pred.png      — PAN | PREDIÇÃO
  lado_a_lado_orig_pred.png — ORIGINAL | PREDIÇÃO
  lado_a_lado_manual_pred.png — MANUAL | PREDIÇÃO

Uso:
  .venv/bin/python scripts/visualizacao/visualizar_predicao.py \
      --modelo-dir modelos/xgb_manual
"""

import argparse
import os
import sys

import numpy as np
from osgeo import gdal
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

ROOT = caminhos.RAIZ   # caminhos relativos da CLI partem da raiz do projeto

PAN = caminhos.WV1_FEATURES
PRED = os.path.join(caminhos.MODELOS, "xgb_01", "mascara_xgb.tif")
MASK_ORIG = caminhos.WV1_MASK_SUAVIZADA
SHP = caminhos.WV1_SHP_MANUAL
OUT = os.path.join(caminhos.MODELOS, "xgb_01", "visualizacao")
LARGURA_MAX = 4096

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelos"))
import config  # noqa: E402

# classes do modelo exibidas pelo código representativo (config.REPR_CODIGO),
# com as cores oficiais do MapBiomas desses códigos
_CORES = {3: "#1f8d49", 4: "#7dc975", 15: "#edde8e", 33: "#2532e4"}
LEGEND = {0: ("Nodata", "#000000")}
LEGEND.update({c: (n, _CORES[c]) for c, n in zip(config.REPR_CODIGO, config.CLASS_NOMES)})

# código MapBiomas -> código representativo da sua classe
CLASS_MAP = {code: config.REPR_CODIGO[cl] for code, cl in config.CLASS_MAP.items()}

# manual -> código de exibição (mesmo esquema de cores)
SHP_MAP = {"Outros": 3, "Área consolidada": 15, "Corpos d'água": 33}


def ler_reduzido(path, banda, largura_max):
    src = gdal.Open(path)
    nx, ny = src.RasterXSize, src.RasterYSize
    fator = max(1, nx // largura_max)
    w, h = nx // fator, ny // fator
    arr = src.GetRasterBand(banda).ReadAsArray(0, 0, nx, ny, w, h)
    src = None
    return np.asarray(arr)


def rasterizar_manual():
    """Rasteriza o shapefile de classificação manual na grade da predição
    (EPSG:32722, 0,5m), classes via SHP_MAP."""
    from osgeo import gdal, ogr
    tmp = "/vsimem/manual_32722.shp"
    gdal.VectorTranslate(tmp, SHP, format="ESRI Shapefile",
                         dstSRS="EPSG:32722")
    shp = ogr.Open(tmp)
    lyr = shp.GetLayer(0)

    ref = gdal.Open(PRED)
    nx, ny = ref.RasterXSize, ref.RasterYSize
    geo = ref.GetGeoTransform()
    ref = None

    drv = gdal.GetDriverByName("MEM")
    dst = drv.Create("manual", nx, ny, 1, gdal.GDT_Byte)
    dst.SetGeoTransform(geo)
    dst.SetProjection("EPSG:32722")
    band = dst.GetRasterBand(1)
    band.Fill(0)
    for nome, val in SHP_MAP.items():
        lyr.SetAttributeFilter("Classe = '%s'" % nome.replace("'", "''"))
        if lyr.GetFeatureCount() == 0:
            continue
        gdal.RasterizeLayer(dst, [1], lyr, burn_values=[val])
    arr = band.ReadAsArray()
    dst = None
    shp = None
    gdal.Unlink(tmp)
    return arr


def colorir(pred):
    h, w = pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, (_, hexc) in LEGEND.items():
        rgb[pred == c] = np.array(
            [int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)],
            dtype=np.uint8)
    return Image.fromarray(rgb)


def legenda():
    """Tira de legenda (quadrados de cor + nome) na base das imagens."""
    from PIL import ImageDraw, ImageFont
    try:
        fonte = ImageFont.load_default(size=30)
    except TypeError:
        fonte = ImageFont.load_default()
    nomes = {c: LEGEND[c][0] for c in config.REPR_CODIGO}
    item_w, item_h, pad = 330, 44, 24
    larg = item_w * len(nomes) + pad * (len(nomes) + 1)
    img = Image.new("RGB", (larg, 66), "white")
    d = ImageDraw.Draw(img)
    x = pad
    for c, nome in nomes.items():
        hexc = LEGEND[c][1]
        cor = tuple(int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        d.rectangle([x, 14, x + 34, 48], fill=cor, outline="black")
        d.text((x + 44, 16), nome, fill="black", font=fonte)
        x += item_w
    return img


def com_legenda(img, lado_a_lado=False):
    """Cola a legenda abaixo da imagem (ou na lateral, p/ lado a lado)."""
    lg = legenda()
    if lado_a_lado:
        base = Image.new("RGB", (img.width, img.height + lg.height + 10),
                         "white")
        base.paste(img, (0, 0))
        base.paste(lg, ((img.width - lg.width) // 2, img.height + 5))
        return base
    base = Image.new("RGB", (img.width + lg.width + 10, img.height), "white")
    base.paste(img, (0, 0))
    base.paste(lg, (img.width + 5, (img.height - lg.height) // 2))
    return base


def main(args=None):
    global PRED, OUT, MASK_ORIG, SHP
    if args is not None:
        modelo_dir = os.path.join(ROOT, args.modelo_dir)
        PRED = os.path.join(modelo_dir, "mascara_xgb.tif")
        OUT = os.path.join(modelo_dir, "visualizacao")
    if args is not None and args.mask_orig:
        MASK_ORIG = os.path.join(ROOT, args.mask_orig)
    if args is not None and args.shp:
        SHP = os.path.join(ROOT, args.shp)
    os.makedirs(OUT, exist_ok=True)
    pan = ler_reduzido(PAN, 1, LARGURA_MAX).astype(np.float32)
    pred = ler_reduzido(PRED, 1, LARGURA_MAX).astype(np.uint8)
    orig_raw = ler_reduzido(MASK_ORIG, 1, LARGURA_MAX).astype(np.uint8)
    h, w = pred.shape
    print("pan: %dx%d | pred: %dx%d | orig: %dx%d"
          % (pan.shape[1], pan.shape[0], w, h,
             orig_raw.shape[1], orig_raw.shape[0]))

    orig = np.zeros_like(pred)
    for code, cl in CLASS_MAP.items():
        orig[orig_raw == code] = cl

    lo, hi = np.percentile(pan[pan > 0], (2, 98))
    gray = np.clip((pan - lo) / (hi - lo), 0, 1)
    pan_im = Image.fromarray((gray * 255).astype(np.uint8))
    pan_im.save(os.path.join(OUT, "pan_pred.png"))

    pred_im = colorir(pred)
    pred_im.save(os.path.join(OUT, "pred.png"))
    orig_im = colorir(orig)
    orig_im.save(os.path.join(OUT, "orig.png"))

    lado = Image.new("RGB", (w * 2, h), "black")
    lado.paste(pan_im, (0, 0))
    lado.paste(pred_im, (w, 0))
    lado.save(os.path.join(OUT, "lado_a_lado_pred.png"))

    # lado a lado: máscara original (MapBiomas remapeada p/ 3 classes) | predito
    lado2 = Image.new("RGB", (w * 2, h), "black")
    lado2.paste(orig_im, (0, 0))
    lado2.paste(pred_im, (w, 0))
    com_legenda(lado2, True).save(
        os.path.join(OUT, "lado_a_lado_orig_pred.png"))

    # lado a lado: classificação manual (shapefile) | predito
    manual = rasterizar_manual()
    fator = max(1, manual.shape[1] // LARGURA_MAX)
    mw, mh = manual.shape[1] // fator, manual.shape[0] // fator
    manual_small = manual[::fator, ::fator][:mh, :mw]
    manual_im = colorir(manual_small)
    manual_im.save(os.path.join(OUT, "manual.png"))
    lado3 = Image.new("RGB", (w * 2, h), "black")
    lado3.paste(manual_im, (0, 0))
    lado3.paste(pred_im, (w, 0))
    com_legenda(lado3, True).save(
        os.path.join(OUT, "lado_a_lado_manual_pred.png"))
    uniq, counts = np.unique(manual, return_counts=True)
    print("manual: %s" % {int(c): int(n) for c, n in zip(uniq, counts)})

    uniq, counts = np.unique(pred, return_counts=True)
    print("pred: %s" % {int(c): int(n) for c, n in zip(uniq, counts)})
    uniq, counts = np.unique(orig, return_counts=True)
    print("orig: %s" % {int(c): int(n) for c, n in zip(uniq, counts)})
    print("OK -> %s" % OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Visualização da predição XGBoost")
    ap.add_argument("--modelo-dir",
                    default="modelos/xgb_01",
                    help="dir do modelo (mascara_xgb.tif + visualizacao/)")
    ap.add_argument("--mask-orig", default=None,
                    help="máscara MapBiomas p/ comparação (default: suavizada 2007)")
    ap.add_argument("--shp", default=None,
                    help="shapefile manual p/ comparação")
    args = ap.parse_args()
    main(args)