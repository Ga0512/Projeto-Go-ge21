#!/usr/bin/env python3
"""Gera a máscara MapBiomas (30 m: Coleção 10.1 Landsat; 10 m: Coleção 3
integração) na grade exata de uma cena.

Genérico para qualquer imagem (CRS projetado ou geográfico, qualquer tamanho).
Feito para cenas grandes (ex.: SPOT5 29488x29067 a 2,5 m):

  1. Lê CRS/transform/dimensões da imagem.
  2. Baixa o MapBiomas pelo GEE numa grade GROSSA ALINHADA à imagem: mesmo
     CRS, mesma origem, pixel = FATOR * pixel da imagem (FATOR = round(RES /
     pixel): 12 para 30 m a 2,5 m, 4 para 10 m a 2,5 m). Cada pixel grosso
     cobre exatamente FATOR x FATOR pixels da imagem -> a "reamostragem" para
     a grade fina é só repetição (nearest exato, sem ambiguidade de borda). O
     GEE reprojeta a classificação com nearest (default). Download em blocos
     se preciso.
  3. Escreve a máscara na grade da imagem em faixas de linhas (memória
     baixa), zerando onde a imagem é nodata (0). GDT_Byte, TILED, DEFLATE.
  4. Valida (CRS/transform/dimensões iguais, códigos oficiais) e gera um
     quicklook PNG com a legenda oficial.

Fontes (não inventar):
  30 m: coleção 10.1, asset por ano
        projects/mapbiomas-public/assets/brazil/lulc/v1/collection_10_1_<ano>_v1,
        banda "classification" (1985–2024).
  10 m: coleção 3, asset único
        projects/mapbiomas-public/assets/brazil/lulc_10m/collection3/
        mapbiomas_10m_collection3_integration_v1, banda "classification_<ano>"
        (2017–2024). Usar ano 2024.

Uso:
  .venv/bin/python scripts/spot5/gerar_mascara_mapbiomas_cena.py \
      --imagem dados/spot5/cena_013-014_2007-06-14/original/013-014_S5_708-378-0_2007-06-14_modificado.tif \
      --ano 2007 --res 30
  (saída default: <pasta da cena>/mapbiomas_<ano>_<res>m/, a pasta da cena é
   a mãe de original/)
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
from osgeo import gdal, osr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

gdal.UseExceptions()

PROJ = "beaming-botany-373619"
MB_COLL_30M = "projects/mapbiomas-public/assets/brazil/lulc/v1"
MB_ASSET_10M = ("projects/mapbiomas-public/assets/brazil/lulc_10m/collection3/"
                "mapbiomas_10m_collection3_integration_v1")
BLOCO_GROSSO = 4096      # px grossos por bloco de download (limite do GEE)

LEGEND = {
    0: ("Nodata / Fora da imagem", "#ffffff"),
    3: ("Formação Florestal", "#1f8d49"),
    4: ("Formação Savânica", "#7dc975"),
    5: ("Mangue", "#04381d"),
    6: ("Floresta Alagável", "#026975"),
    9: ("Silvicultura", "#7a5900"),
    11: ("Campo Alagado e Área Pantanosa", "#519799"),
    12: ("Formação Campestre", "#d6bc74"),
    15: ("Pastagem", "#edde8e"),
    19: ("Lavoura Temporária", "#c27ba0"),
    20: ("Cana", "#db7093"),
    21: ("Mosaico de Usos", "#ffefc3"),
    23: ("Praia, Duna e Areal", "#ffa07a"),
    24: ("Área Urbanizada", "#d4271e"),
    25: ("Outras Áreas não Vegetadas", "#db4d4f"),
    29: ("Afloramento Rochoso", "#ffaa5f"),
    30: ("Mineração", "#9c0027"),
    31: ("Aquicultura", "#091077"),
    32: ("Apicum", "#fc8114"),
    33: ("Rio, Lago e Oceano", "#2532e4"),
    35: ("Dendê", "#9065d0"),
    36: ("Lavoura Perene", "#d082de"),
    39: ("Soja", "#f5b3c8"),
    40: ("Arroz", "#c71585"),
    41: ("Outras Lavouras Temporárias", "#f54ca9"),
    46: ("Café", "#d68fe2"),
    47: ("Citrus", "#9932cc"),
    48: ("Outras Lavouras Perenes", "#e6ccff"),
    49: ("Restinga Arbórea", "#02d659"),
    50: ("Restinga Herbácea", "#ad5100"),
    62: ("Algodão", "#ff69b4"),
    75: ("Usina Fotovoltaica", "#c12100"),
}


def info_imagem(path):
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    if gt[2] != 0 or gt[4] != 0:
        raise ValueError("imagem rotacionada (gt com termos de rotação) — "
                         "não suportado")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    info = {"path": path, "gt": gt, "wkt": ds.GetProjection(),
            "w": ds.RasterXSize, "h": ds.RasterYSize, "srs": srs}
    auth = (srs.GetAuthorityName(None), srs.GetAuthorityCode(None))
    if auth[0] != "EPSG" or auth[1] is None:
        srs.AutoIdentifyEPSG()
        auth = (srs.GetAuthorityName(None), srs.GetAuthorityCode(None))
    if auth[0] != "EPSG" or auth[1] is None:
        raise ValueError("CRS da imagem sem código EPSG — o GEE precisa dele")
    info["epsg"] = "EPSG:%s" % auth[1]
    return info


def grade_grossa(info, res=30.0):
    """Grade alinhada à imagem com pixel ~res m e fator inteiro."""
    px = abs(info["gt"][1])
    if info["srs"].IsGeographic():
        px_m = px * 111320.0          # só para escolher o fator
    else:
        px_m = px * info["srs"].GetLinearUnits()
    fator = max(1, int(round(res / px_m)))
    gw = -(-info["w"] // fator)       # ceil
    gh = -(-info["h"] // fator)
    x0, dx, _, y0, _, dy = info["gt"]
    return {"fator": fator, "w": gw, "h": gh,
            "gt": (x0, dx * fator, 0.0, y0, 0.0, dy * fator)}


def baixar(info, grossa, ano, raw_path, res=30.0):
    """Baixa a classificação na grade grossa (em blocos) -> array uint8."""
    import ee
    ee.Initialize(project=PROJ)
    if res >= 30:
        img = ee.Image("%s/collection_10_1_%d_v1" % (MB_COLL_30M, ano)) \
            .select("classification").toUint8()
    else:
        img = ee.Image(MB_ASSET_10M).select("classification_%d" % ano).toUint8()
    x0, dx, _, y0, _, dy = grossa["gt"]
    out = np.zeros((grossa["h"], grossa["w"]), np.uint8)
    tmp = raw_path + ".bloco.tif"
    nb = 0
    for r in range(0, grossa["h"], BLOCO_GROSSO):
        for c in range(0, grossa["w"], BLOCO_GROSSO):
            bh = min(BLOCO_GROSSO, grossa["h"] - r)
            bw = min(BLOCO_GROSSO, grossa["w"] - c)
            url = img.getDownloadURL({
                "name": "mapbiomas_%d" % ano,
                "crs": info["epsg"],
                "crs_transform": [dx, 0, x0 + c * dx, 0, dy, y0 + r * dy],
                "dimensions": "%dx%d" % (bw, bh),
                "format": "GEO_TIFF",
            })
            urllib.request.urlretrieve(url, tmp)
            ds = gdal.Open(tmp)
            a = ds.GetRasterBand(1).ReadAsArray()
            ds = None
            if a.shape != (bh, bw):
                raise RuntimeError("bloco veio %s, esperado %s"
                                   % (a.shape, (bh, bw)))
            out[r:r + bh, c:c + bw] = a
            nb += 1
    os.remove(tmp)
    # guarda o bruto (grade grossa) para reuso/inspeção
    ds = gdal.GetDriverByName("GTiff").Create(
        raw_path, grossa["w"], grossa["h"], 1, gdal.GDT_Byte,
        ["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(grossa["gt"])
    ds.SetProjection(info["wkt"])
    ds.GetRasterBand(1).WriteArray(out)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds = None
    print("[2/4] MapBiomas %d baixado em %d bloco(s): %dx%d px de %.1f m"
          % (ano, nb, grossa["w"], grossa["h"], res))
    return out


def escrever_fina(info, grossa, M, mask_path, faixa=2048):
    """Máscara na grade da imagem: repete cada pixel grosso fator x fator e
    zera onde a imagem é nodata. Em faixas de linhas."""
    f = grossa["fator"]
    src = gdal.Open(info["path"])
    b_img = src.GetRasterBand(1)
    nd = b_img.GetNoDataValue()
    nd = 0 if nd is None else nd
    ds = gdal.GetDriverByName("GTiff").Create(
        mask_path, info["w"], info["h"], 1, gdal.GDT_Byte,
        ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"])
    ds.SetGeoTransform(info["gt"])
    ds.SetProjection(info["wkt"])
    bo = ds.GetRasterBand(1)
    bo.SetNoDataValue(0)
    hist = np.zeros(256, np.int64)
    faixa -= faixa % f                 # faixa múltipla do fator
    for y in range(0, info["h"], faixa):
        n = min(faixa, info["h"] - y)
        g = M[y // f: (y + n - 1) // f + 1]
        fina = np.repeat(np.repeat(g, f, axis=0), f, axis=1)
        fina = fina[:n, :info["w"]]
        img = b_img.ReadAsArray(0, y, info["w"], n)
        fina = np.where(img == nd, 0, fina).astype(np.uint8)
        bo.WriteArray(fina, 0, y)
        hist += np.bincount(fina.ravel(), minlength=256)
    bo.FlushCache()
    ds = src = None
    print("[3/4] máscara na grade da imagem: %s" % mask_path)
    return hist


def validar(info, mask_path, hist):
    ds = gdal.Open(mask_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    ok_crs = bool(srs.IsSame(info["srs"]))
    ok_gt = np.allclose(ds.GetGeoTransform(), info["gt"])
    ok_dim = (ds.RasterXSize, ds.RasterYSize) == (info["w"], info["h"])
    ok_tipo = ds.GetRasterBand(1).DataType == gdal.GDT_Byte
    ds = None
    cods = [int(c) for c in np.nonzero(hist)[0]]
    desconhecidos = [c for c in cods if c not in LEGEND]
    tot = hist.sum()
    classes = {str(c): {"nome": LEGEND.get(c, ("?", ""))[0],
                        "frac": round(float(hist[c] / tot), 5)}
               for c in cods}
    rel = {"crs_igual": ok_crs, "transform_igual": bool(ok_gt),
           "dimensoes_iguais": ok_dim, "byte": ok_tipo,
           "codigos_desconhecidos": desconhecidos, "classes": classes}
    print("[4/4] validação: CRS %s | transform %s | dimensões %s | Byte %s | "
          "códigos fora da legenda: %s"
          % (ok_crs, ok_gt, ok_dim, ok_tipo, desconhecidos or "nenhum"))
    for c in cods:
        print("      %3d %-32s %6.2f%%" % (c, LEGEND.get(c, ("?",))[0],
                                          100 * hist[c] / tot))
    if not (ok_crs and ok_gt and ok_dim and ok_tipo) or desconhecidos:
        raise SystemExit("validação falhou")
    return rel


def quicklook(info, M, grossa, png_path, res=30.0, max_lado=2000):
    """PNG lado a lado: imagem reduzida | máscara (legenda oficial)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    s = max(1, int(np.ceil(max(M.shape) / max_lado)))
    m = M[::s, ::s]
    ds = gdal.Open(info["path"])
    img = ds.GetRasterBand(1).ReadAsArray(
        buf_xsize=m.shape[1], buf_ysize=m.shape[0],
        resample_alg=gdal.GRIORA_Average).astype(np.float32)
    ds = None
    v = img[img > 0]
    lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)
    m = np.where(img > 0, m, 0)
    cods = sorted(set(np.unique(m).tolist()) | {0})
    cmap = ListedColormap([LEGEND.get(c, ("?", "#888888"))[1] for c in cods])
    lut = np.zeros(256, np.int32)
    lut[cods] = np.arange(len(cods))
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    ax[0].imshow(np.clip((img - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray")
    ax[0].set_title(os.path.basename(info["path"]))
    ax[1].imshow(lut[m], cmap=cmap,
                 norm=BoundaryNorm(np.arange(len(cods) + 1) - 0.5, len(cods)),
                 interpolation="nearest")
    ax[1].set_title("MapBiomas (%.0f m) na grade da imagem" % res)
    ax[1].legend(handles=[Patch(color=LEGEND.get(c, ("", "#888888"))[1],
                                label="%d %s" % (c, LEGEND.get(c, ("?",))[0]))
                          for c in cods if c != 0],
                 loc="lower left", fontsize=8, framealpha=0.9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print("      quicklook: %s" % png_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--ano", type=int, required=True)
    ap.add_argument("--res", type=float, default=30.0, choices=[10.0, 30.0],
                    help="resolução do MapBiomas: 30 (coleção 10.1, 1985–2024) "
                         "ou 10 (coleção 3, 2017–2024)")
    ap.add_argument("--out", default=None,
                    help="pasta de saída (default: <pasta da cena>/mapbiomas_<ano>_<res>m)")
    ap.add_argument("--reusar", action="store_true",
                    help="reusar o bruto já baixado (não chama o GEE)")
    args = ap.parse_args()

    t0 = time.time()
    res = args.res
    info = info_imagem(args.imagem)
    out = args.out or os.path.join(caminhos.pasta_cena(args.imagem),
                                   "mapbiomas_%d_%dm" % (args.ano, int(res)))
    os.makedirs(out, exist_ok=True)
    grossa = grade_grossa(info, res)
    print("[1/4] imagem %dx%d, pixel %.3f, %s | grade MapBiomas %dx%d "
          "(fator %d)" % (info["w"], info["h"], abs(info["gt"][1]),
                          info["epsg"], grossa["w"], grossa["h"],
                          grossa["fator"]))

    raw = os.path.join(out, "_mapbiomas_%d_%dm_raw.tif" % (args.ano, int(res)))
    if args.reusar and os.path.exists(raw):
        ds = gdal.Open(raw)
        M = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        print("[2/4] reusando %s" % raw)
    else:
        M = baixar(info, grossa, args.ano, raw, res)

    mask_path = os.path.join(out, "MASK_MAPBIOMAS_%d.tif" % args.ano)
    hist = escrever_fina(info, grossa, M, mask_path)
    rel = validar(info, mask_path, hist)
    quicklook(info, M, grossa, os.path.join(out, "quicklook_%d.png" % args.ano),
              res)
    colecao = (("%s/collection_10_1_%d_v1" % (MB_COLL_30M, args.ano))
               if res >= 30 else
               (MB_ASSET_10M + " #classification_%d" % args.ano))
    rel.update({"imagem": os.path.abspath(args.imagem), "ano": args.ano,
                "res": res, "colecao": colecao,
                "fator": grossa["fator"], "grade_grossa_gt": grossa["gt"],
                "segundos": round(time.time() - t0, 1)})
    with open(os.path.join(out, "relatorio_%d.json" % args.ano), "w",
              encoding="utf-8") as fp:
        json.dump(rel, fp, ensure_ascii=False, indent=2)
    print("OK -> %s (%.0fs)" % (out, time.time() - t0))


if __name__ == "__main__":
    main()
