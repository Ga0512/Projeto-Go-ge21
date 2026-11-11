#!/usr/bin/env python3
"""Corta uma cena grande (imagem + máscara MapBiomas) em tiles numa pasta, e
junta as máscaras refinadas dos tiles de volta na grade da cena.

Estrutura gerada (cortar):
  <out>/tiles.json                 grade, origem e lista de tiles
  <out>/imagem.vrt                 mosaico virtual das imagens dos tiles
  <out>/mascara_mapbiomas.vrt      mosaico virtual das máscaras originais
  <out>/rRR_cCC/imagem.tif         recorte da imagem (Byte, georreferenciado)
  <out>/rRR_cCC/mascara_mapbiomas.tif   recorte da máscara (Byte, nearest)
  <out>/rRR_cCC/mascara_refinada.tif    criada pelo editor ao salvar o tile

Tiles só com nodata são pulados. Os recortes são cópias exatas de janelas da
grade original (mesmo CRS, pixel e alinhamento) — nada é reamostrado.

Uso:
  .venv/bin/python scripts/spot5/tiles_cena.py cortar \
      --imagem dados/spot5/cena_013-014_2007-06-14/original/013-014_S5_708-378-0_2007-06-14_modificado.tif \
      --mascara dados/spot5/cena_013-014_2007-06-14/mapbiomas_2007_30m/MASK_MAPBIOMAS_2007.tif
  .venv/bin/python scripts/spot5/tiles_cena.py juntar \
      --pasta dados/spot5/cena_013-014_2007-06-14/tiles
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

gdal.UseExceptions()

TILE_PX = 2048
OPC = ["COMPRESS=DEFLATE", "TILED=YES"]


def _gravar(path, arr, gt, wkt, nodata=0):
    ds = gdal.GetDriverByName("GTiff").Create(
        path, arr.shape[1], arr.shape[0], 1, gdal.GDT_Byte, OPC)
    ds.SetGeoTransform(gt)
    ds.SetProjection(wkt)
    b = ds.GetRasterBand(1)
    b.SetNoDataValue(nodata)
    b.WriteArray(arr)
    ds = None


def cortar(imagem, mascara, out, tile=TILE_PX, banda_mascara=1):
    t0 = time.time()
    di, dm = gdal.Open(imagem), gdal.Open(mascara)
    w, h = di.RasterXSize, di.RasterYSize
    if (dm.RasterXSize, dm.RasterYSize) != (w, h) or \
            not np.allclose(dm.GetGeoTransform(), di.GetGeoTransform()):
        raise SystemExit("máscara não está na grade da imagem")
    gt, wkt = di.GetGeoTransform(), di.GetProjection()
    bi, bm = di.GetRasterBand(1), dm.GetRasterBand(banda_mascara)
    nd = bi.GetNoDataValue()
    nd = 0 if nd is None else nd
    os.makedirs(out, exist_ok=True)
    tiles = []
    nr, nc = -(-h // tile), -(-w // tile)
    for r in range(nr):
        y0 = r * tile
        th = min(tile, h - y0)
        faixa = bi.ReadAsArray(0, y0, w, th)        # imagem em faixas de linha
        mfaixa = bm.ReadAsArray(0, y0, w, th)
        for c in range(nc):
            x0 = c * tile
            tw = min(tile, w - x0)
            img = faixa[:, x0:x0 + tw]
            valido = img != nd
            if not valido.any():
                continue
            tid = "r%02d_c%02d" % (r, c)
            d = os.path.join(out, tid)
            os.makedirs(d, exist_ok=True)
            tgt = (gt[0] + x0 * gt[1], gt[1], 0.0, gt[3] + y0 * gt[5], 0.0, gt[5])
            _gravar(os.path.join(d, "imagem.tif"), img, tgt, wkt, nd)
            _gravar(os.path.join(d, "mascara_mapbiomas.tif"),
                    np.where(valido, mfaixa[:, x0:x0 + tw], 0), tgt, wkt)
            tiles.append({"id": tid, "x0": x0, "y0": y0, "w": tw, "h": th,
                          "frac_imagem": round(float(valido.mean()), 4)})
        print("  linha %d/%d: %d tiles até agora" % (r + 1, nr, len(tiles)),
              flush=True)
    di = dm = None
    for nome in ("imagem", "mascara_mapbiomas"):
        gdal.BuildVRT(os.path.join(out, nome + ".vrt"),
                      [os.path.join(out, t["id"], nome + ".tif")
                       for t in tiles],
                      outputBounds=(gt[0], gt[3] + h * gt[5],
                                    gt[0] + w * gt[1], gt[3]),
                      srcNodata=0, VRTNodata=0).FlushCache()
    # relativos à pasta dos tiles: continuam válidos se o projeto mudar de lugar
    meta = {"imagem_original": os.path.relpath(imagem, out),
            "mascara_original": os.path.relpath(mascara, out),
            "wkt": wkt,
            "tile_px": tile, "w": w, "h": h, "gt": gt, "nodata_imagem": nd,
            "tiles": tiles}
    with open(os.path.join(out, "tiles.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=1)
    print("OK: %d tiles de %d px (de %dx%d possíveis) em %s (%.0fs)"
          % (len(tiles), tile, nc, nr, out, time.time() - t0))


def cortar_mascara(pasta, mascara, nome="mascara_mapbiomas_10m.tif",
                   banda=1):
    """Corta uma máscara adicional (ex.: MapBiomas 10 m) nos tiles já
    existentes, sem tocar imagem.tif / mascara_mapbiomas.tif / refinadas.
    Grava <tile>/<nome> + <nome>.vrt e registra em tiles.json."""
    with open(os.path.join(pasta, "tiles.json"), encoding="utf-8") as fp:
        meta = json.load(fp)
    dm = gdal.Open(mascara)
    bm = dm.GetRasterBand(banda)
    if (dm.RasterXSize, dm.RasterYSize) != (meta["w"], meta["h"]) or \
            not np.allclose(dm.GetGeoTransform(), meta["gt"]):
        raise SystemExit("máscara %s não está na grade da cena" % mascara)
    gt = meta["gt"]
    for t in meta["tiles"]:
        d = os.path.join(pasta, t["id"])
        a = bm.ReadAsArray(t["x0"], t["y0"], t["w"], t["h"]).astype(np.uint8)
        tgt = (gt[0] + t["x0"] * gt[1], gt[1], 0.0,
               gt[3] + t["y0"] * gt[5], 0.0, gt[5])
        _gravar(os.path.join(d, nome), np.where(a != 0, a, 0), tgt,
                dm.GetProjection())
    dm = None
    gdal.BuildVRT(os.path.join(pasta, nome[:-4] + ".vrt"),
                  [os.path.join(pasta, t["id"], nome) for t in meta["tiles"]],
                  outputBounds=(gt[0], gt[3] + meta["h"] * gt[5],
                                gt[0] + meta["w"] * gt[1], gt[3]),
                  srcNodata=0, VRTNodata=0).FlushCache()
    meta["mascaras_extra"] = meta.get("mascaras_extra", []) + [nome]
    with open(os.path.join(pasta, "tiles.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=1)
    print("OK: %s em %d tiles de %s (máscara %s)"
          % (nome, len(meta["tiles"]), pasta, mascara))


def juntar(pasta, saida=None):
    """Máscara final na grade da cena: tile refinado onde existe, MapBiomas
    original nos demais."""
    with open(os.path.join(pasta, "tiles.json"), encoding="utf-8") as fp:
        meta = json.load(fp)
    saida = saida or os.path.join(pasta, "MASCARA_REFINADA_CENA.tif")
    wkt = meta.get("wkt")
    if not wkt:   # tiles.json antigo: CRS vem do 1º tile (mesma grade da cena)
        ref = gdal.Open(os.path.join(pasta, meta["tiles"][0]["id"],
                                     "mascara_mapbiomas.tif"))
        wkt = ref.GetProjection()
        ref = None
    ds = gdal.GetDriverByName("GTiff").Create(
        saida, meta["w"], meta["h"], 1, gdal.GDT_Byte,
        OPC + ["BIGTIFF=IF_SAFER"])
    ds.SetGeoTransform(meta["gt"])
    ds.SetProjection(wkt)
    b = ds.GetRasterBand(1)
    b.SetNoDataValue(0)
    n_ref = 0
    for t in meta["tiles"]:
        d = os.path.join(pasta, t["id"])
        p = os.path.join(d, "mascara_refinada.tif")
        n_ref += os.path.exists(p)
        src = gdal.Open(p if os.path.exists(p)
                        else os.path.join(d, "mascara_mapbiomas.tif"))
        b.WriteArray(src.GetRasterBand(1).ReadAsArray(), t["x0"], t["y0"])
        src = None
    b.FlushCache()
    ds = None
    print("OK: %s — %d de %d tiles refinados (demais = MapBiomas original)"
          % (saida, n_ref, len(meta["tiles"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cortar")
    a.add_argument("--imagem", required=True)
    a.add_argument("--mascara", required=True)
    a.add_argument("--banda-mascara", type=int, default=1)
    a.add_argument("--tile", type=int, default=TILE_PX)
    a.add_argument("--out", default=None,
                   help="default: <pasta da cena>/tiles (mãe de original/)")
    j = sub.add_parser("juntar")
    j.add_argument("--pasta", required=True)
    j.add_argument("--saida", default=None)
    m = sub.add_parser("cortar-mascara")
    m.add_argument("--pasta", required=True,
                   help="pasta de tiles já cortada (tem tiles.json)")
    m.add_argument("--mascara", required=True,
                   help="máscara adicional na grade da cena (ex.: 10 m)")
    m.add_argument("--nome", default="mascara_mapbiomas_10m.tif")
    m.add_argument("--banda", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "cortar":
        out = args.out or os.path.join(caminhos.pasta_cena(args.imagem), "tiles")
        cortar(args.imagem, args.mascara, out, args.tile, args.banda_mascara)
    elif args.cmd == "cortar-mascara":
        cortar_mascara(args.pasta, args.mascara, args.nome, args.banda)
    else:
        juntar(args.pasta, args.saida)


if __name__ == "__main__":
    main()
