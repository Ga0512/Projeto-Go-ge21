#!/usr/bin/env python3
"""Inferência de um modelo de segmentação numa CENA INTEIRA (GeoTIFF PAN 2,5 m),
faixa a faixa, resumível.

A cena inteira (857 Mpx na 013-014) não cabe na GPU de uma vez, então é
processada em faixas horizontais com overlap; cada faixa é tratada como uma
imagem INDEPENDENTE pelo MESMO caminho do inference.py (bandas/estatísticas/
predição em janelas de 256 sobrepostas) e gravada num GeoTIFF pequeno
separado (faixa_<y>.tif) — a mesma escrita única e comprovada do inference.py.

- Resumível: a cada faixa concluída o y é registrado em progresso.txt; rodar de
  novo continua de onde parou (para quando quiser e retoma).
- No final (ou quando você quiser, rodando de novo), as faixas são unidas num
  único classes.tif via VRT (gdal.BuildVRT + Translate — escrita única).

Estatísticas: as da cena no receita.json do dataset do modelo (--cena).

Saída em <saida>/:
  faixa_<y>.tif      um GeoTIFF Byte por faixa (grade/CRS da cena, offset correto)
  progresso.txt      faixas já feitas (retomada)
  classes.tif        mapa final unido (gerado quando todas as faixas existirem)
  info.json          modelo, estatísticas, fração de cada classe

Uso (no WSL, a partir de Models/):
  # metade, p/ conferir o PNG antes de gastar tempo:
  ../Data/.venv/bin/python vit_model/inferencia_cena.py \\
      --imagem ../Data/dados/spot5/colinas_do_sul/original/Colinas_do_Sul_mosaico_v2.tif \\
      --modelo unet_cnn/runs/unetpp_v13/modelo_melhor.pth --ate 16312
  # depois o resto (retoma de onde parou):
  ../Data/.venv/bin/python vit_model/inferencia_cena.py ... (sem --ate)
"""

import argparse
import gc
import glob
import json
import os
import sys
import time

import ambiente  # noqa: F401  (caches -> D:; antes de torch/matplotlib)
import numpy as np
import torch

import dados
from inference import (MARGEM, bandas, carregar_modelo, prever,
                       texturas, tipo_modelo)

from osgeo import gdal  # noqa: E402

gdal.UseExceptions()

BASE = os.path.dirname(os.path.abspath(__file__))


def _paleta(band):
    """Paleta MapBiomas + nodata 0 no raster band dado (antes de escrever pixels)."""
    ct = gdal.ColorTable()
    for cod, cor in zip(dados.REPR_CODIGO, dados.CORES_CLASSE):
        ct.SetColorEntry(int(cod), tuple(int(cor[i:i + 2], 16) for i in (1, 3, 5)) + (255,))
    band.SetRasterColorTable(ct)
    band.SetNoDataValue(0)
    band.SetDescription("classe (código MapBiomas representativo)")


def gravar_faixa(caminho, codigos, y0, gt, proj, W):
    """GeoTIFF pequeno da faixa y0..y0+H, grade/CRS da cena, offset correto."""
    H = codigos.shape[0]
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(caminho, W, H, 1, gdal.GDT_Byte, ["COMPRESS=DEFLATE"])
    gt_faixa = (gt[0], gt[1], gt[2], gt[3] + y0 * gt[5], gt[4], gt[5])
    ds.SetGeoTransform(gt_faixa)
    ds.SetProjection(proj)
    b = ds.GetRasterBand(1)
    _paleta(b)
    b.WriteArray(codigos)            # escrita única (mesmo fluxo do inference.py)
    ds.FlushCache()
    ds = None


def unir_faixas(cls_caminho, pasta, W, H, gt, proj):
    """Junta todas as faixa_*.tif num único classes.tif via VRT (escrita única)."""
    faixas = sorted(glob.glob(os.path.join(pasta, "faixa_*.tif")))
    if not faixas:
        return None
    vrt = os.path.join(pasta, "faixas.vrt")
    if os.path.exists(vrt):
        os.remove(vrt)
    gdal.BuildVRT(vrt, faixas)
    drv = gdal.GetDriverByName("GTiff")
    cls_ds = drv.Create(cls_caminho, W, H, 1, gdal.GDT_Byte,
                        ["COMPRESS=DEFLATE", "TILED=YES"])
    cls_ds.SetGeoTransform(gt)
    cls_ds.SetProjection(proj)
    cb = cls_ds.GetRasterBand(1)
    _paleta(cb)
    src = gdal.Open(vrt)
    cb.WriteArray(src.ReadAsArray())
    cls_ds.FlushCache()
    src = None
    cls_ds = None
    os.remove(vrt)
    return len(faixas)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagem", required=True, help="GeoTIFF PAN (1 banda) da cena inteira")
    ap.add_argument("--modelo", required=True, help="modelo_melhor.pth de um run seg_* ou unet_*")
    ap.add_argument("--cena", default=None,
                    help="cena do receita.json cujas estatísticas usar (padrão: pelo caminho "
                         "da imagem)")
    ap.add_argument("--stats-dataset", default=None,
                    help="dataset de onde tirar as estatísticas da cena, quando o dataset do "
                         "modelo não contém essa cena (ex.: modelo treinado só em colinas "
                         "rodando em jataí). O resto (bandas, classes, parâmetros) continua "
                         "vindo do dataset do modelo; a origem fica no info.json")
    ap.add_argument("--saida", default=None, help="pasta de saída")
    ap.add_argument("--faixa", type=int, default=1024, help="linhas por faixa processada")
    ap.add_argument("--ate", type=int, default=0,
                    help="processa só até esta linha (0 = cena inteira; útil p/ teste parcial)")
    ap.add_argument("--overlap", type=int, default=300,
                    help="px extras em cada lado da faixa (texturas + janelas)")
    ap.add_argument("--passo", type=int, default=128, help="px entre janelas de 256")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--unir", action="store_true",
                    help="só junta as faixas já existentes em classes.tif (não infere)")
    args = ap.parse_args()
    t0 = time.monotonic()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, ck = carregar_modelo(args.modelo, dev)
    ta = ck["args"]
    dataset = ta["dataset"] if os.path.isabs(ta["dataset"]) else \
        os.path.normpath(os.path.join(ambiente.MODELS, ta["dataset"]))
    dados.configurar(dataset)             # REPR_CODIGO/CLASS_NOMES/cores seguem o modelo
    receita = json.load(open(os.path.join(dataset, "receita.json"), encoding="utf-8"))
    par = receita["parametros"]
    P = 256
    nome_modelo = os.path.basename(os.path.dirname(os.path.abspath(args.modelo)))
    print("[modelo] %s (epoca %s, val mIoU=%.4f) | dataset %s | %s"
          % (nome_modelo, ck["epoca"], ck["val_miou"], os.path.basename(dataset), dev), flush=True)

    ds = gdal.Open(args.imagem)
    if ds.RasterCount != 1:
        raise SystemExit("esperava 1 banda (PAN), a imagem tem %d" % ds.RasterCount)
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    if abs(gt[1] - par["resolucao"]) > 1e-3 or abs(gt[5] + par["resolucao"]) > 1e-3:
        print("[aviso] resolução %.3f x %.3f m, o modelo foi treinado a %.2f m"
              % (gt[1], -gt[5], par["resolucao"]), flush=True)
    b = ds.GetRasterBand(1)
    nd = b.GetNoDataValue()
    proj = ds.GetProjection()

    origem_st = dataset
    rec_st = receita
    if args.stats_dataset:
        origem_st = args.stats_dataset if os.path.isabs(args.stats_dataset) else \
            os.path.normpath(os.path.join(ambiente.MODELS, args.stats_dataset))
        rec_st = json.load(open(os.path.join(origem_st, "receita.json"), encoding="utf-8"))
        for k in ("janela", "niveis", "resolucao"):
            if rec_st["parametros"][k] != par[k]:
                raise SystemExit("--stats-dataset incompatível: %s = %s lá, %s aqui"
                                 % (k, rec_st["parametros"][k], par[k]))
    cenas = {c: v["estatisticas"] for c, v in rec_st["resultado"]["por_cena"].items()}
    cena = args.cena or next((c for c in cenas if c in os.path.abspath(args.imagem)), None)
    if not cena:
        raise SystemExit("cena não detectada no caminho; use --cena (opções: %s)"
                         % ", ".join(cenas))
    st = cenas[cena]
    print("[estatisticas] %s | p2=%.1f p98=%.1f | de %s"
          % (cena, st["p2"], st["p98"], os.path.basename(origem_st)), flush=True)

    saida = args.saida or os.path.join(
        os.path.join(ambiente.MODELS, "unet_cnn") if tipo_modelo(args.modelo) == "unet" else BASE,
        "inferencias", nome_modelo, cena + "_cena")
    os.makedirs(saida, exist_ok=True)
    cls_caminho = os.path.join(saida, "classes.tif")

    # ---- só unir as faixas já feitas (sem inferir) ----
    if args.unir:
        n = unir_faixas(cls_caminho, saida, W, H, gt, proj)
        if n is None:
            raise SystemExit("nenhuma faixa_*.tif encontrada em %s" % saida)
        print("OK -> %s (%d faixas unidas) | %.0fs" % (cls_caminho, n, time.monotonic() - t0))
        return

    prog = os.path.join(saida, "progresso.txt")
    feitas = set()
    if os.path.exists(prog):
        with open(prog) as f:
            feitas = {int(ln) for ln in f if ln.strip().isdigit()}
        print("[retomar] %d faixas já feitas" % len(feitas), flush=True)

    ol = max(args.overlap, MARGEM + P // 2 + 2)
    H_alvo = args.ate if args.ate > 0 else H
    n_faixas = 0
    t_pred = 0.0
    for y0 in range(0, H_alvo, args.faixa):
        if y0 in feitas:
            continue
        y1 = min(y0 + args.faixa, H)
        yA, yB = max(0, y0 - ol), min(H, y1 + ol)
        pan = b.ReadAsArray(0, yA, W, yB - yA).astype(np.float32)
        if nd is not None:
            pan = np.where(pan == nd, np.nan, pan)
        if nd is None and receita["parametros"].get("pan_zero_nodata"):
            pan = np.where(pan == 0, np.nan, pan)  # PAN 0 = sem dado (moldura de mosaico)
        valido = np.isfinite(pan)
        ini = y0 - yA
        if not valido[ini:ini + (y1 - y0)].any():
            # faixa inteira sem dado (moldura preta do mosaico): grava zeros e
            # pula textura + inferência, que custariam o tempo de uma faixa cheia
            gravar_faixa(os.path.join(saida, "faixa_%06d.tif" % y0),
                         np.zeros((y1 - y0, W), np.uint8), y0, gt, proj, W)
            with open(prog, "a") as f:
                f.write("%d\n" % y0)
            n_faixas += 1
            print("  faixa %2d/%d  y=%d..%d | sem dado, pulada"
                  % (n_faixas, -(-H_alvo // args.faixa), y0, y1), flush=True)
            continue
        X = bandas(pan, valido, st, par["janela"], par["niveis"])
        X[:, ~valido] = 0
        t = time.monotonic()
        probs, _ = prever(modelo, X, P, args.passo, args.lote, dev, ta.get("margem", 15))
        t_pred += time.monotonic() - t
        cls = probs.argmax(0)[ini:ini + (y1 - y0)]
        codigos = np.asarray(dados.REPR_CODIGO, np.uint8)[cls]
        codigos[~valido[ini:ini + (y1 - y0)]] = 0
        gravar_faixa(os.path.join(saida, "faixa_%06d.tif" % y0), codigos, y0, gt, proj, W)
        with open(prog, "a") as f:
            f.write("%d\n" % y0)
        n_faixas += 1
        print("  faixa %2d/%d  y=%d..%d | %.0fs"
              % (n_faixas, -(-H_alvo // args.faixa), y0, y1, time.monotonic() - t0), flush=True)
        del pan, X, probs, cls, codigos
        gc.collect()                          # RAM
        torch.cuda.empty_cache()              # VRAM
    ds = None

    # junta as faixas existentes (todas as feitas até aqui)
    n_unidas = unir_faixas(cls_caminho, saida, W, H, gt, proj)
    if n_unidas is None:
        print("nenhuma faixa_*.tif para unir; rode de novo ou com --unir")
        return

    arr = gdal.Open(cls_caminho).ReadAsArray()
    val = arr != 0
    fr = {n: round(float(((arr == cod) & val).sum()) / max(1, val.sum()), 4)
          for n, cod in zip(dados.CLASS_NOMES, dados.REPR_CODIGO)}
    with open(os.path.join(saida, "info.json"), "w", encoding="utf-8") as f:
        json.dump({"imagem": os.path.abspath(args.imagem), "modelo": os.path.abspath(args.modelo),
                   "epoca_modelo": ck["epoca"], "val_miou_modelo": ck["val_miou"],
                   "dataset_modelo": dataset, "cena": cena, "estatisticas": st,
                   "estatisticas_de": origem_st,
                   "faixas_feitas": len(glob.glob(os.path.join(saida, "faixa_*.tif"))),
                   "faixa_linhas": args.faixa, "overlap": ol,
                   "janelas": {"patch": P, "passo": args.passo},
                   "codigos_saida": dict(zip(dados.CLASS_NOMES, map(int, dados.REPR_CODIGO))),
                   "fracao_prevista": fr, "sem_dado": float((arr == 0).mean()),
                   "tempo_predicao_s": round(t_pred, 1),
                   "tempo_total_s": round(time.monotonic() - t0, 1)},
                  f, indent=2, ensure_ascii=False)
    print("[classes] " + "  ".join("%s=%.1f%%" % (n, 100 * v) for n, v in fr.items()))
    print("OK -> %s (classes.tif, info.json) | %.0fs" % (saida, time.monotonic() - t0))


if __name__ == "__main__":
    main()