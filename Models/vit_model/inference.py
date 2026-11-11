#!/usr/bin/env python3
"""Inferência de um modelo de segmentação (vit_model/runs/seg_*) numa imagem PAN
qualquer (GeoTIFF de 1 banda, 2,5 m), p.ex. um tile que não entrou no dataset.

A imagem passa pelo MESMO caminho do dataset (Cena.texturas/bandas em
Data/scripts/datasets/gerar_dataset_refinado.py):
  PAN esticada p2–p98 -> uint8 -> GLCM entropia (janela 7, 8 níveis), LocalSTD
  7x7 e Sobel (features_block) -> bandas [PAN, GLCM, (z(STD)+z(Grad))/2] em
  z-score com as estatísticas da cena.
Estatísticas: as da cena no receita.json do dataset do modelo (--cena, ou
detectada pelo nome da cena no caminho da imagem); para uma cena nova, são
calculadas na própria imagem com a mesma regra do dataset.

A predição é feita em janelas de 256 px sobrepostas (--passo), com peso maior
no centro de cada janela (a borda do patch é onde o modelo erra mais), e as
probabilidades são somadas antes do argmax.

Saída em vit_model/inferencias/<modelo>/<nome>/:
  classes.tif   GeoTIFF Byte, mesma grade da entrada, códigos MapBiomas
                representativos (config.REPR_CODIGO), 0 = sem dado, com paleta
  classes.png   PAN | predição (| referência, se --referencia) + legenda e %
  info.json     modelo, estatísticas usadas, fração de cada classe
  (--probs)     probabilidades.tif: 1 banda Byte (0–255) por classe

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python vit_model/inference.py \\
      --imagem ../Data/dados/spot5/colinas_do_sul/tiles/r10_c05/imagem.tif \\
      --referencia ../Data/dados/spot5/colinas_do_sul/tiles/r10_c05/mascara_mapbiomas.tif
"""

import argparse
import json
import os
import sys
import time

import ambiente  # noqa: F401  (caches -> D:; antes de torch/matplotlib)
import numpy as np
import scipy.ndimage as ndi
import torch

import dados
from segmentacao import SegViT, criar_encoder

sys.path.insert(0, os.path.join(dados.DATA, "scripts", "wv1"))
from gerar_features_pan import features_block  # noqa: E402

from osgeo import gdal  # noqa: E402

gdal.UseExceptions()

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO_PADRAO = os.path.join(BASE, "runs", "seg_mae_v3_ft_v4", "modelo_melhor.pth")
MARGEM = 8                 # px extras p/ as texturas (como no dataset)


# ------------------------------------------------------------ bandas
def preencher_nodata(pan, valido):
    """Nodata recebe o vizinho válido mais próximo (só p/ calcular texturas)."""
    if valido.all():
        return pan
    idx = ndi.distance_transform_edt(~valido, return_distances=False, return_indices=True)
    return pan[tuple(idx)]


def texturas(pan, valido, p2, p98, janela, niveis):
    """PAN, entropia GLCM, LocalSTD e gradiente da imagem inteira, com margem
    refletida (mesmo recorte de Cena.texturas)."""
    half = janela // 2
    pan = np.pad(preencher_nodata(pan, valido), MARGEM, mode="reflect")
    s = (np.clip((pan - p2) / (p98 - p2), 0, 1) * 255 + 0.5).astype(np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, std, ent, grad = features_block(s, s.shape[0] - 2 * half, janela, niveis)
    H, W = pan.shape[0] - 2 * MARGEM, pan.shape[1] - 2 * MARGEM
    corta = (slice(MARGEM - half, MARGEM - half + H), slice(MARGEM, MARGEM + W))
    return pan[MARGEM:MARGEM + H, MARGEM:MARGEM + W], ent[corta], std[corta], grad[corta]


def bandas(pan, valido, st, janela, niveis):
    pan, ent, std, grad = texturas(pan, valido, st["p2"], st["p98"], janela, niveis)
    z = lambda a, k: (a - st["media"][k]) / st["desvio"][k]  # noqa: E731
    rug = (z(std, "std") + z(grad, "grad")) / 2
    return np.stack([z(pan, "pan"), z(ent, "glcm"), z(rug, "rugosidade")]).astype(np.float32)


def estatisticas_da_imagem(pan, valido, janela, niveis, lado=512, n_blocos=64, seed=42):
    """Mesma regra de estatisticas() do dataset, aplicada à própria imagem."""
    H, W = pan.shape
    pos = [(y, x) for y in range(0, H - lado + 1, lado) for x in range(0, W - lado + 1, lado)]
    np.random.default_rng(seed).shuffle(pos)
    blocos = [(y, x) for y, x in pos if valido[y:y + lado, x:x + lado].mean() >= 0.999][:n_blocos]
    if not blocos:
        raise SystemExit("nenhum bloco %dx%d válido para as estatísticas; use --cena" % (lado, lado))
    p2, p98 = np.percentile(np.concatenate(
        [pan[y:y + lado, x:x + lado][valido[y:y + lado, x:x + lado]] for y, x in blocos]), [2, 98])
    acc = {k: [] for k in ("pan", "glcm", "std", "grad")}
    for y, x in blocos:
        sl = (slice(y, y + lado), slice(x, x + lado))
        for k, a in zip(acc, texturas(pan[sl], valido[sl], p2, p98, janela, niveis)):
            acc[k].append(a.ravel())
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    media = {k: float(v.mean()) for k, v in acc.items()}
    desvio = {k: float(v.std()) for k, v in acc.items()}
    rug = ((acc["std"] - media["std"]) / desvio["std"]
           + (acc["grad"] - media["grad"]) / desvio["grad"]) / 2
    media["rugosidade"], desvio["rugosidade"] = float(rug.mean()), float(rug.std())
    return {"p2": float(p2), "p98": float(p98), "blocos": len(blocos),
            "media": media, "desvio": desvio}


# ------------------------------------------------------------ modelo
def tipo_modelo(caminho):
    """'unet' se o checkpoint for UNet/UNet++ (tem e0.*), senão 'vit' (SegViT)."""
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    return "unet" if "e0.net.0.weight" in ck["state_dict"] else "vit"


def carregar_modelo(caminho, dev):
    """Carrega modelo_melhor.pth de um seg_* (ViT) ou unet_* (CNN).
    Detecta pela presença das chaves do encoder ViT no state_dict."""
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    sd, args = ck["state_dict"], ck["args"]
    n_cls, in_chans = sd["cabeca.weight"].shape[0], sd["raso.0.weight"].shape[1] \
        if "raso.0.weight" in sd else sd["e0.net.0.weight"].shape[1]
    if "raso.0.weight" in sd:                                   # ViT (SegViT)
        enc_fonte = args["encoder"]
        if enc_fonte not in ("dino", "aleatorio") and not os.path.isabs(enc_fonte):
            enc_fonte = os.path.join(ambiente.MODELS, enc_fonte)
        encoder, _ = criar_encoder(enc_fonte, in_chans=in_chans)
        modelo = SegViT(encoder, n_cls, in_chans=in_chans)
    else:                                                       # UNet/UNet++
        sys.path.insert(0, os.path.join(ambiente.MODELS, "unet_cnn"))
        import unet  # noqa: E402
        base = args.get("base", 32)
        modelo = (unet.UNet if args.get("arquitetura", "unetpp") == "unet" else unet.UNetPP)(
            in_chans=in_chans, n_classes=n_cls, base=base)
    modelo.load_state_dict(sd)
    return modelo.to(dev).eval(), ck


def peso_janela(P, margem):
    """1 no interior, rampa linear até 0,05 na borda (a borda do patch erra mais)."""
    r = np.minimum(np.arange(P), np.arange(P)[::-1]).astype(np.float32)
    r = np.clip((r + 1) / margem, 0.05, 1.0)
    return torch.from_numpy(np.outer(r, r))


def posicoes(L, P, passo):
    if L <= P:
        return [0]
    p = list(range(0, L - P + 1, passo))
    return p if p[-1] == L - P else p + [L - P]


@torch.no_grad()
def prever(modelo, X, P, passo, lote, dev, margem):
    """X (C,H,W) -> probabilidades (n_cls,H,W) em janelas sobrepostas."""
    C, H, W = X.shape
    ph, pw = max(0, P - H), max(0, P - W)          # imagem menor que o patch
    if ph or pw:
        X = np.pad(X, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    Xt = torch.from_numpy(X)
    Hp, Wp = Xt.shape[1:]
    w = peso_janela(P, margem).to(dev)
    soma, peso = None, torch.zeros(Hp, Wp, device=dev)
    jan = [(y, x) for y in posicoes(Hp, P, passo) for x in posicoes(Wp, P, passo)]
    for i in range(0, len(jan), lote):
        bloco = jan[i:i + lote]
        xb = torch.stack([Xt[:, y:y + P, x:x + P] for y, x in bloco]).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pr = modelo(xb).float().softmax(1)
        if soma is None:
            soma = torch.zeros(pr.shape[1], Hp, Wp, device=dev)
        for (y, x), p in zip(bloco, pr):
            soma[:, y:y + P, x:x + P] += p * w
            peso[y:y + P, x:x + P] += w
    return (soma / peso).cpu().numpy()[:, :H, :W], len(jan)


# ------------------------------------------------------------ saída
def gravar_classes(caminho, cls, ref_ds):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(caminho, cls.shape[1], cls.shape[0], 1, gdal.GDT_Byte,
                    ["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(ref_ds.GetGeoTransform())
    ds.SetProjection(ref_ds.GetProjection())
    b = ds.GetRasterBand(1)
    ct = gdal.ColorTable()           # paleta antes dos pixels (TIFF não aceita depois)
    for cod, cor in zip(dados.REPR_CODIGO, dados.CORES_CLASSE):
        ct.SetColorEntry(int(cod), tuple(int(cor[i:i + 2], 16) for i in (1, 3, 5)) + (255,))
    b.SetRasterColorTable(ct)
    b.SetNoDataValue(0)
    b.SetDescription("classe (código MapBiomas representativo)")
    b.WriteArray(cls)
    ds = None


def gravar_probs(caminho, probs, ref_ds):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(caminho, probs.shape[2], probs.shape[1], probs.shape[0], gdal.GDT_Byte,
                    ["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(ref_ds.GetGeoTransform())
    ds.SetProjection(ref_ds.GetProjection())
    for i, nome in enumerate(dados.CLASS_NOMES):
        b = ds.GetRasterBand(i + 1)
        b.WriteArray(np.round(probs[i] * 255).astype(np.uint8))
        b.SetDescription("P(%s) x255" % nome)
    ds = None


def referencia_na_grade(caminho, ref_ds):
    """Máscara de referência (códigos MapBiomas) -> índices de classe, na grade da imagem."""
    m = gdal.Open(caminho)
    mesma = (m.RasterXSize, m.RasterYSize) == (ref_ds.RasterXSize, ref_ds.RasterYSize) and \
        np.allclose(m.GetGeoTransform(), ref_ds.GetGeoTransform())
    if not mesma:
        gt, W, H = ref_ds.GetGeoTransform(), ref_ds.RasterXSize, ref_ds.RasterYSize
        m = gdal.Warp("/vsimem/ref.tif", m, dstSRS=ref_ds.GetProjection(), resampleAlg="near",
                      outputBounds=(gt[0], gt[3] + H * gt[5], gt[0] + W * gt[1], gt[3]),
                      width=W, height=H, dstNodata=0)
    return dados.LUT[m.ReadAsArray()]


def visualizar(caminho, pan, valido, cls_idx, st, ref, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(dados.CORES_CLASSE)
    paineis = [("PAN", None), ("predição", cls_idx)] + ([("referência", ref)] if ref is not None else [])
    fig, ax = plt.subplots(1, len(paineis), figsize=(6.2 * len(paineis), 6.8))
    ax = np.atleast_1d(ax)
    for a, (nome, arr) in zip(ax, paineis):
        if arr is None:
            a.imshow(np.where(valido, pan, np.nan), cmap="gray", vmin=st["p2"], vmax=st["p98"])
        else:
            vis = np.ma.masked_where(~valido | (arr >= len(dados.CLASS_NOMES)), arr)
            a.imshow(vis, cmap=cmap, vmin=-0.5, vmax=len(dados.CLASS_NOMES) - 0.5,
                     interpolation="nearest")
        a.set_title(nome)
        a.set_xticks([])
        a.set_yticks([])
    n_val = max(1, int(valido.sum()))
    legenda = [Patch(color=c, label="%s  %.1f%%" % (n, 100 * ((cls_idx == i) & valido).sum() / n_val))
               for i, (n, c) in enumerate(zip(dados.CLASS_NOMES, dados.CORES_CLASSE))]
    fig.legend(handles=legenda, loc="lower center", ncol=len(legenda), fontsize=11,
               title="% da área prevista")
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagem", required=True, help="GeoTIFF PAN (1 banda)")
    ap.add_argument("--modelo", default=MODELO_PADRAO, help="modelo_melhor.pth de um run seg_*")
    ap.add_argument("--cena", default=None,
                    help="cena do receita.json cujas estatísticas usar (padrão: pelo caminho "
                         "da imagem; se não achar, calcula na própria imagem)")
    ap.add_argument("--referencia", default=None,
                    help="máscara (códigos MapBiomas) só p/ comparar no PNG")
    ap.add_argument("--saida", default=None, help="pasta de saída")
    ap.add_argument("--passo", type=int, default=128, help="px entre janelas de 256")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--probs", action="store_true", help="grava também probabilidades.tif")
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
    if abs(gt[1] - par["resolucao"]) > 1e-3 or abs(gt[5] + par["resolucao"]) > 1e-3:
        print("[aviso] resolução %.3f x %.3f m, o modelo foi treinado a %.2f m"
              % (gt[1], -gt[5], par["resolucao"]), flush=True)
    b = ds.GetRasterBand(1)
    pan = b.ReadAsArray().astype(np.float32)
    nd = b.GetNoDataValue()
    if nd is not None:
        pan = np.where(pan == nd, np.nan, pan)
    if nd is None and receita["parametros"].get("pan_zero_nodata"):
        pan = np.where(pan == 0, np.nan, pan)     # PAN 0 = sem dado (moldura de mosaico)
    valido = np.isfinite(pan)
    print("[imagem] %s | %dx%d | %.1f%% válido" % (args.imagem, pan.shape[1], pan.shape[0],
                                                   100 * valido.mean()), flush=True)

    cenas = {c: v["estatisticas"] for c, v in receita["resultado"]["por_cena"].items()}
    cena = args.cena or next((c for c in cenas if c in os.path.abspath(args.imagem)), None)
    if cena:
        if cena not in cenas:
            raise SystemExit("cena %r não está no receita (%s)" % (cena, ", ".join(cenas)))
        st, origem_st = cenas[cena], "receita.json do dataset, cena %s" % cena
    else:
        st = estatisticas_da_imagem(pan, valido, par["janela"], par["niveis"])
        origem_st = "calculadas na própria imagem (%d blocos de 512)" % st["blocos"]
    print("[estatisticas] %s | p2=%.1f p98=%.1f" % (origem_st, st["p2"], st["p98"]), flush=True)

    X = bandas(pan, valido, st, par["janela"], par["niveis"])
    X[:, ~valido] = 0
    print("[bandas] %s | %.0fs" % (receita["bandas"], time.monotonic() - t0), flush=True)

    probs, n_jan = prever(modelo, X, P, args.passo, args.lote, dev, ta.get("margem", 15))
    cls_idx = probs.argmax(0).astype(np.uint8)
    print("[predicao] %d janelas de %d px (passo %d) | %.0fs"
          % (n_jan, P, args.passo, time.monotonic() - t0), flush=True)

    tile = os.path.basename(os.path.dirname(os.path.abspath(args.imagem)))
    stem = os.path.splitext(os.path.basename(args.imagem))[0]
    nome = "_".join(p for p in (cena, tile if stem == "imagem" else stem) if p)
    base_inf = os.path.join(ambiente.MODELS, "unet_cnn") if tipo_modelo(args.modelo) == "unet" \
        else BASE
    saida = args.saida or os.path.join(base_inf, "inferencias", nome_modelo, nome)
    os.makedirs(saida, exist_ok=True)

    codigos = np.asarray(dados.REPR_CODIGO, np.uint8)[cls_idx]
    codigos[~valido] = 0
    gravar_classes(os.path.join(saida, "classes.tif"), codigos, ds)
    if args.probs:
        gravar_probs(os.path.join(saida, "probabilidades.tif"), probs, ds)

    ref = concord = None
    if args.referencia:
        ref = referencia_na_grade(args.referencia, ds)
        ok = valido & (ref < len(dados.CLASS_NOMES))
        concord = float((ref[ok] == cls_idx[ok]).mean()) if ok.any() else None
    titulo = "%s  |  %s (epoca %s)%s" % (
        nome, nome_modelo, ck["epoca"],
        "  |  concordância c/ referência: %.1f%%" % (100 * concord) if concord is not None else "")
    visualizar(os.path.join(saida, "classes.png"), pan, valido, cls_idx, st, ref, titulo)

    n_val = max(1, int(valido.sum()))
    fr = {n: round(float(((cls_idx == i) & valido).sum()) / n_val, 4)
          for i, n in enumerate(dados.CLASS_NOMES)}
    with open(os.path.join(saida, "info.json"), "w", encoding="utf-8") as f:
        json.dump({"imagem": os.path.abspath(args.imagem), "modelo": os.path.abspath(args.modelo),
                   "epoca_modelo": ck["epoca"], "val_miou_modelo": ck["val_miou"],
                   "dataset_modelo": dataset, "estatisticas": {"origem": origem_st, **st},
                   "janelas": {"patch": P, "passo": args.passo, "n": n_jan},
                   "codigos_saida": dict(zip(dados.CLASS_NOMES, map(int, dados.REPR_CODIGO))),
                   "fracao_prevista": fr, "referencia": args.referencia,
                   "concordancia_referencia": concord,
                   "tempo_s": round(time.monotonic() - t0, 1)},
                  f, indent=2, ensure_ascii=False)
    print("[classes] " + "  ".join("%s=%.1f%%" % (n, 100 * v) for n, v in fr.items()))
    if concord is not None:
        print("[referencia] concordância com %s: %.1f%%" % (args.referencia, 100 * concord))
    print("OK -> %s (classes.tif, classes.png, info.json%s) | %.0fs"
          % (saida, ", probabilidades.tif" if args.probs else "", time.monotonic() - t0))


if __name__ == "__main__":
    main()
