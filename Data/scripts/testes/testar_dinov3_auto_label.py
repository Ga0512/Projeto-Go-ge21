#!/usr/bin/env python3
"""Teste experimental: auto-label por similaridade com DINOv3.

Ideia: os patch features do DINOv3 (1 por 16x16 px) separam classes por
similaridade. Usamos o tile refinado r09_c10 como gabarito:
  - extrai as patch features do tile (janelas 512x512, não sobrepostas);
  - computa o CENTROIDE de cada classe (4 classes do config.CLASS_MAP);
  - num tile vizinho NÃO refinado, classifica cada patch pela similaridade de
    cosseno com o centroide mais próximo;
  - remonta a máscara 16x16 e compara com o MapBiomas do tile alvo.

É um teste pode-ficar-ruim: não toca em nada dos dados. Saída em
experimentos/auto_label_dinov3/.

Uso:
  .venv/bin/python scripts/testes/testar_dinov3_auto_label.py
"""
import os
import sys

import numpy as np
import torch
from osgeo import gdal

sys.path.insert(0, "/tmp/dinov3")          # repo oficial do DINOv3
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelos"))
import config  # noqa: E402

gdal.UseExceptions()

CKPT = "/mnt/d/Projects/DinoV3/models/dinov3_vits16.pth"
TILE_GAB = "dados/spot5/cena_013-014_2007-06-14/tiles/r09_c10"
TILE_ALVO = "dados/spot5/cena_013-014_2007-06-14/tiles/r09_c11"
OUT = "experimentos/auto_label_dinov3"
BLOCO = 512          # janela do ViT (2048 = 4x4 janelas), múltiplo de 16
PATCH = 16
EMB = 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def carregar_modelo():
    from dinov3.hub.backbones import dinov3_vits16
    torch.hub.set_dir("/tmp/torchhub")
    m = dinov3_vits16(pretrained=True, weights=CKPT)
    m.eval().cuda()
    return m


def ler(pan, mask):
    ds = gdal.Open(pan)
    a = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None
    ds = gdal.Open(mask)
    m = ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    ds = None
    return a, m


def features(model, pan):
    """Patch features na grade (H/PATCH, W/PATCH, EMB), janela a janela."""
    H, W = pan.shape
    hp, wp = H // PATCH, W // PATCH
    out = np.zeros((hp, wp, EMB), np.float32)
    nby, nbx = H // BLOCO, W // BLOCO
    for r in range(nby):
        for c in range(nbx):
            bloco = pan[r * BLOCO:(r + 1) * BLOCO, c * BLOCO:(c + 1) * BLOCO]
            rgb = np.stack([bloco] * 3, -1)
            x = (rgb / 255.0 - MEAN) / STD
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().cuda()
            with torch.no_grad():
                o = model.forward_features(x)
            f = o["x_norm_patchtokens"][0].cpu().numpy()
            fp = f.reshape(BLOCO // PATCH, BLOCO // PATCH, EMB)
            out[r * BLOCO // PATCH:(r + 1) * BLOCO // PATCH,
                c * BLOCO // PATCH:(c + 1) * BLOCO // PATCH] = fp
    return out


def normalizar(f):
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9)


def centroides(f, mp):
    """Centroide L2-normalizado por classe (config.CLASS_MAP) dos patches cuja
    classe majoritária (por 16x16) é a da classe."""
    fn = normalizar(f.reshape(-1, EMB))
    mpf = mp.ravel()
    cl = np.full(mpf.size, 255, np.int8)
    for cod, idx in config.CLASS_MAP.items():
        cl[mpf == cod] = idx
    cent = {}
    for idx in sorted(set(config.CLASS_MAP.values())):
        sel = cl == idx
        if sel.sum() > 0:
            v = fn[sel].mean(0)
            cent[idx] = v / (np.linalg.norm(v) + 1e-9)
    return cent


def classificar(f, cent):
    fn = normalizar(f.reshape(-1, EMB))
    idxs = sorted(cent)
    sim = fn @ np.stack([cent[i] for i in idxs]).T
    return np.array(idxs)[sim.argmax(1)]


def expandir(grid):
    """Grade de patches (Hf, Wf) -> raster 16x16 expandido."""
    return np.kron(grid, np.ones((PATCH, PATCH), np.uint8))


def main():
    os.makedirs(OUT, exist_ok=True)
    model = carregar_modelo()
    print("modelo carregado")

    pan_gab, mask_gab = ler(os.path.join(TILE_GAB, "imagem.tif"),
                            os.path.join(TILE_GAB, "mascara_refinada.tif"))
    f_gab = features(model, pan_gab)
    mp = np.zeros((f_gab.shape[0], f_gab.shape[1]), np.uint8)
    for r in range(f_gab.shape[0]):
        for c in range(f_gab.shape[1]):
            bloco = mask_gab[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
            mp[r, c] = np.bincount(bloco.ravel()).argmax()
    cent = centroides(f_gab, mp)
    print("centroides por classe:", {i: 1 for i in cent})

    pan_alvo, mask_alvo = ler(os.path.join(TILE_ALVO, "imagem.tif"),
                              os.path.join(TILE_ALVO, "mascara_mapbiomas.tif"))
    f_alvo = features(model, pan_alvo)
    pred_idx = classificar(f_alvo, cent)          # índice de classe do modelo
    repr_cod = np.array(config.REPR_CODIGO)
    pred_cod = repr_cod[pred_idx]
    grid = pred_cod.reshape(f_alvo.shape[0], f_alvo.shape[1])
    auto = expandir(grid)

    ds = gdal.Open(os.path.join(TILE_ALVO, "imagem.tif"))
    gt, wkt = ds.GetGeoTransform(), ds.GetProjection()
    ds = None
    d = gdal.GetDriverByName("GTiff").Create(
        os.path.join(OUT, "auto_label_r09_c11.tif"), auto.shape[1], auto.shape[0],
        1, gdal.GDT_Byte, ["COMPRESS=DEFLATE", "TILED=YES"])
    d.SetGeoTransform(gt); d.SetProjection(wkt)
    b = d.GetRasterBand(1); b.SetNoDataValue(0); b.WriteArray(auto); d = None

    val = (mask_alvo != 0) & (auto != 0)
    if val.any():
        conc = (mask_alvo[val] == auto[val]).mean()
        print("concordância auto-label × MapBiomas (pixel a pixel): %.1f%%"
              % (100 * conc))
        # por classe
        for cod in (3, 4, 15, 33):
            ref = (mask_alvo == cod) & (auto != 0)
            ac = (mask_alvo[ref] == auto[ref]).mean() if ref.any() else float("nan")
            print("  classe %2d: %s %.1f%% (%d px)" %
                  (cod, config.CLASS_NOMES[config.CLASS_MAP[cod]], 100 * ac,
                   int(ref.sum())))

    # PNG: PAN | MapBiomas | auto-label
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p2, p98 = np.percentile(pan_alvo, [2, 98])
    pg = np.clip((pan_alvo - p2) / (p98 - p2), 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(pg, cmap="gray"); ax[0].set_title("PAN alvo (r09_c11)")
    ax[1].imshow(mask_alvo, cmap="tab10", interpolation="nearest")
    ax[1].set_title("MapBiomas")
    ax[2].imshow(auto, cmap="tab10", interpolation="nearest")
    ax[2].set_title("Auto-label DINOv3")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "comparacao.png"), dpi=110)
    print("saída:", OUT)


if __name__ == "__main__":
    main()