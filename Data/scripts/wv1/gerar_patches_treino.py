#!/usr/bin/env python3
"""Gera patches de treino (janelas 512x512) do par features(PAN) + máscara.

Pipeline:
  1. Valida alinhamento (CRS, transform, dimensões) entre img e mask.
  2. Desliza janelas de PATCHxPATCH com STRIDE (overlap configurável),
     cobrindo a imagem inteira incl. bordas (janela final garantida).
  3. Filtra janelas com pouca área válida (máscara != 0).
  4. Aplica augmentations determinísticas (rotações/flips/T) em img+mask
     juntas (mesma transformação nos dois).
  5. Salva img/<idx>.npz (float16/float32, shape PxPx4) e mask/<idx>.npz
     (uint8, shape PxP), + manifest.csv e resumo de classes.

Saída (--out, default dados/wv1/patches/treino):
  treino/
    img/    000000.npz, 000001.npz, ...   (bandas: PAN, LocalSTD, GLCM_Entropy, Gradient)
    mask/   000000.npz, ...               (códigos MapBiomas; 0 = nodata/ignorar)
    manifest.csv                          (idx,x,y,aug,valid_frac)
    resumo_classes.txt

Uso:
  .venv/bin/python scripts/wv1/gerar_patches_treino.py
  .venv/bin/python scripts/wv1/gerar_patches_treino.py \
      --mask dados/wv1/mascaras/mask_2007_suavizada.tif \
      --out dados/wv1/patches/treino_2007 --aug full --stride 256
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

DEFAULT_IMG = caminhos.WV1_FEATURES
DEFAULT_MASK = caminhos.WV1_MASK_2008
DEFAULT_OUT = os.path.join(caminhos.WV1_PATCHES, "treino")

PATCH = 512
STRIDE = 256          # 256 = 50% de overlap (sem perder info nas bordas)
MIN_VALID = 0.90      # fração mínima de pixels com classe (não-nodata) na janela
DTYPE = "float16"
SEED = 42

# augmentations: nome -> função que transforma (img, mask) da mesma forma
AUGS = {
    "id": lambda im, mk: (im, mk),
    "hflip": lambda im, mk: (im[:, ::-1], mk[:, ::-1]),
    "vflip": lambda im, mk: (im[::-1, :], mk[::-1, :]),
    "rot90": lambda im, mk: (np.rot90(im, 1), np.rot90(mk, 1)),
    "rot180": lambda im, mk: (np.rot90(im, 2), np.rot90(mk, 2)),
    "rot270": lambda im, mk: (np.rot90(im, 3), np.rot90(mk, 3)),
    "transpose": lambda im, mk: (im.transpose(1, 0, 2), mk.T),
    "anti_transpose": lambda im, mk: (
        np.rot90(im.transpose(1, 0, 2), 2), np.rot90(mk.T, 2)),
}


def validate_alignment(img_path, mask_path):
    """Confere CRS/transform/dimensões iguais entre img e mask."""
    i = gdal.Open(img_path)
    m = gdal.Open(mask_path)
    assert i.RasterXSize == m.RasterXSize and i.RasterYSize == m.RasterYSize, \
        "dimensões diferentes: img %dx%d mask %dx%d" % (
            i.RasterXSize, i.RasterYSize, m.RasterXSize, m.RasterYSize)
    assert i.GetProjection() == m.GetProjection(), "CRS diferentes"
    gi, gm = i.GetGeoTransform(), m.GetGeoTransform()
    assert all(abs(a - b) < 1e-9 for a, b in zip(gi, gm)), "affine transform diferente"
    return i.RasterXSize, i.RasterYSize


def build_windows(W, H, patch, stride):
    """Janelas (x, y, x1, y1) cobrindo toda a imagem com overlap imparcial."""
    wins = []
    y = 0
    while y + patch <= H:
        x = 0
        while x + patch <= W:
            wins.append((x, y, x + patch, y + patch))
            x += stride
        if x - stride < W - patch:          # garante a janela da última coluna
            wins.append((W - patch, y, W, y + patch))
        y += stride
    if y - stride < H - patch:              # garante a janela da última linha
        x = 0
        while x + patch <= W:
            wins.append((x, H - patch, x + patch, H))
            x += stride
        if x - stride < W - patch:
            wins.append((W - patch, H - patch, W, H))
    return wins


def main():
    ap = argparse.ArgumentParser(description="Gera patches de treino img+mask")
    ap.add_argument("--img", default=DEFAULT_IMG)
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--patch", type=int, default=PATCH)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--min-valid", type=float, default=MIN_VALID)
    ap.add_argument("--aug", choices=list(AUGS) + ["light", "full"],
                    default="light",
                    help="augmentations: id, hflip, vflip, rot90, rot180, rot270, "
                         "transpose, anti_transpose, light (4), full (8)")
    ap.add_argument("--dtype", choices=["float16", "float32"], default=DTYPE)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="Fração das janelas a usar (0.5 = metade, amostragem uniforme)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    patch = args.patch
    stride = args.stride

    os.makedirs(args.out, exist_ok=True)
    img_dir = os.path.join(args.out, "img")
    mask_dir = os.path.join(args.out, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.csv")

    if os.path.exists(manifest_path) and not args.overwrite:
        raise SystemExit("já existe %s (use --overwrite para regenerar)" % manifest_path)

    W, H = validate_alignment(args.img, args.mask)
    wins = build_windows(W, H, patch, stride)
    if args.frac < 1.0:
        keep = max(1, int(round(len(wins) * args.frac)))
        step = len(wins) / keep
        wins = [wins[int(i * step)] for i in range(keep)]
    if args.aug == "light":
        augs = ["id", "hflip", "vflip", "rot180"]
    elif args.aug == "full":
        augs = list(AUGS)
    else:
        augs = [args.aug]
    print("grade %dx%d | patch=%d stride=%d | janelas=%d | aug=%s (%d) | min_valid=%.2f"
          % (W, H, patch, stride, len(wins), augs, len(augs), args.min_valid))

    img_ds = gdal.Open(args.img)
    mask_ds = gdal.Open(args.mask)
    mask_b = mask_ds.GetRasterBand(1)
    nbands = img_ds.RasterCount

    dtype = np.float16 if args.dtype == "float16" else np.float32
    mb = 2 if dtype == np.float16 else 4
    est = len(wins) * len(augs) * (patch * patch * (nbands * mb + 1))
    print("estimativa de disco: %.1f GB (antes de compressão)" % (est / 1e9))

    idx = 0
    t0 = time.time()
    rows_manifest = []
    class_hist = {}

    for (x, y, x1, y1) in wins:
        img_patch = np.stack(
            [img_ds.GetRasterBand(b).ReadAsArray(x, y, patch, patch)
             for b in range(1, nbands + 1)], axis=-1).astype(np.float32)
        mask_patch = mask_b.ReadAsArray(x, y, patch, patch).astype(np.uint8)

        valid_frac = (mask_patch != 0).mean()
        if valid_frac < args.min_valid:
            continue

        for aug in augs:
            im_a, mk_a = AUGS[aug](img_patch, mask_patch)
            # embeddings idênticos para img e mask: nenhuma transformação
            # redimensiona (todas são isometrias do grid), garantido por def.
            name = "%06d" % idx
            np.savez_compressed(os.path.join(img_dir, name + ".npz"),
                                im_a.astype(dtype), allow_pickle=False)
            np.savez_compressed(os.path.join(mask_dir, name + ".npz"),
                                mk_a, allow_pickle=False)
            rows_manifest.append([idx, x, y, aug, round(valid_frac, 4)])
            for c in np.unique(mk_a):
                class_hist[int(c)] = class_hist.get(int(c), 0) + 1
            idx += 1
        if len(rows_manifest) % 5000 < len(augs) and rows_manifest:
            print("  %d patches (%.0fs)" % (idx, time.time() - t0), flush=True)

    with open(manifest_path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["idx", "x", "y", "aug", "valid_frac"])
        cw.writerows(rows_manifest)

    resumo = os.path.join(args.out, "resumo_classes.txt")
    with open(resumo, "w") as f:
        f.write("classes (soma de pixels sobre todos os patches):\n")
        for c, n in sorted(class_hist.items()):
            f.write("%5d : %d\n" % (c, n))
    print("pronto: %d patches em %.0fs" % (idx, time.time() - t0))
    print("  manifest:", manifest_path)
    print("  resumo:  ", resumo)


if __name__ == "__main__":
    main()