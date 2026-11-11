#!/usr/bin/env python3
"""Prévia da regra de água: limiar por cena + mancha grande + vizinhança da água.

Regra (todas as partes configuráveis):
  1. limiar por CENA: p95 da PAN na água INTERIOR da cena inteira;
  2. componente conexo: só escuro que forma mancha >= MIN_PX px (água é densa,
     vegetação escura é pontilhada);
  3. vizinhança da água: só age dentro de VIZ_AGUA px da água existente (33).

Aplica sobre uma CÓPIA da mascara_refinada: onde a regra diz água e a máscara
tem rótulo, vira 33. Gera PNGs antes/depois por área.

Saída: experimentos/agua_pan/<cena>[/<tile>]/mascara_agua_pan.tif + borda_agua.png
"""
import os
import sys

import numpy as np
from osgeo import gdal
from scipy.ndimage import binary_dilation, label

gdal.UseExceptions()
RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
import caminhos  # noqa: E402

OUT = os.path.join(RAIZ, "experimentos", "agua_pan")
BORDA_PX = 8          # interior = longe de qualquer troca de código
MIN_PX = 200          # componente conexo mínimo do "escuro" (mancha grande)
VIZ_AGUA = 12         # px de vizinhança da água (12 px = 30 m) — 0 desliga
QUANTIL = 0.95        # quantil da água interior usado como limiar

CENAS = {
    "recorte_2007-09-10": [
        (caminhos.SPOT5_RECORTE_IMAGEM, caminhos.SPOT5_RECORTE_REFINADA)],
    "cena_013-014_2007-06-14": [
        (os.path.join(caminhos.SPOT5_CENA_013_TILES, "r06_c08", "imagem.tif"),
         os.path.join(caminhos.SPOT5_CENA_013_TILES, "r06_c08", "mascara_refinada.tif"))],
    "colinas_do_sul": [
        (os.path.join(caminhos.SPOT5_COLINAS_TILES, t, "imagem.tif"),
         os.path.join(caminhos.SPOT5_COLINAS_TILES, t, "mascara_refinada.tif"))
        for t in ["r00_c06", "r04_c02", "r06_c05", "r10_c05"]],
}


def ler(band, x0, y0, w, h, W, H, dtype=np.uint8):
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    a = band.ReadAsArray(xa, ya, xb - xa, yb - ya).astype(dtype)
    pad = ((ya - y0, y0 + h - yb), (xa - x0, x0 + w - xb))
    if any(v for p in pad for v in p):
        a = np.pad(a, pad, mode="reflect")
    return a


def limiar_por_cena(areas):
    """p95 da PAN na água interior da cena inteira (concatena as áreas)."""
    vals = []
    FA = 1024
    for pds, mds, pan, masc, W, H, _, _ in areas:
        for y0 in range(0, H, FA):
            h = min(FA, H - y0)
            yp, yq = max(0, y0 - BORDA_PX), min(H, y0 + h + BORDA_PX)
            m = masc.ReadAsArray(0, yp, W, yq - yp).astype(np.uint8)
            p = pan.ReadAsArray(0, yp, W, yq - yp).astype(np.float32)
            c0 = y0 - yp
            mc, pc = m[c0:c0 + h], p[c0:c0 + h]
            agua = mc == 33
            interior = agua & ~binary_dilation(~agua, iterations=BORDA_PX)
            vals.append(pc[interior])
    vals = np.concatenate(vals)
    if vals.size == 0:
        return None
    return float(np.percentile(vals, 100 * QUANTIL))


def aplicar(pan, masc, W, H, limiar, viz, saida, gt, wkt):
    """Copiar a máscara e aplicar a regra. Devolve (n_viram, n_escuro_mancha)."""
    # máscara de "escuro" em toda a cena (só onde há rótulo), com vizinhança
    FA = 1024
    esc = np.zeros((H, W), bool)
    base = masc.ReadAsArray(0, 0, W, H).astype(np.uint8)
    if viz:
        agua_exist = base == 33
        vizinho = binary_dilation(agua_exist, iterations=viz)
    for y0 in range(0, H, FA):
        h = min(FA, H - y0)
        p = pan.ReadAsArray(0, y0, W, h).astype(np.float32)
        m = base[y0:y0 + h]
        e = (p <= limiar) & (m != 0)
        if viz:
            e &= vizinho[y0:y0 + h]
        esc[y0:y0 + h] = e
    lbl, n = label(esc)
    tam = np.bincount(lbl.ravel())
    tam[0] = 0
    manter = tam >= MIN_PX
    filt = manter[lbl]
    # aplica: só vira água onde hoje NÃO é água (33)
    out = base.copy()
    vira = filt & (base != 33)
    nv = int(vira.sum())
    out[vira] = 33
    d = gdal.GetDriverByName("GTiff").Create(saida, W, H, 1, gdal.GDT_Byte,
                                             ["TILED=YES", "COMPRESS=DEFLATE"])
    d.SetGeoTransform(gt)
    d.SetProjection(wkt)
    ob = d.GetRasterBand(1)
    ob.SetNoDataValue(0)
    ob.WriteArray(out)
    d = None
    return nv, int(filt.sum())


def janela_mix(masc, W, H, lado=640):
    L = min(lado, W, H)
    melhores = []
    for y0 in range(0, H - L + 1, max(L // 2, 1)):
        for x0 in range(0, W - L + 1, max(L // 2, 1)):
            m = masc.ReadAsArray(x0, y0, L, L).astype(np.uint8)
            fa = (m == 33).mean()
            fr = ((m != 0) & (m != 33)).mean()
            if 0.15 < fa < 0.85 and fr > 0.15:
                dif = np.abs((m[:, :-1] != 33).astype(np.int8) - (m[:, 1:] != 33).astype(np.int8))
                melhores.append((dif.mean(), y0, x0))
    if not melhores:
        return 0, 0
    _, y0, x0 = max(melhores)
    return y0, x0


def png(pan, masc, W, H, masc_depois, y0, x0, lado, out_png, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x0 = min(max(x0, 0), W - lado)
    y0 = min(max(y0, 0), H - lado)
    p = pan.ReadAsArray(x0, y0, lado, lado).astype(np.float32)
    a = masc.ReadAsArray(x0, y0, lado, lado)
    mds = gdal.Open(masc_depois)
    d = mds.GetRasterBand(1).ReadAsArray(x0, y0, lado, lado)
    p2, p98 = np.percentile(p, [2, 98])
    pg = np.clip((p - p2) / max(p98 - p2, 1), 0, 1)
    agua = np.zeros((lado, lado, 4))
    agua[a == 33] = [0.1, 0.4, 1.0, 0.55]
    agua[d == 33] = [0.1, 0.4, 1.0, 0.7]
    muda = (d == 33) & (a != 33)
    agua[muda] = [1.0, 0.2, 0.1, 0.75]
    sai = (d != 33) & (a == 33)
    agua[sai] = [1.0, 0.8, 0.0, 0.6]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    axes[0].imshow(pg, cmap="gray", vmin=0, vmax=1)
    axes[0].imshow(agua)
    axes[0].set_title("PAN + água\n%s" % titulo, fontsize=10)
    axes[1].imshow(a, cmap="gray", vmin=0, vmax=33, interpolation="nearest")
    axes[1].set_title("máscara refinada (antes)", fontsize=10)
    axes[2].imshow(pg, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(agua)
    axes[2].set_title("depois: vermelho=vira água, amarelo=saiu água", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    preparadas, limiares = {}, {}
    for cena, areas in CENAS.items():
        abs_areas = []
        for pan_p, masc_p in areas:
            pds = gdal.Open(pan_p)
            mds = gdal.Open(masc_p)
            abs_areas.append((pds, mds, pds.GetRasterBand(1), mds.GetRasterBand(1),
                              mds.RasterXSize, mds.RasterYSize,
                              mds.GetGeoTransform(), mds.GetProjection()))
        lim = limiar_por_cena(abs_areas)
        preparadas[cena] = abs_areas
        limiares[cena] = lim
        print("%-24s limiar(p95 agua interior) = %s" % (cena, lim))
    for cena, areas in CENAS.items():
        lim = limiares[cena]
        for i, (pan_p, masc_p) in enumerate(areas):
            pds, mds, pan, masc, W, H, gt, wkt = preparadas[cena][i]
            nome = os.path.basename(os.path.dirname(masc_p)) if len(areas) > 1 else cena
            pasta = os.path.join(OUT, cena, nome)
            os.makedirs(pasta, exist_ok=True)
            saida = os.path.join(pasta, "mascara_agua_pan.tif")
            nv, ne = aplicar(pan, masc, W, H, lim, VIZ_AGUA, saida, gt, wkt)
            print("%-24s %-8s virou água=%6d (%.2f%%) | mancha escura=%6d | limiar=%d" %
                  (cena, nome, nv, 100 * nv / (W * H), ne, lim))
            y0, x0 = janela_mix(masc, W, H)
            png(pan, masc, W, H, saida, y0, x0, 640,
                os.path.join(pasta, "borda_agua.png"),
                "%s %s | limiar %d | mancha>=%dpx | viz %dpx" % (cena, nome, lim, MIN_PX, VIZ_AGUA))
    print("\nSaída:", OUT)


if __name__ == "__main__":
    main()