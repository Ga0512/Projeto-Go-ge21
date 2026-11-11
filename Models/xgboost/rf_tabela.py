#!/usr/bin/env python3
"""Tabela de respostas pré-calculada: a floresta inteira vira um lookup.

O modelo olha só 3 números por pixel (PAN, GLCM, rugosidade), então a saída da
floresta é uma função de 3 variáveis. Dá para avaliá-la UMA vez numa grade e
guardar o resultado: a inferência passa a ser 1 leitura por pixel em vez das
~12.000 que a travessia de 500 árvores exige, e a profundidade das árvores
deixa de importar.

A grade não é uniforme. A PAN é Byte, então seus 256 valores inteiros são a
grade exata -- os 471 limiares que a floresta usa nessa feature caem todos na
grade de 0,5 entre inteiros, e nenhum pixel pode cair entre dois inteiros:
nessa feature a tabela é EXATA. GLCM e rugosidade são float com ~1,1 e ~1,7
milhão de limiares distintos, impossíveis de tabular sem perda; as bordas dos
bins vão nos QUANTIS desses limiares, o que concentra resolução onde a floresta
realmente decide (a rugosidade, por exemplo, vai de -1,7 a 11 mas decide quase
tudo perto de zero -- bins uniformes desperdiçariam metade da tabela na cauda).

O erro da quantização é medido, não estimado: `--verificar` compara a tabela
contra a travessia da MESMA floresta, pixel a pixel.

Uso (no WSL, a partir de `Models/`):
  # montar (uma vez)
  ../Data/.venv/bin/python xgboost/rf_tabela.py --montar \\
      --floresta xgboost/runs/rf_colinas_livre/floresta \\
      --tabela xgboost/runs/rf_colinas_livre/tabela_256.npz --bins 256

  # conferir contra a floresta, num tile
  ../Data/.venv/bin/python xgboost/rf_tabela.py --verificar \\
      --floresta xgboost/runs/rf_colinas_livre/floresta \\
      --tabela xgboost/runs/rf_colinas_livre/tabela_256.npz \\
      --imagem ../Data/dados/spot5/colinas_do_sul/original/colinas_3b.vrt

  # classificar a cena
  ../Data/.venv/bin/python xgboost/rf_tabela.py \\
      --tabela xgboost/runs/rf_colinas_livre/tabela_256.npz \\
      --imagem ../Data/dados/spot5/colinas_do_sul/original/colinas_3b.vrt \\
      --saida xgboost/runs/rf_colinas_livre/classes_tabela.tif
"""

import argparse
import json
import os
import sys
import time

import ambiente  # noqa: F401
import numpy as np
import torch
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rf_gpu import NODATA_SAIDA, FlorestaGPU, ler_bloco  # noqa: E402
from rf_kernel import FlorestaTriton  # noqa: E402

gdal.UseExceptions()

N_PAN = 256          # a PAN é Byte: 256 valores inteiros possíveis, e ponto


# ------------------------------------------------------------ bordas dos bins
def limiares(pasta, f, passo=4_000_000):
    """Limiares distintos que a floresta usa na feature f (só nós internos)."""
    feat = np.load(os.path.join(pasta, "feat.npy"), mmap_mode="r")
    lim = np.load(os.path.join(pasta, "lim.npy"), mmap_mode="r")
    esq = np.load(os.path.join(pasta, "esq.npy"), mmap_mode="r")
    out = []
    for i in range(0, len(feat), passo):
        s = slice(i, i + passo)
        idx = np.arange(i, min(i + passo, len(feat)))
        m = (feat[s] == f) & (esq[s] != idx)
        v = np.asarray(lim[s])[m]
        out.append(v[np.isfinite(v)])
    return np.unique(np.concatenate(out))


def bordas(pasta, f, n):
    """n-1 bordas de bin nos quantis dos limiares, mais os representantes.

    O representante de um bin é o ponto médio dele; é nesse ponto que a
    floresta é avaliada para preencher a tabela.
    """
    u = limiares(pasta, f)
    if len(u) <= n - 1:
        b = u.astype(np.float32)                 # cabe exato
    else:
        pos = np.linspace(0, len(u) - 1, n - 1).round().astype(np.int64)
        b = u[np.unique(pos)].astype(np.float32)
    lo, hi = float(u[0]), float(u[-1])
    ext = np.concatenate([[lo - (float(b[0]) - lo + 1e-3)], b,
                          [hi + (hi - float(b[-1]) + 1e-3)]])
    rep = ((ext[:-1] + ext[1:]) / 2).astype(np.float32)
    return b, rep, len(u)


# ------------------------------------------------------------ montagem
def montar(pasta_fl, destino, n1, n2, dev, chunk=4_000_000):
    fl = FlorestaTriton(FlorestaGPU(pasta_fl, dev))
    print("floresta: %d arvores, profundidade %d, %d nos"
          % (fl.fl.n_arvores, fl.fl.profundidade, fl.fl.n_nos), flush=True)

    b1, r1, u1 = bordas(pasta_fl, 1, n1)
    b2, r2, u2 = bordas(pasta_fl, 2, n2)
    n1, n2 = len(r1), len(r2)
    print("GLCM      : %7d limiares distintos -> %4d bins" % (u1, n1), flush=True)
    print("rugosidade: %7d limiares distintos -> %4d bins" % (u2, n2), flush=True)

    total = N_PAN * n1 * n2
    print("tabela: %d x %d x %d = %d entradas (%.1f MB)"
          % (N_PAN, n1, n2, total, total / 1e6), flush=True)

    v0 = torch.arange(N_PAN, dtype=torch.float32, device=dev)
    v1 = torch.as_tensor(r1, device=dev)
    v2 = torch.as_tensor(r2, device=dev)
    tab = np.empty(total, np.uint8)

    t0 = time.monotonic()
    for i0 in range(0, total, chunk):
        i1 = min(i0 + chunk, total)
        idx = torch.arange(i0, i1, device=dev, dtype=torch.int64)
        a = idx // (n1 * n2)
        b = (idx // n2) % n1
        c = idx % n2
        X = torch.stack([v0[a], v1[b], v2[c]], 1).contiguous()
        tab[i0:i1] = fl.indices(X).to(torch.uint8).cpu().numpy()
        feito = i1 / total
        print("  %6.2f%% | %.0fs decorridos | ~%.0fs restantes"
              % (100 * feito, time.monotonic() - t0,
                 (time.monotonic() - t0) * (1 / feito - 1)), flush=True)

    dt = time.monotonic() - t0
    print("montada em %.0fs (%.3f Mentradas/s)" % (dt, total / 1e6 / dt), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(destino)), exist_ok=True)
    np.savez(destino, tabela=tab.reshape(N_PAN, n1, n2), b1=b1, b2=b2,
             classes=fl.fl.classes, segundos=dt)
    print("-> %s (%.1f MB)" % (destino, os.path.getsize(destino) / 1e6), flush=True)


# ------------------------------------------------------------ uso da tabela
class Tabela:
    """Lookup na GPU: 1 leitura por pixel."""

    def __init__(self, caminho, dev):
        d = np.load(caminho)
        self.classes = d["classes"]
        t = d["tabela"]
        self.n_pan, self.n1, self.n2 = t.shape
        self.tab = torch.as_tensor(t.reshape(-1).copy(), device=dev)
        self.b1 = torch.as_tensor(d["b1"].copy(), device=dev)
        self.b2 = torch.as_tensor(d["b2"].copy(), device=dev)
        self.bytes = self.tab.numel()
        self.dev = dev

    @torch.no_grad()
    def indices(self, X):
        a = X[:, 0].round().clamp(0, self.n_pan - 1).to(torch.int64)
        b = torch.bucketize(X[:, 1].contiguous(), self.b1)
        c = torch.bucketize(X[:, 2].contiguous(), self.b2)
        return self.tab[(a * self.n1 + b) * self.n2 + c]

    @torch.no_grad()
    def predizer(self, X):
        return self.classes[self.indices(X).cpu().numpy()]


# ------------------------------------------------------------ cena
def classificar_cena(tab, img, saida, bloco=2048):
    ds = gdal.Open(img)
    W, H = ds.RasterXSize, ds.RasterYSize
    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(saida, W, H, 1, gdal.GDT_Byte,
                     ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"])
    out.SetProjection(ds.GetProjection())
    out.SetGeoTransform(ds.GetGeoTransform())
    ob = out.GetRasterBand(1)
    ob.SetNoDataValue(NODATA_SAIDA)
    nd = [None] * ds.RasterCount
    t_io = t_gpu = 0.0
    px = 0
    t0 = time.monotonic()
    for y in range(0, H, bloco):
        ny = min(bloco, H - y)
        for x in range(0, W, bloco):
            nx = min(bloco, W - x)
            ta = time.monotonic()
            d, v = ler_bloco(ds, x, y, nx, ny, nd)
            v &= d[..., 0] > 0
            t_io += time.monotonic() - ta
            res = np.full((ny, nx), NODATA_SAIDA, np.uint8)
            if v.any():
                ta = time.monotonic()
                g = torch.as_tensor(np.ascontiguousarray(d[v]), device=tab.dev,
                                    dtype=torch.float32)
                res[v] = tab.predizer(g)
                torch.cuda.synchronize()
                t_gpu += time.monotonic() - ta
                px += int(v.sum())
            ob.WriteArray(res, x, y)
        print("  linha %d/%d | io %.0fs | gpu %.1fs" % (y + ny, H, t_io, t_gpu),
              flush=True)
    ob.FlushCache()
    out = None
    dt = time.monotonic() - t0
    print("cena: %d px validos em %.0fs (io %.0fs, gpu %.1fs) -> %.3f Mpx/s"
          % (px, dt, t_io, t_gpu, px / 1e6 / dt), flush=True)
    return {"pixels": px, "segundos": dt, "io": t_io, "gpu": t_gpu}


# ------------------------------------------------------------ verificação
def verificar(tab, pasta_fl, img, dev, x0, y0, lado):
    """Tabela x travessia da MESMA floresta, no mesmo tile."""
    ds = gdal.Open(img)
    d, v = ler_bloco(ds, x0, y0, lado, lado, [None] * ds.RasterCount)
    v &= d[..., 0] > 0
    X = np.ascontiguousarray(d[v])
    print("tile (%d,%d): %d px validos" % (x0, y0, len(X)), flush=True)
    g = torch.as_tensor(X, device=dev, dtype=torch.float32)

    tab.indices(g[:4096]); torch.cuda.synchronize()
    t0 = time.monotonic()
    pt = tab.predizer(g)
    torch.cuda.synchronize(); tt = time.monotonic() - t0

    fl = FlorestaTriton(FlorestaGPU(pasta_fl, dev))
    fl.indices(g[:4096]); torch.cuda.synchronize()
    t0 = time.monotonic()
    pf = fl.predizer(g)
    torch.cuda.synchronize(); tf = time.monotonic() - t0

    dif = int((pf != pt).sum())
    print("\nfloresta (travessia): %7.3fs -> %8.4f Mpx/s" % (tf, len(X) / 1e6 / tf))
    print("tabela   (lookup)   : %7.3fs -> %8.4f Mpx/s" % (tt, len(X) / 1e6 / tt))
    print("ganho da tabela sobre a travessia: %.1fx" % (tf / tt))
    print("divergencia: %d de %d px (%.4f%%)" % (dif, len(X), 100 * dif / len(X)))
    for c in np.unique(np.concatenate([pf, pt])):
        print("  classe %3d: floresta %8d px | tabela %8d px | %+d"
              % (c, (pf == c).sum(), (pt == c).sum(),
                 int((pt == c).sum()) - int((pf == c).sum())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--montar", action="store_true")
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--floresta")
    ap.add_argument("--tabela", required=True)
    ap.add_argument("--imagem")
    ap.add_argument("--saida")
    ap.add_argument("--bins", type=int, default=256, help="bins de GLCM e rugosidade")
    ap.add_argument("--bins-rug", type=int, default=0, help="0 = igual a --bins")
    ap.add_argument("--bloco", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=4_000_000)
    ap.add_argument("--tile", type=int, nargs=3, default=[12288, 4096, 2048],
                    metavar=("X", "Y", "LADO"))
    args = ap.parse_args()

    dev = ambiente.exigir_gpu()
    if args.montar:
        montar(args.floresta, args.tabela, args.bins,
               args.bins_rug or args.bins, dev, args.chunk)
        return
    tab = Tabela(args.tabela, dev)
    print("tabela %dx%dx%d (%.1f MB na VRAM)"
          % (tab.n_pan, tab.n1, tab.n2, tab.bytes / 1e6), flush=True)
    if args.verificar:
        verificar(tab, args.floresta, args.imagem, dev, *args.tile)
        return
    info = classificar_cena(tab, args.imagem, args.saida, args.bloco)
    with open(os.path.splitext(args.saida)[0] + "_info.json", "w",
              encoding="utf-8") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
