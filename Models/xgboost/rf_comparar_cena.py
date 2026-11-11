#!/usr/bin/env python3
"""Classifica a cena INTEIRA por um método e cronometra. Um método por vez.

Os seis caminhos testados para o Random Forest do RF_macro.py:

  cpu_livre    rf.predict do sklearn, floresta do script dele (profundidade livre)
  gpu_livre    kernel Triton percorrendo a MESMA floresta na GPU
  cpu_d12      rf.predict do sklearn, floresta com max_depth=12
  gpu_d12      kernel Triton na floresta max_depth=12
  cpu_tabela   tabela de respostas pré-calculada, lookup em numpy (1 núcleo)
  gpu_tabela   a mesma tabela, lookup na GPU

Cada método roda em seu PRÓPRIO processo: a floresta profunda do sklearn ocupa
~3,3 GB, e medir CPU com um modelo da GPU carregado (ou vice-versa) falseia o
cronômetro. O `predict` da CPU é fatiado em 262.144 px porque entregar os 4,2 M
de um bloco inteiro de uma vez joga o processo em swap -- e aí o que se mede é
a velocidade do disco, não a do modelo.

O tempo é separado em `io` (ler o bloco do disco + gravar a saída) e `calculo`,
porque a partir de certo ponto o gargalo deixa de ser o modelo.

Uso (no WSL, a partir de `Models/`):
  for M in cpu_livre gpu_livre cpu_d12 gpu_d12 cpu_tabela gpu_tabela; do
    ../Data/.venv/bin/python xgboost/rf_comparar_cena.py --metodo $M
  done
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

gdal.UseExceptions()

RAIZ = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(RAIZ, "..", "..", "Data", "dados", "spot5", "colinas_do_sul",
                   "original", "colinas_3b.vrt")
ROT = os.path.join(RAIZ, "..", "..", "Data", "dados", "spot5", "colinas_do_sul",
                   "mascara_amostra_v1.tif")
SAIDA = os.path.join(RAIZ, "runs", "comparacao_cena")
NODATA_SAIDA = 0
# Fatiar o bloco antes de predizer não é cosmético: 4,19 M linhas de uma vez
# jogam o sklearn em swap, e no kernel Triton levam a um acesso ilegal de
# memória com a floresta de 1,48 GB. Estes são os tamanhos que rodam limpo.
FATIA = {"cpu": 262144, "gpu": 1048576}


def memoria():
    d = {}
    with open("/proc/self/status", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith(("VmRSS:", "VmSwap:")):
                d[ln.split(":")[0]] = int(ln.split()[1]) / 1e6
    return d.get("VmRSS", 0), d.get("VmSwap", 0)


def treinar(profundidade, semente=42, amostras=300000):
    """Mesmos hiperparâmetros do RF_macro.py; profundidade 0 = livre."""
    from rf_treino import coletar, cota
    from sklearn.ensemble import RandomForestClassifier
    t0 = time.monotonic()
    X, y, sp, rng = coletar(IMG, ROT, 2048, semente)
    tr = np.where(sp == 0)[0]
    tr = tr[cota(y[tr], amostras, rng)]
    Xtr, ytr = X[tr].copy(), y[tr].copy()
    del X, y, sp, tr
    gc.collect()
    rf = RandomForestClassifier(n_estimators=500, max_depth=profundidade or None,
                                oob_score=False, random_state=semente, n_jobs=-1)
    rf.fit(Xtr, ytr)
    del Xtr, ytr
    gc.collect()
    nos = sum(e.tree_.node_count for e in rf.estimators_)
    print("  treino: %.0fs | %d nos | profundidade %d"
          % (time.monotonic() - t0, nos, max(e.tree_.max_depth for e in rf.estimators_)),
          flush=True)
    return rf, nos


# --------------------------------------------------- os seis preditores
def preditor(metodo):
    """Devolve (funcao(X)->classes uint8, descricao, nos, bytes_modelo)."""
    runs = os.path.join(RAIZ, "runs")

    if metodo.startswith("cpu_") and metodo != "cpu_tabela":
        rf, nos = treinar(0 if metodo == "cpu_livre" else 12)

        def f(X):
            return rf.predict(X).astype(np.uint8)
        return f, "sklearn rf.predict (CPU)", nos, 0

    if metodo.startswith("gpu_") and metodo != "gpu_tabela":
        import ambiente
        import torch
        from rf_gpu import FlorestaGPU
        from rf_kernel import FlorestaTriton
        dev = ambiente.exigir_gpu()
        pasta = os.path.join(runs, "rf_colinas_livre" if metodo == "gpu_livre"
                             else "rf_colinas_d12", "floresta")
        fl = FlorestaTriton(FlorestaGPU(pasta, dev))
        fl.indices(torch.zeros((4096, 3), device=dev)); torch.cuda.synchronize()

        def f(X):
            g = torch.as_tensor(X, device=dev, dtype=torch.float32)
            r = fl.predizer(g)
            torch.cuda.synchronize()
            return r.astype(np.uint8)
        return f, "kernel Triton na GPU", fl.fl.n_nos, fl.fl.bytes

    cam = os.path.join(runs, "rf_colinas_livre", "tabela_256.npz")
    d = np.load(cam)
    tab = d["tabela"]
    n1, n2 = tab.shape[1], tab.shape[2]
    plano = np.ascontiguousarray(tab.reshape(-1))
    b1, b2, classes = d["b1"], d["b2"], d["classes"]

    if metodo == "cpu_tabela":
        def f(X):
            a = np.clip(np.rint(X[:, 0]), 0, 255).astype(np.int64)
            b = np.searchsorted(b1, X[:, 1])
            c = np.searchsorted(b2, X[:, 2])
            return classes[plano[(a * n1 + b) * n2 + c]].astype(np.uint8)
        return f, "lookup na tabela (numpy, 1 nucleo)", 0, plano.nbytes

    import ambiente
    import torch
    dev = ambiente.exigir_gpu()
    t_tab = torch.as_tensor(plano, device=dev)
    t_b1 = torch.as_tensor(b1, device=dev)
    t_b2 = torch.as_tensor(b2, device=dev)
    t_cl = torch.as_tensor(classes.astype(np.uint8), device=dev)

    def f(X):
        g = torch.as_tensor(X, device=dev, dtype=torch.float32)
        a = g[:, 0].round().clamp(0, 255).to(torch.int64)
        b = torch.bucketize(g[:, 1].contiguous(), t_b1)
        c = torch.bucketize(g[:, 2].contiguous(), t_b2)
        r = t_cl[t_tab[(a * n1 + b) * n2 + c].to(torch.int64)].cpu().numpy()
        torch.cuda.synchronize()
        return r
    return f, "lookup na tabela (GPU)", 0, plano.nbytes


# --------------------------------------------------- varredura da cena
def rodar(metodo, bloco, limite_blocos=0, y0=0, fatia=0):
    os.makedirs(SAIDA, exist_ok=True)
    tif = os.path.join(SAIDA, metodo + ".tif")
    print("=== %s ===" % metodo, flush=True)
    f, desc, nos, nbytes = preditor(metodo)
    fatia = fatia or FATIA[metodo.split("_")[0]]
    print("  fatias de %d px" % fatia, flush=True)
    print("  %s | %s nos | modelo %.1f MB"
          % (desc, "%d" % nos if nos else "-", nbytes / 1e6), flush=True)

    ds = gdal.Open(IMG)
    W, H = ds.RasterXSize, ds.RasterYSize
    nd = [ds.GetRasterBand(i + 1).GetNoDataValue() for i in range(ds.RasterCount)]
    out = gdal.GetDriverByName("GTiff").Create(
        tif, W, H, 1, gdal.GDT_Byte,
        ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"])
    out.SetProjection(ds.GetProjection())
    out.SetGeoTransform(ds.GetGeoTransform())
    ob = out.GetRasterBand(1)
    ob.SetNoDataValue(NODATA_SAIDA)

    t_io = t_calc = 0.0
    px = nb = 0
    pior_swap = 0.0
    t0 = time.monotonic()
    for y in range(y0, H, bloco):
        ny = min(bloco, H - y)
        for x in range(0, W, bloco):
            nx = min(bloco, W - x)
            ta = time.monotonic()
            d = ds.ReadAsArray(x, y, nx, ny).astype(np.float32, copy=False)
            v = np.all(np.isfinite(d), axis=0) & (d[0] > 0)
            for i, n in enumerate(nd):
                if n is not None and not np.isnan(n):
                    v &= d[i] != n
            t_io += time.monotonic() - ta
            res = np.full((ny, nx), NODATA_SAIDA, np.uint8)
            if v.any():
                X = np.ascontiguousarray(np.moveaxis(d, 0, -1)[v])
                ta = time.monotonic()
                saida = np.empty(len(X), np.uint8)
                for i in range(0, len(X), fatia):
                    saida[i:i + fatia] = f(X[i:i + fatia])
                res[v] = saida
                t_calc += time.monotonic() - ta
                px += len(X)
            ta = time.monotonic()
            ob.WriteArray(res, x, y)
            t_io += time.monotonic() - ta
            nb += 1
            if limite_blocos and nb >= limite_blocos:
                break
        pior_swap = max(pior_swap, memoria()[1])
        feito = min(1.0, (y + ny) / H)
        print("  linha %6d/%d (%5.1f%%) | io %5.0fs | calculo %6.0fs | "
              "~%.0f min restantes | swap %.2f GB"
              % (y + ny, H, 100 * feito, t_io, t_calc,
                 (time.monotonic() - t0) * (1 / feito - 1) / 60, pior_swap), flush=True)
        if limite_blocos and nb >= limite_blocos:
            break
    ob.FlushCache()
    out = None
    dt = time.monotonic() - t0

    info = {"metodo": metodo, "descricao": desc, "nos": int(nos),
            "modelo_bytes": int(nbytes), "pixels": int(px), "segundos": dt,
            "io": t_io, "calculo": t_calc, "mpx_s_total": px / 1e6 / dt,
            "mpx_s_calculo": px / 1e6 / t_calc if t_calc else None,
            "pior_swap_gb": pior_swap, "bloco": bloco, "fatia": fatia}
    with open(os.path.join(SAIDA, metodo + ".json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    print("  %d px em %.0fs (io %.0fs, calculo %.0fs) -> %.4f Mpx/s total, "
          "%.4f Mpx/s so calculo%s"
          % (px, dt, t_io, t_calc, info["mpx_s_total"], info["mpx_s_calculo"] or 0,
             "  <-- HOUVE SWAP, medida suspeita" if pior_swap > 0.05 else ""),
          flush=True)
    print("  -> %s" % tif, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metodo", required=True,
                    choices=["cpu_livre", "gpu_livre", "cpu_d12", "gpu_d12",
                             "cpu_tabela", "gpu_tabela"])
    ap.add_argument("--bloco", type=int, default=2048)
    ap.add_argument("--limite-blocos", type=int, default=0,
                    help="para teste rápido do código; 0 = cena inteira")
    ap.add_argument("--fatia", type=int, default=0,
                    help="px por chamada do preditor; 0 = padrão do método")
    ap.add_argument("--y0", type=int, default=0,
                    help="linha inicial; só para teste (a borda da cena é vazia)")
    a = ap.parse_args()
    rodar(a.metodo, a.bloco, a.limite_blocos, a.y0, a.fatia)


if __name__ == "__main__":
    main()
