#!/usr/bin/env python3
"""CPU x GPU com O MESMO Random Forest, nos MESMOS tiles da imagem.

Treina o RF com os hiperparâmetros do RF_macro.py (500 árvores, profundidade
livre), recorta alguns tiles da cena e classifica cada um duas vezes: pelo
`rf.predict` do sklearn na CPU e pelo kernel Triton na GPU. Compara os tempos
e confere pixel a pixel que as duas saídas são iguais.

A ordem importa por causa de memória: a CPU roda primeiro, com o modelo do
sklearn em RAM; só depois a floresta é achatada para disco (o que esvazia o
modelo) e carregada na GPU. Os dois juntos não cabem nos 6 GB do WSL.

Uso (no WSL, a partir de `Models/`):
  ../Data/.venv/bin/python xgboost/rf_duelo.py \\
      --imagem ../Data/dados/spot5/colinas_do_sul/original/colinas_3b.vrt \\
      --rotulos ../Data/dados/spot5/colinas_do_sul/mascara_amostra_v1.tif
"""

import argparse
import gc
import os
import shutil
import sys
import time

import ambiente
import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rf_treino import coletar, cota, ler_bloco  # noqa: E402

gdal.UseExceptions()


def memoria():
    """RSS e swap deste processo. Se houver swap, a cronometragem não vale:
    o que se mede passa a ser a velocidade do disco, não a do modelo."""
    d = {}
    with open("/proc/self/status", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith(("VmRSS:", "VmSwap:")):
                d[ln.split(":")[0]] = int(ln.split()[1]) / 1e6
    aviso = "  <-- EM SWAP, medida invalida" if d.get("VmSwap", 0) > 0.05 else ""
    return "RAM %.2f GB | swap %.2f GB%s" % (d.get("VmRSS", 0), d.get("VmSwap", 0), aviso)


def escolher_tiles(img, lado, quantos, amostra=64):
    """Tiles cheios de dado válido, espalhados pela cena.

    A validade NÃO pode sair de GetMaskBand(): a banda PAN é Byte e não declara
    nodata, então o GDAL devolve "tudo válido" e a borda preta da cena passa.
    Quem sabe onde há dado são as bandas de textura, com nodata -9999.
    """
    ds = gdal.Open(img)
    W, H = ds.RasterXSize, ds.RasterYSize
    b1, b2 = ds.GetRasterBand(1), ds.GetRasterBand(2)
    cand = []
    for y in range(0, H - lado, lado):
        for x in range(0, W - lado, lado):
            p = b1.ReadAsArray(x, y, lado, lado, buf_xsize=amostra, buf_ysize=amostra)
            t = b2.ReadAsArray(x, y, lado, lado, buf_xsize=amostra, buf_ysize=amostra)
            f = float(((p > 0) & (t != -9999) & np.isfinite(t)).mean())
            if f > 0.98:
                cand.append((x, y, f))
    if not cand:
        raise SystemExit("nenhum tile cheio encontrado")
    # espalha: pega de partes diferentes da lista, que está em ordem de varredura
    passo = max(1, len(cand) // quantos)
    return [cand[i * passo] for i in range(quantos)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--rotulos", required=True)
    ap.add_argument("--arvores", type=int, default=500)
    ap.add_argument("--max-profundidade", type=int, default=0, help="0 = livre")
    ap.add_argument("--max-amostras", type=int, default=300000)
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--lado", type=int, default=2048)
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier

    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "_duelo")
    os.makedirs(tmp, exist_ok=True)

    print("=== 1. treino (mesmos hiperparametros do RF_macro.py) ===", flush=True)
    X, y, sp, rng = coletar(args.imagem, args.rotulos, 2048, args.semente)
    tr = np.where(sp == 0)[0]
    tr = tr[cota(y[tr], args.max_amostras, rng)]
    # Só os pixels sorteados seguem adiante. Manter os 34 milhões coletados
    # junto com um modelo de 3 GB estoura os 6 GB do WSL, e o que vai para o
    # swap não volta sozinho: a medição da CPU viraria medição de disco.
    Xtr, ytr, n_tr = X[tr].copy(), y[tr].copy(), len(tr)
    del X, y, sp, tr
    gc.collect()

    rf = RandomForestClassifier(n_estimators=args.arvores,
                                max_depth=args.max_profundidade or None,
                                oob_score=False, random_state=args.semente, n_jobs=-1)
    t0 = time.monotonic()
    rf.fit(Xtr, ytr)
    print("  %d arvores, profundidade %s | %d px de treino | %.0fs"
          % (args.arvores, args.max_profundidade or "livre", n_tr,
             time.monotonic() - t0), flush=True)
    nos = sum(e.tree_.node_count for e in rf.estimators_)
    prof = max(e.tree_.max_depth for e in rf.estimators_)
    print("  floresta: %d nos | profundidade %d | ~%.2f GB"
          % (nos, prof, nos * 80 / 1e9), flush=True)
    del Xtr, ytr
    gc.collect()

    print("\n=== 2. escolhendo %d tiles de %dx%d ===" % (args.tiles, args.lado, args.lado),
          flush=True)
    tiles = escolher_tiles(args.imagem, args.lado, args.tiles)
    ds = gdal.Open(args.imagem)
    dados = []
    for x, y_, f in tiles:
        d, v = ler_bloco(ds, x, y_, args.lado, args.lado, None)
        v &= d[..., 0] > 0          # PAN 0 é nodata nesta cena, sem declaração
        if not v.any():
            raise SystemExit("tile em (%d,%d) sem pixel valido" % (x, y_))
        dados.append((x, y_, d[v], v.shape))
        print("  tile em (%d,%d): %d px validos (%.1f%% da amostragem)"
              % (x, y_, len(d[v]), 100 * f), flush=True)

    print("\n=== 3. CPU: rf.predict do sklearn ===", flush=True)
    print("  %s" % memoria(), flush=True)
    cpu = []
    for x, y_, Xt, _ in dados:
        t0 = time.monotonic()
        p = rf.predict(Xt)
        dt = time.monotonic() - t0
        cpu.append((p, dt))
        print("  tile (%d,%d): %7d px | %8.2fs | %.4f Mpx/s"
              % (x, y_, len(Xt), dt, len(Xt) / 1e6 / dt), flush=True)

    print("\n=== 4. achatando a floresta e liberando a RAM ===", flush=True)
    from rf_gpu import achatar
    pasta = os.path.join(tmp, "floresta")
    t0 = time.monotonic()
    achatar(rf, pasta)
    del rf
    gc.collect()
    print("  %.0fs" % (time.monotonic() - t0), flush=True)

    print("\n=== 5. GPU: kernel triton, MESMA floresta ===", flush=True)
    import torch
    from rf_gpu import FlorestaGPU
    from rf_kernel import FlorestaTriton
    dev = ambiente.exigir_gpu()
    t0 = time.monotonic()
    fl = FlorestaTriton(FlorestaGPU(pasta, dev))
    print("  floresta carregada na VRAM | %.0fs" % (time.monotonic() - t0), flush=True)
    # aquece o JIT fora da cronometragem
    fl.predizer(torch.as_tensor(dados[0][2][:4096], device=dev, dtype=torch.float32))
    torch.cuda.synchronize()

    gpu = []
    for (x, y_, Xt, _), (pc, _) in zip(dados, cpu):
        g = torch.as_tensor(Xt, device=dev, dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        p = fl.predizer(g)
        torch.cuda.synchronize()
        dt = time.monotonic() - t0
        dif = int((p != pc).sum())
        gpu.append((dt, dif))
        print("  tile (%d,%d): %7d px | %8.2fs | %.4f Mpx/s | divergem %d (%.6f%%)"
              % (x, y_, len(Xt), dt, len(Xt) / 1e6 / dt, dif, 100 * dif / len(Xt)),
              flush=True)

    print("\n=== RESULTADO ===")
    print("floresta: %d nos, profundidade %d (modelo do RF_macro.py, intocado)"
          % (nos, prof))
    print("%-18s %10s %12s %12s %9s %10s"
          % ("tile", "pixels", "CPU (s)", "GPU (s)", "ganho", "divergem"))
    tc = tg = tp = td = 0
    for (x, y_, Xt, _), (pc, dc), (dg, dif) in zip(dados, cpu, gpu):
        print("%-18s %10d %12.2f %12.2f %8.2fx %10d"
              % ("(%d,%d)" % (x, y_), len(Xt), dc, dg, dc / dg, dif))
        tc += dc
        tg += dg
        tp += len(Xt)
        td += dif
    print("%-18s %10d %12.2f %12.2f %8.2fx %10d" % ("TOTAL", tp, tc, tg, tc / tg, td))
    print("\nCPU %.4f Mpx/s | GPU %.4f Mpx/s | divergencia %.6f%%"
          % (tp / 1e6 / tc, tp / 1e6 / tg, 100 * td / tp))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
