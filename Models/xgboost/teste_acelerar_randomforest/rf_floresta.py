#!/usr/bin/env python3
"""Random Forest por travessia das árvores, cena inteira. GPU se houver, CPU se não.

Mesmas features e hiperparâmetros do RF_macro.py (PAN, GLCM entropia,
rugosidade; 500 árvores, n_jobs=-1, blocos de 2048). `max_depth` é parâmetro de
`--treinar`: 0 é a profundidade livre do original, 12 é a variante medida aqui.

  CPU   rf.predict do sklearn, lido de `<modelo>/modelo.joblib`.
  GPU   kernel Triton sobre `<modelo>/floresta/` (ver rf_kernel.py).

As duas vias leem o mesmo modelo e produzem o mesmo mapa: conferido
100,0000% dos pixels na cena inteira.

PROFUNDIDADE
------------
A diferença de velocidade entre livre e 12 não vem da quantidade de
comparações, e sim de onde os nós ficam. Os primeiros 12 níveis das 500 árvores
somam ~2 MB, por onde todo pixel passa, e isso cabe no cache do processador. A
floresta livre tem 1,48 GB e cada travessia faz ~43.000 leituras espalhadas por
ela, nenhuma em cache. As leituras são sequencialmente obrigatórias: o endereço
do próximo nó só existe depois de ler o nó atual, então não há como adiantá-las.

  modelo               profundidade        nos    memoria   acuracia(*)
  livre                          75   37.111.334    1,48 GB      0,7836
  max_depth=12                   12    1.552.486      60 MB      0,8022

  (*) validacao de 3 M px, amostra de 3 classes deste projeto.

FATIAMENTO
----------
O predict é chamado em fatias (262.144 px na CPU, 1.048.576 na GPU). Entregar
os 4,19 M px de um bloco 2048x2048 de uma vez joga o sklearn em swap -- com o
MESMO modelo, fatiar deu 2,4x. No kernel Triton, a fatia grande com a floresta
de 1,48 GB causa acesso ilegal de memória.

TEMPOS -- colinas_do_sul (29068x32624 = 295.863.990 px validos)
---------------------------------------------------------------
  modelo         dispositivo    total      io   calculo   Mpx/s calculo
  livre          CPU                -       -   ~4300 s          0,0688  (*)
  livre          GPU                -       -         -          0,0795  (*)
  max_depth=12   CPU           1169 s   125 s    1035 s          0,2859
  max_depth=12   GPU            166 s   114 s      43 s          6,8550

  (*) medido num tile de 2048x2048 (4.192.723 px, 16 fatias, zero swap) e
      extrapolado. A cena inteira nao foi rodada por esse caminho; em GPU ela
      nao completa num notebook, porque apos ~160 s de kernel saturado a placa
      vai a 86 C, o clock cai de 1777 para 1537 MHz e o driver devolve
      `illegal memory access`.

Para cena grande (> ~4096x4096 px) o caminho da tabela é muito mais rápido:
ver `rf_tabela.py` (8 s de cálculo na mesma cena).

Uso:
  # treinar (grava modelo.joblib para a CPU e floresta/ para a GPU)
  python rf_floresta.py --treinar --modelo runs/rf_d12 \
      --imagem <cena>.tif --rotulos <rotulos>.tif --max-profundidade 12

  python rf_floresta.py --treinar --modelo runs/rf_livre \
      --imagem <cena>.tif --rotulos <rotulos>.tif --max-profundidade 0

  # classificar (GPU se houver)
  python rf_floresta.py --modelo runs/rf_d12 \
      --imagem <cena>.tif --saida classes.tif

  # forçar CPU, para comparar na mesma máquina
  python rf_floresta.py --modelo runs/rf_d12 \
      --imagem <cena>.tif --saida classes.tif --dispositivo cpu

  # teste rápido do código (a borda da cena costuma ser vazia, daí o --y0)
  python rf_floresta.py --modelo runs/rf_d12 --imagem <cena>.tif \
      --saida /tmp/t.tif --y0 12288 --limite-blocos 2

Dependências: GDAL, numpy, scikit-learn, joblib. Para GPU: torch + triton.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

gdal.UseExceptions()

NODATA_SAIDA = 0
BLOCO_SPLIT = 4096          # lado do bloco espacial que decide treino/validação
FATIA = {"cpu": 262144, "gpu": 1048576}


# ------------------------------------------------------------ leitura
def ler_bloco(ds, x, y, nx, ny):
    """Devolve (X com shape (ny,nx,bandas) float32, mascara de validos)."""
    d = ds.ReadAsArray(x, y, nx, ny)
    if d.ndim == 2:
        d = d[np.newaxis]
    d = d.astype(np.float32, copy=False)
    v = np.all(np.isfinite(d), axis=0) & (d[0] > 0)
    for i in range(ds.RasterCount):
        n = ds.GetRasterBand(i + 1).GetNoDataValue()
        if n is not None and not np.isnan(n):
            v &= d[i] != n
    return np.moveaxis(d, 0, -1), v


# ------------------------------------------------------------ treino
def treinar(modelo, img, rot, profundidade, arvores, amostras, semente, bloco):
    """Treina, grava modelo.joblib (CPU) e floresta/ (GPU), e mede a validação.

    Treino e validação são separados por BLOCO ESPACIAL, não por pixel
    sorteado: pixels vizinhos são quase idênticos, então sortear pixel a pixel
    põe o mesmo conteúdo nos dois lados e infla a acurácia.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    ds, rs = gdal.Open(img), gdal.Open(rot)
    if (ds.RasterXSize, ds.RasterYSize) != (rs.RasterXSize, rs.RasterYSize):
        raise SystemExit("imagem e rótulos têm grades diferentes")
    rng = np.random.default_rng(semente)
    W, H = ds.RasterXSize, ds.RasterYSize

    Xs, ys, sp = [], [], []
    for y in range(0, H, bloco):
        ny = min(bloco, H - y)
        for x in range(0, W, bloco):
            nx = min(bloco, W - x)
            r = rs.GetRasterBand(1).ReadAsArray(x, y, nx, ny)
            if not (r > 0).any():
                continue
            d, v = ler_bloco(ds, x, y, nx, ny)
            sel = (r > 0) & v
            if not sel.any():
                continue
            Xs.append(d[sel])
            ys.append(r[sel])
            h = abs(hash((x // BLOCO_SPLIT, y // BLOCO_SPLIT, semente))) % 100
            sp.append(np.full(int(sel.sum()), 0 if h < 70 else 1, np.uint8))
    if not ys:
        raise SystemExit("nenhum pixel rotulado sobre pixel válido")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    sp = np.concatenate(sp)
    del Xs, ys
    print("  %d px rotulados (%d treino, %d validacao)"
          % (len(y), int((sp == 0).sum()), int((sp == 1).sum())), flush=True)

    def cota(yy, maximo):
        """Índices com cota igual por classe, até `maximo` no total."""
        if not maximo or len(yy) <= maximo:
            return np.arange(len(yy))
        cls = np.unique(yy)
        por = maximo // len(cls)
        out = [rng.choice(np.where(yy == c)[0], min(por, int((yy == c).sum())),
                          replace=False) for c in cls]
        return np.sort(np.concatenate(out))

    tr = np.where(sp == 0)[0]
    tr = tr[cota(y[tr], amostras)]
    Xtr, ytr = X[tr].copy(), y[tr].copy()
    print("  treino: %d px, cota igual por classe" % len(ytr), flush=True)

    t0 = time.monotonic()
    rf = RandomForestClassifier(n_estimators=arvores,
                                max_depth=profundidade or None,
                                oob_score=False, random_state=semente,
                                n_jobs=-1)
    rf.fit(Xtr, ytr)
    dt = time.monotonic() - t0
    nos = sum(e.tree_.node_count for e in rf.estimators_)
    prof = max(e.tree_.max_depth for e in rf.estimators_)
    print("  %.0fs | %d nos | profundidade %d" % (dt, nos, prof), flush=True)
    del Xtr, ytr

    va = np.where(sp == 1)[0]
    if len(va) > 3_000_000:
        va = np.sort(rng.choice(va, 3_000_000, replace=False))
    acc = float((rf.predict(X[va]) == y[va]).mean()) if len(va) else None
    print("  acuracia na validacao: %.4f (%d px)" % (acc or 0, len(va)), flush=True)
    del X, y, sp

    os.makedirs(modelo, exist_ok=True)
    cam = os.path.join(modelo, "modelo.joblib")
    joblib.dump(rf, cam, compress=0)
    print("  -> %s (%.1f MB)" % (cam, os.path.getsize(cam) / 1e6), flush=True)

    with open(os.path.join(modelo, "info.json"), "w", encoding="utf-8") as f:
        json.dump({"arvores": arvores, "max_profundidade": profundidade or None,
                   "profundidade": prof, "nos": nos, "treino_px": int(len(tr)),
                   "segundos_treino": dt, "acuracia_validacao": acc,
                   "validacao_px": int(len(va)),
                   "classes": rf.classes_.tolist()}, f, indent=2)

    # achatar para a GPU. No MESMO processo: recarregar o pickle de uma
    # floresta livre faz pico de ~6,6 GB de RAM.
    try:
        from rf_kernel import achatar
        achatar(rf, os.path.join(modelo, "floresta"))
    except ImportError as e:
        print("  (sem torch/triton: floresta/ nao gerada, a GPU nao vai rodar"
              " -- %s)" % e, flush=True)


# ------------------------------------------------------------ preditores
def preditor(modelo, pedido):
    """Devolve (f(X)->uint8, dispositivo, descricao, nos, bytes)."""
    if pedido != "cpu":
        try:
            import torch
            from rf_kernel import FlorestaGPU, FlorestaTriton
            if not torch.cuda.is_available():
                raise RuntimeError("não há CUDA visível")
            print("  GPU: %s" % torch.cuda.get_device_name(0), flush=True)
            dev = torch.device("cuda")
            fl = FlorestaTriton(FlorestaGPU(os.path.join(modelo, "floresta"), dev))
            fl.indices(torch.zeros((4096, 3), device=dev))   # compila o kernel
            torch.cuda.synchronize()

            def f(X):
                g = torch.as_tensor(X, device=dev, dtype=torch.float32)
                r = fl.predizer(g)
                torch.cuda.synchronize()
                return r.astype(np.uint8)
            return (f, "gpu", "kernel Triton, 1 thread por pixel",
                    fl.fl.n_nos, fl.fl.bytes)
        except Exception as e:
            if pedido == "gpu":
                raise SystemExit("--dispositivo gpu pedido mas falhou: %s" % e)
            print("  sem GPU utilizável (%s) -> CPU" % e, flush=True)

    import joblib
    cam = os.path.join(modelo, "modelo.joblib")
    if not os.path.exists(cam):
        raise SystemExit("falta %s -- gere com --treinar" % cam)
    rf = joblib.load(cam)
    nos = sum(e.tree_.node_count for e in rf.estimators_)
    prof = max(e.tree_.max_depth for e in rf.estimators_)

    def f(X):
        return rf.predict(X).astype(np.uint8)
    return (f, "cpu", "rf.predict do sklearn (profundidade %d)" % prof,
            nos, os.path.getsize(cam))


# ------------------------------------------------------------ a cena
def classificar(f, img, saida, bloco, fatia, y0=0, limite_blocos=0):
    ds = gdal.Open(img)
    W, H = ds.RasterXSize, ds.RasterYSize
    os.makedirs(os.path.dirname(os.path.abspath(saida)) or ".", exist_ok=True)
    out = gdal.GetDriverByName("GTiff").Create(
        saida, W, H, 1, gdal.GDT_Byte,
        ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"])
    out.SetProjection(ds.GetProjection())
    out.SetGeoTransform(ds.GetGeoTransform())
    ob = out.GetRasterBand(1)
    ob.SetNoDataValue(NODATA_SAIDA)

    t_io = t_calc = 0.0
    px = nb = 0
    t0 = time.monotonic()
    for y in range(y0, H, bloco):
        ny = min(bloco, H - y)
        for x in range(0, W, bloco):
            nx = min(bloco, W - x)
            ta = time.monotonic()
            d, v = ler_bloco(ds, x, y, nx, ny)
            t_io += time.monotonic() - ta

            res = np.full((ny, nx), NODATA_SAIDA, np.uint8)
            if v.any():
                X = np.ascontiguousarray(d[v])
                ta = time.monotonic()
                r = np.empty(len(X), np.uint8)
                for i in range(0, len(X), fatia):
                    r[i:i + fatia] = f(X[i:i + fatia])
                res[v] = r
                t_calc += time.monotonic() - ta
                px += len(X)

            ta = time.monotonic()
            ob.WriteArray(res, x, y)
            t_io += time.monotonic() - ta
            nb += 1
            if limite_blocos and nb >= limite_blocos:
                break
        feito = min(1.0, (y + ny - y0) / max(H - y0, 1))
        print("  linha %6d/%d (%5.1f%%) | io %5.0fs | calculo %6.0fs | "
              "~%.0f min restantes"
              % (y + ny, H, 100 * feito, t_io, t_calc,
                 (time.monotonic() - t0) * (1 / max(feito, 1e-9) - 1) / 60),
              flush=True)
        if limite_blocos and nb >= limite_blocos:
            break
    ob.FlushCache()
    out = None
    return px, time.monotonic() - t0, t_io, t_calc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True,
                    help="pasta com modelo.joblib (CPU) e floresta/ (GPU)")
    ap.add_argument("--treinar", action="store_true")
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--rotulos", help="só com --treinar")
    ap.add_argument("--saida", help="GeoTIFF de saída (sem --treinar)")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--max-profundidade", type=int, default=12,
                    help="0 = livre, como no RF_macro.py original")
    ap.add_argument("--arvores", type=int, default=500)
    ap.add_argument("--max-amostras", type=int, default=300000, help="0 = todas")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--bloco", type=int, default=2048)
    ap.add_argument("--fatia", type=int, default=0, help="0 = padrão do dispositivo")
    ap.add_argument("--y0", type=int, default=0, help="linha inicial (só p/ teste)")
    ap.add_argument("--limite-blocos", type=int, default=0,
                    help="0 = cena inteira; >0 para testar o código")
    a = ap.parse_args()

    if a.treinar:
        if not a.rotulos:
            raise SystemExit("--treinar exige --rotulos")
        print("=== treino (max_depth=%s) ==="
              % (a.max_profundidade or "livre"), flush=True)
        treinar(a.modelo, a.imagem, a.rotulos, a.max_profundidade, a.arvores,
                a.max_amostras, a.semente, a.bloco)
        return

    if not a.saida:
        raise SystemExit("--saida é obrigatório sem --treinar")
    print("=== floresta: travessia das árvores ===", flush=True)
    f, disp, desc, nos, nbytes = preditor(a.modelo, a.dispositivo)
    fatia = a.fatia or FATIA[disp]
    print("  %s" % desc, flush=True)
    print("  modelo: %s" % a.modelo, flush=True)
    print("  %d nós | %.1f MB | fatias de %d px" % (nos, nbytes / 1e6, fatia),
          flush=True)

    px, dt, t_io, t_calc = classificar(f, a.imagem, a.saida, a.bloco, fatia,
                                       a.y0, a.limite_blocos)
    info = {"caminho": "floresta", "dispositivo": disp, "descricao": desc,
            "modelo": a.modelo, "nos": int(nos), "modelo_bytes": int(nbytes),
            "pixels": int(px), "segundos": dt, "io": t_io, "calculo": t_calc,
            "mpx_s_total": px / 1e6 / dt if dt else None,
            "mpx_s_calculo": px / 1e6 / t_calc if t_calc else None,
            "bloco": a.bloco, "fatia": fatia}
    cam = os.path.splitext(a.saida)[0] + ".json"
    with open(cam, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    print("\n  %d px em %.0fs (io %.0fs, calculo %.0fs)" % (px, dt, t_io, t_calc))
    print("  %.4f Mpx/s total | %.4f Mpx/s só cálculo"
          % (info["mpx_s_total"] or 0, info["mpx_s_calculo"] or 0))
    print("  -> %s\n  -> %s" % (a.saida, cam), flush=True)


if __name__ == "__main__":
    main()
