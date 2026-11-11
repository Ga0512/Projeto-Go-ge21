#!/usr/bin/env python3
"""Random Forest por tabela pré-calculada, cena inteira. GPU se houver, CPU se não.

Mesmo modelo de `rf_floresta.py`, mesma resposta -- mas sem percorrer árvore
nenhuma na inferência.

A IDEIA
-------
O modelo olha 3 números por pixel e nada mais: PAN, GLCM entropia, rugosidade.
A PAN é Byte, logo só pode ter 256 valores. Cada textura é cortada em 256
faixas, com as bordas nos quantis dos limiares que a própria floresta usa --
resolução fina onde ela decide, grossa onde ela não pergunta nada.

Isso fecha o espaço de entrada em 256 x 256 x 256 = 16.777.216 combinações
possíveis. `--montar` roda a floresta uma vez em cada combinação (a PAN
inteira, e o valor do centro de cada faixa de textura, em escala real) e grava
a resposta em um byte: 16,8 MB, 11 s na GPU, sem imagem aberta.

Na inferência nenhuma árvore é percorrida:

  PAN  142   -> a = 142         (é o próprio valor, é Byte)
  GLCM 0,83  -> b = 171         (em qual das 256 faixas cai)
  rug  1,21  -> c =  94

  indice = (142 * 256 + 171) * 256 + 94 = 9.343.838
  classe = tabela[9.343.838]

1 leitura por pixel, numa tabela de 16,8 MB que cabe em cache -- contra as
~43.000 leituras espalhadas por 1,48 GB que a travessia das 500 árvores exige,
e que são sequencialmente obrigatórias (o endereço do próximo nó só existe
depois de ler o nó atual).

POR QUE DÁ A MESMA RESPOSTA
---------------------------
A floresta é uma função de 3 variáveis e nada mais: sem contexto, sem vizinho,
sem posição. Dois pixels com os mesmos 3 números recebem obrigatoriamente a
mesma classe, então a resposta pode ser guardada e reaproveitada.

Na PAN é exato: os 471 limiares que a floresta usa nessa feature caem todos na
grade de 0,5 entre inteiros, e nenhum pixel pode cair entre dois inteiros. Nas
texturas é aproximado: elas têm ~1,1 e ~1,7 milhão de limiares distintos, e
valores na mesma faixa recebem a mesma resposta.

O CUSTO DA APROXIMAÇÃO, MEDIDO (--verificar)
--------------------------------------------
  pixels em que a tabela discorda da floresta livre:  3,5%
  nesses pixels, quem acertou:   tabela 48,3%  |  floresta 50,6%
  acuracia (validacao, 3 M px):  tabela 0,7843 |  floresta 0,7836

Os 3,5% caem em bordas de decisão, onde a própria floresta está indecisa.
Grades mais finas não melhoraram: 512 deu 0,7837 e 1024 deu 0,7833, contra
0,7843 do 256 -- 256 faixas já estão mais finas que a precisão real do modelo.

TEMPOS -- colinas_do_sul (29068x32624 = 295.863.990 px validos)
---------------------------------------------------------------
  dispositivo    total      io   calculo   Mpx/s calculo
  CPU (numpy)     143 s   109 s      26 s         11,5059
  GPU             131 s   113 s       8 s         36,8285

Para comparar, a mesma cena por `rf_floresta.py`: 1035 s de cálculo na CPU com
max_depth=12, 43 s na GPU, e ~4300 s na CPU com profundidade livre.

Note que o disco virou 86% do tempo (113 s de io contra 8 s de cálculo). A
partir daí otimizar o modelo não altera mais o resultado prático; o que resta é
leitura paralela, armazenamento mais rápido, ou dividir a cena entre processos
ou máquinas -- cada uma leva uma cópia de 16,8 MB e não precisa de GPU.

QUANDO NÃO USAR
---------------
O custo da tabela é FIXO (16,7 M travessias na montagem), não por pixel:

  8x8 = 64 px          floresta percorre          64  | tabela 16.777.216
  2048x2048 = 4,19 M   floresta percorre   4.194.304  | tabela 16.777.216
  ~4096x4096 = 16,7 M  EMPATE
  295,9 M (colinas)    floresta percorre 295.863.990  | tabela 16.777.216
  80 Gpx               floresta percorre      80 e+09 | tabela 16.777.216

Ponto de virada: ~4096x4096 px. Abaixo disso use `rf_floresta.py`.

E duas restrições estruturais: depende de haver poucas features e uma delas ser
Byte (com 4 features a tabela iria a 4,3 GB, com 5 a 1,1 TB), e é uma tabela
por modelo treinado.

Uso:
  # montar (uma vez, 11 s na GPU)
  python rf_tabela.py --montar --modelo runs/rf_livre --tabela tabela_256.npz

  # conferir contra a travessia da MESMA floresta, num tile
  python rf_tabela.py --verificar --modelo runs/rf_livre \
      --tabela tabela_256.npz --imagem <cena>.tif

  # classificar (GPU se houver)
  python rf_tabela.py --tabela tabela_256.npz \
      --imagem <cena>.tif --saida classes.tif

  # forçar CPU, para comparar na mesma máquina
  python rf_tabela.py --tabela tabela_256.npz \
      --imagem <cena>.tif --saida classes.tif --dispositivo cpu

Dependências: GDAL, numpy (+ scikit-learn e joblib para montar sem GPU).
Para GPU: torch + triton.
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
N_PAN = 256          # a PAN é Byte: 256 valores inteiros possíveis, e ponto
FATIA = {"cpu": 262144, "gpu": 1048576}


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


# ------------------------------------------------------------ bordas das faixas
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
    """n-1 bordas de faixa nos QUANTIS dos limiares, mais os representantes.

    Não são faixas de largura igual. A rugosidade, por exemplo, vai de -1,7 a
    +11 mas a floresta decide quase tudo perto de zero: faixas uniformes
    desperdiçariam metade da tabela na cauda. Nos quantis dos limiares, a faixa
    fica estreita onde ela decide muito e larga onde ela não pergunta nada --
    e faixa larga num trecho sem limiar não muda resposta nenhuma.

    O representante de uma faixa é o ponto médio dela; é nesse ponto que a
    floresta é avaliada para preencher a tabela. As faixas das pontas vão até
    o infinito, então nenhum valor pode cair fora.
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
def montar(modelo, destino, n1, n2, chunk=4_000_000):
    """Roda a floresta em TODAS as combinações possíveis e grava as respostas.

    Nenhuma imagem é aberta aqui. A grade regular é, por si, mais rápida de
    percorrer que pixels reais: vizinhos da grade caem no mesmo caminho da
    árvore, então a GPU lê o nó uma vez e serve 32 threads. Medido: 1,504
    Mentradas/s na grade contra 0,0795 Mpx/s nos pixels reais, 19x.
    """
    pasta = os.path.join(modelo, "floresta")
    if not os.path.exists(pasta):
        raise SystemExit("falta %s -- gere com rf_floresta.py --treinar" % pasta)

    b1, r1, u1 = bordas(pasta, 1, n1)
    b2, r2, u2 = bordas(pasta, 2, n2)
    n1, n2 = len(r1), len(r2)
    u0 = len(limiares(pasta, 0))
    print("PAN       : %7d limiares distintos -> %4d valores (EXATO, e Byte)"
          % (u0, N_PAN), flush=True)
    print("GLCM      : %7d limiares distintos -> %4d faixas" % (u1, n1), flush=True)
    print("rugosidade: %7d limiares distintos -> %4d faixas" % (u2, n2), flush=True)

    total = N_PAN * n1 * n2
    print("tabela: %d x %d x %d = %d entradas (%.1f MB)"
          % (N_PAN, n1, n2, total, total / 1e6), flush=True)

    tab = np.empty(total, np.uint8)
    t0 = time.monotonic()
    usou_gpu = False
    try:
        import torch
        from rf_kernel import FlorestaGPU, FlorestaTriton
        if not torch.cuda.is_available():
            raise RuntimeError("não há CUDA visível")
        dev = torch.device("cuda")
        fl = FlorestaTriton(FlorestaGPU(pasta, dev))
        print("GPU: %s | floresta %d arvores, profundidade %d, %d nos"
              % (torch.cuda.get_device_name(0), fl.fl.n_arvores,
                 fl.fl.profundidade, fl.fl.n_nos), flush=True)
        classes = fl.fl.classes
        v0 = torch.arange(N_PAN, dtype=torch.float32, device=dev)
        v1 = torch.as_tensor(r1, device=dev)
        v2 = torch.as_tensor(r2, device=dev)
        for i0 in range(0, total, chunk):
            i1 = min(i0 + chunk, total)
            idx = torch.arange(i0, i1, device=dev, dtype=torch.int64)
            X = torch.stack([v0[idx // (n1 * n2)],
                             v1[(idx // n2) % n1],
                             v2[idx % n2]], 1).contiguous()
            tab[i0:i1] = fl.indices(X).to(torch.uint8).cpu().numpy()
            print("  %6.2f%% | %.0fs" % (100 * i1 / total, time.monotonic() - t0),
                  flush=True)
        usou_gpu = True
    except Exception as e:
        print("sem GPU utilizavel (%s) -> montando na CPU com o sklearn" % e,
              flush=True)
        import joblib
        cam = os.path.join(modelo, "modelo.joblib")
        if not os.path.exists(cam):
            raise SystemExit("falta %s para montar sem GPU" % cam)
        rf = joblib.load(cam)
        classes = np.asarray(rf.classes_)
        cl_idx = {c: i for i, c in enumerate(classes)}
        for i0 in range(0, total, chunk):
            i1 = min(i0 + chunk, total)
            idx = np.arange(i0, i1, dtype=np.int64)
            X = np.stack([(idx // (n1 * n2)).astype(np.float32),
                          r1[(idx // n2) % n1],
                          r2[idx % n2]], 1)
            tab[i0:i1] = np.vectorize(cl_idx.get)(rf.predict(X)).astype(np.uint8)
            print("  %6.2f%% | %.0fs" % (100 * i1 / total, time.monotonic() - t0),
                  flush=True)

    dt = time.monotonic() - t0
    print("montada em %.0fs (%.3f Mentradas/s, %s)"
          % (dt, total / 1e6 / dt, "GPU" if usou_gpu else "CPU"), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(destino)) or ".", exist_ok=True)
    np.savez(destino, tabela=tab.reshape(N_PAN, n1, n2), b1=b1, b2=b2,
             classes=classes, segundos=dt)
    print("-> %s (%.1f MB)" % (destino, os.path.getsize(destino) / 1e6), flush=True)


# ------------------------------------------------------------ preditores
def preditor(cam, pedido):
    """Devolve (f(X)->uint8, dispositivo, descricao, bytes)."""
    d = np.load(cam)
    plano = np.ascontiguousarray(d["tabela"].reshape(-1))
    n_pan, n1, n2 = d["tabela"].shape
    b1, b2, classes = d["b1"], d["b2"], d["classes"]
    forma = "%dx%dx%d" % (n_pan, n1, n2)

    if pedido != "cpu":
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("não há CUDA visível")
            print("  GPU: %s" % torch.cuda.get_device_name(0), flush=True)
            dev = torch.device("cuda")
            t_tab = torch.as_tensor(plano, device=dev)
            t_b1 = torch.as_tensor(b1.copy(), device=dev)
            t_b2 = torch.as_tensor(b2.copy(), device=dev)
            t_cl = torch.as_tensor(classes.astype(np.uint8), device=dev)

            def f(X):
                g = torch.as_tensor(X, device=dev, dtype=torch.float32)
                a = g[:, 0].round().clamp(0, n_pan - 1).to(torch.int64)
                b = torch.bucketize(g[:, 1].contiguous(), t_b1)
                c = torch.bucketize(g[:, 2].contiguous(), t_b2)
                r = t_cl[t_tab[(a * n1 + b) * n2 + c].to(torch.int64)].cpu().numpy()
                torch.cuda.synchronize()
                return r
            return f, "gpu", "lookup na tabela %s (GPU)" % forma, plano.nbytes
        except Exception as e:
            if pedido == "gpu":
                raise SystemExit("--dispositivo gpu pedido mas falhou: %s" % e)
            print("  sem GPU utilizável (%s) -> CPU" % e, flush=True)

    def f(X):
        a = np.clip(np.rint(X[:, 0]), 0, n_pan - 1).astype(np.int64)
        b = np.searchsorted(b1, X[:, 1])
        c = np.searchsorted(b2, X[:, 2])
        return classes[plano[(a * n1 + b) * n2 + c]].astype(np.uint8)
    return f, "cpu", "lookup na tabela %s (numpy, 1 núcleo)" % forma, plano.nbytes


# ------------------------------------------------------------ verificação
def verificar(cam_tab, modelo, img, x0, y0, lado):
    """Tabela x travessia da MESMA floresta, no mesmo tile. Precisa de GPU."""
    import torch
    from rf_kernel import FlorestaGPU, FlorestaTriton
    dev = torch.device("cuda")
    ds = gdal.Open(img)
    d, v = ler_bloco(ds, x0, y0, lado, lado)
    X = np.ascontiguousarray(d[v])
    print("tile (%d,%d): %d px validos" % (x0, y0, len(X)), flush=True)
    g = torch.as_tensor(X, device=dev, dtype=torch.float32)

    ft, _, _, _ = preditor(cam_tab, "gpu")
    ft(X[:4096])
    t0 = time.monotonic()
    pt = ft(X)
    tt = time.monotonic() - t0

    fl = FlorestaTriton(FlorestaGPU(os.path.join(modelo, "floresta"), dev))
    fl.indices(g[:4096]); torch.cuda.synchronize()
    t0 = time.monotonic()
    pf = fl.predizer(g).astype(np.uint8)
    torch.cuda.synchronize()
    tf = time.monotonic() - t0

    dif = int((pf != pt).sum())
    print("\nfloresta (travessia): %7.3fs -> %8.4f Mpx/s" % (tf, len(X) / 1e6 / tf))
    print("tabela   (lookup)   : %7.3fs -> %8.4f Mpx/s" % (tt, len(X) / 1e6 / tt))
    print("ganho da tabela sobre a travessia: %.1fx" % (tf / tt))
    print("divergencia: %d de %d px (%.4f%%)" % (dif, len(X), 100 * dif / len(X)))
    for c in np.unique(np.concatenate([pf, pt])):
        print("  classe %3d: floresta %8d px | tabela %8d px | %+d"
              % (c, (pf == c).sum(), (pt == c).sum(),
                 int((pt == c).sum()) - int((pf == c).sum())))


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
    ap.add_argument("--tabela", required=True, help=".npz da tabela")
    ap.add_argument("--montar", action="store_true")
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--modelo", help="pasta do modelo (só com --montar/--verificar)")
    ap.add_argument("--imagem")
    ap.add_argument("--saida")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--bins", type=int, default=256, help="faixas de GLCM e rug")
    ap.add_argument("--bins-rug", type=int, default=0, help="0 = igual a --bins")
    ap.add_argument("--chunk", type=int, default=4_000_000)
    ap.add_argument("--tile", type=int, nargs=3, default=[12288, 4096, 2048],
                    metavar=("X", "Y", "LADO"))
    ap.add_argument("--bloco", type=int, default=2048)
    ap.add_argument("--fatia", type=int, default=0, help="0 = padrão do dispositivo")
    ap.add_argument("--y0", type=int, default=0, help="linha inicial (só p/ teste)")
    ap.add_argument("--limite-blocos", type=int, default=0,
                    help="0 = cena inteira; >0 para testar o código")
    a = ap.parse_args()

    if a.montar:
        if not a.modelo:
            raise SystemExit("--montar exige --modelo")
        print("=== montagem da tabela (nenhuma imagem aberta) ===", flush=True)
        montar(a.modelo, a.tabela, a.bins, a.bins_rug or a.bins, a.chunk)
        return

    if not os.path.exists(a.tabela):
        raise SystemExit("tabela não encontrada: %s\nMonte com --montar" % a.tabela)

    if a.verificar:
        if not (a.modelo and a.imagem):
            raise SystemExit("--verificar exige --modelo e --imagem")
        print("=== tabela x travessia da MESMA floresta ===", flush=True)
        verificar(a.tabela, a.modelo, a.imagem, *a.tile)
        return

    if not (a.imagem and a.saida):
        raise SystemExit("--imagem e --saida são obrigatórios")
    print("=== tabela: sem travessia de árvore ===", flush=True)
    f, disp, desc, nbytes = preditor(a.tabela, a.dispositivo)
    fatia = a.fatia or FATIA[disp]
    print("  %s" % desc, flush=True)
    print("  tabela: %s" % a.tabela, flush=True)
    print("  %.1f MB | fatias de %d px" % (nbytes / 1e6, fatia), flush=True)

    px, dt, t_io, t_calc = classificar(f, a.imagem, a.saida, a.bloco, fatia,
                                       a.y0, a.limite_blocos)
    info = {"caminho": "tabela", "dispositivo": disp, "descricao": desc,
            "tabela": a.tabela, "modelo_bytes": int(nbytes), "pixels": int(px),
            "segundos": dt, "io": t_io, "calculo": t_calc,
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
