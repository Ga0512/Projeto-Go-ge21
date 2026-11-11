#!/usr/bin/env python3
"""Inferência de Random Forest do scikit-learn na GPU, com as MESMAS árvores.

O gargalo do RF_macro.py não é o Python: `rf.predict` já recebe o bloco inteiro
de uma vez e `n_jobs=-1` já divide as árvores entre os núcleos. O que custa é
percorrer 500 árvores nó a nó para cada pixel, um padrão de acesso à memória
que a CPU não consegue prever, somado ao float64 que o `predict_proba` aloca
para acumular as árvores.

Aqui a floresta treinada pelo sklearn é achatada em arrays contíguos (todas as
árvores concatenadas, com um deslocamento por árvore) e a travessia vira um
laço sobre a PROFUNDIDADE, não sobre as árvores: a cada passo, todos os pixels
de todas as árvores avançam um nível de uma vez, via `gather`. Como os
limiares e a regra de decisão (`x <= limiar` vai para a esquerda) são os mesmos
do sklearn, a predição é a mesma árvore por árvore.

A única diferença numérica possível está na soma das probabilidades das 500
árvores, feita aqui em float32 e no sklearn em float64. Isso só muda o
resultado quando duas classes empatam em ~1e-7, e `verificar()` mede
exatamente em quantos pixels isso acontece.

Uso (no WSL, a partir de `Models/`):
  ../Data/.venv/bin/python xgboost/rf_gpu.py --modelo runs/rf_x/modelo.joblib \\
      --imagem ../Data/dados/spot5/colinas_do_sul/original/colinas_3b.vrt \\
      --saida runs/rf_x/classes_gpu.tif
"""

import argparse
import json
import os
import time

import ambiente  # noqa: F401  (antes de torch; redireciona caches p/ Models/.cache)
import numpy as np
import torch
from osgeo import gdal

gdal.UseExceptions()

NODATA_ENTRADA = -9999
NODATA_SAIDA = 0


# ------------------------------------------------------------ floresta
class FlorestaGPU:
    """Floresta do sklearn achatada em tensores na GPU.

    Todas as árvores viram um único vetor de nós; `base[t]` é onde a árvore t
    começa. Um nó folha tem `esquerda == direita == ele mesmo`, de modo que
    quem chega nela para de andar sem precisar de máscara extra.
    """

    def __init__(self, pasta, dev, lote_arvores=0):
        with open(os.path.join(pasta, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.classes = np.asarray(meta["classes"])
        self.n_classes = int(meta["n_classes"])
        self.n_arvores = int(meta["n_arvores"])
        self.profundidade = int(meta["profundidade"])

        def sobe(nome):
            """memmap -> GPU em fatias, para não materializar o array em RAM."""
            a = np.load(os.path.join(pasta, nome + ".npy"), mmap_mode="r")
            t = torch.empty(a.shape, dtype=getattr(torch, str(a.dtype)), device=dev)
            passo = max(1, 8_000_000 // max(1, int(np.prod(a.shape[1:]))))
            for i in range(0, a.shape[0], passo):
                t[i:i + passo] = torch.from_numpy(np.ascontiguousarray(a[i:i + passo]))
            return t

        # Os quatro campos do nó viram UMA tabela (n_nos, 4) int32. Assim cada
        # passo da travessia faz um gather só, de 16 bytes contíguos, em vez de
        # quatro gathers espalhados: menos lançamentos de kernel e leitura
        # coalescida. O limiar vai como float32 reinterpretado em int32.
        self.base = sobe("base")
        self.feat = sobe("feat")
        self.lim = sobe("lim")
        self.esq = sobe("esq")
        self.dir = sobe("dir")
        self.val = sobe("val")
        self.dev = dev
        self.n_nos = int(self.feat.numel())
        self.lote_arvores = lote_arvores or self.n_arvores
        self.bytes = sum(t.numel() * t.element_size() for t in
                         (self.feat, self.lim, self.esq, self.dir, self.val))

    @torch.no_grad()
    def prob(self, X):
        """X: tensor (N, n_features) float32 na GPU -> (N, n_classes) float32."""
        N = X.shape[0]
        soma = torch.zeros((N, self.n_classes), dtype=torch.float32, device=self.dev)
        for i0 in range(0, self.n_arvores, self.lote_arvores):
            b = self.base[i0:i0 + self.lote_arvores]
            no = b.unsqueeze(0).expand(N, -1).contiguous()   # (N, T) int64
            for passo in range(self.profundidade + 1):
                x = torch.gather(X, 1, self.feat[no])
                direita = x > self.lim[no]
                prox = torch.where(direita, self.dir[no], self.esq[no])
                # a folha aponta para si mesma, então prox == no significa que
                # todo mundo chegou. A folha média está no nível ~24 de 76: sem
                # sair cedo, dois terços dos passos seriam desperdício. Mas a
                # comparação sincroniza com a GPU, então só do nível 12 em diante
                # e a cada 4 passos.
                se_parou = passo >= 12 and passo % 4 == 3 and torch.equal(prox, no)
                no = prox
                if se_parou:
                    break
            soma += self.val[no].sum(1)
        return soma / self.n_arvores

    @torch.no_grad()
    def predizer(self, X):
        """Devolve os rótulos originais do sklearn (rf.classes_)."""
        idx = self.prob(X).argmax(1).cpu().numpy()
        return self.classes[idx]


def achatar(rf, destino):
    """Grava a floresta do sklearn como arrays achatados (.npy) numa pasta.

    Chame isto no MESMO processo que treinou: o pickle de um RF de 500
    árvores com profundidade livre passa de 3 GB, e recarregá-lo faz pico de
    ~6,6 GB (buffer + array), o que derruba a VM do WSL nesta máquina de
    15 GB. Aqui o modelo já está em RAM e os arrays vão direto para disco por
    memmap, com cada árvore descartada assim que copiada.
    """
    from numpy.lib.format import open_memmap
    os.makedirs(destino, exist_ok=True)
    classes = np.asarray(rf.classes_)
    nc = len(classes)
    n = np.array([e.tree_.node_count for e in rf.estimators_], np.int64)
    base = np.concatenate([[0], np.cumsum(n)[:-1]])
    total = int(n.sum())
    prof = int(max(e.tree_.max_depth for e in rf.estimators_))
    print("  %d arvores | %d nos | profundidade %d | %d classes"
          % (len(n), total, prof, nc), flush=True)

    def mm(nome, forma, tipo):
        return open_memmap(os.path.join(destino, nome + ".npy"), mode="w+",
                           dtype=tipo, shape=forma)

    feat = mm("feat", (total,), np.int64)
    lim = mm("lim", (total,), np.float32)
    esq = mm("esq", (total,), np.int64)
    dir_ = mm("dir", (total,), np.int64)
    val = mm("val", (total, nc), np.float32)

    for t in range(len(n)):
        a = rf.estimators_[t].tree_
        b, m = int(base[t]), int(n[t])
        ce = a.children_left.astype(np.int64)
        cd = a.children_right.astype(np.int64)
        folha = ce < 0
        idx = np.arange(m)
        # a folha aponta para si mesma: quem chega nela para de andar
        esq[b:b + m] = np.where(folha, idx, ce) + b
        dir_[b:b + m] = np.where(folha, idx, cd) + b
        feat[b:b + m] = np.where(folha, 0, a.feature.astype(np.int64))
        lim[b:b + m] = np.where(folha, np.inf, a.threshold).astype(np.float32)
        # sklearn guarda contagens por folha; predict_proba normaliza por árvore
        v = a.value.reshape(m, -1).astype(np.float64)
        val[b:b + m] = (v / np.maximum(v.sum(1, keepdims=True), 1e-12)).astype(np.float32)
        rf.estimators_[t] = None          # libera a árvore já copiada
        if (t + 1) % 100 == 0:
            print("    %d/%d arvores" % (t + 1, len(n)), flush=True)
    del rf
    for a in (feat, lim, esq, dir_, val):
        a.flush()
    np.save(os.path.join(destino, "base.npy"), base)
    with open(os.path.join(destino, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes.tolist(), "n_arvores": int(len(n)),
                   "profundidade": prof, "n_nos": total, "n_classes": nc}, f)
    tam = sum(os.path.getsize(os.path.join(destino, a))
              for a in os.listdir(destino))
    print("  -> %s (%.2f GB)" % (destino, tam / 1e9), flush=True)


# ------------------------------------------------------------ verificação
def verificar(floresta, pasta, dev, lote=100000):
    """Compara a GPU contra as predições que o sklearn gravou no treino.

    `rf_treino.py` guarda em `verificacao.npz` uma amostra de pixels e o que
    o `rf.predict` da CPU respondeu para cada um. Aqui só reexecutamos na GPU
    e contamos as divergências, sem precisar do modelo do sklearn em RAM.
    """
    d = np.load(os.path.join(pasta, "verificacao.npz"))
    X, esperado = d["X"], d["pred"]
    dif = 0
    for i in range(0, len(X), lote):
        b = floresta.predizer(
            torch.as_tensor(X[i:i + lote], device=dev, dtype=torch.float32))
        dif += int((b != esperado[i:i + lote]).sum())
    return dif, len(X)


# ------------------------------------------------------------ cena
def janelas(largura, altura, bloco):
    for y in range(0, altura, bloco):
        ny = min(bloco, altura - y)
        for x in range(0, largura, bloco):
            yield x, y, min(bloco, largura - x), ny


def ler_bloco(ds, x, y, nx, ny, mascaras):
    """Igual ao ler_bloco do RF_macro.py, mas lendo a máscara uma vez só."""
    dados = ds.ReadAsArray(x, y, nx, ny)
    if dados.ndim == 2:
        dados = dados[np.newaxis]
    dados = dados.astype(np.float32, copy=False)
    validos = np.all(np.isfinite(dados), axis=0)
    validos &= np.all(dados != NODATA_ENTRADA, axis=0)
    for i, m in enumerate(mascaras):
        nd = ds.GetRasterBand(i + 1).GetNoDataValue()
        if nd is not None and not np.isnan(nd):
            validos &= dados[i] != nd
        if m is not None:
            validos &= m.ReadAsArray(x, y, nx, ny) != 0
    return np.moveaxis(dados, 0, -1), validos


def classificar_cena(floresta, img, saida, bloco=2048, chunk=32768, dev=None):
    ds = gdal.Open(img)
    W, H = ds.RasterXSize, ds.RasterYSize
    mascaras = [ds.GetRasterBand(i + 1).GetMaskBand() for i in range(ds.RasterCount)]
    # bandas do mesmo mosaico costumam compartilhar máscara: lê só a primeira
    mascaras = [mascaras[0]] + [None] * (ds.RasterCount - 1)

    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(saida, W, H, 1, gdal.GDT_Byte,
                     options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"])
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    bo = out.GetRasterBand(1)
    bo.SetNoDataValue(NODATA_SAIDA)

    total = ((H + bloco - 1) // bloco) * ((W + bloco - 1) // bloco)
    t0 = time.monotonic()
    t_io = t_gpu = 0.0
    n_px = 0
    for k, (x, y, nx, ny) in enumerate(janelas(W, H, bloco), 1):
        ta = time.monotonic()
        dados, validos = ler_bloco(ds, x, y, nx, ny, mascaras)
        t_io += time.monotonic() - ta
        cls = np.full((ny, nx), NODATA_SAIDA, np.uint8)
        if validos.any():
            Xb = dados[validos]
            ta = time.monotonic()
            saidas = np.empty(len(Xb), np.uint8)
            for i in range(0, len(Xb), chunk):
                t = torch.as_tensor(Xb[i:i + chunk], device=dev, dtype=torch.float32)
                saidas[i:i + chunk] = floresta.predizer(t).astype(np.uint8)
            torch.cuda.synchronize()
            t_gpu += time.monotonic() - ta
            cls[validos] = saidas
            n_px += len(Xb)
        bo.WriteArray(cls, x, y)
        if k % 10 == 0 or k == total:
            print("  bloco %d/%d | %.1f Mpx válidos | io %.0fs gpu %.0fs | %.0fs"
                  % (k, total, n_px / 1e6, t_io, t_gpu, time.monotonic() - t0), flush=True)
    bo = None
    out = None
    return {"pixels": n_px, "segundos": time.monotonic() - t0,
            "io": t_io, "gpu": t_gpu}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", help="joblib do sklearn; com --exportar, vira .npz")
    ap.add_argument("--floresta", help=".npz já exportado (dispensa sklearn e RAM)")
    ap.add_argument("--exportar", help="destino .npz; converte --modelo e sai")
    ap.add_argument("--imagem")
    ap.add_argument("--saida")
    ap.add_argument("--bloco", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=1048576,
                    help="pixels por chamada à GPU. No backend triton pode ser grande; "
                         "no torch use ~32768, que já aloca (chunk x arvores) int64")
    ap.add_argument("--lote-arvores", type=int, default=0,
                    help="árvores por passada (0 = todas); só no backend torch")
    ap.add_argument("--backend", choices=("triton", "torch"), default="triton",
                    help="triton = kernel próprio (rápido); torch = gather (referência)")
    ap.add_argument("--verificar", action="store_true",
                    help="compara contra verificacao.npz gravado pelo rf_treino.py")
    args = ap.parse_args()

    if args.exportar:
        if not args.modelo:
            raise SystemExit("--exportar precisa de --modelo")
        print("[exportando] %s" % args.modelo, flush=True)
        print("  AVISO: recarregar um .joblib grande faz pico de 2x o tamanho;"
              " prefira achatar no mesmo processo do treino (rf_treino.py).",
              flush=True)
        import joblib
        t0 = time.monotonic()
        achatar(joblib.load(args.modelo), args.exportar)
        print("OK | %.0fs" % (time.monotonic() - t0))
        return
    if not args.floresta:
        raise SystemExit("precisa de --floresta")
    if not (args.imagem and args.saida) and not args.verificar:
        raise SystemExit("precisa de --imagem e --saida (ou só --verificar)")

    dev = ambiente.exigir_gpu()
    t0 = time.monotonic()
    fl = FlorestaGPU(args.floresta, dev, args.lote_arvores)
    print("[floresta na gpu] %d nos | profundidade %d | %.2f GB | %.0fs"
          % (fl.n_nos, fl.profundidade, fl.bytes / 1e9, time.monotonic() - t0), flush=True)
    if args.backend == "triton":
        from rf_kernel import FlorestaTriton
        fl = FlorestaTriton(fl)
        print("[backend] kernel triton (uma thread por pixel)", flush=True)
    else:
        print("[backend] torch gather (referencia, mais lento)", flush=True)

    if args.verificar:
        t0 = time.monotonic()
        dif, n = verificar(fl, args.floresta, dev)
        dt = time.monotonic() - t0
        print("[verificacao] %d/%d pixels divergem do sklearn (%.6f%%) | %.1fs "
              "-> %.3f Mpx/s na GPU"
              % (dif, n, 100 * dif / n, dt, n / 1e6 / dt), flush=True)
        if not (args.imagem and args.saida):
            return

    os.makedirs(os.path.dirname(os.path.abspath(args.saida)) or ".", exist_ok=True)
    r = classificar_cena(fl, args.imagem, args.saida, args.bloco, args.chunk, dev)
    print("OK -> %s | %.1f Mpx em %.0fs (%.2f Mpx/s) | io %.0fs gpu %.0fs"
          % (args.saida, r["pixels"] / 1e6, r["segundos"],
             r["pixels"] / 1e6 / r["segundos"], r["io"], r["gpu"]))


if __name__ == "__main__":
    main()
