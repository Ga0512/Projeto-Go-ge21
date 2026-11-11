#!/usr/bin/env python3
"""Parte de GPU: a floresta achatada em tensores e o kernel que a percorre.

Importado por `rf_floresta.py` e `rf_tabela.py` apenas quando há CUDA. Sem GPU
nenhum dos dois precisa deste arquivo.

Três coisas vivem aqui:

  achatar(rf, pasta)   grava a floresta do sklearn como arrays .npy planos
  FlorestaGPU          carrega esses arrays na VRAM
  FlorestaTriton       percorre, uma thread por pixel

POR QUE ACHATAR
---------------
Todas as árvores viram um único vetor de nós; `base[t]` diz onde a árvore t
começa. Um nó folha aponta para si mesmo (`esquerda == direita == ele`), então
a thread que chega nela simplesmente relê o mesmo nó -- já em cache -- até o
laço acabar, sem precisar de máscara nem de desvio de fluxo.

O achatamento também é a única forma de exportar um RF de profundidade livre
nesta máquina: o pickle de 500 árvores sem `max_depth` passa de 3 GB e
recarregá-lo faz pico de ~6,6 GB. Aqui o modelo já está em RAM e os arrays vão
direto para disco por memmap, descartando cada árvore assim que copiada.

POR QUE UM KERNEL, E NÃO OPERAÇÕES DE TENSOR
--------------------------------------------
A versão em torch puro (um `gather` por nível sobre tensores de pixels x
árvores) perde para a CPU: cada nível vira ~6 lançamentos de kernel e o índice
do nó vai e volta da memória global a cada passo. O custo não está na
aritmética, está no tráfego.

No kernel, cada thread cuida de um pixel: as features ficam em registradores,
o índice do nó também, e a cada nível a thread lê só os 16 bytes do seu nó. O
tráfego cai para `pixels x árvores x profundidade x 16 B`, que é o mínimo
possível sem comprimir a floresta.

Ainda assim o ganho sobre a CPU é modesto na floresta de profundidade livre
(1,2x medido). O gargalo é latência de memória aleatória: ~12.000 nós por
pixel, espalhados por 1,48 GB, e cada endereço só existe depois de ler o nó
anterior. A GPU tem latência de VRAM pior que a CPU (~600 ns contra ~80 ns) e
compensa com acesso coalescido, que uma travessia de árvore não tem. Com
`max_depth=12` o modelo cai para 60 MB, os primeiros níveis ficam em cache e aí
a GPU rende 24x.

Suporta até 4 features e até 4 classes.
"""

import json
import os

import numpy as np
import torch
import triton
import triton.language as tl


# ------------------------------------------------------------ achatamento
def achatar(rf, destino):
    """Grava a floresta do sklearn como arrays achatados (.npy) numa pasta.

    Chame no MESMO processo que treinou -- ver a nota sobre o pickle acima.
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
    for a in (feat, lim, esq, dir_, val):
        a.flush()
    np.save(os.path.join(destino, "base.npy"), base)
    with open(os.path.join(destino, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes.tolist(), "n_arvores": len(n),
                   "profundidade": prof, "n_nos": total, "n_classes": nc}, f)
    print("  -> %s" % destino, flush=True)


# ------------------------------------------------------------ carga na VRAM
class FlorestaGPU:
    """Arrays achatados -> tensores na VRAM."""

    def __init__(self, pasta, dev):
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
                # np.array copia de verdade: uma fatia de memmap é read-only, e
                # torch.from_numpy sobre ela emite UserWarning
                t[i:i + passo] = torch.from_numpy(np.array(a[i:i + passo]))
            return t

        self.base = sobe("base")
        self.feat = sobe("feat")
        self.lim = sobe("lim")
        self.esq = sobe("esq")
        self.dir = sobe("dir")
        self.val = sobe("val")
        self.dev = dev
        self.n_nos = int(self.feat.numel())
        self.bytes = sum(t.numel() * t.element_size() for t in
                         (self.feat, self.lim, self.esq, self.dir, self.val))


# ------------------------------------------------------------ o kernel
@triton.jit
def _percorrer(X, feat, lim, esq, dire, val, base, saida,
               N, T, F: tl.constexpr, C: tl.constexpr,
               PROF: tl.constexpr, BLOCO: tl.constexpr):
    pid = tl.program_id(0)
    linha = pid * BLOCO + tl.arange(0, BLOCO)
    m = linha < N

    # features do pixel, uma vez só, direto para registrador
    x0 = tl.load(X + linha * F + 0, mask=m, other=0.0)
    x1 = tl.load(X + linha * F + 1, mask=m & (F > 1), other=0.0)
    x2 = tl.load(X + linha * F + 2, mask=m & (F > 2), other=0.0)
    x3 = tl.load(X + linha * F + 3, mask=m & (F > 3), other=0.0)

    a0 = tl.zeros([BLOCO], tl.float32)
    a1 = tl.zeros([BLOCO], tl.float32)
    a2 = tl.zeros([BLOCO], tl.float32)
    a3 = tl.zeros([BLOCO], tl.float32)

    for t in range(T):
        no = tl.load(base + t) + tl.zeros([BLOCO], tl.int64)
        # A folha aponta para si mesma. Quando NENHUMA thread do bloco se move,
        # todas chegaram e o resto dos níveis seria desperdício: na floresta de
        # profundidade livre a folha média está no nível 24 de 76.
        andando = 1
        nivel = 0
        while andando > 0 and nivel < PROF:
            f = tl.load(feat + no, mask=m, other=0)
            v = tl.where(f == 0, x0, tl.where(f == 1, x1, tl.where(f == 2, x2, x3)))
            thr = tl.load(lim + no, mask=m, other=0.0)
            e = tl.load(esq + no, mask=m, other=0)
            d = tl.load(dire + no, mask=m, other=0)
            prox = tl.where(v > thr, d, e)
            prox = tl.where(m, prox, no)   # thread fora do bloco não anda
            andando = tl.sum((prox != no).to(tl.int32))
            no = prox
            nivel += 1
        a0 += tl.load(val + no * C + 0, mask=m, other=0.0)
        a1 += tl.load(val + no * C + 1, mask=m & (C > 1), other=0.0)
        a2 += tl.load(val + no * C + 2, mask=m & (C > 2), other=0.0)
        a3 += tl.load(val + no * C + 3, mask=m & (C > 3), other=0.0)

    # argmax com desempate pelo menor índice, igual ao numpy do sklearn
    melhor = tl.zeros([BLOCO], tl.int32)
    topo = a0
    melhor = tl.where((C > 1) & (a1 > topo), 1, melhor)
    topo = tl.where((C > 1) & (a1 > topo), a1, topo)
    melhor = tl.where((C > 2) & (a2 > topo), 2, melhor)
    topo = tl.where((C > 2) & (a2 > topo), a2, topo)
    melhor = tl.where((C > 3) & (a3 > topo), 3, melhor)
    tl.store(saida + linha, melhor, mask=m)


class FlorestaTriton:
    """Travessia da floresta no kernel, uma thread por pixel."""

    def __init__(self, fl, bloco=128):
        self.fl = fl
        self.bloco = bloco
        self.classes = fl.classes
        # o kernel indexa val como val[no * C + c]: precisa ser contíguo
        self.val = fl.val.contiguous()
        self.feat = fl.feat.contiguous()
        self.lim = fl.lim.contiguous()
        self.esq = fl.esq.contiguous()
        self.dir = fl.dir.contiguous()
        self.base = fl.base.contiguous()

    @torch.no_grad()
    def indices(self, X):
        """X: (N, F) float32 na GPU -> (N,) int32 com o índice da classe."""
        X = X.contiguous()
        N, F = X.shape
        out = torch.empty(N, dtype=torch.int32, device=X.device)
        grade = (triton.cdiv(N, self.bloco),)
        _percorrer[grade](X, self.feat, self.lim, self.esq, self.dir,
                          self.val, self.base, out,
                          N, self.fl.n_arvores, F, self.fl.n_classes,
                          self.fl.profundidade + 1, self.bloco,
                          num_warps=4)
        return out

    @torch.no_grad()
    def predizer(self, X):
        """Devolve os rótulos originais do sklearn (rf.classes_)."""
        return self.classes[self.indices(X).cpu().numpy()]
