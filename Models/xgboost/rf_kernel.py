#!/usr/bin/env python3
"""Travessia de floresta em kernel Triton: uma thread por pixel.

A versão em operações de tensor (`FlorestaGPU.prob`) perde para a CPU porque
cada nível da árvore vira ~6 kernels sobre tensores de (pixels x árvores), e
o índice do nó vai e volta da memória global a cada passo. O custo não está na
aritmética, está no tráfego.

Aqui cada thread cuida de um pixel: as features ficam em registradores, o
índice do nó também, e a cada nível a thread lê só os 16 bytes do seu nó. O
tráfego cai para `pixels x árvores x profundidade x 16 B`, que é o mínimo
possível sem comprimir a floresta.

A folha aponta para si mesma, então uma thread que chegou ao fim simplesmente
relê o mesmo nó (já em cache) até o laço acabar. Isso evita divergência de
fluxo entre threads do mesmo warp.

Suporta até 4 features e até 4 classes, que é o que os modelos deste projeto
usam (PAN/GLCM/rugosidade e 3 ou 4 classes).
"""

import torch
import triton
import triton.language as tl


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
    """Mesma interface da FlorestaGPU, com a travessia no kernel."""

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
        """X: (N, F) float32 contíguo na GPU -> (N,) int32 com o índice da classe."""
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
        return self.classes[self.indices(X).cpu().numpy()]
