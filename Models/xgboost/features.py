"""Features por pixel, calculadas na GPU.

Para cada pixel: as bandas originais + média e desvio padrão local de cada
banda em janelas quadradas de vários tamanhos (contexto espacial multiescala).
Com 3 bandas e janelas (5, 15, 31): 3 + 3*3*2 = 21 features.

Borda: padding por reflexão. Em patch de treino isso cria valores artificiais
perto da borda; por isso o treino e a avaliação usam só pixels a mais de
`margem` (= maior janela // 2) da borda, onde a janela cabe inteira no patch.
"""

import torch
import torch.nn.functional as F


def nomes(bandas, janelas):
    out = list(bandas)
    for j in janelas:
        out += ["%s_media%d" % (b, j) for b in bandas]
        out += ["%s_desvio%d" % (b, j) for b in bandas]
    return out


def _media(x, j):
    """Média em janela j x j, calculada como 1xj seguida de jx1.

    A média de caixa é separável, e o padding por reflexão também, então o
    resultado é o mesmo de um avg_pool2d quadrado -- mas custa 2j operações por
    pixel em vez de j². Medido numa RTX 3060 em tensor 512x512x3:

        j      quadrado   separável   ganho
        5         1,1 ms     1,74 ms   0,6x   (kernel pequeno não compensa)
        15        3,2 ms     2,16 ms   1,5x
        31        8,3 ms     2,82 ms   3,0x
        75       34,4 ms     4,34 ms   7,9x
        155     164,4 ms     8,59 ms  19,1x

    Somando as janelas usadas: SPOT5 [5,15,31] 12,6 -> 6,7 ms (1,9x) e
    WorldView [25,75,155] 203,5 -> 15,1 ms (13,5x).

    Precisão: contra uma referência em float64, o erro do separável é MENOR que
    o do quadrado (1,8e-06 contra 6,2e-06 em j=31) -- somar 2j termos acumula
    menos que somar j².

    O desvio, porém, sai de `sqrt(E[x²] - E[x]²)`, e em região plana (variância
    ~0) essa subtração cancela: uma diferença de 1e-12 nos termos vira 1e-2 no
    resultado. Isso faz ~0,8% dos pixels do interior trocarem de classe no
    xgb_spot5_v18. NÃO é efeito desta mudança: medido nos mesmos 510.760 px,

        quadrado float32 x quadrado float64   0,7957%
        quadrado float32 x separável float32  0,8016%

    ou seja, calcular as features em float64 troca os mesmos pixels. São pixels
    em cima de um limiar, onde o ruído de float32 decide. Qualquer troca de GPU
    ou de versão do CUDA produz a mesma variação.
    """
    r = j // 2
    y = F.avg_pool2d(F.pad(x, (r, r, 0, 0), mode="reflect"), (1, j), stride=1)
    return F.avg_pool2d(F.pad(y, (0, 0, r, r), mode="reflect"), (j, 1), stride=1)


def calcular(x, janelas):
    """x: (B,C,H,W) float32 na GPU -> (B, C*(1+2*len(janelas)), H, W)."""
    feats = [x]
    xx = x * x
    for j in janelas:
        media = _media(x, j)
        desvio = (_media(xx, j) - media * media).clamp_min(0).sqrt()
        feats += [media, desvio]
    return torch.cat(feats, dim=1)
