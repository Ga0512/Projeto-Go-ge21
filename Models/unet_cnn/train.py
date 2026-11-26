#!/usr/bin/env python3
"""Treino da U-Net CNN (unet.py) nos patches SPOT5 refinados (4 classes).

Espelha train_seg.py para comparação direta com o ViT (seg_mae_v3_*) e o
XGBoost no mesmo dataset: mesma augmentation (flip/rot90 imagem+máscara,
brilho/contraste só na PAN), mesma loss (CE + Dice), mesmas métricas no patch
inteiro e no "interior" (--margem px da borda, igual ao XGBoost).

- Treino: splits/treino.txt (patches 256 inteiros).
- Melhor epoch escolhido pelo mIoU da validação (com paciência); teste só no fim.
- Trava de espaço: para e preserva modelo_melhor.pth se /mnt/c ficar com menos
  de --min-livre-c-gb livres.

Saída em unet_cnn/runs/<nome>/: modelo_melhor.pth, log.csv, relatorio.json,
visualizacao/{validacao,teste}.png (metade de cada cena).

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/train.py --nome unet_v1
"""

import argparse
import copy
import csv
import io
import json
import math
import os
import shutil
import time

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch)
import numpy as np
import torch
import torch.nn.functional as F

import dados
from unet import ATTENTION_NODES, build_model, normalized_state_dict

BASE = os.path.dirname(os.path.abspath(__file__))
N = dados.N_CLASSES
IGN = dados.IGNORE
PESO = None          # peso por classe na CE (--peso-classe); None = sem peso
BORDA_RAIO = 0       # --borda-raio: faixa (px) ao redor da fronteira entre classes
BORDA_ALFA = 0.0     # --borda-alfa: peso extra da CE nessa faixa


def c_livre_gb():
    """Espaço livre no C: visto do WSL (pagefile do Windows e swap do WSL moram lá)."""
    return shutil.disk_usage("/mnt/c").free / 1e9 if os.path.isdir("/mnt/c") else float("inf")


def aumentar(x, y, rng, ganho_pan, offset_pan, escala=0.0, cutout=0,
             cutout_frac=0.25):
    """Uma das 8 combinações de rot90/flip (imagem e máscara juntas) + PAN
    com ganho/deslocamento aleatórios (as texturas não mudam com brilho).

    `escala` e `cutout` sao opcionais e saem DESLIGADOS, para nao mudar o
    significado dos runs antigos. Existem porque o limite medido no
    wv_jatai_v3 e falta de PAISAGEM, nao de otimizacao: sao 5 blocos de treino,
    a rede chega ao teto na epoca 2 e dali em diante a loss de treino cai
    enquanto a validacao piora. Sem rotular mais blocos, a saida e simular
    variedade.

    escala > 0: recorta 1/s do patch (s em [1, 1+escala]) e reamostra de volta
        ao tamanho original -- a rede passa a ver talhao e copa em tamanhos
        diferentes. Uma escala por LOTE, nao por amostra, para gastar uma
        interpolacao so; com 350 passos por epoca a variedade vem do numero de
        passos. Mascara por vizinho mais proximo, senao surgem classes que nao
        existem.

    cutout > 0: apaga N retangulos por patch, preenchendo com ruido N(0,1) e
        PRESERVANDO o rotulo -- forca a decidir pelo contexto em vez de decorar
        a mancha. O preenchimento e ruido, e nao zero, porque neste pipeline
        zero ja significa "pixel invalido" (dados.carregar zera a imagem e
        marca IGNORE); as bandas sao z-score, entao ruido N(0,1) e indistinto
        de textura comum e nao ensina a confundir as duas coisas.
    """
    B = len(x)
    rot, flip = rng.integers(0, 4, B), rng.integers(0, 2, B)
    xs, ys = [], []
    for i in range(B):
        a, b = x[i], y[i]
        if rot[i]:
            a, b = torch.rot90(a, int(rot[i]), (1, 2)), torch.rot90(b, int(rot[i]), (0, 1))
        if flip[i]:
            a, b = a.flip(2), b.flip(1)
        xs.append(a)
        ys.append(b)
    x, y = torch.stack(xs).float(), torch.stack(ys).long()

    H, W = x.shape[-2:]
    if escala > 0:
        s = float(rng.uniform(1.0, 1.0 + escala))
        h, w = max(16, int(round(H / s))), max(16, int(round(W / s)))
        if h < H or w < W:
            oy = int(rng.integers(0, H - h + 1))
            ox = int(rng.integers(0, W - w + 1))
            x = F.interpolate(x[:, :, oy:oy + h, ox:ox + w], size=(H, W),
                              mode="bilinear", align_corners=False)
            y = F.interpolate(y[:, None, oy:oy + h, ox:ox + w].float(),
                              size=(H, W), mode="nearest")[:, 0].long()

    if ganho_pan or offset_pan:
        g = torch.from_numpy(rng.uniform(1 - ganho_pan, 1 + ganho_pan, B)).to(x)
        o = torch.from_numpy(rng.uniform(-offset_pan, offset_pan, B)).to(x)
        x[:, 0] = x[:, 0] * g[:, None, None] + o[:, None, None]

    if cutout > 0 and cutout_frac > 0:
        lado = max(8, int(round(min(H, W) * cutout_frac)))
        for i in range(B):
            for _ in range(int(cutout)):
                ch = int(rng.integers(8, lado + 1))
                cw = int(rng.integers(8, lado + 1))
                oy = int(rng.integers(0, H - ch + 1))
                ox = int(rng.integers(0, W - cw + 1))
                x[i, :, oy:oy + ch, ox:ox + cw] = torch.randn(
                    x.shape[1], ch, cw, device=x.device, dtype=x.dtype)
    return x, y


def pesos_reamostragem(Y, cenas, n_classes, k, presenca_min=0.01, lote=256):
    """Peso de sorteio por patch que corrige o par (CENA x CLASSE) mais torto.

    O PROBLEMA. No v18 a nao-vegetacao e 43,8% dos pixels de treino -- nada
    rara -- e mesmo assim o IoU dela e 0,0302 em colinas_do_sul contra 0,9522
    em jatai. O desequilibrio nao esta na frequencia da classe, esta em QUAL
    CENA ela vem: dos pixels de nao-vegetacao do treino, 85,6% sao de jatai e
    14,4% de colinas. A aparencia difere entre as duas (clareira pequena no
    meio da mata contra talhao grande e claro), e o detector sai esculpido pela
    cena dominante. Nem peso de classe nem Dice corrigem isso: para os dois a
    classe esta boa, porque ambos olham o agregado.

    O LIMITE. Equalizar todos os pares (cena x classe) de uma vez e impossivel
    com peso por patch, porque o patch mistura classes em proporcao fixa:
    multiplicar colinas por 7,5 para igualar a nao-vegetacao levaria a
    vegetacao de colinas de 75,9% para 96% do total, quebrando outra classe
    para consertar esta. Entao se escolhe UM par e se aceita o colateral.

    O METODO. Acha o par (cena, classe) com o maior deficit de share, entre os
    pares em que a classe realmente existe naquela cena (>= presenca_min da
    cena -- sem isso o par escolhido seria (jatai, Agua), que esta em 0,1% do
    share nao por desequilibrio mas porque jatai nao tem agua: 0,16% da cena).
    Depois soma ao peso dos patches daquela cena, proporcionalmente a quanto
    cada um carrega da classe escassa:

        w_i = 1 + k * deficit * frac(c*, i) / max_j frac(c*, j)

    Medido na composicao do v18 (share de colinas na nao-vegetacao, pico do
    sorteio em multiplos da media, e tamanho efetivo da amostra):

        k       share      pico    amostra efetiva
        0 (off) 14,4%      1,00x       100%
        3       30,9%      3,36x        84%
        8       47,6%      5,96x        54%
        20      66,9%      9,24x        29%

    k=3 mais que dobra o share com pico de 3,4x, deixando a Agua intocada
    (99,9% -> 99,9%) e a Vegetacao quase (75,9% -> 79,3%). k alto chega perto
    da paridade mas concentra o sorteio nos mesmos poucos patches, que e
    overfitting contratado -- colinas so tem 599 patches com >10% de
    nao-vegetacao.

    As fracoes saem do rotulo em memoria, nao do manifest.csv: assim valem
    depois do LUT do dados.carregar() e de --juntar-agua, e nao dependem do
    nome das colunas do manifest.

    -> (pesos somando 1, matriz frac n x n_classes, (cena*, classe*, deficit))
    """
    n = len(Y)
    frac = np.zeros((n, n_classes), np.float64)
    for i in range(0, n, lote):
        y = Y[i:i + lote]
        y = y.reshape(len(y), -1).long()
        cls = torch.where(y >= n_classes, n_classes, y)       # IGNORE -> bin extra
        linha = torch.arange(len(y), device=y.device)[:, None].expand_as(cls)
        c = torch.bincount((linha * (n_classes + 1) + cls).reshape(-1),
                           minlength=len(y) * (n_classes + 1)
                           ).reshape(len(y), n_classes + 1)
        c = c[:, :n_classes].double().cpu().numpy()           # descarta o IGNORE
        frac[i:i + len(y)] = c / np.maximum(1.0, c.sum(1, keepdims=True))

    lista = sorted(set(cenas.tolist()))
    if k <= 0 or len(lista) < 2:
        return np.full(n, 1.0 / n), frac, None

    alvo = 1.0 / len(lista)
    pior, melhor_def = None, 0.0
    for ci in range(n_classes):
        tot = frac[:, ci].sum()
        if tot <= 0:
            continue
        for nome in lista:
            m = cenas == nome
            # a classe existe nesta cena? share DA CENA, nao do total
            if frac[m, ci].mean() < presenca_min:
                continue
            share = frac[m, ci].sum() / tot
            deficit = (alvo - share) / alvo
            if deficit > melhor_def:
                pior, melhor_def = (nome, ci), deficit

    if pior is None:
        print("[reamostrar] nenhum par (cena x classe) deficitario: uniforme",
              flush=True)
        return np.full(n, 1.0 / n), frac, None

    nome, ci = pior
    m = cenas == nome
    w = np.ones(n)
    pico = max(frac[m, ci].max(), 1e-9)
    w[m] = 1.0 + k * melhor_def * frac[m, ci] / pico
    return w / w.sum(), frac, (nome, ci, melhor_def)


def relatar_reamostragem(pesos, frac, cenas, n_classes, k, alvo=None):
    """Composicao antes -> depois. E a auditoria do experimento: sem ela,
    "reamostrei" e uma afirmacao sem numero."""
    n = len(pesos)
    if alvo is None:
        print("[reamostrar] k=%.1f | sem par deficitario (sorteio uniforme)"
              % k, flush=True)
    else:
        print("[reamostrar] k=%.1f | par mais torto: (%s, %s) com deficit de "
              "%.0f%% do share" % (k, alvo[0], dados.CLASS_NOMES[alvo[1]],
                                   100 * alvo[2]), flush=True)
    print("    share dos pixels de cada classe, por cena (uniforme -> sorteado):",
          flush=True)
    for c in range(n_classes):
        tot_u = frac[:, c].sum()
        tot_p = float((pesos * frac[:, c]).sum())
        if tot_u <= 0:
            continue
        partes = []
        for k in sorted(set(cenas.tolist())):
            m = cenas == k
            u = frac[m, c].sum() / tot_u
            p = float((pesos[m] * frac[m, c]).sum()) / max(tot_p, 1e-12)
            partes.append("%-14s %5.1f%% -> %5.1f%%" % (k[:14], 100 * u, 100 * p))
        print("    %-26s %s" % (dados.CLASS_NOMES[c], " | ".join(partes)), flush=True)
    ef = 1.0 / float((pesos ** 2).sum()) / n
    print("    pico %.2fx a media | amostra efetiva %.0f%% dos %d patches"
          % (pesos.max() * n, 100 * ef, n), flush=True)


def faixa_fronteira(y, raio):
    """bool (B,H,W): pixels a <= `raio` da fronteira entre classes do ROTULO.

    Mede-se a fronteira no rotulo, nao na predicao, para a faixa nao mudar
    conforme o modelo aprende -- senao o peso (e a metrica) viram alvo movel.

    A dilatacao e um max_pool: distancia de Chebyshev, nao euclidiana. Para
    este uso tanto faz (a diferenca e o canto do quadrado contra o circulo) e
    sai numa chamada de GPU, em vez de uma transformada de distancia por
    amostra na CPU.
    """
    v = y != IGN
    yy = torch.where(v, y, torch.full_like(y, -1))
    b = torch.zeros_like(v)
    dif = (yy[:, :, 1:] != yy[:, :, :-1]) & v[:, :, 1:] & v[:, :, :-1]
    b[:, :, 1:] |= dif
    b[:, :, :-1] |= dif
    dif = (yy[:, 1:, :] != yy[:, :-1, :]) & v[:, 1:, :] & v[:, :-1, :]
    b[:, 1:, :] |= dif
    b[:, :-1, :] |= dif
    if raio <= 0:
        return b
    d = F.max_pool2d(b.float()[:, None], 2 * raio + 1, stride=1, padding=raio)
    return d[:, 0] > 0


def perda(logits, y):
    """Cross-entropy (ignora sem rótulo) + Dice suave médio das classes.

    O PESO (--peso-classe) entra SÓ na CE, nunca no Dice: o Dice já é média por
    classe, ou seja já dá 1/N a cada uma independente do tamanho. Pôr peso nos
    dois seria contar duas vezes, e com um agravante -- o Dice é somado sobre o
    lote inteiro, misturando as cenas, então ele não enxerga uma classe que vai
    bem numa cena e mal noutra (ver --reamostrar). Quem desloca o limiar de
    decisão por pixel é a CE; é nela que o peso tem efeito.
    """
    logits = logits.float()
    if BORDA_ALFA > 0:
        # Olhando os recortes ampliados: o erro nao e imprecisao do traco, e
        # VEGETACAO ESPARSA (cerrado ralo) na transicao entre dossel fechado e
        # campo aberto -- uma faixa de ~30 m que o modelo chama de nao
        # vegetacao inteira. Ela e 5,7% dos pixels e carrega quase todo o erro
        # (acerto 50% junto a fronteira contra 98,8% no interior), entao sem
        # peso o gradiente e dominado por pixel que o modelo ja acerta.
        pp = F.cross_entropy(logits, y, weight=PESO, ignore_index=IGN,
                             reduction="none")
        w = 1.0 + BORDA_ALFA * faixa_fronteira(y, BORDA_RAIO).float()
        w = w * (y != IGN).float()
        ce = (pp * w).sum() / w.sum().clamp_min(1.0)
    else:
        ce = F.cross_entropy(logits, y, weight=PESO, ignore_index=IGN)
    valido = (y != IGN)[:, None]
    p = logits.softmax(1) * valido
    t = F.one_hot(torch.where(y == IGN, 0, y), N).permute(0, 3, 1, 2) * valido
    inter = (p * t).sum(dim=(0, 2, 3))
    soma = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    return ce + (1 - ((2 * inter + 1) / (soma + 1)).mean())


@torch.no_grad()
def prever(modelo, X, lote=32):
    """Inferência em lotes. CUIDADO com o `lote` padrão: ele foi calibrado para
    patch 256, e a validação/avaliação chamavam sem argumento. Em patch 512 um
    lote de 32 é 4x a área -- equivale a 128 patches de 256 -- e derrubou o
    treino do WorldView com `CUDA error: device not ready` no fim do epoch 1,
    DEPOIS de o treino em si ter rodado liso com --lote 2 (3085 de 6144 MiB).
    Quem chama agora passa o lote explicitamente, derivado de --lote."""
    modelo.eval()
    d = next(modelo.parameters()).device
    out = []
    for i in range(0, len(X), lote):
        xb = X[i:i + lote]
        if xb.device != d:
            xb = xb.to(d)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(modelo(xb.float()).argmax(1).to(torch.uint8))
    return torch.cat(out)


def _ix(t, idx):
    """t[idx] com o indice no device de t -- com --dados-na-ram os splits ficam
    na RAM e `pred` na GPU, entao um indice so nao serve para os dois."""
    return t[idx.to(t.device)]


def matriz(pred, y, mascara=None):
    if y.device != pred.device:
        y = y.to(pred.device)
    if mascara is not None and mascara.device != pred.device:
        mascara = mascara.to(pred.device)
    v = y != IGN
    if mascara is not None:
        v &= mascara
    return torch.bincount(y[v].long() * N + pred[v].long(), minlength=N * N) \
        .reshape(N, N).cpu().numpy()


def _lista(v):
    return [None if np.isnan(a) else float(a) for a in v]


def metricas(cm, rotulo=None):
    """OA, mIoU e recall/precisão/F1/IoU por classe (mesmo schema do XGBoost)."""
    tp = np.diag(cm).astype(np.float64)
    ref, prd = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall, prec = tp / ref, tp / prd
        f1, iou = 2 * tp / (ref + prd), tp / (ref + prd - tp)
    oa = tp.sum() / max(cm.sum(), 1)
    miou = float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan")
    # mIoU simples dá peso igual a toda classe, inclusive a que mal aparece: na
    # Jataí a água é 0,15% dos pixels e pesa 1/3 do agregado. Não dá para
    # excluí-la (ela EXISTE e o modelo erra 100% dela -- isso é defeito, não
    # artefato), então somam-se duas leituras em vez de mexer na primeira:
    #   miou_presentes  média só das classes com referência > 0 (classe de fato
    #                   ausente deixa de contar; na Jataí não muda nada)
    #   fwiou           IoU ponderado pela frequência -- resume sem esconder
    pres = ref > 0
    miou_pres = (float(np.nanmean(iou[pres]))
                 if pres.any() and np.isfinite(iou[pres]).any() else float("nan"))
    fw = ref / max(cm.sum(), 1)
    fwiou = float(np.nansum(np.where(np.isfinite(iou), iou, 0.0) * fw))
    if rotulo:
        linhas = ["%s  OA=%.4f  mIoU=%.4f  (%d pixels)" % (rotulo, oa, miou, cm.sum()),
                  "  %2s  %-26s  %7s  %7s  %7s  %7s  %6s"
                  % ("cl", "nome", "recall", "precis", "F1", "IoU", "ref%")]
        for c in range(N):
            linhas.append("  %2d  %-26s  %7.4f  %7.4f  %7.4f  %7.4f  %5.1f%%"
                          % (c, dados.CLASS_NOMES[c], recall[c], prec[c], f1[c], iou[c],
                             100 * ref[c] / max(cm.sum(), 1)))
        print("\n".join(linhas), flush=True)
    return {"oa": float(oa), "miou": miou, "miou_presentes": miou_pres, "fwiou": fwiou,
            "acuracia": _lista(recall),
            "precisao": _lista(prec), "f1": _lista(f1), "iou": _lista(iou),
            "matriz_confusao": cm.tolist(), "n_pixels": int(cm.sum())}


def visualizar(imgs, rotulos, preds, nomes, cenas, margem, caminho, n, titulo):
    """PAN | referência | predição | erro; metade dos patches de cada cena."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    cmap = ListedColormap(dados.CORES_CLASSE)
    cmap.set_bad("#bdbdbd")
    cmap_erro = ListedColormap(["#2ca02c", "#d62728"])
    cmap_erro.set_bad("#bdbdbd")
    unicas = list(dict.fromkeys(cenas))
    por_cena = max(1, n // len(unicas))
    grupos = []
    for c in unicas:
        ids = np.flatnonzero(cenas == c)
        pos = np.unique(np.linspace(0, len(ids) - 1, min(por_cena, len(ids))).astype(int))
        grupos.append((c, ids[pos], len(ids)))
    n_lin = sum(len(s) for _, s, _ in grupos)
    fig = plt.figure(figsize=(13, 3.4 * n_lin + 0.8 * len(grupos)), layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(len(grupos), 1,
                                           height_ratios=[len(s) for _, s, _ in grupos]))
    for k, (sf, (c, sel, total)) in enumerate(zip(subfigs, grupos)):
        sf.set_facecolor("#f4f4f4" if k % 2 == 0 else "#e6eef7")
        sf.suptitle("%s  |  cena %s  (%d de %d patches)" % (titulo, c, len(sel), total),
                    fontsize=14, fontweight="bold")
        axs = sf.subplots(len(sel), 4, squeeze=False)
        for lin, i in enumerate(sel):
            img, ref, pred = imgs[i], rotulos[i], preds[i]
            h, w = ref.shape
            sem = ref == IGN
            aval = np.zeros_like(sem)
            aval[margem:h - margem, margem:w - margem] = True
            aval &= ~sem
            oa = (pred[aval] == ref[aval]).mean() if aval.any() else float("nan")
            v0, v1 = np.percentile(img[0], [2, 98])
            rotulo = nomes[i][len(c) + 1:] if nomes[i].startswith(c + "_") else nomes[i]
            a = axs[lin]
            a[0].imshow(img[0], cmap="gray", vmin=v0, vmax=v1)
            a[1].imshow(np.ma.masked_where(sem, ref), cmap=cmap, vmin=0, vmax=N - 1,
                        interpolation="nearest")
            a[2].imshow(pred, cmap=cmap, vmin=0, vmax=N - 1, interpolation="nearest")
            a[3].imshow(np.ma.masked_where(sem, pred != ref), cmap=cmap_erro, vmin=0, vmax=1,
                        interpolation="nearest")
            for col, tit2 in enumerate(["PAN  (%s)" % rotulo, "referência", "predição",
                                        "erro (OA interior=%.3f)" % oa]):
                a[col].set_title(tit2, fontsize=9)
                a[col].set_xticks([])
                a[col].set_yticks([])
            for col in (2, 3):
                a[col].add_patch(Rectangle((margem - .5, margem - .5), w - 2 * margem,
                                           h - 2 * margem, fill=False, ls="--", lw=.8, ec="white"))
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cor) for cor in dados.CORES_CLASSE],
               labels=dados.CLASS_NOMES, loc="outside lower center", ncol=N, fontsize=10)
    fig.savefig(caminho, dpi=80)
    plt.close(fig)


def _hms(seg):
    """segundos -> '7m12s', para o tempo restante do epoch."""
    seg = int(max(seg, 0))
    return "%dm%02ds" % (seg // 60, seg % 60) if seg >= 60 else "%ds" % seg


def main():
    global N, IGN, PESO                         # N/IGN/PESO usados por perda()/matriz()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", required=True, help="pasta de saída em unet_cnn/runs/")
    ap.add_argument("--dataset", default=dados.DATASET_PADRAO)
    ap.add_argument("--inicializar", default=None,
                    help="modelo_melhor.pth de um run anterior p/ warm-start (fine-tuning): "
                         "carrega os pesos que baterem (o nº de classes pode diferir — a "
                         "cabeça com shape novo fica re-inicializada). Use LR baixa.")
    ap.add_argument("--base", type=int, default=32, help="canais do 1º nível do encoder")
    ap.add_argument("--architecture", default="unetpp", choices=["unet", "unetpp"],
                    help="unet = classic U-Net; unetpp = U-Net++")
    ap.add_argument("--arquitetura", dest="architecture", choices=["unet", "unetpp"],
                    help=argparse.SUPPRESS)
    ap.add_argument("--attention-input", action="store_true",
                    help="channel attention on the original input bands")
    ap.add_argument("--attention-input-mode", choices=["softmax", "sigmoid"],
                    default="softmax",
                    help="softmax weights sum to 1 across bands; sigmoid loads legacy runs")
    ap.add_argument("--attention-encoder", action="store_true",
                    help="CBAM after every encoder block")
    ap.add_argument("--attention-decoder", action="store_true",
                    help="CBAM after every decoder block")
    ap.add_argument("--attention-nodes", nargs="+", choices=ATTENTION_NODES,
                    default=None, help="specific CBAM blocks; combines with encoder/decoder")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--juntar-agua", action="store_true",
                    help="funde Água em Não vegetação (2 classes). Desligado por padrão: "
                         "só faz sentido onde a água é rara demais para ser aprendida e "
                         "medida -- ver a nota em dados._classes()")
    ap.add_argument("--peso-classe", nargs="+", type=float, default=None, metavar="W",
                    help="peso por classe na cross-entropy, na ordem das classes "
                         "(ex.: 1 2 1 dobra o custo de errar nao-vegetacao). Entra "
                         "so na CE, nunca no Dice -- ver perda(). Desligado por "
                         "padrao, para os runs anteriores seguirem reproduziveis")
    ap.add_argument("--reamostrar", action="store_true",
                    help="sorteia os patches do treino com peso que equaliza os "
                         "pares (cena x classe), em vez de percorrer todos uma vez "
                         "por epoch -- ver pesos_reamostragem()")
    ap.add_argument("--reamostrar-k", type=float, default=3.0,
                    help="forca da reamostragem. Medido no v18: k=3 leva o share "
                         "de colinas na nao-vegetacao de 14,4%% para 30,9%% com "
                         "pico de 3,4x no sorteio; k=8 chega a 47,6%% com 6,0x; "
                         "k=20 a 66,9%% com 9,2x e amostra efetiva de 29%%. "
                         "0 desliga -- ver pesos_reamostragem()")
    ap.add_argument("--amostras-por-epoca", type=int, default=None,
                    help="numero de sorteios por epoca com --reamostrar. "
                         "Use para comparar datasets de tamanhos diferentes "
                         "com o mesmo numero de passos por epoca. "
                         "Padrao: numero de patches de treino.")
    ap.add_argument("--borda-raio", type=int, default=0, metavar="R",
                    help="faixa de R px ao redor da fronteira entre classes. "
                         "Com --borda-alfa, a CE pesa (1+alfa) ali. 16 px = "
                         "~8 m a 0,486 m, que cobre a transicao medida.")
    ap.add_argument("--borda-alfa", type=float, default=0.0, metavar="A",
                    help="peso extra da CE na faixa de fronteira. 0 desliga "
                         "(padrao). 3 a 5 e um ponto de partida; alto demais "
                         "ensina a rede a copiar a imprecisao do rotulo.")
    ap.add_argument("--escala-jitter", type=float, default=0.0, metavar="S",
                    help="recorta 1/s do patch (s em [1,1+S]) e reamostra de "
                         "volta: a rede ve as feicoes em tamanhos variados. "
                         "0 desliga (padrao, para nao mexer nos runs antigos). "
                         "0.3 e um ponto de partida razoavel.")
    ap.add_argument("--cutout", type=int, default=0, metavar="N",
                    help="apaga N retangulos por patch (ruido N(0,1), rotulo "
                         "preservado), forcando decisao por contexto. 0 desliga.")
    ap.add_argument("--cutout-frac", type=float, default=0.25, metavar="F",
                    help="lado maximo de cada retangulo do --cutout, como "
                         "fracao do patch.")
    ap.add_argument("--dados-na-ram", action="store_true",
                    help="mantem os splits na RAM e manda so o lote para a GPU. "
                         "O padrao e deixar os tres splits residentes na VRAM, "
                         "o que custa 4,7 GB com patch 512 e 4 bandas -- numa "
                         "placa de 6 GB sobra o bastante so para --lote 2, e "
                         "ainda assim o treino arrasta por reciclagem de blocos. "
                         "Com esta opcao a VRAM fica para as ativacoes e da para "
                         "subir o --lote, que costuma sair mais rapido apesar da "
                         "transferencia por passo (~33 MB com lote 16). Nao custa "
                         "RAM extra: torch.from_numpy compartilha o buffer.")
    ap.add_argument("--log-cada", type=int, default=50, metavar="N",
                    help="imprime vazao e VRAM a cada N passos (0 desliga). "
                         "Sem isto o unico sinal de vida e a linha de fim de "
                         "epoca, que no WorldView leva mais de 20 min -- e nao "
                         "ha como distinguir 'lento' de 'travado'.")
    ap.add_argument("--resume", action="store_true",
                    help="retomar de runs/<nome>/ultimo.pth (gravado a cada epoch): "
                         "continua do epoch seguinte, com otimizador, histórico e "
                         "melhor modelo preservados")
    ap.add_argument("--paciencia", type=int, default=25,
                    help="para se o mIoU de validação não melhora em N epochs")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--lote-val", type=int, default=0,
                    help="patches por lote na inferência de validação/teste. "
                         "0 = 4x --lote (o padrão histórico). Não altera "
                         "métrica nenhuma, só o agrupamento: baixe se der "
                         "OOM ou 'device not ready' -- com U-Net++ em patch "
                         "256 nesta GPU de 6 GB, 64 não cabe e 8 cabe")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--aquecimento", type=int, default=5, help="epochs de warmup")
    ap.add_argument("--ganho-pan", type=float, default=0.2)
    ap.add_argument("--offset-pan", type=float, default=0.2)
    ap.add_argument("--margem", type=int, default=15,
                    help="borda excluída nas métricas 'interior' (igual ao XGBoost)")
    ap.add_argument("--n-png", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-livre-c-gb", type=float, default=1.0,
                    help="trava: para se o C: (/mnt/c) ficar abaixo disso")
    args = ap.parse_args()

    dados.configurar(args.dataset, args.juntar_agua)   # nº de classes seguem o dataset (v13=4, v15=3)
    N = dados.N_CLASSES
    IGN = dados.IGNORE

    dev = ambiente.exigir_gpu()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    global PESO, BORDA_RAIO, BORDA_ALFA
    rng = np.random.default_rng(args.seed)
    saida = os.path.join(BASE, "runs", args.nome)
    os.makedirs(os.path.join(saida, "visualizacao"), exist_ok=True)

    t0 = time.monotonic()
    cena_de = dados.cenas(args.dataset)
    nomes, brutos, cenas, X, Y = {}, {}, {}, {}, {}
    for s in ("treino", "validacao", "teste"):
        nomes[s] = dados.ler_split(args.dataset, s)
        brutos[s] = dados.carregar_cache(args.dataset, s, nomes[s], np.float16)
        cenas[s] = np.array([cena_de[n] for n in nomes[s]])
        if args.dados_na_ram:
            X[s] = torch.from_numpy(brutos[s][0])          # compartilha o buffer
            Y[s] = torch.from_numpy(brutos[s][1])
        else:
            X[s] = torch.from_numpy(brutos[s][0]).to(dev, torch.float16)
            Y[s] = torch.from_numpy(brutos[s][1]).to(dev)
    print("[dados] treino=%d val=%d teste=%d | %.0fs"
          % (len(X["treino"]), len(X["validacao"]), len(X["teste"]), time.monotonic() - t0))

    # Quanto da placa o dataset ocupa PARADO. O train.py mantem os tres splits
    # residentes na VRAM; com patch 256 do SPOT5 isso e barato, mas no 512 do
    # WorldView com 4 bandas sao 4,7 GB numa placa de 6 GB -- e o que sobra nao
    # paga as ativacoes. Quando aperta, o PyTorch recicla blocos a cada passo e
    # o treino fica 4x mais lento com a GPU marcando 100%. Sem esta linha isso
    # so aparecia como lentidao inexplicada.
    if dev.type == "cuda":
        bytes_dados = sum(t.nelement() * t.element_size()
                          for d in (X, Y) for t in d.values())
        livre_v, total_v = torch.cuda.mem_get_info()
        onde = "na VRAM" if X["treino"].is_cuda else "na RAM (fora da VRAM)"
        print("[vram] dados %.2f GB %s | reservado na GPU %.2f | livre %.2f de %.2f GB"
              % (bytes_dados / 2**30, onde, torch.cuda.memory_reserved() / 2**30,
                 livre_v / 2**30, total_v / 2**30), flush=True)
        if livre_v / 2**30 < 1.5:
            print("[vram] AVISO: menos de 1,5 GB livres para ativacoes e gradientes. "
                  "Espere lentidao por reciclagem de blocos. Baixe --lote, ou "
                  "--base, ou use um dataset com menos patches.", flush=True)

    BORDA_RAIO, BORDA_ALFA = args.borda_raio, args.borda_alfa
    faixa_val = None
    if args.borda_raio > 0:
        # calculada UMA vez: a fronteira vem do rotulo, que nao muda.
        faixa_val = torch.cat([faixa_fronteira(Y["validacao"][i:i + 32].to(dev),
                                               args.borda_raio).cpu()
                               for i in range(0, len(Y["validacao"]), 32)])
        fr = float(faixa_val.float().mean())
        res = float(json.load(io.open(os.path.join(args.dataset, "receita.json"),
                                      encoding="utf-8"))["parametros"]["resolucao"])
        print("[borda] faixa de %d px (%.1f m) = %.1f%% dos pixels de validacao"
              " | peso da CE ali: %.1fx"
              % (args.borda_raio, args.borda_raio * res, 100 * fr,
                 1 + args.borda_alfa), flush=True)

    if args.peso_classe:
        if len(args.peso_classe) != N:
            raise SystemExit("--peso-classe exige %d valores, um por classe (%s); "
                             "recebeu %d" % (N, ", ".join(dados.CLASS_NOMES),
                                             len(args.peso_classe)))
        PESO = torch.tensor(args.peso_classe, dtype=torch.float32, device=dev)
        print("[peso-classe] CE com %s | Dice segue sem peso"
              % ", ".join("%s=%.2f" % (nm, w) for nm, w
                          in zip(dados.CLASS_NOMES, args.peso_classe)), flush=True)

    pesos = None
    if args.reamostrar:
        pesos, frac_tr, alvo = pesos_reamostragem(Y["treino"], cenas["treino"], N,
                                                  args.reamostrar_k)
        relatar_reamostragem(pesos, frac_tr, cenas["treino"], N,
                             args.reamostrar_k, alvo)

    modelo = build_model(args.architecture, X["treino"].shape[1], N, args.base,
                         args.attention_input, args.attention_encoder,
                         args.attention_decoder, args.attention_nodes,
                         args.attention_input_mode).to(dev)
    if args.inicializar:
        ck = torch.load(args.inicializar, map_location="cpu", weights_only=False)
        sd = normalized_state_dict(ck["state_dict"])
        estado = modelo.state_dict()
        filtrado = {k: v for k, v in sd.items() if k in estado and estado[k].shape == v.shape}
        n_carregadas = len(filtrado)
        n_puladas = len(sd) - n_carregadas
        if n_carregadas:
            modelo.load_state_dict(filtrado, strict=False)
        print("[warm-start] %s | %d/%d tensores carregados, %d pulados por shape "
              "(cabeça nova p/ %d classes)" % (args.inicializar, n_carregadas, len(sd),
                                               n_puladas, N), flush=True)
    n_params = sum(p.numel() for p in modelo.parameters())
    print("[modelo] %s base=%d | %.1fM params" % (args.architecture, args.base, n_params / 1e6), flush=True)

    grupos = []
    for nome, p in modelo.named_parameters():
        sem_wd = p.ndim <= 1
        grupos.append({"params": p, "weight_decay": 0.0 if sem_wd else args.wd})
    opt = torch.optim.AdamW(grupos, lr=args.lr)
    if args.amostras_por_epoca is not None and (not args.reamostrar or
                                               args.amostras_por_epoca < 1):
        raise SystemExit("--amostras-por-epoca exige --reamostrar e valor positivo")
    amostras_ep = args.amostras_por_epoca or len(X["treino"])
    passos_ep = math.ceil(amostras_ep / args.lote)
    print("[amostras] %d sorteios por epoca | %d passos | %d patches unicos"
          % (amostras_ep, passos_ep, len(X["treino"])), flush=True)
    total, aquec = args.epocas * passos_ep, args.aquecimento * passos_ep

    H, W = X["treino"].shape[2:]
    interior = torch.zeros(H, W, dtype=torch.bool, device=dev)
    interior[args.margem:H - args.margem, args.margem:W - args.margem] = True

    # Sem gradiente a inferência cabe num lote maior que o do treino, mas o 4x
    # NÃO cabe nesta GPU com U-Net++: os skips densos concatenam 160 canais na
    # resolução cheia, e 64 x 160 x 256 x 256 x 4 B = 2,5 GB num tensor só --
    # exatamente a alocação que o alocador reportou não conseguir, e exatamente
    # o `x03` onde o `device not ready` matou o epoch 2 do v19a. Daí --lote-val.
    lote_inf = args.lote_val if args.lote_val else max(1, args.lote * 4)
    historico, melhor, melhor_ep, estado = [], -1.0, 0, None
    ep_inicio, t_acumulado = 0, 0.0
    cam_ultimo = os.path.join(saida, "ultimo.pth")
    if args.resume:
        if not os.path.exists(cam_ultimo):
            raise SystemExit("--resume: %s não existe" % cam_ultimo)
        ck = torch.load(cam_ultimo, map_location="cpu", weights_only=False)
        previous_args = ck.get("args", {})
        if args.attention_input and previous_args.get("attention_input_mode", "sigmoid") != args.attention_input_mode:
            raise SystemExit("--resume: input attention mode differs from checkpoint; "
                             "pass --attention-input-mode %s" %
                             previous_args.get("attention_input_mode", "sigmoid"))
        modelo.load_state_dict(ck["state_dict"])
        # `opt` nulo = checkpoint remendado de um treino anterior ao --resume (os
        # momentos do AdamW se refazem em algumas dezenas de passos, mas o
        # primeiro epoch depois disso não é idêntico ao de um treino contínuo).
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        else:
            print("[resume] sem estado do otimizador: os momentos do AdamW "
                  "recomeçam do zero", flush=True)
        historico, melhor, melhor_ep = ck["historico"], ck["melhor"], ck["melhor_ep"]
        estado, ep_inicio, t_acumulado = ck["estado"], ck["epoca"], ck["tempo_s"]
        if ck.get("rng"):
            rng.bit_generator.state = ck["rng"]
        print("[resume] %s | retomando no epoch %d | melhor até aqui: epoch %d "
              "(val mIoU=%.4f)" % (cam_ultimo, ep_inicio + 1, melhor_ep, melhor),
              flush=True)
    # O `metricas()` devolve IoU/recall/precisão por classe a cada epoch; sem uma
    # coluna por classe aqui, só o agregado sobrevive, e o agregado esconde
    # justamente a classe rara (ver o `por_cena` do RF e do XGBoost no v18).
    campos = (["epoca", "lr", "loss_treino", "val_oa", "val_miou"]
              + (["val_oa_borda", "val_miou_borda"] if faixa_val is not None else [])
              + ["val_iou_%d" % c for c in range(N)]
              + ["tempo_s"])
    livre = c_livre_gb()
    if livre < args.min_livre_c_gb:
        raise SystemExit("[trava] C: com %.2f GB livres (< %.1f GB): não vou começar"
                         % (livre, args.min_livre_c_gb))
    print("[trava] C: com %.2f GB livres; para sozinho se cair abaixo de %.1f GB"
          % (livre, args.min_livre_c_gb), flush=True)
    t_inicio = time.monotonic()
    for ep in range(ep_inicio, args.epocas):
        modelo.train()
        if pesos is None:
            perm = rng.permutation(len(X["treino"]))
        else:
            # com reposicao: alguns patches saem 2x no epoch, outros nenhuma. O
            # --amostras-por-epoca permite igualar passos entre datasets.
            perm = rng.choice(len(X["treino"]), size=amostras_ep,
                              replace=True, p=pesos)
        soma, n, t0 = 0.0, 0, time.monotonic()
        t_bloco, soma_bloco, n_bloco = time.monotonic(), 0.0, 0
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for b in range(passos_ep):
            if b % 20 == 0 and c_livre_gb() < args.min_livre_c_gb:
                raise SystemExit("[trava] C: com %.2f GB livres (< %.1f GB) na epoca %d: parando. "
                                 "O melhor modelo até aqui (epoca %d) está em modelo_melhor.pth."
                                 % (c_livre_gb(), args.min_livre_c_gb, ep + 1, melhor_ep))
            idx = torch.from_numpy(perm[b * args.lote:(b + 1) * args.lote])
            if X["treino"].is_cuda:
                idx = idx.to(dev)
            xb, yb = X["treino"][idx], Y["treino"][idx]
            if xb.device != dev:                      # --dados-na-ram: so o lote sobe
                xb, yb = xb.to(dev), yb.to(dev)
            x, y = aumentar(xb, yb, rng, args.ganho_pan, args.offset_pan,
                            args.escala_jitter, args.cutout, args.cutout_frac)
            lr = args.lr
            if total:
                if ep * passos_ep + b < aquec:
                    lr = args.lr * (ep * passos_ep + b + 1) / aquec
                else:
                    t = (ep * passos_ep + b - aquec) / max(1, total - aquec)
                    lr = args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = perda(modelo(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 3.0)
            opt.step()
            soma += float(loss.detach()) * len(idx)
            n += len(idx)
            soma_bloco += float(loss.detach()) * len(idx)
            n_bloco += len(idx)
            if args.log_cada and (b + 1) % args.log_cada == 0:
                dt = time.monotonic() - t_bloco
                vram = ""
                if dev.type == "cuda":
                    vram = " | VRAM %.2f/%.2f GB pico %.2f" % (
                        torch.cuda.memory_allocated() / 2**30,
                        torch.cuda.memory_reserved() / 2**30,
                        torch.cuda.max_memory_allocated() / 2**30)
                print("  passo %4d/%d | loss=%.4f | %.2f passos/s %.1f patches/s"
                      " | faltam %s%s"
                      % (b + 1, passos_ep, soma_bloco / max(n_bloco, 1),
                         args.log_cada / max(dt, 1e-9),
                         n_bloco / max(dt, 1e-9),
                         _hms((passos_ep - b - 1) * dt / max(args.log_cada, 1)),
                         vram), flush=True)
                t_bloco, soma_bloco, n_bloco = time.monotonic(), 0.0, 0

        pred_val = prever(modelo, X["validacao"], lote_inf)
        m = metricas(matriz(pred_val, Y["validacao"]))
        # mIoU SO na faixa de fronteira: e onde mora o erro, e no agregado ele
        # some sob os 75% de interior que ja estao em 0,98.
        mb = (metricas(matriz(pred_val, Y["validacao"], faixa_val))
              if faixa_val is not None else None)
        reg = {"epoca": ep + 1, "lr": lr, "loss_treino": soma / n, "val_oa": m["oa"],
               "val_miou": m["miou"], "tempo_s": round(time.monotonic() - t0, 2)}
        if mb is not None:
            reg["val_miou_borda"] = mb["miou"]
            reg["val_oa_borda"] = mb["oa"]
        reg.update(("val_iou_%d" % c, m["iou"][c]) for c in range(N))
        historico.append(reg)
        marca = ""
        if m["miou"] > melhor:
            melhor, melhor_ep, marca = m["miou"], ep + 1, "  *"
            estado = copy.deepcopy({k: v.cpu() for k, v in modelo.state_dict().items()})
            torch.save({"state_dict": estado, "epoca": ep + 1, "val_miou": melhor,
                        "args": vars(args)}, os.path.join(saida, "modelo_melhor.pth"))
        extra = ("" if mb is None else
                 " | BORDA OA=%.4f mIoU=%.4f" % (mb["oa"], mb["miou"]))
        print("epoca %3d | lr=%.2e | loss=%.4f | val OA=%.4f mIoU=%.4f | IoU %s%s | %.1fs%s"
              % (ep + 1, lr, reg["loss_treino"], m["oa"], m["miou"],
                 " ".join("%s=%.3f" % (dados.CLASS_NOMES[c][:3], m["iou"][c])
                          for c in range(N)),
                 extra, reg["tempo_s"], marca),
              flush=True)
        with open(os.path.join(saida, "log.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=campos)
            wr.writeheader()
            wr.writerows(historico)
        # Tudo que o --resume precisa para continuar como se nada tivesse
        # acontecido. O LR vem do cronograma por passo global (ep*passos_ep+b),
        # então basta retomar no epoch certo para a curva do cosseno seguir.
        torch.save({"state_dict": {k: v.cpu() for k, v in modelo.state_dict().items()},
                    "opt": opt.state_dict(), "historico": historico,
                    "melhor": melhor, "melhor_ep": melhor_ep, "estado": estado,
                    "epoca": ep + 1, "tempo_s": t_acumulado + time.monotonic() - t_inicio,
                    "rng": rng.bit_generator.state, "args": vars(args)}, cam_ultimo)
        if ep + 1 - melhor_ep >= args.paciencia:
            print("[parada] %d epochs sem melhorar (melhor epoca %d)" % (args.paciencia, melhor_ep))
            break
    t_treino = t_acumulado + time.monotonic() - t_inicio

    modelo.load_state_dict(estado)
    print("\n[final] modelo da epoca %d (val mIoU=%.4f) | treino %.0fs" % (melhor_ep, melhor, t_treino))
    res = {}
    for s in ("validacao", "teste"):
        pred = prever(modelo, X[s], lote_inf)
        res[s] = {}
        for escopo, masc in (("inteiro", None), ("interior", interior)):
            mk = None if masc is None else masc.expand_as(Y[s])
            imprimir = s == "teste"
            r = metricas(matriz(pred, Y[s], mk),
                         "%s | %s | geral" % (s.upper(), escopo) if imprimir else None)
            r["por_cena"] = {}
            for c in dict.fromkeys(cenas[s]):
                sel = torch.from_numpy(np.flatnonzero(cenas[s] == c))
                r["por_cena"][c] = metricas(
                    matriz(_ix(pred, sel), _ix(Y[s], sel),
                           None if mk is None else _ix(mk, sel)),
                    "%s | %s | cena %s" % (s.upper(), escopo, c)
                    if imprimir and escopo == "interior" else None)
            res[s][escopo] = r
        visualizar(brutos[s][0], brutos[s][1], pred.cpu().numpy(), nomes[s], cenas[s],
                   args.margem, os.path.join(saida, "visualizacao", s + ".png"), args.n_png,
                   {"validacao": "VALIDAÇÃO", "teste": "TESTE"}[s])

    with open(os.path.join(saida, "relatorio.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "params": n_params,
                   "melhor_epoca": melhor_ep, "epocas_rodadas": len(historico),
                   "tempo_treino_s": round(t_treino, 1), "classes": dados.CLASS_NOMES,
                   "validacao": res["validacao"], "teste": res["teste"]},
                  f, indent=2, ensure_ascii=False)
    print("OK -> %s" % saida)


if __name__ == "__main__":
    main()
