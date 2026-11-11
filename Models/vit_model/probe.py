"""Linear probe: mede se o encoder aprendeu algo útil para as nossas classes.

Encoder congelado -> features de cada token (1 token = célula de 16x16 px).
Rótulo da célula = classe majoritária dos pixels rotulados (célula com menos
de `min_rotulado` rotulada é ignorada). Um classificador linear é treinado nos
tokens do split de treino e avaliado nos de validação (geral e por cena).

Rodar no epoch 0 (pesos DINO puros) e ao longo do treino do MAE mostra se o
pré-treino está melhorando a representação.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dados

N = dados.N_CLASSES


def rotulos_celula(rotulos, p, min_rotulado=0.5):
    """(N,H,W) uint8 (IGNORE = sem rótulo) -> (N, L) int64, -1 = ignorada."""
    t = torch.from_numpy(rotulos).long()
    B, H, W = t.shape
    cel = t.reshape(B, H // p, p, W // p, p).permute(0, 1, 3, 2, 4).reshape(B, -1, p * p)
    cont = F.one_hot(cel.clamp(max=N), N + 1)[..., :N].sum(dim=2)   # (B, L, N)
    y = cont.argmax(dim=-1)
    y[cont.sum(dim=-1) < min_rotulado * p * p] = -1
    return y


@torch.no_grad()
def extrair(mae, imgs, lote=32):
    """imgs (N,C,H,W) na GPU -> tokens (N, L, D) float32."""
    mae.eval()
    out = []
    for i in range(0, len(imgs), lote):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(mae.tokens(imgs[i:i + lote].float()).float())
    return torch.cat(out)


def metricas(pred, y):
    cm = torch.bincount(y * N + pred, minlength=N * N).reshape(N, N).double()
    tp, ref, prd = cm.diag(), cm.sum(1), cm.sum(0)
    uniao = ref + prd - tp
    iou = torch.where(uniao > 0, tp / uniao.clamp(min=1), torch.full_like(tp, float("nan")))
    return {"oa": float(tp.sum() / cm.sum().clamp(min=1)),
            "miou": float(np.nanmean(iou.cpu().numpy())),
            "iou": [None if np.isnan(v) else float(v) for v in iou.cpu().numpy()],
            "n": int(cm.sum())}


def rodar(mae, imgs_tr, rot_tr, imgs_va, rot_va, cenas_va, passos=500, lr=1e-2, seed=0):
    """-> dict com métricas de validação (geral e por cena)."""
    dev = imgs_tr.device
    ytr = rotulos_celula(rot_tr, mae.p).to(dev).reshape(-1)
    yva = rotulos_celula(rot_va, mae.p).to(dev)
    Xtr = extrair(mae, imgs_tr)
    Xva = extrair(mae, imgs_va)
    L, D = Xtr.shape[1], Xtr.shape[2]
    cena_tok = np.repeat(np.asarray(cenas_va), L)
    Xtr, Xva, yva = Xtr.reshape(-1, D), Xva.reshape(-1, D), yva.reshape(-1)

    ok = ytr >= 0
    Xtr, ytr = Xtr[ok], ytr[ok]
    media, desvio = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - media) / desvio, (Xva - media) / desvio

    cont = torch.bincount(ytr, minlength=N).float()
    peso = torch.where(cont > 0, cont.sum() / (N * cont.clamp(min=1)), torch.zeros_like(cont))
    torch.manual_seed(seed)
    lin = nn.Linear(D, N).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(passos):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(lin(Xtr), ytr, weight=peso).backward()
        opt.step()

    with torch.no_grad():
        pred = lin(Xva).argmax(1)
    ok = (yva >= 0).cpu().numpy()
    res = metricas(pred[yva >= 0], yva[yva >= 0])
    res["por_cena"] = {}
    for c in dict.fromkeys(cenas_va):
        m = torch.from_numpy(ok & (cena_tok == c)).to(dev)
        res["por_cena"][c] = metricas(pred[m], yva[m])
    return res
