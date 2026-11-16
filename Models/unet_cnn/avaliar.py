#!/usr/bin/env python3
"""Avaliação completa de um run da CNN (unet_cnn/runs/<nome>/modelo_melhor.pth).

Gêmeo do xgboost/avaliar.py, pelos mesmos dois motivos:

1. O `train.py` só mede validação e teste, no fim do treino, e não há como pedir
   o conjunto de treino -- que é o que mostra o quanto o modelo decorou. Aqui a
   matriz de confusão é acumulada lote a lote, então a memória é O(--lote
   patches) e dá para medir os 4304 patches de treino (~219 M px no interior).
2. O relatório do treino se perde se o processo morrer antes do fim, e nesta
   máquina isso aconteceu quatro vezes. Com o modelo_melhor.pth em disco,
   reavaliar não exige retreinar.

Mede nos dois escopos do train.py (`inteiro` e `interior`), para o
relatorio_completo.json poder ser comparado com o relatorio.json do próprio
run. Comparações entre famílias usam o `interior` (ver CLAUDE.md).

Uso (no WSL, a partir de `Models/`):
  ../Data/.venv/bin/python unet_cnn/avaliar.py --nome unetpp_v18 \\
      --dataset ../Data/dados/datasets/spot5_refinado_v18
"""

import argparse
import json
import os
import time

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch)
import numpy as np
import torch

import dados
from unet import UNet, UNetPP

BASE = os.path.dirname(os.path.abspath(__file__))
N = dados.N_CLASSES


def _lista(v):
    return [None if np.isnan(a) else float(a) for a in v]


def metricas(rotulo, cm):
    """Mesmo schema do relatorio.json do train.py, a partir da matriz já
    acumulada -- é o que permite não guardar predição nem rótulo inteiros."""
    tp = np.diag(cm).astype(np.float64)
    ref, prd = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall, prec = tp / ref, tp / prd
        f1 = 2 * tp / (ref + prd)
        iou = tp / (ref + prd - tp)
    oa = tp.sum() / max(1, cm.sum())
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
    fw = ref / max(1, cm.sum())
    fwiou = float(np.nansum(np.where(np.isfinite(iou), iou, 0.0) * fw))
    linhas = ["%s  OA=%.4f  mIoU=%.4f  (%d pixels)" % (rotulo, oa, miou, cm.sum()),
              "  %2s  %-26s  %7s  %7s  %7s  %7s  %6s"
              % ("cl", "nome", "recall", "precis", "F1", "IoU", "ref%")]
    for c in range(N):
        linhas.append("  %2d  %-26s  %7.4f  %7.4f  %7.4f  %7.4f  %5.1f%%"
                      % (c, dados.CLASS_NOMES[c], recall[c], prec[c], f1[c], iou[c],
                         100 * ref[c] / max(1, cm.sum())))
    print("\n".join(linhas), flush=True)
    return {"rotulo": rotulo, "oa": float(oa), "miou": miou,
            "miou_presentes": miou_pres, "fwiou": fwiou,
            "acuracia": _lista(recall), "precisao": _lista(prec),
            "f1": _lista(f1), "iou": _lista(iou),
            "matriz_confusao": cm.tolist(), "n_pixels": int(cm.sum())}


@torch.no_grad()
def confusao(modelo, dataset, nomes, cena_de, margem, lote, dev):
    """-> {escopo: (cm, {cena: cm})} para escopo em ('inteiro', 'interior').

    Os patches entram em lotes de `lote`; de cada lote só sobrevivem matrizes
    N x N, então o pico de memória é o lote, não o split.
    """
    escopos = ("inteiro", "interior")
    cenas = list(dict.fromkeys(cena_de[n] for n in nomes))
    cm = {e: np.zeros((N, N), np.int64) for e in escopos}
    por_cena = {e: {c: np.zeros((N, N), np.int64) for c in cenas} for e in escopos}
    interior = None
    for i in range(0, len(nomes), lote):
        bloco = nomes[i:i + lote]
        imgs, rots = dados.carregar(dataset, bloco)
        x = torch.from_numpy(imgs).to(dev)
        y = torch.from_numpy(rots).to(dev).long()
        if interior is None:
            h, w = y.shape[1:]
            interior = torch.zeros(h, w, dtype=torch.bool, device=dev)
            interior[margem:h - margem, margem:w - margem] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = modelo(x.float()).argmax(1)
        for k in range(len(bloco)):
            valido = y[k] != dados.IGNORE
            for e in escopos:
                v = valido if e == "inteiro" else (valido & interior)
                if not bool(v.any()):
                    continue
                c = torch.bincount(y[k][v] * N + pred[k][v],
                                   minlength=N * N).reshape(N, N).cpu().numpy()
                cm[e] += c
                por_cena[e][cena_de[bloco[k]]] += c
        del imgs, rots, x, y, pred
        torch.cuda.empty_cache()
    return {e: (cm[e], por_cena[e]) for e in escopos}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", required=True, help="pasta em unet_cnn/runs/")
    ap.add_argument("--dataset", default=dados.DATASET_PADRAO)
    ap.add_argument("--splits", nargs="+", default=["treino", "validacao", "teste"])
    ap.add_argument("--margem", type=int, default=15,
                    help="px da borda fora do escopo 'interior' (o mesmo do treino)")
    ap.add_argument("--lote", type=int, default=16,
                    help="patches por lote; é ele que define o pico de memória")
    args = ap.parse_args()

    global N
    dados.configurar(args.dataset)
    N = dados.N_CLASSES
    print("[classes] %d: %s" % (N, ", ".join(dados.CLASS_NOMES)))

    dev = ambiente.exigir_gpu()
    saida = os.path.join(BASE, "runs", args.nome)
    cam_modelo = os.path.join(saida, "modelo_melhor.pth")
    if not os.path.exists(cam_modelo):
        raise SystemExit("%s não existe" % cam_modelo)
    ck = torch.load(cam_modelo, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    arq = a.get("arquitetura", "unetpp")
    base = a.get("base", 32)
    sd = ck["state_dict"]
    canais = next(v for k, v in sd.items() if k.endswith("weight") and v.ndim == 4).shape[1]
    modelo = (UNet if arq == "unet" else UNetPP)(
        in_chans=canais, n_classes=N, base=base).to(dev)
    modelo.load_state_dict(sd)
    modelo.eval()
    print("[modelo] %s | %s base=%d | epoca %s (val mIoU=%.4f)"
          % (cam_modelo, arq, base, ck.get("epoca", "?"), ck.get("val_miou", float("nan"))),
          flush=True)

    cena_de = dados.cenas(args.dataset)
    cam = os.path.join(saida, "relatorio_completo.json")
    rel = {"run": args.nome, "dataset": os.path.abspath(args.dataset),
           "margem": args.margem, "classes": dados.CLASS_NOMES,
           "epoca": ck.get("epoca"), "val_miou_treino": ck.get("val_miou")}
    # Aproveita os splits já medidos numa execução anterior: sem isso, rodar de
    # novo só o treino apagaria validação e teste do arquivo.
    if os.path.exists(cam):
        with open(cam, encoding="utf-8") as f:
            anterior = json.load(f)
        for s in ("treino", "validacao", "teste"):
            if s in anterior and s not in args.splits:
                rel[s] = anterior[s]
                print("[mantido] %s do relatório anterior" % s, flush=True)

    for s in args.splits:
        nomes = dados.ler_split(args.dataset, s)
        t0 = time.time()
        res = confusao(modelo, args.dataset, nomes, cena_de, args.margem,
                       args.lote, dev)
        print("\n[%s] %d patches | %d px (interior) | %.0fs"
              % (s, len(nomes), res["interior"][0].sum(), time.time() - t0), flush=True)
        rel[s] = {}
        for e, (cm, por_cena) in res.items():
            rel[s][e] = metricas("%s | %s" % (s, e), cm)
            rel[s][e]["por_cena"] = {c: metricas("  %s | %s / %s" % (s, e, c), m)
                                     for c, m in sorted(por_cena.items()) if m.sum()}
        # Grava a cada split: o treino sozinho leva vários minutos, e um
        # relatório escrito só no fim já se perdeu inteiro aqui.
        with open(cam, "w", encoding="utf-8") as f:
            json.dump(rel, f, indent=2, ensure_ascii=False)
        print("-> %s (%s)" % (cam, ", ".join(x for x in ("treino", "validacao", "teste")
                                             if x in rel)), flush=True)


if __name__ == "__main__":
    main()
