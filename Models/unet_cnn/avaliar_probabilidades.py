#!/usr/bin/env python3
"""Mede ROC-AUC e average precision por classe no melhor checkpoint.

Para segmentacao semantica, cada pixel rotulado e uma observacao e cada classe
e avaliada um-contra-resto. Usa histogramas de probabilidades em lotes para
nao guardar dezenas de milhoes de scores na memoria. A aproximacao tem 4096
intervalos por padrao. Classes sem positivos ou sem negativos recebem null.

Salva unet_cnn/runs/<nome>/relatorio_probabilidades.json. O arquivo identifica
o dataset e os splits, evitando misturar avaliacoes com reguas diferentes.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python -u unet_cnn/avaliar_probabilidades.py \
    --nome unetpp_spot5_colinas_v32 \
    --dataset ../Data/dados/datasets/spot5_colinas_v32_7b
"""

import argparse
import json
from pathlib import Path

import ambiente  # noqa: F401: caches antes de torch/numpy
import numpy as np
import torch

import dados
from unet import model_from_checkpoint


BASE = Path(__file__).resolve().parent


def resumo_histograma(hist):
    """ROC-AUC e AP one-vs-rest a partir de hist[classe, pos/neg, bin]."""
    saida = []
    for classe in range(hist.shape[0]):
        pos = hist[classe, 1, ::-1].astype(np.float64)
        neg = hist[classe, 0, ::-1].astype(np.float64)
        n_pos, n_neg = pos.sum(), neg.sum()
        if not n_pos or not n_neg:
            saida.append({"auc_roc": None, "average_precision": None,
                          "positivos": int(n_pos), "negativos": int(n_neg)})
            continue
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        recall = np.r_[0.0, tp / n_pos]
        fpr = np.r_[0.0, fp / n_neg]
        precision = tp / np.maximum(tp + fp, 1.0)
        saida.append({"auc_roc": float(np.trapz(recall, fpr)),
                      "average_precision": float(np.sum(np.diff(recall) * precision)),
                      "positivos": int(n_pos), "negativos": int(n_neg)})
    return {"por_classe": {dados.CLASS_NOMES[i]: item for i, item in enumerate(saida)},
            "auc_roc_macro": float(np.mean([r["auc_roc"] for r in saida
                                             if r["auc_roc"] is not None]))
            if any(r["auc_roc"] is not None for r in saida) else None,
            "average_precision_macro": float(np.mean([r["average_precision"]
                                                        for r in saida
                                                        if r["average_precision"] is not None]))
            if any(r["average_precision"] is not None for r in saida) else None}


@torch.no_grad()
def medir(modelo, dataset, nomes, lote, bins, margem, dev):
    n = dados.N_CLASSES
    hist = {escopo: np.zeros((n, 2, bins), dtype=np.int64)
            for escopo in ("inteiro", "interior")}
    for inicio in range(0, len(nomes), lote):
        imgs, rots = dados.carregar(dataset, nomes[inicio:inicio + lote])
        x = torch.from_numpy(imgs).to(dev)
        y = torch.from_numpy(rots).to(dev).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prob = torch.softmax(modelo(x.float()).float(), dim=1)
        valido = y != dados.IGNORE
        interior = valido.clone()
        if margem:
            interior[:, :margem, :] = False
            interior[:, -margem:, :] = False
            interior[:, :, :margem] = False
            interior[:, :, -margem:] = False
        for escopo, mask in (("inteiro", valido), ("interior", interior)):
            rotulo = y[mask]
            for classe in range(n):
                indices = (prob[:, classe][mask].mul(bins).long()
                           .clamp_(min=0, max=bins - 1))
                contagens = torch.bincount(indices + bins * (rotulo == classe).long(),
                                           minlength=2 * bins)
                hist[escopo][classe] += contagens.cpu().numpy().reshape(2, bins)
        if (inicio // lote + 1) % 10 == 0 or inicio + lote >= len(nomes):
            print("[auc] %d/%d patches" % (min(inicio + lote, len(nomes)), len(nomes)),
                  flush=True)
    return {escopo: resumo_histograma(valores) for escopo, valores in hist.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nome", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["validacao", "teste"])
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--bins", type=int, default=4096)
    ap.add_argument("--margem", type=int, default=15)
    args = ap.parse_args()
    if args.lote < 1 or args.bins < 2 or args.margem < 0:
        ap.error("lote >= 1, bins >= 2 e margem >= 0")
    dados.configurar(args.dataset)
    dev = ambiente.exigir_gpu()
    pasta = BASE / "runs" / args.nome
    cam_modelo = pasta / "modelo_melhor.pth"
    ck = torch.load(cam_modelo, map_location="cpu", weights_only=False)
    modelo = model_from_checkpoint(ck, dados.N_CLASSES).to(dev).eval()
    rel = {"nome": args.nome, "dataset": str(Path(args.dataset).resolve()),
           "checkpoint": str(cam_modelo), "epoca": ck.get("epoca"),
           "metodo": "histograma de probabilidades; pixels rotulados; um-contra-resto",
           "bins": args.bins, "margem": args.margem, "splits": {}}
    for split in args.splits:
        nomes = dados.ler_split(args.dataset, split)
        rel["splits"][split] = medir(modelo, args.dataset, nomes, args.lote,
                                      args.bins, args.margem, dev)
        print("[%s] ROC-AUC macro=%.4f | AP macro=%.4f" %
              (split, rel["splits"][split]["inteiro"]["auc_roc_macro"],
               rel["splits"][split]["inteiro"]["average_precision_macro"]),
              flush=True)
    destino = pasta / "relatorio_probabilidades.json"
    destino.write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK -> " + str(destino), flush=True)


if __name__ == "__main__":
    main()
