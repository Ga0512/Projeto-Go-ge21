#!/usr/bin/env python3
"""Avaliação completa de um run de árvore: XGBoost (modelo.ubj) ou Random
Forest do rf_patches.py (modelo.joblib), detectado pelo arquivo presente em
xgboost/runs/<nome>/.

Existe por dois motivos:

1. O `train.py` avalia validação/teste com todos os pixels do interior, o que
   custa ~2,9 GB de features por split e já matou treinos nesta máquina. O
   `--max-amostras-aval` contorna isso amostrando, mas aí o relatório sai
   marcado `(amostrado)` e não é o mesmo conjunto de pixels em que o RF e a
   U-Net++ foram medidos. Aqui a matriz de confusão é **acumulada lote a lote**,
   então a memória é O(--lote patches), não O(split): dá para medir todos os
   pixels de qualquer split, treino inclusive (~198 M px, que como matriz densa
   seriam 16 GB).
2. O treino não é retomável. Com o modelo.ubj em disco, reavaliar não exige
   retreinar.

As métricas de treino aqui NÃO são as do `relatorio.json` do treino: lá o nível
`treino` é medido na amostra de ~6 M pixels que ajustou o modelo (por isso
`TREINO (amostrado)`); aqui são todos os pixels rotulados do interior dos
patches de treino.

Saída: runs/<nome>/relatorio_completo.json, mesmo schema do relatorio.json
(oa, miou, acuracia/precisao/f1/iou por classe, matriz_confusao, por_cena),
com `n_pixels` provando quantos pixels entraram.

Uso (no WSL, a partir de `Models/`):
  ../Data/.venv/bin/python xgboost/avaliar.py --nome xgb_spot5_v18 \\
      --dataset ../Data/dados/datasets/spot5_refinado_v18
  ../Data/.venv/bin/python xgboost/avaliar.py --nome rf_v18_d12 \\
      --dataset ../Data/dados/datasets/spot5_refinado_v18
"""

import argparse
import json
import os
import time

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch/xgboost)
import numpy as np
import torch
import xgboost as xgb

import dados
import features

BASE = os.path.dirname(os.path.abspath(__file__))
N = dados.N_CLASSES


def _lista(v):
    return [None if np.isnan(a) else float(a) for a in v]


def metricas(rotulo, cm):
    """Mesmo schema de train.py:metricas(), mas a partir da matriz de confusão
    já acumulada -- é o que permite não guardar predição nem rótulo inteiros."""
    tp = np.diag(cm).astype(np.float64)
    ref, prd = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall, prec = tp / ref, tp / prd
        f1 = 2 * tp / (ref + prd)
        iou = tp / (ref + prd - tp)
    oa = tp.sum() / max(1, cm.sum())
    miou = float(np.nanmean(iou))
    linhas = ["%s  OA=%.4f  mIoU=%.4f  (%d pixels)" % (rotulo, oa, miou, cm.sum()),
              "  %2s  %-26s  %7s  %7s  %7s  %7s  %6s"
              % ("cl", "nome", "recall", "precis", "F1", "IoU", "ref%")]
    for c in range(N):
        linhas.append("  %2d  %-26s  %7.4f  %7.4f  %7.4f  %7.4f  %5.1f%%"
                      % (c, dados.CLASS_NOMES[c], recall[c], prec[c], f1[c], iou[c],
                         100 * ref[c] / max(1, cm.sum())))
    print("\n".join(linhas), flush=True)
    return {"rotulo": rotulo, "oa": float(oa), "miou": miou,
            "acuracia": _lista(recall), "precisao": _lista(prec),
            "f1": _lista(f1), "iou": _lista(iou),
            "matriz_confusao": cm.tolist(), "n_pixels": int(cm.sum())}


def confusao(predizer, usa_features, dataset, nomes, cena_de, janelas, margem,
             lote, dev):
    """-> (cm global, {cena: cm}) sobre TODOS os pixels rotulados do interior.

    Os patches entram em lotes de `lote`; de cada lote só sobrevive a matriz
    de confusão (N x N inteiros), então o pico de memória é o lote, não o split.
    """
    cm = np.zeros((N, N), np.int64)
    por_cena = {c: np.zeros((N, N), np.int64) for c in dict.fromkeys(cena_de[n] for n in nomes)}
    interior = None
    for i in range(0, len(nomes), lote):
        bloco = nomes[i:i + lote]
        imgs, rots = dados.carregar(dataset, bloco)
        if interior is None:
            _, _, h, w = imgs.shape
            interior = torch.zeros(h, w, dtype=torch.bool, device=dev)
            interior[margem:h - margem, margem:w - margem] = True
        x = torch.from_numpy(imgs).to(dev)
        y = torch.from_numpy(rots).to(dev)
        # O RF do rf_patches.py recebe as 3 bandas do pixel e nada mais; o
        # XGBoost recebe as 21 features derivadas. A margem é a mesma nos dois,
        # para o conjunto avaliado ser idêntico (ver CLAUDE.md).
        f = (features.calcular(x, janelas) if usa_features else x).permute(0, 2, 3, 1)
        f = f.reshape(f.shape[0], -1, f.shape[-1])
        for k in range(f.shape[0]):
            sel = torch.nonzero(((y[k] != dados.IGNORE) & interior).reshape(-1)).squeeze(1)
            if not len(sel):
                continue
            X = f[k, sel].cpu().numpy()
            alvo = y[k].reshape(-1)[sel].cpu().numpy()
            pred = predizer(X)
            c = np.bincount(alvo.astype(np.int64) * N + pred,
                            minlength=N * N).reshape(N, N)
            cm += c
            por_cena[cena_de[bloco[k]]] += c
        del imgs, rots, x, y, f
        torch.cuda.empty_cache()
    return cm, por_cena


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", required=True, help="pasta em xgboost/runs/")
    ap.add_argument("--dataset", default=dados.DATASET_PADRAO)
    ap.add_argument("--splits", nargs="+", default=["treino", "validacao", "teste"])
    ap.add_argument("--janelas", type=int, nargs="+", default=[5, 15, 31])
    ap.add_argument("--margem", type=int, default=-1,
                    help="-1 = maior janela // 2 (o mesmo do treino)")
    ap.add_argument("--lote", type=int, default=16,
                    help="patches por lote; é ele que define o pico de memória")
    args = ap.parse_args()

    global N
    dados.configurar(args.dataset)
    N = dados.N_CLASSES
    print("[classes] %d: %s" % (N, ", ".join(dados.CLASS_NOMES)))
    margem = max(args.janelas) // 2 if args.margem < 0 else args.margem

    dev = ambiente.exigir_gpu()
    saida = os.path.join(BASE, "runs", args.nome)
    cam_ubj = os.path.join(saida, "modelo.ubj")
    cam_joblib = os.path.join(saida, "modelo.joblib")
    if os.path.exists(cam_ubj):
        bst = xgb.Booster()
        bst.load_model(cam_ubj)
        # O XGBoost avisa que cai no caminho de DMatrix porque as features
        # chegam em numpy (CPU) e o booster está em "cuda". O aviso é real, mas
        # trocar o booster para "cpu" custou 6,3x no tempo (0,83 s/patch contra
        # 0,13 s), e a cópia extra é de um lote de --lote patches, não do
        # split: fica em "cuda".
        bst.set_param({"device": "cuda"})
        usa_features = True
        def predizer(X):
            return bst.inplace_predict(X).argmax(1).astype(np.uint8)
        print("[modelo] %s | XGBoost | %d árvores"
              % (cam_ubj, bst.num_boosted_rounds()), flush=True)
    elif os.path.exists(cam_joblib):
        import joblib
        rf = joblib.load(cam_joblib)
        usa_features = False          # o RF usa as 3 bandas cruas
        def predizer(X):
            return rf.predict(X).astype(np.uint8)
        print("[modelo] %s | RandomForest | %d árvores | profundidade %d"
              % (cam_joblib, len(rf.estimators_),
                 max(e.tree_.max_depth for e in rf.estimators_)), flush=True)
    else:
        raise SystemExit("nem modelo.ubj nem modelo.joblib em %s" % saida)

    cena_de = dados.cenas(args.dataset)
    cam = os.path.join(saida, "relatorio_completo.json")
    # Aproveita os splits já medidos numa execução anterior: sem isso, rodar de
    # novo só o treino apagaria validação e teste do arquivo.
    rel = {"run": args.nome, "dataset": os.path.abspath(args.dataset),
           "margem": margem, "classes": dados.CLASS_NOMES,
           "janelas": args.janelas, "amostrado": False}
    if os.path.exists(cam):
        with open(cam, encoding="utf-8") as f:
            anterior = json.load(f)
        for s in ("treino", "validacao", "teste"):
            if s in anterior and s not in args.splits:
                rel[s] = anterior[s]
                print("[mantido] %s do relatório anterior (%d px)"
                      % (s, anterior[s]["n_pixels"]), flush=True)
    for s in args.splits:
        nomes = dados.ler_split(args.dataset, s)
        t0 = time.time()
        cm, por_cena = confusao(predizer, usa_features, args.dataset, nomes,
                                cena_de, args.janelas,
                                margem, args.lote, dev)
        print("\n[%s] %d patches | %d px | %.0fs"
              % (s, len(nomes), cm.sum(), time.time() - t0), flush=True)
        rel[s] = metricas("%s (interior, margem %d)" % (s, margem), cm)
        rel[s]["por_cena"] = {c: metricas("  %s / %s" % (s, c), m)
                              for c, m in sorted(por_cena.items()) if m.sum()}

        # Grava a cada split: o treino sozinho leva ~10 min, e um relatório
        # escrito só no fim já se perdeu inteiro quando o processo morreu no
        # meio.
        with open(cam, "w", encoding="utf-8") as f:
            json.dump(rel, f, indent=2, ensure_ascii=False)
        print("-> %s (%s)" % (cam, ", ".join(x for x in args.splits if x in rel)),
              flush=True)


if __name__ == "__main__":
    main()
