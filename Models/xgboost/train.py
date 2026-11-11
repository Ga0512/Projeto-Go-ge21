#!/usr/bin/env python3
"""XGBoost por pixel (GPU) no dataset de patches SPOT5 refinado.

Cada pixel rotulado é uma amostra. Features (features.py, calculadas na GPU):
as 3 bandas + média/desvio local de cada banda em janelas multiescala.
Classes: CLASS_MAP de Data/scripts/modelos/config.py (0 = sem rótulo, ignorado).

- Treino: splits/treino.txt, até --max-amostras pixels sorteados (mesma cota
  por patch), com peso por classe (inverso da frequência).
- Parada antecipada: mlogloss em splits/validacao.txt (todos os pixels, ou
  até --max-amostras-aval sorteados -- o pico de RAM mora aqui).
- Teste: splits/teste.txt, só no final (nada é escolhido olhando o teste).
- Treino e avaliação usam só o interior dos patches (--margem da borda), onde
  as janelas das features cabem inteiras.

Saída em xgboost/runs/<nome>/: modelo.ubj (já cortado na melhor rodada),
relatorio.json (métricas gerais e por cena), curva.csv,
visualizacao/{validacao,teste}.png (metade dos patches de cada cena).

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python xgboost/train.py --nome xgb_spot5_v1
"""

import argparse
import csv
import gc
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


def extrair(imgs, rotulos, janelas, margem, por_patch, gerador, lote, dev):
    """-> X (n,F) float32, y (n,) uint8, patch (n,) int32: pixels rotulados do
    interior dos patches e o índice do patch de origem de cada um.
    por_patch > 0 sorteia no máximo esse número de pixels por patch."""
    _, _, h, w = imgs.shape
    interior = torch.zeros(h, w, dtype=torch.bool, device=dev)
    interior[margem:h - margem, margem:w - margem] = True
    Xs, ys, ps = [], [], []
    for i in range(0, len(imgs), lote):
        x = torch.from_numpy(imgs[i:i + lote]).to(dev)
        y = torch.from_numpy(rotulos[i:i + lote]).to(dev)
        f = features.calcular(x, janelas).permute(0, 2, 3, 1)
        f = f.reshape(f.shape[0], -1, f.shape[-1])
        for k in range(f.shape[0]):
            sel = torch.nonzero(((y[k] != dados.IGNORE) & interior).reshape(-1)).squeeze(1)
            if por_patch and len(sel) > por_patch:
                sel = sel[torch.randperm(len(sel), generator=gerador, device=dev)[:por_patch]]
            Xs.append(f[k, sel].cpu())
            ys.append(y[k].reshape(-1)[sel].cpu())
            ps.append(torch.full((len(sel),), i + k, dtype=torch.int32))
    return torch.cat(Xs).numpy(), torch.cat(ys).numpy(), torch.cat(ps).numpy()


def pesos_classe(y):
    cont = np.bincount(y, minlength=N).astype(np.float64)
    w = np.where(cont > 0, len(y) / (N * np.maximum(cont, 1)), 0.0)
    return w[y].astype(np.float32), w


def _lista(v):
    return [None if np.isnan(a) else float(a) for a in v]


def metricas(rotulo, pred, y):
    """OA, mIoU e recall/precisão/F1/IoU por classe (schema do relatorio de
    Data/scripts/modelos/xgb_lulc.py + miou e matriz de confusão)."""
    cm = np.bincount(y.astype(np.int64) * N + pred, minlength=N * N).reshape(N, N)
    tp = np.diag(cm).astype(np.float64)
    ref, prd = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = tp / ref
        prec = tp / prd
        f1 = 2 * tp / (ref + prd)
        iou = tp / (ref + prd - tp)
    oa = tp.sum() / cm.sum()
    miou = float(np.nanmean(iou))
    linhas = ["%s  OA=%.4f  mIoU=%.4f  (%d pixels)" % (rotulo, oa, miou, cm.sum()),
              "  %2s  %-26s  %7s  %7s  %7s  %7s  %6s"
              % ("cl", "nome", "recall", "precis", "F1", "IoU", "ref%")]
    for c in range(N):
        linhas.append("  %2d  %-26s  %7.4f  %7.4f  %7.4f  %7.4f  %5.1f%%"
                      % (c, dados.CLASS_NOMES[c], recall[c], prec[c], f1[c], iou[c],
                         100 * ref[c] / cm.sum()))
    print("\n".join(linhas), flush=True)
    return {"rotulo": rotulo, "oa": float(oa), "miou": miou,
            "acuracia": _lista(recall), "precisao": _lista(prec),
            "f1": _lista(f1), "iou": _lista(iou),
            "matriz_confusao": cm.tolist(), "n_pixels": int(cm.sum())}


def prever(bst, X, lote=1_000_000):
    """Classe prevista, em lotes: a matriz de probabilidades inteira de um split
    grande (15 M pixels x 4) estourava a memória do WSL."""
    out = np.empty(len(X), np.uint8)
    for i in range(0, len(X), lote):
        out[i:i + lote] = bst.inplace_predict(X[i:i + lote]).argmax(1)
    return out


def _linha(axs, bst, img, ref, rotulo, janelas, margem, dev, cmap, cmap_erro):
    """Uma linha da figura: PAN | referência | predição | erro."""
    from matplotlib.patches import Rectangle
    x = torch.from_numpy(img[None]).to(dev)
    f = features.calcular(x, janelas)[0].permute(1, 2, 0)
    h, w = f.shape[:2]
    pred = prever(bst, f.reshape(h * w, -1).cpu().numpy()).reshape(h, w)
    sem = ref == dados.IGNORE
    aval = np.zeros_like(sem)
    aval[margem:h - margem, margem:w - margem] = True
    aval &= ~sem
    oa = (pred[aval] == ref[aval]).mean() if aval.any() else float("nan")
    v0, v1 = np.percentile(img[0], [2, 98])
    axs[0].imshow(img[0], cmap="gray", vmin=v0, vmax=v1)
    axs[1].imshow(np.ma.masked_where(sem, ref), cmap=cmap, vmin=0, vmax=N - 1,
                  interpolation="nearest")
    axs[2].imshow(pred, cmap=cmap, vmin=0, vmax=N - 1, interpolation="nearest")
    axs[3].imshow(np.ma.masked_where(sem, pred != ref), cmap=cmap_erro, vmin=0, vmax=1,
                  interpolation="nearest")
    for c, tit in enumerate(["PAN  (%s)" % rotulo, "referência", "predição",
                             "erro (OA interior=%.3f)" % oa]):
        axs[c].set_title(tit, fontsize=9)
        axs[c].set_xticks([])
        axs[c].set_yticks([])
    for c in (2, 3):
        axs[c].add_patch(Rectangle((margem - .5, margem - .5), w - 2 * margem, h - 2 * margem,
                                   fill=False, ls="--", lw=.8, ec="white"))


def visualizar(bst, imgs, rotulos, nomes, cenas, janelas, margem, dev, caminho, n, titulo):
    """Figura com n patches divididos igualmente entre as cenas do split, um
    bloco (com cabeçalho) por cena; dentro da cena, patches espaçados."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

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

    n_lin = sum(len(sel) for _, sel, _ in grupos)
    fig = plt.figure(figsize=(13, 3.4 * n_lin + 0.8 * len(grupos)), layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(len(grupos), 1,
                                           height_ratios=[len(s) for _, s, _ in grupos]))
    for k, (sf, (c, sel, total)) in enumerate(zip(subfigs, grupos)):
        sf.set_facecolor("#f4f4f4" if k % 2 == 0 else "#e6eef7")
        sf.suptitle("%s  |  cena %s  (%d de %d patches)" % (titulo, c, len(sel), total),
                    fontsize=14, fontweight="bold")
        axs = sf.subplots(len(sel), 4, squeeze=False)
        for lin, i in enumerate(sel):
            rotulo = nomes[i][len(c) + 1:] if nomes[i].startswith(c + "_") else nomes[i]
            _linha(axs[lin], bst, imgs[i], rotulos[i], rotulo, janelas, margem, dev,
                   cmap, cmap_erro)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cor) for cor in dados.CORES_CLASSE],
               labels=dados.CLASS_NOMES, loc="outside lower center", ncol=N, fontsize=10)
    fig.savefig(caminho, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=dados.DATASET_PADRAO)
    ap.add_argument("--nome", default="xgb_spot5", help="pasta de saída em xgboost/runs/")
    ap.add_argument("--janelas", type=int, nargs="+", default=[5, 15, 31],
                    help="janelas (ímpares) da média/desvio local")
    ap.add_argument("--margem", type=int, default=-1,
                    help="pixels da borda fora do treino/avaliação (-1 = maior janela // 2)")
    ap.add_argument("--max-amostras", type=int, default=6_000_000,
                    help="pixels de treino (0 = todos)")
    ap.add_argument("--max-amostras-aval", type=int, default=0,
                    help="pixels por split de validação/teste (0 = todos). Todos os"
                         " pixels rotulados do interior dos 740+656 patches são ~64 M"
                         " linhas x 21 features = 5,4 GB de RAM, e validação e teste"
                         " ficam vivos ao mesmo tempo; amostrar derruba o pico sem"
                         " mexer em OA/mIoU além da 3a casa")
    ap.add_argument("--sem-balanceamento", action="store_true",
                    help="não pesar as classes pelo inverso da frequência")
    ap.add_argument("--n-estimadores", type=int, default=2000)
    ap.add_argument("--profundidade", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--min-child-weight", type=float, default=4)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--verbose", type=int, default=50)
    ap.add_argument("--lote", type=int, default=32, help="patches por lote no cálculo das features")
    ap.add_argument("--n-png", type=int, default=8,
                    help="patches por figura (validação/teste), divididos igualmente entre as cenas")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # O nº de classes vem do dataset (v13=4, v15 e v18=3), e `N` foi fixado no
    # import a partir do DATASET_PADRAO. Reavaliar aqui, antes de qualquer
    # carregamento, senão o treino monta silenciosamente uma cabeça do tamanho
    # errado.
    global N
    dados.configurar(args.dataset)
    N = dados.N_CLASSES
    print("[classes] %d: %s" % (N, ", ".join(dados.CLASS_NOMES)))

    dev = ambiente.exigir_gpu()
    janelas = sorted(set(args.janelas))
    if any(j % 2 == 0 or j < 3 for j in janelas):
        ap.error("--janelas devem ser ímpares >= 3")
    margem = max(janelas) // 2 if args.margem < 0 else args.margem

    saida = os.path.join(BASE, "runs", args.nome)
    os.makedirs(os.path.join(saida, "visualizacao"), exist_ok=True)
    nomes_bandas = dados.bandas(args.dataset)
    nomes_feat = features.nomes(nomes_bandas, janelas)

    t0 = time.time()
    splits = {s: dados.ler_split(args.dataset, s) for s in ("treino", "validacao", "teste")}
    cena_de = dados.cenas(args.dataset)
    cenas = {s: np.array([cena_de[n] for n in nomes]) for s, nomes in splits.items()}
    print("[dados] %s | bandas=%s | %.0fs" % (args.dataset, nomes_bandas, time.time() - t0))
    for s in splits:
        c, k = np.unique(cenas[s], return_counts=True)
        print("  patches %-9s %3d  (%s)" % (s, len(splits[s]),
                                            ", ".join("%s=%d" % ck for ck in zip(c, k))))

    t0 = time.time()
    g = torch.Generator(device=dev).manual_seed(args.seed)
    por_patch = args.max_amostras // len(splits["treino"]) if args.max_amostras else 0
    def cota_aval(s):
        return args.max_amostras_aval // len(splits[s]) if args.max_amostras_aval else 0
    comum = dict(janelas=janelas, margem=margem, gerador=g, lote=args.lote, dev=dev)
    def extrair_split(s, por_patch=0):
        """Lê os patches do split, extrai as features e solta os patches: carregar
        os 3 splits juntos (patches + features) estourava a memória do WSL."""
        X, y, pos = extrair(*dados.carregar(args.dataset, splits[s]), por_patch=por_patch, **comum)
        gc.collect()
        torch.cuda.empty_cache()
        fr = np.bincount(y, minlength=N) / len(y)
        print("  classes %-9s %s" % (s, "  ".join("%s=%.1f%%" % (dados.CLASS_NOMES[c], 100 * fr[c])
                                                  for c in range(N))), flush=True)
        return X, y, pos

    Xtr, ytr, _ = extrair_split("treino", por_patch)
    Xva, yva, pva = extrair_split("validacao", cota_aval("validacao"))
    print("[features] %d features | margem=%d | amostras treino=%d val=%d | %.0fs"
          % (len(nomes_feat), margem, len(ytr), len(yva), time.time() - t0))

    if args.sem_balanceamento:
        wtr, w_cl = None, np.ones(N)
    else:
        wtr, w_cl = pesos_classe(ytr)
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr, max_bin=args.max_bin,
                              feature_names=nomes_feat)
    dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr, feature_names=nomes_feat)

    params = {
        "objective": "multi:softprob",
        "num_class": N,
        "tree_method": "hist",
        "device": "cuda",
        "eta": args.lr,
        "max_depth": args.profundidade,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "min_child_weight": args.min_child_weight,
        "max_bin": args.max_bin,
        "eval_metric": ["merror", "mlogloss"],   # a última (mlogloss) manda na parada
        "seed": args.seed,
    }
    curva = {}
    t0 = time.time()
    bst = xgb.train(params, dtr, num_boost_round=args.n_estimadores,
                    evals=[(dtr, "treino"), (dva, "validacao")],
                    early_stopping_rounds=args.early_stopping,
                    evals_result=curva, verbose_eval=args.verbose)
    t_treino = time.time() - t0
    melhor = bst.best_iteration + 1
    bst = bst[:melhor]
    bst.save_model(os.path.join(saida, "modelo.ubj"))
    val_mlogloss = float(curva["validacao"]["mlogloss"][melhor - 1])
    print("[treino] %.0fs | melhor rodada=%d (val mlogloss=%.4f)"
          % (t_treino, melhor, val_mlogloss), flush=True)

    del dtr, dva                      # as QuantileDMatrix guardam outra cópia dos dados
    gc.collect()
    rels = {"treino": metricas("TREINO (amostrado)", prever(bst, Xtr), ytr)}
    del Xtr
    gc.collect()
    Xte, yte, pte = extrair_split("teste", cota_aval("teste"))  # só agora: treino saiu
    feats = {"validacao": Xva, "teste": Xte}
    del Xva, Xte                      # só a cópia do dict segura as features daqui p/ frente
    marca = " (amostrado)" if args.max_amostras_aval else ""
    for s, rot, y, p in (("validacao", "VALIDACAO" + marca, yva, pva),
                         ("teste", "TESTE" + marca, yte, pte)):
        pred = prever(bst, feats.pop(s))   # libera as features do split logo após prever
        gc.collect()
        rels[s] = metricas(rot, pred, y)
        cena_px = cenas[s][p]
        rels[s]["por_cena"] = {c: metricas("%s | cena %s" % (rot, c), pred[cena_px == c],
                                           y[cena_px == c])
                               for c in dict.fromkeys(cenas[s])}

    with open(os.path.join(saida, "curva.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rodada", "treino_merror", "treino_mlogloss",
                     "validacao_merror", "validacao_mlogloss"])
        for r in range(len(curva["treino"]["mlogloss"])):
            wr.writerow([r + 1] + [curva[s][m][r] for s in ("treino", "validacao")
                                   for m in ("merror", "mlogloss")])

    ganho = bst.get_score(importance_type="gain")
    tot = sum(ganho.values()) or 1.0
    importancia = {k: v / tot for k, v in sorted(ganho.items(), key=lambda kv: -kv[1])}
    print("[importancia] top 8 (ganho): %s"
          % ", ".join("%s=%.3f" % kv for kv in list(importancia.items())[:8]))

    for s, titulo in (("validacao", "VALIDAÇÃO"), ("teste", "TESTE")):
        visualizar(bst, *dados.carregar(args.dataset, splits[s]), splits[s], cenas[s],
                   janelas, margem, dev,
                   os.path.join(saida, "visualizacao", s + ".png"), args.n_png, titulo)
        gc.collect()

    with open(os.path.join(saida, "relatorio.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "dataset": os.path.abspath(args.dataset),
                   "bandas": nomes_bandas, "features": nomes_feat, "margem": margem,
                   "class_map": {str(k): v for k, v in dados.CLASS_MAP.items()},
                   "classes": dados.CLASS_NOMES, "peso_classes": w_cl.tolist(),
                   "amostras": {"treino": len(ytr), "validacao": len(yva), "teste": len(yte)},
                   "melhor_rodada": melhor, "val_mlogloss": val_mlogloss,
                   "tempo_treino_s": round(t_treino, 1), "xgboost": xgb.__version__,
                   "treino": rels["treino"], "validacao": rels["validacao"],
                   "teste": rels["teste"], "importancia_ganho": importancia},
                  f, indent=2, ensure_ascii=False)
    print("OK -> %s" % saida)


if __name__ == "__main__":
    main()
