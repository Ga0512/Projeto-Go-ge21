#!/usr/bin/env python3
"""Random Forest do RF_macro.py treinado e avaliado nos patches, para comparar.

O `rf_treino.py` treina sobre uma cena + raster de rótulos, que é o fluxo do
RF_macro.py original. Isso NÃO é comparável com o nosso XGBoost nem com a
U-Net++, que treinam no dataset de patches, com splits por bloco espacial e
métricas de interior. Este script fecha essa lacuna: mesmo dataset, mesmos
splits, mesma margem, mesma função de métricas, mesmo schema de relatorio.json.

A única diferença em relação ao XGBoost é o modelo -- e é justamente o que se
quer medir:

  RF (aqui)      3 features: as 3 bandas do patch, pixel isolado
  XGBoost        21 features: as 3 bandas + média e desvio local em 5/15/31 px

A margem de avaliação é a MESMA (`--margem -1` = maior janela // 2 = 15), ainda
que o RF não precise dela: sem isso o RF seria avaliado em pixels de borda que o
XGBoost não vê, e a comparação ficaria a favor de quem tem o conjunto mais
fácil. Ver CLAUDE.md: "comparações entre famílias usam sempre o interior".

`--max-profundidade 0` é a configuração original (livre); 12 é a variante que
mediu melhor na amostra da cena. Rode as duas e compare.

Saída, igual às outras famílias: runs/<nome>/relatorio.json (níveis
validacao/teste, com `por_cena`), modelo.joblib, treino.json e
visualizacao/*.png.

O `fit` das 500 árvores leva ~28 min e a avaliação vem depois; se o processo
morrer no meio (aqui o guarda de memória do Claude Code já matou um),
`--retomar` lê o modelo.joblib gravado e vai direto para validação/teste, sem
treinar de novo. Os metadados do treino (tempo, pesos de classe, nº de
amostras) vêm do treino.json, gravado junto do modelo; sem ele, esses campos
do relatório ficam nulos e `retomado: true` registra o motivo.

Uso (no WSL, a partir de `Models/`):
  ../Data/.venv/bin/python xgboost/rf_patches.py --nome rf_v18_livre \\
      --dataset ../Data/dados/datasets/spot5_refinado_v18 --max-profundidade 0

  ../Data/.venv/bin/python xgboost/rf_patches.py --nome rf_v18_d12 \\
      --dataset ../Data/dados/datasets/spot5_refinado_v18 --max-profundidade 12
"""

import argparse
import gc
import json
import os
import sys
import time

import ambiente  # noqa: F401
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import dados  # noqa: E402

N = dados.N_CLASSES          # reavaliado em main(), depois de dados.configurar()


# ------------------------------------------------------------ amostragem
def extrair(imgs, rotulos, margem, por_patch, rng):
    """-> X (n,3) float32, y (n,) uint8, patch (n,) int32.

    Só pixels rotulados do INTERIOR, igual ao extrair() do train.py -- mas sem
    features derivadas: o RF recebe as 3 bandas do pixel e nada mais.
    """
    n, c, h, w = imgs.shape
    interior = np.zeros((h, w), bool)
    interior[margem:h - margem, margem:w - margem] = True
    Xs, ys, ps = [], [], []
    for k in range(n):
        sel = np.flatnonzero(((rotulos[k] != dados.IGNORE) & interior).reshape(-1))
        if por_patch and len(sel) > por_patch:
            sel = rng.choice(sel, por_patch, replace=False)
        f = np.moveaxis(imgs[k], 0, -1).reshape(-1, c)
        Xs.append(f[sel])
        ys.append(rotulos[k].reshape(-1)[sel])
        ps.append(np.full(len(sel), k, np.int32))
    return (np.concatenate(Xs).astype(np.float32),
            np.concatenate(ys), np.concatenate(ps))


def pesos_classe(y):
    cont = np.bincount(y, minlength=N).astype(np.float64)
    w = np.where(cont > 0, len(y) / (N * np.maximum(cont, 1)), 0.0)
    return w[y].astype(np.float32), w


def _lista(v):
    return [None if np.isnan(a) else float(a) for a in v]


def metricas(rotulo, pred, y):
    """Idêntica à de train.py: mesmo schema, para comparar_seg.py agrupar junto."""
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


def prever(rf, X, lote=2_000_000):
    out = np.empty(len(X), np.uint8)
    for i in range(0, len(X), lote):
        out[i:i + lote] = rf.predict(X[i:i + lote]).astype(np.uint8)
    return out


# ------------------------------------------------------------ figura
def visualizar(rf, imgs, rotulos, nomes, cenas_arr, margem, saida, rotulo, n_png):
    """PAN | referência | predição | erro, para alguns patches de cada cena."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    cmap = ListedColormap(dados.CORES_CLASSE)
    cmap_erro = ListedColormap(["#d9d9d9", "#d62728"])
    unicas = sorted(set(cenas_arr))
    por = max(1, n_png // len(unicas))
    idx = []
    for c in unicas:
        idx += list(np.flatnonzero(cenas_arr == c)[:por])
    if not idx:
        return

    fig, axs = plt.subplots(len(idx), 4, figsize=(13, 3.3 * len(idx)),
                            squeeze=False, layout="constrained")
    for linha, k in enumerate(idx):
        img, ref = imgs[k], rotulos[k]
        c, h, w = img.shape
        pred = prever(rf, np.moveaxis(img, 0, -1).reshape(-1, c)).reshape(h, w)
        sem = ref == dados.IGNORE
        aval = np.zeros_like(sem)
        aval[margem:h - margem, margem:w - margem] = True
        erro = ((pred != ref) & ~sem & aval).astype(np.uint8)

        a = axs[linha]
        a[0].imshow(img[0], cmap="gray")
        a[0].set_title("%s\n%s" % (cenas_arr[k], nomes[k]), fontsize=8)
        a[1].imshow(np.ma.masked_where(sem, ref), cmap=cmap, vmin=0, vmax=N - 1,
                    interpolation="nearest")
        a[1].set_title("referência", fontsize=9)
        a[2].imshow(pred, cmap=cmap, vmin=0, vmax=N - 1, interpolation="nearest")
        a[2].set_title("predição", fontsize=9)
        a[3].imshow(np.ma.masked_where(~aval | sem, erro), cmap=cmap_erro,
                    vmin=0, vmax=1, interpolation="nearest")
        a[3].set_title("erro (%.1f%% do interior)"
                       % (100 * erro.sum() / max(1, (aval & ~sem).sum())), fontsize=9)
        for eixo in a:
            eixo.add_patch(Rectangle((margem - .5, margem - .5), w - 2 * margem,
                                     h - 2 * margem, fill=False, ec="w", lw=.8,
                                     ls=":"))
            eixo.axis("off")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cor)
                        for cor in dados.CORES_CLASSE],
               labels=dados.CLASS_NOMES, loc="outside lower center",
               ncol=N, fontsize=10)
    cam = os.path.join(saida, "visualizacao", "%s.png" % rotulo)
    fig.savefig(cam, dpi=110)
    plt.close(fig)
    print("  -> %s" % cam, flush=True)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=dados.DATASET_PADRAO)
    ap.add_argument("--nome", default="rf_patches", help="pasta em xgboost/runs/")
    ap.add_argument("--margem", type=int, default=-1,
                    help="-1 = 15 (maior janela do XGBoost // 2), para o "
                         "conjunto de avaliação ser o MESMO")
    ap.add_argument("--max-amostras", type=int, default=6_000_000,
                    help="pixels de treino (0 = todos)")
    ap.add_argument("--arvores", type=int, default=500)
    ap.add_argument("--max-profundidade", type=int, default=0, help="0 = livre")
    ap.add_argument("--sem-balanceamento", action="store_true")
    ap.add_argument("--retomar", action="store_true",
                    help="não treinar: ler runs/<nome>/modelo.joblib e ir direto "
                         "para a avaliação de validação/teste")
    ap.add_argument("--n-png", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global N
    dados.configurar(args.dataset)
    N = dados.N_CLASSES
    print("[classes] %d: %s" % (N, ", ".join(dados.CLASS_NOMES)))
    margem = 15 if args.margem < 0 else args.margem

    import joblib
    from sklearn.ensemble import RandomForestClassifier

    saida = os.path.join(BASE, "runs", args.nome)
    os.makedirs(os.path.join(saida, "visualizacao"), exist_ok=True)
    nomes_bandas = dados.bandas(args.dataset)

    t0 = time.time()
    splits = {s: dados.ler_split(args.dataset, s)
              for s in ("treino", "validacao", "teste")}
    cena_de = dados.cenas(args.dataset)
    cenas = {s: np.array([cena_de[n] for n in nomes]) for s, nomes in splits.items()}
    print("[dados] %s | bandas=%s | %.0fs"
          % (args.dataset, nomes_bandas, time.time() - t0))
    for s in splits:
        c, k = np.unique(cenas[s], return_counts=True)
        print("  patches %-9s %3d  (%s)"
              % (s, len(splits[s]), ", ".join("%s=%d" % ck for ck in zip(c, k))))

    rng = np.random.default_rng(args.seed)
    por_patch = (args.max_amostras // len(splits["treino"])
                 if args.max_amostras else 0)
    cam_modelo = os.path.join(saida, "modelo.joblib")
    cam_meta = os.path.join(saida, "treino.json")
    n_treino = None

    if args.retomar:
        if not os.path.exists(cam_modelo):
            sys.exit("--retomar: %s não existe" % cam_modelo)
        rf = joblib.load(cam_modelo)
        meta = {}
        if os.path.exists(cam_meta):
            with open(cam_meta, encoding="utf-8") as f:
                meta = json.load(f)
        dt_treino = meta.get("tempo_treino_s")
        w_cl = (np.array(meta["peso_classes"], dtype=np.float64)
                if meta.get("peso_classes") else None)
        n_treino = meta.get("amostras_treino")
        nos = int(sum(e.tree_.node_count for e in rf.estimators_))
        prof = int(max(e.tree_.max_depth for e in rf.estimators_))
        print("[retomar] %s | %d árvores | %d nos | profundidade %d | treino %s"
              % (cam_modelo, len(rf.estimators_), nos, prof,
                 "%.0fs" % dt_treino if dt_treino else "(sem treino.json)"),
              flush=True)
    else:
        t0 = time.time()
        imgs, rots = dados.carregar(args.dataset, splits["treino"])
        Xtr, ytr, _ = extrair(imgs, rots, margem, por_patch, rng)
        del imgs, rots
        gc.collect()
        n_treino = int(len(ytr))
        fr = np.bincount(ytr, minlength=N) / len(ytr)
        print("  classes treino    %s"
              % "  ".join("%s=%.1f%%" % (dados.CLASS_NOMES[c], 100 * fr[c])
                           for c in range(N)))
        print("[amostras] %d features | margem=%d | treino=%d | %.0fs"
              % (Xtr.shape[1], margem, len(ytr), time.time() - t0), flush=True)

        if args.sem_balanceamento:
            wtr, w_cl = None, np.ones(N)
        else:
            wtr, w_cl = pesos_classe(ytr)

        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=args.arvores,
                                    max_depth=args.max_profundidade or None,
                                    oob_score=False, random_state=args.seed,
                                    n_jobs=-1)
        rf.fit(Xtr, ytr, sample_weight=wtr)
        dt_treino = time.time() - t0
        nos = int(sum(e.tree_.node_count for e in rf.estimators_))
        prof = int(max(e.tree_.max_depth for e in rf.estimators_))
        print("[treino] %.0fs | %d nos | profundidade %d" % (dt_treino, nos, prof),
              flush=True)
        del Xtr, ytr, wtr
        gc.collect()

        joblib.dump(rf, cam_modelo, compress=3)
        print("[modelo] %s (%.1f MB)"
              % (cam_modelo, os.path.getsize(cam_modelo) / 1e6), flush=True)
        # Gravado junto do modelo para que um --retomar depois de uma morte no
        # meio da avaliação recupere o que só o treino sabia.
        with open(cam_meta, "w", encoding="utf-8") as f:
            json.dump({"tempo_treino_s": dt_treino, "nos": nos,
                       "profundidade": prof, "arvores": len(rf.estimators_),
                       "amostras_treino": n_treino,
                       "peso_classes": w_cl.tolist()}, f, indent=2)

    rel = {"args": vars(args), "dataset": args.dataset, "bandas": nomes_bandas,
           "features": nomes_bandas, "margem": margem,
           "class_map": {str(k): v for k, v in dados.CLASS_MAP.items()},
           "classes": dados.CLASS_NOMES,
           "peso_classes": w_cl.tolist() if w_cl is not None else None,
           "modelo": "RandomForestClassifier",
           "max_profundidade": args.max_profundidade or None,
           "profundidade": prof, "nos": nos, "arvores": len(rf.estimators_),
           "tempo_treino_s": dt_treino, "retomado": bool(args.retomar),
           "amostras": {"treino": n_treino}}

    for s in ("validacao", "teste"):
        t0 = time.time()
        imgs, rots = dados.carregar(args.dataset, splits[s])
        X, y, pos = extrair(imgs, rots, margem, 0, rng)
        rel["amostras"][s] = int(len(y))
        print("\n[%s] %d px | %.0fs" % (s, len(y), time.time() - t0), flush=True)
        pred = prever(rf, X)
        rel[s] = metricas("%s (interior, margem %d)" % (s, margem), pred, y)
        rel[s]["por_cena"] = {}
        for c in sorted(set(cenas[s])):
            m = cenas[s][pos] == c
            if m.any():
                rel[s]["por_cena"][c] = metricas("  %s / %s" % (s, c),
                                                 pred[m], y[m])
        del X, y, pos
        gc.collect()
        visualizar(rf, imgs, rots, splits[s], cenas[s], margem, saida, s,
                   args.n_png)
        del imgs, rots
        gc.collect()

    with open(os.path.join(saida, "relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=2, ensure_ascii=False)
    print("\n-> %s" % os.path.join(saida, "relatorio.json"), flush=True)


if __name__ == "__main__":
    main()
