#!/usr/bin/env python3
"""Avaliação de um modelo já treinado + importância dos canais por OCLUSÃO.

Serve para (a) obter as métricas de teste de um run que foi interrompido antes
do fim (sem relatorio.json) e (b) medir de quanto o modelo DEPENDE de cada
canal de entrada, sem retreinar.

Como as três bandas estão em z-score por cena, **zerar um canal equivale a
substituí-lo pela média da cena**: o canal continua existindo para a rede (o
nº de canais da entrada não muda), mas não carrega mais informação espacial.
Além da oclusão por zero, calcula a variante por PERMUTAÇÃO (o canal é trocado
pelo mesmo canal de outro patch do lote), que preserva o histograma do canal e
destrói só a correspondência espacial com as outras bandas.

ATENÇÃO na leitura: oclusão mede a dependência DESTE modelo treinado, não o
desempenho alcançável sem o canal — um modelo retreinado sem a banda poderia
compensar parte da perda. Para essa pergunta é preciso retreinar.

Métricas idênticas às do train.py (patch inteiro e "interior", --margem px da
borda), por cena e com matriz de confusão.

Saída em unet_cnn/runs/<nome>/avaliacao_canais.json (e tabela no terminal).

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/avaliar_canais.py \\
      --modelo unet_cnn/runs/unetpp_v15_ft/modelo_melhor.pth
  ../Data/.venv/bin/python unet_cnn/avaliar_canais.py \\
      --modelo unet_cnn/runs/unetpp_v15_ft/modelo_melhor.pth --split validacao
"""

import argparse
import json
import os

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch)
import numpy as np
import torch

import dados
from unet import modelo_do_checkpoint

BASE = os.path.dirname(os.path.abspath(__file__))


def metricas(cm, n):
    """OA, mIoU e recall/precisão/F1/IoU por classe (schema do train.py)."""
    tp = np.diag(cm).astype(np.float64)
    ref, prd = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall, prec = tp / ref, tp / prd
        f1, iou = 2 * tp / (ref + prd), tp / (ref + prd - tp)
    oa = tp.sum() / max(cm.sum(), 1)
    def lista(v):
        return [None if np.isnan(a) else float(a) for a in v]
    return {"oa": float(oa),
            "miou": float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan"),
            "miou_classes_presentes": float(np.nanmean(iou[ref > 0]))
            if (ref > 0).any() and np.isfinite(iou[ref > 0]).any() else float("nan"),
            "acuracia": lista(recall), "precisao": lista(prec), "f1": lista(f1),
            "iou": lista(iou), "ref_frac": lista(ref / max(cm.sum(), 1)),
            "matriz_confusao": cm.tolist(), "n_pixels": int(cm.sum())}


def matriz(pred, y, n, ign, mascara=None):
    v = y != ign
    if mascara is not None:
        v &= mascara
    return torch.bincount(y[v].long() * n + pred[v].long(), minlength=n * n) \
        .reshape(n, n).cpu().numpy()


@torch.no_grad()
def prever(modelo, X, ocultar=(), modo="zero", lote=16, rng=None):
    """Predição com os canais de `ocultar` neutralizados.
    modo 'zero'      -> canal = 0 (= média da cena, pois as bandas são z-score)
    modo 'permuta'   -> canal embaralhado entre os patches do lote."""
    modelo.eval()
    out = []
    for i in range(0, len(X), lote):
        x = X[i:i + lote].float()
        if ocultar:
            x = x.clone()
            for c in ocultar:
                if modo == "zero":
                    x[:, c] = 0.0
                else:
                    perm = torch.from_numpy(rng.permutation(len(x))).to(x.device)
                    x[:, c] = x[perm, c]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(modelo(x).argmax(1).to(torch.uint8))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True, help="modelo_melhor.pth de um run unet*")
    ap.add_argument("--split", default="teste", choices=["teste", "validacao", "treino"])
    ap.add_argument("--dataset", default=None,
                    help="padrão: o dataset gravado no checkpoint")
    ap.add_argument("--margem", type=int, default=None,
                    help="padrão: a margem do treino (interior)")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev = ambiente.exigir_gpu()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = np.random.default_rng(args.seed)

    caminho = args.modelo if os.path.isabs(args.modelo) \
        else os.path.join(ambiente.MODELS, args.modelo)
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    ta = ck["args"]
    dataset = args.dataset or ta["dataset"]
    if not os.path.isabs(dataset):
        dataset = os.path.normpath(os.path.join(ambiente.MODELS, dataset))
    margem = args.margem if args.margem is not None else ta.get("margem", 15)

    dados.configurar(dataset)
    N, IGN = dados.N_CLASSES, dados.IGNORE
    bandas = dados.bandas(dataset)
    nome_run = os.path.basename(os.path.dirname(caminho))
    print("[modelo] %s | epoca %s | val mIoU=%.4f | %s | %d classes | bandas %s"
          % (nome_run, ck.get("epoca"), ck.get("val_miou", float("nan")),
             os.path.basename(dataset), N, bandas), flush=True)

    nomes = dados.ler_split(dataset, args.split)
    cena_de = dados.cenas(dataset)
    imgs, rot = dados.carregar(dataset, nomes)
    cenas = np.array([cena_de[n] for n in nomes])
    X = torch.from_numpy(imgs).to(dev, torch.float16)
    Y = torch.from_numpy(rot).to(dev)
    H, W = X.shape[2:]
    interior = torch.zeros(H, W, dtype=torch.bool, device=dev)
    interior[margem:H - margem, margem:W - margem] = True
    print("[dados] split %s: %d patches | %d canais | interior = margem %d px"
          % (args.split, len(X), X.shape[1], margem), flush=True)

    modelo = modelo_do_checkpoint(ck, N)
    modelo.load_state_dict(ck["state_dict"])
    modelo = modelo.to(dev)
    print("[modelo] %s base=%d | %.2fM params"
          % (ta.get("arquitetura", "unetpp"), ta.get("base", 32),
             sum(p.numel() for p in modelo.parameters()) / 1e6), flush=True)

    # conjuntos de oclusão: nome -> canais neutralizados
    conf = [("completo (PAN+GLCM+Rug)", ()),
            ("sem GLCM", (1,)),
            ("sem Rugosidade", (2,)),
            ("sem textura (so PAN)", (1, 2)),
            ("sem PAN (so texturas)", (0,))]

    res, base_miou = {}, {}
    for modo in ("zero", "permuta"):
        print("\n=== oclusao por %s ===" % modo.upper(), flush=True)
        print("  %-26s %-28s %-28s" % ("", "patch inteiro", "interior (margem %d)" % margem))
        print("  %-26s %8s %8s %7s   %8s %8s %7s"
              % ("configuracao", "OA", "mIoU", "dmIoU", "OA", "mIoU", "dmIoU"))
        for rot_c, oc in conf:
            if modo == "permuta" and not oc:
                continue
            pred = prever(modelo, X, oc, modo, args.lote, rng)
            linha = {"canais_ocultos": list(oc), "modo": modo}
            for escopo, masc in (("inteiro", None), ("interior", interior)):
                mk = None if masc is None else masc.expand_as(Y)
                m = metricas(matriz(pred, Y, N, IGN, mk), N)
                m["por_cena"] = {}
                for c in dict.fromkeys(cenas):
                    sel = torch.from_numpy(np.flatnonzero(cenas == c)).to(dev)
                    m["por_cena"][c] = metricas(
                        matriz(pred[sel], Y[sel], N, IGN,
                               None if mk is None else mk[sel]), N)
                linha[escopo] = m
            if not oc:
                base_miou = {e: linha[e]["miou"] for e in ("inteiro", "interior")}
            print("  %-26s %8.4f %8.4f %+7.4f   %8.4f %8.4f %+7.4f"
                  % (rot_c, linha["inteiro"]["oa"], linha["inteiro"]["miou"],
                     linha["inteiro"]["miou"] - base_miou["inteiro"],
                     linha["interior"]["oa"], linha["interior"]["miou"],
                     linha["interior"]["miou"] - base_miou["interior"]), flush=True)
            res["%s|%s" % (modo, rot_c)] = linha

    print("\n=== IoU por classe (interior, oclusao por zero) ===")
    print("  %-26s %s" % ("configuracao", " ".join("%-20s" % c[:20] for c in dados.CLASS_NOMES)))
    for rot_c, oc in conf:
        m = res["zero|%s" % rot_c]["interior"]
        print("  %-26s %s" % (rot_c, " ".join("%-20.4f" % v for v in m["iou"])))

    print("\n=== por cena (interior, modelo completo) ===")
    m = res["zero|completo (PAN+GLCM+Rug)"]["interior"]
    for c, mc in m["por_cena"].items():
        print("  %-26s n=%9d OA=%.4f mIoU=%.4f (presentes %.4f) IoU=%s ref=%s"
              % (c, mc["n_pixels"], mc["oa"], mc["miou"], mc["miou_classes_presentes"],
                 " ".join("%.3f" % (v if v is not None else float("nan")) for v in mc["iou"]),
                 " ".join("%.2f" % v for v in mc["ref_frac"])))

    saida = os.path.join(os.path.dirname(caminho), "avaliacao_canais.json")
    with open(saida, "w", encoding="utf-8") as f:
        json.dump({"modelo": caminho, "run": nome_run, "epoca": ck.get("epoca"),
                   "val_miou": ck.get("val_miou"), "dataset": dataset, "split": args.split,
                   "margem": margem, "bandas": bandas, "classes": dados.CLASS_NOMES,
                   "args_treino": ta, "resultados": res}, f, indent=2, ensure_ascii=False)
    print("\nOK -> %s" % saida)


if __name__ == "__main__":
    main()
