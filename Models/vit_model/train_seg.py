#!/usr/bin/env python3
"""Segmentação 3 classes (SPOT5 refinado) com encoder ViT + decoder UNETR leve.

Serve para comparar encoders com o MESMO treino: DINO puro × MAE v1 × MAE v2.

- Treino: splits/treino.txt (patches 256 inteiros), aug flip/rot90 em imagem e
  máscara juntas + brilho/contraste só na PAN.
- Encoder com lr x --lr-encoder-mult (fine-tuning suave), decoder com --lr;
  com --congelar-encoder o encoder fica fixo e só o decoder treina.
- Loss: cross-entropy (0 = sem rótulo, ignorado) + Dice.
- Melhor epoch escolhido pelo mIoU da validação (com paciência); o teste só é
  avaliado no final, com o modelo desse epoch.
- Métricas no patch inteiro e no "interior" (sem --margem px da borda) — o
  interior é o mesmo recorte usado pelo XGBoost, para comparar direto.

Saída em vit_model/runs/<nome>/: modelo_melhor.pth, log.csv, relatorio.json,
visualizacao/{validacao,teste}.png (metade de cada cena).

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python vit_model/train_seg.py --encoder dino --nome seg_dino
  ../Data/.venv/bin/python vit_model/train_seg.py \\
      --encoder vit_model/runs/mae_vits_dino_v1/encoder.pth --nome seg_mae_v1
Comparar: ../Data/.venv/bin/python vit_model/comparar_seg.py
"""

import argparse
import copy
import csv
import json
import math
import os
import shutil
import time

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch/timm)
import numpy as np
import torch
import torch.nn.functional as F

import dados
from mae import lr_cosseno
from segmentacao import SegViT, criar_encoder

BASE = os.path.dirname(os.path.abspath(__file__))
N = dados.N_CLASSES
IGN = dados.IGNORE


def c_livre_gb():
    """Espaço livre no C: visto do WSL (pagefile do Windows e swap do WSL moram lá)."""
    return shutil.disk_usage("/mnt/c").free / 1e9 if os.path.isdir("/mnt/c") else float("inf")


def aumentar(x, y, rng, ganho_pan, offset_pan):
    """Uma das 8 combinações de rot90/flip (imagem e máscara juntas) + PAN
    com ganho/deslocamento aleatórios (as texturas não mudam com brilho)."""
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
    if ganho_pan or offset_pan:
        g = torch.from_numpy(rng.uniform(1 - ganho_pan, 1 + ganho_pan, B)).to(x)
        o = torch.from_numpy(rng.uniform(-offset_pan, offset_pan, B)).to(x)
        x[:, 0] = x[:, 0] * g[:, None, None] + o[:, None, None]
    return x, y


def perda(logits, y):
    """Cross-entropy (ignora sem rótulo) + Dice suave médio das classes."""
    logits = logits.float()
    ce = F.cross_entropy(logits, y, ignore_index=IGN)
    valido = (y != IGN)[:, None]
    p = logits.softmax(1) * valido
    t = F.one_hot(torch.where(y == IGN, 0, y), N).permute(0, 3, 1, 2) * valido
    inter = (p * t).sum(dim=(0, 2, 3))
    soma = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    return ce + (1 - ((2 * inter + 1) / (soma + 1)).mean())


@torch.no_grad()
def prever(modelo, X, lote=32):
    modelo.eval()
    out = []
    for i in range(0, len(X), lote):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(modelo(X[i:i + lote].float()).argmax(1).to(torch.uint8))
    return torch.cat(out)


def matriz(pred, y, mascara=None):
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
    if rotulo:
        linhas = ["%s  OA=%.4f  mIoU=%.4f  (%d pixels)" % (rotulo, oa, miou, cm.sum()),
                  "  %2s  %-26s  %7s  %7s  %7s  %7s  %6s"
                  % ("cl", "nome", "recall", "precis", "F1", "IoU", "ref%")]
        for c in range(N):
            linhas.append("  %2d  %-26s  %7.4f  %7.4f  %7.4f  %7.4f  %5.1f%%"
                          % (c, dados.CLASS_NOMES[c], recall[c], prec[c], f1[c], iou[c],
                             100 * ref[c] / max(cm.sum(), 1)))
        print("\n".join(linhas), flush=True)
    return {"oa": float(oa), "miou": miou, "acuracia": _lista(recall),
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
            for col, tit in enumerate(["PAN  (%s)" % rotulo, "referência", "predição",
                                       "erro (OA interior=%.3f)" % oa]):
                a[col].set_title(tit, fontsize=9)
                a[col].set_xticks([])
                a[col].set_yticks([])
            for col in (2, 3):
                a[col].add_patch(Rectangle((margem - .5, margem - .5), w - 2 * margem,
                                           h - 2 * margem, fill=False, ls="--", lw=.8, ec="white"))
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cor) for cor in dados.CORES_CLASSE],
               labels=dados.CLASS_NOMES, loc="outside lower center", ncol=N, fontsize=10)
    fig.savefig(caminho, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True,
                    help="'dino', 'aleatorio' ou caminho de um encoder.pth do MAE")
    ap.add_argument("--nome", required=True, help="pasta de saída em vit_model/runs/")
    ap.add_argument("--dataset", default=dados.DATASET_PADRAO)
    ap.add_argument("--inicializar", default=None,
                    help="modelo_melhor.pth de um run anterior para warm-start "
                         "(fine-tuning): carrega os pesos do mesmo encoder/decoder "
                         "antes de treinar. Use LR baixa e patience curto.")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--paciencia", type=int, default=25,
                    help="para se o mIoU de validação não melhora em N epochs")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4, help="lr do decoder")
    ap.add_argument("--lr-encoder-mult", type=float, default=0.1)
    ap.add_argument("--congelar-encoder", action="store_true",
                    help="encoder fixo: só o decoder treina")
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

    dev = ambiente.exigir_gpu()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    saida = os.path.join(BASE, "runs", args.nome)
    os.makedirs(os.path.join(saida, "visualizacao"), exist_ok=True)

    t0 = time.monotonic()
    cena_de = dados.cenas(args.dataset)
    nomes, brutos, cenas, X, Y = {}, {}, {}, {}, {}
    for s in ("treino", "validacao", "teste"):
        nomes[s] = dados.ler_split(args.dataset, s)
        brutos[s] = dados.carregar(args.dataset, nomes[s])
        cenas[s] = np.array([cena_de[n] for n in nomes[s]])
        X[s] = torch.from_numpy(brutos[s][0]).to(dev, torch.float16)
        Y[s] = torch.from_numpy(brutos[s][1]).to(dev)
    print("[dados] treino=%d val=%d teste=%d | %.0fs"
          % (len(X["treino"]), len(X["validacao"]), len(X["teste"]), time.monotonic() - t0))

    encoder, origem = criar_encoder(args.encoder, in_chans=X["treino"].shape[1])
    modelo = SegViT(encoder, N, in_chans=X["treino"].shape[1]).to(dev)
    if args.inicializar:
        ck = torch.load(args.inicializar, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        ausentes = [k for k in sd if k not in modelo.state_dict()]
        if ausentes:
            raise SystemExit("--inicializar %s: chaves %s não existem no modelo (mesma "
                             "arquitetura/encoder?)" % (args.inicializar, ausentes[:5]))
        carregadas = modelo.load_state_dict(sd, strict=False)
        print("[warm-start] %s (epoca %s, val mIoU=%.4f) | não carregadas: %d"
              % (args.inicializar, ck.get("epoca"), ck.get("val_miou", float("nan")),
                 len(carregadas.unexpected_keys)), flush=True)
    if args.congelar_encoder:
        modelo.congelar_encoder()
    n_enc = sum(p.numel() for p in modelo.encoder.parameters())
    n_dec = sum(p.numel() for n, p in modelo.named_parameters() if not n.startswith("encoder."))
    print("[modelo] encoder: %s%s | encoder=%.1fM decoder=%.1fM params"
          % (origem, " [CONGELADO]" if args.congelar_encoder else "", n_enc / 1e6, n_dec / 1e6),
          flush=True)

    grupos = {}
    for nome, p in modelo.named_parameters():
        if not p.requires_grad:
            continue
        parte = "encoder" if nome.startswith("encoder.") else "decoder"
        sem_wd = p.ndim <= 1 or nome.endswith(("pos_embed", "cls_token"))
        g = grupos.setdefault((parte, sem_wd), {
            "params": [], "weight_decay": 0.0 if sem_wd else args.wd,
            "lr_mult": args.lr_encoder_mult if parte == "encoder" else 1.0})
        g["params"].append(p)
    opt = torch.optim.AdamW(list(grupos.values()), lr=args.lr)
    passos_ep = math.ceil(len(X["treino"]) / args.lote)
    total, aquec = args.epocas * passos_ep, args.aquecimento * passos_ep

    H, W = X["treino"].shape[2:]
    interior = torch.zeros(H, W, dtype=torch.bool, device=dev)
    interior[args.margem:H - args.margem, args.margem:W - args.margem] = True

    historico, melhor, melhor_ep, estado = [], -1.0, 0, None
    campos = ["epoca", "lr", "loss_treino", "val_oa", "val_miou", "tempo_s"]
    livre = c_livre_gb()
    if livre < args.min_livre_c_gb:
        raise SystemExit("[trava] C: com %.2f GB livres (< %.1f GB): não vou começar"
                         % (livre, args.min_livre_c_gb))
    print("[trava] C: com %.2f GB livres; para sozinho se cair abaixo de %.1f GB"
          % (livre, args.min_livre_c_gb), flush=True)
    t_inicio = time.monotonic()
    for ep in range(args.epocas):
        modelo.train()
        perm = rng.permutation(len(X["treino"]))
        soma, n, t0 = 0.0, 0, time.monotonic()
        for b in range(passos_ep):
            if b % 20 == 0 and c_livre_gb() < args.min_livre_c_gb:
                raise SystemExit("[trava] C: com %.2f GB livres (< %.1f GB) na epoca %d: parando. "
                                 "O melhor modelo até aqui (epoca %d) está em modelo_melhor.pth."
                                 % (c_livre_gb(), args.min_livre_c_gb, ep + 1, melhor_ep))
            idx = torch.from_numpy(perm[b * args.lote:(b + 1) * args.lote]).to(dev)
            if len(idx) < 2:                                   # lote final com 1 patch:
                idx = idx.repeat(2)                            # duplica (BN do PPM exige >= 2)
            x, y = aumentar(X["treino"][idx], Y["treino"][idx], rng, args.ganho_pan, args.offset_pan)
            lr = lr_cosseno(ep * passos_ep + b, total, aquec, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = perda(modelo(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 3.0)
            opt.step()
            soma += float(loss.detach()) * len(idx)
            n += len(idx)

        m = metricas(matriz(prever(modelo, X["validacao"]), Y["validacao"]))
        reg = {"epoca": ep + 1, "lr": lr, "loss_treino": soma / n, "val_oa": m["oa"],
               "val_miou": m["miou"], "tempo_s": round(time.monotonic() - t0, 2)}
        historico.append(reg)
        marca = ""
        if m["miou"] > melhor:
            melhor, melhor_ep, marca = m["miou"], ep + 1, "  *"
            estado = copy.deepcopy({k: v.cpu() for k, v in modelo.state_dict().items()})
            torch.save({"state_dict": estado, "encoder": args.encoder, "origem": origem,
                        "epoca": ep + 1, "val_miou": melhor, "args": vars(args)},
                       os.path.join(saida, "modelo_melhor.pth"))
        print("epoca %3d | lr=%.2e | loss=%.4f | val OA=%.4f mIoU=%.4f | %.1fs%s"
              % (ep + 1, lr, reg["loss_treino"], m["oa"], m["miou"], reg["tempo_s"], marca),
              flush=True)
        with open(os.path.join(saida, "log.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=campos)
            wr.writeheader()
            wr.writerows(historico)
        if ep + 1 - melhor_ep >= args.paciencia:
            print("[parada] %d epochs sem melhorar (melhor epoca %d)" % (args.paciencia, melhor_ep))
            break
    t_treino = time.monotonic() - t_inicio

    modelo.load_state_dict(estado)
    print("\n[final] modelo da epoca %d (val mIoU=%.4f) | treino %.0fs" % (melhor_ep, melhor, t_treino))
    res = {}
    for s in ("validacao", "teste"):
        pred = prever(modelo, X[s])
        res[s] = {}
        for escopo, masc in (("inteiro", None), ("interior", interior)):
            mk = None if masc is None else masc.expand_as(Y[s])
            imprimir = s == "teste"
            r = metricas(matriz(pred, Y[s], mk),
                         "%s | %s | geral" % (s.upper(), escopo) if imprimir else None)
            r["por_cena"] = {}
            for c in dict.fromkeys(cenas[s]):
                sel = torch.from_numpy(np.flatnonzero(cenas[s] == c)).to(dev)
                r["por_cena"][c] = metricas(
                    matriz(pred[sel], Y[s][sel], None if mk is None else mk[sel]),
                    "%s | %s | cena %s" % (s.upper(), escopo, c)
                    if imprimir and escopo == "interior" else None)
            res[s][escopo] = r
        visualizar(brutos[s][0], brutos[s][1], pred.cpu().numpy(), nomes[s], cenas[s],
                   args.margem, os.path.join(saida, "visualizacao", s + ".png"), args.n_png,
                   {"validacao": "VALIDAÇÃO", "teste": "TESTE"}[s])

    with open(os.path.join(saida, "relatorio.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "encoder_origem": origem,
                   "encoder_congelado": args.congelar_encoder,
                   "params": {"encoder": n_enc, "decoder": n_dec},
                   "melhor_epoca": melhor_ep, "epocas_rodadas": len(historico),
                   "tempo_treino_s": round(t_treino, 1), "classes": dados.CLASS_NOMES,
                   "validacao": res["validacao"], "teste": res["teste"]},
                  f, indent=2, ensure_ascii=False)
    print("OK -> %s" % saida)


if __name__ == "__main__":
    main()
