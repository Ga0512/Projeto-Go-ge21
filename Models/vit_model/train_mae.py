#!/usr/bin/env python3
"""Pré-treino MAE do encoder ViT nos patches SPOT5 (pasta images/ do dataset).

O MAE não usa as máscaras: treina com as imagens de splits/treino.txt e mede a
reconstrução em splits/validacao.txt. splits/teste.txt fica de fora (reservado
para avaliar a segmentação depois). Com --sem-rotulo, soma ao treino os patches
sem rótulo (uso=mae) empacotados por preparar_sem_rotulo.py.

- Encoder ViT-S/16 com pesos DINO (timm), decoder MAE leve (descartado no fim).
- Augmentation (aumentar()): crop com zoom (área --escala-min..1, proporção
  3:4..4:3) redimensionado para --img, 8 combinações de flip/rot90 e
  brilho/contraste aleatórios só na PAN. Validação sem augmentation.
- Primeiros --congelar epochs: encoder congelado, só o decoder treina; depois
  o encoder treina com lr x --lr-encoder-mult.
- Probe linear (probe.py) no epoch 0 (DINO puro) e a cada --probe-cada epochs:
  mostra se o MAE está melhorando a representação para as nossas classes.

Saída em vit_model/runs/<nome>/:
  encoder.pth               só o encoder (formato timm) — o que vai para a segmentação
  encoder_melhor_probe.pth  encoder com o melhor mIoU de probe na validação
  ultimo.pth                checkpoint completo (--resume)
  log.csv, probe.json, relatorio.json, recon/epoca_XXXX.png (metade de cada cena),
  exemplos_aug.png (conferência da augmentation)

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python vit_model/train_mae.py --nome mae_vits_dino_v1
"""

import argparse
import csv
import json
import math
import os
import shutil
import time

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch/timm)
import numpy as np
import timm
import torch

import dados
import probe
from mae import MAE, grupos_parametros, lr_cosseno

BASE = os.path.dirname(os.path.abspath(__file__))


def aumentar(x, tam, rng, escala=(0.35, 1.0), proporcao=(3 / 4, 4 / 3),
             ganho_pan=0.2, offset_pan=0.2):
    """Augmentation por amostra, tudo numa amostragem afim na GPU:
    - crop aleatório com área em `escala` do patch e proporção em `proporcao`,
      redimensionado (bilinear) para tam x tam;
    - uma das 8 combinações de rot90/flip;
    - PAN (banda 0) com ganho 1±ganho_pan e deslocamento ±offset_pan (em
      desvios). As texturas foram calculadas sobre a PAN esticada p2–p98, então
      não mudam com brilho/contraste: só a PAN varia. Com norm_pix_loss o alvo
      por patch não muda — o encoder aprende a ignorar o brilho da cena."""
    B, _, H, W = x.shape
    area = rng.uniform(*escala, B) * H * W
    r = np.exp(rng.uniform(np.log(proporcao[0]), np.log(proporcao[1]), B))
    w = np.clip(np.sqrt(area * r), 16, W)
    h = np.clip(np.sqrt(area / r), 16, H)
    cx = rng.uniform(w / 2, W - w / 2)
    cy = rng.uniform(h / 2, H - h / 2)
    ang = rng.integers(0, 4, B) * (np.pi / 2)
    flip = np.where(rng.integers(0, 2, B) == 1, -1.0, 1.0)
    cos, sin = np.round(np.cos(ang)), np.round(np.sin(ang))
    # theta: coords de saída [-1,1] -> coords de entrada; A = escala_do_crop @ flip @ rot
    theta = np.zeros((B, 2, 3), dtype=np.float32)
    theta[:, 0, 0] = (w / W) * flip * cos
    theta[:, 0, 1] = (w / W) * flip * -sin
    theta[:, 1, 0] = (h / H) * sin
    theta[:, 1, 1] = (h / H) * cos
    theta[:, 0, 2] = 2 * cx / W - 1
    theta[:, 1, 2] = 2 * cy / H - 1
    theta = torch.from_numpy(theta).to(x.device)
    grade = torch.nn.functional.affine_grid(theta, (B, x.shape[1], tam, tam), align_corners=False)
    out = torch.nn.functional.grid_sample(x.float(), grade, mode="bilinear",
                                          padding_mode="reflection", align_corners=False)
    if ganho_pan or offset_pan:
        g = torch.from_numpy(rng.uniform(1 - ganho_pan, 1 + ganho_pan, B)).to(out)
        o = torch.from_numpy(rng.uniform(-offset_pan, offset_pan, B)).to(out)
        out[:, 0] = out[:, 0] * g[:, None, None] + o[:, None, None]
    return out


def exemplos_aug(imgs, tam, rng, caminho, aug_kw, n=6, n_aug=4):
    """PNG de conferência: PAN original (256) e n_aug versões aumentadas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.linspace(0, len(imgs) - 1, n).astype(int)
    fig, axs = plt.subplots(n, n_aug + 1, figsize=(2.2 * (n_aug + 1), 2.2 * n), squeeze=False)
    for lin, i in enumerate(idx):
        x = imgs[i:i + 1].float()
        v0, v1 = np.percentile(x[0, 0].cpu().numpy(), [2, 98])
        axs[lin, 0].imshow(x[0, 0].cpu().numpy(), cmap="gray", vmin=v0, vmax=v1)
        axs[lin, 0].set_title("original 256", fontsize=8)
        for k in range(n_aug):
            a = aumentar(x, tam, rng, **aug_kw)[0, 0].cpu().numpy()
            axs[lin, k + 1].imshow(a, cmap="gray", vmin=v0, vmax=v1)   # mesma escala: vê o brilho
            axs[lin, k + 1].set_title("aug %d (%d)" % (k + 1, tam), fontsize=8)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(caminho, dpi=70)
    plt.close(fig)


class LeitorBlocos:
    """Lotes de um .npy grande num HD, sem multiprocessing (fork depois do
    CUDA trava os workers do DataLoader).

    Os índices viram pedaços de até `bloco` patches CONTÍGUOS no arquivo; cada
    lote junta lote // bloco pedaços sorteados (poucos saltos de disco em vez de
    um por patch) e a ordem dos pedaços é sorteada a cada epoch. Uma thread lê
    adiantado com readinto (libera o GIL durante o I/O) enquanto a GPU treina."""

    def __init__(self, caminho, indices, lote, bloco, rng, adiantados=4):
        with open(caminho, "rb") as f:
            versao = np.lib.format.read_magic(f)
            leitor = (np.lib.format.read_array_header_1_0 if versao == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, _, dtype = leitor(f)
            self.offset = f.tell()
        self.caminho, self.forma, self.dtype = caminho, tuple(shape[1:]), np.dtype(dtype)
        self.bytes_patch = int(np.prod(self.forma)) * self.dtype.itemsize
        idx = np.sort(np.asarray(indices))
        self.pedacos = []
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
            for k in range(0, len(seg), bloco):
                self.pedacos.append((int(seg[k]), int(min(bloco, len(seg) - k))))
        self.por_lote = max(1, lote // bloco)
        self.passos = len(self.pedacos) // self.por_lote
        self.rng, self.adiantados = rng, adiantados

    def epoca(self):
        import queue
        import threading
        ordem = self.rng.permutation(len(self.pedacos))
        fila = queue.Queue(maxsize=self.adiantados)

        def produzir():
            try:
                with open(self.caminho, "rb", buffering=0) as f:
                    for p in range(self.passos):
                        sel = [self.pedacos[i]
                               for i in ordem[p * self.por_lote:(p + 1) * self.por_lote]]
                        buf = np.empty((sum(n for _, n in sel), *self.forma), dtype=self.dtype)
                        pos = 0
                        for ini, n in sel:
                            ini_b = self.offset + ini * self.bytes_patch
                            f.seek(ini_b)
                            if f.readinto(memoryview(buf[pos:pos + n]).cast("B")) != n * self.bytes_patch:
                                raise IOError("leitura curta em %s (patch %d)" % (self.caminho, ini))
                            # tira o trecho lido do cache de disco do Linux: sem isso a RAM do
                            # WSL incha com o .npy e o Windows empurra memória p/ o pagefile (C:)
                            try:
                                os.posix_fadvise(f.fileno(), ini_b, n * self.bytes_patch,
                                                 os.POSIX_FADV_DONTNEED)
                            except OSError:
                                pass
                            pos += n
                        fila.put(torch.from_numpy(buf).pin_memory())
                fila.put(None)
            except BaseException as e:   # repassa o erro para o laço de treino
                fila.put(e)

        threading.Thread(target=produzir, daemon=True).start()
        while (x := fila.get()) is not None:
            if isinstance(x, BaseException):
                raise x
            yield x


def c_livre_gb():
    """Espaço livre no C: visto do WSL (pagefile/swap do WSL moram lá)."""
    return shutil.disk_usage("/mnt/c").free / 1e9 if os.path.isdir("/mnt/c") else float("inf")


def centro(x, tam):
    oy, ox = (x.shape[2] - tam) // 2, (x.shape[3] - tam) // 2
    return x[:, :, oy:oy + tam, ox:ox + tam].float()


@torch.no_grad()
def validar(mae, imgs, tam, razao, seed, lote=64):
    """Loss de reconstrução com crop central e máscara fixa (mesma a cada chamada)."""
    mae.eval()
    g = torch.Generator(device=imgs.device).manual_seed(seed)
    tot, bandas, n = 0.0, 0.0, 0
    for i in range(0, len(imgs), lote):
        x = centro(imgs[i:i + lote], tam)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, por_banda, _, _ = mae(x, razao, g)
        tot += float(loss) * len(x)
        bandas = bandas + por_banda.float() * len(x)
        n += len(x)
    return tot / n, (bandas / n).tolist()


@torch.no_grad()
def visualizar_recon(mae, imgs, nomes, cenas, bandas, tam, razao, seed, caminho, n, titulo):
    """Original / mascarada / reconstruída por banda; metade dos patches de cada cena."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mae.eval()
    unicas = list(dict.fromkeys(cenas))
    por_cena = max(1, n // len(unicas))
    grupos = []
    for c in unicas:
        ids = np.flatnonzero(cenas == c)
        pos = np.unique(np.linspace(0, len(ids) - 1, min(por_cena, len(ids))).astype(int))
        grupos.append((c, ids[pos], len(ids)))
    todos = np.concatenate([sel for _, sel, _ in grupos])
    x = centro(imgs[torch.from_numpy(todos).to(imgs.device)], tam)
    g = torch.Generator(device=imgs.device).manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, pred, mask = mae(x, razao, g)
    recon, mascarada = mae.reconstruir(x, pred, mask)
    x, recon, mascarada = x.cpu().numpy(), recon.cpu().numpy(), mascarada.cpu().numpy()
    linha_de = {int(i): k for k, i in enumerate(todos)}

    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#3b6ea5")
    colunas = [(0, "orig"), (0, "masc"), (0, "recon")] + \
              [(b, t) for b in range(1, len(bandas)) for t in ("orig", "recon")]
    n_lin = len(todos)
    fig = plt.figure(figsize=(2.1 * len(colunas), 2.3 * n_lin + 0.8 * len(grupos)),
                     layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(len(grupos), 1,
                                           height_ratios=[len(s) for _, s, _ in grupos]))
    for k, (sf, (c, sel, total)) in enumerate(zip(subfigs, grupos)):
        sf.set_facecolor("#f4f4f4" if k % 2 == 0 else "#e6eef7")
        sf.suptitle("%s  |  cena %s  (%d de %d patches)" % (titulo, c, len(sel), total),
                    fontsize=12, fontweight="bold")
        axs = sf.subplots(len(sel), len(colunas), squeeze=False)
        for lin, i in enumerate(sel):
            r = linha_de[int(i)]
            rotulo = nomes[i][len(c) + 1:] if nomes[i].startswith(c + "_") else nomes[i]
            for col, (b, tipo) in enumerate(colunas):
                v0, v1 = np.percentile(x[r, b], [2, 98])
                img = {"orig": x, "masc": mascarada, "recon": recon}[tipo][r, b]
                ax = axs[lin, col]
                ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=v0, vmax=v1)
                ax.set_xticks([])
                ax.set_yticks([])
                if lin == 0:
                    ax.set_title("%s\n%s" % (bandas[b], {"orig": "original", "masc": "mascarada",
                                                         "recon": "reconstruída"}[tipo]),
                                 fontsize=8)
            axs[lin, 0].set_ylabel(rotulo, fontsize=7)
    fig.savefig(caminho, dpi=75)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=dados.DATASET_PADRAO)
    ap.add_argument("--nome", default="mae_vits_dino", help="pasta de saída em vit_model/runs/")
    ap.add_argument("--arquitetura", default="vit_small_patch16_224.dino",
                    help="modelo do timm usado como encoder")
    ap.add_argument("--sem-pesos", action="store_true", help="encoder do zero (sem pré-treino)")
    ap.add_argument("--img", type=int, default=224, help="lado do crop de treino")
    ap.add_argument("--escala-min", type=float, default=0.35,
                    help="área mínima do crop (fração do patch); 1 = sem zoom")
    ap.add_argument("--ganho-pan", type=float, default=0.2, help="contraste da PAN: ganho 1±x")
    ap.add_argument("--offset-pan", type=float, default=0.2, help="brilho da PAN: ±x desvios")
    ap.add_argument("--epocas", type=int, default=800)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4, help="lr do decoder (pico do cosseno)")
    ap.add_argument("--lr-encoder-mult", type=float, default=0.25)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--aquecimento", type=int, default=20, help="epochs de warmup do lr")
    ap.add_argument("--congelar", type=int, default=10, help="epochs iniciais só com o decoder")
    ap.add_argument("--razao-mascara", type=float, default=0.75)
    ap.add_argument("--dec-dim", type=int, default=256)
    ap.add_argument("--dec-prof", type=int, default=4)
    ap.add_argument("--val-cada", type=int, default=20)
    ap.add_argument("--probe-cada", type=int, default=100)
    ap.add_argument("--n-png", type=int, default=8, help="patches por PNG, metade de cada cena")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="retomar de runs/<nome>/ultimo.pth")
    ap.add_argument("--sem-rotulo", default=None,
                    help="pasta de preparar_sem_rotulo.py (nome em vit_model/dados/ ou caminho): "
                         "treina com os patches uso=mae dela + os de treino rotulados")
    ap.add_argument("--bloco-leitura", type=int, default=8,
                    help="patches contíguos lidos de uma vez do .npy (--sem-rotulo)")
    ap.add_argument("--min-livre-c-gb", type=float, default=1.0,
                    help="trava: salva checkpoint e para se o C: (/mnt/c) ficar abaixo disso")
    args = ap.parse_args()

    dev = ambiente.exigir_gpu()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    saida = os.path.join(BASE, "runs", args.nome)
    os.makedirs(os.path.join(saida, "recon"), exist_ok=True)
    bandas = dados.bandas(args.dataset)
    cena_de = dados.cenas(args.dataset)
    t0 = time.monotonic()
    nomes_tr = dados.ler_split(args.dataset, "treino")
    nomes_va = dados.ler_split(args.dataset, "validacao")
    imgs_tr, rot_tr = dados.carregar(args.dataset, nomes_tr)
    imgs_va, rot_va = dados.carregar(args.dataset, nomes_va)
    cenas_tr = np.array([cena_de[n] for n in nomes_tr])
    cenas_va = np.array([cena_de[n] for n in nomes_va])
    X_tr = torch.from_numpy(imgs_tr).to(dev, torch.float16)
    X_va = torch.from_numpy(imgs_va).to(dev, torch.float16)
    print("[dados] treino=%d val=%d patches %s | bandas=%s | %.0fs"
          % (len(X_tr), len(X_va), tuple(X_tr.shape[1:]), bandas, time.monotonic() - t0), flush=True)
    for rot, cs in (("treino", cenas_tr), ("val", cenas_va)):
        c, k = np.unique(cs, return_counts=True)
        print("  %-6s %s" % (rot, ", ".join("%s=%d" % ck for ck in zip(c, k))))

    mae = MAE(args.arquitetura, pesos=not args.sem_pesos, img_size=args.img,
              in_chans=X_tr.shape[1], dec_dim=args.dec_dim, dec_prof=args.dec_prof).to(dev)
    n_enc = sum(p.numel() for p in mae.encoder.parameters())
    n_dec = sum(p.numel() for n, p in mae.named_parameters() if not n.startswith("encoder."))
    print("[modelo] %s | encoder=%.1fM decoder=%.1fM params | timm %s"
          % (args.arquitetura, n_enc / 1e6, n_dec / 1e6, timm.__version__), flush=True)

    opt = torch.optim.AdamW(grupos_parametros(mae, args.lr, args.lr_encoder_mult, args.wd),
                            lr=args.lr, betas=(0.9, 0.95))

    info_sem_rotulo = None
    if args.sem_rotulo:
        pasta = args.sem_rotulo if os.path.isdir(args.sem_rotulo) \
            else os.path.join(BASE, "dados", args.sem_rotulo)
        info = json.load(open(os.path.join(pasta, "info.json"), encoding="utf-8"))
        if info["bandas"] != bandas or tuple(info["shape"][1:]) != tuple(X_tr.shape[1:]):
            raise SystemExit("%s: bandas/shape %s %s diferentes do dataset rotulado %s %s"
                             % (pasta, info["bandas"], info["shape"][1:], bandas,
                                list(X_tr.shape[1:])))
        with open(os.path.join(pasta, "indice.csv"), newline="", encoding="utf-8") as f:
            ids_mae = [int(r["i"]) for r in csv.DictReader(f) if r["uso"] == "mae"]
        leitor = LeitorBlocos(os.path.join(pasta, "patches.npy"), ids_mae, args.lote,
                              args.bloco_leitura, rng)
        passos_ep = leitor.passos
        # rotulados de treino (já na GPU) entram em cada lote na proporção do total
        n_rot = max(1, round(args.lote * len(X_tr) / len(ids_mae)))
        info_sem_rotulo = {"pasta": os.path.abspath(pasta), "patches_mae": len(ids_mae),
                           "patches_rotulados_treino": len(imgs_tr),
                           "rotulados_por_lote": n_rot, "bloco_leitura": args.bloco_leitura}
        print("[dados] pré-treino: %d sem rótulo (%s) + %d rotulados de treino (%d por lote) | "
              "%d passos/epoca" % (len(ids_mae), os.path.basename(pasta), len(imgs_tr), n_rot,
                                   passos_ep), flush=True)

        def lotes():
            for x in leitor.epoca():
                rot = X_tr[torch.from_numpy(rng.integers(0, len(X_tr), n_rot)).to(dev)]
                yield torch.cat([x.to(dev, non_blocking=True), rot])
    else:
        passos_ep = math.ceil(len(X_tr) / args.lote)

        def lotes():
            perm = rng.permutation(len(X_tr))
            for b in range(passos_ep):
                yield X_tr[torch.from_numpy(perm[b * args.lote:(b + 1) * args.lote]).to(dev)]
    total, aquec = args.epocas * passos_ep, args.aquecimento * passos_ep

    ck_path = os.path.join(saida, "ultimo.pth")
    historico, probes, inicio, melhor_probe = [], [], 0, -1.0
    if args.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        mae.load_state_dict(ck["mae"])
        opt.load_state_dict(ck["opt"])
        inicio, historico, probes = ck["epoca"], ck["historico"], ck["probes"]
        melhor_probe = ck["melhor_probe"]
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())      # map_location levou p/ a GPU
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        print("[resume] epoca %d" % inicio)

    def fazer_probe(ep):
        nonlocal melhor_probe
        t = time.monotonic()
        r = probe.rodar(mae, X_tr, rot_tr, X_va, rot_va, cenas_va)
        pc = "  ".join("%s: mIoU=%.3f" % (c, v["miou"]) for c, v in r["por_cena"].items())
        print("[probe] epoca %d | val OA=%.4f mIoU=%.4f | %s | %.0fs"
              % (ep, r["oa"], r["miou"], pc, time.monotonic() - t), flush=True)
        probes.append({"epoca": ep, **r})
        with open(os.path.join(saida, "probe.json"), "w", encoding="utf-8") as f:
            json.dump(probes, f, indent=2, ensure_ascii=False)
        if r["miou"] > melhor_probe:
            melhor_probe = r["miou"]
            mae.exportar_encoder(os.path.join(saida, "encoder_melhor_probe.pth"), epoca=ep,
                                 probe_val_miou=r["miou"], bandas=bandas)
        return r

    aug_kw = dict(escala=(args.escala_min, 1.0), ganho_pan=args.ganho_pan,
                  offset_pan=args.offset_pan)
    if inicio == 0:
        exemplos_aug(X_tr, args.img, np.random.default_rng(0),
                     os.path.join(saida, "exemplos_aug.png"), aug_kw)
        fazer_probe(0)

    campos = ["epoca", "lr", "loss_treino", "loss_val"] + ["val_" + b for b in bandas] + \
             ["probe_oa", "probe_miou", "encoder_congelado", "tempo_s"]

    def salvar_checkpoint(epoca_feita):
        with open(os.path.join(saida, "log.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=campos)
            wr.writeheader()
            wr.writerows(historico)
        torch.save({"mae": mae.state_dict(), "opt": opt.state_dict(), "epoca": epoca_feita,
                    "historico": historico, "probes": probes, "melhor_probe": melhor_probe,
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(), "args": vars(args)},
                   ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

    livre = c_livre_gb()
    if livre < args.min_livre_c_gb:
        raise SystemExit("[trava] C: com %.2f GB livres (< %.1f GB): não vou começar"
                         % (livre, args.min_livre_c_gb))
    print("[trava] C: com %.2f GB livres; para sozinho se cair abaixo de %.1f GB"
          % (livre, args.min_livre_c_gb), flush=True)
    t_inicio = time.monotonic()
    for ep in range(inicio, args.epocas):
        mae.train()
        congelado = ep < args.congelar
        for p in mae.encoder.parameters():
            p.requires_grad_(not congelado)
        soma, n, t0 = 0.0, 0, time.monotonic()
        parar = False
        for b, lote in enumerate(lotes()):
            if b % 20 == 0 and c_livre_gb() < args.min_livre_c_gb:
                parar = True
                break
            x = aumentar(lote, args.img, rng, **aug_kw)
            lr = lr_cosseno(ep * passos_ep + b, total, aquec, args.lr)
            for gr in opt.param_groups:
                gr["lr"] = lr * gr["lr_mult"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _, _ = mae(x, args.razao_mascara)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in mae.parameters() if p.requires_grad], 3.0)
            opt.step()
            soma += float(loss.detach()) * len(x)
            n += len(x)

        if parar:
            print("[trava] C: com %.2f GB livres (< %.1f GB): checkpoint salvo com %d epochs "
                  "completos; parando. Retome com --resume." % (c_livre_gb(), args.min_livre_c_gb, ep),
                  flush=True)
            salvar_checkpoint(ep)
            raise SystemExit(2)

        reg = {"epoca": ep + 1, "lr": lr, "loss_treino": soma / n,
               "encoder_congelado": int(congelado), "tempo_s": round(time.monotonic() - t0, 2)}
        ultima = ep + 1 == args.epocas
        if (ep + 1) % args.val_cada == 0 or ultima or ep == inicio:
            lv, pb = validar(mae, X_va, args.img, args.razao_mascara, args.seed)
            reg["loss_val"] = lv
            reg.update({"val_" + b: v for b, v in zip(bandas, pb)})
            visualizar_recon(mae, X_va, nomes_va, cenas_va, bandas, args.img,
                             args.razao_mascara, args.seed,
                             os.path.join(saida, "recon", "epoca_%04d.png" % (ep + 1)),
                             args.n_png, "VALIDAÇÃO epoca %d" % (ep + 1))
        if (ep + 1) % args.probe_cada == 0 or ultima:
            r = fazer_probe(ep + 1)
            reg.update(probe_oa=r["oa"], probe_miou=r["miou"])
        historico.append(reg)

        if "loss_val" in reg or ep < inicio + 3:
            print("epoca %4d/%d | lr=%.2e | loss treino=%.4f%s%s | %.1fs/epoca"
                  % (ep + 1, args.epocas, lr, reg["loss_treino"],
                     " val=%.4f (%s)" % (reg["loss_val"], " ".join(
                         "%s=%.3f" % (b[:4], reg["val_" + b]) for b in bandas))
                     if "loss_val" in reg else "",
                     " [encoder congelado]" if congelado else "", reg["tempo_s"]), flush=True)
            salvar_checkpoint(ep + 1)

    mae.exportar_encoder(os.path.join(saida, "encoder.pth"), epoca=args.epocas, bandas=bandas)
    with open(os.path.join(saida, "relatorio.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "dataset": os.path.abspath(args.dataset),
                   "bandas": bandas, "patches": {"treino": len(X_tr), "validacao": len(X_va)},
                   "sem_rotulo": info_sem_rotulo,
                   "params": {"encoder": n_enc, "decoder": n_dec},
                   "tempo_treino_s": round(time.monotonic() - t_inicio, 1),
                   "timm": timm.__version__, "torch": torch.__version__,
                   "ultimo": historico[-1] if historico else None,
                   "probe_inicial": probes[0] if probes else None,
                   "probe_final": probes[-1] if probes else None,
                   "melhor_probe_miou": melhor_probe},
                  f, indent=2, ensure_ascii=False)
    print("OK -> %s  (encoder.pth = só o encoder; o decoder MAE ficou para trás)" % saida)


if __name__ == "__main__":
    main()
