#!/usr/bin/env python3
"""Treinamento do U-Net (leve, UNet++ ou UNet++/ResNet pre-treinado).

Uso (treino com labels manuais e splits do XGB, comparacao justa):
  .venv/bin/python scripts/modelos/train.py \
      --treino dados/wv1/patches/treino_manual --saida modelos/unet_manual \
      --usar-splits --arquitetura unetpp_r34

  # retomar de checkpoint (se interrompido):
  ... --resume

Uso (inferencia -> GeoTIFF na grade da PAN):
  .venv/bin/python scripts/modelos/train.py \
      --inferir modelos/unet_manual/unet_best.pth \
      --saida modelos/unet_manual/mascara_unet.tif

Arquiteturas (--arquitetura):
  unetleve    UNetLeve com SE (3.38M params)
  unetpp      UNet++ com deep supervision, treinado do zero (~n0=32)
  unetpp_r18  UNet++ com encoder ResNet18 pre-treinado (ImageNet)
  unetpp_r34  UNet++ com encoder ResNet34 pre-treinado (default)

Obs: a 1a execucao de unetpp_r18/r34 baixa os pesos ImageNet do servidor
oficial do PyTorch (cache em ~/.cache/torch). Use --no-pretrained para
pular o download (encoder aleatorio).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import config  # noqa: E402
import dataset_lulc as dl  # noqa: E402
from modelo_unet import UNetLeve, UNetPP, UNetPPResNet  # noqa: E402
import xgb_lulc as xl  # noqa: E402  (metricas_por_classe/relatorio: mesmo schema)

try:
    from osgeo import gdal
except ImportError:
    gdal = None

# pesos da deep supervision nos 4 heads X^{0,1..4} (o final domina)
PESOS_SUP = [1.0, 0.5, 0.25, 0.125]


def criar_modelo(arquitetura, se=True, pretrained=True):
    """Instancia o modelo pela arquitetura (strings de --arquitetura)."""
    if arquitetura == "unetleve":
        return UNetLeve(se=se)
    if arquitetura == "unetpp":
        return UNetPP(se=se)
    backbone = arquitetura.split("_")[2]  # unetpp_r18|r34
    return UNetPPResNet(backbone=backbone, pretrained=pretrained, se=se)


def logits_finais(saida):
    """Saida final do modelo: tensor logits (aceita tupla da deep
    supervision — UNetLeve devolve tensor direto)."""
    return saida[0] if isinstance(saida, tuple) else saida


def loss_modelo(saida, y, alpha, gamma):
    """Loss do modelo; com deep supervision soma os 4 heads ponderados."""
    if isinstance(saida, tuple):
        _, sups = saida
        tot = None
        for j, p in enumerate(sups):
            l = PESOS_SUP[j] * focal_dice_loss(p, y, alpha, gamma)
            tot = l if tot is None else tot + l
        return tot
    return focal_dice_loss(saida, y, alpha, gamma)


def pesos_classes(idxs, base_dir):
    """Pesos por classe = inverso da frequencia de pixels no split de treino,
    cap 10x a media (evita peso extremo p/ a agua), normalizado media=1."""
    counts = np.zeros(config.N_CLASSES, dtype=np.float64)
    for idx in idxs:
        mk = np.load(os.path.join(base_dir, "mask", "%06d.npz" % idx))["arr_0"]
        for code, cl in config.CLASS_MAP.items():
            counts[cl] += int((mk == code).sum())
    freq = counts / max(counts.sum(), 1)
    w = 1.0 / np.maximum(freq, 1e-6)
    w = np.minimum(w, 10.0 * w.mean())
    w = w / w.mean()
    print("[loss] pesos por classe: %s" % np.round(w, 3).tolist())
    return w


def focal_dice_loss(logits, lbl, alpha, gamma=2.0):
    """0.5*Focal(alpha,gamma) + 0.5*Dice — combina foco em pixels
    dificeis/raros com uma metrica invariante a escala por classe."""
    valid = lbl != config.IGNORE
    y = torch.where(valid, lbl, torch.zeros_like(lbl))
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(y, config.N_CLASSES).permute(0, 3, 1, 2).float()
    vf = valid.float().unsqueeze(1)
    pv = probs * vf
    gv = onehot * vf

    pt = (pv * gv).sum(dim=1)
    av = alpha[y.clamp(max=config.N_CLASSES - 1)].unsqueeze(1)
    fl = -(av * (1 - pt).pow(gamma) * pt.clamp_min(1e-7).log())
    focal = (fl * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    inter = (pv * gv).sum(dim=(0, 2, 3))
    den = pv.sum(dim=(0, 2, 3)) + gv.sum(dim=(0, 2, 3))
    dice_c = 2 * inter / den.clamp_min(1e-6)
    presentes = gv.sum(dim=(0, 2, 3)) > 0
    dice = dice_c[presentes].mean() if bool(presentes.any()) else \
        torch.tensor(1.0, device=logits.device)
    return 0.5 * focal + 0.5 * (1.0 - dice)


def avaliar_splits(model, idxs, base_dir, batch, dev, amp=True):
    """Predicao full-512 (sem aug) e metricas por classe; devolve
    (dict relatorio, mIoU medio)."""
    dset = dl.LULCDataset(base_dir, idxs, aug=False, seed=config.SEED)
    load = torch.utils.data.DataLoader(dset, batch_size=batch, shuffle=False,
                                       num_workers=0)
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for x, y in load:
            x = x.to(dev)
            with autocast(enabled=amp):
                logits = logits_finais(model(x))
            preds.append(logits.argmax(1).cpu().numpy())
            lbls.append(y.numpy())
    pred = np.concatenate(preds).reshape(-1)
    lbl = np.concatenate(lbls).reshape(-1)
    m = lbl != config.IGNORE
    acc_c, prec_c, f1_c, iou_c = xl.metricas_por_classe(pred[m], lbl[m])
    iou = np.nanmean(iou_c)
    return {"rotulo": "", "oa": float((pred[m] == lbl[m]).mean()),
            "acuracia": [float("nan") if np.isnan(v) else float(v)
                         for v in acc_c],
            "precisao": [float("nan") if np.isnan(v) else float(v)
                         for v in prec_c],
            "f1": [float("nan") if np.isnan(v) else float(v) for v in f1_c],
            "iou": [float("nan") if np.isnan(v) else float(v) for v in iou_c],
            "_n": int(m.sum())}, iou


def treinar(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = os.path.join(args.treino, "manifest.csv")
    idxs = dl.ler_idxs(manifest, frac=1.0, seed=args.seed)
    if args.usar_splits:
        tr_idxs, va_idxs, te_idxs = dl.ler_splits(args.treino)
    else:
        tr_idxs, va_idxs, te_idxs = dl.split_train_test_val(
            idxs, args.val_frac, args.test_frac, args.seed)
    print("[data] patches=%d | treino=%d | val=%d | teste=%d"
          % (len(idxs), len(tr_idxs), len(va_idxs), len(te_idxs)))

    dev = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print("[device] %s" % dev)

    dset_tr = dl.LULCDataset(args.treino, tr_idxs, aug=True, seed=args.seed)
    dset_va = dl.LULCDataset(args.treino, va_idxs, aug=False, seed=args.seed)
    load_tr = torch.utils.data.DataLoader(dset_tr, batch_size=args.batch,
                                          shuffle=True, num_workers=args.workers,
                                          pin_memory=True, drop_last=True)
    load_va = torch.utils.data.DataLoader(dset_va, batch_size=args.eval_batch,
                                          shuffle=False, num_workers=0)

    model = criar_modelo(args.arquitetura, args.se, args.pretrained).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print("[modelo] %s se=%s pretrained=%s params=%d (%.2fM)"
          % (args.arquitetura, args.se, args.pretrained, nparams,
             nparams / 1e6))

    alpha = torch.tensor(pesos_classes(tr_idxs, args.treino),
                         dtype=torch.float32, device=dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=0.0)
    scaler = GradScaler(enabled=args.amp)

    hist = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_iou, best_ep, ep0 = -1.0, 0, 1
    ckpt_path = os.path.join(args.saida, "checkpoint.pth")
    best_path = os.path.join(args.saida, "unet_best.pth")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        ep0, best_iou, best_ep = ck["epoch"] + 1, ck["best_iou"], ck["best_ep"]
        hist = ck["hist"]
        print("[checkpoint] retomando da epoca %d (best_iou=%.4f ep=%d)"
              % (ep0 - 1, best_iou, best_ep))

    for ep in range(ep0, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for x, y in load_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            for _ in range(args.crops_pp):
                if args.crops > 0 and x.size(-1) > args.crops:
                    s = args.crops
                    c0 = int(torch.randint(0, x.size(-2) - s + 1, ()))
                    c1 = int(torch.randint(0, x.size(-1) - s + 1, ()))
                    xc = x[:, :, c0:c0 + s, c1:c1 + s]
                    yc = y[:, c0:c0 + s, c1:c1 + s]
                else:
                    xc, yc = x, y
                optim.zero_grad()
                with autocast(enabled=args.amp):
                    loss = loss_modelo(model(xc), yc, alpha, args.gamma)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                tot += loss.item()
                nb += 1

        model.eval()
        vloss, viou = 0.0, 0.0
        nvb = 0
        with torch.no_grad():
            for x, y in load_va:
                x, y = x.to(dev), y.to(dev)
                with autocast(enabled=args.amp):
                    logits = logits_finais(model(x))
                    vloss += focal_dice_loss(logits, y, alpha,
                                             args.gamma).item()
                pred = logits.argmax(1).cpu().numpy()
                m = y.cpu().numpy() != config.IGNORE
                viou += np.nanmean(xl.metricas_por_classe(pred[m], y.cpu().numpy()[m])[3])
                nvb += 1
        vloss /= nvb
        viou /= nvb
        sched.step()
        hist["train_loss"].append(tot / nb)
        hist["val_loss"].append(vloss)
        hist["val_iou"].append(viou)
        print("ep %02d  train=%.4f  val=%.4f  val_iou=%.4f  lr=%.2e  (%.0fs)"
              % (ep, tot / nb, vloss, viou, sched.get_last_lr()[0],
                 time.time() - t0), flush=True)

        melhorou = viou > best_iou
        if melhorou:
            best_iou, best_ep = viou, ep
            torch.save(model.state_dict(), best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
        torch.save({"epoch": ep, "model": model.state_dict(),
                    "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best_iou": best_iou,
                    "best_ep": best_ep, "hist": hist},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        if ep - best_ep >= args.patience:
            print("[early stop] val_iou nao melhora ha %d epocas (best=%d)"
                  % (args.patience, best_ep))
            break

    model.load_state_dict(torch.load(best_path, map_location=dev))
    print("[best] epoca %d val_iou=%.4f" % (best_ep, best_iou))
    rels = {}
    for rotulo, s_idxs in (("treino", tr_idxs), ("validacao", va_idxs),
                           ("teste", te_idxs)):
        r, _ = avaliar_splits(model, s_idxs, args.treino, args.eval_batch, dev,
                              args.amp)
        r["rotulo"] = rotulo.upper()
        print("\n%s  n=%d" % (rotulo.upper(), r["_n"]))
        print("  OA=%.4f  IoU=%s" % (r["oa"], np.round(r["iou"], 4).tolist()))
        r.pop("_n", None)
        rels[rotulo] = r

    hist["args"] = vars(args)
    with open(os.path.join(args.saida, "historico.json"), "w") as f:
        json.dump(hist, f, indent=2)
    with open(os.path.join(args.saida, "relatorio.json"), "w") as f:
        json.dump({"args": vars(args), "best_iteration": best_ep,
                   "treino": rels["treino"], "validacao": rels["validacao"],
                   "teste": rels["teste"], "features": config.BANDAS},
                  f, indent=2, ensure_ascii=False)
    print("OK -> %s" % args.saida)


def inferir(args):
    if gdal is None:
        print("osgeo.gdal indisponivel", file=sys.stderr)
        sys.exit(1)
    dev = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print("[device] %s" % dev)

    model = criar_modelo(args.arquitetura, args.se, args.pretrained).to(dev)
    model.load_state_dict(torch.load(args.inferir, map_location=dev))
    model.eval()
    print("[modelo] %s (%s, se=%s)" % (args.inferir, args.arquitetura, args.se))

    src = gdal.Open(args.imagem, gdal.GA_ReadOnly)
    nx, ny = src.RasterXSize, src.RasterYSize
    if src.RasterCount < 4:
        print("imagem de features deve ter 4 bandas", file=sys.stderr)
        sys.exit(1)
    geo, proj = src.GetGeoTransform(), src.GetProjection()

    prog_path = args.saida + ".progress"
    tile, ovl = args.tile, args.overlay
    feitos = set()
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            linhas = f.read().split()
        cab = linhas[0] if linhas else ""
        if cab == "tile=%d ovl=%d" % (tile, ovl):
            feitos = {tuple(int(v) for v in ln.split())
                      for ln in linhas[1:] if ln.strip()}
            print("[progresso] %d tiles ja concluidos; retomando" % len(feitos))
        else:
            print("[progresso] grade de tiles antiga (%s); recomecando"
                  % (cab or "vazia"))

    if os.path.exists(args.saida) and feitos:
        dst = gdal.Open(args.saida, gdal.GA_Update)
    else:
        drv = gdal.GetDriverByName("GTiff")
        dst = drv.Create(args.saida, nx, ny, 1, gdal.GDT_Byte,
                         ["TILED=YES", "COMPRESS=DEFLATE"])
        dst.SetGeoTransform(geo)
        dst.SetProjection(proj)

    media = torch.tensor(config.MEAN, dtype=torch.float32, device=dev).view(1, 4, 1, 1)
    desvio = torch.tensor(config.STD, dtype=torch.float32, device=dev).view(1, 4, 1, 1)
    lookup = torch.tensor(config.REPR_CODIGO, dtype=torch.uint8, device=dev)

    t0 = time.time()
    n_tiles = 0
    total = ((ny + tile - 1) // tile) * ((nx + tile - 1) // tile)
    with open(prog_path, "a") as fprog:
        if not feitos:
            fprog.write("tile=%d ovl=%d\n" % (tile, ovl))
            fprog.flush()
        for y0 in range(0, ny, tile):
            for x0 in range(0, nx, tile):
                if (x0, y0) in feitos:
                    n_tiles += 1
                    continue
                cy1, cx1 = min(y0 + tile, ny), min(x0 + tile, nx)
                ay0, ax0 = max(0, y0 - ovl), max(0, x0 - ovl)
                ay1, ax1 = min(ny, cy1 + ovl), min(nx, cx1 + ovl)
                bandas = np.stack([src.GetRasterBand(b).ReadAsArray(
                    ax0, ay0, ax1 - ax0, ay1 - ay0)
                    for b in (1, 2, 3, 4)], axis=-1).astype(np.float32)
                t = torch.from_numpy(bandas).permute(2, 0, 1).unsqueeze(0).to(dev)
                t = (t - media) / desvio
                valid = t[0, 0] > -media[0, 0, 0, 0] / desvio[0, 0, 0, 0]
                pred = torch.zeros(1, 1, ay1 - ay0, ax1 - ax0,
                                   dtype=torch.uint8, device=dev)
                with torch.no_grad():
                    with autocast(enabled=args.amp):
                        logits = logits_finais(model(t))
                    cl = logits.argmax(1)
                pred[0, 0][valid] = lookup[cl[valid]]
                dst.GetRasterBand(1).WriteArray(pred[0, 0].cpu().numpy(), ax0, ay0)
                dst.FlushCache()
                fprog.write("%d %d\n" % (x0, y0))
                fprog.flush()
                del t, logits, cl, pred, bandas
                torch.cuda.empty_cache()
                n_tiles += 1
                if n_tiles % 10 == 0 or n_tiles == total:
                    smi = ""
                    try:
                        import subprocess
                        smi = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"],
                            text=True).strip()
                    except Exception:
                        pass
                    print("  tiles=%d/%d (%.0fs, ~%.0fs restantes, vram=%sMB)"
                          % (n_tiles, total, time.time() - t0,
                             (time.time() - t0) / n_tiles * (total - n_tiles),
                             smi), flush=True)
    dst = None
    src = None
    print("OK -> %s (%.0fs, %d tiles)" % (args.saida, time.time() - t0, n_tiles))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treino", default=config.DEFAULT_TREINO)
    ap.add_argument("--saida", default=os.path.join(config.DEFAULT_SAIDA, "unet_01"))
    ap.add_argument("--arquitetura", default="unetpp_r34",
                    choices=["unetleve", "unetpp", "unetpp_r18", "unetpp_r34"],
                    help="arquitetura (default: unetpp_r34 = UNet++/ResNet34)")
    ap.add_argument("--no-pretrained", action="store_false", dest="pretrained",
                    default=True,
                    help="nao baixar pesos ImageNet do encoder (init aleatoria)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=None,
                    help="batch de treino (default: 8 p/ ResNet, 24 senao)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--crops", type=int, default=256,
                    help="tamanho do crop aleatorio de treino (0 = 512 cheio)")
    ap.add_argument("--crops-pp", type=int, default=4,
                    help="crops aleatorios por patch por epoca")
    ap.add_argument("--eval-batch", type=int, default=None,
                    help="batch da validacao/teste 512x512 (default: 4 p/ "
                         "ResNet, 8 senao)")
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="gamma do focal loss")
    ap.add_argument("--patience", type=int, default=15,
                    help="early stop em epocas sem melhora de val_iou")
    ap.add_argument("--usar-splits", action="store_true",
                    help="usar splits/{treino,validacao,teste}.txt existentes")
    ap.add_argument("--val-frac", type=float, default=config.SPLIT_VAL)
    ap.add_argument("--test-frac", type=float, default=config.SPLIT_TEST)
    ap.add_argument("--resume", action="store_true",
                    help="retomar treino do checkpoint.pth em --saida")
    ap.add_argument("--device", default="cuda", help="'cuda' ou 'cpu'")
    ap.add_argument("--amp", action="store_true", default=True,
                    help="mixed precision (fp16) ligado")
    ap.add_argument("--no-se", action="store_false", dest="se", default=True,
                    help="desliga o SE attention do modelo")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--inferir", metavar="MODELO.pth",
                    help="modo inferencia: pesos salvos p/ aplicar na imagem")
    ap.add_argument("--imagem", default=config.DEFAULT_IMAGEM)
    ap.add_argument("--tile", type=int, default=None,
                    help="tile de inferencia (default: 512 p/ ResNet, "
                         "1024 senao); menor = menos VRAM")
    ap.add_argument("--overlay", type=int, default=64,
                    help="margem de sobreposicao (contexto nas bordas)")
    args = ap.parse_args()

    if args.batch is None:
        args.batch = 8 if args.arquitetura.startswith("unetpp_r") else 24
    if args.eval_batch is None:
        args.eval_batch = 4 if args.arquitetura.startswith("unetpp_r") else 8
    if args.tile is None:
        args.tile = 512 if args.arquitetura.startswith("unetpp_r") else 1024

    os.makedirs(args.saida, exist_ok=True)
    if args.inferir:
        if not args.saida.endswith(".tif"):
            args.saida = os.path.join(os.path.dirname(args.inferir),
                                      "mascara_unet.tif")
        inferir(args)
    else:
        treinar(args)


if __name__ == "__main__":
    main()