#!/usr/bin/env python3
"""XGBoost (CUDA) para LULC — classificação por pixel.

Cada pixel é uma amostra com 10 features: 4 bandas (PAN, LocalSTD,
GLCM_Entropy, Gradient) + 4 vizinhos PAN (N/S/L/O) + media e desvio PAN em
janela 7x7. Labels = índice de classe (config.CLASS_MAP), nodata descartado.

Treino usa device="cuda" (xgboost >= 2.0; wheel PyPI inclui runtime CUDA,
em WSL2 usa o driver CUDA do Windows). Sem GPU, use --device cpu.

Uso (treino):
  .venv/bin/python scripts/modelos/xgb_lulc.py \
      --saida modelos/xgb_01 --max-amostras 3000000

  # retomar de checkpoint (se interrompido):
  .venv/bin/python scripts/modelos/xgb_lulc.py \
      --saida modelos/xgb_01 --resume

Uso (inferencia -> GeoTIFF na grade da PAN):
  .venv/bin/python scripts/modelos/xgb_lulc.py \
      --inferir modelos/xgb_01/modelo.ubj \
      --imagem dados/wv1/features/pan_features.tif \
      --saida modelos/xgb_01/mascara_xgb.tif
"""

import argparse
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import config  # noqa: E402
import dataset_lulc as dl  # noqa: E402

try:
    import xgboost as xgb
except ImportError:
    print("xgboost nao instalado. Instale no venv: "
          ".venv/bin/pip install xgboost", file=sys.stderr)
    sys.exit(1)

try:
    from osgeo import gdal
except ImportError:
    gdal = None

LOOKUP_CLASSE = np.array(config.REPR_CODIGO, dtype=np.uint8)


class _CheckpointCallback(xgb.callback.TrainingCallback):
    """Salva checkpoint.ubj a cada N iterações (atômico via .tmp + rename)."""

    def __init__(self, path, interval):
        super().__init__()
        self.path = path
        self.interval = interval
        self.ultimo = -1

    def after_iteration(self, model, epoch, evals_log):
        if epoch - self.ultimo >= self.interval:
            self.ultimo = epoch
            tmp = self.path + ".tmp"
            model.save_model(tmp)
            os.replace(tmp, self.path)
        return False


def build_features(im):
    """im: (H, W, 4) float32 -> (H*W, 10) float32."""
    pan = im[..., 0]
    p = np.pad(pan, 1, mode="edge")
    vizinhos = np.stack([
        p[0:-2, 1:-1], p[2:, 1:-1], p[1:-1, 0:-2], p[1:-1, 2:],
    ], axis=-1)
    j = 3
    pan7 = np.pad(pan, j, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pan7, (2 * j + 1, 2 * j + 1))
    media7 = win.mean(axis=(2, 3))
    var7 = win.var(axis=(2, 3))
    desvio7 = np.sqrt(np.maximum(var7, 0))
    return np.concatenate(
        [im, vizinhos, media7[..., None], desvio7[..., None]], axis=-1).reshape(-1, 10)


def amostrar_patch(im, mk, cap_por_classe, rng, aug_agua=0, cap_agua=0):
    """Amostra pixels por classe (cap por classe). cap_agua>0 substitui o cap
    da classe água (0 = todos os pixels de água). Se aug_agua>0 e o patch tem
    água, adiciona flip h, flip v e rot90 dos pixels de água (x3 amostras)."""
    im32 = im.astype(np.float32)
    y = np.full(mk.shape, -1, dtype=np.int64)
    for code, cl in config.CLASS_MAP.items():
        y[mk == code] = cl
    sel_ids = []
    for cl in range(config.N_CLASSES):
        ids = np.flatnonzero(y.ravel() == cl)
        cap = cap_por_classe
        if cl == config.IDX_AGUA and cap_agua is not None:
            cap = cap_agua if cap_agua > 0 else len(ids)
        if len(ids) > cap:
            ids = rng.choice(ids, cap, replace=False)
        sel_ids.append(ids)
    sel = np.concatenate(sel_ids) if sel_ids else np.array([], dtype=np.int64)
    rng.shuffle(sel)
    X = build_features(im32)[sel]
    Y = y.ravel()[sel]

    if aug_agua > 0:
        tem_agua = int((y == config.IDX_AGUA).sum())
        if tem_agua > 0:
            im_aug = [np.flip(im32, 0), np.flip(im32, 1),
                      np.rot90(im32, 1, axes=(0, 1))]
            y_aug = [np.flip(y, 0), np.flip(y, 1), np.rot90(y, 1)]
            for fim, ym in zip(im_aug, y_aug):
                F = build_features(fim)
                m = ym.ravel() == config.IDX_AGUA
                X = np.concatenate([X, F[m]])
                Y = np.concatenate([Y, np.full(int(m.sum()), config.IDX_AGUA,
                                               dtype=np.int64)])
    return X, Y


def coletar(idxs, base_dir, cap_por_classe, max_amostras, seed, aug_agua=0,
            cap_agua=None):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    tot = 0
    t0 = time.time()
    for k, idx in enumerate(idxs):
        im = np.load(os.path.join(base_dir, "img", "%06d.npz" % idx))["arr_0"]
        mk = np.load(os.path.join(base_dir, "mask", "%06d.npz" % idx))["arr_0"]
        X, y = amostrar_patch(im, mk, cap_por_classe, rng, aug_agua, cap_agua)
        Xs.append(X)
        ys.append(y)
        tot += len(y)
        if max_amostras and tot >= max_amostras:
            Xs = Xs[:k + 1]
            ys = ys[:k + 1]
            break
        if (k + 1) % 100 == 0:
            print("  patch %4d/%d  amostras=%d (%.0fs)"
                  % (k + 1, len(idxs), tot, time.time() - t0), flush=True)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def pesos_por_classe(y):
    n = len(y)
    uniq, counts = np.unique(y, return_counts=True)
    w = np.empty(n, dtype=np.float64)
    for cl, c in zip(uniq, counts):
        w[y == cl] = n / (config.N_CLASSES * max(c, 1))
    return w


def metricas_por_classe(pred, lbl):
    """Acurácia (recall), precisão, F1 e IoU por classe."""
    acc = np.zeros(config.N_CLASSES)
    prec = np.zeros(config.N_CLASSES)
    f1 = np.zeros(config.N_CLASSES)
    iou = np.zeros(config.N_CLASSES)
    for c in range(config.N_CLASSES):
        p = pred == c
        g = lbl == c
        inter = np.logical_and(p, g).sum()
        acc[c] = inter / g.sum() if g.sum() else float("nan")
        prec[c] = inter / p.sum() if p.sum() else float("nan")
        union = np.logical_or(p, g).sum()
        iou[c] = inter / union if union else float("nan")
        r, pr = acc[c], prec[c]
        f1[c] = 2 * r * pr / (r + pr) if r + pr else float("nan")
    return acc, prec, f1, iou


def relatorio(rotulo, pred, y):
    acc = (pred == y).mean()
    acc_c, prec_c, f1_c, iou_c = metricas_por_classe(pred, y)
    linhas = [rotulo, "OA=%.4f" % acc, "por classe (acuracia=recall):"]
    linhas.append("  %2s  %-22s  %7s  %7s  %7s  %7s"
                  % ("cl", "nome", "acurac", "precis", "F1", "IoU"))
    for i in range(config.N_CLASSES):
        nome = config.CLASS_NOMES[i] if i < len(config.CLASS_NOMES) else "?"
        linhas.append("  %2d  %-22s  %7.4f  %7.4f  %7.4f  %7.4f"
                      % (i, nome, acc_c[i], prec_c[i], f1_c[i], iou_c[i]))
    print("\n".join(linhas))
    return {"rotulo": rotulo, "oa": float(acc),
            "acuracia": [float("nan") if np.isnan(v) else float(v)
                         for v in acc_c],
            "precisao": [float("nan") if np.isnan(v) else float(v)
                         for v in prec_c],
            "f1": [float("nan") if np.isnan(v) else float(v) for v in f1_c],
            "iou": [float("nan") if np.isnan(v) else float(v) for v in iou_c]}


def treinar(args):
    manifest = os.path.join(args.treino, "manifest.csv")
    idxs = dl.ler_idxs(manifest, frac=args.frac, seed=args.seed)
    if args.usar_splits:
        tr_idxs, va_idxs, te_idxs = dl.ler_splits(args.treino)
        print("[data] usando splits gravados em %s/splits/" % args.treino)
    elif args.split_estratificado:
        tr_idxs, va_idxs, te_idxs = dl.split_estratificado_agua(
            idxs, args.treino, args.val_frac, args.test_frac, args.seed)
        dl.salvar_splits(args.treino, tr_idxs, va_idxs, te_idxs)
        print("  -> splits estratificados gravados em %s/splits/"
              % args.treino)
    else:
        tr_idxs, va_idxs, te_idxs = dl.split_train_test_val(
            idxs, args.val_frac, args.test_frac, args.seed)
    print("[data] patches=%d | treino=%d | val=%d | teste=%d"
          % (len(idxs), len(tr_idxs), len(va_idxs), len(te_idxs)))

    Xtr, ytr = coletar(tr_idxs, args.treino, args.cap_por_classe,
                       args.max_amostras, args.seed, aug_agua=args.aug_agua,
                       cap_agua=args.cap_agua)
    Xva, yva = coletar(va_idxs, args.treino, args.cap_por_classe, 0, args.seed)
    Xte, yte = coletar(te_idxs, args.treino, args.cap_por_classe, 0, args.seed)
    print("[data] amostras: treino=%d (aug_agua=%d cap_agua=%d) val=%d "
          "teste=%d | features=%d"
          % (len(ytr), args.aug_agua, args.cap_agua, len(yva), len(yte),
             Xtr.shape[1]))

    wtr = pesos_por_classe(ytr)

    params = {
        "objective": "multi:softprob",
        "num_class": config.N_CLASSES,
        "tree_method": "hist",
        "device": args.device,
        "eta": args.lr,
        "max_depth": args.profundidade,
        "subsample": args.subsample,
        "colsample_bytree": 0.8,
        "min_child_weight": 4,
        "eval_metric": "mlogloss",
        "seed": config.SEED,
        "nthread": args.nthread,
    }
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dva = xgb.DMatrix(Xva, label=yva)

    os.makedirs(args.saida, exist_ok=True)
    checkpoint_path = os.path.join(args.saida, "checkpoint.ubj")
    modelo_path = os.path.join(args.saida, "modelo.ubj")

    bst0 = None
    if args.resume and os.path.exists(checkpoint_path):
        bst0 = xgb.Booster()
        bst0.load_model(checkpoint_path)
        print("[checkpoint] retomando de %s (arborização atual=%d)"
              % (checkpoint_path, bst0.num_boosted_rounds()))

    t0 = time.time()
    try:
        bst = xgb.train(params, dtr, num_boost_round=args.n_estimadores,
                        evals=[(dtr, "train"), (dva, "val")],
                        early_stopping_rounds=args.early_stopping,
                        verbose_eval=args.verbose,
                        xgb_model=bst0,
                        callbacks=[_CheckpointCallback(checkpoint_path,
                                                       args.checkpoint_interval)])
    except KeyboardInterrupt:
        if os.path.exists(checkpoint_path):
            print("[checkpoint] interrompido; retome com --resume. "
                  "ultimo checkpoint: %s" % checkpoint_path, file=sys.stderr)
        else:
            print("[checkpoint] interrompido antes do 1o checkpoint",
                  file=sys.stderr)
        raise

    print("[treino] %.0fs | best_iteration=%d"
          % (time.time() - t0, bst.best_iteration))

    bst.save_model(modelo_path)

    dte = xgb.DMatrix(Xte, label=yte)
    pred_tr = bst.predict(dtr, iteration_range=(0, bst.best_iteration + 1)).argmax(1)
    pred_va = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1)).argmax(1)
    pred_te = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1)).argmax(1)
    r_tr = relatorio("TREINO", pred_tr, ytr)
    r_va = relatorio("VALIDACAO", pred_va, yva)
    r_te = relatorio("TESTE", pred_te, yte)

    with open(os.path.join(args.saida, "relatorio.json"), "w") as f:
        json.dump({"args": vars(args), "best_iteration": bst.best_iteration,
                   "treino": r_tr, "validacao": r_va, "teste": r_te,
                   "features": config.BANDAS + ["PAN_N", "PAN_S", "PAN_L",
                                                "PAN_O", "PAN_media7",
                                                "PAN_desvio7"]},
                  f, indent=2, ensure_ascii=False)
    print("OK -> %s" % args.saida)


def avaliar(args):
    """Avalia um modelo salvo nos splits gravados (sem retreinar)."""
    modelo = os.path.join(args.saida, "modelo.ubj")
    if not os.path.exists(modelo):
        print("modelo nao encontrado: %s" % modelo, file=sys.stderr)
        sys.exit(1)
    tr_idxs, va_idxs, te_idxs = dl.ler_splits(args.treino)
    print("[data] treino=%d | val=%d | teste=%d"
          % (len(tr_idxs), len(va_idxs), len(te_idxs)))

    bst = xgb.Booster()
    bst.load_model(modelo)
    n = bst.num_boosted_rounds()
    print("[modelo] %s | arborizacao=%d | best=%d"
          % (modelo, n, getattr(bst, "best_iteration", n)))

    rels = {}
    for rotulo, idxs in (("TREINO", tr_idxs), ("VALIDACAO", va_idxs),
                         ("TESTE", te_idxs)):
        X, y = coletar(idxs, args.treino, args.cap_por_classe, 0, args.seed)
        dm = xgb.DMatrix(X, label=y)
        pred = bst.predict(dm, iteration_range=(0, n)).argmax(1)
        rels[rotulo.lower()] = relatorio(rotulo, pred, y)

    with open(os.path.join(args.saida, "relatorio.json"), "w") as f:
        json.dump({"args": vars(args), "best_iteration": n,
                   "treino": rels["treino"], "validacao": rels["validacao"],
                   "teste": rels["teste"],
                   "features": config.BANDAS + ["PAN_N", "PAN_S", "PAN_L",
                                                "PAN_O", "PAN_media7",
                                                "PAN_desvio7"]},
                  f, indent=2, ensure_ascii=False)
    print("OK -> %s/relatorio.json" % args.saida)


def build_features_gpu(t, j=3):
    """t: (1,4,H,W) float32 CUDA -> (1,10,H,W) CUDA.

    Igual a build_features (CPU): vizinhos PAN N/S/L/O + media/desvio 7x7,
    mas com ops torch (avg_pool/unfold), duas passadas p/ var (mesma
    semantica do np.var: media dos desvios ao quadrado)."""
    import torch
    import torch.nn.functional as F
    pan = t[:, 0:1]
    p = F.pad(pan, (1, 1, 1, 1), mode="replicate")
    viz = torch.cat([
        p[:, :, 0:-2, 1:-1], p[:, :, 2:, 1:-1],
        p[:, :, 1:-1, 0:-2], p[:, :, 1:-1, 2:],
    ], dim=1)
    p7 = F.pad(pan, (j, j, j, j), mode="replicate")
    win = p7.unfold(2, 2 * j + 1, 1).unfold(3, 2 * j + 1, 1)
    media = win.mean(dim=(4, 5))
    cent = win - media[..., None, None]
    var = cent.pow(2).mean(dim=(4, 5))
    desvio = torch.sqrt(torch.clamp(var, min=0))
    return torch.cat([t, viz, media, desvio], dim=1)


def inferir(args):
    """Inferencia GPU por tiles com overlay (margem >= contexto das
    features), gravando so a regiao central de cada tile. Progresso por
    tile e gravado em <saida>.progress; retoma onde parou se o arquivo
    existir."""
    if gdal is None:
        print("osgeo.gdal indisponivel", file=sys.stderr)
        sys.exit(1)
    try:
        import torch
    except ImportError:
        print("torch indisponivel (necessario p/ inferencia GPU)", file=sys.stderr)
        sys.exit(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA indisponivel; use --device cpu", file=sys.stderr)
        sys.exit(1)

    dev = "cuda" if args.device == "cuda" else "cpu"
    bst = xgb.Booster()
    bst.load_model(args.inferir)
    bst.set_param({"device": args.device})
    print("[modelo] %s | arborizacao=%d" % (args.inferir,
                                            bst.num_boosted_rounds()))

    src = gdal.Open(args.imagem, gdal.GA_ReadOnly)
    nx, ny = src.RasterXSize, src.RasterYSize
    if src.RasterCount < 4:
        print("imagem de features deve ter 4 bandas", file=sys.stderr)
        sys.exit(1)
    geo, proj = src.GetGeoTransform(), src.GetProjection()

    prog_path = args.saida + ".progress"
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

    tile, ovl = args.tile, args.overlay
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
                feats = build_features_gpu(t).reshape(10, -1).t().contiguous()
                valid = t[0, 0].reshape(-1) > 0
                pred = torch.zeros(len(feats), dtype=torch.uint8, device=dev)
                if bool(valid.any()):
                    dm = xgb.DMatrix(feats[valid].cpu().numpy())
                    cl = bst.predict(dm).argmax(1)
                    del dm
                    pred[valid] = lookup[torch.from_numpy(
                        np.asarray(cl).astype(np.int64)).to(dev)]
                ih, iw = cy1 - y0, cx1 - x0
                iy0, ix0 = y0 - ay0, x0 - ax0
                inner = pred.reshape(ay1 - ay0, ax1 - ax0)[
                    iy0:iy0 + ih, ix0:ix0 + iw]
                dst.GetRasterBand(1).WriteArray(inner.cpu().numpy(), x0, y0)
                dst.FlushCache()
                fprog.write("%d %d\n" % (x0, y0))
                fprog.flush()
                del t, feats, pred, inner
                torch.cuda.empty_cache()
                n_tiles += 1
                if n_tiles % 5 == 0 or n_tiles == total:
                    smi = ""
                    try:
                        import subprocess
                        smi = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"],
                            text=True).strip()
                    except Exception:
                        pass
                    print("  tiles=%d/%d (%.0fs, ~%.0fs restantes, "
                          "vram=%sMB)" % (n_tiles, total, time.time() - t0,
                                          (time.time() - t0) / n_tiles
                                          * (total - n_tiles), smi),
                          flush=True)
    dst = None
    src = None
    print("OK -> %s (%.0fs, %d tiles, device=%s)"
          % (args.saida, time.time() - t0, n_tiles, dev))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treino", default=config.DEFAULT_TREINO)
    ap.add_argument("--saida", default=os.path.join(config.DEFAULT_SAIDA, "xgb_01"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=config.SPLIT_VAL)
    ap.add_argument("--test-frac", type=float, default=config.SPLIT_TEST)
    ap.add_argument("--split-estratificado", action="store_true",
                    help="splits com os dois tipos de agua em cada partição")
    ap.add_argument("--usar-splits", action="store_true",
                    help="usar splits/{treino,validacao,teste}.txt existentes")
    ap.add_argument("--device", default="cuda", help="'cuda' ou 'cpu'")
    ap.add_argument("--max-amostras", type=int, default=3000000,
                    help="cap de amostras (pixels) de treino")
    ap.add_argument("--cap-por-classe", type=int, default=4000,
                    help="cap de pixels por classe em cada patch")
    ap.add_argument("--aug-agua", type=int, default=1,
                    help="quando patch tem agua, incluir flip h/v + rot90 "
                         "dos pixels de agua (1 = ligado)")
    ap.add_argument("--cap-agua", type=int, default=0,
                    help="cap de pixels de agua no treino (0 = todos)")
    ap.add_argument("--n-estimadores", type=int, default=2000)
    ap.add_argument("--profundidade", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--early-stopping", type=int, default=50,
                    help="parada antecipada se val nao melhora em N rodadas")
    ap.add_argument("--checkpoint-interval", type=int, default=100,
                    help="salvar checkpoint.ubj a cada N rodadas")
    ap.add_argument("--resume", action="store_true",
                    help="retomar treino do checkpoint.ubj existente em --saida")
    ap.add_argument("--verbose", type=int, default=50)
    ap.add_argument("--nthread", type=int, default=0)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--inferir", metavar="MODELO.ubj",
                    help="modo inferencia: modelo salvo p/ aplicar na imagem")
    ap.add_argument("--avaliar", action="store_true",
                    help="avaliar modelo.ubj de --saida nos splits gravados")
    ap.add_argument("--imagem", default=config.DEFAULT_IMAGEM)
    ap.add_argument("--tile", type=int, default=1024,
                    help="tile de inferencia (GPU); menor = menos VRAM")
    ap.add_argument("--overlay", type=int, default=64,
                    help="margem de sobreposicao (>=3 p/ contexto 7x7)")
    args = ap.parse_args()

    if args.avaliar:
        avaliar(args)
    elif args.inferir:
        if not args.saida.endswith(".tif"):
            args.saida = os.path.join(os.path.dirname(args.inferir),
                                      "mascara_xgb.tif")
        inferir(args)
    else:
        treinar(args)


if __name__ == "__main__":
    main()