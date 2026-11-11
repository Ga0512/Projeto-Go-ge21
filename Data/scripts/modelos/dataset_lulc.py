#!/usr/bin/env python3
"""Dataset PyTorch dos patches de treino LULC (npz img/mask).

Lê em tempo de treino:  mask/<idx>.npz (códigos MapBiomas, 0=nodata)
  e img/<idx>.npz (4 bandas float16). Normaliza por banda (config.MEAN/STD),
  mapeia códigos -> 0..N-1 (config.CLASS_MAP) e nodata -> IGNORE.
  Augment determinística (flip h/v, rot 90) aplicada em img+mask juntas.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

import config


class LULCDataset(Dataset):
    def __init__(self, base_dir, idxs, aug=False, seed=config.SEED):
        self.img_dir = os.path.join(base_dir, "img")
        self.mask_dir = os.path.join(base_dir, "mask")
        self.idxs = idxs
        self.aug = aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx = self.idxs[i]
        im = np.load(os.path.join(self.img_dir, "%06d.npz" % idx))["arr_0"]
        mk = np.load(os.path.join(self.mask_dir, "%06d.npz" % idx))["arr_0"]

        im = im.astype(np.float32)
        im = (im - np.array(config.MEAN, np.float32)) / np.array(config.STD, np.float32)

        lbl = np.full(mk.shape, config.IGNORE, dtype=np.int64)
        for code, cl in config.CLASS_MAP.items():
            lbl[mk == code] = cl

        if self.aug:
            im, lbl = self._aug(im, lbl)

        return torch.from_numpy(im.transpose(2, 0, 1).copy()), torch.from_numpy(lbl)

    def _aug(self, im, lbl):
        if self.rng.random() < 0.5:
            im, lbl = im[::-1, :, :], lbl[::-1, :]
        if self.rng.random() < 0.5:
            im, lbl = im[:, ::-1, :], lbl[:, ::-1]
        k = int(self.rng.integers(0, 4))
        if k:
            im = np.rot90(im, k, axes=(0, 1))
            lbl = np.rot90(lbl, k)
        return im.copy(), lbl.copy()


def ler_idxs(manifest_path, frac=1.0, seed=config.SEED):
    """Índices (ordenação do manifest) com subamostragem uniforme opcional."""
    idxs = []
    with open(manifest_path) as f:
        next(f)  # header
        for line in f:
            idxs.append(int(line.split(",")[0]))
    idxs.sort()
    if frac < 1.0:
        rng = np.random.default_rng(seed)
        idxs = sorted(rng.choice(idxs, size=int(round(len(idxs) * frac)),
                                 replace=False).tolist())
    return idxs


def split_train_val(idxs, val_frac=config.SPLIT_VAL, seed=config.SEED):
    """Split determinístico por seed; mantém consistência entre execuções."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(idxs))
    nval = int(round(len(idxs) * val_frac))
    return [idxs[i] for i in perm[nval:]], [idxs[i] for i in perm[:nval]]


def split_train_test_val(idxs, val_frac=config.SPLIT_VAL,
                         test_frac=config.SPLIT_TEST, seed=config.SEED):
    """Split 3 vias (treino/validacao/teste), determinístico por seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(idxs))
    nval = int(round(len(idxs) * val_frac))
    ntest = int(round(len(idxs) * test_frac))
    va = [idxs[i] for i in perm[:nval]]
    te = [idxs[i] for i in perm[nval:nval + ntest]]
    tr = [idxs[i] for i in perm[nval + ntest:]]
    return tr, va, te


def salvar_splits(base_dir, tr, va, te):
    """Grava listas de índices em splits/{treino,validacao,teste}.txt."""
    import os
    d = os.path.join(base_dir, "splits")
    os.makedirs(d, exist_ok=True)
    for nome, idxs in (("treino", tr), ("validacao", va), ("teste", te)):
        with open(os.path.join(d, nome + ".txt"), "w") as f:
            f.write("\n".join("%06d" % i for i in sorted(idxs)) + "\n")


def ler_splits(base_dir):
    """Lê splits/{treino,validacao,teste}.txt (escritos por salvar_splits)."""
    import os
    d = os.path.join(base_dir, "splits")
    out = {}
    for nome in ("treino", "validacao", "teste"):
        with open(os.path.join(d, nome + ".txt")) as f:
            out[nome] = [int(x) for x in f.read().split()]
    return out["treino"], out["validacao"], out["teste"]


def split_estratificado_agua(idxs, base_dir, val_frac=config.SPLIT_VAL,
                             test_frac=config.SPLIT_TEST, seed=config.SEED):
    """Split 3 vias onde cada split contém os dois tipos de água (escura e
    clara, pela mediana do PAN dos pixels de água). Garante cobertura de
    classes raras em todos os splits; sem isso o sorteio puro pode mandar
    um tipo inteiro de água só pro treino ou só pro teste."""
    import numpy as np

    tr, va, te = split_train_test_val(idxs, val_frac, test_frac, seed)

    tipos = {}   # idx -> 'escura'|'clara' (só patches com água)
    n_px = {}    # idx -> n° de pixels de água (p/ doar o menor possível)
    for i in tr + va + te:
        mk = np.load(os.path.join(base_dir, "mask", "%06d.npz" % i))["arr_0"]
        c = int((mk == 33).sum())
        if c == 0:
            continue
        n_px[i] = c
        im = np.load(os.path.join(base_dir, "img", "%06d.npz" % i))["arr_0"]
        pan_med = im[..., 0][mk == 33].mean()
        tipos[i] = "escura" if pan_med < 22 else "clara"

    splits = {"treino": tr, "validacao": va, "teste": te}
    for nome in splits:
        n = {t: sum(tipos.get(i) == t for i in splits[nome])
             for t in ("escura", "clara")}
        print("  split %-10s agua: escura=%d clara=%d"
              % (nome, n["escura"], n["clara"]))

    # rebalanceia priorizando TREINO RICO: cada split sem um tipo recebe um
    # patch daquele tipo; doador = split com mais sobra do tipo (val/teste
    # preferidos antes de treino) e doa o patch com MENOS pixels de água.
    for t in ("escura", "clara"):
        for nome in ("teste", "validacao", "treino"):
            if any(tipos.get(i) == t for i in splits[nome]):
                continue
            candidatos = [o for o in splits
                          if o != nome and
                          sum(tipos.get(i) == t for i in splits[o]) >= 2]
            if not candidatos:
                continue

            def excesso(o):
                return sum(tipos.get(i) == t for i in splits[o])

            doador_nome = max(candidatos, key=lambda o: (excesso(o),
                                                         o != "treino"))
            doador = min((i for i in splits[doador_nome]
                          if tipos.get(i) == t), key=lambda i: n_px[i])
            sem_agua = [i for i in splits[nome] if i not in tipos]
            if not sem_agua:
                continue
            troca = sem_agua[0]
            splits[nome].remove(troca)
            splits[nome].append(doador)
            splits[doador_nome].remove(doador)
            splits[doador_nome].append(troca)
            print("  -> %s recebe agua %s (patch %d, %d px); %s recebe %d"
                  % (nome, t, doador, n_px[doador], doador_nome, troca))
    return splits["treino"], splits["validacao"], splits["teste"]