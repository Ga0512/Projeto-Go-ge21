"""Leitura do dataset de patches SPOT5 (Data/dados/datasets/<nome>/).

Estrutura esperada (gerada por Data/scripts/datasets/gerar_dataset_refinado.py):
  images/<nome>.tif   3 bandas Float32, já em z-score por cena
  masks/<nome>.tif    Byte, códigos MapBiomas, 0 = sem rótulo
  splits/{treino,validacao,teste}.txt
  manifest.csv        (cena de origem de cada patch)
  receita.json        (nomes das bandas)

O mapeamento código MapBiomas -> classe vem de Data/scripts/modelos/config.py,
a fonte única de CLASS_MAP do projeto.
"""

import csv
import json
import io
import os
import sys

import numpy as np

from ambiente import MODELS

DATA = os.path.normpath(os.path.join(MODELS, "..", "Data"))
DATASET_PADRAO = os.path.join(DATA, "dados", "datasets", "spot5_refinado_v6b")

sys.path.insert(0, os.path.join(DATA, "scripts", "modelos"))
import config as cfg  # noqa: E402  (Data/scripts/modelos/config.py)

from osgeo import gdal  # noqa: E402

gdal.UseExceptions()


def _classes(dataset, juntar_agua=False):
    """-> (CLASS_MAP, CLASS_NOMES, N_CLASSES, REPR_CODIGO) do dataset, detectando
    '--juntar-veg' (3 classes: Vegetação / Não vegetação / Água) pelo manifest.csv
    (coluna 'frac_veg' em vez de 'frac_veg_densa')."""
    with open(os.path.join(dataset, "manifest.csv"), newline="", encoding="utf-8") as f:
        cols = next(csv.reader(f))
    if "frac_veg" in cols and "frac_veg_densa" not in cols:
        cp = {}
        for cod, idx in cfg.CLASS_MAP.items():
            cp[cod] = 0 if idx <= 1 else (1 if idx == 2 else 2)
        if juntar_agua:
            # Água vira "Não vegetação". NÃO é padrão: existe para datasets onde
            # a água é rara demais para ser aprendida e, pior, para ser MEDIDA.
            # No wv_jatai_v2 ela é 0,1% do treino e 0,4% do teste; o U-Net++
            # zerou essa classe em 4 dos 11 epochs, e como o mIoU é média das
            # três, ela sozinha descartou o epoch 8 (Veg 0,776 / NVeg 0,921)
            # em favor do epoch 1 (0,616 / 0,839). Com 2 classes a seleção
            # volta a medir o que o modelo faz em 99,9% dos pixels.
            cp = {cod: min(i, 1) for cod, i in cp.items()}
            return (cp, ["Vegetação", "Não vegetação"], 2, [3, 15])
        return (cp, ["Vegetação", "Não vegetação", "Água"], 3, [3, 15, 33])
    return (cfg.CLASS_MAP, cfg.CLASS_NOMES, cfg.N_CLASSES, cfg.REPR_CODIGO)


CLASS_MAP, CLASS_NOMES, N_CLASSES, REPR_CODIGO = _classes(DATASET_PADRAO)
IGNORE = cfg.IGNORE

# cores oficiais da legenda MapBiomas (LEGEND em Data/scripts/*) dos REPR_CODIGO
CORES_CODIGO = {3: "#1f8d49", 4: "#7dc975", 15: "#edde8e", 33: "#2532e4"}
CORES_CLASSE = [CORES_CODIGO[c] for c in REPR_CODIGO]

LUT = np.full(256, IGNORE, dtype=np.uint8)
for _codigo, _classe in CLASS_MAP.items():
    LUT[_codigo] = _classe


def configurar(dataset, juntar_agua=False):
    """Reavalia CLASS_MAP/CLASS_NOMES/N_CLASSES/REPR_CODIGO/LUT/CORES para o dataset
    dado (o nº de classes muda entre datasets: v13=4, v15=3). Chamado pelo train.py
    com o --dataset real antes de carregar."""
    global CLASS_MAP, CLASS_NOMES, N_CLASSES, REPR_CODIGO, LUT, CORES_CLASSE
    CLASS_MAP, CLASS_NOMES, N_CLASSES, REPR_CODIGO = _classes(dataset, juntar_agua)
    CORES_CLASSE = [CORES_CODIGO[c] for c in REPR_CODIGO]
    LUT = np.full(256, IGNORE, dtype=np.uint8)
    for _codigo, _classe in CLASS_MAP.items():
        LUT[_codigo] = _classe


def bandas(dataset):
    with open(os.path.join(dataset, "receita.json"), encoding="utf-8") as f:
        return json.load(f)["bandas"]


def cenas(dataset):
    """arquivo -> cena de origem (manifest.csv)."""
    with open(os.path.join(dataset, "manifest.csv"), newline="", encoding="utf-8") as f:
        return {r["arquivo"]: r["cena"] for r in csv.DictReader(f)}


def ler_split(dataset, split):
    with open(os.path.join(dataset, "splits", split + ".txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _ler_tif(caminho):
    ds = gdal.Open(caminho)
    arr = ds.ReadAsArray()
    ds = None
    return arr


def carregar(dataset, nomes, dtype=np.float32):
    """-> imgs (N,C,H,W) no dtype pedido, rotulos (N,H,W) uint8 em índices de
    classe (IGNORE = sem rótulo, código fora do CLASS_MAP ou não finito).

    Aloca o array final de uma vez e preenche no lugar. O np.stack() anterior
    segurava a lista E a cópia empilhada ao mesmo tempo, dobrando o pico de
    RAM: com os patches de 512 px e 4 bandas do wv_jatai_v3 isso pedia 16,8 GB
    e o treino morria sem deixar log (WSL com teto de 11 GB numa máquina de
    15,4 GB).

    `dtype`: os train*.py mandam o X para a GPU em float16 logo em seguida,
    entao passar np.float16 aqui nao perde nenhum valor e corta o pico pela
    metade. O `brutos` que sobra para a visualizacao fica igualmente menor.
    """
    imgs, rotulos = None, None
    fora_do_mapa = {}
    for i, nome in enumerate(nomes):
        im = _ler_tif(os.path.join(dataset, "images", nome + ".tif")).astype(np.float32)
        mk = _ler_tif(os.path.join(dataset, "masks", nome + ".tif"))
        if im.ndim != 3 or im.shape[1:] != mk.shape:
            raise ValueError("%s: imagem %s e mascara %s incompatíveis"
                             % (nome, im.shape, mk.shape))
        y = LUT[mk]
        desconhecidos = (y == IGNORE) & (mk != 0)
        if desconhecidos.any():
            for c in np.unique(mk[desconhecidos]):
                fora_do_mapa[int(c)] = fora_do_mapa.get(int(c), 0) + int((mk == c).sum())
        invalido = ~np.isfinite(im).all(axis=0)
        if invalido.any():
            im[:, invalido] = 0
            y[invalido] = IGNORE
        if imgs is None:
            imgs = np.empty((len(nomes),) + im.shape, dtype)
            rotulos = np.empty((len(nomes),) + y.shape, y.dtype)
        imgs[i] = im
        rotulos[i] = y
    if fora_do_mapa:
        print("[dados] AVISO: códigos fora do CLASS_MAP (ignorados): %s" % fora_do_mapa)
    if imgs is None:
        raise ValueError("split vazio: nenhum patch para carregar")
    return imgs, rotulos


def carregar_cache(dataset, split, nomes, dtype=np.float32, usar=True):
    '''carregar() com cache em .npy, lido por mmap.

    POR QUE. Abrir 2.000 GeoTIFF comprimidos um a um leva 14 min no
    wv_jatai_v3 -- mais que uma epoca inteira -- e o custo se repete a cada
    execucao. Calibrar --lote exige varias execucoes, entao isso passa a
    dominar o tempo de trabalho. Com o cache, da segunda vez em diante sai em
    segundos.

    A chave guarda a lista de nomes e o dtype: mudou o split, o dataset ou a
    precisao, o cache e refeito. Nada e escrito fora de <dataset>/cache/, e
    apagar essa pasta so custa uma releitura.

    mmap_mode='r' devolve array read-only paginado sob demanda pelo SO. O
    torch.from_numpy avisa que nao e gravavel; e inofensivo aqui porque o
    treino so le (o indexamento por lote ja devolve copia).
    '''
    base = os.path.join(dataset, "cache")
    nome_dt = np.dtype(dtype).name
    cx = os.path.join(base, "%s_x_%s.npy" % (split, nome_dt))
    cy = os.path.join(base, "%s_y.npy" % split)
    ck = os.path.join(base, "%s_%s.chave" % (split, nome_dt))
    nl = chr(10)
    chave = str(len(nomes)) + nl + nl.join(nomes)
    if usar and os.path.exists(cx) and os.path.exists(cy) and os.path.exists(ck):
        try:
            if io.open(ck, encoding="utf-8").read() == chave:
                print("[cache] %s lido de %s" % (split, base), flush=True)
                return (np.load(cx, mmap_mode="r"), np.load(cy, mmap_mode="r"))
            print("[cache] %s: lista de patches mudou, refazendo" % split)
        except Exception as e:
            print("[cache] %s: ilegivel (%s), refazendo" % (split, e))
    x, y = carregar(dataset, nomes, dtype)
    if usar:
        try:
            os.makedirs(base, exist_ok=True)
            np.save(cx, x)
            np.save(cy, y)
            io.open(ck, "w", encoding="utf-8").write(chave)
            print("[cache] %s gravado (%.2f GB)"
                  % (split, (x.nbytes + y.nbytes) / 2**30), flush=True)
        except Exception as e:
            print("[cache] %s: nao gravou (%s); seguindo sem cache" % (split, e))
    return x, y
