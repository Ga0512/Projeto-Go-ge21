#!/usr/bin/env python3
"""Dataset de patches (imagem 3 bandas + máscara) a partir de pares PAN + máscara.

Cada cena entra como PAN + máscara total (códigos MapBiomas, 0 = sem rótulo)
+ peso (% do dataset). A máscara pode ser um GeoTIFF ou uma pasta tiles/ do
editor (tiles_cena.py): aí só os tiles com mascara_refinada.tif entram e o
resto conta como sem rótulo. Tudo (cenas, parâmetros, estatísticas, arquivos
de entrada) vai para <saida>/receita.json; rodar com --receita refaz o
dataset idêntico.

Pipeline por cena:
  1. Grade comum: UTM a --resolucao m (cena geográfica vira a UTM do seu
     centro). Cena já nessa grade não é reamostrada; senão a PAN vai com
     cúbica e a máscara SEMPRE com nearest.
  2. Estatísticas da cena (blocos sorteados na cena inteira): p2/p98 da PAN
     (esticamento que deixa as texturas comparáveis entre cenas de brilho
     diferente) e média/desvio de cada banda (z-score por cena).
  3. Janelas candidatas (--patch) em qualquer posição de uma grade fina
     (passo --stride) com >= --min-rotulo de pixels rotulados e PAN 100%
     válida.
  4. Split espacial: blocos de --bloco px percorridos em serpentina;
     validação e teste pegam trechos seguidos (regiões contíguas, pouca
     fronteira). O ponto de partida é o que deixa validação e teste com a
     distribuição de classes mais parecida com a da cena. Janela que encosta
     em blocos de splits diferentes é descartada (nenhum pixel é
     compartilhado entre splits).
  5. Sorteio com teto de repetição: as candidatas são visitadas em ordem
     aleatória e uma janela só entra se não divide mais que o teto da sua
     área com nenhuma já escolhida da cena.
  6. Balanceamento: --total patches (0 = o máximo que cabe), cada cena com
     total x peso, repartidos pelos splits. As cotas são calculadas com o
     teto --max-sobreposicao; depois cada cena usa o MENOR teto (passos de
     5%) que ainda entrega a sua cota, então cena grande repete pouco e só a
     cena pequena chega perto do teto. Se uma cena não comporta a sua cota,
     todas encolhem juntas (a proporção manda).
  7. Grava images/<cena>_<y>_<x>.tif, masks/<cena>_<y>_<x>.tif, manifest.csv,
     splits/, receita.json e visualizacao/. --simular só mostra quanto cabe.

Bandas das imagens (Float32, z-score por cena):
  1 PAN          PAN original
  2 GLCM         entropia GLCM (janela 7, 8 níveis, offset (0,1)) sobre a
                 PAN esticada p2–p98
  3 Rugosidade   média de z(LocalSTD 7x7) e z(Gradiente Sobel) sobre a PAN
                 esticada (as duas medem rugosidade por caminhos diferentes)
Máscaras: Byte, códigos MapBiomas (39/41/62 unificados em 19), 0 = sem rótulo.

Uso:
  .venv/bin/python scripts/datasets/gerar_dataset_refinado.py \\
      --saida dados/datasets/spot5_refinado --total 500 \\
      --cena dados/spot5/recorte_2007-09-10/original/IMAGEM_SPOT5_10092007.tif \\
             dados/spot5/recorte_2007-09-10/mascara_refinada.tif 50 \\
      --cena dados/spot5/cena_013-014_2007-06-14/original/013-014_S5_708-378-0_2007-06-14_modificado.tif \\
             dados/spot5/cena_013-014_2007-06-14/tiles 50
  # refazer idêntico (mesmas entradas -> mesmos arquivos):
  .venv/bin/python scripts/datasets/gerar_dataset_refinado.py \\
      --receita dados/datasets/spot5_refinado/receita.json --sobrescrever
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import scipy.ndimage as ndi
from osgeo import gdal, osr

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
sys.path.insert(0, os.path.join(AQUI, "..", "wv1"))
sys.path.insert(0, os.path.join(AQUI, "..", "spot5"))
sys.path.insert(0, os.path.join(AQUI, "..", "modelos"))
import caminhos  # noqa: E402
import config  # noqa: E402
from gerar_features_pan import features_block  # noqa: E402
from gerar_mascara_mapbiomas_cena import LEGEND  # noqa: E402
from suavizar_mascara import CLASS_REMAP  # noqa: E402

gdal.UseExceptions()

PARAMETROS = {
    "resolucao": 2.5,      # m, grade comum
    "patch": 256,          # px
    "stride": 32,          # px entre posições candidatas (grade fina = mais visões)
    "bloco": 512,          # px, unidade do split espacial
    "min_rotulo": 0.9,     # fração mínima de pixels rotulados na janela
    "min_valido": 1.0,     # fração mínima de PAN válida na janela. 1.0 = só janela 100%
                           # válida (o default histórico). Abaixo de 1 a janela entra e os
                           # pixels inválidos saem como SEM RÓTULO na máscara. Serve para
                           # imagem comprimida com perda, onde o escuro satura em 0 e seria
                           # lido como "sem imagem": na WV1 a água tem ~3-8% de DN=0, e
                           # exigir 100% descartava quase toda janela com água.
    "max_sobreposicao": 0.7,   # teto: fração máx. da área que um patch divide com outro
    "total": 0,            # patches no dataset (0 = o máximo que cabe)
    "janela": 7,           # janela do GLCM e do LocalSTD
    "niveis": 8,           # níveis de quantização do GLCM
    "frac_validacao": config.SPLIT_VAL,
    "frac_teste": config.SPLIT_TEST,
    "blocos_estatistica": 64,   # blocos sorteados p/ as estatísticas da cena
    "lado_estatistica": 512,
    "seed": config.SEED,
    "pan_zero_nodata": 1,  # PAN sem nodata declarado: 0 = sem dado (moldura preta de mosaico)
    "grade_nativa": 0,     # 1 = usa a grade do PAN COMO ESTÁ, sem reprojetar para UTM. Para
                           # cena geográfica enorme (a reamostragem gravaria um Float32 da cena
                           # inteira em disco) e para as bandas casarem com texturas já geradas
                           # na grade nativa. `resolucao` passa a ser só a escala métrica
                           # efetiva, usada em distâncias/áreas, não na grade.
    "zona_morta": 0,       # m entre grupos (blockCV): janela perto de bloco de outro
                           # grupo é descartada. >= alcance da autocorrelação
    "dobras": 0,           # k-fold espacial: 0 = treino/validação/teste; k = k dobras
    "ordenar_raras": 0,    # 1 = visita as candidatas em ordem decrescente de
                           # conteúdo de classe rara da cena, em vez de sorteio.
                           # Muda a COMPOSIÇÃO do dataset sem repetir tensor: o
                           # teto de sobreposição continua valendo. Use com
                           # --total para escolher onde parar na curva (ver a
                           # docstring de sortear()). Com isso ligado, a busca
                           # pelo menor teto é desligada: baixar o teto espalha
                           # as janelas e desfaz o rebalanceamento em silêncio.
    "sem_repetir_terreno": 1,  # cenas podem cobrir o mesmo chão (Colinas x 013-014): a cena
                           # seguinte não pega patches onde a anterior já rotulou
    "buffer": 0,           # m, buffer NEGATIVO nas máscaras (erode cada classe): remove a
                           # faixa de transição entre classes (pixels mistos/desalinhados) —
                           # o modelo treina com o interior representativo de cada classe
}
BANDAS = ["PAN", "GLCM_Entropia", "Rugosidade_STD_Gradiente"]
SPLITS = ["treino", "validacao", "teste"]
MARGEM = 8                 # px extras lidos em volta da janela p/ as texturas
LUT = np.arange(256, dtype=np.uint8)
for _k, _v in CLASS_REMAP.items():
    LUT[_k] = _v


# ------------------------------------------------------------ utilidades
def rel(p):
    """Caminho relativo à raiz do projeto quando está dentro dele."""
    a = os.path.abspath(p)
    return os.path.relpath(a, caminhos.RAIZ) if a.startswith(caminhos.RAIZ + os.sep) else a


def absoluto(p):
    return p if os.path.isabs(p) else os.path.join(caminhos.RAIZ, p)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ler(band, x0, y0, w, h, W, H, dtype=np.float32):
    """Lê a janela; o que cair fora da imagem é preenchido por reflexão."""
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    a = band.ReadAsArray(xa, ya, xb - xa, yb - ya)
    a = a if dtype is None else a.astype(dtype)
    pad = ((ya - y0, y0 + h - yb), (xa - x0, x0 + w - xb))
    if any(v for par in pad for v in par):
        a = np.pad(a, pad, mode="reflect")
    return a


def epsg_utm(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def erodir_mascara(masc, W, H, raio, path, gt, wkt):
    """Grava em <path> a máscara erodida (buffer NEGATIVO): cada classe encolhe
    `raio` px. Um pixel mantém a classe só se o disco de raio `raio` em volta
    for inteiramente da mesma classe — remove a faixa de transição entre
    classes (pixels mistos) e contra o sem-rótulo. Processa em faixas com
    margem de `raio` linhas (memória constante)."""
    disk = np.zeros((2 * raio + 1, 2 * raio + 1), bool)
    yy, xx = np.ogrid[-raio:raio + 1, -raio:raio + 1]
    disk[xx * xx + yy * yy <= raio * raio] = True
    d = gdal.GetDriverByName("GTiff").Create(path, W, H, 1, gdal.GDT_Byte,
                                             ["TILED=YES", "COMPRESS=DEFLATE"])
    d.SetGeoTransform(gt)
    d.SetProjection(wkt)
    ob = d.GetRasterBand(1)
    ob.SetNoDataValue(0)
    FA = 2048
    for y0 in range(0, H, FA):
        h = min(FA, H - y0)
        ya = max(0, y0 - raio)
        yb = min(H, y0 + h + raio)
        a = masc.ReadAsArray(0, ya, W, yb - ya)
        if ya > y0 or yb < y0 + h:   # margem de raio p/ o disco na borda da cena
            a = np.pad(a, ((max(0, raio - y0), max(0, y0 + h + raio - H)), (0, 0)),
                       constant_values=0)
        out = np.zeros_like(a)
        for c in np.unique(a):
            if c == 0:
                continue
            out[ndi.binary_erosion(a == c, structure=disk, border_value=0)] = c
        ob.WriteArray(out[raio:raio + h], 0, y0)
    d = None
    return gdal.Open(path)



# ---------------------------------------------------- preparação da cena
class Cena:
    """PAN e máscara na grade comum; guarda os datasets GDAL abertos."""

    def __init__(self, spec, par, tmp):
        self.nome, self.peso = spec["nome"], float(spec["peso"])
        self.pan_path = absoluto(spec["pan"])
        self.masc_path = absoluto(spec["mascara"]) if spec.get("mascara") else None
        res = par["resolucao"]
        ds = gdal.Open(self.pan_path)
        gt, srs = ds.GetGeoTransform(), osr.SpatialReference(wkt=ds.GetProjection())
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if srs.IsGeographic():
            cx = gt[0] + gt[1] * ds.RasterXSize / 2
            cy = gt[3] + gt[5] * ds.RasterYSize / 2
            alvo = osr.SpatialReference()
            alvo.ImportFromEPSG(epsg_utm(cx, cy))
        else:
            alvo = srs
        if par.get("grade_nativa"):
            nativa = True          # a grade do PAN é a grade comum: nada é reamostrado
        else:
            nativa = (srs.IsSame(alvo) and gt[2] == 0 and gt[4] == 0
                      and abs(gt[1] - res) < 1e-6 and abs(gt[5] + res) < 1e-6)
        self.reamostrada = not nativa
        self.tiles = None

        nd_origem = ds.GetRasterBand(1).GetNoDataValue()
        if nd_origem is None and par.get("pan_zero_nodata"):
            nd_origem = 0
            print("   PAN sem nodata declarado: 0 tratado como nodata", flush=True)
        if nativa:
            self.pan_ds = ds
            self.nodata = nd_origem
            self.alpha = None
        else:
            dst = os.path.join(tmp, self.nome + "_pan.tif")
            print("   reamostrando PAN -> EPSG:%s %.2f m (cúbica)"
                  % (alvo.GetAuthorityCode(None), res), flush=True)
            gdal.Warp(dst, ds, dstSRS=alvo.ExportToWkt(), xRes=res, yRes=res,
                      targetAlignedPixels=True, resampleAlg="cubic", srcNodata=nd_origem,
                      outputType=gdal.GDT_Float32, dstAlpha=True,
                      creationOptions=["TILED=YES", "COMPRESS=DEFLATE"])
            ds = None
            self.pan_ds = gdal.Open(dst)
            self.nodata = None
            self.alpha = self.pan_ds.GetRasterBand(2)
        self.W, self.H = self.pan_ds.RasterXSize, self.pan_ds.RasterYSize
        self.gt, self.wkt = self.pan_ds.GetGeoTransform(), self.pan_ds.GetProjection()
        self.epsg = osr.SpatialReference(wkt=self.wkt).GetAuthorityCode(None)
        self.pan = self.pan_ds.GetRasterBand(1)

        # máscara: GeoTIFF ou pasta tiles/ (só tiles refinados); sem máscara = cena só de imagem
        if self.masc_path is None:
            self.masc_ds = self.masc = None
            self.entradas_mascara = {}
            return
        if os.path.isdir(self.masc_path):
            meta = json.load(open(os.path.join(self.masc_path, "tiles.json"), encoding="utf-8"))
            ids = spec.get("tiles_refinados")
            if ids is None:
                ids = [t["id"] for t in meta["tiles"] if os.path.exists(
                    os.path.join(self.masc_path, t["id"], "mascara_refinada.tif"))]
            if not ids:
                raise SystemExit("%s: nenhum tile refinado em %s" % (self.nome, self.masc_path))
            self.tiles = ids
            arqs = [os.path.join(self.masc_path, i, "mascara_refinada.tif") for i in ids]
            g = meta["gt"]
            fonte = "/vsimem/%s_mascara.vrt" % self.nome
            gdal.BuildVRT(fonte, arqs, outputBounds=(g[0], g[3] + meta["h"] * g[5],
                                                     g[0] + meta["w"] * g[1], g[3]),
                          srcNodata=0, VRTNodata=0).FlushCache()
            self.entradas_mascara = {rel(a): md5(a) for a in arqs}
        else:
            fonte = self.masc_path
            self.entradas_mascara = {rel(fonte): md5(fonte)}
        self.masc_fonte = fonte     # p/ outra cena checar terreno já usado
        m = gdal.Open(fonte)
        mesma = (m.RasterXSize, m.RasterYSize) == (self.W, self.H) and \
            np.allclose(m.GetGeoTransform(), self.gt) and \
            osr.SpatialReference(wkt=m.GetProjection()).IsSame(osr.SpatialReference(wkt=self.wkt))
        if mesma:
            self.masc_ds = m
        else:
            dst = os.path.join(tmp, self.nome + "_mascara.tif")
            print("   levando a máscara para a grade da PAN (nearest)", flush=True)
            x0, y0 = self.gt[0], self.gt[3]
            gdal.Warp(dst, m, dstSRS=self.wkt, resampleAlg="near",
                      outputBounds=(x0, y0 + self.H * self.gt[5], x0 + self.W * self.gt[1], y0),
                      width=self.W, height=self.H, dstNodata=0, outputType=gdal.GDT_Byte,
                      creationOptions=["TILED=YES", "COMPRESS=DEFLATE"])
            m = None
            self.masc_ds = gdal.Open(dst)
            assert (self.masc_ds.RasterXSize, self.masc_ds.RasterYSize) == (self.W, self.H)
        self.masc = self.masc_ds.GetRasterBand(1)
        # buffer negativo: erode cada classe na grade comum (candidatos e labels usam isto)
        if par.get("buffer", 0) > 0:
            raio = max(1, int(round(par["buffer"] / res)))
            print("   buffer negativo: erodindo máscara %.0f m = %d px"
                  % (par["buffer"], raio), flush=True)
            dst = os.path.join(tmp, self.nome + "_mascara_buffer.tif")
            self.masc_ds = erodir_mascara(self.masc, self.W, self.H, raio, dst, self.gt, self.wkt)
            self.masc = self.masc_ds.GetRasterBand(1)


    def valido(self, x0, y0, w, h):
        if self.alpha is not None:
            return ler(self.alpha, x0, y0, w, h, self.W, self.H, None) > 0
        if self.nodata is None:
            return np.ones((h, w), bool)
        return ler(self.pan, x0, y0, w, h, self.W, self.H, None) != self.nodata

    def texturas(self, x0, y0, w, h, par):
        """PAN + texturas brutas (sem z-score) da janela, calculadas com
        margem da própria cena para não ter artefato de borda."""
        M, half = MARGEM, par["janela"] // 2
        pan = ler(self.pan, x0 - M, y0 - M, w + 2 * M, h + 2 * M, self.W, self.H)
        ok = self.valido(x0 - M, y0 - M, w + 2 * M, h + 2 * M)
        if not ok.all():   # nodata na margem: copia o vizinho válido mais próximo
            idx = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
            pan = pan[tuple(idx)]
        p2, p98 = self.stats["p2"], self.stats["p98"]
        s = (np.clip((pan - p2) / (p98 - p2), 0, 1) * 255 + 0.5).astype(np.uint8)
        with np.errstate(divide="ignore", invalid="ignore"):   # log(0) descartado pelo where
            _, std, ent, grad = features_block(s, s.shape[0] - 2 * half, par["janela"], par["niveis"])
        corta = (slice(M - half, M - half + h), slice(M, M + w))
        return pan[M:M + h, M:M + w], ent[corta], std[corta], grad[corta]

    def bandas(self, x0, y0, w, h, par):
        pan, ent, std, grad = self.texturas(x0, y0, w, h, par)
        z = lambda a, k: (a - self.stats["media"][k]) / self.stats["desvio"][k]  # noqa: E731
        rug = (z(std, "std") + z(grad, "grad")) / 2
        return np.stack([z(pan, "pan"), z(ent, "glcm"), z(rug, "rugosidade")]).astype(np.float32)


# --------------------------------------------------------- estatísticas
def estatisticas(c, par, rng):
    """p2/p98 da PAN e média/desvio das bandas em blocos sorteados na cena."""
    L = par["lado_estatistica"]
    pos = [(y, x) for y in range(0, c.H - L + 1, L) for x in range(0, c.W - L + 1, L)]
    rng.shuffle(pos)
    blocos, pans = [], []
    for y, x in pos:
        if len(blocos) == par["blocos_estatistica"]:
            break
        ok = c.valido(x, y, L, L)
        if ok.mean() >= 0.999:
            blocos.append((x, y))
            pans.append(ler(c.pan, x, y, L, L, c.W, c.H)[ok])
    if not blocos:
        raise SystemExit("%s: nenhum bloco %dx%d totalmente válido" % (c.nome, L, L))
    p2, p98 = np.percentile(np.concatenate(pans), [2, 98])
    c.stats = {"p2": float(p2), "p98": float(p98), "blocos": len(blocos)}
    acc = {k: [] for k in ("pan", "glcm", "std", "grad")}
    for x, y in blocos:
        pan, ent, std, grad = c.texturas(x, y, L, L, par)
        for k, a in zip(acc, (pan, ent, std, grad)):
            acc[k].append(a.ravel())
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    media = {k: float(v.mean()) for k, v in acc.items()}
    desvio = {k: float(v.std()) for k, v in acc.items()}
    rug = ((acc["std"] - media["std"]) / desvio["std"]
           + (acc["grad"] - media["grad"]) / desvio["grad"]) / 2
    media["rugosidade"], desvio["rugosidade"] = float(rug.mean()), float(rug.std())
    c.stats.update({"media": media, "desvio": desvio})


# ------------------------------------------------ candidatos e splits
def _construir_classe_lut():
    """CLASSE_LUT (código MapBiomas -> classe do modelo) do config ATUAL."""
    lut = np.full(256, 255, np.uint8)
    for _k, _v in config.CLASS_MAP.items():
        lut[_k] = _v
    return lut


CLASSE_LUT = _construir_classe_lut()


def fracoes_celulas(c, cel, validade_total=False):
    """Por célula de cel x cel: fração rotulada, fração de PAN válida e fração
    de cada classe do modelo (config.CLASS_MAP). A validade da PAN só é lida
    onde há rótulo, a menos que validade_total (patches sem rótulo)."""
    ny, nx = -(-c.H // cel), -(-c.W // cel)
    rot, val = np.zeros((ny, nx)), np.zeros((ny, nx))
    cls = np.zeros((config.N_CLASSES, ny, nx))
    # varre em blocos nos DOIS eixos: a faixa de largura inteira estourava a RAM em
    # cena muito larga (na WV1, 2048 x 274176 dá 561 MB por temporário). O lado é
    # múltiplo de `cel`, então nenhuma célula cruza a borda de um bloco: o resultado
    # é idêntico ao da varredura por faixas.
    faixa = cel * max(1, 2048 // cel)
    for y0 in range(0, c.H, faixa):
        h = min(faixa, c.H - y0)
        r0, nr = y0 // cel, -(-h // cel)
        for x0 in range(0, c.W, faixa):
            w = min(faixa, c.W - x0)
            c0, nc = x0 // cel, -(-w // cel)
            codigos = (LUT[c.masc.ReadAsArray(x0, y0, w, h)] if c.masc is not None
                       else np.zeros((h, w), np.uint8))
            m = codigos != 0

            def reduz(a, nr=nr, nc=nc):
                p = np.zeros((nr * cel, nc * cel), bool)
                p[:a.shape[0], :a.shape[1]] = a
                return p.reshape(nr, cel, nc, cel).mean(axis=(1, 3))
            rot[r0:r0 + nr, c0:c0 + nc] = reduz(m)
            if m.any() or validade_total:   # validade da PAN só onde precisa (poupa leitura)
                val[r0:r0 + nr, c0:c0 + nc] = reduz(c.valido(x0, y0, w, h))
            if m.any():
                k = CLASSE_LUT[codigos]
                for i in range(config.N_CLASSES):
                    cls[i, r0:r0 + nr, c0:c0 + nc] = reduz(k == i)
    return rot, val, cls


def celulas_ocupadas(c, fontes, cel, tmp):
    """Fração, por célula, de terreno já rotulado por cenas anteriores (as
    máscaras delas trazidas para a grade desta cena com nearest). Duas cenas
    podem cobrir o mesmo chão (Colinas x 013-014): sem isso, o mesmo lugar
    entraria no treino por uma cena e no teste pela outra."""
    ny, nx = -(-c.H // cel), -(-c.W // cel)
    acc = np.zeros((ny, nx))
    for j, fonte in enumerate(fontes):
        dst = os.path.join(tmp, "%s_ocupado%d.tif" % (c.nome, j))
        x0, y0 = c.gt[0], c.gt[3]
        gdal.Warp(dst, fonte, dstSRS=c.wkt, resampleAlg="near",
                  outputBounds=(x0, y0 + c.H * c.gt[5], x0 + c.W * c.gt[1], y0),
                  width=c.W, height=c.H, dstNodata=0, outputType=gdal.GDT_Byte,
                  creationOptions=["TILED=YES", "COMPRESS=DEFLATE"])
        ds = gdal.Open(dst)
        b = ds.GetRasterBand(1)
        faixa = cel * max(1, 2048 // cel)
        for y in range(0, c.H, faixa):
            h = min(faixa, c.H - y)
            a = b.ReadAsArray(0, y, c.W, h) != 0
            nr = -(-h // cel)
            p = np.zeros((nr * cel, nx * cel), bool)
            p[:a.shape[0], :a.shape[1]] = a
            acc[y // cel:y // cel + nr] += p.reshape(nr, cel, nx, cel).mean(axis=(1, 3))
        ds = None
    return acc


def _split_por_regiao(regioes, ys, xs, P, nome):
    """Split por regiões EXPLÍCITAS no lugar da serpentina: cada região é um
    retângulo em pixels da cena com o seu grupo. A janela só fica se couber
    INTEIRA em exatamente uma região — as faixas entre as regiões são a zona
    morta, sem precisar de `zona_morta`. Serve quando o split tem de respeitar
    um desenho conhecido (p. ex. blocos de GT isolados, e um bloco cortado em
    pedaços para distribuir uma classe rara entre os grupos)."""
    dentro = np.zeros((len(SPLITS), len(ys)), bool)
    usadas = 0
    for r in regioes:
        if r.get("cena") not in (None, nome):
            continue
        if r["grupo"] not in SPLITS:
            raise SystemExit("grupo '%s' desconhecido (esperado %s)" % (r["grupo"], SPLITS))
        g = SPLITS.index(r["grupo"])
        x0, y0, x1, y1 = r["janela"]
        dentro[g] |= (xs >= x0) & (xs + P <= x1) & (ys >= y0) & (ys + P <= y1)
        usadas += 1
    if not usadas:
        raise SystemExit("%s: nenhuma região de split para esta cena" % nome)
    n = dentro.sum(0)
    if (n > 1).any():
        raise SystemExit("%s: %d janelas caem em regiões de grupos diferentes "
                         "(regiões sobrepostas?)" % (nome, int((n > 1).sum())))
    fica = n == 1
    ref = dentro.argmax(0)
    por_split = {s: list(zip(ys[fica & (ref == i)].tolist(), xs[fica & (ref == i)].tolist()))
                 for i, s in enumerate(SPLITS)}
    return por_split, int((~fica).sum())


def candidatos_e_splits(c, par, rng, ocupado=None):
    """Janelas candidatas por split (listas de (y, x)), nº de candidatas e
    nº descartadas por cruzar a fronteira entre splits."""
    P, S, B = par["patch"], par["stride"], par["bloco"]
    rot, val, cls = fracoes_celulas(c, S)
    k = P // S

    def media_janela(a):   # média de k x k células em cada posição (tabela de somas)
        ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) / (k * k)
    ny, nx = (c.H - P) // S + 1, (c.W - P) // S + 1     # janela inteira na imagem
    ok = (media_janela(rot)[:ny, :nx] >= par["min_rotulo"]) & \
        (media_janela(val)[:ny, :nx] >= par.get("min_valido", 1.0) - 1e-6)
    if ocupado is not None:      # janela que pega terreno já usado por cena anterior: fora
        ok &= media_janela(ocupado)[:ny, :nx] <= 0
    cy, cx = np.nonzero(ok)
    ys, xs = cy * S, cx * S
    # escore = fração da janela ocupada pelas classes RARAS DA CENA (as que não
    # são a majoritária ali). É o que o --ordenar-raras usa para decidir a ordem
    # de visita; fica em `c.escores` para o sorteio achar depois.
    fc = np.stack([media_janela(cls[j])[:ny, :nx] for j in range(config.N_CLASSES)])
    tot = np.clip(fc.sum(0), 1e-9, None)
    dom = int(np.argmax(fc.sum((1, 2))))          # classe majoritária da cena
    raro = (fc.sum(0) - fc[dom]) / tot
    c.escores = {(int(a), int(b)): float(raro[i, j])
                 for a, b, i, j in zip(ys, xs, cy, cx)}
    if not len(ys):
        raise SystemExit("%s: nenhuma janela com >= %.0f%% rotulado"
                         % (c.nome, 100 * par["min_rotulo"]))
    if par.get("split_regioes"):      # regiões desenhadas à mão no lugar da serpentina
        por_split, fora = _split_por_regiao(par["split_regioes"], ys, xs, P, c.nome)
        return por_split, len(ys), fora
    # blocos pelo centro da janela, percorridos em serpentina (linha a linha,
    # alternando o sentido): validação e teste pegam trechos seguidos ->
    # regiões contíguas, pouca fronteira entre splits. O ponto de partida é o
    # que deixa validação e teste com a distribuição de classes mais parecida
    # com a da cena (empate: sorteio).
    by, bx = (ys + P // 2) // B, (xs + P // 2) // B
    blocos, contagem = np.unique(np.stack([by, bx], 1), axis=0, return_counts=True)
    serp = np.lexsort((np.where(blocos[:, 0] % 2, -blocos[:, 1], blocos[:, 1]), blocos[:, 0]))
    kb = B // S                                   # células por bloco
    pc = np.zeros((config.N_CLASSES, -(-cls.shape[1] // kb) * kb, -(-cls.shape[2] // kb) * kb))
    pc[:, :cls.shape[1], :cls.shape[2]] = cls
    cls_bloco = pc.reshape(config.N_CLASSES, pc.shape[1] // kb, kb, pc.shape[2] // kb, kb).sum(axis=(2, 4))
    cls_b = cls_bloco[:, blocos[:, 0], blocos[:, 1]].T          # (n_blocos, classes)
    dist_cena = cls_b.sum(0) / max(cls_b.sum(), 1e-9)
    # grupos: splits clássicos (validação e teste pegam blocos primeiro, o resto
    # é treino) ou k dobras iguais (k-fold espacial do blockCV)
    if par.get("dobras"):
        alvo = {i: len(ys) / len(SPLITS) for i in range(len(SPLITS) - 1)}
        resto = len(SPLITS) - 1
    else:
        alvo = {SPLITS.index("validacao"): par["frac_validacao"] * len(ys),
                SPLITS.index("teste"): par["frac_teste"] * len(ys)}
        resto = SPLITS.index("treino")

    def atribuir(inicio):
        acum = {i: 0 for i in alvo}
        sp = np.full(len(blocos), resto)
        for i in np.roll(serp, -inicio):
            s = next((s for s in alvo if acum[s] < alvo[s]), resto)
            sp[i] = s
            if s in acum:
                acum[s] += contagem[i]
        return sp

    def desvio(sp):
        d = 0.0
        for s in range(len(SPLITS)):
            v = cls_b[sp == s].sum(0)
            d += np.abs(v / max(v.sum(), 1e-9) - dist_cena).sum()
        return d
    inicios = rng.permutation(len(serp))
    melhor = min(inicios, key=lambda i: desvio(atribuir(int(i))))
    sp = atribuir(int(melhor))
    grade = np.full(((c.H - 1) // B + 1, (c.W - 1) // B + 1), -1)
    grade[blocos[:, 0], blocos[:, 1]] = sp
    # janela fica só se, dentro dela MAIS a zona morta, só há blocos de um grupo
    zm = int(round(par.get("zona_morta", 0) / par["resolucao"]))
    ny_b, nx_b = grade.shape
    ii = [np.pad((grade == gi).astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
          for gi in range(len(SPLITS))]
    y0b = np.clip((ys - zm) // B, 0, ny_b - 1); y1b = np.clip((ys + P - 1 + zm) // B, 0, ny_b - 1)
    x0b = np.clip((xs - zm) // B, 0, nx_b - 1); x1b = np.clip((xs + P - 1 + zm) // B, 0, nx_b - 1)
    cnt = np.stack([I[y1b + 1, x1b + 1] - I[y0b, x1b + 1] - I[y1b + 1, x0b] + I[y0b, x0b]
                    for I in ii])
    fica = (cnt > 0).sum(0) == 1
    ref = cnt.argmax(0)
    por_split = {s: list(zip(ys[fica & (ref == i)].tolist(), xs[fica & (ref == i)].tolist()))
                 for i, s in enumerate(SPLITS)}
    return por_split, len(ys), int((~fica).sum())


def sortear(cand, P, teto, rng, escores=None):
    """Visita as candidatas e aceita a janela se ela não divide mais que `teto`
    da área com nenhuma já aceita. Devolve as aceitas na ordem de aceitação (um
    prefixo dela é uma amostra espalhada).

    Sem `escores`, a ordem é aleatória -- o comportamento histórico. Com
    `escores` ({(y, x): valor}), visita em ordem DECRESCENTE de escore: é assim
    que se força composição de classes, já que o `balancear()` depois corta um
    prefixo. Medido na Colinas: visitando por conteúdo de classe rara, o
    prefixo de 3400 patches sai com 68/16/15 em vez dos 79/11/10 do sorteio,
    perdendo 12% de área distinta. O teto continua valendo, então as janelas
    seguem sendo recortes distintos, não repetição do mesmo tensor."""
    if not cand:
        return []
    ys, xs = np.array(cand).T
    ay, ax = np.empty(len(ys), int), np.empty(len(ys), int)
    n, lim = 0, teto * P * P + 1e-6
    if escores is None:
        ordem = rng.permutation(len(ys))
    else:
        v = np.array([escores.get((int(y), int(x)), 0.0) for y, x in zip(ys, xs)])
        # desempate aleatório: sem isso, patches de escore igual (muito comum,
        # p.ex. 0 em toda janela só de vegetação) entrariam em ordem de varredura
        ordem = np.lexsort((rng.random(len(v)), -v))
    for i in ordem:
        if n:
            ov = np.clip(P - np.abs(ax[:n] - xs[i]), 0, None) * np.clip(P - np.abs(ay[:n] - ys[i]), 0, None)
            if ov.max() > lim:
                continue
        ay[n], ax[n] = ys[i], xs[i]
        n += 1
    return list(zip(ay[:n].tolist(), ax[:n].tolist()))


def balancear(cenas, aceitas, par):
    """--total repartido pelos splits e, em cada split, pelas cenas conforme o
    peso. Cena que não comporta a cota faz todas encolherem juntas."""
    soma = sum(c.peso for c in cenas)
    frac = ({s: 1.0 / len(SPLITS) for s in SPLITS} if par.get("dobras") else
            {"treino": 1 - par["frac_validacao"] - par["frac_teste"],
             "validacao": par["frac_validacao"], "teste": par["frac_teste"]})
    cabe = {c.nome: {s: len(aceitas[c.nome][s]) for s in SPLITS} for c in cenas}
    avisos = []
    if par["total"]:
        # cota da cena = total * peso, repartida pelos splits (maior resto: fecha a conta)
        cota = {}
        for c in cenas:
            n = int(round(par["total"] * c.peso / soma))
            q = {s: n * frac[s] for s in SPLITS}
            base = {s: int(math.floor(q[s])) for s in SPLITS}
            for s in sorted(SPLITS, key=lambda s: base[s] - q[s])[:n - sum(base.values())]:
                base[s] += 1
            cota[c.nome] = base
        for s in SPLITS:   # split que não cabe encolhe em todas as cenas (mantém a proporção)
            f = min(1.0, min(cabe[c.nome][s] / cota[c.nome][s] for c in cenas if cota[c.nome][s]))
            if f < 1:
                avisos.append("%s: pedidos %d, cabem %d com sobreposição <= %.0f%%"
                              % (s, sum(cota[c.nome][s] for c in cenas),
                                 int(f * sum(cota[c.nome][s] for c in cenas)),
                                 100 * par["max_sobreposicao"]))
                for c in cenas:
                    cota[c.nome][s] = int(math.floor(cota[c.nome][s] * f + 1e-9))
    else:
        cota = {c.nome: {} for c in cenas}
        for s in SPLITS:
            T = min(cabe[c.nome][s] / (c.peso / soma) for c in cenas)
            for c in cenas:
                cota[c.nome][s] = int(math.floor(T * c.peso / soma + 1e-9))
    escolha = {c.nome: {s: sorted(aceitas[c.nome][s][:cota[c.nome][s]]) for s in SPLITS}
               for c in cenas}
    return escolha, avisos


def repeticao(sel, P, res, S=1):
    """Área distinta coberta (km²), vezes que cada pixel aparece (média nos
    pixels cobertos, máx.) e maior sobreposição entre dois patches.

    A cobertura é contada em células de S x S px, não pixel a pixel: toda
    origem de patch é múltipla do stride e P também, então a cobertura é
    uniforme dentro de cada célula e a conta é EXATA — só não aloca o
    retângulo que envolve todos os patches (na WV1 isso daria 44 GB)."""
    ys, xs = np.array(sel).T
    if P % S:                       # fora do contrato: cai para pixel a pixel
        S = 1
    y0, x0 = ys.min() // S, xs.min() // S
    k = P // S
    cob = np.zeros((ys.max() // S - y0 + k, xs.max() // S - x0 + k), np.uint16)
    for y, x in sel:
        cob[y // S - y0:y // S - y0 + k, x // S - x0:x // S - x0 + k] += 1
    # maior sobreposição entre dois patches, em blocos (a matriz n x n inteira
    # passa de 7 GB quando --simular enche a cena de candidatas)
    maior = 0
    for i in range(0, len(ys), 2048):
        dx = np.clip(P - np.abs(xs[i:i + 2048, None] - xs), 0, None)
        dy = np.clip(P - np.abs(ys[i:i + 2048, None] - ys), 0, None)
        ov = dx * dy
        ov[np.arange(ov.shape[0]), np.arange(i, min(i + 2048, len(ys)))] = 0
        maior = max(maior, int(ov.max()))
    return {"area_distinta_km2": round(float((cob > 0).sum()) * (S * res) ** 2 / 1e6, 2),
            "vezes_por_pixel_media": round(float(cob[cob > 0].mean()), 2),
            "vezes_por_pixel_max": int(cob.max()),
            "maior_sobreposicao": round(maior / P / P, 3)}


# ---------------------------------------------------------------- saída
def gravar(path, arr, gt, wkt, dtype, nodata=None, descricoes=None):
    arr = arr if arr.ndim == 3 else arr[None]
    opc = ["COMPRESS=DEFLATE", "TILED=YES", "PREDICTOR=%d" % (3 if dtype == gdal.GDT_Float32 else 2)]
    ds = gdal.GetDriverByName("GTiff").Create(path, arr.shape[2], arr.shape[1], arr.shape[0], dtype, opc)
    ds.SetGeoTransform(gt)
    ds.SetProjection(wkt)
    for i, a in enumerate(arr, 1):
        b = ds.GetRasterBand(i)
        b.WriteArray(a)
        if nodata is not None:
            b.SetNoDataValue(nodata)
        if descricoes:
            b.SetDescription(descricoes[i - 1])
    ds = None


def fracoes_classes(m):
    n = m.size
    out = {"rotulado": float((m != 0).sum()) / n}
    for i, nome in enumerate(config.CLASS_CURTOS):
        out[nome] = float(np.isin(m, [k for k, v in config.CLASS_MAP.items() if v == i]).sum()) / n
    out["outros"] = float(((m != 0) & ~np.isin(m, list(config.CLASS_MAP))).sum()) / n
    return out


def visualizar(saida, cenas, escolha, amostras, par):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    vis = os.path.join(saida, "visualizacao")
    os.makedirs(vis, exist_ok=True)
    ciclo = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    cores = {s: ciclo[i % len(ciclo)] for i, s in enumerate(SPLITS)}
    P = par["patch"]
    for c in cenas:
        xy = [p for s in SPLITS for p in escolha[c.nome][s]]
        ys, xs = [p[0] for p in xy], [p[1] for p in xy]
        x0, y0 = max(0, min(xs) - P), max(0, min(ys) - P)
        x1, y1 = min(c.W, max(xs) + 2 * P), min(c.H, max(ys) + 2 * P)
        f = max(1, max(x1 - x0, y1 - y0) // 1400)
        a = c.pan.ReadAsArray(x0, y0, x1 - x0, y1 - y0, buf_xsize=(x1 - x0) // f,
                              buf_ysize=(y1 - y0) // f).astype(np.float32)
        fig, ax = plt.subplots(figsize=(12, 12 * a.shape[0] / a.shape[1] + 0.8))
        ax.imshow(a, cmap="gray", vmin=c.stats["p2"], vmax=c.stats["p98"])
        for s in SPLITS:
            for y, x in escolha[c.nome][s]:
                ax.add_patch(Rectangle(((x - x0) / f, (y - y0) / f), P / f, P / f, fill=False,
                                       lw=0.8, ec=cores[s]))
        ax.legend(handles=[Patch(ec=cores[s], fill=False, label="%s (%d)" % (s, len(escolha[c.nome][s])))
                           for s in SPLITS], loc="upper right")
        ax.set_title("%s — patches por split" % c.nome)
        ax.axis("off")
        fig.savefig(os.path.join(vis, "cena_%s.png" % c.nome), dpi=110, bbox_inches="tight")
        plt.close(fig)

    codigos = sorted(LEGEND)
    cmap = ListedColormap([LEGEND[k][1] for k in codigos])
    lut = np.zeros(256, np.uint8)
    for i, k in enumerate(codigos):
        lut[k] = i
    fig, axs = plt.subplots(len(amostras), 4, figsize=(12, 3 * len(amostras)))
    axs = np.atleast_2d(axs)
    for lin, (nome, img, msk) in enumerate(amostras):
        x, m = gdal.Open(img).ReadAsArray(), gdal.Open(msk).ReadAsArray()
        for col, (a, t) in enumerate(zip(list(x) + [None], BANDAS + ["Máscara"])):
            ax = axs[lin, col]
            if a is None:
                ax.imshow(lut[m], cmap=cmap, vmin=0, vmax=len(codigos) - 1, interpolation="nearest")
            else:
                ax.imshow(a, cmap="gray", vmin=-2.5, vmax=2.5)
            ax.set_title("%s\n%s" % (os.path.basename(img), t) if col == 0 else t, fontsize=8)
            ax.axis("off")
    presentes = sorted({int(v) for _, _, msk in amostras for v in np.unique(gdal.Open(msk).ReadAsArray())})
    fig.legend(handles=[Patch(color=LEGEND[k][1], label="%d %s" % (k, LEGEND[k][0])) for k in presentes],
               loc="lower center", ncol=4, fontsize=8)
    fig.savefig(os.path.join(vis, "amostras.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("Uso:"):])
    ap.add_argument("--cena", nargs="+", action="append",
                    metavar="ARG",
                    help="PAN MASCARA PESO [TILE...] — PAN, máscara total "
                         "(GeoTIFF ou pasta tiles/) e peso (%% do dataset); "
                         "opcional: tiles refinados a usar (ex.: r09_c10 "
                         "r10_c10) — sem eles, todos os com mascara_refinada "
                         "entram")
    ap.add_argument("--receita", help="receita.json de um dataset anterior (refaz idêntico)")
    ap.add_argument("--saida", help="pasta do dataset (default com --receita: a pasta dela)")
    ap.add_argument("--sobrescrever", action="store_true")
    ap.add_argument("--juntar-veg", action="store_true",
                    help="v3 do modelo: vegetação densa + não densa viram UMA "
                         "classe (3 finais: Vegetação / Não vegetação / Água) — "
                         "redefine classes/config na execução, sem mexer no "
                         "config.py de modelos")
    ap.add_argument("--split-regioes", metavar="JSON",
                    help="JSON com as regiões do split no lugar da serpentina: lista de "
                         "{grupo, janela:[x0,y0,x1,y1] em px da cena, cena (opcional)}. "
                         "A janela só entra se couber inteira em uma região; as faixas "
                         "entre regiões são a zona morta")
    ap.add_argument("--simular", nargs="*", type=float, metavar="TETO",
                    help="não grava nada: mostra quantos patches cabem para cada teto de "
                         "sobreposição dado (sem valores: usa --max-sobreposicao)")
    for k, v in PARAMETROS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None,
                        help="default %s" % v)
    args = ap.parse_args()

    par = dict(PARAMETROS)
    global CLASSE_LUT
    if args.juntar_veg:
        # 4 classes -> 3: 0=Vegetação (densa+não densa), 1=Não vegetação, 2=Água
        cp = {}
        for cod, idx in config.CLASS_MAP.items():
            cp[cod] = 0 if idx <= 1 else (1 if idx == 2 else 2)
        config.CLASS_MAP = cp
        config.CLASS_NOMES = ["Vegetação", "Não vegetação", "Água"]
        config.CLASS_CURTOS = ["veg", "nao_veg", "agua"]
        config.N_CLASSES = 3
        config.REPR_CODIGO = [3, 15, 33]
        CLASSE_LUT = _construir_classe_lut()
        print("classes juntadas: veg densa + não densa -> Vegetação (3 classes)",
              flush=True)
    if args.receita:
        rec = json.load(open(args.receita, encoding="utf-8"))
        par.update(rec["parametros"])
        if "pan_zero_nodata" not in rec["parametros"]:   # receita anterior à opção: refaz idêntico
            par["pan_zero_nodata"] = 0
        specs = rec["cenas"]
        saida = args.saida or os.path.dirname(os.path.abspath(args.receita))
    else:
        if not args.cena or not args.saida:
            ap.error("informe --saida e pelo menos um --cena (ou --receita)")
        specs, nomes = [], set()
        for pan, masc, peso, *tiles in args.cena:
            nome = os.path.basename(caminhos.pasta_cena(pan))
            while nome in nomes:
                nome += "_b"
            nomes.add(nome)
            spec = {"nome": nome, "pan": rel(pan), "mascara": rel(masc), "peso": float(peso)}
            if tiles:
                spec["tiles_refinados"] = list(tiles)
            specs.append(spec)
        saida = args.saida
    for k in PARAMETROS:
        v = getattr(args, k)
        if v is not None:
            par[k] = v
    if args.split_regioes:
        par["split_regioes"] = json.load(open(args.split_regioes, encoding="utf-8"))
        par["split_regioes_arquivo"] = rel(args.split_regioes)
    if par["patch"] % par["stride"] or par["bloco"] % par["stride"] or par["janela"] % 2 == 0:
        raise SystemExit("patch e bloco devem ser múltiplos do stride; janela ímpar")
    if par.get("grade_nativa"):
        # a grade é a do PAN; `resolucao` vira só a escala métrica efetiva (distâncias e áreas)
        _d = gdal.Open(absoluto(specs[0]["pan"]))
        _g = _d.GetGeoTransform()
        _s = osr.SpatialReference(wkt=_d.GetProjection())
        _s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if _s.IsGeographic():
            _lat = _g[3] + _g[5] * _d.RasterYSize / 2.0
            _rx = abs(_g[1]) * 111320.0 * math.cos(math.radians(_lat))
            _ry = abs(_g[5]) * 110574.0
        else:
            _rx, _ry = abs(_g[1]), abs(_g[5])
        par["resolucao"] = round((_rx + _ry) / 2, 6)
        _d = None
        print("grade NATIVA do PAN (nada reamostrado); resolução efetiva %.4f m "
              "(só para distâncias e áreas)" % par["resolucao"], flush=True)
    if par.get("dobras"):        # k-fold espacial: os grupos passam a ser as dobras
        if int(par["dobras"]) < 2:
            raise SystemExit("--dobras deve ser >= 2 (0 = treino/validação/teste)")
        SPLITS[:] = ["dobra_%d" % (i + 1) for i in range(int(par["dobras"]))]
    if any(float(s["peso"]) <= 0 for s in specs):
        raise SystemExit("pesos devem ser > 0")

    simular = args.simular is not None
    saida = os.path.abspath(saida)
    if simular:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="dataset_sim_")
    else:
        geradas = ["images", "masks", "splits", "visualizacao", "manifest.csv", "_tmp"]
        if any(os.path.exists(os.path.join(saida, g)) for g in geradas):
            if not args.sobrescrever:
                raise SystemExit("%s já tem um dataset; use --sobrescrever" % saida)
            for g in geradas:
                p = os.path.join(saida, g)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.exists(p):
                    os.remove(p)
        tmp = os.path.join(saida, "_tmp")
        for d in ("images", "masks", "splits"):
            os.makedirs(os.path.join(saida, d))
        os.makedirs(tmp)

    t0 = time.time()
    P = par["patch"]
    cenas, candidatos, info = [], {}, {}
    for i, spec in enumerate(specs):
        print("[%s] preparando" % spec["nome"], flush=True)
        c = Cena(spec, par, tmp)
        if args.receita:
            for arq, h in spec.get("md5_mascara", {}).items():
                if c.entradas_mascara.get(arq) != h:
                    print("   AVISO: %s mudou desde a receita — o dataset não será idêntico" % arq)
        if not simular:
            estatisticas(c, par, np.random.default_rng([par["seed"], i, 1]))
        ocupado = None
        if par.get("sem_repetir_terreno") and cenas:
            ocupado = celulas_ocupadas(c, [x.masc_fonte for x in cenas], par["stride"], tmp)
            if ocupado.any():
                print("   terreno já rotulado por cena anterior: %.1f km² fora"
                      % (ocupado.sum() * (par["stride"] * par["resolucao"]) ** 2 / 1e6), flush=True)
        por_split, ncand, desc = candidatos_e_splits(c, par, np.random.default_rng([par["seed"], i, 2]), ocupado)
        print("   grade EPSG:%s %dx%d (%s) | posições candidatas %d (%d descartadas na fronteira "
              "de split) -> %s" % (c.epsg, c.W, c.H, "reamostrada" if c.reamostrada else "nativa",
                                   ncand, desc, {s: len(v) for s, v in por_split.items()}), flush=True)
        cenas.append(c)
        candidatos[c.nome] = por_split
        spec["md5_mascara"] = c.entradas_mascara
        if c.tiles is not None:
            spec["tiles_refinados"] = c.tiles
        info[c.nome] = {"grade": {"epsg": c.epsg, "largura": c.W, "altura": c.H,
                                  "geotransform": list(c.gt), "reamostrada": c.reamostrada},
                        "candidatas": ncand, "descartadas_fronteira": desc,
                        "candidatas_por_split": {s: len(v) for s, v in por_split.items()}}
        if not simular:
            info[c.nome]["estatisticas"] = c.stats

    def sorteio(i, c, teto):
        esc = getattr(c, "escores", None) if par.get("ordenar_raras") else None
        return {s: sortear(candidatos[c.nome][s], P, teto,
                           np.random.default_rng([par["seed"], i, 3, j]), esc)
                for j, s in enumerate(SPLITS)}

    def selecionar(teto):
        """Cotas pelo teto máximo; depois cada cena usa o MENOR teto (passos de
        5%) que ainda entrega a sua cota -> repetição mínima em cada cena."""
        p = dict(par, max_sobreposicao=teto)
        aceitas = {c.nome: sorteio(i, c, teto) for i, c in enumerate(cenas)}
        escolha, avisos = balancear(cenas, aceitas, p)
        tetos = {}
        if par.get("ordenar_raras"):      # ver o comentário em PARAMETROS
            return aceitas, escolha, avisos, {c.nome: teto for c in cenas}
        for i, c in enumerate(cenas):
            cota = {s: len(escolha[c.nome][s]) for s in SPLITS}
            for t in [round(v, 2) for v in np.arange(0.05, teto, 0.05)] + [teto]:
                a = sorteio(i, c, t) if t != teto else aceitas[c.nome]
                if all(len(a[s]) >= cota[s] for s in SPLITS):
                    escolha[c.nome] = {s: sorted(a[s][:cota[s]]) for s in SPLITS}
                    tetos[c.nome] = t
                    break
        return aceitas, escolha, avisos, tetos

    if simular:
        print("\nquanto cabe (patches de %d px, total pedido: %s):"
              % (P, par["total"] or "o máximo"))
        for teto in args.simular or [par["max_sobreposicao"]]:
            aceitas, escolha, avisos, tetos = selecionar(teto)
            print("teto máximo de sobreposição %.0f%%:" % (100 * teto))
            for c in cenas:
                sel = [p for s in SPLITS for p in escolha[c.nome][s]]
                r = repeticao(sel, P, par["resolucao"], par["stride"]) if sel else {}
                print("   %-26s cabem %4d | no dataset %4d %s | teto usado %.0f%% | pixel aparece "
                      "em média %.1fx (máx %d)"
                      % (c.nome, sum(len(v) for v in aceitas[c.nome].values()), len(sel),
                         {s: len(escolha[c.nome][s]) for s in SPLITS}, 100 * tetos[c.nome],
                         r.get("vezes_por_pixel_media", 0), r.get("vezes_por_pixel_max", 0)))
            for a in avisos:
                print("   AVISO", a)
        shutil.rmtree(tmp)
        return

    aceitas, escolha, avisos, tetos = selecionar(par["max_sobreposicao"])
    for a in avisos:
        print("AVISO", a)
    for c in cenas:
        info[c.nome]["cabem_por_split"] = {s: len(aceitas[c.nome][s]) for s in SPLITS}
        info[c.nome]["teto_sobreposicao_usado"] = tetos[c.nome]
        info[c.nome]["repeticao"] = repeticao([p for s in SPLITS for p in escolha[c.nome][s]],
                                              P, par["resolucao"], par["stride"])
        print("[%s] %s" % (c.nome, info[c.nome]["repeticao"]))

    linhas, amostras = [], []
    classes = {s: np.zeros(256, np.int64) for s in SPLITS}
    for i, c in enumerate(cenas):
        n = sum(len(v) for v in escolha[c.nome].values())
        print("[%s] gravando %d patches" % (c.nome, n), flush=True)
        for s in SPLITS:
            for y, x in escolha[c.nome][s]:
                nome = "%s_%05d_%05d" % (c.nome, y, x)
                gt = (c.gt[0] + x * c.gt[1], c.gt[1], 0.0, c.gt[3] + y * c.gt[5], 0.0, c.gt[5])
                img = c.bandas(x, y, P, P, par)
                m = LUT[c.masc.ReadAsArray(x, y, P, P)]
                # pixel sem PAN válida nunca é rótulo (com min_valido = 1 isto é no-op,
                # a janela já é 100% válida; com min_valido < 1 tira o preto saturado)
                m[~c.valido(x, y, P, P)] = 0
                gravar(os.path.join(saida, "images", nome + ".tif"), img, gt, c.wkt,
                       gdal.GDT_Float32, descricoes=BANDAS)
                gravar(os.path.join(saida, "masks", nome + ".tif"), m, gt, c.wkt,
                       gdal.GDT_Byte, nodata=0)
                classes[s] += np.bincount(m.ravel(), minlength=256)
                fr = fracoes_classes(m)
                linhas.append([nome, c.nome, s, x, y] + [round(fr[k], 4) for k in
                                                         ["rotulado"] + config.CLASS_CURTOS + ["outros"]])
        info[c.nome]["selecionadas_por_split"] = {s: len(escolha[c.nome][s]) for s in SPLITS}
        rng = np.random.default_rng([par["seed"], i, 4])
        todos = [r[0] for r in linhas if r[1] == c.nome]
        for nome in rng.choice(todos, min(3, len(todos)), replace=False):
            amostras.append((c.nome, os.path.join(saida, "images", nome + ".tif"),
                             os.path.join(saida, "masks", nome + ".tif")))

    with open(os.path.join(saida, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "cena", "split", "x", "y", "frac_rotulado"]
                   + ["frac_" + k for k in config.CLASS_CURTOS] + ["frac_outros"])
        w.writerows(linhas)
    for s in SPLITS:
        with open(os.path.join(saida, "splits", s + ".txt"), "w") as f:
            f.write("".join(r[0] + "\n" for r in linhas if r[2] == s))

    dist = {}
    for s in SPLITS:
        tot = max(1, classes[s][1:].sum())
        dist[s] = {str(k): round(float(classes[s][k]) / tot, 4) for k in np.nonzero(classes[s][1:])[0] + 1}
    receita = {
        "script": rel(__file__),
        "gerado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "bandas": BANDAS,
        "mascara": "códigos MapBiomas (39/41/62 -> 19), 0 = sem rótulo",
        "parametros": par,
        "cenas": specs,
        "resultado": {"por_cena": info, "avisos": avisos,
                      "total_por_split": {s: sum(1 for r in linhas if r[2] == s) for s in SPLITS},
                      "classes_por_split": dist},
    }
    with open(os.path.join(saida, "receita.json"), "w", encoding="utf-8") as f:
        json.dump(receita, f, ensure_ascii=False, indent=1)

    visualizar(saida, cenas, escolha, amostras, par)
    for c in cenas:
        c.pan_ds = c.masc_ds = c.pan = c.masc = c.alpha = None
    shutil.rmtree(tmp)
    print("OK: %d patches em %s (%.0fs) | por split: %s"
          % (len(linhas), saida, time.time() - t0, receita["resultado"]["total_por_split"]))
    print("classes por split (fração dos pixels rotulados):")
    for s in SPLITS:
        print("   %-9s %s" % (s, dist[s]))


if __name__ == "__main__":
    main()
