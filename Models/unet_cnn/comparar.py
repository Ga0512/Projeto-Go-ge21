#!/usr/bin/env python3
"""Compara runs no MESMO conjunto de teste, com agregado ponderado pela cena.

POR QUE ISTO EXISTE
-------------------
O `mIoU` dá peso igual às três classes, mas o split de teste não tem a
composição da cena. Em Colinas o teste saiu com 15,22% de água e 4,51% de não
vegetação, enquanto a cena tem 5,54% e 12,47% (MapBiomas 2007). Resultado: a
água, que quase não existe no terreno, pesa um terço do agregado, e um modelo
bom em água e ruim em não vegetação parece melhor do que é no produto final.

Foi o que aconteceu com o `unetpp_v19b_reamostra`: mIoU de teste empatado com o
`unetpp_v20_zero` em Colinas (0,7039 x 0,7051) enquanto o mapa de cena inteira
dele era visivelmente pior -- com água classificada como não vegetação, efeito
da precisão de 0,308 nessa classe. Ponderando pela cena, o v20_zero ganha
(0,8579 x 0,8386), que é o que o olho via.

OS PESOS
--------
Vêm da MÁSCARA DE AMOSTRAGEM da cena (`mascara_amostra_v1.tif`), que é o ground
truth do projeto -- desenhada à mão, na resolução nativa de 2,5 m. Não se usa
MapBiomas aqui. A composição é contada sobre a máscara inteira, não sobre os
patches selecionados (a seleção é enviesada por construção) nem sobre o split
de teste (é justamente o que está torto) nem sobre a predição (seria circular).

A contagem é cacheada em `composicao_mascara.json` ao lado da máscara, porque
varrer 948 Mpx a cada comparação é desperdício. Apagar o arquivo força recontar.

Classe ausente na cena (água em Jataí: 0,14%) entra com peso ~0 em vez de puxar
o agregado para baixo como um zero -- mas o IoU por classe continua impresso,
porque 0,0000 ali é defeito real do modelo, não artefato.

Uso (no WSL, a partir de `Models/`):
  ../Data/.venv/bin/python unet_cnn/comparar.py \\
      --runs unetpp_v19b_reamostra unetpp_v20_zero unetpp_v21 unetpp_v22
"""
import argparse
import io
import json
import os

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

BASE = os.path.dirname(os.path.abspath(__file__))
DADOS = os.path.normpath(os.path.join(BASE, "..", "..", "Data", "dados", "spot5"))

# mesmo agrupamento do config.CLASS_MAP do projeto
VEG = {3, 4, 11, 12}
NVEG = {15, 21, 24, 25, 29, 39, 41, 62}
AGUA = {33}


def pesos_da_cena(cena, _ano=None):
    """-> ([p_veg, p_nveg, p_agua], fonte) pela máscara de amostragem da cena."""
    msk = os.path.join(DADOS, cena, "mascara_amostra_v1.tif")
    if not os.path.exists(msk):
        return None, None
    cache = os.path.join(DADOS, cena, "composicao_mascara.json")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(msk):
        c = json.load(io.open(cache, encoding="utf-8"))
        return c["pesos"], os.path.basename(msk) + " (cache)"
    d = gdal.Open(msk)
    W, H = d.RasterXSize, d.RasterYSize
    cont = {}
    for y0 in range(0, H, 2048):                 # em faixas: 948 Mpx não cabem
        a = d.GetRasterBand(1).ReadAsArray(0, y0, W, min(2048, H - y0))
        v, n = np.unique(a, return_counts=True)
        for k, q in zip(v.tolist(), n.tolist()):
            cont[k] = cont.get(k, 0) + q
    acc = [0.0, 0.0, 0.0]
    for c, q in cont.items():
        if c in VEG:
            acc[0] += q
        elif c in NVEG:
            acc[1] += q
        elif c in AGUA:
            acc[2] += q
    s_ = sum(acc)
    if s_ <= 0:
        return None, None
    pesos = [a / s_ for a in acc]
    json.dump({"pesos": pesos, "pixels": {str(k): v for k, v in cont.items()}},
              io.open(cache, "w", encoding="utf-8"), indent=1)
    return pesos, os.path.basename(msk)


def carregar(nome):
    """relatorio_completo.json (do avaliar.py) tem precedência sobre o do treino."""
    for arq in ("relatorio_completo.json", "relatorio.json"):
        p = os.path.join(BASE, "runs", nome, arq)
        if os.path.exists(p):
            r = json.load(io.open(p, encoding="utf-8"))
            return (r["teste"]["interior"] if "teste" in r else r["interior"]), arq
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    dados = {}
    for n in args.runs:
        t, arq = carregar(n)
        if t is None:
            print("!! sem relatório: %s" % n)
        else:
            dados[n] = (t, arq)
    if not dados:
        raise SystemExit("nada para comparar")

    qq = next(iter(dados.values()))[0]
    cenas = sorted(qq.get("por_cena", {}))

    print("=" * 94)
    print("GERAL (todas as cenas juntas)")
    print("%-28s %8s %8s %8s" % ("run", "OA", "mIoU", "origem"))
    for n, (t, arq) in dados.items():
        print("%-28s %8.4f %8.4f  %s" % (n, t["oa"], t["miou"], arq))

    for cena in cenas:
        pesos, fonte = pesos_da_cena(cena)
        print()
        print("=" * 94)
        if pesos:
            print("%s  —  pesos do ground truth (%s): Veg %.1f%% | NaoVeg %.1f%% | Agua %.1f%%"
                  % (cena.upper(), fonte, 100 * pesos[0], 100 * pesos[1],
                     100 * pesos[2]))
        else:
            print("%s  —  sem máscara de amostragem; só o mIoU simples"
                  % cena.upper())
        print("%-28s %-8s %-8s %-8s | %-9s %-9s"
              % ("run", "Veg", "NaoVeg", "Agua", "mIoU", "PONDERADO"))
        linhas = []
        for n, (t, _) in dados.items():
            c = t["por_cena"][cena]
            iou = [0.0 if v is None else float(v) for v in c["iou"]]
            m = sum(iou) / len(iou)
            w = sum(a * b for a, b in zip(iou, pesos)) if pesos else float("nan")
            linhas.append((n, iou, m, w))
        melhor_m = max(l[2] for l in linhas)
        melhor_w = max(l[3] for l in linhas) if pesos else None
        for n, iou, m, w in linhas:
            marca_m = " *" if abs(m - melhor_m) < 1e-9 else "  "
            marca_w = " *" if pesos and abs(w - melhor_w) < 1e-9 else "  "
            print("%-28s %-8.4f %-8.4f %-8.4f | %7.4f%s %7.4f%s"
                  % (n, iou[0], iou[1], iou[2], m, marca_m, w, marca_w))

    print()
    print("* = melhor da coluna. Diferença abaixo de 0,02 está dentro do ruído")
    print("  do conjunto de teste (798 patches) e não deve ser tratada como")
    print("  diferença. E nenhum modelo se escolhe só por esta tabela: conferir")
    print("  o mapa de cena inteira é parte da decisão.")


if __name__ == "__main__":
    main()
