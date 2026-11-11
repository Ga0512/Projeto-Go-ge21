#!/usr/bin/env python3
"""Alcance da autocorrelação espacial das 3 bandas (blockCV, Valavi et al. 2019).

O tamanho do bloco do split e a zona morta entre treino/validação/teste devem
ser MAIORES que a distância em que as bandas deixam de ser correlacionadas;
senão treino e teste são vizinhos parecidos e a métrica sai otimista.

Mede, em janelas das áreas rotuladas: autocorrelação 2D por FFT (Wiener–
Khinchin) de cada banda, média radial -> rho(d). Reporta a distância em que
rho cai abaixo de --limite (default 0.1) e a que corresponde a 95% do patamar
do variograma (semivariância = 0.95 * sill, equivalente ao "range").

Uso:
  .venv/bin/python scripts/testes/alcance_autocorrelacao.py
  .venv/bin/python scripts/testes/alcance_autocorrelacao.py --lado 2048 --limite 0.1
"""

import argparse
import json
import os
import sys

import numpy as np
from osgeo import gdal

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
sys.path.insert(0, os.path.join(AQUI, "..", "datasets"))
import caminhos  # noqa: E402
import gerar_dataset_refinado as g  # noqa: E402

gdal.UseExceptions()


def rho_radial(a):
    """Autocorrelação média por distância (px), via FFT."""
    a = a - a.mean()
    n = a.shape[0]
    f = np.fft.rfft2(a, s=(2 * n, 2 * n))
    ac = np.fft.irfft2(f * np.conj(f), s=(2 * n, 2 * n))[:n, :n]
    ac /= ac[0, 0]
    yy, xx = np.mgrid[:n, :n]
    d = np.sqrt(yy * yy + xx * xx).astype(int)
    soma = np.bincount(d.ravel(), ac.ravel())
    cont = np.bincount(d.ravel())
    return soma[:n] / np.maximum(cont[:n], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lado", type=int, default=2048, help="lado da janela analisada (px)")
    ap.add_argument("--limite", type=float, default=0.1, help="rho considerado 'sem correlação'")
    args = ap.parse_args()

    par = dict(g.PARAMETROS)
    tmp = os.path.join(caminhos.EXPERIMENTOS, "_tmp_autocorr")
    os.makedirs(tmp, exist_ok=True)
    res = par["resolucao"]
    alvos = [
        ("cena_013-014_2007-06-14", caminhos.SPOT5_CENA_013_IMAGEM, caminhos.SPOT5_CENA_013_TILES),
        ("colinas_do_sul", caminhos.SPOT5_COLINAS_IMAGEM, caminhos.SPOT5_COLINAS_TILES),
    ]
    print("janela %d px (%.1f km) | resolução %.1f m\n" % (args.lado, args.lado * res / 1000, res))
    for nome, pan, tiles in alvos:
        meta = json.load(open(os.path.join(tiles, "tiles.json"), encoding="utf-8"))
        refin = [t for t in meta["tiles"]
                 if os.path.exists(os.path.join(tiles, t["id"], "mascara_refinada.tif"))]
        c = g.Cena({"nome": nome, "pan": g.rel(pan), "mascara": g.rel(tiles), "peso": 1}, par, tmp)
        g.estatisticas(c, par, np.random.default_rng([par["seed"], 0, 1]))
        for t in refin:
            L = min(args.lado, t["w"], t["h"])
            b = c.bandas(t["x0"], t["y0"], L, L, par)
            saida = []
            for i, banda in enumerate(g.BANDAS):
                r = rho_radial(b[i].astype(np.float64))
                fora = np.nonzero(r < args.limite)[0]
                d_rho = int(fora[0]) if len(fora) else L
                # variograma: gamma(d) = sill * (1 - rho(d)); range = 95% do sill
                g95 = np.nonzero((1 - r) >= 0.95 * (1 - r[:L // 2].min()))[0]
                d_var = int(g95[0]) if len(g95) else L
                saida.append("%s rho<%.1f em %d px (%.0f m) | range 95%% em %d px (%.0f m)"
                             % (banda.split("_")[0], args.limite, d_rho, d_rho * res,
                                d_var, d_var * res))
            print("%s %s:" % (nome, t["id"]))
            for s in saida:
                print("   " + s)
        c.pan_ds = c.masc_ds = c.pan = c.masc = c.alpha = None
    print("\nRegra do blockCV: bloco do split e zona morta >= o maior alcance acima.")


if __name__ == "__main__":
    main()
