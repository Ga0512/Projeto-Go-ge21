#!/usr/bin/env python3
"""Patches SEM rótulo (só imagem, 3 bandas) de uma cena inteira.

Mesmas 3 bandas e mesma grade do gerar_dataset_refinado.py (PAN, GLCM
entropia, rugosidade; Float32, z-score por cena, UTM a 2,5 m), para servir de
imagem para inferência, pré-treino ou pseudo-rótulo. Não grava máscaras, e a
saída deve ficar numa pasta própria, separada dos datasets rotulados.

  --excluir     máscara (GeoTIFF ou pasta tiles/) de área já rotulada: janela
                que toca qualquer pixel rotulado fica de fora (não repete os
                patches rotulados, nem vaza validação/teste deles). Pode
                repetir; máscara de OUTRA cena/grade (ex.: tiles da Colinas do
                Sul sobre a 013-014) é levada para a grade da PAN com nearest,
                o que exclui o mesmo terreno visto por outra imagem.
  --estatisticas receita.json de um dataset rotulado: copia dela as
                estatísticas desta cena (p2/p98 e média/desvio das bandas),
                para as bandas ficarem na mesma escala dos patches rotulados.
                Sem isso, calcula como o gerar_dataset_refinado.py.

Janelas de --patch px a cada --stride px (default = patch: lado a lado, sem
sobreposição), só as com >= --min-valido de imagem válida. Tudo vai para
<saida>/receita.json; --receita refaz idêntico.

Uso:
  .venv/bin/python scripts/datasets/gerar_patches_sem_rotulo.py \\
      --saida dados/datasets/spot5_013-014_sem_rotulo \\
      --pan dados/spot5/cena_013-014_2007-06-14/original/013-014_S5_708-378-0_2007-06-14_modificado.tif \\
      --excluir dados/spot5/cena_013-014_2007-06-14/tiles \\
      --excluir dados/spot5/colinas_do_sul/tiles \\
      --estatisticas dados/datasets/spot5_refinado_v3/receita.json
  # refazer idêntico:
  .venv/bin/python scripts/datasets/gerar_patches_sem_rotulo.py \\
      --receita dados/datasets/spot5_013-014_sem_rotulo/receita.json --sobrescrever
"""

import argparse
import csv
import datetime
import json
import math
import os
import shutil
import sys
import time

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gerar_dataset_refinado as g  # noqa: E402
from gerar_dataset_refinado import caminhos  # noqa: E402

PARAMETROS = {
    "resolucao": g.PARAMETROS["resolucao"],
    "patch": 256,
    "stride": 0,              # 0 = igual ao patch (lado a lado)
    "min_valido": 1.0,        # fração mínima de imagem válida na janela
    "janela": g.PARAMETROS["janela"],
    "niveis": g.PARAMETROS["niveis"],
    "blocos_estatistica": g.PARAMETROS["blocos_estatistica"],
    "lado_estatistica": g.PARAMETROS["lado_estatistica"],
    "seed": g.PARAMETROS["seed"],
}


def visualizar(saida, c, sel, P, cel, excl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis = os.path.join(saida, "visualizacao")
    os.makedirs(vis, exist_ok=True)
    f = max(1, max(c.W, c.H) // 1400)
    a = c.pan.ReadAsArray(0, 0, c.W, c.H, buf_xsize=c.W // f, buf_ysize=c.H // f).astype(np.float32)
    cob = np.zeros(a.shape, bool)
    for y, x in sel:
        cob[y // f:(y + P) // f, x // f:(x + P) // f] = True
    fig, ax = plt.subplots(figsize=(12, 12 * a.shape[0] / a.shape[1] + 0.6))
    ax.imshow(a, cmap="gray", vmin=c.stats["p2"], vmax=c.stats["p98"])
    ax.imshow(np.ma.masked_where(~cob, cob), cmap="autumn", alpha=0.25)
    if excl is not None:
        linhas = np.minimum(np.arange(a.shape[0]) * f // cel, excl.shape[0] - 1)
        colunas = np.minimum(np.arange(a.shape[1]) * f // cel, excl.shape[1] - 1)
        e = (excl > 0)[linhas][:, colunas]
        ax.contour(e, levels=[0.5], colors="cyan", linewidths=1.2)
    ax.set_title("%s — %d patches sem rótulo (laranja)%s"
                 % (c.nome, len(sel), "; ciano = área rotulada excluída" if excl is not None else ""))
    ax.axis("off")
    fig.savefig(os.path.join(vis, "cobertura.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    if not sel:
        return
    rng = np.random.default_rng(0)
    amostra =[sel[i] for i in sorted(rng.choice(len(sel), min(4, len(sel)), replace=False))]
    fig, axs = plt.subplots(len(amostra), 3, figsize=(9, 3 * len(amostra)))
    axs = np.atleast_2d(axs)
    for lin, (y, x) in enumerate(amostra):
        nome = "%s_%05d_%05d" % (c.nome, y, x)
        img = gdal.Open(os.path.join(saida, "images", nome + ".tif")).ReadAsArray()
        for col in range(3):
            axs[lin, col].imshow(img[col], cmap="gray", vmin=-2.5, vmax=2.5)
            axs[lin, col].set_title(("%s\n" % nome if col == 0 else "") + g.BANDAS[col], fontsize=8)
            axs[lin, col].axis("off")
    fig.savefig(os.path.join(vis, "amostras.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("Uso:"):])
    ap.add_argument("--pan", help="PAN da cena")
    ap.add_argument("--excluir", action="append", default=[],
                    help="máscara/pasta tiles de área rotulada a deixar de fora (pode repetir)")
    ap.add_argument("--estatisticas", help="receita.json de onde copiar as estatísticas da cena")
    ap.add_argument("--saida")
    ap.add_argument("--receita", help="receita.json de uma rodada anterior (refaz idêntico)")
    ap.add_argument("--sobrescrever", action="store_true")
    for k, v in PARAMETROS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None, help="default %s" % v)
    args = ap.parse_args()

    par = dict(PARAMETROS)
    if args.receita:
        rec = json.load(open(args.receita, encoding="utf-8"))
        par.update(rec["parametros"])
        spec, stats = rec["cena"], rec["estatisticas"]
        if "excluir" not in spec:      # receita antiga: uma máscara só, em "mascara"
            ex = {"mascara": spec.pop("mascara", None), "md5": spec.pop("md5_excluir", {})}
            if "tiles_refinados" in spec:
                ex["tiles_refinados"] = spec.pop("tiles_refinados")
            spec["excluir"] = [ex] if ex["mascara"] else []
        saida = args.saida or os.path.dirname(os.path.abspath(args.receita))
    else:
        if not args.pan or not args.saida:
            ap.error("informe --pan e --saida (ou --receita)")
        spec = {"nome": os.path.basename(caminhos.pasta_cena(args.pan)), "pan": g.rel(args.pan),
                "excluir": [{"mascara": g.rel(e)} for e in args.excluir]}
        stats = None
        if args.estatisticas:
            rec = json.load(open(args.estatisticas, encoding="utf-8"))
            for cs in rec["cenas"]:
                if os.path.abspath(g.absoluto(cs["pan"])) == os.path.abspath(args.pan):
                    stats = rec["resultado"]["por_cena"][cs["nome"]]["estatisticas"]
                    spec["estatisticas_de"] = {"receita": g.rel(args.estatisticas), "cena": cs["nome"]}
            if stats is None:
                raise SystemExit("%s não tem estatísticas desta PAN" % args.estatisticas)
        saida = args.saida
    for k in PARAMETROS:
        if getattr(args, k) is not None:
            par[k] = getattr(args, k)
    P = par["patch"]
    S = par["stride"] or P

    saida = os.path.abspath(saida)
    if os.path.exists(os.path.join(saida, "images")):
        if not args.sobrescrever:
            raise SystemExit("%s já tem patches; use --sobrescrever" % saida)
        for d in ("images", "visualizacao", "_tmp", "manifest.csv"):
            p = os.path.join(saida, d)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
    tmp = os.path.join(saida, "_tmp")
    os.makedirs(os.path.join(saida, "images"))
    os.makedirs(tmp)

    t0 = time.time()
    print("[%s] preparando" % spec["nome"], flush=True)
    par_cena = dict(g.PARAMETROS, **par)
    c = g.Cena({"nome": spec["nome"], "pan": spec["pan"], "mascara": None, "peso": 1}, par_cena, tmp)
    if stats is None:
        g.estatisticas(c, par, np.random.default_rng([par["seed"], 0, 1]))
    else:
        c.stats = stats

    cel = math.gcd(P, S)
    rot, val, _ = g.fracoes_celulas(c, cel, validade_total=True)   # rot = 0: cena sem máscara
    for j, ex in enumerate(spec["excluir"]):
        print("   área a excluir: %s" % ex["mascara"], flush=True)
        ce = g.Cena(dict({"nome": "%s_excluir%d" % (spec["nome"], j), "pan": spec["pan"],
                          "mascara": ex["mascara"], "peso": 1},
                         **({"tiles_refinados": ex["tiles_refinados"]} if "tiles_refinados" in ex else {})),
                    par_cena, tmp)
        if args.receita:
            for arq, h in ex.get("md5", {}).items():
                if ce.entradas_mascara.get(arq) != h:
                    print("   AVISO: %s mudou desde a receita — o resultado não será idêntico" % arq)
        ex["md5"] = ce.entradas_mascara
        if ce.tiles is not None:
            ex["tiles_refinados"] = ce.tiles
        rot = rot + g.fracoes_celulas(ce, cel)[0]
        ce.pan_ds = ce.masc_ds = ce.pan = ce.masc = ce.alpha = None
    k = P // cel

    def soma_janela(a):
        ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    ny, nx = (c.H - P) // cel + 1, (c.W - P) // cel + 1
    fv = soma_janela(val)[:ny, :nx] / (k * k)
    fr = soma_janela(rot)[:ny, :nx]
    passo = S // cel
    ok = np.zeros_like(fv, bool)
    ok[::passo, ::passo] = True
    validas = ok & (fv >= par["min_valido"] - 1e-6)
    excluidas = int((validas & (fr > 0)).sum())
    ok = validas & (fr <= 0)
    cy, cx = np.nonzero(ok)
    sel = list(zip((cy * cel).tolist(), (cx * cel).tolist()))
    print("   grade EPSG:%s %dx%d (%s) | %d janelas com imagem válida, %d excluídas por tocar a "
          "área rotulada -> %d patches" % (c.epsg, c.W, c.H, "reamostrada" if c.reamostrada else "nativa",
                                           int(validas.sum()), excluidas, len(sel)), flush=True)

    linhas = []
    for n, (y, x) in enumerate(sel, 1):
        nome = "%s_%05d_%05d" % (c.nome, y, x)
        gt = (c.gt[0] + x * c.gt[1], c.gt[1], 0.0, c.gt[3] + y * c.gt[5], 0.0, c.gt[5])
        g.gravar(os.path.join(saida, "images", nome + ".tif"), c.bandas(x, y, P, P, par), gt, c.wkt,
                 gdal.GDT_Float32, descricoes=g.BANDAS)
        linhas.append([nome, x, y, round(float(fv[y // cel, x // cel]), 4)])
        if n % 500 == 0 or n == len(sel):
            print("   %d/%d (%.0fs)" % (n, len(sel), time.time() - t0), flush=True)

    with open(os.path.join(saida, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "x", "y", "frac_valido"])
        w.writerows(linhas)
    receita = {
        "script": g.rel(__file__),
        "gerado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "bandas": g.BANDAS,
        "rotulo": "nenhum (só imagem)",
        "parametros": par,
        "cena": spec,
        "estatisticas": c.stats,
        "resultado": {"grade": {"epsg": c.epsg, "largura": c.W, "altura": c.H,
                                "geotransform": list(c.gt), "reamostrada": c.reamostrada},
                      "janelas_validas": int(validas.sum()), "excluidas_area_rotulada": excluidas,
                      "patches": len(sel),
                      "area_km2": round(len(sel) * (P * par["resolucao"]) ** 2 / 1e6, 1)},
    }
    with open(os.path.join(saida, "receita.json"), "w", encoding="utf-8") as f:
        json.dump(receita, f, ensure_ascii=False, indent=1)
    visualizar(saida, c, sel, P, cel, rot if spec["excluir"] else None)
    c.pan_ds = c.masc_ds = c.pan = c.masc = c.alpha = None
    shutil.rmtree(tmp)
    print("OK: %d patches em %s (%.0fs)" % (len(sel), saida, time.time() - t0))


if __name__ == "__main__":
    main()
