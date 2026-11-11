#!/usr/bin/env python3
"""Máscara base para edição: MapBiomas + água detectada na PAN.

O MapBiomas erra a água (borda de 30 m, corpos pequenos que ele não vê). Este
script gera, ao lado da máscara MapBiomas de cada tile, uma
`mascara_base.tif` = MapBiomas com a água AMPLIADA pelo que a PAN mostra.
Serve como ponto de partida da edição manual no editor, no lugar do MapBiomas
puro. **Só acrescenta água; nunca tira** — o que o MapBiomas marcou como água
continua água, e o resto continua com a classe que tinha.

Regra (dois critérios, calibrados na água que JÁ existe na cena):
  1. escuro:  PAN <= quantil --quantil-brilho da PAN na água interior;
  2. liso:    desvio-padrão local (janela --janela) <= quantil
              --quantil-suavidade do mesmo desvio na água interior;
  3. mancha:  componente conexo >= --min-px px (tira ruído pontilhado);
  4. opcional --vizinhanca N: só a N px da água já existente (0 = a cena toda,
     que é o que permite achar corpo d'água novo).
"Água interior" = água da máscara erodida --borda px, para o limiar não ser
contaminado pela margem. O critério de suavidade é o que separa água de
sombra de encosta: as duas são escuras, só a água é lisa.

A calibração usa a máscara mais confiável de cada área: `mascara_refinada.tif`
se já existir, senão a do MapBiomas. A saída é gerada SEMPRE a partir do
MapBiomas (a refinada é trabalho manual e não é tocada).

Uso:
  .venv/bin/python scripts/spot5/agua_pan.py --tiles dados/spot5/colinas_do_sul/tiles
  .venv/bin/python scripts/spot5/agua_pan.py --tiles ... --simular        # só os números
  .venv/bin/python scripts/spot5/agua_pan.py --imagem <PAN.tif> --mascara <mask.tif> --saida <out.tif>
Depois é só abrir a cena no editor: onde houver `mascara_base.tif`, ela é o
ponto de partida (a `mascara_refinada.tif` continua tendo prioridade).
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.ndimage as ndi
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

gdal.UseExceptions()

PADRAO = {
    "quantil_brilho": 0.95,    # quantil da PAN na água interior -> limiar de "escuro"
    "quantil_suavidade": 0.95,  # idem para o desvio-padrão local -> limiar de "liso"
    "min_px": 200,             # componente conexo mínimo
    "vizinhanca": 0,           # px de vizinhança da água existente (0 = cena toda)
    "janela": 7,               # janela do desvio-padrão local
    "borda": 8,                # px erodidos da água para pegar só o interior
    "max_novo": 0.25,          # trava: água nova acima disso (fração do rotulado) = aborta
}
FAIXA = 2048                   # linhas por bloco de leitura


def local_std(pan, janela):
    """Desvio-padrão local (mesma definição do gerar_features_pan.py)."""
    m = ndi.uniform_filter(pan, size=janela, mode="reflect")
    m2 = ndi.uniform_filter(pan * pan, size=janela, mode="reflect")
    return np.sqrt(np.clip(m2 - m * m, 0, None))


def _blocos(H, altura, margem):
    """(y0, h, topo) por bloco, com margem de leitura para filtros de janela."""
    for y0 in range(0, H, altura):
        h = min(altura, H - y0)
        ya, yb = max(0, y0 - margem), min(H, y0 + h + margem)
        yield y0, h, ya, yb


def calibrar(areas, par):
    """Limiares de brilho e de suavidade a partir da água interior da cena."""
    pans, stds = [], []
    meia = par["janela"] // 2
    for a in areas:
        W, H = a["w"], a["h"]
        mb = gdal.Open(a["calib"]); bm = mb.GetRasterBand(1)
        pb = gdal.Open(a["imagem"]); bp = pb.GetRasterBand(1)
        for y0, h, ya, yb in _blocos(H, FAIXA, max(par["borda"], meia)):
            m = bm.ReadAsArray(0, ya, W, yb - ya).astype(np.uint8)
            agua = m == 33
            if not agua.any():
                continue
            p = bp.ReadAsArray(0, ya, W, yb - ya).astype(np.float32)
            interior = agua & ~ndi.binary_dilation(~agua, iterations=par["borda"])
            c0 = y0 - ya
            sel = interior[c0:c0 + h]
            if not sel.any():
                continue
            pans.append(p[c0:c0 + h][sel])
            stds.append(local_std(p, par["janela"])[c0:c0 + h][sel])
        mb = pb = None
    if not pans:
        return None, None, 0
    pans, stds = np.concatenate(pans), np.concatenate(stds)
    return (float(np.quantile(pans, par["quantil_brilho"])),
            float(np.quantile(stds, par["quantil_suavidade"])), int(pans.size))


def aplicar(a, par, lim_brilho, lim_std, saida, lut=None):
    """Grava <saida> = máscara MapBiomas com os códigos remapeados (lut) e a água
    ampliada (se houver limiar). Devolve stats."""
    W, H = a["w"], a["h"]
    meia = par["janela"] // 2
    base_ds = gdal.Open(a["mapbiomas"])
    base = base_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    if lut is not None:
        base = lut[base]
    pb = gdal.Open(a["imagem"]); bp = pb.GetRasterBand(1)
    cand = np.zeros((H, W), bool)
    if lim_brilho is not None:      # lim_brilho None = só o remapeamento, sem água
        for y0, h, ya, yb in _blocos(H, FAIXA, meia):
            p = bp.ReadAsArray(0, ya, W, yb - ya).astype(np.float32)
            s = local_std(p, par["janela"])
            c0 = y0 - ya
            cand[y0:y0 + h] = ((p[c0:c0 + h] <= lim_brilho) & (s[c0:c0 + h] <= lim_std)
                               & (base[y0:y0 + h] != 0))
        if par["vizinhanca"]:
            cand &= ndi.binary_dilation(base == 33, iterations=par["vizinhanca"])
        lbl, _ = ndi.label(cand)
        tam = np.bincount(lbl.ravel())
        tam[0] = 0
        cand = (tam >= par["min_px"])[lbl]
    novo = cand & (base != 33)
    out = base.copy()
    out[novo] = 33
    stats = {"agua_antes_px": int((base == 33).sum()), "agua_nova_px": int(novo.sum()),
             "rotulado_px": int((base != 0).sum())}
    if saida:
        d = gdal.GetDriverByName("GTiff").Create(saida, W, H, 1, gdal.GDT_Byte,
                                                 ["TILED=YES", "COMPRESS=DEFLATE"])
        d.SetGeoTransform(base_ds.GetGeoTransform())
        d.SetProjection(base_ds.GetProjection())
        b = d.GetRasterBand(1)
        b.SetNoDataValue(0)
        b.WriteArray(out)
        d = None
    base_ds = pb = None
    return stats, out, base


def png_comparacao(a, base, out, png, titulo, lado=640):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = base.shape
    muda = (out == 33) & (base != 33)
    if muda.any():   # centraliza onde mais mudou
        ys, xs = np.nonzero(muda)
        cy, cx = int(np.median(ys)), int(np.median(xs))
    else:
        cy, cx = H // 2, W // 2
    y0 = min(max(cy - lado // 2, 0), max(H - lado, 0))
    x0 = min(max(cx - lado // 2, 0), max(W - lado, 0))
    lado_y, lado_x = min(lado, H), min(lado, W)
    ds = gdal.Open(a["imagem"])
    p = ds.GetRasterBand(1).ReadAsArray(x0, y0, lado_x, lado_y).astype(np.float32)
    ds = None
    b = base[y0:y0 + lado_y, x0:x0 + lado_x]
    o = out[y0:y0 + lado_y, x0:x0 + lado_x]
    p2, p98 = np.percentile(p[p > 0], [2, 98]) if (p > 0).any() else (0, 1)
    cinza = np.clip((p - p2) / max(p98 - p2, 1), 0, 1)
    cor = np.zeros(b.shape + (4,))
    cor[b == 33] = [0.1, 0.4, 1.0, 0.55]
    cor[(o == 33) & (b != 33)] = [1.0, 0.2, 0.1, 0.75]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.6))
    ax[0].imshow(cinza, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("PAN", fontsize=10)
    ax[1].imshow(cinza, cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(cor)
    ax[1].set_title("azul = água do MapBiomas · vermelho = água nova (PAN)", fontsize=10)
    for e in ax:
        e.axis("off")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def areas_de_tiles(pasta, nome_saida):
    meta = json.load(open(os.path.join(pasta, "tiles.json"), encoding="utf-8"))
    areas = []
    for t in meta["tiles"]:
        d = os.path.join(pasta, t["id"])
        mb = os.path.join(d, "mascara_mapbiomas.tif")
        ref = os.path.join(d, "mascara_refinada.tif")
        ds = gdal.Open(mb)
        areas.append({"nome": t["id"], "imagem": os.path.join(d, "imagem.tif"),
                      "mapbiomas": mb, "calib": ref if os.path.exists(ref) else mb,
                      "refinada": os.path.exists(ref),
                      "saida": os.path.join(d, nome_saida),
                      "w": ds.RasterXSize, "h": ds.RasterYSize})
        ds = None
    return areas


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("Uso:"):])
    ap.add_argument("--tiles", help="pasta tiles/ de uma cena (gera em cada tile)")
    ap.add_argument("--imagem", help="PAN (uso avulso, sem tiles)")
    ap.add_argument("--mascara", help="máscara MapBiomas na grade da PAN")
    ap.add_argument("--saida", help="GeoTIFF de saída (uso avulso)")
    ap.add_argument("--nome-saida", default="mascara_base.tif",
                    help="nome do arquivo por tile (default mascara_base.tif)")
    ap.add_argument("--mapear", default="",
                    help="remapeia códigos na base, ex.: '39=15,41=15' (soja e outras "
                         "lavouras viram pastagem). Não muda a classe do modelo — serve "
                         "para a edição manual não ter dezenas de retalhos minúsculos")
    ap.add_argument("--png", action="store_true", help="grava PNG antes/depois por área")
    ap.add_argument("--simular", action="store_true", help="só os números, não grava")
    for k, v in PADRAO.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v,
                        help="default %s" % v)
    args = ap.parse_args()
    par = {k: getattr(args, k) for k in PADRAO}

    if args.tiles:
        areas = areas_de_tiles(args.tiles, args.nome_saida)
        cena = os.path.basename(os.path.dirname(os.path.abspath(args.tiles)))
    elif args.imagem and args.mascara:
        ds = gdal.Open(args.mascara)
        areas = [{"nome": os.path.basename(args.imagem), "imagem": args.imagem,
                  "mapbiomas": args.mascara, "calib": args.mascara, "refinada": False,
                  "saida": args.saida, "w": ds.RasterXSize, "h": ds.RasterYSize}]
        ds = None
        cena = os.path.basename(caminhos.pasta_cena(args.imagem))
        if not args.saida and not args.simular:
            ap.error("informe --saida (ou --simular)")
    else:
        ap.error("informe --tiles ou (--imagem e --mascara)")

    lut = None
    if args.mapear:
        lut = np.arange(256, dtype=np.uint8)
        for par_ in args.mapear.split(","):
            de, para = par_.split("=")
            lut[int(de)] = int(para)
        print("remapeando códigos: %s" % args.mapear)

    lim_b, lim_s, n = calibrar(areas, par)
    if lim_b is None:
        if lut is None:
            raise SystemExit("%s: não há água (33) na máscara para calibrar" % cena)
        print("   AVISO: sem água para calibrar — só o remapeamento será aplicado")
        lim_s = None
    # TRAVA 1 (prévia barata): roda a regra em algumas áreas sorteadas e vê quanto
    # ela pintaria. Onde o MapBiomas marca rio mais estreito que o pixel de 30 m
    # (ex.: Jataí), a "água" dele é terra na PAN, o limiar sai claro e a regra
    # pintaria a cena toda — melhor abortar antes de processar tudo.
    if lim_b is not None:
        rng = np.random.default_rng(0)
        amostra = [areas[i] for i in rng.permutation(len(areas))[:min(8, len(areas))]]
        novo = rot = 0
        for a in amostra:
            st, _, _ = aplicar(a, par, lim_b, lim_s, None, lut)
            novo += st["agua_nova_px"]
            rot += st["rotulado_px"]
        prev = novo / max(rot, 1)
        print("   prévia em %d áreas: água nova = %.1f%% da área rotulada"
              % (len(amostra), 100 * prev))
        if prev > par["max_novo"]:
            msg = ("a prévia daria %.0f%% de água nova (teto --max-novo %.0f%%). A água do "
                   "MapBiomas nesta cena não é escura na PAN (limiar %.0f), então a "
                   "calibração não vale." % (100 * prev, 100 * par["max_novo"], lim_b))
            if lut is None:
                raise SystemExit("ABORTADO: %s Nada foi gerado; a base continua o MapBiomas." % msg)
            print("   AVISO: %s Sigo só com o remapeamento, sem mexer na água." % msg)
            lim_b = lim_s = None
    if lim_b is None:
        print("%s: regra de água DESLIGADA (só remapeamento)" % cena, flush=True)
    else:
        print("%s: %d px de água interior | limiar brilho <= %.0f | limiar desvio local <= %.2f"
              % (cena, n, lim_b, lim_s), flush=True)
    print("   (calibrado com %d refinada(s) e %d MapBiomas)"
          % (sum(a["refinada"] for a in areas), sum(not a["refinada"] for a in areas)))

    tot_novo = tot_antes = tot_rotulado = 0
    for a in areas:
        st, out, base = aplicar(a, par, lim_b, lim_s, None if args.simular else a["saida"], lut)
        tot_novo += st["agua_nova_px"]
        tot_antes += st["agua_antes_px"]
        tot_rotulado += st["rotulado_px"]
        if st["agua_nova_px"] or a["refinada"]:
            print("   %-10s água MapBiomas %6.2f%% -> +%.2f%% nova%s"
                  % (a["nome"], 100 * st["agua_antes_px"] / max(st["rotulado_px"], 1),
                     100 * st["agua_nova_px"] / max(st["rotulado_px"], 1),
                     "  [tem refinada]" if a["refinada"] else ""), flush=True)
        if args.png and st["agua_nova_px"]:
            vis = os.path.join(caminhos.EXPERIMENTOS, "agua_pan", cena)
            os.makedirs(vis, exist_ok=True)
            png_comparacao(a, base, out, os.path.join(vis, "base_%s.png" % a["nome"]),
                           "%s %s | brilho<=%.0f desvio<=%.2f mancha>=%d viz=%d"
                           % (cena, a["nome"], lim_b, lim_s, par["min_px"], par["vizinhanca"]))
    print("TOTAL: água do MapBiomas %d px, água nova %d px (+%.0f%%)%s"
          % (tot_antes, tot_novo, 100 * tot_novo / max(tot_antes, 1),
             "  [simulação: nada gravado]" if args.simular else ""))
    # TRAVA 2: acréscimo absurdo = regra errada para esta cena
    frac = tot_novo / max(tot_rotulado, 1)
    if frac > par["max_novo"]:
        raise SystemExit("ABORTADO: água nova = %.0f%% da área rotulada (teto --max-novo %.0f%%). "
                         "Revise os quantis ou não use a regra nesta cena."
                         % (100 * frac, 100 * par["max_novo"]))


if __name__ == "__main__":
    main()
