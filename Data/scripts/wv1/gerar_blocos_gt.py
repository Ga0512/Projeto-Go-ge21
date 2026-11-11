#!/usr/bin/env python3
"""Pares imagem+máscara (PAN + mapa de classes) na grade NATIVA da PAN, um par
por bloco rotulado do GT.

O GT da Jataí WorldView são 10 blocos de 5x5 km rotulados de parede a parede
(campo `id` = bloco, `classe` = código do GT). Para cada bloco este script
grava, em `<saida>/b<id>/`:

    imagem.tif      recorte exato da PAN (sem reamostragem, cópia lossless)
    mascara_gt.tif  classes rasterizadas na MESMA grade, códigos MapBiomas

A janela é o retângulo envolvente do bloco ajustado a limites inteiros de pixel,
então imagem e máscara têm mesmo CRS, mesmo affine e mesmas dimensões (regra do
projeto). O bloco é um quadrado em UTM e a PAN é geográfica, então sobra uma
margem fina fora do bloco: lá a máscara fica 0 (sem rótulo).

**Códigos**: o `classe` do GT NÃO é código MapBiomas — `classe=4` é "Área
consolidada" (= não vegetação, 15), enquanto 4 no MapBiomas é Savânica
(vegetação). `--mapear` faz a tradução; o default é

    1=3    Vegetação natural  -> 3   (o GT não separa densa de não densa)
    4=15   Área consolidada   -> 15  (não vegetação)
    10=33  Corpo d'água       -> 33
    11=0   Não identificado   -> 0   (sem rótulo)

ATENÇÃO: com `1=3`, toda a vegetação sai no código 3. Sob o CLASS_MAP de 4
classes isso cai em "vegetação densa" e o índice de "não densa" fica vazio.
Este GT só faz sentido com o agrupamento de vegetação única (3 classes).

Só grava bloco que tenha imagem na PAN (`--min-valido`); os blocos que caem num
vão do mosaico são listados e pulados.
"""

import argparse
import json
import os
import sys

import numpy as np
from osgeo import gdal, ogr, osr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

MAPEAMENTO_PADRAO = "1=3,4=15,10=33,11=0"
CO_IMG = ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=IF_SAFER"]


def ler_mapeamento(txt):
    m = {}
    for par in txt.split(","):
        par = par.strip()
        if not par:
            continue
        de, para = par.split("=")
        m[int(de)] = int(para)
    return m


def camada_na_grade(gt_path, wkt_destino, campo_id, campo_classe, mapa):
    """Lê o GT, reprojeta para o CRS da PAN e devolve uma camada em memória com
    o campo `codigo` já traduzido: (ds_mem, camada, envelopes_por_bloco)."""
    src = ogr.Open(gt_path)
    if src is None:
        raise SystemExit("ERRO: não abriu o GT %s" % gt_path)
    lay_src = src.GetLayer()
    srs_src = lay_src.GetSpatialRef()
    srs_dst = osr.SpatialReference(wkt=wkt_destino)
    for s in (srs_src, srs_dst):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = None if srs_src.IsSame(srs_dst) else osr.CoordinateTransformation(srs_src, srs_dst)

    mem = ogr.GetDriverByName("Memory").CreateDataSource("gt")
    lay = mem.CreateLayer("gt", srs_dst, ogr.wkbPolygon)
    lay.CreateField(ogr.FieldDefn("bloco", ogr.OFTInteger))
    lay.CreateField(ogr.FieldDefn("codigo", ogr.OFTInteger))
    envs, sem_mapa = {}, set()
    for f in lay_src:
        g = f.GetGeometryRef()
        if g is None:
            continue
        g = g.Clone()
        g.FlattenTo2D()
        if tr is not None:
            g.Transform(tr)
        c = f.GetField(campo_classe)
        if c not in mapa:
            sem_mapa.add(c)
            continue
        b = int(f.GetField(campo_id))
        nf = ogr.Feature(lay.GetLayerDefn())
        nf.SetGeometry(g)
        nf.SetField("bloco", b)
        nf.SetField("codigo", mapa[c])
        lay.CreateFeature(nf)
        e = g.GetEnvelope()
        envs[b] = e if b not in envs else (min(envs[b][0], e[0]), max(envs[b][1], e[1]),
                                           min(envs[b][2], e[2]), max(envs[b][3], e[3]))
    if sem_mapa:
        print("AVISO: classes do GT sem tradução, ignoradas: %s" % sorted(sem_mapa), flush=True)
    return mem, lay, envs


def janela(env, gt, W, H):
    """Retângulo envolvente -> janela em pixels inteiros, recortada na imagem."""
    x0, x1, y0, y1 = env
    px = [(x0 - gt[0]) / gt[1], (x1 - gt[0]) / gt[1]]
    py = [(y1 - gt[3]) / gt[5], (y0 - gt[3]) / gt[5]]
    a = max(0, int(np.floor(min(px))))
    c = min(W, int(np.ceil(max(px))))
    p = max(0, int(np.floor(min(py))))
    q = min(H, int(np.ceil(max(py))))
    return a, p, c - a, q - p


def fracao_valida(band, a, p, w, h, nd, amostra=600):
    arr = band.ReadAsArray(a, p, w, h, buf_xsize=min(w, amostra), buf_ysize=min(h, amostra))
    return float((arr != nd).mean())


def contar(msk_path, pan_path, win):
    """Contagem exata de códigos na máscara, só onde a PAN tem imagem."""
    dm = gdal.Open(msk_path)
    bm = dm.GetRasterBand(1)
    dp = gdal.Open(pan_path)
    bp = dp.GetRasterBand(1)
    nd = bp.GetNoDataValue()
    nd = 0 if nd is None else nd
    W, H = dm.RasterXSize, dm.RasterYSize
    a, p = win[0], win[1]
    cont = np.zeros(256, np.int64)
    validos = 0
    for y in range(0, H, 2048):
        h = min(2048, H - y)
        m = bm.ReadAsArray(0, y, W, h)
        v = bp.ReadAsArray(a, p + y, W, h) != nd
        validos += int(v.sum())
        cont += np.bincount(m[v].ravel(), minlength=256)
    return cont, validos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pan", default=caminhos.WV1_JATAI_PAN)
    ap.add_argument("--gt", default=caminhos.WV1_JATAI_GT)
    ap.add_argument("--saida", default=caminhos.WV1_JATAI_BLOCOS)
    ap.add_argument("--campo-id", default="id")
    ap.add_argument("--campo-classe", default="classe")
    ap.add_argument("--mapear", default=MAPEAMENTO_PADRAO,
                    help="tradução classe_do_GT=codigo_MapBiomas (default: %s)" % MAPEAMENTO_PADRAO)
    ap.add_argument("--min-valido", type=float, default=0.01,
                    help="fração mínima de pixel com imagem para gravar o bloco (default 0.01)")
    ap.add_argument("--mascara-cena", metavar="SAIDA.tif",
                    help="em vez dos pares por bloco, grava UMA máscara na grade inteira do "
                         "PAN (é o formato que o gerador de dataset consome como --cena). "
                         "Os blocos sem imagem não precisam ser excluídos: o PAN é nodata lá "
                         "e o gerador descarta as janelas pelo critério de PAN válida")
    ap.add_argument("--simular", action="store_true", help="só mostra o que faria")
    ap.add_argument("--sobrescrever", action="store_true")
    args = ap.parse_args()

    gdal.UseExceptions()
    gdal.SetCacheMax(256 * 1024 * 1024)
    mapa = ler_mapeamento(args.mapear)

    dp = gdal.Open(args.pan)
    bp = dp.GetRasterBand(1)
    gt, W, H = dp.GetGeoTransform(), dp.RasterXSize, dp.RasterYSize
    wkt = dp.GetProjection()
    nd = bp.GetNoDataValue()
    nd = 0 if nd is None else nd
    # area de 1 px em m2. Se o CRS e geografico, o grau de longitude encurta com cos(lat)
    # (sem isso a area sai ~5% alta nesta latitude); se e projetado, o gt ja esta em metros.
    _srs = osr.SpatialReference(wkt=wkt)
    _srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if _srs.IsGeographic():
        lat = gt[3] + gt[5] * H / 2.0
        grau_m2 = abs(gt[1] * 111320.0 * np.cos(np.radians(lat))) * abs(gt[5] * 110574.0)
    else:
        grau_m2 = abs(gt[1] * gt[5])
    print("PAN  %s" % args.pan)
    print("     %dx%d  pixel %.8g x %.8g  nodata=%g" % (W, H, gt[1], abs(gt[5]), nd))
    print("GT   %s" % args.gt)
    print("mapeamento %s\n" % " ".join("%d->%d" % (k, v) for k, v in sorted(mapa.items())))

    mem, lay, envs = camada_na_grade(args.gt, wkt, args.campo_id, args.campo_classe, mapa)
    print("%d blocos no GT\n" % len(envs))

    if args.mascara_cena:
        if os.path.exists(args.mascara_cena) and not args.sobrescrever:
            raise SystemExit("ERRO: %s já existe (use --sobrescrever)" % args.mascara_cena)
        if args.simular:
            print("gravaria %s (%dx%d Byte na grade do PAN)" % (args.mascara_cena, W, H))
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.mascara_cena)), exist_ok=True)
        dm = gdal.GetDriverByName("GTiff").Create(args.mascara_cena, W, H, 1, gdal.GDT_Byte,
                                                 options=CO_IMG + ["BIGTIFF=YES"])
        dm.SetGeoTransform(gt)
        dm.SetProjection(wkt)
        dm.GetRasterBand(1).SetNoDataValue(0)
        dm.GetRasterBand(1).Fill(0)
        gdal.RasterizeLayer(dm, [1], lay, options=["ATTRIBUTE=codigo"])
        dm.SetMetadata({"fonte_gt": os.path.basename(args.gt), "mapeamento": args.mapear,
                        "blocos": " ".join("b%d" % b for b in sorted(envs)),
                        "aviso": "codigos MapBiomas; vegetacao natural toda em 3 (o GT nao "
                                 "separa densa de nao densa) -> agrupamento de vegetacao unica"})
        dm.FlushCache()
        dm = None
        # conferência: quanto de rótulo caiu onde o PAN tem imagem
        dm = gdal.Open(args.mascara_cena); bm = dm.GetRasterBand(1)
        cont = np.zeros(256, np.int64)
        for y in range(0, H, 2048):
            h = min(2048, H - y)
            m = bm.ReadAsArray(0, y, W, h)
            v = bp.ReadAsArray(0, y, W, h) != nd
            cont += np.bincount(m[v].ravel(), minlength=256)
        rot = int(cont[1:].sum())
        print("%s  %dx%d" % (args.mascara_cena, W, H))
        print("   rotulado COM imagem: %.2f km2" % (rot * grau_m2 / 1e6))
        for c in np.nonzero(cont)[0]:
            if c:
                print("     codigo %-3d %6.2f%%  %8.2f km2"
                      % (c, 100.0 * cont[c] / rot, cont[c] * grau_m2 / 1e6))
        return

    reg = {"pan": os.path.relpath(args.pan, caminhos.RAIZ),
           "gt": os.path.relpath(args.gt, caminhos.RAIZ),
           "mapeamento": {str(k): v for k, v in mapa.items()},
           "blocos": {}}
    feitos = pulados = 0
    for b in sorted(envs):
        a, p, w, h = janela(envs[b], gt, W, H)
        if w < 2 or h < 2:
            print("b%-6d fora da grade da PAN — pulado" % b)
            pulados += 1
            continue
        frac = fracao_valida(bp, a, p, w, h, nd)
        if frac < args.min_valido:
            print("b%-6d %5dx%-5d  %5.1f%% com imagem — PULADO (vão do mosaico)"
                  % (b, w, h, 100 * frac))
            pulados += 1
            continue
        pasta = os.path.join(args.saida, "b%d" % b)
        img = os.path.join(pasta, "imagem.tif")
        msk = os.path.join(pasta, "mascara_gt.tif")
        if args.simular:
            print("b%-6d %5dx%-5d  %5.1f%% com imagem  -> %s"
                  % (b, w, h, 100 * frac, os.path.relpath(pasta, caminhos.RAIZ)))
            feitos += 1
            continue
        if os.path.exists(img) and not args.sobrescrever:
            raise SystemExit("ERRO: %s já existe (use --sobrescrever)" % img)
        os.makedirs(pasta, exist_ok=True)

        # 1) recorte exato da PAN: sem reamostragem, re-embalagem lossless
        gdal.Translate(img, dp, srcWin=[a, p, w, h], format="GTiff",
                       noData=nd, creationOptions=CO_IMG)
        # 2) máscara na MESMA grade, só os polígonos deste bloco
        gsub = (gt[0] + a * gt[1], gt[1], 0.0, gt[3] + p * gt[5], 0.0, gt[5])
        dm = gdal.GetDriverByName("GTiff").Create(msk, w, h, 1, gdal.GDT_Byte, options=CO_IMG)
        dm.SetGeoTransform(gsub)
        dm.SetProjection(wkt)
        dm.GetRasterBand(1).SetNoDataValue(0)
        dm.GetRasterBand(1).Fill(0)
        lay.SetAttributeFilter("bloco = %d" % b)
        gdal.RasterizeLayer(dm, [1], lay, options=["ATTRIBUTE=codigo"])
        lay.SetAttributeFilter(None)
        dm.SetMetadata({
            "fonte_gt": os.path.basename(args.gt),
            "bloco": str(b),
            "mapeamento": args.mapear,
            "aviso": "codigos MapBiomas; vegetacao natural toda em 3 (o GT nao separa "
                     "densa de nao densa) -> usar agrupamento de vegetacao unica",
        })
        dm.FlushCache()
        dm = None

        cont, validos = contar(msk, args.pan, (a, p, w, h))
        pix = {str(i): int(cont[i]) for i in range(256) if cont[i]}
        reg["blocos"]["b%d" % b] = {
            "janela": [a, p, w, h], "geotransform": list(gsub),
            "largura": w, "altura": h,
            "px_com_imagem": validos, "frac_com_imagem": round(validos / float(w * h), 5),
            "px_por_codigo": pix,
        }
        rot = int(validos - cont[0])
        print("b%-6d %5dx%-5d  %5.1f%% com imagem  %6.2f km2 rotulados  %s"
              % (b, w, h, 100.0 * validos / (w * h), rot * grau_m2 / 1e6,
                 "  ".join("%s:%4.1f%%" % (c, 100.0 * n / max(validos, 1))
                           for c, n in sorted(pix.items(), key=lambda t: -t[1]))))
        feitos += 1

    if not args.simular and feitos:
        os.makedirs(args.saida, exist_ok=True)
        with open(os.path.join(args.saida, "blocos.json"), "w") as f:
            json.dump(reg, f, indent=1)
        print("\nblocos.json em %s" % os.path.relpath(args.saida, caminhos.RAIZ))
    print("\n%d bloco(s) gravado(s), %d pulado(s)" % (feitos, pulados))


if __name__ == "__main__":
    main()
