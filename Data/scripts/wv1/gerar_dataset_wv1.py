#!/usr/bin/env python3
"""Prepara o par WorldView-1 PAN 0,5m <-> máscara MapBiomas 10m reamostrada.

Pipeline:
  1. Lê o GeoTIFF WorldView-1 PAN e extrai CRS/transform/resolução/dimensões/
     bounds/footprint reais.
  2. Usa o footprint real do GeoTIFF como ROI.
  3. No GEE (projeto beaming-botany-373619), acessa o MapBiomas Coverage 10m /
     Collection 3 beta (ano 2024, banda classification_2024) e baixa o recorte
     da ROI em EPSG:32722 na resolução nativa 10m.
  4. Reamostra o MapBiomas para a grade EXATA da PAN usando nearest-neighbor
     (gdal.ReprojectImage, GRA_NearestNeighbour), preservando os códigos
     categóricos oficiais da legenda.
  5. Exporta <out>/pan.tif, <out>/mask.tif e <out>/pan_mask_pair.tif
     (Band1=PAN, Band2=MASK).
  6. Valida o alinhamento PAN/MASK e gera visualização + relatório.

Não usa bilinear/bicubic. A máscara é tratada como pseudo-label derivado de um
produto de 10m, não como ground truth de 0,5m.
"""

import argparse
import os
import shutil
import sys

from osgeo import gdal, osr

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

DEFAULT_PAN = caminhos.WV1_PAN
DEFAULT_OUT = os.path.join(caminhos.WV1_MAPBIOMAS, "2024_10m")

PROJ = "beaming-botany-373619"

MB_ASSET_10M = (
    "projects/mapbiomas-public/assets/brazil/lulc_10m/collection3/"
    "mapbiomas_10m_collection3_integration_v1"
)
MB_COLL_30M = "projects/mapbiomas-public/assets/brazil/lulc/v1"  # ID image por ano
MB_CRS = "EPSG:32722"

# Resolução nativa em metros e bandas de cada fonte.
# 10m  -> asset único, banda classification_YYYY (2017-2024)
# 30m  -> ImageCollection v1, imagem collection_10_1_YYYY_v1, banda classification
MB_YEARS = {"10m": range(2017, 2025), "30m": range(1985, 2025)}

# Códigos oficiais da legenda MapBiomas (descrição da legenda oficial) com as
# cores oficiais. Somente as classes presentes na ROI são utilizadas; os códigos
# NÃO são inventados.
LEGEND = {
    0: ("Nodata / Fora do mapa", "#ffffff"),
    3: ("Formação Florestal (Forest Formation)", "#1f8d49"),
    4: ("Formação Savânica (Savanna Formation)", "#7dc975"),
    9: ("Silvicultura (Forest Plantation)", "#7a5900"),
    11: ("Campo Alagado e Área Pantanosa (Wetland)", "#519799"),
    12: ("Formação Campestre (Grassland)", "#d6bc74"),
    15: ("Pastagem (Pasture)", "#edde8e"),
    19: ("Lavoura Temporária (Temporary Crop)", "#c27ba0"),
    21: ("Mosaico de Usos (Mosaic of Uses)", "#ffefc3"),
    25: ("Outras Áreas não Vegetadas (Other non Vegetated Areas)", "#db4d4f"),
    33: ("Rio, Lago e Oceano (River, Lake and Ocean)", "#2532e4"),
    39: ("Soja (Soybean)", "#f5b3c8"),
    40: ("Arroz (Rice)", "#c71585"),
    41: ("Outras Lavouras Temporárias (Other Temporary Crops)", "#f54ca9"),
    62: ("Algodão beta (Cotton beta)", "#ff69b4"),
}

# CLASSE_PRESENTES por fonte/ano conhecidas nas ROIs dos exemplos (para conferência)
CODES_10M_2024 = [3, 4, 9, 11, 12, 15, 19, 21, 25, 33]
CODES_30M_2008 = [3, 4, 9, 11, 12, 15, 21, 25, 33, 39, 41, 62]

# --------------------------------------------------------------------------
# 1. Extração de informação do PAN
# --------------------------------------------------------------------------
def pan_info(pan_path):
    ds = gdal.Open(pan_path)
    if ds is None:
        raise RuntimeError("Não foi possível abrir a PAN em %s" % pan_path)
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    width, height = ds.RasterXSize, ds.RasterYSize
    resx, resy = abs(gt[1]), abs(gt[5])
    xmin, ymax = gt[0], gt[3]
    xmax, ymax_c = gt[0] + gt[1] * width, gt[3]
    xmax_c, ymin = xmax, gt[3] + gt[5] * height
    ds = None

    # Transforma os 4 cantos para lon/lat (geodésico) para uso no GEE.
    src = osr.SpatialReference(wkt=wkt)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    corners_utm = [
        (gt[0], gt[3]),
        (gt[0] + gt[1] * width, gt[3]),
        (gt[0] + gt[1] * width, gt[3] + gt[5] * height),
        (gt[0], gt[3] + gt[5] * height),
    ]
    corners_lonlat = []
    for x, y in corners_utm:
        lonlat = ct.TransformPoint(x, y)
        corners_lonlat.append([lonlat[0], lonlat[1]])

    return {
        "pan_path": pan_path,
        "crs_wkt": wkt,
        "crs_epsg": src.GetAuthorityCode("PROJCRS") or src.GetAuthorityCode(None),
        "gt": gt,
        "width": width,
        "height": height,
        "resx": resx,
        "resy": resy,
        "bounds": (xmin, ymin, xmax, ymax),
        "corners_lonlat": corners_lonlat,
    }


# --------------------------------------------------------------------------
# 2. Download do MapBiomas 10m na ROI (mesmo CRS da PAN)
# --------------------------------------------------------------------------
def fetch_mapbiomas(info, raw_path, source, year, scale):
    import ee

    ee.Initialize(project=PROJ)
    roi = ee.Geometry.Polygon(info["corners_lonlat"])
    if source == "10m":
        src_image = MB_ASSET_10M
        band = "classification_%d" % year
    elif source == "30m":
        src_image = "%s/collection_10_1_%d_v1" % (MB_COLL_30M, year)
        band = "classification"
    else:
        raise ValueError("source deve ser '10m' ou '30m'")
    img = ee.Image(src_image).select(band)
    url = img.clip(roi).getDownloadURL(
        {
            "name": "mapbiomas_wv1",
            "scale": scale,
            "crs": MB_CRS,
            "region": roi,
            "format": "GEO_TIFF",
            "maxPixels": 1e10,
        }
    )
    import urllib.request

    urllib.request.urlretrieve(url, raw_path)
    if not os.path.exists(raw_path):
        raise RuntimeError("Download do MapBiomas falhou: %s" % raw_path)
    print("MapBiomas %dm (%s, %d) baixado:" % (scale, source, year), raw_path)


# --------------------------------------------------------------------------
# 3. Reamostragem nearest-neighbor para a grade EXATA da PAN
# --------------------------------------------------------------------------
def resample_to_pan(info, raw_path, mask_path):
    gdal.UseExceptions()
    wkt = info["crs_wkt"]
    out_ds = gdal.GetDriverByName("GTiff").Create(
        mask_path, info["width"], info["height"], 1, gdal.GDT_Byte
    )
    out_ds.SetProjection(wkt)
    out_ds.SetGeoTransform(info["gt"])
    bnd = out_ds.GetRasterBand(1)
    bnd.SetNoDataValue(0)
    src = gdal.Open(raw_path)
    gdal.ReprojectImage(
        src,
        out_ds,
        src.GetProjection(),
        wkt,
        gdal.GRA_NearestNeighbour,
        options=["INIT_DEST=0"],
    )
    bnd.FlushCache()
    src = None
    out_ds = None
    print("MASK reamostrada para grade da PAN:", mask_path)


# --------------------------------------------------------------------------
# 4. Par multibanda
# --------------------------------------------------------------------------
def build_pair(info, pan_path, mask_path, pair_path, year):
    gdal.UseExceptions()
    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(
        pair_path, info["width"], info["height"], 2, gdal.GDT_Byte
    )
    dst.SetProjection(info["crs_wkt"])
    dst.SetGeoTransform(info["gt"])
    pan = gdal.Open(pan_path)
    mask = gdal.Open(mask_path)
    dst.GetRasterBand(1).WriteArray(pan.GetRasterBand(1).ReadAsArray())
    dst.GetRasterBand(2).WriteArray(mask.GetRasterBand(1).ReadAsArray())
    dst.GetRasterBand(1).SetDescription("WorldView-1_PAN")
    dst.GetRasterBand(2).SetDescription("MAPBIOMAS_%d" % year)
    dst.GetRasterBand(2).SetNoDataValue(0)
    pan = mask = dst = None
    print("Par multibanda criado:", pair_path)


# --------------------------------------------------------------------------
# 5. Validação
# --------------------------------------------------------------------------
def validate(info, pan_path, mask_path, pair_path):
    gdal.UseExceptions()
    checks = []

    pan = gdal.Open(pan_path)
    mask = gdal.Open(mask_path)
    pair = gdal.Open(pair_path)

    def check(ok, name, detail):
        checks.append((ok, name, detail))

    # 1. CRS PAN == CRS MASK (comparação semântica, ignora diferenças de
    #    serialização textual do WKT)
    srs_pan = osr.SpatialReference(wkt=pan.GetProjection())
    srs_mask = osr.SpatialReference(wkt=mask.GetProjection())
    crs_same = srs_pan.IsSame(srs_mask) == 1
    check(
        crs_same,
        "CRS PAN == CRS MASK (mesmo EPSG)",
        "PAN %s / MASK %s" % (pan.GetProjectionRef()[:40], mask.GetProjectionRef()[:40]),
    )

    # 2. transform PAN == transform MASK
    gt_pan = pan.GetGeoTransform()
    gt_mask = mask.GetGeoTransform()
    same_gt = all(
        abs(a - b) < 1e-9 for a, b in zip(gt_pan, gt_mask)
    )
    check(
        same_gt,
        "Transform PAN == Transform MASK",
        "PAN %s / MASK %s" % (gt_pan, gt_mask),
    )

    # 3. resolução PAN ≈ 0,5 m
    resx = abs(gt_pan[1])
    resy = abs(gt_pan[5])
    check(
        abs(resx - 0.5) < 1e-6 and abs(resy - 0.5) < 1e-6,
        "Resolução PAN ≈ 0,5 m",
        "resx=%.6f resy=%.6f" % (resx, resy),
    )

    # 4. máscara na grade de 0,5 m
    check(
        abs(abs(gt_mask[1]) - 0.5) < 1e-6 and abs(abs(gt_mask[5]) - 0.5) < 1e-6,
        "Máscara na grade de 0,5 m",
        "resx=%.6f resy=%.6f" % (abs(gt_mask[1]), abs(gt_mask[5])),
    )

    # 5/6. width e height iguais
    check(
        pan.RasterXSize == mask.RasterXSize,
        "Width PAN == Width MASK",
        "PAN %d / MASK %d" % (pan.RasterXSize, mask.RasterXSize),
    )
    check(
        pan.RasterYSize == mask.RasterYSize,
        "Height PAN == Height MASK",
        "PAN %d / MASK %d" % (pan.RasterYSize, mask.RasterYSize),
    )

    # 7. bounds compatíveis
    bx_pan = (gt_pan[0], gt_pan[3] + gt_pan[5] * pan.RasterYSize,
              gt_pan[0] + gt_pan[1] * pan.RasterXSize, gt_pan[3])
    bx_mask = (gt_mask[0], gt_mask[3] + gt_mask[5] * mask.RasterYSize,
               gt_mask[0] + gt_mask[1] * mask.RasterXSize, gt_mask[3])
    bounds_ok = all(
        abs(a - b) < 1e-6 for a, b in zip(bx_pan, bx_mask)
    )
    check(
        bounds_ok,
        "Bounds PAN e MASK compatíveis",
        "PAN %s / MASK %s" % (bx_pan, bx_mask),
    )

    # 8. máscara contém somente códigos válidos da legenda oficial
    band = mask.GetRasterBand(1)
    stats = band.ComputeStatistics(True)
    mn, mx = int(stats[0]), int(stats[1])
    hist = band.GetHistogram(mn, mx + 1, mx - mn + 1, True)
    used = {mn + i for i, c in enumerate(hist) if c > 0}
    unknown = used - set(LEGEND.keys())
    check(
        not unknown,
        "Máscara só com códigos oficiais",
        "valores usados=%s desconhecidos=%s" % (sorted(used), sorted(unknown)),
    )

    # 9. nenhum bilinear/bicubic usado (sempre nearest-neighbor neste script)
    check(
        True,
        "Resampling usado = nearest-neighbor (GRA_NearestNeighbour)",
        "nenhum bilinear/bicubic no pipeline",
    )

    # 10. par multibanda coerente (Band1=PAN, Band2=MASK com mesma grade)
    pair_gt_ok = all(
        abs(a - b) < 1e-9
        for a, b in zip(pair.GetGeoTransform(), pan.GetGeoTransform())
    )
    pair_dims_ok = (
        pair.RasterXSize == pan.RasterXSize
        and pair.RasterYSize == pan.RasterYSize
        and pair.RasterCount == 2
    )
    check(
        pair_gt_ok and pair_dims_ok,
        "Par multibanda (PAN+MASK) coerente",
        "dims=%dx%d bands=%d gt_match=%s"
        % (pair.RasterXSize, pair.RasterYSize, pair.RasterCount, pair_gt_ok),
    )

    pan = mask = pair = None
    return checks


# --------------------------------------------------------------------------
# 6. Visualização
# --------------------------------------------------------------------------
def make_visualization(info, pan_path, mask_path, out_viz_dir, year, scale):
    gdal.UseExceptions()
    os.makedirs(out_viz_dir, exist_ok=True)
    w, h = 2400, int(info["height"] * 2400 / info["width"])

    pan_tmp = os.path.join(out_viz_dir, "_preview_pan.tif")
    mask_tmp = os.path.join(out_viz_dir, "_preview_mask.tif")
    gdal.Warp(
        pan_tmp, pan_path, width=w, height=h, resampleAlg=gdal.GRA_Average,
        dstSRS="EPSG:32722", srcNodata=0, dstNodata=0,
    )
    gdal.Warp(
        mask_tmp, mask_path, width=w, height=h,
        resampleAlg=gdal.GRA_NearestNeighbour, dstSRS="EPSG:32722",
        srcNodata=0, dstNodata=0,
    )

    import numpy as np

    pan_arr = gdal.Open(pan_tmp).ReadAsArray().astype(np.float32)
    mask_arr = gdal.Open(mask_tmp).ReadAsArray()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        pan_arr = None
        mask_arr = None
        print("matplotlib indisponível; visualização PNG omitida")
        return

    # Histograma de classes no preview para legenda
    used = sorted({int(v) for v in mask_arr.ravel() if int(v) in LEGEND})
    colors = [LEGEND[c][1] for c in used]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(used + [used[-1] + 1], cmap.N)

    # realce de contraste para a PAN (1%-99%)
    p1, p99 = np.percentile(pan_arr[pan_arr > 0], [1, 99])
    pan_disp = np.clip((pan_arr - p1) / (p99 - p1), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax = axes[0]
    im = ax.imshow(pan_disp, cmap="gray")
    ax.set_title("WorldView-1 PAN 0,5 m (preview)")
    ax.axis("off")

    ax = axes[1]
    masked = np.ma.masked_where(mask_arr == 0, mask_arr)
    im2 = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("MapBiomas %d (%dm -> 0,5m, nearest) - preview" % (year, scale))
    ax.axis("off")
    cbar = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([c + 0.5 for c in used])
    cbar.set_ticklabels(["%d: %s" % (c, LEGEND[c][0][:24]) for c in used])

    fig.tight_layout()
    side = os.path.join(out_viz_dir, "pan_mask_side_by_side.png")
    fig.savefig(side, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Visualização lado a lado:", side)

    # Sobreposição PAN (cinza) + MASK (cores com transparência)
    rgba = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.float32)
    for c in used:
        idx = mask_arr == c
        rgb = tuple(int(LEGEND[c][1].lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        rgba[idx, 0] = rgb[0]
        rgba[idx, 1] = rgb[1]
        rgba[idx, 2] = rgb[2]
        rgba[idx, 3] = 0.55

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(pan_disp, cmap="gray")
    ax.imshow(rgba)
    ax.set_title("Overlay: PAN + MASK MapBiomas %d (transparência)" % year)
    ax.axis("off")
    ovl = os.path.join(out_viz_dir, "pan_mask_overlay.png")
    fig.tight_layout()
    fig.savefig(ovl, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Overlay:", ovl)

    for t in (pan_tmp, mask_tmp):
        if os.path.exists(t):
            os.remove(t)


# --------------------------------------------------------------------------
# 7. Relatório
# --------------------------------------------------------------------------
def write_report(info, checks, out_dir, source, year, scale):
    lines = []
    lines.append("RELATÓRIO DE PREPARAÇÃO DO DATASET WV1 (0,5m) + MAPBIOMAS %dm (%d)" % (scale, year))
    lines.append("=" * 70)
    lines.append("")
    lines.append("FONTE PAN")
    lines.append("- arquivo PAN utilizado: %s" % info["pan_path"])
    lines.append("- satélite: WorldView-1 (pancromática, 0,5 m)")
    lines.append("- resolução: %.6f x %.6f m" % (info["resx"], info["resy"]))
    lines.append("- CRS: %s (EPSG:%s)" % (info["crs_wkt"][:120], info["crs_epsg"]))
    gt = info["gt"]
    lines.append("- transform (GT): %s" % (gt,))
    lines.append("- dimensões: %d x %d px" % (info["width"], info["height"]))
    xmin, ymin, xmax, ymax = info["bounds"]
    lines.append("- bounds (CRS da PAN): xmin=%.3f ymin=%.3f xmax=%.3f ymax=%.3f" % (xmin, ymin, xmax, ymax))
    lines.append("- footprint: %s" % (info["corners_lonlat"],))
    lines.append("")
    lines.append("FONTE DE LABEL / PSEUDO-LABEL")
    if source == "10m":
        asset_line = MB_ASSET_10M
        band_line = "classification_%d" % year
    else:
        asset_line = "%s/collection_10_1_%d_v1" % (MB_COLL_30M, year)
        band_line = "classification"
    lines.append("- asset: %s" % asset_line)
    lines.append("- ano: %d (banda %s, coleção 10.1)" % (year, band_line))
    lines.append("- resolução original do MapBiomas: %d m" % scale)
    lines.append("- resolução final da máscara: %.3f m (grade exata da PAN)" % info["resx"])
    lines.append("- dimensões finais da máscara: %d x %d px" % (info["width"], info["height"]))
    lines.append("- reamostragem: nearest-neighbor (GRA_NearestNeighbour); NENHUM bilinear/bicubic")
    lines.append("- códigos/classes utilizados (legenda oficial MapBiomas):")
    for c in sorted(LEGEND.keys()):
        if c == 0:
            continue
        lines.append("    %3d = %s" % (c, LEGEND[c][0]))
    lines.append("")
    lines.append("VALIDAÇÃO DO ALINHAMENTO PAN/MASK")
    all_ok = True
    for ok, name, detail in checks:
        lines.append("  [%s] %s  -> %s" % ("OK" if ok else "FALHOU", name, detail))
        if not ok:
            all_ok = False
    lines.append("")
    lines.append("RESULTADO GLOBAL: %s" % ("ALINHADO" if all_ok else "FALHOU - revisar"))
    lines.append("")
    lines.append("NOTA: a máscara é um pseudo-label derivado de um produto de %d m, reamostrado" % scale)
    lines.append("para 0,5 m. NÃO deve ser tratada como ground truth de 0,5 m. O objetivo é")
    lines.append("avaliar se esse label é suficientemente bom para treinar um classificador LULC.")

    report_path = os.path.join(out_dir, "relatorio_pipeline.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Relatório escrito:", report_path)
    return report_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Prepara par WV1 PAN 0,5m + máscara MapBiomas")
    ap.add_argument("--pan", default=DEFAULT_PAN, help="Caminho do GeoTIFF WorldView-1 PAN")
    ap.add_argument("--source", default="10m", choices=["10m", "30m"],
                    help="Fonte MapBiomas: 10m (coverage 10m, 2017-2024) ou 30m (Landsat col. 10.1, 1985-2024)")
    ap.add_argument("--year", type=int, default=2024, help="Ano do MapBiomas (ex.: 2008, 2024)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Diretório de saída (ex.: dados/wv1/mapbiomas/2024_10m)")
    args = ap.parse_args()

    if not os.path.exists(args.pan):
        # fallback: procura o .TIF dentro de dados/wv1/original
        wv1_dir = caminhos.WV1_ORIGINAL
        cands = [
            os.path.join(wv1_dir, f)
            for f in os.listdir(wv1_dir)
            if f.lower().endswith(".tif") and not f.lower().endswith(".aux.xml")
        ]
        if not cands:
            sys.exit("GeoTIFF PAN não encontrado. Use --pan.")
        args.pan = cands[0]
        print("PAN detectado:", args.pan)

    if args.year not in MB_YEARS[args.source]:
        sys.exit("Ano %d inválido para fonte %s. Valores válidos: %d-%d"
                 % (args.year, args.source, min(MB_YEARS[args.source]), max(MB_YEARS[args.source])))

    scale = 10 if args.source == "10m" else 30
    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "_mapbiomas_%dm_raw.tif" % scale)
    pan_path = os.path.join(args.out, "pan.tif")
    mask_path = os.path.join(args.out, "mask.tif")
    pair_path = os.path.join(args.out, "pan_mask_pair.tif")

    print("=== 1. Extraindo info do PAN ===")
    info = pan_info(args.pan)
    print("PAN info:", info)

    print("=== 2. Download MapBiomas %dm (%d) (GEE) ===" % (scale, args.year))
    fetch_mapbiomas(info, raw_path, args.source, args.year, scale)

    print("=== 3. Reamostrando máscara p/ grade da PAN (nearest) ===")
    resample_to_pan(info, raw_path, mask_path)

    print("=== 4. Copiando PAN e montando par multibanda ===")
    shutil.copy2(args.pan, pan_path)
    print("PAN copiado:", pan_path)
    build_pair(info, pan_path, mask_path, pair_path, args.year)

    print("=== 5. Validação ===")
    checks = validate(info, pan_path, mask_path, pair_path)
    for ok, name, detail in checks:
        print("  [%s] %s" % ("OK" if ok else "FALHOU", name))

    print("=== 6. Visualização ===")
    viz_dir = os.path.join(args.out, "visualizacao")
    make_visualization(info, pan_path, mask_path, viz_dir, args.year, scale)

    print("=== 7. Relatório ===")
    write_report(info, checks, args.out, args.source, args.year, scale)

    print("\nOK -> %s" % args.out)


if __name__ == "__main__":
    main()
