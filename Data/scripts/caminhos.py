#!/usr/bin/env python3
"""Caminhos do projeto — fonte única. Mudou uma pasta de lugar? Mude só aqui.

Os scripts em scripts/<assunto>/ importam este módulo com:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import caminhos
"""

import os

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPTS = os.path.join(RAIZ, "scripts")
DADOS = os.path.join(RAIZ, "dados")
DATASETS = os.path.join(DADOS, "datasets")         # datasets de patches gerados
MODELOS = os.path.join(RAIZ, "modelos")            # saídas de treino (xgb_*, unet_*)
EXPERIMENTOS = os.path.join(RAIZ, "experimentos")  # testes exploratórios arquivados

# ------------------------------------------------ WorldView-1 (0,5 m, 2008-07-10)
WV1 = os.path.join(DADOS, "wv1")
WV1_ORIGINAL = os.path.join(WV1, "original")
WV1_PAN = os.path.join(WV1_ORIGINAL, "08JUL10134220-P3DS-050316155010_01_P001.TIF")
WV1_SHP_MANUAL = os.path.join(WV1_ORIGINAL, "CLASSIFICACAO_WV1.shp")
# MapBiomas na grade da PAN, uma pasta por <ano>_<fonte> (gerar_dataset_wv1.py)
WV1_MAPBIOMAS = os.path.join(WV1, "mapbiomas")
WV1_MASK_2007 = os.path.join(WV1_MAPBIOMAS, "2007_30m", "mask.tif")
WV1_MASK_2008 = os.path.join(WV1_MAPBIOMAS, "2008_30m", "mask.tif")
WV1_FEATURES = os.path.join(WV1, "features", "pan_features.tif")
# máscaras derivadas usadas como label
WV1_MASCARAS = os.path.join(WV1, "mascaras")
WV1_MASK_SUAVIZADA = os.path.join(WV1_MASCARAS, "mask_2007_suavizada.tif")
WV1_MASK_MANUAL = os.path.join(WV1_MASCARAS, "mask_manual.tif")
# patches 512x512 de treino (.npz), uma pasta por fonte de label
WV1_PATCHES = os.path.join(WV1, "patches")
WV1_TREINO_2007 = os.path.join(WV1_PATCHES, "treino_2007")
WV1_TREINO_MANUAL = os.path.join(WV1_PATCHES, "treino_manual")

# ------------------------------- WorldView Jataí (0,5 m, EPSG:4674, 274176x293760)
# imagem nova de 2026-10; ~70% do retângulo é fundo 0 (mosaico com vãos).
# As texturas são as bandas 2 e 3 do dataset (gerar_features_cena.py --byte):
# Byte quantizado com offset/escala declarados, 0 = sem imagem.
WV1_JATAI = os.path.join(WV1, "jatai")
WV1_JATAI_PAN = os.path.join(WV1_JATAI, "jatai_vantor_4674.tif")
WV1_JATAI_GLCM = os.path.join(WV1_JATAI, "jatai_vantor_4674_glcm.tif")
WV1_JATAI_RUGOSIDADE = os.path.join(WV1_JATAI, "jatai_vantor_4674_rugosidade.tif")
# GT de classes desenhado sobre esta imagem: 10 blocos de 5x5 km rotulados de
# parede a parede (EPSG:31982). Campos: id (bloco), classe, macro.
WV1_JATAI_GT = os.path.join(WV1_JATAI, "gt", "jatai_amostra_v1.shp")
WV1_JATAI_BLOCOS = os.path.join(WV1_JATAI, "blocos")

# ------------------------------------------------------------------- SPOT5
SPOT5 = os.path.join(DADOS, "spot5")
# recorte pequeno de 2007-09-10 (2758x1465, EPSG:4326, pixel ~5 m) + classificação manual
SPOT5_RECORTE = os.path.join(SPOT5, "recorte_2007-09-10")
SPOT5_RECORTE_IMAGEM = os.path.join(SPOT5_RECORTE, "original", "IMAGEM_SPOT5_10092007.tif")
SPOT5_RECORTE_SHP_MANUAL = os.path.join(SPOT5_RECORTE, "original", "CLASSIFICACAO_SPOT5.shp")
SPOT5_RECORTE_MB_2007 = os.path.join(SPOT5_RECORTE, "mapbiomas_2007_30m")
SPOT5_RECORTE_MB_2024 = os.path.join(SPOT5_RECORTE, "mapbiomas_2024_10m")
SPOT5_RECORTE_REFINADA = os.path.join(SPOT5_RECORTE, "mascara_refinada.tif")  # edição manual
# cena grande 013-014 de 2007-06-14 (29488x29067, EPSG:32722), editada em tiles
SPOT5_CENA_013 = os.path.join(SPOT5, "cena_013-014_2007-06-14")
SPOT5_CENA_013_IMAGEM = os.path.join(
    SPOT5_CENA_013, "original", "013-014_S5_708-378-0_2007-06-14_modificado.tif")
SPOT5_CENA_013_MB_2007 = os.path.join(SPOT5_CENA_013, "mapbiomas_2007_30m")
SPOT5_CENA_013_TILES = os.path.join(SPOT5_CENA_013, "tiles")
# mosaico Colinas do Sul (29068x32624, EPSG:32722, 2,5 m, sem data; 0 = fundo),
# sobrepõe boa parte da cena 013-014 (inclusive o tile r06_c08), editado em tiles
SPOT5_COLINAS = os.path.join(SPOT5, "colinas_do_sul")
SPOT5_COLINAS_IMAGEM = os.path.join(SPOT5_COLINAS, "original", "Colinas_do_Sul_mosaico_v2.tif")
SPOT5_COLINAS_MB_2007 = os.path.join(SPOT5_COLINAS, "mapbiomas_2007_30m")
SPOT5_COLINAS_TILES = os.path.join(SPOT5_COLINAS, "tiles")
# Jataí, sudoeste de GO (44327x49234, EPSG:32722, 2,5 m, sem data no arquivo;
# 0 = fundo). ~12.600 km² com imagem; longe das outras cenas (não se sobrepõe)
SPOT5_JATAI = os.path.join(SPOT5, "jatai")
SPOT5_JATAI_IMAGEM = os.path.join(SPOT5_JATAI, "original", "W052S18.tif")
SPOT5_JATAI_MB_2007 = os.path.join(SPOT5_JATAI, "mapbiomas_2007_30m")
SPOT5_JATAI_TILES = os.path.join(SPOT5_JATAI, "tiles")


def pasta_cena(imagem):
    """Pasta da cena de uma imagem: a mãe de original/ (layout dados/<sensor>/<cena>/original/),
    senão a própria pasta da imagem."""
    d = os.path.dirname(os.path.abspath(imagem))
    return os.path.dirname(d) if os.path.basename(d) == "original" else d
