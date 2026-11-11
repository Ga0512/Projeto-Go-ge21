#!/usr/bin/env python3
"""Configuração central do pipeline de modelos LULC (WV1 0,5m + MapBiomas).

Códigos MapBiomas presentes nas máscaras -> índices de classe do modelo.
0 (nodata) é ignorado no treino (ignore class).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

DEFAULT_TREINO = caminhos.WV1_TREINO_2007
DEFAULT_SAIDA = caminhos.MODELOS
DEFAULT_IMAGEM = caminhos.WV1_FEATURES   # cena inteira p/ inferência

PATCH = 512
BANDAS = ["PAN", "LocalSTD", "GLCM_Entropy", "Gradient"]
N_BANDAS = len(BANDAS)

# código MapBiomas -> índice de classe (ordem fixa e documentada)
# 4 classes finais (definidas pelo usuário em 2026-09-15). Nas máscaras
# refinadas à mão só entram 3 (floresta = veg. densa), 4 (savana = veg. não
# densa), 15 (pastagem = TUDO que não é vegetação natural) e 33 (água); os
# demais códigos (restos do MapBiomas não editados) seguem a mesma lógica:
#   0 = Vegetação densa:      3 Formação Florestal, 5 Mangue, 6 Floresta Alagável,
#                             49 Restinga Arbórea
#   1 = Vegetação não densa:  4 Formação Savânica, 11 Campo Alagado, 12 Formação
#                             Campestre, 50 Restinga Herbácea
#   2 = Não vegetação:        15 Pastagem e todo o resto não natural: 9 Silvicultura,
#                             19/39/40/41/62 lavouras temporárias, 20 Cana, 21 Mosaico
#                             de Usos, 23 Praia/Areal, 24 Urbano, 25 Outras não
#                             Vegetadas, 29 Afloramento Rochoso, 30 Mineração, 32 Apicum,
#                             35/36/46/47/48 lavouras perenes, 75 Usina Fotovoltaica
#   3 = Água:                 33 Rio/Lago/Oceano, 31 Aquicultura
CLASS_MAP = {3: 0, 5: 0, 6: 0, 49: 0,
             4: 1, 11: 1, 12: 1, 50: 1,
             15: 2, 9: 2, 19: 2, 20: 2, 21: 2, 23: 2, 24: 2, 25: 2, 29: 2, 30: 2,
             32: 2, 35: 2, 36: 2, 39: 2, 40: 2, 41: 2, 46: 2, 47: 2, 48: 2, 62: 2,
             75: 2,
             33: 3, 31: 3}
CLASS_NOMES = ["Vegetação densa", "Vegetação não densa", "Não vegetação", "Água"]
CLASS_CURTOS = ["veg_densa", "veg_nao_densa", "nao_veg", "agua"]   # colunas de manifest etc.
N_CLASSES = len(CLASS_NOMES)
IDX_AGUA = CLASS_NOMES.index("Água")
IGNORE = 255          # classe dummy p/ pixels sem label (nodata PAN ou mask 0)

# código MapBiomas representativo de cada classe p/ o GeoTIFF de saída
REPR_CODIGO = [3, 4, 15, 33]

# normalização por banda (stats reais: 349.6M pixels válidos, mask!=0, treino_2007)
MEAN = [30.1842, 4.5271, 0.9376, 24.2900]   # PAN, LocalSTD, GLCM_Entropy, Gradient
STD = [6.9151, 1.7515, 0.4314, 17.1077]

SPLIT_VAL = 0.15      # fração dos patches para validação
SPLIT_TEST = 0.15     # fração dos patches para teste (restante = treino)
SEED = 42