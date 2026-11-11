from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import time

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from osgeo import gdal, ogr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
)

gdal.UseExceptions()

# ---------------------------------------------------------
# PARÂMETROS
# ---------------------------------------------------------

img_path = r"C:\Users\felipe.ferreira\OneDrive - GE21 Consultoria\Arquivos de Gabriel Cicotoste - SEMAD_GO_SPOT\Base Cartografica\TIFF\colinas_sul_mosaico_3b.tif"

training_path = r"C:\Users\felipe.ferreira\OneDrive - GE21 Consultoria\Arquivos de Gabriel Cicotoste - SEMAD_GO_SPOT\Base Cartografica\GEOJSON\Amostras_RF_Colinas_sul_treinamento.geojson"
validation_path = r"C:\Users\felipe.ferreira\OneDrive - GE21 Consultoria\Arquivos de Gabriel Cicotoste - SEMAD_GO_SPOT\Base Cartografica\GEOJSON\Amostras_RF_Colinas_sul_validacao.geojson"

pasta_saida = Path(img_path).parent / "Macro_02"

class_img_path = str(pasta_saida / "Class_RF_PAN_Texturas_Original.tif")
filter_img_path = str(pasta_saida / "Class_RF_PAN_Texturas_Filtrada.tif")
results_path = str(pasta_saida / "Results_RF_PAN_Texturas.txt")

n_trees = 500
n_cores = -1
attribute = "Macro_id"
num_pixels = 200
BLOCO = 2048

# NoData da entrada e da classificação.
NODATA_ENTRADA = -9999
NODATA_SAIDA = 0

pasta_saida.mkdir(parents=True, exist_ok=True)
stime = time.time()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(results_path, mode="w", encoding="utf-8"),
    ],
    force=True,
)

# ---------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------

def janelas(largura, altura):
    for y in range(0, altura, BLOCO):
        ny = min(BLOCO, altura - y)
        for x in range(0, largura, BLOCO):
            nx = min(BLOCO, largura - x)
            yield x, y, nx, ny


def ler_bloco(dataset, x, y, nx, ny):
    dados = dataset.ReadAsArray(x, y, nx, ny)

    # Também permite entrada com apenas uma banda.
    if dados.ndim == 2:
        dados = dados[np.newaxis, :, :]

    validos = np.all(np.isfinite(dados), axis=0)
    validos &= np.all(dados != NODATA_ENTRADA, axis=0)

    for i in range(dataset.RasterCount):
        banda = dataset.GetRasterBand(i + 1)
        nodata = banda.GetNoDataValue()

        if nodata is not None:
            if np.isnan(nodata):
                validos &= ~np.isnan(dados[i])
            else:
                validos &= dados[i] != nodata

        validos &= banda.GetMaskBand().ReadAsArray(
            x, y, nx, ny
        ) != 0

    return np.moveaxis(dados, 0, -1), validos


def rasterizar_amostras(caminho, destino, referencia):
    vetor = ogr.Open(caminho)
    camada = vetor.GetLayer()

    if camada.GetLayerDefn().GetFieldIndex(attribute) < 0:
        raise ValueError(f"Campo {attribute} ausente em: {caminho}")

    # Verifica os códigos antes de gravá-los como Byte.
    for feicao in camada:
        valor = feicao.GetField(attribute)

        if (
            valor is None
            or not np.isfinite(float(valor))
            or float(valor) != int(float(valor))
            or not 1 <= int(float(valor)) <= 255
        ):
            raise ValueError(
                f"{attribute} deve conter inteiros de 1 a 255: {caminho}"
            )

    camada.ResetReading()

    raster = gdal.GetDriverByName("GTiff").Create(
        destino,
        referencia.RasterXSize,
        referencia.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=[
            "TILED=YES",
            "COMPRESS=LZW",
            "BIGTIFF=IF_SAFER",
        ],
    )
    raster.SetProjection(referencia.GetProjection())
    raster.SetGeoTransform(referencia.GetGeoTransform())

    banda = raster.GetRasterBand(1)
    banda.Fill(0)
    banda.SetNoDataValue(0)

    erro = gdal.RasterizeLayer(
        raster,
        [1],
        camada,
        options=[
            f"ATTRIBUTE={attribute}",
        ],
    )
    if erro != gdal.CE_None:
        raise RuntimeError(f"Falha ao rasterizar: {caminho}")

    banda = None
    raster = None
    camada = None
    vetor = None


# ---------------------------------------------------------
# ABERTURA DA IMAGEM
# ---------------------------------------------------------

logging.info("Iniciando classificação.")
logging.info("Imagem: %s", img_path)

img_ds = gdal.Open(img_path)
row = img_ds.RasterYSize
col = img_ds.RasterXSize
band_number = img_ds.RasterCount
geot = img_ds.GetGeoTransform()
proj = img_ds.GetProjection()

logging.info(
    "Dimensões: %s linhas × %s colunas; %s bandas.",
    row, col, band_number,
)

total_blocos = (
    ((row + BLOCO - 1) // BLOCO)
    * ((col + BLOCO - 1) // BLOCO)
)

with TemporaryDirectory(
    prefix="rf_amostras_",
    dir=str(pasta_saida),
) as pasta_temporaria:

    treinamento_tif = str(
        Path(pasta_temporaria) / "treinamento.tif"
    )
    validacao_tif = str(
        Path(pasta_temporaria) / "validacao.tif"
    )

    # -----------------------------------------------------
    # TREINAMENTO: EXTRAI APENAS PIXELS DAS AMOSTRAS
    # -----------------------------------------------------

    logging.info("Rasterizando amostras de treinamento.")

    rasterizar_amostras(training_path, treinamento_tif, img_ds)
    treinamento_ds = gdal.Open(treinamento_tif)

    partes_X = []
    partes_y = []

    logging.info("Extraindo treinamento por blocos.")

    for indice, (x, y, nx, ny) in enumerate(
        janelas(col, row), start=1
    ):
        classes = treinamento_ds.ReadAsArray(x, y, nx, ny)

        if np.any(classes > 0):
            dados, validos = ler_bloco(img_ds, x, y, nx, ny)
            selecionados = (classes > 0) & validos

            if np.any(selecionados):
                partes_X.append(
                    dados[selecionados].astype(np.float32)
                )
                partes_y.append(classes[selecionados])

        print(
            f"\rTreinamento: {indice / total_blocos * 100:6.2f}%",
            end="",
            flush=True,
        )

    treinamento_ds = None
    print()

    if not partes_y:
        raise ValueError(
            "Nenhuma amostra de treinamento sobre pixels válidos."
        )

    X = np.concatenate(partes_X)
    y = np.concatenate(partes_y)
    del partes_X, partes_y

    labels = np.unique(y)
    n_samples = len(y)

    if len(labels) < 2:
        raise ValueError("O treinamento precisa de pelo menos duas classes.")

    logging.info("Pixels de treinamento: %s", n_samples)
    logging.info("Classes: %s", labels)
    logging.info("Treinando Random Forest.")

    rf = RandomForestClassifier(
        n_estimators=n_trees,
        oob_score=True,
        random_state=42,
        verbose=1,
        n_jobs=n_cores,
    )
    rf.fit(X, y)

    logging.info("Acurácia OOB: %.4f%%", rf.oob_score_ * 100)

    for i, importancia in enumerate(
        rf.feature_importances_, start=1
    ):
        nome = img_ds.GetRasterBand(i).GetDescription() or f"Banda {i}"
        logging.info(
            "Importância de %s: %.6f", nome, importancia
        )

    del X, y

    # -----------------------------------------------------
    # CLASSIFICA TODA A EXTENSÃO, POR BLOCOS
    # -----------------------------------------------------

    logging.info("Classificando toda a extensão da entrada.")

    out_ds = gdal.GetDriverByName("GTiff").Create(
        class_img_path,
        col,
        row,
        1,
        gdal.GDT_Byte,
        options=[
            "TILED=YES",
            "COMPRESS=LZW",
            "BIGTIFF=IF_SAFER",
        ],
    )
    out_ds.SetGeoTransform(geot)
    out_ds.SetProjection(proj)

    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(NODATA_SAIDA)
    out_band.SetDescription("Classes Random Forest")

    for indice, (x, y, nx, ny) in enumerate(
        janelas(col, row), start=1
    ):
        dados, validos = ler_bloco(img_ds, x, y, nx, ny)

        classificacao = np.full(
            (ny, nx), NODATA_SAIDA, dtype=np.uint8
        )

        if np.any(validos):
            classificacao[validos] = rf.predict(
                dados[validos].astype(np.float32)
            ).astype(np.uint8)

        out_band.WriteArray(classificacao, x, y)

        print(
            f"\rClassificação: {indice / total_blocos * 100:6.2f}%",
            end="",
            flush=True,
        )

    out_band = None
    out_ds = None
    print()

    # -----------------------------------------------------
    # FILTRAGEM: PRESERVA O NODATA
    # -----------------------------------------------------

    logging.info("Criando classificação filtrada.")

    original_ds = gdal.Open(class_img_path)
    filtrado_ds = gdal.GetDriverByName("GTiff").CreateCopy(
        filter_img_path,
        original_ds,
        options=[
            "TILED=YES",
            "COMPRESS=LZW",
            "BIGTIFF=IF_SAFER",
        ],
    )
    original_ds = None

    banda_filtro = filtrado_ds.GetRasterBand(1)
    mascara_filtro = banda_filtro.GetMaskBand()

    gdal.SieveFilter(
        srcBand=banda_filtro,
        maskBand=mascara_filtro,
        dstBand=banda_filtro,
        threshold=num_pixels,
        connectedness=8,
        callback=lambda p, msg, data: (
            print(
                f"\rFiltro: {p * 100:6.2f}%",
                end="",
                flush=True,
            ) or 1
        ),
    )

    mascara_filtro = None
    banda_filtro = None
    filtrado_ds = None
    print()

    # -----------------------------------------------------
    # VALIDAÇÃO DA CLASSIFICAÇÃO ORIGINAL
    # -----------------------------------------------------

    logging.info("Rasterizando amostras de validação.")

    rasterizar_amostras(validation_path, validacao_tif, img_ds)

    validacao_ds = gdal.Open(validacao_tif)
    classificacao_ds = gdal.Open(class_img_path)

    partes_referencia = []
    partes_predicao = []

    for x, y, nx, ny in janelas(col, row):
        referencia = validacao_ds.ReadAsArray(x, y, nx, ny)

        if not np.any(referencia > 0):
            continue

        predicao = classificacao_ds.ReadAsArray(x, y, nx, ny)

        selecionados = (
            (referencia > 0)
            & (predicao != NODATA_SAIDA)
        )

        if np.any(selecionados):
            partes_referencia.append(referencia[selecionados])
            partes_predicao.append(predicao[selecionados])

    validacao_ds = None
    classificacao_ds = None

    if not partes_referencia:
        raise ValueError(
            "Nenhuma amostra de validação sobre pixels classificados."
        )

    y_v = np.concatenate(partes_referencia)
    pred_v = np.concatenate(partes_predicao)
    del partes_referencia, partes_predicao

    # Inclui classes treinadas, observadas e preditas.
    labels_v = np.union1d(
        labels, np.union1d(y_v, pred_v)
    )

    relatorio = classification_report(
        y_v,
        pred_v,
        labels=labels_v,
        target_names=[str(c) for c in labels_v],
        zero_division=0,
    )

    overall_accuracy = accuracy_score(y_v, pred_v)
    kappa = cohen_kappa_score(y_v, pred_v)

    avg_precision = precision_score(
        y_v, pred_v, labels=labels_v,
        average="macro", zero_division=0,
    )
    avg_recall = recall_score(
        y_v, pred_v, labels=labels_v,
        average="macro", zero_division=0,
    )
    avg_f1 = f1_score(
        y_v, pred_v, labels=labels_v,
        average="macro", zero_division=0,
    )

    cm_val = confusion_matrix(
        y_v, pred_v, labels=labels_v
    )

    logging.info("Validação da classificação ORIGINAL:\n%s", relatorio)
    logging.info("Pixels de validação: %s", len(y_v))
    logging.info("Acurácia: %.4f%%", overall_accuracy * 100)
    logging.info("Kappa: %.4f", kappa)
    logging.info("Precisão macro: %.4f", avg_precision)
    logging.info("Recall macro: %.4f", avg_recall)
    logging.info("F1 macro: %.4f", avg_f1)

    matriz = pd.DataFrame(
        cm_val,
        index=pd.Index(labels_v, name="Referencia"),
        columns=pd.Index(labels_v, name="Predicao"),
    )

    csv_path = str(
        pasta_saida / "conf_matrix_validation.csv"
    )
    matriz.to_csv(csv_path)

    plt.figure(figsize=(10, 7))
    sn.heatmap(
        matriz,
        annot=True,
        fmt="g",
        cmap="viridis",
    )
    plt.title("Matriz de confusão — classificação original")
    plt.xlabel("Classe predita")
    plt.ylabel("Classe de referência")
    plt.tight_layout()
    plt.savefig(
        pasta_saida / "conf_matrix_validation.png",
        dpi=300,
    )
    plt.close()

img_ds = None

logging.info("Original: %s", class_img_path)
logging.info("Filtrada: %s", filter_img_path)
logging.info("Tempo total: %.1f minutos", (time.time() - stime) / 60)