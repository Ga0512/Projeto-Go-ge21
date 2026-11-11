# Classificação LULC — Goiás (WorldView-1 e SPOT5)

Classificação de uso e cobertura do solo em imagens pancromáticas de alta
resolução, usando o MapBiomas (via Google Earth Engine) como pseudo-label.
Python + GDAL + GEE; modelos XGBoost e U-Net/UNet++.

## Mapa do projeto

```
Data/
├─ scripts/                    todo o código
│  ├─ caminhos.py              ← TODOS os caminhos do projeto (mude pastas só aqui)
│  ├─ wv1/                     pipeline WorldView-1: MapBiomas, features, máscaras, patches
│  ├─ spot5/                   MapBiomas na grade de cenas SPOT5 + corte/junção em tiles
│  │  └─ legado/               scripts antigos do recorte SPOT5 (substituídos por *_cena.py)
│  ├─ datasets/                dataset de patches a partir de PAN + máscara refinada
│  ├─ editor/guia_edicao.py    GUI (PySide6) para corrigir a máscara à mão
│  ├─ modelos/                 config, dataset, XGBoost (xgb_lulc.py), U-Net (train.py)
│  ├─ visualizacao/            PNGs de predição e de comparação
│  └─ testes/                  experimentos pontuais (SLIC) e smoke test do editor
│
├─ dados/
│  ├─ wv1/                     WorldView-1, 0,5 m, 2008-07-10, 22400x18336, EPSG:32722
│  │  ├─ original/             PAN .TIF + classificação manual (CLASSIFICACAO_WV1.shp)
│  │  ├─ mapbiomas/<ano>_<res>/  MapBiomas reamostrado p/ grade da PAN (mask.tif,
│  │  │                        pan_mask_pair.tif, relatório, visualizacao/)
│  │  ├─ features/             pan_features.tif (4 bandas: PAN, LocalSTD, GLCM, Gradiente)
│  │  ├─ mascaras/             labels derivados: mask_2007_suavizada, mask_manual
│  │  └─ patches/              patches 512x512 de treino: treino_2007/, treino_manual/
│  ├─ spot5/                   SPOT5 — uma pasta por cena
│  │  ├─ recorte_2007-09-10/   recorte pequeno (~5 m) + classificação manual
│  │  │  ├─ original/
│  │  │  ├─ mapbiomas_2007_30m/  mapbiomas_2024_10m/
│  │  │  └─ mascara_refinada.tif   ← edição manual (não sobrescrever)
│  │  ├─ cena_013-014_2007-06-14/  cena grande (2,5 m, 29488x29067), editada em tiles
│  │  │  ├─ original/
│  │  │  ├─ mapbiomas_2007_30m/
│  │  │  └─ tiles/rRR_cCC/     imagem, mascara_mapbiomas, mascara_refinada (edição manual)
│  │  └─ colinas_do_sul/       mosaico do município (2,5 m), sobrepõe a 013-014; editado em tiles
│  │     ├─ original/          Colinas_do_Sul_mosaico_v2.tif
│  │     ├─ mapbiomas_2007_30m/
│  │     └─ tiles/rRR_cCC/     95 tiles
│  └─ datasets/
│     ├─ spot5_refinado/       ROTULADO v1 (recorte + tile 013-014): 500 patches, 3 classes
│     ├─ spot5_refinado_v2/    ROTULADO v2 (+ tile Colinas r06_c05): 750 patches, 3 classes
│     ├─ spot5_refinado_v3/    ROTULADO v3 (+ Colinas r04_c02, 4 classes): 750 patches (250 por cena)
│     ├─ spot5_refinado_v4/    ROTULADO v4 (4 classes, 250 por tile refinado): 1000 patches
│     ├─ spot5_refinado_v5/    ROTULADO v5 (ATUAL; + Colinas r00_c06): 1250 patches —
│     │                        images/ + masks/ (GeoTIFF), manifest.csv, splits/, receita.json, visualizacao/
│     └─ spot5_013-014_sem_rotulo/  SEM RÓTULO: só images/ (cena 013-014 inteira, menos
│                              as áreas rotuladas), manifest.csv, receita.json
│
├─ modelos/                    saídas de treino: modelos/<nome>/ (modelo, relatório, máscara)
├─ experimentos/               testes exploratórios arquivados (slic_test, sandbox_30m)
├─ _para_apagar/               cópias duplicadas (ver LEIAME.txt) — pode apagar
└─ .reorganizacao/             registro da reorganização de 2026-09-11 + desfazer.sh
```

Regra geral: `original/` é entrada bruta (não editar); tudo fora dela é gerado
por algum script de `scripts/`, exceto os `mascara_refinada.tif`, que são
trabalho manual feito no editor.

## Como rodar

O ambiente é Linux/WSL (`.venv`). A partir da raiz do projeto
(`/mnt/d/Projects/projeto-go/Data` no WSL):

```bash
# editor de máscaras
.venv/bin/python scripts/editor/guia_edicao.py

# WV1: MapBiomas -> features -> máscaras -> patches
.venv/bin/python scripts/wv1/gerar_dataset_wv1.py --source 30m --year 2008 --out dados/wv1/mapbiomas/2008_30m
.venv/bin/python scripts/wv1/gerar_features_pan.py
.venv/bin/python scripts/wv1/suavizar_mascara.py --mask dados/wv1/mapbiomas/2007_30m/mask.tif \
    --out dados/wv1/mascaras/mask_2007_suavizada.tif --alias 2007_30m
.venv/bin/python scripts/wv1/gerar_mascara_manual.py
.venv/bin/python scripts/wv1/gerar_patches_treino.py --mask dados/wv1/mascaras/mask_2007_suavizada.tif \
    --out dados/wv1/patches/treino_2007 --aug full --stride 256
.venv/bin/python scripts/wv1/gerar_patches_manual.py

# modelos
.venv/bin/python scripts/modelos/xgb_lulc.py --treino dados/wv1/patches/treino_manual \
    --saida modelos/xgb_manual --usar-splits --max-amostras 6000000
.venv/bin/python scripts/modelos/train.py --treino dados/wv1/patches/treino_manual \
    --saida modelos/unet_manual --usar-splits --arquitetura unetpp_r34

# dataset de patches: para cada cena, PAN + máscara total (0 = sem rótulo; ou a pasta tiles/) + peso;
# regra: 250 patches por área refinada -> peso = nº de tiles refinados da cena, --total = 250 x soma dos pesos
# --simular mostra quanto cabe sem gravar
.venv/bin/python scripts/datasets/gerar_dataset_refinado.py --saida dados/datasets/spot5_refinado_v5 --total 1250 \
    --cena dados/spot5/recorte_2007-09-10/original/IMAGEM_SPOT5_10092007.tif dados/spot5/recorte_2007-09-10/mascara_refinada.tif 1 \
    --cena dados/spot5/cena_013-014_2007-06-14/original/013-014_S5_708-378-0_2007-06-14_modificado.tif dados/spot5/cena_013-014_2007-06-14/tiles 1 \
    --cena dados/spot5/colinas_do_sul/original/Colinas_do_Sul_mosaico_v2.tif dados/spot5/colinas_do_sul/tiles 3
# refazer o mesmo dataset, idêntico:
.venv/bin/python scripts/datasets/gerar_dataset_refinado.py --receita dados/datasets/spot5_refinado_v5/receita.json --sobrescrever

# patches SEM rótulo de uma cena inteira (pasta própria!), na mesma escala do dataset rotulado
.venv/bin/python scripts/datasets/gerar_patches_sem_rotulo.py --saida dados/datasets/spot5_013-014_sem_rotulo \
    --pan dados/spot5/cena_013-014_2007-06-14/original/013-014_S5_708-378-0_2007-06-14_modificado.tif \
    --excluir dados/spot5/cena_013-014_2007-06-14/tiles --excluir dados/spot5/colinas_do_sul/tiles \
    --estatisticas dados/datasets/spot5_refinado_v5/receita.json

# SPOT5: máscara MapBiomas de uma cena nova, tiles para o editor, máscara final
.venv/bin/python scripts/spot5/gerar_mascara_mapbiomas_cena.py --imagem dados/spot5/<cena>/original/<img>.tif --ano 2007
.venv/bin/python scripts/spot5/tiles_cena.py cortar --imagem dados/spot5/<cena>/original/<img>.tif \
    --mascara dados/spot5/<cena>/mapbiomas_2007_30m/MASK_MAPBIOMAS_2007.tif
.venv/bin/python scripts/spot5/tiles_cena.py juntar --pasta dados/spot5/<cena>/tiles
```

Detalhes técnicos (regras de reamostragem, gotchas GDAL/GEE, classes): `CLAUDE.md`.
