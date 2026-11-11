# AGENTS.md

Dataset de classificação LULC (Goiás) a partir de imagens pancromáticas de alta resolução, usando MapBiomas (GEE) como fonte de labels/pseudo-labels. Python + GDAL + Google Earth Engine.

## Ambiente (CRÍTICO)

- **Sempre rodar Python via o venv do projeto**: `.venv/bin/python ...`, a partir da raiz.
- O Python global é quebrado para GDAL: o gdal do sistema (3.8.4, em `/usr/lib/python3/dist-packages`) foi compilado para numpy 1.x, mas o user-site (`~/.local`) tem numpy 2.4.6. Importar `osgeo.gdal_array` no python global falha (`_ARRAY_API not found`).
- O venv (`python3 -m venv --system-site-packages .venv`) resolve com numpy 1.26.4 instalado dentro dele (faz sombra ao user-site).
- **Não instalar nada no Python global** (PEP 668). Todo `pip install` no venv, com `.venv/bin/python -m pip install ...` — o `.venv/bin/pip` está quebrado (shebang aponta para a pasta antiga `.../GOIAS_EXEMPLO_CLASSIFICACAO_USO_SOLO/.venv`).
- GEE autenticado no projeto `beaming-botany-373619` (`ee.Initialize(project=...)`).
- Sem git e sem testes automatizados — validação é feita pelos checks embutidos nos scripts (`validate()` em `gerar_dataset_wv1.py`, `--receita --sobrescrever` byte-a-byte em `gerar_dataset_refinado.py`) e pelos PNGs/relatórios gerados.
- `scripts/testes/teste_editor_headless.py` está desatualizado e tem `sys.exit` no topo: NÃO rodar (o `salvar()` dele pode sobrescrever uma `mascara_refinada.tif` real).
- Arquivos grandes em `/mnt/d` (WSL): renomear/mover pode falhar com "Permission denied" ou "No such device" se o drive estiver instável ou algo segurar a pasta no Windows — retentar após `wsl --shutdown` ou fechar apps com a pasta aberta.

## Estrutura

Mapa completo em `README.md`. Todos os caminhos vêm de `scripts/caminhos.py` (não hardcodar caminhos nos scripts).

- `dados/<sensor>/.../original/` — entradas brutas: GeoTIFF PAN (SPOT5: recorte ~5 m, cena 013-014 2,5 m; WV1 0,5 m), shapefile de classificação manual (atributos: `Classe`, `Area_km2`, `Area_ha`), `.qmd` (metadados QGIS, EPSG:4326).
- `dados/wv1/` — WV1:
  - `mapbiomas/<ano>_<res>/` (2007_30m, 2008_30m, 2024_10m) — saída de `gerar_dataset_wv1.py`: mask.tif, pan_mask_pair.tif, relatorio_pipeline.txt, visualizacao/ (PNGs side-by-side/overlay/RGB das features).
  - `features/pan_features.tif` — GeoTIFF 4 bandas (Float32, BIGTIFF, ~6.2G): band1=PAN, band2=LocalSTD (janela 7), band3=GLCM_Entropy (janela 7, L=8), band4=Gradient (sobel).
  - `mascaras/` — labels derivados (`mask_2007_suavizada.tif`, `mask_manual.tif`); `patches/` — patches de treino.
- `dados/spot5/<cena>/` — uma pasta por cena SPOT5 (`recorte_2007-09-10`, `cena_013-014_2007-06-14`, `colinas_do_sul`, `jatai`), com `original/`, `mapbiomas_<ano>_<res>/` (30m 2007 e 10m 2024), e a máscara refinada à mão (`mascara_refinada.tif` ou `tiles/rRR_cCC/mascara_refinada.tif`). **Colinas sobrepõe a 013-014** (mesmo terreno, outra imagem/brilho) e tem fundo 0 = sem imagem.
- `scripts/wv1/` — `gerar_dataset_wv1.py` (pipeline PAN+MASK) e `gerar_features_pan.py` (features 4 bandas), entre outros.
- `scripts/datasets/` — `gerar_dataset_refinado.py` (patches rotulados a partir de PAN + máscara total + peso por cena) e `gerar_patches_sem_rotulo.py` (cena inteira sem rótulo, pasta separada). Ambos reamostram para a grade comum 2,5 m EPSG:32722 e usam o `receita.json` para reprodutibilidade byte-a-byte.
- `scripts/modelos/` — `config.py` é a fonte única das 4 classes finais (`CLASS_MAP` código MapBiomas → índice; `REPR_CODIGO=[3,4,15,33]`; `IGNORE=255`); `xgb_lulc.py` e `train.py` (U-Net). **Alterar `CLASS_MAP` invalida modelos já treinados.**
- `scripts/spot5/gerar_mascara_mapbiomas_cena.py` — MapBiomas na grade de qualquer cena (genérico), `--res 30` (coleção 10.1, anos 1985–2024) ou `--res 10` (coleção 3, 2017–2024, banda `classification_<ano>`; usar ano 2024). `scripts/spot5/legado/gerar_mascara_mapbiomas.py` (2024 10m) e `gerar_mascara_mapbiomas_2007.py` (2007 30m) geraram as máscaras do recorte SPOT5. `scripts/spot5/tiles_cena.py` (`cortar`, `cortar-mascara`, `juntar`) cria os tiles do editor e monta a máscara final.
- `scripts/editor/guia_edicao.py` — **GUI PySide6** para corrigir bordas das classes, com a PAN ao fundo e as classes por cima, **coloridas pelo código MapBiomas (LEGEND oficial)**. A máscara raster é a fonte da verdade; os polígonos são revetorizados a cada edição (`gdal.Polygonize` + `shapely.coverage_simplify`). Atalhos e modos estão no docstring do script. As cenas editáveis ficam na lista `CENAS`. Seletor `cb_fonte` escolhe a máscara MapBiomas original da edição: **30 m (2007)** ou **10 m (2024)** — para cenas em tiles usa `mascara_mapbiomas{,_10m}.vrt`, priorizando `mascara_refinada` > `mascara_base` > MapBiomas.
- `dados/datasets/` — datasets de patches. **Atual: `spot5_refinado_v12`** (4 classes, `--total 3000`, pesos recorte 1 / 013-014 3 / Colinas 6 / Jatai 2; 2981 patches, splits 2081/451/449, `bloco=1024`, `zona_morta=320`). A regra é **250 patches por área refinada** → peso da cena = nº de tiles refinados; refinar tile novo aumenta o peso e o `--total` em 250.
- `dados/datasets/spot5_refinado_v6b_buffer5m/` — **cópia do v6b com `--buffer 5`** (m): erode as máscaras 5 m antes de selecionar/gabar patches (o modelo treina com o interior das classes, sem a faixa de transição). Mesmas cotas do v6b (1447 patches); só labels erodidos mudam.

## Dados de label (MapBiomas)

- 10m: asset `projects/mapbiomas-public/assets/brazil/lulc_10m/collection3/mapbiomas_10m_collection3_integration_v1` — bandas `classification_YYYY` (2017–2024). Usar ano 2024. Fator na grade SPOT5 2,5 m = 4.
- 30m: coleção 10.1 Landsat, asset por ano `projects/mapbiomas-public/assets/brazil/lulc/v1/collection_10_1_YYYY_v1`, banda `classification` (1985–2024). Usar 2008 (imagem WV1 de julho/2008).
- Códigos são os oficiais da legenda MapBiomas (não inventar). Presentes na ROI da WV1: 2024 (10m) = 3, 4, 9, 11, 12, 15, 19, 21, 25, 33; 2008 (30m) = 3, 4, 9, 11, 12, 15, 21, 25, 33, 39, 41, 62 (ver `LEGEND` no script para nomes/cores).
- Fatores de reamostragem: 10m/0.5m = 20; 30m/0.5m = 60 → cada pixel MapBiomas ≈ bloco 20×20 ou 60×60 na grade WV1.

## Regras do pipeline (não violar)

- Máscara **categórica inteira** (GDT_Byte), reamostrada com **nearest-neighbor** (`gdal.GRA_NearestNeighbour`). Nunca bilinear/bicubic.
- PAN e MASK devem ter o **mesmo CRS, mesmo affine transform, mesmas dimensões** (grade exata da PAN).
- Máscara é pseudo-label derivado de produto coarse (10m/30m) — nunca tratar como ground truth 0.5m.
- Passos: extrair footprint do GeoTIFF → usar como ROI → `ee.Image(MB_ASSET).select(band).clip(roi).getDownloadURL(scale=<10|30>, crs=EPSG:32722)` → `gdal.ReprojectImage` para a grade da PAN → validar (10 checks).
- GeoTIFF de features: Float32, TILED, COMPRESS=DEFLATE, PREDICTOR=2, **BIGTIFF=YES** (sem isso, >4GB falha em `WriteEncodedTile`).

## Gotchas GDAL

- Ao transformar cantos UTM → lon/lat para o GEE: forçar `dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)` no EPSG:4326, senão retorna (lat, lon) e o GEE interpreta errado.
- Comparar CRS com `osr.SpatialReference.IsSame()`, não string de WKT (serialização textual pode diferir).
- Passar footprint ao GEE como `ee.Geometry.Polygon` em lon/lat (geodésico) — retângulo em UTM falha (`Even/odd interiors only supported for geodesic cylindrical projections`).
- `gdal.Warp` com `width=` e `height=` arredonda dimensões — ler `RasterXSize/RasterYSize` reais do output (reshape assumindo altura calculada pode quebrar).

## Comandos

- Rodar pipeline WV1: `.venv/bin/python scripts/wv1/gerar_dataset_wv1.py --source 10m --year 2024 --out dados/wv1/mapbiomas/2024_10m` (ou `--source 30m --year 2008 --out dados/wv1/mapbiomas/2008_30m`)
- Rodar features: `.venv/bin/python scripts/wv1/gerar_features_pan.py` (saída padrão: `dados/wv1/features/pan_features.tif`)
- Editor de máscaras: `.venv/bin/python scripts/editor/guia_edicao.py`
- SPOT5: `.venv/bin/python scripts/spot5/gerar_mascara_mapbiomas_cena.py --imagem <cena>/original/<img>.tif --ano 2007 --res 30` (ou `--res 10 --ano 2024`); tiles: `scripts/spot5/tiles_cena.py cortar --imagem <cena> ...`, `... cortar-mascara --pasta <cena>/tiles --mascara <10m>/MASK_MAPBIOMAS_2024.tif` e `... juntar --pasta <cena>/tiles`
- Dataset de patches: `.venv/bin/python scripts/datasets/gerar_dataset_refinado.py --saida dados/datasets/spot5_refinado_v12 --total 3000 --cena <PAN> <mask|tiles/> <peso> ...`; **refazer idêntico** com `--receita dados/datasets/spot5_refinado_v12/receita.json --sobrescrever`; `--simular` mostra quanto cabe sem gravar; `--buffer <m>` erode as máscaras (buffer negativo) antes de selecionar/gabar patches.
- Validar alinhamento (sem re-download GEE): importar `pan_info`/`validate` de `scripts/wv1/gerar_dataset_wv1.py` e chamar `validate(info, <PAN original>, <out>/mask.tif, <out>/pan_mask_pair.tif)`.
- Ferramentas CLI disponíveis no sistema: `ogrinfo`, `gdalinfo`, `gdalwarp`.