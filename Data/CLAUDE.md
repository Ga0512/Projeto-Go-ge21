# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Mapa das pastas: `README.md`. Detalhes da geração de dados via GEE/MapBiomas: `AGENTS.md`.

## Ambiente (CRÍTICO)

- **Sempre rodar Python via o venv do projeto**: `.venv/bin/python ...`, a partir da raiz. O venv é Linux/WSL (`.venv/bin`, python3.12, numpy 1.26.4, GDAL 3.8.4, torch cu130, xgboost 2.1). No WSL o projeto fica em `/mnt/d/Projects/projeto-go/Data` (no Windows: `D:\Projects\projeto-go\Data`, também acessível pela junction `C:\Users\gabriel.cicotoste\HDSeagate\Projects\projeto-go\Data`).
- Se o Claude Code estiver rodando no Windows, chamar pelo WSL a partir da ferramenta Bash (o PowerShell estraga as aspas aninhadas): `wsl.exe -e bash -c 'cd /mnt/d/Projects/projeto-go/Data && .venv/bin/python ...'`. Para passar um caminho `/mnt/...` como argumento a partir do Git Bash, prefixar `MSYS_NO_PATHCONV=1`. Os caminhos `/tmp` dos scripts são do WSL.
- O Python global é quebrado para GDAL: o gdal do sistema (3.8.4) foi compilado para numpy 1.x, mas o user-site tem numpy 2.x — `osgeo.gdal_array` falha com `_ARRAY_API not found`. O venv (`--system-site-packages`) resolve com numpy 1.26.4 instalado dentro dele.
- **Não instalar nada no Python global** (PEP 668). Instalar no venv com **`.venv/bin/python -m pip install ...`**. O `.venv/bin/pip` (e os demais scripts de console do venv) está quebrado: o shebang ainda aponta para o nome antigo da pasta (`.../GOIAS_EXEMPLO_CLASSIFICACAO_USO_SOLO/.venv`).
- GEE autenticado no projeto `beaming-botany-373619` (`ee.Initialize(project=...)`).
- Sem git. Sem testes automatizados — a validação é feita pelos checks embutidos nos scripts (`validate()` em `gerar_dataset_wv1.py`) e pelos PNGs/relatórios gerados.
- CLI GDAL disponível: `ogrinfo`, `gdalinfo`, `gdalwarp`.
- A pasta irmã `../Models` está fora do escopo deste projeto: não mexer.

## Organização

- **`scripts/caminhos.py` é a fonte única de todos os caminhos.** Todo script faz `sys.path.insert(0, <scripts/>)` + `import caminhos` e usa as constantes de lá (`caminhos.WV1_FEATURES`, `caminhos.WV1_TREINO_2007`, `caminhos.SPOT5_CENA_013_TILES`...). Nunca hardcodar caminho em script; se uma pasta mudar, mudar só `caminhos.py`. `caminhos.pasta_cena(img)` devolve a pasta da cena (a mãe de `original/`).
- Código por assunto em `scripts/{wv1,spot5,editor,modelos,visualizacao,testes}/`. Dados por sensor/cena em `dados/{wv1,spot5/<cena>}/`, cada um com `original/` (entrada bruta, não editar) e as pastas derivadas ao lado. Saídas de treino em `modelos/<nome>/`. `experimentos/` é só arquivo de testes exploratórios.
- Symlinks e metadados guardam caminhos **relativos** (`dados/wv1/patches/treino_manual/img -> ../treino_2007/img`; `tiles.json` e os `.vrt` relativos à pasta `tiles/`). Um caminho absoluto quebrou quando a pasta do projeto foi renomeada.
- `.reorganizacao/` registra a reorganização de 2026-09-11 (`movimentos.tsv`, `desfazer.sh`). `_para_apagar/` guarda cópias byte-idênticas que o usuário ainda não decidiu apagar; nada lê de lá.

## Arquitetura do pipeline

Três estágios encadeados, cada um consumindo o produto do anterior:

**1. Dados (GEE + GDAL)** — `scripts/wv1/gerar_dataset_wv1.py`, `gerar_features_pan.py`
Extrai footprint da PAN → baixa MapBiomas pelo GEE na ROI → reprojeta para a grade exata da PAN, em `dados/wv1/mapbiomas/<ano>_<res>/` (`mask.tif`, `pan_mask_pair.tif`, relatório, PNGs). O script também grava um `pan.tif` que é cópia idêntica da PAN original (as cópias antigas foram para `_para_apagar/`). Gera `dados/wv1/features/pan_features.tif` (4 bandas Float32, ~6.2 GB, BIGTIFF): `PAN, LocalSTD(7), GLCM_Entropy(7,L=8), Gradient(sobel)`.

**2. Labels e patches** — `scripts/wv1/`: `gerar_mascara_manual.py`, `suavizar_mascara.py`, `gerar_patches_treino.py`, `gerar_patches_manual.py`
Existem **duas fontes de label paralelas**, treinadas e comparadas com os **mesmos splits e as mesmas janelas (x,y)**:
- `dados/wv1/patches/treino_2007/` — pseudo-labels MapBiomas 2007 30m, suavizados por watershed guiado pelo gradiente da PAN (`dados/wv1/mascaras/mask_2007_suavizada.tif`).
- `dados/wv1/patches/treino_manual/` — labels da classificação manual (`dados/wv1/original/CLASSIFICACAO_WV1.shp` rasterizado em `mascaras/mask_manual.tif`). Reusa `img/` do treino_2007 por symlink e copia `splits/`; só a máscara muda. Isso é intencional: mantém a comparação justa entre as duas fontes.

Patches são `512x512`, salvos como `img/%06d.npz` (float16, HxWx4) e `mask/%06d.npz` (uint8, códigos MapBiomas, 0=nodata; chave `arr_0`), mais `manifest.csv` (`idx,x,y,aug,valid_frac`) e `splits/{treino,validacao,teste}.txt` (936/200/200).

**3. Modelos** — `scripts/modelos/`
`config.py` é a fonte única de verdade compartilhada por todos os modelos: `CLASS_MAP` (código MapBiomas → índice), `IGNORE=255`, `REPR_CODIGO` (índice → código representativo, usado ao escrever o GeoTIFF de saída), `MEAN`/`STD` por banda (stats reais de 349.6M pixels do treino_2007), frações de split e os defaults `DEFAULT_TREINO`/`DEFAULT_SAIDA`/`DEFAULT_IMAGEM` (vindos de `caminhos`). Alterar `CLASS_MAP` invalida os modelos já treinados.
- `dataset_lulc.py` — `LULCDataset` (lê os `.npz`, normaliza com `MEAN/STD`, mapeia códigos → índices, nodata → `IGNORE`, aug flip/rot90 em img+mask juntas) e os helpers de split (`ler_idxs`, `ler_splits`, `salvar_splits`, `split_estratificado_agua`) usados pelos dois modelos.
- `xgb_lulc.py` — XGBoost por pixel, 10 features (4 bandas + 4 vizinhos PAN + média/std 7x7), device CUDA por padrão. `--avaliar` reavalia o `modelo.ubj` de `--saida` nos splits gravados.
- `train.py` + `modelo_unet.py` — U-Net/UNet++ (`unetleve`, `unetpp`, `unetpp_r18`, `unetpp_r34`), deep supervision com pesos `[1, .5, .25, .125]`, loss `0.5*Focal + 0.5*Dice`, AMP, treino em crops aleatórios (`--crops 256`, `--crops-pp 4`), early stop por `val_iou` (`--patience`).
- Splits: sem `--usar-splits` o split é refeito em memória a partir do `manifest.csv` (seed `config.SEED`). **`xgb_lulc.py --split-estratificado` sobrescreve `splits/` em disco** — isso quebra a comparação justa com o `treino_manual` (que tem cópia dos splits do `treino_2007`).
- Ambos os treinos compartilham o schema de métricas/relatório (`train.py` importa `metricas_por_classe`/`relatorio` de `xgb_lulc`), gravam `relatorio.json` com os args completos, e suportam `--resume` a partir de checkpoint.
- Ambos fazem inferência tile a tile com overlap (`--tile`/`--overlay`) sobre a cena inteira, escrevendo um GeoTIFF na grade da PAN com os códigos de `REPR_CODIGO`.
- Saídas em `modelos/<nome>/` (`modelo.ubj`|`*.pth`, `checkpoint.*`, `relatorio.json`, `mascara_*.tif`, `visualizacao/`). Hoje vazia: nenhum modelo treinado está no disco.

## SPOT5 e editor de máscaras

Cada cena SPOT5 é uma pasta em `dados/spot5/`:
- `recorte_2007-09-10/` — recorte pequeno (2758x1465, EPSG:4326, pixel **~5 m**, não 2,5) + classificação manual `CLASSIFICACAO_SPOT5.shp` (mesma extensão do recorte). MapBiomas em `mapbiomas_2007_30m/` e `mapbiomas_2024_10m/` (gerados por `scripts/spot5/legado/`; o editor usa a banda 2 de `PAR_SPOT5_PAN_MASK_2007.tif`).
- `cena_013-014_2007-06-14/` — cena grande (29488x29067 a 2,5 m, EPSG:32722, NoData=0).
- `colinas_do_sul/` — `Colinas_do_Sul_mosaico_v2.tif`: mosaico recortado pelo limite do município (29068x32624 a 2,5 m, EPSG:32722, sem data no arquivo, sem NoData declarado — 69% é fundo 0, que os scripts tratam como sem imagem). MapBiomas **2007** (ano assumido, igual às outras SPOT5) em `mapbiomas_2007_30m/`, 95 tiles em `tiles/`, cena "SPOT5 Colinas do Sul (tiles)" no editor. **Sobrepõe a 013-014**: 1139 km² em comum (62% da Colinas), inclusive 71% do tile rotulado `r06_c08`. Não é recorte da 013-014 (na área comum: brilho diferente, correlação 0,83, deslocamento ~5 m) — outra data ou processamento. Tem códigos 24 (urbano) e 29 (afloramento rochoso), fora do `config.CLASS_MAP`. Ao montar datasets com as duas cenas, cuidar para a mesma área do terreno não cair em splits diferentes.

- `jatai/` — `W052S18.tif`: sudoeste de GO (44327x49234 a 2,5 m, EPSG:32722, sem data, sem NoData declarado — 7% é fundo 0). **~12.600 km² com imagem, a maior cena** (7x a Colinas), 506 tiles, cena "SPOT5 Jataí (tiles)" no editor. Não se sobrepõe às outras. MapBiomas 2007 validado; composição bem diferente: **36% soja, 24% pastagem, 10% floresta, 8% savana, só 0,13% água** — é o que falta de "não vegetação" nas outras cenas. **Não tem `mascara_base.tif`**: a regra de água foi abortada de propósito (ver abaixo).

Cena nova: `scripts/spot5/gerar_mascara_mapbiomas_cena.py --imagem dados/spot5/<cena>/original/<img>.tif --ano 2007` baixa o MapBiomas numa grade 30 m alinhada à origem da imagem (fator inteiro 12 → reamostragem é repetição exata) e grava `<cena>/mapbiomas_<ano>_30m/`; `tiles_cena.py cortar` gera `<cena>/tiles/rRR_cCC/{imagem,mascara_mapbiomas}.tif` (2048 px; tiles só-nodata pulados) + `.vrt` da cena + `tiles.json`; o editor grava `rRR_cCC/mascara_refinada.tif`; `tiles_cena.py juntar --pasta <cena>/tiles` monta a máscara final (`MASCARA_REFINADA_CENA.tif`). Tiles > ~2048 px deixam a revetorização do editor lenta demais (3072 px → ~70 s por edição).

**Água pela PAN (base de edição)** — `scripts/spot5/agua_pan.py --tiles <cena>/tiles [--simular] [--png]`: grava `<tile>/mascara_base.tif` = MapBiomas com a **água ampliada pelo que a PAN mostra** (o MapBiomas erra a borda de 30 m e perde corpos pequenos). **Só acrescenta água, nunca tira.** Regra calibrada na água que já existe na cena (água erodida `--borda 8` px = interior): **escuro** (PAN ≤ quantil `--quantil-brilho`, default 0.95) **E liso** (desvio-padrão local janela 7 ≤ quantil `--quantil-suavidade`) + componente conexo ≥ `--min-px` (200) + opcional `--vizinhanca N` px da água existente (0 = cena toda, acha corpo novo). **O critério de suavidade é essencial**: só com brilho, a sombra da encosta na margem entra como água (foi o que aconteceu no experimento `scripts/testes/testar_agua_pan.py`, que não tem esse critério). Calibra com a `mascara_refinada.tif` onde existir, senão com o MapBiomas; a saída sai sempre do MapBiomas (não toca no trabalho manual). Resultado: +3% de água na Colinas e na 013-014; falso positivo típico = mancha pequena em fundo de vale sombreado. **A regra só vale onde há corpo d'água grande e escuro de verdade.** Na Jataí ela foi abortada pela trava: lá o MapBiomas marca rios mais estreitos que o pixel de 30 m, a "água" dele tem brilho de terra (mediana 79 contra 103 da cena, p95 = 138), o limiar sai em 128 e a regra pintaria 77% da cena. Duas travas cuidam disso: uma **prévia em 8 áreas sorteadas** antes de processar e o teto `--max-novo` (0,25 da área rotulada); passando de qualquer um, o script aborta sem gravar e a base segue sendo o MapBiomas. PNGs de conferência (`--png`) em `experimentos/agua_pan/<cena>/base_<tile>.png`.
- `--mapear "39=15,41=15,62=15,20=15"` remapeia códigos ao gravar a base (soja e lavouras viram pastagem). **Não muda a classe do modelo** (39/41/62 já viravam 19 = não vegetação nos datasets); serve para a edição manual não ter dezenas de retalhos minúsculos que o balde não pega. Usado na Jataí (38% de soja): as 506 bases saíram sem nenhum código de lavoura, e nada além de lavoura→15 mudou. Quando a regra de água é recusada mas há `--mapear`, o script segue só com o remapeamento.
- **O editor prefere `mascara_refinada.tif` > `mascara_base.tif` > `mascara_mapbiomas.tif`** (`_vrt_mascara_atual`), então basta rodar o script e abrir a cena. Só vale para cenas em tiles; o recorte não-tilado continua saindo do MapBiomas.

**Texturas da cena inteira** — `scripts/spot5/gerar_features_cena.py --pan <PAN.tif>` (ou `--todos`, as 3 cenas UTM): grava **dois GeoTIFF de 1 banda ao lado do PAN**, em `original/`: `<pan>_glcm.tif` (GLCM entropia) e `<pan>_rugosidade.tif` (média de z(LocalSTD) e z(Gradiente), com **segundo z-score**). Float32, grade NATIVA do PAN, DEFLATE/PREDICTOR=3, TILED, BIGTIFF, nodata −9999 onde o PAN é fundo. São as **bandas 2 e 3 do dataset de patches**: fórmulas e estatísticas (p2/p98 do esticamento, média/desvio de cada banda) saem do `receita.json` do `spot5_refinado_v15` quando a cena está lá e não foi reamostrada — verificado contra os patches do v15: GLCM bate em 1e−6 e rugosidade em 1e−3 (arredondamento GPU×CPU, em valores de −3 a +9); cobertura varrida em 60 linhas por arquivo, zero buraco.
- **A rugosidade leva DOIS z-scores** (o do `std`/`grad` e mais um da média dos dois, `media/desvio["rugosidade"]` na receita). Esquecer o segundo foi o bug da primeira versão: dava diferença de até 0,8 contra os patches (`sd_rug` ≈ 0,93).
- **A banda 1 (PAN) não é gravada** — na inferência ela sai do PAN original com `(pan_cru_em_DN − mu_pan) / sd_pan`. **É a PAN CRUA, não a esticada**: o esticamento p2–p98 alimenta só o cálculo das texturas (`gerar_dataset_refinado.py`, `texturas()` devolve `pan` bruto e `bandas()` faz `z(pan)`). Os valores de `p2`, `p98`, `mu_pan`, `sd_pan` e das texturas ficam nos **metadados dos dois GeoTIFF** (`gdalinfo`).
- Roda na **GPU por padrão** (torch; `--cpu` volta para o scipy de `gerar_features_pan.py`). Medido na 013-014 com bloco de 1024 linhas: GPU 8,3 s por bloco contra 139 s na CPU (17×), 1,9 GB de VRAM.
- Processa em blocos de linhas com margem refletida de 3 px; `--block` (0 = automático pela largura, mirando ~2,5 GB de VRAM). O gargalo é **RAM do sistema**, não VRAM: o WSL tem 7 GB e uma rodada `--todos` já foi morta por falta de memória — usar `--block 512`. O cache de blocos do GDAL está travado em 256 MB.
- Cada bloco faz **flush do dataset** (fecha o diretório do TIFF) e anota a linha em `<pan>_features_progresso.json`, então uma rodada interrompida fica legível até onde chegou e `--retomar` continua de lá. Sem o flush do dataset (só da banda) um kill perde o arquivo inteiro: foi o que aconteceu na Jataí — 9,4 GB gravados e 100% ilegível.
- Tempo/tamanho (GPU, `--block 512`): 013-014 (29488x29067) 285 s, 1,60 + 2,02 GB; Colinas (29068x32624) 215 s, 0,95 + 1,03 GB; Jatai (44327x49234) 794 s, 5,30 + 6,79 GB.

`scripts/editor/guia_edicao.py` — GUI PySide6. As cenas editáveis ficam na lista `CENAS`. A máscara raster é a fonte da verdade e os polígonos são revetorizados a cada edição com `shapely.coverage_simplify` (partição sem fresta/sobreposição; requer `shapely>=2.1`). Correção automática = sieve + watershed no gradiente da PAN + suavização de cobertura.
- Cena não-tilada: a máscara de trabalho é um único `mascara_refinada.tif` da cena inteira (ex.: `dados/spot5/recorte_2007-09-10/mascara_refinada.tif`, nasce no 1º salvar como cópia da MapBiomas). Cada salvar grava só o recorte aberto, e `<refinada>.recortes.json` lista os recortes já editados (esses não recebem correção automática ao reabrir). Cenas até 16 Mpx abrem inteiras; acima disso, em recortes de 3072 px.
- **Os `mascara_refinada.tif` são edição manual do usuário: não sobrescrever.**
- `scripts/testes/teste_editor_headless.py` está desatualizado (modo antigo de recortes) e tem um `sys.exit` no topo: no modo em tiles o `salvar()` dele gravaria num `mascara_refinada.tif` real. Reescrever antes de usar.

## Datasets de patches (SPOT5 refinado)

`scripts/datasets/gerar_dataset_refinado.py` monta um dataset de patches a partir de pares **PAN + máscara total** (0 = sem rótulo) com **peso por cena** (% do dataset); a máscara também pode ser uma pasta `tiles/` (só os tiles com `mascara_refinada.tif` entram). Saída em `dados/datasets/<nome>/`: `images/*.tif` (3 bandas Float32: PAN, GLCM entropia, rugosidade = média de z(LocalSTD) e z(Gradiente)), `masks/*.tif` (Byte, códigos MapBiomas, 39/41/62→19), `manifest.csv`, `splits/`, `receita.json`, `visualizacao/`.
- Grade comum UTM a 2,5 m: cena já nessa grade não é reamostrada; o recorte de ~5 m vai com cúbica (PAN) e nearest (máscara).
- Normalização por cena: as texturas usam a PAN esticada p2–p98 da cena (as cenas têm brilho bem diferente) e as 3 bandas saem em z-score com estatísticas de 64 blocos sorteados na cena inteira (não só na área rotulada — por isso a média no tile 013-014, 42% água, fica em −0,5).
- Posições candidatas numa grade fina (`--stride 32`), não só lado a lado. Seleção por sorteio com teto de sobreposição: um patch só entra se não divide mais que o teto da sua área com nenhum outro da cena. `--max-sobreposicao` (0.7) é o teto máximo; cada cena usa o **menor teto** (passos de 5%) que entrega a sua cota — cena grande repete pouco, só a cena pequena chega perto do teto.
- `--total N` (0 = o máximo que cabe) com cota por cena = N × peso, repartida pelos splits; se uma cena não comporta a sua cota, todas encolhem juntas (a proporção manda) e sai AVISO. `--simular [TETOS...]` mostra quanto cabe e a repetição sem gravar nada.
- Split espacial: blocos de 512 px em serpentina; validação e teste são trechos contíguos, e o ponto de partida é o que deixa a distribuição de classes de validação/teste mais parecida com a da cena (com partida sorteada, o teste do tile 013-014 chegou a ficar com 2,6% de água contra 31% no treino). Janelas que tocam blocos de splits diferentes são descartadas (nenhum pixel compartilhado entre splits).
- `--receita <dataset>/receita.json --sobrescrever` refaz o dataset byte a byte idêntico (verificado); a receita guarda md5 das máscaras e a lista de tiles usados e avisa se mudaram.
- `patch`/`bloco` devem ser múltiplos de `stride`. Limite físico atual: o tile refinado da 013-014 tem 2048 px, onde cabem só 64 patches de 256 sem sobreposição; 250 patches dele implicam cada pixel em ~4 patches, qualquer que seja o método. Mais tiles refinados = menos repetição.
- Datasets: `spot5_refinado` (v1: recorte + 013-014 r06_c08, 500, 3 classes), `spot5_refinado_v2` (+ Colinas r06_c05, 750, 3 classes), `spot5_refinado_v3` (+ Colinas r04_c02, 4 classes, 250 por cena = 750) , `spot5_refinado_v4` (Colinas 2 tiles, 1000) e **`spot5_refinado_v5`** (atual: + Colinas r00_c06; pesos 1/1/3, `--total 1250` → 250/250/750, 875/189/186; Colinas = 216 r00_c06 + 178 r04_c02 + 356 r06_c05; repetição Colinas 5,6×). **Regra do usuário: 250 patches por área refinada** (o recorte inteiro conta 1, cada tile refinado conta 1) → peso da cena = nº de tiles refinados; v4 = `--total 1000` pesos recorte 1 / 013-014 1 / Colinas 2 → 250/250/500, 700/151/149. A cota é por cena, então dentro da Colinas os tiles não saem 250/250: 202 do r04_c02 (43% dele é fundo sem rótulo) + 298 do r06_c05. Repetição: recorte 1,3× por pixel, 013-014 4,2×, Colinas 5,1×. Ao refinar tile novo, aumentar o peso da cena e o `--total` em 250. As estatísticas por cena não mudam entre versões enquanto a ordem das cenas na lista for a mesma (mesmo sorteio de blocos).
**Split calibrado pelo blockCV** (Valavi et al. 2019, MEE 10.1111/2041-210X.13107): o bloco do split e a zona morta devem ser >= o alcance da autocorrelação espacial das bandas, senão treino e teste são vizinhos parecidos e a métrica sai otimista.
- `scripts/testes/alcance_autocorrelacao.py` mede esse alcance (autocorrelação 2D por FFT, média radial) nas janelas rotuladas. **Medido (2026-09-24): rho<0,1 entre 130 e 990 px; range de 95% do variograma entre 130 e 890 px (até ~2,2 km).** O bloco antigo de 512 px (1,28 km) era MENOR que isso.
- `--zona-morta <m>` (default 0): janela perto de bloco de outro grupo é descartada. `--dobras k` (default 0): k-fold espacial no lugar de treino/validação/teste (grava `splits/dobra_i.txt`). **k-fold não é viável hoje**: a 013-014 tem um tile só (26 km²), que com blocos de 1024 px dá 4 blocos — em 4 ou 5 dobras sempre sobra dobra vazia e o balanceamento zera o dataset. Só depois de mais tiles espalhados.
- **`spot5_refinado_v9` (atual)**: `--bloco 1024 --zona-morta 320` + 6 tiles Colinas, `--total 2000` pesos 1/1/6 → 1995 patches (1395/301/299). Efeito medido: no v8 havia patch de teste ENCOSTADO no treino (mínimo 0 m, mediana 160–880 m); no v9 o mínimo é 640–720 m e a mediana 1,3–1,8 km. Ainda abaixo do alcance máximo medido (2,2 km) — com tiles de 26 km² não dá para ter bloco e zona morta de 2 km sem ficar sem dados.

- Colinas e 013-014 cobrem o mesmo terreno: hoje os tiles rotulados das duas não se sobrepõem (verificado: 0 pares de patches de cenas diferentes no mesmo lugar). Se um dia forem refinados tiles sobre o mesmo terreno nas duas cenas, o split espacial (que é por cena) pode pôr o mesmo lugar em splits diferentes — checar antes.

**Patches sem rótulo** — `scripts/datasets/gerar_patches_sem_rotulo.py --pan X --saida Y [--excluir <máscara|tiles/>] [--estatisticas <receita rotulada>]`: só `images/` (mesmas 3 bandas e grade), janelas lado a lado por padrão (`--stride` 0 = patch), só com 100% de imagem válida. `--excluir` (pode repetir) tira as janelas que tocam a área rotulada (não repete nem vaza validação/teste); máscara de outra cena/grade é levada para a grade da PAN com nearest — por isso o sem rótulo da 013-014 exclui também os tiles rotulados da Colinas (mesmo terreno, outra imagem). `--estatisticas` copia as estatísticas da cena da receita rotulada, e as bandas saem bit a bit iguais às do dataset rotulado na mesma posição (verificado). **Salvar sempre numa pasta própria**, nunca dentro de um dataset rotulado. Atual: `dados/datasets/spot5_013-014_sem_rotulo/` (cena 013-014 inteira menos o tile r06_c08 e o terreno dos tiles rotulados da Colinas; estatísticas copiadas do `spot5_refinado_v3`, idênticas às do v4/v5; o r00_c06 da Colinas fica fora da imagem 013-014, então não afeta o sem rótulo). `Cena` e `fracoes_celulas` do `gerar_dataset_refinado.py` aceitam cena sem máscara (`mascara: None`, `validade_total=True`).

Outros: `scripts/testes/testar_slic.py` (snapping de bordas por SLIC → `experimentos/slic_test/`), `scripts/visualizacao/visualizar_30m.py` (PAN/máscara agregadas a 30m → `experimentos/sandbox_30m/`).

## Classes

**4 classes finais** (definidas pelo usuário em 2026-09-15; agrupamento em `config.CLASS_MAP`, índice de água em `config.IDX_AGUA`):
`0` Vegetação densa (3 Floresta; 5, 6, 49) · `1` Vegetação não densa (4 Savânica; 11, 12, 50) · `2` Não vegetação (15 Pastagem = **tudo que não é vegetação natural**: lavouras, silvicultura, mosaico, urbano, areia, afloramento, mineração...) · `3` Água (33; 31). `REPR_CODIGO = [3, 4, 15, 33]`.
- Nas máscaras refinadas à mão o usuário usa só os códigos **3, 4, 15 e 33**; os demais são restos do MapBiomas não editados e caem no grupo pela mesma lógica. Todo código da legenda oficial está no mapa.
- Editor e `visualizar_predicao.py` leem as classes do `config` (sem cópias locais). Os datasets guardam códigos MapBiomas, então trocar o agrupamento não exige refazer máscaras — mas `manifest.csv` (colunas `frac_<classe>`) e o balanceamento de classes dos splits usam o `CLASS_MAP` do momento da geração (v1/v2 foram gerados com 3 classes).
- WV1: a classificação manual (`CLASSIFICACAO_WV1.shp`) não separa vegetação densa de não densa — `gerar_mascara_manual.py` põe "Outros" em 3 (hoje = vegetação densa); rever antes de treinar o WV1 com 4 classes.
- `suavizar_mascara.py` e os datasets unificam antes as lavouras específicas (39 Soja, 41 Outras Temporárias, 62 Algodão) em 19.

## Regras que não podem ser violadas

- Máscara é **categórica inteira** (GDT_Byte) e sempre reamostrada com **nearest-neighbour**. Nunca bilinear/bicubic.
- PAN e MASK devem ter **mesmo CRS, mesmo affine transform, mesmas dimensões** (a grade da PAN manda: 22400x18336, EPSG:32722).
- A máscara MapBiomas é **pseudo-label derivado de produto coarse** (10m/30m para uma imagem de 0,5m) — nunca tratar como ground truth.
- Códigos MapBiomas são os oficiais da legenda; não inventar códigos nem cores (ver `LEGEND` nos scripts).
- GeoTIFF de features: Float32, TILED, COMPRESS=DEFLATE, PREDICTOR=2, **BIGTIFF=YES** (sem isso, >4 GB falha em `WriteEncodedTile`).

## Gotchas GDAL

- Ao converter cantos UTM → lon/lat para o GEE, forçar `dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)` no EPSG:4326, senão vem (lat, lon) e o GEE interpreta errado.
- Comparar CRS com `osr.SpatialReference.IsSame()`, nunca por string WKT.
- Footprint para o GEE deve ser `ee.Geometry.Polygon` em lon/lat (geodésico); retângulo em UTM falha.
- `gdal.Warp` com `width=`/`height=` arredonda — ler `RasterXSize/RasterYSize` reais do output antes de dar reshape.
- Não encadear `gdal.Open(p).GetRasterBand(1)`: o dataset é coletado e a banda vira ponteiro inválido (`TypeError ... GDALRasterBandShadow`). Guardar o dataset numa variável.

## Comandos

```bash
# 1. dados WV1
.venv/bin/python scripts/wv1/gerar_dataset_wv1.py --source 30m --year 2008 --out dados/wv1/mapbiomas/2008_30m
.venv/bin/python scripts/wv1/gerar_features_pan.py            # -> dados/wv1/features/pan_features.tif

# 2. labels + patches
.venv/bin/python scripts/wv1/gerar_mascara_manual.py
.venv/bin/python scripts/wv1/suavizar_mascara.py --mask dados/wv1/mapbiomas/2007_30m/mask.tif \
    --out dados/wv1/mascaras/mask_2007_suavizada.tif --alias 2007_30m
.venv/bin/python scripts/wv1/gerar_patches_treino.py --mask dados/wv1/mascaras/mask_2007_suavizada.tif \
    --out dados/wv1/patches/treino_2007 --aug full --stride 256
.venv/bin/python scripts/wv1/gerar_patches_manual.py          # reusa janelas/splits do treino_2007

# 3. treino (--usar-splits p/ comparação justa entre fontes de label)
.venv/bin/python scripts/modelos/xgb_lulc.py --treino dados/wv1/patches/treino_manual \
    --saida modelos/xgb_manual --usar-splits --max-amostras 6000000
.venv/bin/python scripts/modelos/train.py --treino dados/wv1/patches/treino_manual \
    --saida modelos/unet_manual --usar-splits --arquitetura unetpp_r34
# retomar treino interrompido: acrescentar --resume

# inferência na cena inteira (--imagem default = dados/wv1/features/pan_features.tif)
.venv/bin/python scripts/modelos/xgb_lulc.py --inferir modelos/xgb_manual/modelo.ubj \
    --saida modelos/xgb_manual/mascara_xgb.tif
.venv/bin/python scripts/modelos/train.py --inferir modelos/unet_manual/unet_best.pth \
    --saida modelos/unet_manual/mascara_unet.tif

# SPOT5 + editor
.venv/bin/python scripts/editor/guia_edicao.py
.venv/bin/python scripts/spot5/tiles_cena.py juntar --pasta dados/spot5/cena_013-014_2007-06-14/tiles
# texturas (bandas 2 e 3) da cena inteira, ao lado do PAN, na GPU
.venv/bin/python scripts/spot5/gerar_features_cena.py --todos --block 512
.venv/bin/python scripts/spot5/gerar_features_cena.py --pan dados/spot5/jatai/original/W052S18.tif --retomar

# dataset de patches SPOT5 refinado (PAN + máscara + peso por cena) / refazer idêntico
.venv/bin/python scripts/datasets/gerar_dataset_refinado.py --saida dados/datasets/spot5_refinado_v5 --total 1250 \
    --cena dados/spot5/recorte_2007-09-10/original/IMAGEM_SPOT5_10092007.tif dados/spot5/recorte_2007-09-10/mascara_refinada.tif 1 \
    --cena dados/spot5/cena_013-014_2007-06-14/original/013-014_S5_708-378-0_2007-06-14_modificado.tif dados/spot5/cena_013-014_2007-06-14/tiles 1 \
    --cena dados/spot5/colinas_do_sul/original/Colinas_do_Sul_mosaico_v2.tif dados/spot5/colinas_do_sul/tiles 3
.venv/bin/python scripts/datasets/gerar_dataset_refinado.py --receita dados/datasets/spot5_refinado_v5/receita.json --sobrescrever
# patches SEM rótulo da cena inteira (pasta separada), mesma escala do dataset rotulado
.venv/bin/python scripts/datasets/gerar_patches_sem_rotulo.py --saida dados/datasets/spot5_013-014_sem_rotulo \
    --pan dados/spot5/cena_013-014_2007-06-14/original/013-014_S5_708-378-0_2007-06-14_modificado.tif \
    --excluir dados/spot5/cena_013-014_2007-06-14/tiles --excluir dados/spot5/colinas_do_sul/tiles \
    --estatisticas dados/datasets/spot5_refinado_v5/receita.json

# visualização e validação
.venv/bin/python scripts/visualizacao/visualizar_predicao.py --modelo-dir modelos/xgb_manual
# alinhamento sem re-download do GEE: importar pan_info/validate de scripts/wv1/gerar_dataset_wv1.py e chamar
#   validate(info, caminhos.WV1_PAN, <out>/mask.tif, <out>/pan_mask_pair.tif)
```
