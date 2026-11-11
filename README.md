# projeto-go — LULC a partir de imagem pancromática

Classificação de uso e cobertura do solo (vegetação / não vegetação / água) a
partir de **uma única banda pancromática**, no estado de Goiás. Três famílias de
modelo treinadas e comparadas no mesmo dado e sob o mesmo contrato de avaliação:
**U-Net++** (CNN), **ViT+MAE** (transformer com pré-treino auto-supervisionado)
e **XGBoost** por pixel.

O método não depende do sensor — ele depende de ter PAN e um rótulo. Dois
sensores já foram percorridos de ponta a ponta:

| sensor | resolução | patch | datasets | runs |
|---|---|---|---|---|
| SPOT5 | 2,5 m | 256 px | `spot5_refinado` … `_v18` | `unetpp_v*`, `seg_mae_v3_*`, `xgb_spot5_v*` |
| WorldView-1 | 0,486 m | 512 px | `wv_jatai_v1`, `_v2` | `unetpp_wv_b16`, `unetpp_wv_ft`, `xgb_wv`, `rf_wv_d12` |

O patch dobra no WorldView justamente para compensar a resolução 5× maior: o
que importa para o modelo é quanto **terreno** cabe na janela, não quantos
pixels. 256 px a 2,5 m e 512 px a 0,486 m são 640 m e 249 m de chão — ainda não
é equivalência, e essa diferença é um dos eixos em aberto do projeto.

O repositório tem duas metades que **precisam continuar vizinhas**:

| pasta | papel |
|---|---|
| `Data/` | geração dos dados: recorte das cenas, cálculo das bandas de textura, montagem dos datasets de patches |
| `Models/` | treino, avaliação e inferência dos modelos |

Elas não são independentes: os scripts do `Models/` importam
`Data/scripts/modelos/config.py` por caminho relativo (`../Data/...`), que é a
fonte única das classes. Por isso as duas estão no mesmo repositório — clonar só
uma deixa código que não roda.

Cada metade tem o seu próprio `CLAUDE.md` com a documentação detalhada da
arquitetura (`Data/CLAUDE.md` e `Models/CLAUDE.md`). **Leia-os antes de mexer**;
este README cobre só o que é preciso para sair do zero.

---

## O que NÃO está aqui

**Nenhuma imagem, nenhum dataset, nenhum modelo treinado.** Não é descuido:

- os mosaicos originais passam de 30 GB (`jatai_sirgas.tif` sozinho tem 2,9 GB,
  `W052S18_rugosidade.tif` tem 6,8 GB) e o GitHub recusa arquivo acima de 100 MB;
- os checkpoints vão de 23 MB (`modelo_melhor.pth`) a 64 MB (`ultimo.pth`);
- as inferências de cena inteira produzem GeoTIFFs e faixas intermediárias em GB.

O que **está** versionado:

- o registro de cada experimento — `relatorio.json` e `log.csv` dentro de
  `Models/*/runs/` — que torna os resultados auditáveis sem precisar dos pesos;
- as **receitas** dos 21 datasets já gerados, em `Data/receitas/`, que permitem
  reconstruir qualquer um deles a partir das cenas originais (ver abaixo).

Para obter as imagens originais, procure o responsável pelo projeto; elas não são
redistribuíveis por aqui.

---

## Ambiente

O projeto roda em **WSL2 (Ubuntu) sobre Windows**, com GPU NVIDIA. A GPU é
obrigatória: `ambiente.exigir_gpu()` aborta sem CUDA.

```bash
# dentro do WSL, na raiz do repositório
python3.12 -m venv Data/.venv
./Data/.venv/bin/python -m pip install -r requirements.txt
```

Três armadilhas conhecidas na instalação:

1. **`torch==2.13.0+cu130` não vem do PyPI padrão.** É preciso apontar o índice
   do PyTorch na instalação, ou o pip não encontra o wheel:
   `pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130`
2. **`GDAL==3.8.4` precisa da libgdal do sistema na mesma versão**
   (`sudo apt install libgdal-dev` no Ubuntu 24.04 entrega a 3.8.4). O pip não
   compila o binding sem ela.
3. **O venv tem que ser o do Linux.** O Python do Windows é quebrado para GDAL
   neste projeto. Todos os comandos abaixo assumem `../Data/.venv/bin/python`.

O caminho do venv é convenção do projeto (`Data/.venv`), não exigência técnica —
mas os comandos documentados nos `CLAUDE.md` todos assumem ele.

### Cache em disco

`Models/ambiente.py` redireciona `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME` e
`TMPDIR` para `Models/.cache`. Isso existe porque o vhdx do WSL mora no C: e já
encheu no meio de um treino. **Todo script novo precisa de `import ambiente`
antes de importar torch/timm/xgboost/matplotlib** — essa ordem importa.

---

## Caminho completo, do zero ao mapa

```bash
# 1. gerar um dataset de patches a partir das cenas (no WSL, a partir de Data/)
../Data/.venv/bin/python scripts/datasets/gerar_dataset_refinado.py --help

# 2. treinar (a partir de Models/) -- --dataset SEMPRE explícito
../Data/.venv/bin/python unet_cnn/train.py --nome unetpp_v18 \
    --dataset ../Data/dados/datasets/spot5_refinado_v18

# 3. avaliar por cena
../Data/.venv/bin/python unet_cnn/avaliar.py --nome unetpp_v18

# 4. inferir uma cena inteira (resumível, horas)
../Data/.venv/bin/python vit_model/inferencia_cena.py \
    --modelo unet_cnn/runs/unetpp_v18/modelo_melhor.pth \
    --imagem ../Data/dados/spot5/colinas_do_sul/original/Colinas_do_Sul_mosaico_v2.tif
```

`--dataset` é sempre obrigatório na prática: os `DATASET_PADRAO` dos três
`dados.py` apontam para datasets antigos e **nunca** para o atual.

---

## Você tem um TIF pancromático. E agora?

Este é o caminho completo para quem chega de fora com uma cena PAN (SPOT5 2,5 m
ou WorldView) e quer chegar a um mapa classificado.

### 1. O que você precisa ter em mãos

| item | obrigatório? | observação |
|---|---|---|
| GeoTIFF da PAN | sim | uma banda, qualquer CRS — a grade é reprojetada para UTM na resolução pedida |
| máscara de rótulo | para treinar | GeoTIFF com códigos MapBiomas, `0` = sem rótulo |
| — | — | para **só inferir** com um modelo já treinado, a máscara não é necessária |

As três bandas que o modelo consome **não são dados que você fornece** — elas
são derivadas da própria PAN pelo gerador:

1. **PAN** original;
2. **entropia GLCM** (janela 7, 8 níveis) sobre a PAN esticada p2–p98;
3. **rugosidade**, média de `z(LocalSTD 7×7)` e `z(Sobel)` sobre a mesma PAN
   esticada.

O esticamento p2–p98 é calculado **por cena**, em blocos sorteados na cena
inteira, e fica gravado na receita. É isso que deixa texturas de cenas com
brilho diferente comparáveis entre si — e é por isso que usar a estatística
errada na inferência degrada o resultado em silêncio.

### 2. Gerar o dataset de patches

```bash
# a partir de Data/
.venv/bin/python scripts/datasets/gerar_dataset_refinado.py \
    --saida dados/datasets/meu_dataset_v1 --total 3000 \
    --cena dados/<cena>/original/<PAN>.tif dados/<cena>/mascara_refinada.tif 100

# antes de gastar tempo, veja quanto cabe sem gravar nada:
.venv/bin/python scripts/datasets/gerar_dataset_refinado.py ... --simular
```

Cada `--cena` recebe **PAN, máscara e peso** (% do dataset). Várias cenas somam
100. A máscara pode ser um GeoTIFF único ou uma pasta `tiles/` do editor, e nesse
caso só os tiles com `mascara_refinada.tif` entram.

O split de validação e teste é **espacial**, não aleatório: blocos contíguos, e
qualquer janela que encoste em blocos de splits diferentes é descartada. Nenhum
pixel é compartilhado entre treino, validação e teste — o que torna a métrica
honesta e, de quebra, mais pessimista que um split aleatório.

### 3. Refazer um dos datasets deste projeto, idêntico

As receitas dos 21 datasets já gerados estão versionadas em `Data/receitas/`.
Cada uma guarda resolução, patch, stride, fração de validação/teste, semente,
tetos de sobreposição e as estatísticas p2–p98 de cada cena:

```bash
.venv/bin/python scripts/datasets/gerar_dataset_refinado.py \
    --receita ../Data/receitas/spot5_refinado_v18/receita.json --sobrescrever
```

Com as cenas originais nas mãos, isso reconstrói o dataset byte a byte. **Os
caminhos das cenas dentro da receita são os da máquina original** — se a sua
árvore de pastas for diferente, ajuste-os antes de rodar.

### 4. Treinar, avaliar, inferir

```bash
# a partir de Models/ -- --dataset SEMPRE explícito
../Data/.venv/bin/python unet_cnn/train.py --nome meu_modelo_v1 \
    --dataset ../Data/dados/datasets/meu_dataset_v1

../Data/.venv/bin/python unet_cnn/avaliar.py --nome meu_modelo_v1

../Data/.venv/bin/python vit_model/inferencia_cena.py \
    --modelo unet_cnn/runs/meu_modelo_v1/modelo_melhor.pth \
    --imagem ../Data/dados/<cena>/original/<PAN>.tif
```

A inferência de cena inteira é **resumível**: ela grava faixa por faixa e um
`progresso.txt`, então rodar de novo continua de onde parou. Uma cena de 2,9 Gpx
levou ~3 h numa RTX 3060 Laptop de 6 GB.

### 5. WorldView: o mesmo método, um preparo diferente

O treino, a avaliação e a inferência são **os mesmos scripts** — o que muda é
como a cena e o rótulo chegam até eles.

No SPOT5 a máscara é desenhada à mão sobre a própria imagem (o editor em
`Data/scripts/spot5/tiles_cena.py`), e é ground truth de verdade na resolução da
cena. No WorldView-1 não existe rótulo nessa resolução, então
`Data/scripts/wv1/gerar_dataset_wv1.py` monta o par de outro jeito:

1. lê o GeoTIFF WorldView-1 PAN (0,486 m) e extrai CRS, transform e footprint;
2. usa o footprint real como ROI e baixa do Google Earth Engine o **MapBiomas
   Coverage 10 m, Collection 3 beta** (ano 2024) recortado nessa ROI;
3. reamostra o MapBiomas para a grade **exata** da PAN com **nearest-neighbor**
   (nunca bilinear ou bicúbica — são códigos categóricos, interpolar inventa
   classe que não existe);
4. grava `pan.tif`, `mask.tif` e o par empilhado, valida o alinhamento e gera
   relatório.

A consequência metodológica está escrita no próprio script e vale repetir aqui:
**a máscara do WorldView é pseudo-rótulo derivado de um produto de 10 m, não
ground truth de 0,5 m.** Um erro de borda de 10 m são ~20 pixels de WorldView.
Comparar o número do WorldView com o do SPOT5 sem levar isso em conta é
comparar coisas diferentes.

Depois do par montado, o resto dos scripts de `Data/scripts/wv1/`
(`gerar_features_pan.py`, `gerar_patches_treino.py`, `gerar_blocos_gt.py`,
`gerar_mascara_manual.py`, `suavizar_mascara.py`, `agregar_pan.py`) produzem os
patches, e daí em diante o caminho é idêntico ao do SPOT5.

Aviso honesto: existe um `Models/sensores.py` com presets por sensor, mas ele
**ainda não está conectado a nenhum script**. Hoje o ajuste por sensor é feito
nos argumentos de linha de comando, não por preset.

---

## Convenções que importam

- **Run = `<modelo>_v<N>`**, onde `v<N>` é a versão do dataset: `unetpp_v18` foi
  treinado em `spot5_refinado_v18`. Os runs de WorldView fogem dessa regra e
  levam o sensor no nome (`unetpp_wv_b16`, `xgb_wv`), porque os datasets deles
  não seguem a mesma numeração.
- As métricas são calculadas duas vezes: no patch **inteiro** e no **interior**
  (margem de 15 px). Comparação entre famílias de modelo usa **sempre o
  interior**, porque as janelas de feature do XGBoost só cabem inteiras ali.
- O melhor epoch é escolhido pelo **mIoU de validação**; o teste só é avaliado no
  fim, com os pesos daquele epoch. Nada é escolhido olhando o teste.
- `ambiente.py` e `dados.py` são **cópias** em cada uma das três pastas de modelo,
  não um pacote compartilhado. Mexeu em um, replique nos outros — é o erro mais
  fácil de cometer aqui.
- Código, docstrings, argumentos e saída são **em português**.
