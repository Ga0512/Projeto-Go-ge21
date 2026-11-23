# U-Net CNN no RunPod: treino, teste e mapa de cena

Este guia usa um **GPU Pod direto**, sem depender de uma GPU ou região específica. O código vem do GitHub; imagens, máscaras e rasters são transferidos separadamente. Os comandos usam o mesmo `unet_cnn/train.py`, `avaliar.py` e `inferencia_cena.py` usados no PC. Leia o docstring de cada script antes de rodar.

## 1. Defina o experimento

Preencha estes valores para cada rodada; nenhum ID de Pod ou nome de cena precisa entrar no código:

```bash
REPO_URL=https://github.com/Ga0512/Projeto-Go-ge21.git
DATASET=<nome-da-pasta-do-dataset>
RUN=<nome-do-run>
CENA=<chave-da-cena-no-receita.json>
PAN=<caminho-relativo-a-Data-do-raster-pan.tif>
POD_HOST=<ip-ou-host-do-pod>
POD_PORT=<porta-ssh>
SSH_KEY=<caminho-da-chave-privada>
```

No `receita.json` do dataset, confira tamanho do patch, ordem e número de bandas, classes, estatísticas por cena e splits. O `Data/scripts/modelos/config.py` é a fonte dos códigos de classe. Não altere o mapeamento de classes depois de treinar. Guarde o commit do código e o `receita.json` com o resultado.

## 2. Pod e ambiente

Escolha qualquer GPU disponível com VRAM suficiente para **um teste real do lote**. Reserve disco para dataset, rasters, checkpoints e mapa. Um Pod pode ser criado no console ou pela API; anote host, porta SSH e ID. Network Volume não é obrigatório: o disco do Pod recebe os dados. Um Global Volume pode guardar uma cópia persistente do dataset, mas não substitui conferir os arquivos no Pod antes do treino.

No Pod, clone um commit conhecido e prepare `Data/.venv`. A imagem precisa expor GPU, PyTorch com CUDA, GDAL com bindings Python e as dependências do projeto. Os requisitos concretos estão no `Dockerfile` e nos imports dos scripts; valide o ambiente antes de transferir muitos GB:

```bash
cd /workspace
git clone "$REPO_URL" Projeto-Go-ge21
cd Projeto-Go-ge21
git rev-parse HEAD
cd Models
../Data/.venv/bin/python -c 'import ambiente; import torch; from osgeo import gdal; print(torch.__version__, torch.cuda.is_available(), gdal.VersionInfo())'
```

`torch.cuda.is_available()` deve retornar `True`. No PC, rode Python a partir de `Models/` com `../Data/.venv/bin/python`; o Python global não serve para GDAL neste projeto.

## 3. Transferir o dataset

Execute no PC via WSL. O layout remoto deve manter `Models/` e `Data/` como irmãos:

```bash
rsync -a --partial --append-verify \
  -e "ssh -i $SSH_KEY -p $POD_PORT" \
  "/mnt/d/Projects/projeto-go/Data/dados/datasets/$DATASET/" \
  "$POD_HOST:/workspace/Projeto-Go-ge21/Data/dados/datasets/$DATASET/"
```

O `rsync` pode ser repetido após queda. Confira a quantidade de imagens/máscaras e o `receita.json` no destino. Para cena inteira, envie também a PAN original e cada raster auxiliar usado pelo modelo. Não é preciso enviar texturas prontas quando `inferencia_cena.py` as recalcula da PAN; use `--texturas` apenas se forem exatamente as texturas da receita.

## 4. Treinar do zero e avaliar

No Pod, a partir de `/workspace/Projeto-Go-ge21/Models`:

```bash
../Data/.venv/bin/python -u unet_cnn/train.py \
  --nome "$RUN" --dataset "../Data/dados/datasets/$DATASET" \
  --arquitetura unetpp --base 32 --lote <lote-testado> --lote-val <lote-val-testado> \
  --epocas <limite> --paciencia <paciencia> --reamostrar --reamostrar-k 3

../Data/.venv/bin/python -u unet_cnn/avaliar.py \
  --nome "$RUN" --dataset "../Data/dados/datasets/$DATASET" \
  --splits validacao teste --lote <lote-val-testado>
```

Sem `--inicializar`, o treino começa do zero. `--resume` retoma `ultimo.pth` do **mesmo** run. `modelo_melhor.pth` é escolhido pela validação; `ultimo.pth` é a última época e pode ser avaliado separadamente copiando-o para outro run com o nome `modelo_melhor.pth`. Compare modelos nos **mesmos patches de teste**, com IoU de cada classe e mIoU; validações com composição diferente não são comparáveis.

`--reamostrar-k 3` é a configuração usada, mas com **uma única cena** a implementação não encontra par cena × classe deficitário: faz sorteio uniforme. O log mostra isso. Não descreva esse caso como balanceamento efetivo.

Teste o lote em uma época real. Um lote que cabe em inferência pode falhar no treino por causa dos gradientes. Se ocorrer erro CUDA, reduza o lote e confira se não sobrou processo na GPU.

## 5. Inferir a cena inteira

Envie a PAN e, se o modelo tiver bandas auxiliares, os rasters correspondentes. Faça primeiro um recorte válido como piloto para conferir bandas, classes, memória, tempo e códigos do GeoTIFF. Depois, no Pod:

```bash
../Data/.venv/bin/python -u unet_cnn/inferencia_cena.py \
  --modelo "unet_cnn/runs/$RUN/modelo_melhor.pth" \
  --dataset "../Data/dados/datasets/$DATASET" \
  --imagem "../Data/$PAN" --cena "$CENA" \
  --ladrilho <lado-testado> --passo <passo-testado> --lote <lote-testado> \
  --saida "unet_cnn/runs/$RUN/cena_$CENA"
```

Acrescente `--ndvi-temporal <arquivo.tif>` somente quando o modelo foi treinado com as quatro bandas temporais de NDVI, na ordem `median`, `min`, `max`, `stdDev`. O script reprojeta as bandas por ladrilho e usa as estatísticas do `receita.json`; não crie um raster inteiro em 2,5 m só para a inferência. Para outros tipos de banda, siga os argumentos documentados no docstring do script.

Os ladrilhos `bloco_*.tif` e `progresso.txt` permitem retomar o comando com os mesmos parâmetros. A etapa final escreve `classes.tif` e `info.json`. Confira `gdalinfo` para dimensões, projeção e códigos de classe. Para visualização, depois de baixar o GeoTIFF, gere uma prévia PNG reduzida com `unet_cnn/gerar_previa.py --imagem <classes.tif> --saida <classes.png> --largura 4096`. O PNG não substitui o mapa georreferenciado.

## 6. Trazer o resultado e parar a cobrança da GPU

No PC, baixe **antes** de parar ou remover o Pod:

```bash
rsync -a --partial --append-verify \
  -e "ssh -i $SSH_KEY -p $POD_PORT" \
  "$POD_HOST:/workspace/Projeto-Go-ge21/Models/unet_cnn/runs/$RUN/" \
  "/mnt/d/Projects/projeto-go/Models/unet_cnn/runs/$RUN/"
sha256sum "/mnt/d/Projects/projeto-go/Models/unet_cnn/runs/$RUN/cena_$CENA/classes.tif"
```

Compare esse SHA256 com o remoto e confira `gdalinfo` local. Verifique também o conteúdo real do backup persistente, se existir; consultar só os metadados do volume não prova que o dataset está nele. Com o mapa local validado, rode `rp pod stop <pod-id>` e confirme o estado retornado pela API. Pare o Pod mesmo se a execução falhar, depois de salvar logs e checkpoints úteis.

## Exemplo medido: Jataí WorldView v4

`wv_jatai_v4`: 2.000 patches de 512 px, 7 bandas (PAN, GLCM, rugosidade, NDVI median/min/max/stdDev), 3 classes. U-Net++ base 32, lote 20, lote de validação 16, `k=3` uniforme por ser cena única. Treino iniciado do zero. O melhor checkpoint foi a época 5. No mesmo teste de 300 patches, mIoU: 3 bandas **0,5903**, 4 bandas **0,6115**, 7 bandas época 5 **0,6329**. Na época 5, IoU por classe: vegetação **0,7492**, não vegetação **0,9006**, água **0,2488**. O `modelo_melhor.pth` e os relatórios estão em `Models/unet_cnn/runs/unetpp_wv_v4/`.

Para esta cena, o piloto de inferência usou `--ladrilho 4096 --passo 384 --lote 16` e as quatro bandas do TIFF temporal original. Lotes 24 e 32 funcionaram, mas não aceleraram o piloto enquanto o upload ocorria. A cena tem 274.176 × 293.760 pixels; 24 GB de VRAM permitem o lote, mas o tempo total depende do número de ladrilhos com imagem válida. Meça o progresso no `progresso.txt` antes de estimar a duração final.
