# Treinar num pod com GPU

O código vai na imagem, o dado vai no volume. O layout do repositório é
recriado dentro do container, então os caminhos relativos dos scripts
(`../Data/dados/...` a partir de `Models/`) continuam valendo sem alteração.

## 1. Construir e publicar a imagem

```bash
# na raiz do repositório
docker build -t ghcr.io/ga0512/projeto-go:$(git rev-parse --short HEAD) .
docker push  ghcr.io/ga0512/projeto-go:$(git rev-parse --short HEAD)
```

Usar o SHA do commit como tag não é capricho: é o que liga um resultado no
volume à versão exata do código que o produziu.

## 2. Encher o volume (uma vez)

O volume precisa de três pastas. Só o dataset é obrigatório para treinar:

```
/vol/dados/datasets/spot5_refinado_v20/     ~5 GB   obrigatório p/ treinar
/vol/dados/spot5/<cena>/original/*.tif      ~3 GB   só p/ inferência de cena
/vol/runs/                                  vazio   saída do treino
/vol/inferencias/                           vazio   saída dos mapas
```

Enviar por `runpodctl send`, `rsync` sobre SSH, ou o que o provedor oferecer.
São ~5 GB para treinar e ~8 GB se for fazer inferência de cena inteira. **Não
subir os originais inteiros**: passam de 30 GB e não são usados no treino.

## 3. Treinar

```bash
docker run --rm --gpus all \
    -v /runpod-volume:/vol \
    -e IMAGEM_SHA=$(git rev-parse --short HEAD) \
    ghcr.io/ga0512/projeto-go:<sha> \
    python3 -u unet_cnn/train.py \
        --nome unetpp_v23 \
        --dataset ../Data/dados/datasets/spot5_refinado_v20 \
        --lote 64 --lote-val 32 --epocas 12 --lr 3e-4 --seed 42
```

Os resultados aparecem em `/vol/runs/unetpp_v23/`.

### O que mudar numa GPU grande

| | 6 GB (local) | 40 GB (A100) |
|---|---|---|
| `--lote` | 16 | 48–64 |
| `--lote-val` | 8 | 32 |
| `--patch` (no dataset) | 256 | **512** |

O ganho que importa é o **patch 512**: a 2,5 m, 256 px enxergam 640 m de chão e
512 px enxergam 1.280 m — um talhão inteiro em vez de um pedaço. Lote maior e
mais velocidade são consequência, não objetivo.

**Mais épocas não ajuda.** Medido: todas as rodadas saturam por volta da 6ª a
10ª época, e da 12ª em diante a perda de treino cai com a validação parada.

## 4. Inferência de cena inteira

```bash
docker run --rm --gpus all -v /runpod-volume:/vol \
    ghcr.io/ga0512/projeto-go:<sha> \
    python3 -u vit_model/inferencia_cena.py \
        --modelo unet_cnn/runs/unetpp_v23/modelo_melhor.pth \
        --imagem ../Data/dados/spot5/colinas_do_sul/original/colinasdosul_sirgas_utm22s.tif \
        --cena colinas_do_sul --lote 32
```

## Observações

- A imagem parte do GDAL oficial do OSGeo (3.8.4) e instala o torch por cima. A
  wheel do torch traz as bibliotecas CUDA, então a máquina hospedeira precisa só
  do driver NVIDIA e do container toolkit.
- O `ambiente.py` manda os caches para `Models/.cache`, que no container fica na
  camada escrevível e some com o pod. É cache, não faz falta.
- Sem `/vol` montado o container sobe mesmo assim, sem dado e sem persistência —
  serve para testar se a imagem está boa (`... python3 -c "import torch"`).

---

# RunPod e Vast.ai na prática

A imagem é a mesma nas duas. O que muda é **como o dado chega e se ele
sobrevive ao fim do pod**.

| | RunPod | Vast.ai |
|---|---|---|
| armazenamento | Network Volume: persiste e reconecta em qualquer pod | disco da instância: morre com ela |
| transferência | `runpodctl send` / `receive` (ponto a ponto) | `scp` / `rsync` por SSH, ou baixar de um bucket |
| indicado para | iterar várias rodadas sobre o mesmo dado | rodada única, sensível a preço |

Tamanhos: `spot5_refinado_v20` 3,5 GB · `_v21` ~3,4 GB · `_v22` ~2,3 GB.
Para inferência de cena somar `colinasdosul_sirgas_utm22s.tif` (252 MB) e
`jatai_sirgas.tif` (2,9 GB).

## RunPod (recomendado)

**1. Criar o Network Volume** (50 GB basta) na mesma região onde você vai pegar
a GPU — volume e pod precisam estar na mesma região.

**2. Subir um pod CPU barato** com o volume montado, só para receber o dado.

**3. Enviar o dataset.** Na sua máquina:

```bash
cd Data/dados/datasets
tar czf - spot5_refinado_v20 | runpodctl send -
```

`runpodctl` imprime um código de uma linha; no pod, rodar o `receive` com esse
código dentro de `/runpod-volume/dados/datasets/`. Repetir para cada dataset.

Alternativa sem `runpodctl`: o pod expõe SSH, então `rsync -avP` funciona e é
retomável se a conexão cair — o que, com 3,5 GB, conta.

**4. Conferir o layout no volume:**

```
/runpod-volume/dados/datasets/spot5_refinado_v20/
/runpod-volume/dados/spot5/colinas_do_sul/original/...   (só p/ inferência)
/runpod-volume/runs/
/runpod-volume/inferencias/
```

**5. Matar o pod CPU.** O volume fica.

**6. Subir o pod GPU** com a sua imagem (`ghcr.io/ga0512/projeto-go:<sha>`, as
credenciais do GHCR vão em *Container Registry Auth* se o repo for privado),
o mesmo volume montado em `/vol`, e o comando de treino.

## Vast.ai

Mesma imagem, dois ajustes:

- **Disco:** pedir pelo menos 60 GB na instância, já que não há volume externo.
- **Dado:** ou `rsync` a cada instância nova, ou — melhor — deixar os datasets
  num bucket (B2/R2, centavos por mês) e baixar no início:

```bash
# dentro da instância, antes do treino
mkdir -p /vol/dados/datasets && cd /vol/dados/datasets
curl -sL "$URL_DO_BUCKET/spot5_refinado_v20.tar.gz" | tar xz
```

Com um bucket, subir uma vez resolve as duas plataformas e qualquer instância
futura.

## Erros comuns

- **Volume em região diferente da GPU** — no RunPod eles não se enxergam.
- **Esquecer `--gpus all`** — o `entrada.sh` avisa, imprime "GPU NENHUMA" e o
  `exigir_gpu()` aborta.
- **Montar o volume em outro caminho que não `/vol`** — os links do
  `entrada.sh` não são criados e o treino não acha o dataset.
- **Subir os originais inteiros** — passam de 30 GB e não entram no treino.
