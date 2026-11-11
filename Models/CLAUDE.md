# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Modelos LULC (SPOT5 pancromática 2,5 m) treinados sobre os datasets de patches do projeto irmão `../Data`. Regras operacionais do dia a dia: `AGENTS.md` (leia junto — **mas veja "Onde o AGENTS.md está desatualizado"** no fim). Pipeline de geração dos dados: `../Data/CLAUDE.md`.

Código, docstrings, nomes de argumentos e saída são **em português**. Todo script tem docstring com "Uso (no WSL, a partir de `Models/`)" — leia a docstring antes de rodar, ela é a documentação canônica de cada script.

## Ambiente (CRÍTICO)

- **Sempre** `../Data/.venv/bin/python <script>` a partir de `Models/`. O venv é Linux/WSL (python3.12, GDAL 3.8.4, numpy 1.26.4, torch 2.13.0+cu130, xgboost 2.1.4, timm 1.0.29); o Python global é quebrado para GDAL.
- Rodando no Windows, chamar pela ferramenta Bash via WSL (o PowerShell estraga as aspas aninhadas):
  `wsl.exe -e bash -c "cd /mnt/d/Projects/projeto-go/Models && ../Data/.venv/bin/python vit_model/train_seg.py ..."`
  No WSL o projeto é `/mnt/d/Projects/projeto-go/Models`; no Windows, `D:\Projects\projeto-go\Models`, também acessível pela junction `C:\Users\gabriel.cicotoste\HDSeagate\Projects\projeto-go\Models`.
- **`import ambiente` antes de torch/timm/xgboost/matplotlib** em qualquer script, novo inclusive. Ele redireciona `HF_HOME`/`TORCH_HOME`/`XDG_CACHE_HOME`/`TMPDIR`/etc. para `Models/.cache` (no D:), porque o vhdx do WSL mora no C: e já encheu no meio de um treino.
- GPU obrigatória: `ambiente.exigir_gpu()` aborta sem CUDA. Os scripts de inferência aceitam CPU no papel, mas na prática não.
- **Trava de C:**: os três `train*.py` medem `shutil.disk_usage("/mnt/c")` a cada 20 lotes e abortam se o livre cair abaixo de `--min-livre-c-gb` (1 GB), preservando `modelo_melhor.pth`. Só `train_mae.py` é retomável (`--resume` lê `runs/<nome>/ultimo.pth`).
- Sem git, sem testes, sem linter. **Validação = `relatorio.json` + `log.csv` + os PNGs em `runs/<nome>/visualizacao/`.** Depois de treinar, conferir o PNG é parte do trabalho.

## Comandos

```bash
# segmentação ViT (encoder: 'dino' | 'aleatorio' | caminho de um encoder.pth do MAE)
../Data/.venv/bin/python vit_model/train_seg.py --nome seg_mae_v3_ft_v13_uper \
    --encoder vit_model/runs/mae_vits_dino_v3_8k/encoder_epoca70.pth \
    --dataset ../Data/dados/datasets/spot5_refinado_v13
# U-Net / U-Net++ CNN (--arquitetura unetpp é o padrão)
../Data/.venv/bin/python unet_cnn/train.py --nome unetpp_v13 \
    --dataset ../Data/dados/datasets/spot5_refinado_v13
# XGBoost por pixel
../Data/.venv/bin/python xgboost/train.py --nome xgb_spot5_v13 \
    --dataset ../Data/dados/datasets/spot5_refinado_v13

# pré-treino MAE (só imagens; retomável)
../Data/.venv/bin/python vit_model/preparar_sem_rotulo.py            # empacota patches sem rótulo
../Data/.venv/bin/python vit_model/train_mae.py --nome mae_x --sem-rotulo spot5_013-014_sem_rotulo
../Data/.venv/bin/python vit_model/train_mae.py --nome mae_x --resume
../Data/.venv/bin/python vit_model/exportar_encoder.py --nome mae_x  # encoder de um treino interrompido

# inferência (aceita modelo seg_* OU unet*; a saída vai para a pasta do tipo de modelo)
../Data/.venv/bin/python vit_model/inference.py --modelo unet_cnn/runs/unetpp_v13/modelo_melhor.pth \
    --imagem ../Data/dados/spot5/jatai/tiles/r02_c05/imagem.tif \
    --referencia ../Data/dados/spot5/jatai/tiles/r02_c05/mascara_mapbiomas.tif
../Data/.venv/bin/python vit_model/inferencia_cena.py --modelo ... --imagem <cena inteira> --ate 16312
../Data/.venv/bin/python vit_model/inferencia_cena.py --modelo ... --imagem <cena inteira> --unir

# avaliação / diagnóstico
../Data/.venv/bin/python vit_model/comparar_seg.py [--split validacao]
../Data/.venv/bin/python vit_model/diagnostico_bordas.py --modelo <modelo_melhor.pth>
../Data/.venv/bin/python vit_model/avaliar_mapbiomas.py
```

**`--dataset` sempre explícito.** Os `DATASET_PADRAO` dos três `dados.py` apontam para datasets antigos (`spot5_refinado` v1 em `vit_model/`/`xgboost/`, `spot5_refinado_v6b` em `unet_cnn/`) e **nunca** para o atual.

Não há `__init__.py` nem entrypoint empacotado: cada script é rodado por caminho e conta com o Python pondo a **própria pasta** no `sys.path` — é assim que `import ambiente`/`import dados` resolvem para a cópia local do pacote. `python -m vit_model.train_seg` não funciona.

## Arquitetura

### Três famílias de modelos, um mesmo contrato de avaliação

`vit_model/` (ViT+MAE), `unet_cnn/` (CNN) e `xgboost/` (por pixel) são deliberadamente **paralelos e comparáveis**, não camadas de um sistema. Cada um treina no mesmo dataset, com a mesma augmentation (8 combinações de flip/rot90 em imagem+máscara; ganho/offset aleatórios **só na banda PAN**, porque as texturas são z-score e não mudam com brilho), a mesma loss (cross-entropy ignorando `IGNORE` + Dice suave), as mesmas métricas e o mesmo esquema de arquivos de saída:

- melhor epoch escolhido pelo **mIoU de validação** (com `--paciencia`); o **teste só é avaliado no fim**, com os pesos daquele epoch. Nada é escolhido olhando o teste.
- métricas calculadas duas vezes: no **patch inteiro** e no **"interior"** (`--margem 15` px da borda). O interior existe porque as janelas de features do XGBoost (`[5,15,31]`, margem = maior janela // 2) só cabem inteiras ali — **comparações entre famílias usam sempre o interior**.
- `relatorio.json` com o mesmo schema (`oa`, `miou`, `acuracia`/`precisao`/`f1`/`iou` por classe, `matriz_confusao`, `por_cena`) nos níveis `validacao`/`teste` × `inteiro`/`interior` (no XGBoost, sem o nível `inteiro`). `comparar_seg.py` só funciona porque esse schema é idêntico; ele agrupa por (classes, cenas), já que runs de 3 e de 4 classes não cabem na mesma tabela.

Consequência prática: ao mexer em augmentation, loss ou métrica em um dos `train*.py`, **replique nos outros** ou os runs antigos deixam de ser comparáveis com os novos.

### `ambiente.py` e `dados.py` são copiados, não compartilhados

As três pastas têm **cópias byte-a-byte** de `ambiente.py` e cópias quase idênticas de `dados.py` (não há pacote comum). Editar um e esquecer os outros é o erro mais fácil de cometer aqui. As divergências atuais entre as cópias de `dados.py` são intencionais e descritas abaixo.

### A fonte única de classes mora em `../Data`

Os três `dados.py` fazem `sys.path.insert` em `../Data/scripts/modelos/` e importam `config.py`: `CLASS_MAP` (código MapBiomas → índice), `CLASS_NOMES`, `IGNORE=255`, `REPR_CODIGO=[3,4,15,33]` (código escrito de volta no GeoTIFF de saída). **Alterar `CLASS_MAP` invalida todos os modelos treinados.** `dados.carregar()` aplica um LUT de 256 entradas sobre a máscara, manda código fora do `CLASS_MAP` para `IGNORE` (com aviso), e zera pixels não finitos da imagem marcando o rótulo como `IGNORE`.

### 4 classes × 3 classes: o detalhe mais perigoso do repositório

O dataset v15 ("juntar-veg") tem as **mesmas máscaras** dos outros (os códigos MapBiomas 3 e 4 continuam distintos no disco); a fusão vegetação densa + não densa acontece **do lado do Models**, em `dados._classes()`, que decide 3 ou 4 classes farejando o header do `manifest.csv` (`frac_veg` sem `frac_veg_densa` ⇒ 3 classes). Isso só vale para quem chama `dados.configurar(dataset)`:

| script | `configurar()` | efeito |
|---|---|---|
| `unet_cnn/train.py` | sim (e refaz os globais `N`/`IGN`) | único treino que aceita dataset de 3 classes |
| `vit_model/inference.py`, `inferencia_cena.py` | sim, com o dataset gravado no checkpoint | saída segue o modelo |
| `vit_model/train_seg.py` | **não** — `N = dados.N_CLASSES` é fixado no import, a partir do `DATASET_PADRAO` | com `--dataset ...v15` treina **silenciosamente** uma cabeça de 4 classes com rótulos de 4 classes; não quebra, só não é o experimento pretendido |
| `xgboost/dados.py` | **nem existe** (sem `_classes`/`configurar`) | sempre 4 classes |
| `probe.py`, `avaliar_mapbiomas.py`, `diagnostico_bordas.py` | não | assumem o nº de classes do import |

Para rodar um experimento de 3 classes no ViT ou no XGBoost, **é preciso portar `configurar()` primeiro** (e, no ViT, reavaliar `N`/`IGN` dentro de `main()` como o `unet_cnn/train.py` faz).

### Caminho ViT: MAE → segmentação

`mae.py` monta um Masked Autoencoder sobre um ViT do timm (`vit_small_patch16_224.dino`, `dynamic_img_size=True`, para pré-treinar em 224 e segmentar em 256 reamostrando o `pos_embed`). A adaptação de canais é o ponto fino: os pesos pré-treinados são RGB, nossas bandas são PAN/GLCM/rugosidade, então **o canal PAN recebe a soma dos três filtros RGB e os canais de textura começam em zero**. O decoder do MAE é descartável — `exportar_encoder()` grava só o encoder, no formato do timm, e é esse `.pth` que `train_seg.py --encoder <caminho>` consome. `train_mae.py` grava `encoder.pth` **apenas no último epoch**; se o treino parar antes, use `exportar_encoder.py` (gera `encoder_epoca<N>.pth` a partir de `ultimo.pth`). `probe.py` roda um probe linear sobre tokens congelados (rótulo da célula de 16×16 px = classe majoritária) no epoch 0 e a cada `--probe-cada`: é o termômetro de se o MAE está melhorando a representação.

`segmentacao.py` (`SegViT`) é um **UPerNet sobre ViT**: as features dos blocos 2/5/8/11 (todas em 16×16 px, porque o ViT não tem pirâmide) passam por convs 1×1 laterais, o nível mais fundo entra num PPM (pooling 1/2/3/6) para contexto global, a fusão é top-down e densa, o mapa sobe 16× para 256 e é somado a um **ramo convolucional raso sobre a imagem de entrada** — é esse ramo que devolve a borda que os tokens de 16 px perdem. Runs com sufixo `_uper` usam esse decoder; os anteriores usavam um UNETR mais simples (a docstring do `train_seg.py` ainda diz "UNETR leve").

### Caminho CNN e caminho XGBoost

`unet_cnn/unet.py` traz `UNet` (clássica, 4 níveis, base 32) e `UNetPP` (U-Net++, skips densos `X_i^j`, padrão atual). `unet_cnn/train.py` espelha `train_seg.py` e tem o único warm-start tolerante do repositório: `--inicializar` carrega só os tensores com shape compatível, então **dá para fine-tunar um modelo de 4 classes num dataset de 3** (a cabeça nova fica reinicializada) — foi assim que `unetpp_v15_ft` saiu de `unetpp_v13`. O `--inicializar` do `train_seg.py`, ao contrário, é estrito e aborta em chave desconhecida.

`xgboost/features.py` calcula, na GPU, as 3 bandas + média e desvio local por janela `[5,15,31]` = **21 features por pixel** (desvio via `E[x²]−E[x]²` com `avg_pool2d` e padding por reflexão — daí a margem). `train.py` sorteia até `--max-amostras` pixels com cota igual por patch, pesa classes pelo inverso da frequência, usa early stopping por `mlogloss` na validação e grava `modelo.ubj` já cortado na melhor rodada.

### Inferência

`inference.py` refaz **o mesmo caminho de bandas do gerador do dataset** (PAN esticada p2–p98 → uint8 → GLCM entropia janela 7 / 8 níveis, LocalSTD 7×7, Sobel → `[PAN, GLCM, (z(STD)+z(Grad))/2]`), com as estatísticas da cena lidas do `receita.json` do dataset do modelo (`--cena`, ou detectadas pelo nome da cena no caminho). Prediz em janelas de 256 px com passo `--passo 128`, **ponderadas com peso maior no centro** (a borda do patch é onde o modelo erra), somando probabilidades antes do argmax. `carregar_modelo()` descobre ViT × CNN pela presença de `raso.0.weight` no `state_dict`, e a saída vai para `vit_model/inferencias/` ou `unet_cnn/inferencias/<modelo>/<cena>_<tile>/` conforme o tipo — ou seja, `vit_model/inference.py` grava em `unet_cnn/` quando o modelo é CNN.

`inferencia_cena.py` roda a cena inteira (857 Mpx na 013-014) em faixas horizontais de `--faixa 1024` linhas com `--overlap 300`, cada faixa tratada como imagem independente e gravada em `faixa_<y>.tif`; é **retomável** por `progresso.txt` e junta tudo num `classes.tif` via VRT (`--unir` só junta, sem inferir). Não misture, na mesma pasta, faixas vindas de execuções com parâmetros diferentes.

`avaliar_mapbiomas.py` é a única avaliação **externa**: usa os patches `uso=aval` que `preparar_sem_rotulo.py` separou por blocos espaciais (fora do pré-treino do MAE e fora do tile refinado, e — diferente do teste — com agropecuária), comparando por célula de 30 m (12×12 px) contra o MapBiomas 2007.

## Convenções

- **Run = `<modelo>_v<N>`, onde `v<N>` é a versão do dataset**: `unetpp_v13` ⇒ `spot5_refinado_v13`. `seg_mae_v3_*` = encoder `mae_vits_dino_v3_8k/encoder_epoca70.pth`; `cong` = encoder congelado, `ft` = fine-tuning, `_uper` = decoder UPerNet, `_ft` em `unetpp_v15_ft` = warm-start de outro run.
- `.cache/`, `*/runs/`, `*/inferencias/` e `vit_model/dados/` são **artefatos**, não código (~23 MB por `modelo_melhor.pth`, GB em `ultimo.pth` e nas faixas de cena).
- Arquivos grandes no `/mnt/d`: renomear/mover pode falhar com o drive instável — retentar após `wsl --shutdown`. O vhdx do Ubuntu cresce e não encolhe; compactar com `Optimize-VHD -Mode Full` (WSL parado) quando o C: apertar.
- Nem todo run está completo. Sem `relatorio.json` = treino interrompido (só `log.csv` + `modelo_melhor.pth`); é o caso de `unetpp_v12`, `unetpp_v13`, `unetpp_v15_ft`, `xgb_spot5_v8` e `xgb_spot5_v11`. Para esses, o único número que existe é o **mIoU de validação** do `log.csv`, que não é comparável com o mIoU de teste dos outros.

## Estado atual (out/2026)

Datasets em `../Data/dados/datasets/`: `spot5_refinado` (v1) e `_v4`…`_v15`, mais `spot5_013-014_sem_rotulo`. Todos com patches 256, 3 bandas `PAN`/`GLCM_Entropia`/`Rugosidade_STD_Gradiente`, 4 cenas (`colinas_do_sul`, `cena_013-014_2007-06-14`, `jatai`, `recorte_2007-09-10`).

| dataset | splits (treino/val/teste) | classes | observação |
|---|---|---|---|
| v11 | 1864/413/412 | 4 | base das comparações consolidadas |
| v12 / v13 | 2343/459/476 (v13) | 4 | `total` 3000 / 3500 |
| v14 | 14/76/74 | 4 | dataset minúsculo, só serve para testar código |
| v15 | 2343/459/476 | **3** | vegetação densa + não densa fundidas na leitura |

Melhores mIoU de **teste (interior)** por família: ViT `seg_mae_v3_ft_v11_uper` 0.757 (v11) e `seg_mae_v3_ft_v13_uper` 0.723 (v13); CNN `unetpp_v11` 0.734; XGBoost `xgb_spot5_v9` 0.641 — o último XGBoost concluído. A linha de frente está sem `relatorio.json`: `unetpp_v13` parou no epoch 30 com val mIoU 0.858, e `unetpp_v15_ft` (3 classes, warm-start de `unetpp_v13`) no epoch 19 com val mIoU 0.946 — número inflado pela fusão das duas classes de vegetação, que era justamente a confusão dominante. As cenas inteiras já inferidas com `unetpp_v15_ft` (`colinas_do_sul`, `cena_013-014_2007-06-14`) são o produto mais recente.

## Onde o AGENTS.md está desatualizado

`AGENTS.md` foi escrito na época do v11 e descreve bem o ambiente, mas não o estado atual do código. Ao usá-lo, corrija:

- "4 classes" não é mais universal: v15 tem 3 (ver a tabela acima).
- O dataset atual não é o v11; existem v12…v15, e o trabalho recente é em v13/v15.
- Os decoders mudaram: `SegViT` é UPerNet (não "UNETR leve"), e `unet_cnn` tem U-Net++ como padrão (`--arquitetura`), não só a U-Net base 32.
- `inferencia_cena.py` (cena inteira, retomável) e o warm-start `--inicializar` não estão lá.
- Os runs de referência citados (`seg_mae_v3_ft_v11` 0.728, `unet_v11` 0.724) foram superados por `seg_mae_v3_ft_v11_uper` (0.757) e `unetpp_v11` (0.734) no mesmo v11.
