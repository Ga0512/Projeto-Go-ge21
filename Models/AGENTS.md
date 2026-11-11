# AGENTS.md

Modelos LULC (SPOT5 2,5 m, 4 classes) sobre os datasets do projeto irmão `../Data`. Código e saída em português; cada script tem docstring com "Uso (no WSL, a partir de Models/)" — leia o docstring antes de rodar.

## Ambiente (CRÍTICO)

- **Rodar Python sempre via** `../Data/.venv/bin/python <script>` a partir de `Models/`. O Python global é quebrado para GDAL. Venv tem torch 2.13.0+cu130, xgboost 2.1.4 (GPU), timm 1.0.29, numpy 1.26.4.
- **Importar `ambiente` ANTES de torch/xgboost/timm/matplotlib** em qualquer script (novo script também). Ele move todos os caches (`HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TMPDIR`, etc.) para `Models/.cache` (D:) porque o disco do WSL (vhdx) vive no C:, que já chegou a 100% e derrubava o WSL no meio do treino (foi compactado com `Optimize-VHD`).
- **GPU obrigatória** em treino e cálculo de features (`ambiente.exigir_gpu()`). `inference.py` e `diagnostico_bordas.py` caem para CPU, mas na prática exigem GPU.
- **Trava de C:**: `train_mae.py`/`train_seg.py`/`unet_cnn/train.py` abortam (com checkpoint/`modelo_melhor.pth` preservado) se `/mnt/c` livre cair abaixo de `--min-livre-c-gb` (padrão 1 GB). `train_mae.py` é retomável: `--resume` lê `runs/<nome>/ultimo.pth`.
- Sem git, sem testes automatizados, sem linter. Validação = `relatorio.json`, `log.csv`, PNGs em `runs/<nome>/`.

## Dados e classes

- Fontes vêm de `../Data` (ver `../Data/AGENTS.md`). `vit_model/dados.py`, `xgboost/dados.py` e `unet_cnn/dados.py` definem `DATA` = `../Data` e importam **`Data/scripts/modelos/config.py`** via `sys.path` — fonte única de `CLASS_MAP`/`CLASS_NOMES` (4 classes), `IGNORE=255`, `REPR_CODIGO=[3,4,15,33]`. **Alterar `CLASS_MAP` invalida modelos treinados.**
- **`dados.DATASET_PADRAO` está DESATUALIZADO**: em `vit_model/` e `xgboost/` aponta para `spot5_refinado` (v1), e em `unet_cnn/` para `spot5_refinado_v6b`. **Sempre passe `--dataset` explicitamente.** Dataset atual: `../Data/dados/datasets/spot5_refinado_v11` (patches 256, 3 bandas `PAN`/`GLCM_Entropia`/`Rugosidade_STD_Gradiente`, splits 1864/413/412, inclui a cena `jatai`). Linhagem: `v6b` → `v6b_buffer5m` (**experimento abandonado**; v7+ usa `buffer: 0`) → `v7`…`v11` (v8 `bloco 512`, v9/v10/v11 `bloco 1024`; v11 `zona_morta 320`, `total 2750`, `sem_repetir_terreno`, cena `jatai`).
- **Buffer 5 m erode só as máscaras** (labels → 0 = sem rótulo; imagens idênticas). No teste o buffer5m exclui ~6% dos pixels (a faixa erodida), então comparar mIoU com o dataset sem buffer **não é maçã com maçã** — o teste fica menor e mais fácil. Para comparar de forma justa, avalie o modelo buffer5m no teste do dataset original.
- Texturas (GLCM/rugosidade) são z-score por cena; só a PAN varia com brilho/contraste na augmentation. ViT usa 3 canais: **PAN = soma dos pesos RGB do DINO, texturas começam em zero** (ver `mae.py` `_adaptar_canais`).

## Comandos (rodar de `Models/`)

- `vit_model/preparar_sem_rotulo.py` — empacota o dataset sem rótulo em `patches.npy` float16 em `vit_model/dados/<nome>/`; separa patches `uso=aval` por blocos espaciais (ficam fora do MAE).
- `vit_model/train_mae.py` — pré-treino MAE (encoder ViT-S/16 DINO do timm + decoder descartável). **`encoder.pth` só é gravado no último epoch**; `encoder_melhor_probe.pth` sai do probe. Treino interrompido? `vit_model/exportar_encoder.py --nome <run>` extrai `encoder_epoca<N>.pth` do `ultimo.pth`.
- `vit_model/train_seg.py` — segmentação UNETR leve. `--encoder` = `dino` | `aleatorio` | caminho de `encoder.pth`. Métricas no patch inteiro e no "interior" (`--margem 15` px, igual ao XGBoost). Saída `runs/<nome>/modelo_melhor.pth` (melhor mIoU de validação; teste só no fim).
- `vit_model/inference.py` — roda um `seg_*` numa PAN GeoTIFF inteira; saída em `vit_model/inferencias/<modelo>/<nome>/` (classes.tif com códigos MapBiomas, classes.png, info.json).
- `vit_model/comparar_seg.py` — tabela no teste (interior) entre `runs/seg_*` e `xgboost/runs/*`.
- `vit_model/diagnostico_bordas.py` — erro × distância à borda de classe (distingue confusão de transição da de modelo).
- `vit_model/avaliar_mapbiomas.py` — avaliação externa nos patches `uso=aval` × MapBiomas 2007 30 m, por célula de 30 m (12×12 px).
- `xgboost/train.py` — XGBoost por pixel na GPU: 3 bandas + média/desvio local em janelas `[5,15,31]` = 21 features (`xgboost/features.py`); treino/avaliação só no interior (`margem` = maior janela // 2). Saída `xgboost/runs/<nome>/modelo.ubj` + `relatorio.json`.
- `unet_cnn/train.py` — U-Net CNN (base 32, 3,3 M params), espelha `train_seg.py` (mesma aug/loss/métricas, saída `unet_cnn/runs/<nome>/`).

## Convenções

- **Os runs seguem a versão do dataset**: `*_v<N>` = treinado em `spot5_refinado_v<N>` (ex.: `seg_mae_v3_ft_v11`, `unet_v11`, `xgb_spot5_v9`). Segmentações: `seg_mae_v3_*` com encoder `vit_model/runs/mae_vits_dino_v3_8k/encoder_epoca70.pth` (`cong` = encoder congelado, `ft` = fine-tuning).
- Runs atuais de referência (v11, teste interior): `seg_mae_v3_ft_v11` (mIoU 0.728), `unet_v11` (0.724). `xgb_spot5_v11` está **incompleto** (só `treino.log`, sem `relatorio.json`/`modelo.ubj`); `xgb_spot5_v8` idem; `xgb_spot5_v9` (v9, mIoU 0.641) é o último XGBoost concluído.
- `Models/.cache`, `vit_model/runs`, `xgboost/runs`, `unet_cnn/runs` e `vit_model/dados/` são artefatos — não tratar como código.
- Arquivos grandes no `/mnt/d` (WSL): renomear/mover pode falhar se o drive estiver instável — retentar após `wsl --shutdown`. O vhdx do Ubuntu (`C:\Users\...\AppData\Local\wsl\{...}\ext4.vhdx`) cresce e não encolhe sozinho; compactar com `Optimize-VHD -Mode Full` (WSL parado) quando o C: apertar.