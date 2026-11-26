# Atenção na UNet++

`atencao.py` oferece `AtencaoCanais`, `AtencaoEspacial` e `CBAM` como módulos
PyTorch independentes. Na `UNetPP`, as opções são independentes:

- `--atencao-entrada`: aplica `AtencaoCanais` às bandas antes do encoder.
- `--cbam-encoder`: aplica `CBAM` depois de `e0`, `e1`, `e2`, `e3`.
- `--cbam-decoder`: aplica `CBAM` depois dos seis blocos convolucionais do
  decoder denso (`x01`, `x11`, `x21`, `x02`, `x12`, `x03`).
- `--cbam-nos e1 x03`: aplica `CBAM` só nos nós indicados. Pode ser combinado
  com as opções de grupos acima.

Para treinar **somente com CBAM no encoder**, passe `--cbam-encoder` e omita
`--atencao-entrada` e `--cbam-decoder`. As mesmas opções existem na rodada
`runpod/rodada_unet_s3.py`; não é preciso editar o modelo para mover a atenção.
O nó `bot` não é opção porque o forward atual da UNet++ não usa sua saída;
inseri-lo ali não alteraria a predição. Essa limitação também existe no
baseline, então não foi alterada neste experimento.

Sem essas opções, a arquitetura e as chaves dos checkpoints antigos continuam
iguais. `train.py` grava as opções em `modelo_melhor.pth`; `avaliar.py` e
`inferencia_cena.py` reconstroem a arquitetura a partir do checkpoint.

## Treino comparável em Colinas

Use `spot5_colinas_v28` para comparar com `unetpp_spot5_colinas_v28`. Ele contém
sete canais: PAN, GLCM, rugosidade e quatro estatísticas temporais de NDVI. O
baseline usou seed 42, base 32, lote 32, 30 épocas, paciência 10, aquecimento
2, borda raio 5/alfa 4 e a opção de reamostragem k=3. Como há uma única cena,
essa reamostragem resulta em sorteio uniforme.

No WSL, a partir de `Models/`:

```bash
../Data/.venv/bin/python unet_cnn/train.py \
  --nome unetpp_cbam_spot5_colinas_7b_p256_v28 \
  --dataset ../Data/dados/datasets/spot5_colinas_v28 \
  --arquitetura unetpp --base 32 --atencao-entrada --cbam-decoder \
  --lote 32 --lote-val 64 --epocas 30 --paciencia 10 --aquecimento 2 \
  --reamostrar --reamostrar-k 3 --borda-raio 5 --borda-alfa 4 \
  --dados-na-ram --seed 42
```

Ajuste `--lote` e `--lote-val` à GPU; ao comparar resultados, registre a
alteração. A seleção do checkpoint usa somente a validação. Compare o mIoU e
os IoUs por classe no **mesmo split de teste** do baseline.

## Auditar os sete pesos de entrada

```bash
../Data/.venv/bin/python unet_cnn/auditar_atencao.py \
  --nome unetpp_cbam_spot5_colinas_7b_p256_v28 \
  --dataset ../Data/dados/datasets/spot5_colinas_v28
```

Os arquivos `atencao_validacao.csv` e `atencao_teste.csv` trazem um vetor de
sete pesos por patch, com ID, cena e fração rotulada de cada classe. Os JSONs
correspondentes resumem a distribuição por banda e por classe predominante.
São portas adaptativas por patch, **não** uma medida causal de importância de
banda. Para testar importância, compare o IoU por classe ao ocultar cada banda
no mesmo conjunto de patches.

No RunPod, `runpod/rodada_unet_s3.py` aceita as mesmas duas opções e envia os
arquivos de auditoria ao volume S3 junto com o modelo e os relatórios. O treino
e a inferência locais continuam usando os scripts habituais.
