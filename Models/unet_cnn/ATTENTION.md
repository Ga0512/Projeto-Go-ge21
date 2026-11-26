# Configurable attention for U-Net and U-Net++

`attention.py` provides reusable `ChannelAttention`, `SpatialAttention`,
`CBAM`, and `AttentionRouter` modules. The same placement options work in
`unet_cnn/train.py` and `runpod/rodada_unet_s3.py`:

| Option | Effect |
| --- | --- |
| `--attention-input` | Channel attention on the input bands |
| `--attention-encoder` | CBAM after every encoder block |
| `--attention-decoder` | CBAM after every decoder block |
| `--attention-nodes e1 x03` | CBAM after selected named blocks; combines with the group options |

U-Net supports encoder nodes `e0`–`e3`, bottleneck `bot`, and decoder nodes
`d0`–`d3`. U-Net++ supports encoder nodes `e0`–`e3` and decoder nodes `x01`,
`x11`, `x21`, `x02`, `x12`, `x03`. Its existing `bot` output is disconnected;
placing attention there would not affect predictions, so it is excluded.

To switch from input plus decoder to **encoder only**, remove the input and
decoder options and use `--attention-encoder`. No model source edit is needed.
The input gate produces one weight per band and patch. Encoder and decoder
gates operate on learned features, so their channels do not correspond to
individual input bands.

Old checkpoints without attention still load strictly. Checkpoints created
before the English naming change are translated when loaded. New checkpoints
record the chosen locations in English under `args`.

## Comparable SPOT5 Colinas training

The reference run is `unetpp_spot5_colinas_v28`. Its dataset has seven inputs:
PAN, GLCM entropy, gradient roughness, and four temporal NDVI statistics.
The reference used seed 42, base 32, batch 32, 30 epochs, patience 10,
warmup 2, border radius 5/alpha 4, and resampling k=3. With one scene,
resampling k=3 produces uniform sampling.

Uso (no WSL, a partir de Models/):

```bash
../Data/.venv/bin/python unet_cnn/train.py \
  --nome unetpp_spot5_colinas_v28_input_decoder_attention \
  --dataset ../Data/dados/datasets/spot5_colinas_v28 \
  --architecture unetpp --base 32 \
  --attention-input --attention-decoder \
  --lote 32 --lote-val 64 --epocas 30 --paciencia 10 --aquecimento 2 \
  --reamostrar --reamostrar-k 3 --borda-raio 5 --borda-alfa 4 \
  --dados-na-ram --seed 42
```

For encoder only, replace the two attention options with
`--attention-encoder`. Use the same test split for comparison with the
reference run. An epoch on a tiny smoke dataset verifies code execution;
it does not estimate segmentation quality.

## Audit input band weights

```bash
../Data/.venv/bin/python unet_cnn/audit_attention.py \
  --name unetpp_spot5_colinas_v28_input_decoder_attention \
  --dataset ../Data/dados/datasets/spot5_colinas_v28
```

`attention_validation.csv` and `attention_test.csv` contain seven weights per
patch, a patch ID, scene, and label fractions. The matching `_summary.json`
files aggregate them by band and majority class. These weights are adaptive
gates, not causal importance scores; per-band occlusion is needed to measure
the effect of removing a band. The RunPod workflow uploads audit files to S3
when `--attention-input` is enabled.
