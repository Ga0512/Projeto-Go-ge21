#!/usr/bin/env bash
# Treino comparavel ao v32: dataset Colinas SPOT5 v33 com NDVI variacional.
# Uso (no WSL, a partir de Models/):
#   bash ../runpod/treinar_colinas_v33_autostop.sh
# No Pod, executar em segundo plano. O proprio script para o Pod ao sair.
set -Eeuo pipefail

POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID ausente}
VOLUME_ID=${RUNPOD_VOLUME_ID:-g643dn7nah}
DATACENTER=${RUNPOD_DATACENTER:-US-GA-2}
DATASET=spot5_colinas_v33_7b_fusion
RUN=unetpp_spot5_colinas_v33
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODELS="$ROOT/Models"
PY="$ROOT/Data/.venv/bin/python"
RUN_KEY="Models/unet_cnn/runs/$RUN"
RUN_DIR="$ROOT/$RUN_KEY"
mkdir -p "$RUN_DIR"

stop_pod() {
    python3 - "$POD_ID" <<'PY'
import json
from pathlib import Path
import sys
import time
import tomllib
from urllib.request import Request, urlopen

pod = sys.argv[1]
config = tomllib.loads((Path.home() / '.runpod' / 'config.toml').read_text())
key = config.get('default', {}).get('api_key')
if not key:
    raise RuntimeError('API key do RunPod ausente; Pod nao pode se desligar')
request = Request(
    f'https://api.runpod.io/v2/pods/{pod}/action',
    data=json.dumps({'action': 'stop'}).encode(),
    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
    method='POST',
)
for attempt in range(4):
    try:
        with urlopen(request, timeout=20) as response:
            print(f'[autostop] HTTP {response.status}', flush=True)
        break
    except Exception as exc:
        if attempt == 3:
            raise
        print(f'[autostop] tentativa {attempt + 1} falhou: {exc}', flush=True)
        time.sleep(5)
PY
}

if [[ ${1:-} == --stop-only ]]; then
    stop_pod
    exit 0
fi

finish() {
    status=$?
    trap - EXIT
    echo "[fim] codigo=$status; parando Pod $POD_ID" >> "$RUN_DIR/dispatch.log"
    if [[ -x "$PY" ]]; then
        "$PY" "$ROOT/runpod/subir.py" "$RUN_DIR/dispatch.log" \
            --para "$RUN_KEY/dispatch.log" --volume "$VOLUME_ID" \
            --datacenter "$DATACENTER" --forcar >> "$RUN_DIR/dispatch.log" 2>&1 || true
    fi
    stop_pod >> "$RUN_DIR/dispatch.log" 2>&1 || echo '[autostop] FALHOU' >> "$RUN_DIR/dispatch.log"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM

# Failsafe caso o processo principal seja morto sem executar o trap.
( sleep 5400; bash "$0" --stop-only ) > "$RUN_DIR/watchdog.log" 2>&1 < /dev/null &

exec > >(tee -a "$RUN_DIR/dispatch.log") 2>&1
echo "[inicio] $(date -u +%FT%TZ) Pod=$POD_ID dataset=$DATASET"
cd "$MODELS"

# A imagem PyTorch do RunPod ja contem torch/CUDA. Instalar apenas o que faltar.
if ! command -v gdal-config >/dev/null || ! python -c 'from osgeo import gdal' >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        gdal-bin libgdal-dev python3-gdal python3-venv
fi
python -m pip install --disable-pip-version-check -q boto3 timm rasterio scipy matplotlib
if ! python -c 'from osgeo import gdal' >/dev/null 2>&1; then
    python -m pip install --disable-pip-version-check -q \
        --no-build-isolation "GDAL==$(gdal-config --version)"
fi
mkdir -p "$ROOT/Data/.venv/bin"
if [[ ! -x "$PY" ]]; then
    cat > "$PY" <<'WRAPPER'
#!/bin/sh
exec python "$@"
WRAPPER
    chmod +x "$PY"
fi

"$PY" - <<'PY'
import ambiente
import torch
from osgeo import gdal
import boto3
assert torch.cuda.is_available(), 'GPU indisponivel'
print('[ambiente]', torch.__version__, torch.cuda.get_device_name(0),
      'GDAL', gdal.VersionInfo(), flush=True)
PY

"$PY" -u "$ROOT/runpod/executar_s3.py" \
    --volume "$VOLUME_ID" --datacenter "$DATACENTER" \
    --entrada "Data/dados/datasets/$DATASET" --saida "$RUN_KEY" -- \
    "$PY" -u unet_cnn/train.py \
      --nome "$RUN" --dataset "../Data/dados/datasets/$DATASET" \
      --architecture unetpp --base 32 --lote 32 --lote-val 64 \
      --epocas 30 --paciencia 10 --aquecimento 2 \
      --borda-raio 5 --borda-alfa 4 --reamostrar --reamostrar-k 3 \
      --dados-na-ram --seed 42

"$PY" -u "$ROOT/runpod/executar_s3.py" \
    --volume "$VOLUME_ID" --datacenter "$DATACENTER" \
    --saida "$RUN_KEY" -- \
    "$PY" -u unet_cnn/avaliar.py \
      --nome "$RUN" --dataset "../Data/dados/datasets/$DATASET" \
      --splits validacao teste --lote 64

echo '[concluido] treino e avaliacao enviados ao S3'
