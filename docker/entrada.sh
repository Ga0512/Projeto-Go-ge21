#!/bin/sh
# Liga o volume ao layout que os scripts esperam, e só depois executa o comando.
#
# O volume do pod entra em /vol e deve conter:
#   /vol/dados/     -> vira /work/Data/dados   (datasets, cenas, máscaras)
#   /vol/runs/      -> vira /work/Models/unet_cnn/runs   (saída, persiste)
#   /vol/inferencias/ -> vira /work/Models/unet_cnn/inferencias  (opcional)
#
# Sem /vol o container ainda roda (útil para --help e teste da imagem), só não
# enxerga dado nem persiste resultado.
set -e

ligar() {   # $1 = origem no volume, $2 = destino no repositório
    [ -d "$1" ] || mkdir -p "$1"
    rm -rf "$2"
    mkdir -p "$(dirname "$2")"
    ln -s "$1" "$2"
}

if [ -d /vol ]; then
    ligar /vol/dados        /work/Data/dados
    ligar /vol/runs         /work/Models/unet_cnn/runs
    ligar /vol/inferencias  /work/Models/unet_cnn/inferencias
    echo "[container] volume ligado: $(ls /vol)"
else
    echo "[container] AVISO: /vol não montado — sem dado e sem persistência"
fi

# Registra no log o que está rodando: sem isso um resultado no volume não tem
# proveniência (é a dívida de rastreabilidade levantada na auditoria).
echo "[container] imagem: ${IMAGEM_SHA:-nao informado}"
python3 - <<'PY'
import torch
print("[container] torch %s | CUDA %s | GPU %s"
      % (torch.__version__, torch.version.cuda,
         torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NENHUMA"))
PY

exec "$@"
