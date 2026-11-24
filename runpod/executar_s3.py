#!/usr/bin/env python3
"""Executa um comando no Pod com entradas e saídas em um Network Volume S3.

O Pod baixa as entradas antes do comando e envia cada saída existente ao final,
inclusive se o comando falhar. Caminhos são relativos à raiz do repositório;
o comando é executado a partir de Models/. O PC não participa da transferência.

Uso (no WSL, a partir de Models/):

    ../Data/.venv/bin/python ../runpod/executar_s3.py \
      --volume ID --datacenter US-GA-2 \
      --entrada Data/dados/datasets/wv_jatai_v4 \
      --saida Models/unet_cnn/runs/NOME \
      -- ../Data/.venv/bin/python -u unet_cnn/train.py \
           --nome NOME --dataset ../Data/dados/datasets/wv_jatai_v4

Credenciais no Pod: AWS_ACCESS_KEY_ID e AWS_SECRET_ACCESS_KEY, ou os equivalentes
RUNPOD_S3_ACCESS_KEY e RUNPOD_S3_SECRET_KEY. Boto3 deve estar instalado na venv.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from subir import baixar, cliente, enviar, remotos


RAIZ = Path(__file__).resolve().parent.parent
MODELS = RAIZ / "Models"


def _caminho_relativo(valor):
    caminho = Path(valor)
    if caminho.is_absolute() or ".." in caminho.parts or not caminho.parts:
        raise argparse.ArgumentTypeError("use caminho relativo à raiz do repositório")
    return caminho.as_posix().rstrip("/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", default=os.environ.get("RUNPOD_VOLUME_ID"), required=False)
    ap.add_argument("--datacenter", default=os.environ.get("RUNPOD_DATACENTER"), required=False)
    ap.add_argument("--entrada", action="append", type=_caminho_relativo, default=[])
    ap.add_argument("--saida", action="append", type=_caminho_relativo, default=[])
    ap.add_argument("comando", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if not a.volume or not a.datacenter:
        ap.error("informe --volume e --datacenter, ou RUNPOD_VOLUME_ID e RUNPOD_DATACENTER")
    comando = a.comando[1:] if a.comando[:1] == ["--"] else a.comando
    if not comando:
        ap.error("informe o comando depois de --")

    cli = cliente(a.datacenter)
    for chave in a.entrada:
        destino = RAIZ / chave
        if not remotos(cli, a.volume, chave):
            raise FileNotFoundError("entrada ausente no S3: " + chave)
        print("[s3] baixando " + chave, flush=True)
        baixar(cli, a.volume, chave, str(destino))

    codigo = 1
    ausentes = []
    try:
        print("[exec] " + " ".join(comando), flush=True)
        codigo = subprocess.call(comando, cwd=MODELS)
    finally:
        for chave in a.saida:
            origem = RAIZ / chave
            if not origem.exists():
                print("[s3] saída ausente: " + chave, file=sys.stderr, flush=True)
                ausentes.append(chave)
                continue
            print("[s3] enviando " + chave, flush=True)
            enviar(cli, a.volume, str(origem), chave, forcar=True, tudo=True)
    if codigo == 0 and ausentes:
        raise FileNotFoundError("saídas esperadas ausentes: " + ", ".join(ausentes))
    return codigo


if __name__ == "__main__":
    sys.exit(main())
