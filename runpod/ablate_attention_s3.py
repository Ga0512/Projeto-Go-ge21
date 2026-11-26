#!/usr/bin/env python3
"""Treina variantes de atencao no mesmo dataset e salva cada run no S3.

Uso (no WSL, a partir de Models/; tambem dentro do Pod):
  ../Data/.venv/bin/python -u ../runpod/ablate_attention_s3.py \
    --dataset spot5_colinas_v28 --name-prefix unetpp_spot5_colinas_v28 \
    --variants input_decoder encoder decoder --volume g643dn7nah \
    --datacenter US-GA-2

O dataset e baixado uma vez. Cada variante usa a mesma receita, seed e splits.
O checkpoint, relatorios e logs sao enviados ao S3 apos cada treino.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from rodada_unet_s3 import RAIZ, MODELS, PY, comando
from subir import baixar, cliente, enviar


VARIANTS = {
    "input_decoder": ("--attention-input", "--attention-decoder"),
    "encoder": ("--attention-encoder",),
    "decoder": ("--attention-decoder",),
    "input": ("--attention-input",),
    "input_encoder": ("--attention-input", "--attention-encoder"),
    "all": ("--attention-input", "--attention-encoder", "--attention-decoder"),
}
RESULT_FILES = (
    "modelo_melhor.pth", "ultimo.pth", "relatorio.json",
    "relatorio_completo.json", "log.csv", "rodada_s3.log",
    "attention_validation.csv", "attention_validation_summary.json",
    "attention_test.csv", "attention_test_summary.json",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name-prefix", required=True)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    ap.add_argument("--volume", default=os.environ.get("RUNPOD_VOLUME_ID"))
    ap.add_argument("--datacenter", default=os.environ.get("RUNPOD_DATACENTER"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--input-attention-mode", choices=["softmax", "sigmoid"],
                    default="softmax")
    ap.add_argument("--border-radius", type=int, default=5)
    ap.add_argument("--border-alpha", type=float, default=4.0)
    a = ap.parse_args()
    if not a.volume or not a.datacenter:
        ap.error("configure --volume e --datacenter (ou variaveis RUNPOD_*)")
    if any("/" in value or "\\" in value or value in ("", ".", "..")
           for value in (a.dataset, a.name_prefix)):
        ap.error("dataset e name-prefix devem ser nomes simples")

    cli = cliente(a.datacenter)
    dataset_key = "Data/dados/datasets/" + a.dataset
    dataset = RAIZ / dataset_key
    print("[s3] baixando dataset " + dataset_key, flush=True)
    baixar(cli, a.volume, dataset_key, str(dataset))
    recipe = json.loads((dataset / "receita.json").read_text(encoding="utf-8"))
    expected_bands = ["PAN", "GLCM_Entropia", "Rugosidade_STD_Gradiente",
                      "NDVI_median", "NDVI_min", "NDVI_max", "NDVI_stdDev"]
    if recipe["bandas"] != expected_bands:
        raise ValueError("ordem de bandas inesperada: " + repr(recipe["bandas"]))
    expected_patches = sum(recipe["resultado"]["total_por_split"].values())
    for folder in ("images", "masks"):
        actual = len(list((dataset / folder).glob("*.tif")))
        if actual != expected_patches:
            raise ValueError("dataset incompleto: %s=%d, esperado=%d" %
                             (folder, actual, expected_patches))
    print("[dataset] %d pares; 7 bandas" % expected_patches, flush=True)

    for variant in a.variants:
        name = a.name_prefix + "_attn_" + variant
        run = MODELS / "unet_cnn" / "runs" / name
        if (run / "modelo_melhor.pth").exists():
            raise FileExistsError("run already has a checkpoint; use a new name: " + name)
        run.mkdir(parents=True, exist_ok=True)
        log = run / "rodada_s3.log"
        print("\n[variant] %s" % name, flush=True)

        def upload(path):
            if not path.is_file():
                return
            key = path.relative_to(RAIZ).as_posix()
            enviar(cli, a.volume, str(path), key, forcar=True, tudo=True)
            remote_size = cli.head_object(Bucket=a.volume, Key=key)["ContentLength"]
            if remote_size != path.stat().st_size:
                raise RuntimeError("upload incompleto: " + key)
            print("[s3] salvo " + key, flush=True)

        def upload_results():
            for filename in RESULT_FILES:
                upload(run / filename)

        def upload_checkpoint(line):
            if not line.startswith("epoca "):
                return
            try:
                epoch = int(line.split()[1])
            except (IndexError, ValueError):
                return
            if epoch % 5 == 0:
                upload(run / "modelo_melhor.pth")

        start = time.monotonic()
        try:
            train = [PY, "-u", "unet_cnn/train.py", "--nome", name,
                     "--dataset", "../Data/dados/datasets/" + a.dataset,
                     "--architecture", "unetpp", "--base", "32",
                     "--lote", str(a.batch_size), "--lote-val", str(a.val_batch_size),
                     "--epocas", str(a.epochs), "--paciencia", str(a.patience),
                     "--aquecimento", str(a.warmup), "--reamostrar",
                     "--reamostrar-k", "3", "--dados-na-ram",
                     "--borda-raio", str(a.border_radius),
                     "--borda-alfa", str(a.border_alpha), "--seed", "42"]
            train.extend(VARIANTS[variant])
            if "--attention-input" in VARIANTS[variant]:
                train.extend(("--attention-input-mode", a.input_attention_mode))
            comando(train, log, ao_ler=upload_checkpoint)
            if not (run / "modelo_melhor.pth").is_file():
                raise FileNotFoundError("treino sem modelo_melhor.pth: " + name)
            upload_results()
            evaluate = [PY, "-u", "unet_cnn/avaliar.py", "--nome", name,
                        "--dataset", "../Data/dados/datasets/" + a.dataset,
                        "--splits", "validacao", "teste", "--lote",
                        str(a.val_batch_size)]
            comando(evaluate, log)
            if "--attention-input" in VARIANTS[variant]:
                audit = [PY, "-u", "unet_cnn/audit_attention.py", "--name", name,
                         "--dataset", "../Data/dados/datasets/" + a.dataset,
                         "--batch-size", str(a.val_batch_size)]
                comando(audit, log)
            upload_results()
            print("[done] %s em %.1f min" %
                  (name, (time.monotonic() - start) / 60), flush=True)
        finally:
            upload_results()


if __name__ == "__main__":
    main()
