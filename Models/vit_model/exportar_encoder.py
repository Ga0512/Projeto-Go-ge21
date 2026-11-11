#!/usr/bin/env python3
"""Exporta só o encoder (formato timm) de um checkpoint do MAE (ultimo.pth),
para quando o treino foi interrompido antes do fim (o encoder.pth só é gravado
no último epoch).

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python vit_model/exportar_encoder.py --nome mae_vits_dino_v3_8k
  -> vit_model/runs/<nome>/encoder_epoca<N>.pth
"""

import argparse
import os

import ambiente  # noqa: F401  (caches -> D:)
import torch

from mae import MAE

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", required=True, help="run em vit_model/runs/")
    args = ap.parse_args()

    pasta = os.path.join(BASE, "runs", args.nome)
    ck = torch.load(os.path.join(pasta, "ultimo.pth"), map_location="cpu", weights_only=False)
    a = ck["args"]
    in_chans = ck["mae"]["encoder.patch_embed.proj.weight"].shape[1]
    mae = MAE(a["arquitetura"], pesos=False, img_size=a["img"], in_chans=in_chans,
              dec_dim=a["dec_dim"], dec_prof=a["dec_prof"])
    mae.load_state_dict(ck["mae"])
    ultimo_probe = ck["probes"][-1] if ck.get("probes") else None
    destino = os.path.join(pasta, "encoder_epoca%d.pth" % ck["epoca"])
    mae.exportar_encoder(destino, epoca=ck["epoca"],
                         probe_val_miou=ultimo_probe["miou"] if ultimo_probe else None)
    print("OK -> %s (epoca %d%s)" % (destino, ck["epoca"],
                                     ", último probe mIoU=%.4f (epoca %d)"
                                     % (ultimo_probe["miou"], ultimo_probe["epoca"])
                                     if ultimo_probe else ""))


if __name__ == "__main__":
    main()
