#!/usr/bin/env python3
"""Exercise configurable attention in U-Net and U-Net++ on the local GPU.

Checks backward gradients, selected nodes, input weights, inference loading,
old checkpoints and the checkpoint format published before the name cleanup.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/smoke_test_attention.py
"""

import os
import tempfile

import ambiente  # noqa: F401  (configure caches before importing torch)
import torch
import torch.nn.functional as F

from inferencia_cena import carregar_modelo
from unet import build_model, model_from_checkpoint


def train_step(architecture, device, **attention):
    model = build_model(architecture, 7, 3, base=32, **attention).to(device).train()
    image = torch.randn(1, 7, 256, 256, device=device)
    label = torch.randint(0, 3, (1, 256, 256), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(image)
        loss = F.cross_entropy(prediction.float(), label)
    loss.backward()
    assert prediction.shape == (1, 3, 256, 256) and torch.isfinite(loss)
    for prefix in ("attention.input.", "attention.nodes."):
        gradients = [p.grad for name, p in model.named_parameters()
                     if name.startswith(prefix)]
        if gradients:
            assert any(g is not None and bool(g.abs().sum() > 0)
                       for g in gradients), prefix
    return model, image, float(loss.detach())


def checkpoint(model, architecture, **attention):
    return {"args": {"architecture": architecture, "base": 32, **attention},
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}


def legacy_checkpoint(current):
    """Simulate attention checkpoints created before English names were used."""
    old = {}
    for key, value in current["state_dict"].items():
        if key.startswith("attention.input."):
            key = key.replace("attention.input.", "atencao_entrada.", 1)
        elif key.startswith("attention.nodes."):
            node = key.split(".")[2]
            group = "cbam_encoder" if node.startswith("e") else "cbam_decoder"
            key = key.replace("attention.nodes.", group + ".", 1)
        key = key.replace(".channel.mlp.", ".canais.projecao.")
        key = key.replace(".spatial.", ".espacial.")
        key = key.replace(".mlp.", ".projecao.")
        old[key] = value
    return {"args": {"arquitetura": "unetpp", "base": 32,
                     "atencao_entrada": True, "cbam_decoder": True},
            "state_dict": old}


def main():
    device = ambiente.exigir_gpu()
    torch.manual_seed(42)
    unetpp, image, first_loss = train_step(
        "unetpp", device, attention_input=True, attention_decoder=True)
    with torch.inference_mode():
        weights = unetpp.input_attention_weights(image)
    assert weights.shape == (1, 7) and bool(((weights > 0) & (weights < 1)).all())
    new = checkpoint(unetpp, "unetpp", attention_input=True,
                     attention_decoder=True)
    model_from_checkpoint(new, 3)
    model_from_checkpoint(legacy_checkpoint(new), 3)
    unetpp_selective = build_model("unetpp", 7, 3, attention_nodes=["e1", "x03"])
    assert set(unetpp_selective.attention.nodes) == {"e1", "x03"}

    cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as directory:
        path = os.path.join(directory, "model.pth")
        torch.save(new, path)
        loaded, _, _, _ = carregar_modelo(path, "cpu", 3)
        assert loaded.input_attention_weights(torch.zeros(1, 7, 32, 32)).shape == (1, 7)

    del unetpp, image
    torch.cuda.empty_cache()
    unet, _, second_loss = train_step("unet", device, attention_encoder=True)
    assert set(unet.attention.nodes) == {"e0", "e1", "e2", "e3"}
    model_from_checkpoint(checkpoint(unet, "unet", attention_encoder=True), 3)
    unet_selective = build_model("unet", 7, 3, attention_nodes=["bot", "d3"])
    assert set(unet_selective.attention.nodes) == {"bot", "d3"}
    for architecture in ("unet", "unetpp"):
        baseline = checkpoint(build_model(architecture, 7, 3), architecture)
        model_from_checkpoint(baseline, 3)
    print("OK: %s | U-Net++ loss %.4f | U-Net loss %.4f | 7 input weights | old/new checkpoints"
          % (torch.cuda.get_device_name(device), first_loss, second_loss), flush=True)


if __name__ == "__main__":
    main()
