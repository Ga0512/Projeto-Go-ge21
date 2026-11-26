#!/usr/bin/env python3
"""Teste curto da UNet++ com atenção na GPU local, sem carregar datasets.

Verifica dimensões, gradientes nas duas atenções, pesos auditáveis e leitura
de checkpoints novos e antigos. Não executa um treino completo.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/testar_cbam.py
"""

import os
import tempfile

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch)
import torch
import torch.nn.functional as F

from unet import criar_modelo, modelo_do_checkpoint
from inferencia_cena import carregar_modelo


def main():
    dev = ambiente.exigir_gpu()
    torch.manual_seed(42)
    modelo = criar_modelo("unetpp", 7, 3, base=32,
                          atencao_entrada=True, cbam_decoder=True).to(dev)
    x = torch.randn(1, 7, 256, 256, device=dev)
    y = torch.randint(0, 3, (1, 256, 256), device=dev)
    modelo.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = modelo(x)
        loss = F.cross_entropy(pred.float(), y)
    loss.backward()
    assert pred.shape == (1, 3, 256, 256)
    assert torch.isfinite(loss)
    for prefixo in ("atencao_entrada.", "cbam_decoder."):
        gradientes = [p.grad for nome, p in modelo.named_parameters()
                      if nome.startswith(prefixo)]
        assert gradientes and any(g is not None and bool(g.abs().sum() > 0)
                                 for g in gradientes), prefixo
    modelo.eval()
    with torch.inference_mode():
        pesos = modelo.pesos_entrada(x)
    assert pesos.shape == (1, 7)
    assert bool(((pesos > 0) & (pesos < 1)).all())
    ck = {"args": {"arquitetura": "unetpp", "base": 32,
                   "atencao_entrada": True, "cbam_decoder": True},
          "state_dict": {k: v.cpu() for k, v in modelo.state_dict().items()}}
    novo = modelo_do_checkpoint(ck, 3)
    novo.load_state_dict(ck["state_dict"], strict=True)
    antigo = criar_modelo("unetpp", 7, 3, base=32)
    ck_antigo = {"args": {"arquitetura": "unetpp", "base": 32},
                 "state_dict": antigo.state_dict()}
    modelo_do_checkpoint(ck_antigo, 3).load_state_dict(ck_antigo["state_dict"],
                                                       strict=True)
    cache = os.path.join(os.path.dirname(__file__), "..", ".cache")
    os.makedirs(cache, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as pasta:
        caminho = os.path.join(pasta, "modelo_melhor.pth")
        torch.save(ck, caminho)
        inferido, _, _, _ = carregar_modelo(caminho, "cpu", 3)
        assert inferido.pesos_entrada(torch.zeros(1, 7, 32, 32)).shape == (1, 7)
    print("OK: GPU %s | loss %.4f | pesos %s | checkpoints novo/antigo/inferência"
          % (torch.cuda.get_device_name(dev), loss.item(),
             [round(v, 3) for v in pesos[0].tolist()]), flush=True)


if __name__ == "__main__":
    main()
