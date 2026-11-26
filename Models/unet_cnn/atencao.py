"""Blocos de atenção reutilizáveis para mapas de características de CNN.

CBAM aplica atenção de canais seguida de atenção espacial. A etapa de canais
também pode ser usada sozinha nas bandas de entrada. Os pesos são calculados
para cada imagem do lote; ``AtencaoCanais.pesos`` permite auditá-los depois.

Uso (no WSL, a partir de Models/):
  PYTHONPATH=unet_cnn ../Data/.venv/bin/python -c "from atencao import CBAM; print(CBAM(32))"
"""

import ambiente  # noqa: F401  (caches -> D:; precisa vir antes de torch)
import torch
from torch import nn


class AtencaoCanais(nn.Module):
    """Porta CBAM de canais; preserva pelo menos ``min_ocultos`` unidades."""

    def __init__(self, canais, reducao=16, min_ocultos=4):
        super().__init__()
        if canais < 1 or reducao < 1 or min_ocultos < 1:
            raise ValueError("canais, reducao e min_ocultos precisam ser positivos")
        ocultos = max(min_ocultos, canais // reducao)
        self.projecao = nn.Sequential(
            nn.Conv2d(canais, ocultos, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ocultos, canais, 1, bias=False),
        )

    def pesos(self, x):
        media = torch.mean(x, dim=(2, 3), keepdim=True)
        maximo = torch.amax(x, dim=(2, 3), keepdim=True)
        return torch.sigmoid(self.projecao(media) + self.projecao(maximo))

    def forward(self, x):
        return x * self.pesos(x)


class AtencaoEspacial(nn.Module):
    """Porta CBAM espacial após a seleção de canais."""

    def __init__(self, kernel=7):
        super().__init__()
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError("kernel espacial precisa ser positivo e impar")
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def pesos(self, x):
        media = torch.mean(x, dim=1, keepdim=True)
        maximo = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat((media, maximo), dim=1)))

    def forward(self, x):
        return x * self.pesos(x)


class CBAM(nn.Module):
    """Atenção sequencial de canais e espaço para qualquer mapa ``B,C,H,W``."""

    def __init__(self, canais, reducao=16, kernel=7):
        super().__init__()
        self.canais = AtencaoCanais(canais, reducao=reducao)
        self.espacial = AtencaoEspacial(kernel=kernel)

    def forward(self, x):
        return self.espacial(self.canais(x))
