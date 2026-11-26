"""Reusable channel, spatial and CBAM attention for convolutional models.

The input gate returns one weight per channel and sample. ``AttentionRouter``
places CBAM after selected feature blocks without tying the module to a model.

Uso (no WSL, a partir de Models/):
  PYTHONPATH=unet_cnn ../Data/.venv/bin/python -c "from attention import CBAM; print(CBAM(32))"
"""

import ambiente  # noqa: F401  (configure caches before importing torch)
import torch
from torch import nn


class ChannelAttention(nn.Module):
    """CBAM channel gate; keeps at least ``min_hidden`` hidden channels."""

    def __init__(self, channels, reduction=16, min_hidden=4):
        super().__init__()
        if channels < 1 or reduction < 1 or min_hidden < 1:
            raise ValueError("channels, reduction and min_hidden must be positive")
        hidden = max(min_hidden, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def weights(self, x):
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        maximum = torch.amax(x, dim=(2, 3), keepdim=True)
        return torch.sigmoid(self.mlp(avg) + self.mlp(maximum))

    def forward(self, x):
        return x * self.weights(x)


class SpatialAttention(nn.Module):
    """CBAM spatial gate applied after the channel gate."""

    def __init__(self, kernel=7):
        super().__init__()
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError("spatial kernel must be positive and odd")
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def weights(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat((avg, maximum), dim=1)))

    def forward(self, x):
        return x * self.weights(x)


class CBAM(nn.Module):
    """Sequential channel and spatial gates for any ``B,C,H,W`` feature map."""

    def __init__(self, channels, reduction=16, kernel=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel=kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


class AttentionRouter(nn.Module):
    """Attach gates to named blocks of any CNN without changing those blocks."""

    def __init__(self, input_channels, node_channels, input_enabled=False,
                 selected_nodes=()):
        super().__init__()
        selected = set(selected_nodes)
        unknown = selected.difference(node_channels)
        if unknown:
            raise ValueError("unknown attention nodes: %s" % sorted(unknown))
        self.input = (ChannelAttention(input_channels) if input_enabled
                      else nn.Identity())
        self.nodes = nn.ModuleDict({name: CBAM(channels)
                                    for name, channels in node_channels.items()
                                    if name in selected})

    def apply_input(self, x):
        return self.input(x)

    def apply_node(self, name, x):
        return self.nodes[name](x) if name in self.nodes else x

    def input_weights(self, x):
        if not isinstance(self.input, ChannelAttention):
            raise RuntimeError("input channel attention is disabled")
        return self.input.weights(x).flatten(1)
