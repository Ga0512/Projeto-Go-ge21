"""U-Net and U-Net++ segmentation models with configurable attention.

Input channels, class count and patch size come from the dataset. Attention
can be placed on input bands or selected encoder and decoder feature blocks.

Uso (no WSL, a partir de Models/):
  PYTHONPATH=unet_cnn ../Data/.venv/bin/python -c "from unet import build_model; print(build_model('unetpp', 7, 3))"
"""

import ambiente  # noqa: F401  (configure caches before importing torch)
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionRouter

ENCODER_NODES = ("e0", "e1", "e2", "e3")
UNET_DECODER_NODES = ("d0", "d1", "d2", "d3")
UNETPP_DECODER_NODES = ("x01", "x11", "x21", "x02", "x12", "x03")
ATTENTION_NODES = ENCODER_NODES + ("bot",) + UNET_DECODER_NODES + UNETPP_DECODER_NODES


def _selected_nodes(encoder, decoder, decoder_nodes, extra):
    selected = set(extra or ())
    if encoder:
        selected.update(ENCODER_NODES)
    if decoder:
        selected.update(decoder_nodes)
    return selected


class ConvBloco(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class Subir(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    """Classic U-Net; channel and class counts are dataset dependent."""

    def __init__(self, in_chans=3, n_classes=4, base=32,
                 attention_input=False, attention_encoder=False,
                 attention_decoder=False, attention_nodes=None,
                 attention_input_mode="softmax"):
        super().__init__()
        node_channels = {"e0": base, "e1": base * 2, "e2": base * 4,
                         "e3": base * 8, "bot": base * 8,
                         "d0": base * 4, "d1": base * 2,
                         "d2": base, "d3": base}
        self.attention = AttentionRouter(
            in_chans, node_channels, attention_input,
            _selected_nodes(attention_encoder, attention_decoder,
                            UNET_DECODER_NODES, attention_nodes),
            input_mode=attention_input_mode)
        self.e0 = ConvBloco(in_chans, base)                            #  32 @ 256
        self.e1 = nn.Sequential(nn.MaxPool2d(2), ConvBloco(base, base * 2))    #  64 @ 128
        self.e2 = nn.Sequential(nn.MaxPool2d(2), ConvBloco(base * 2, base * 4))  # 128 @  64
        self.e3 = nn.Sequential(nn.MaxPool2d(2), ConvBloco(base * 4, base * 8))  # 256 @  32
        self.bot = nn.Sequential(nn.MaxPool2d(2), ConvBloco(base * 8, base * 8)) # 256 @  16
        # decoder: subida (cin->cout) concatena o skip simétrico (canais do encoder)
        self.d0 = nn.Sequential(Subir(base * 8, base * 4), ConvBloco(base * 8 + base * 4, base * 4))  # 128 @ 32
        self.d1 = nn.Sequential(Subir(base * 4, base * 2), ConvBloco(base * 4 + base * 2, base * 2))  #  64 @ 64
        self.d2 = nn.Sequential(Subir(base * 2, base), ConvBloco(base * 2 + base, base))              #  32 @ 128
        self.d3 = nn.Sequential(Subir(base, base), ConvBloco(base + base, base))                      #  32 @ 256
        self.cabeca = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x = self.attention.apply_input(x)
        e0 = self.attention.apply_node("e0", self.e0(x))
        e1 = self.attention.apply_node("e1", self.e1(e0))
        e2 = self.attention.apply_node("e2", self.e2(e1))
        e3 = self.attention.apply_node("e3", self.e3(e2))
        b = self.attention.apply_node("bot", self.bot(e3))
        x = self.attention.apply_node("d0", self.d0[1](torch.cat([self.d0[0](b), e3], dim=1)))
        x = self.attention.apply_node("d1", self.d1[1](torch.cat([self.d1[0](x), e2], dim=1)))
        x = self.attention.apply_node("d2", self.d2[1](torch.cat([self.d2[0](x), e1], dim=1)))
        x = self.attention.apply_node("d3", self.d3[1](torch.cat([self.d3[0](x), e0], dim=1)))
        return self.cabeca(x)

    def input_attention_weights(self, x):
        return self.attention.input_weights(x)


class UNetPP(nn.Module):
    """U-Net++ (Zhou et al., 2018) — skip connections DENSAS entre níveis.

    A diferença para a UNet clássica: no nível i, o nó X_i^j recebe a
    concatenação de TODOS os nós X_i^{0..j-1} do MESMO nível (e o subido do
    nível acima), não só o do encoder. Isso reduz o "semantic gap" entre o
    encoder e o decoder e costuma melhorar principalmente bordas e objetos
    pequenos. Fica mais pesado que a UNet (mais blocos convolucionais).

    Encoder igual à UNet (base -> base*2 -> base*4 -> base*8 -> base*8 no
    bottleneck); o decoder denso tem profundidade 4 (X0..X3).
    """

    def __init__(self, in_chans=3, n_classes=4, base=32,
                 attention_input=False, attention_encoder=False,
                 attention_decoder=False, attention_nodes=None,
                 attention_input_mode="softmax"):
        super().__init__()
        c = [base * (2 ** i) for i in range(5)]          # 32,64,128,256,512 (base 32)
        node_channels = {"e0": c[0], "e1": c[1], "e2": c[2], "e3": c[3],
                         "x01": c[0], "x11": c[1], "x21": c[2],
                         "x02": c[0], "x12": c[1], "x03": c[0]}
        self.attention = AttentionRouter(
            in_chans, node_channels, attention_input,
            _selected_nodes(attention_encoder, attention_decoder,
                            UNETPP_DECODER_NODES, attention_nodes),
            input_mode=attention_input_mode)
        self.e0 = ConvBloco(in_chans, c[0])              # 256
        self.e1 = nn.Sequential(nn.MaxPool2d(2), ConvBloco(c[0], c[1]))    # 128
        self.e2 = nn.Sequential(nn.MaxPool2d(2), ConvBloco(c[1], c[2]))    #  64
        self.e3 = nn.Sequential(nn.MaxPool2d(2), ConvBloco(c[2], c[3]))    #  32
        self.bot = nn.Sequential(nn.MaxPool2d(2), ConvBloco(c[3], c[4]))   #  16

        # nós do decoder denso: X_ij (nível i, profundidade j); X_i0 = encoder
        # X_0^1 = cat(e0, up(X_1^0))         -> c0
        # X_1^1 = cat(e1, up(X_2^0))         -> c1
        # X_2^1 = cat(e2, up(X_3^0))         -> c2
        # X_0^2 = cat(X_0^0, X_0^1, up(X_1^1))      -> c0
        # X_1^2 = cat(X_1^0, X_1^1, up(X_2^1))      -> c1
        # X_0^3 = cat(X_0^0, X_0^1, X_0^2, up(X_1^2)) -> c0
        self.x01 = ConvBloco(c[0] + c[1], c[0])
        self.x11 = ConvBloco(c[1] + c[2], c[1])
        self.x21 = ConvBloco(c[2] + c[3], c[2])
        self.x02 = ConvBloco(c[0] * 2 + c[1], c[0])
        self.x12 = ConvBloco(c[1] * 2 + c[2], c[1])
        self.x03 = ConvBloco(c[0] * 3 + c[1], c[0])
        self.cabeca = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        x = self.attention.apply_input(x)
        x00 = self.attention.apply_node("e0", self.e0(x))
        x10 = self.attention.apply_node("e1", self.e1(x00))
        x20 = self.attention.apply_node("e2", self.e2(x10))
        x30 = self.attention.apply_node("e3", self.e3(x20))
        b = self.bot(x30)

        def up(t):
            return F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)

        x01 = self.attention.apply_node("x01", self.x01(torch.cat([x00, up(x10)], dim=1)))
        x11 = self.attention.apply_node("x11", self.x11(torch.cat([x10, up(x20)], dim=1)))
        x21 = self.attention.apply_node("x21", self.x21(torch.cat([x20, up(x30)], dim=1)))
        x02 = self.attention.apply_node("x02", self.x02(torch.cat([x00, x01, up(x11)], dim=1)))
        x12 = self.attention.apply_node("x12", self.x12(torch.cat([x10, x11, up(x21)], dim=1)))
        x03 = self.attention.apply_node("x03", self.x03(torch.cat([x00, x01, x02, up(x12)], dim=1)))
        return self.cabeca(x03)

    def input_attention_weights(self, x):
        """Return B,C input gate weights without storing activations."""
        return self.attention.input_weights(x)


def build_model(architecture, in_chans, n_classes, base=32,
                attention_input=False, attention_encoder=False,
                attention_decoder=False, attention_nodes=None,
                attention_input_mode="softmax"):
    """Single model factory for training, evaluation and inference."""
    options = (in_chans, n_classes, base, attention_input, attention_encoder,
               attention_decoder, attention_nodes, attention_input_mode)
    if architecture == "unetpp":
        return UNetPP(*options)
    if architecture == "unet":
        return UNet(*options)
    raise ValueError("unknown architecture: %s" % architecture)


def normalized_state_dict(state_dict):
    """Read checkpoints made before the English attention names were adopted."""
    renamed = {}
    for key, value in state_dict.items():
        if key.startswith("atencao_entrada."):
            key = key.replace("atencao_entrada.", "attention.input.", 1)
        elif key.startswith("cbam_encoder.") or key.startswith("cbam_decoder."):
            key = key.replace("cbam_encoder.", "attention.nodes.", 1)
            key = key.replace("cbam_decoder.", "attention.nodes.", 1)
        key = key.replace(".canais.projecao.", ".channel.mlp.")
        key = key.replace(".espacial.", ".spatial.")
        key = key.replace(".projecao.", ".mlp.")
        renamed[key] = value
    return renamed


def model_from_checkpoint(checkpoint, n_classes):
    """Build and strictly load old or new model checkpoints."""
    args = checkpoint.get("args", {})
    state = normalized_state_dict(checkpoint["state_dict"])
    channels = state["e0.net.0.weight"].shape[1]
    model = build_model(
        args.get("architecture", args.get("arquitetura", "unetpp")),
        channels, n_classes,
        args.get("base", 32),
        args.get("attention_input", args.get("atencao_entrada", False)),
        args.get("attention_encoder", args.get("cbam_encoder", False)),
        args.get("attention_decoder", args.get("cbam_decoder", False)),
        args.get("attention_nodes", args.get("cbam_nos")),
        args.get("attention_input_mode", "sigmoid"))
    model.load_state_dict(state, strict=True)
    return model
