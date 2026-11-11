#!/usr/bin/env python3
"""Modelo de segmentação semântica leve (U-Net-ish) p/ patches 512x512x4.

Estrutura simples e treinável em CPU: encoder Conv-ReLU-MaxPool em 4 estágios,
bottleneck, decoder com upsampling + jump concat, conv final 1x1.
"""

import torch
import torch.nn as nn

import config


def _conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class _SE(nn.Module):
    """Squeeze-and-Excitation (channel attention) — recalibra canais pelo
    contexto global; ajuda a detectar alvos pequenos (água)."""

    def __init__(self, c, r=8):
        super().__init__()
        h = max(c // r, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, h, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class UNetLeve(nn.Module):
    def __init__(self, n_bandas=config.N_BANDAS, n_classes=config.N_CLASSES,
                 n0=32, se=True):
        super().__init__()
        c = [n0, n0 * 2, n0 * 4, n0 * 8]
        self.enc1 = _conv_block(n_bandas, c[0])
        self.enc2 = _conv_block(c[0], c[1])
        self.enc3 = _conv_block(c[1], c[2])
        self.enc4 = _conv_block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _conv_block(c[3], c[3])
        self.se4 = _SE(c[3]) if se else nn.Identity()
        self.seb = _SE(c[3]) if se else nn.Identity()
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = _conv_block(c[3] + c[3], c[2])
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _conv_block(c[2] + c[2], c[1])
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _conv_block(c[1] + c[1], c[0])
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _conv_block(c[0] + c[0], c[0])
        self.out = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.se4(self.enc4(self.pool(e3)))
        b = self.seb(self.bottleneck(self.pool(e4)))
        d = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d = self.dec3(torch.cat([self.up3(d), e3], dim=1))
        d = self.dec2(torch.cat([self.up2(d), e2], dim=1))
        d = self.dec1(torch.cat([self.up1(d), e1], dim=1))
        return self.out(d)


def _conv_block_s2(in_c, out_c):
    """Igual a _conv_block, mas a 1a conv tem stride 2 (downsample embutido)."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class _UNetPPDecoder(nn.Module):
    """Decoder U-Net++ (N=4, skip connections densas + deep supervision).

    Nós X^{i,j} (i=profundidade, j=nivel de resolucao 2^j):
      X^{i,0} = features do encoder; X^{0,0} = entrada (n_bandas).
      X^{i,j} = H(concat[X^{i,0..j-1}, up(X^{i+1,j-1})]),  i+j <= 4.
    Saidas em resolucao cheia X^{0,1..4} com heads 1x1 (deep supervision);
    a predicao final e X^{0,4} (sups[-1]).
    """

    def __init__(self, n_bandas, n_classes, c, se=True, sup=True):
        super().__init__()
        self.n_bandas = n_bandas
        self.c = c
        self.sup_on = sup
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=False)
        self.nodes = nn.ModuleDict()
        for j in range(1, 5):
            for i in range(0, 5 - j):
                in_c = sum(self._canal(i, k) for k in range(j)) + \
                    self._canal(i + 1, j - 1)
                self.nodes["%d_%d" % (i, j)] = _conv_block(in_c, self._canal(i, j))
        if sup:
            self.sup = nn.ModuleDict({
                "%d" % j: nn.Conv2d(self._canal(0, j), n_classes, 1)
                for j in range(1, 5)})

    def _canal(self, i, k):
        """Canais do no X^{i,k}: encoder na coluna k=0; nos da coluna k>0
        com out = c[k-1], cap 128 nos nos em resolucao cheia (i=0)."""
        if k == 0:
            return self.n_bandas if i == 0 else self.c[i - 1]
        return min(self.c[k - 1], 128) if i == 0 else self.c[k - 1]

    def forward(self, enc, x0):
        """enc: dict "i_0" -> tensor do encoder (i=1..4); x0: entrada."""
        X = {"0_0": x0}
        for i in range(1, 5):
            X["%d_0" % i] = enc["%d_0" % i]
        for j in range(1, 5):
            for i in range(0, 5 - j):
                srcs = [X["%d_%d" % (i, k)] for k in range(j)]
                srcs.append(self.up(X["%d_%d" % (i + 1, j - 1)]))
                X["%d_%d" % (i, j)] = self.nodes["%d_%d" % (i, j)](
                    torch.cat(srcs, dim=1))
        if self.sup_on:
            sups = [self.sup["%d" % j](X["0_%d" % j]) for j in range(1, 5)]
            return sups[-1], sups
        return X["0_4"], None


class UNetPP(nn.Module):
    """U-Net++ (deep supervision, N=4) treinado do zero (sem backbone).

    Encoder Conv-ReLU-MaxPool em 4 estagios (stride 2,4,8,16) + SE nos 2
    ultimos; decoder denso (dense skip connections) com heads 1x1."""

    def __init__(self, n_bandas=config.N_BANDAS, n_classes=config.N_CLASSES,
                 n0=32, se=True, sup=True):
        super().__init__()
        c = [n0, n0 * 2, n0 * 4, n0 * 8]
        self.enc1 = _conv_block_s2(n_bandas, c[0])
        self.enc2 = _conv_block(c[0], c[1])
        self.enc3 = _conv_block(c[1], c[2])
        self.enc4 = _conv_block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.se3 = _SE(c[2]) if se else nn.Identity()
        self.se4 = _SE(c[3]) if se else nn.Identity()
        self.decoder = _UNetPPDecoder(n_bandas, n_classes, c, se, sup)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.se3(self.enc3(self.pool(e2)))
        e4 = self.se4(self.enc4(self.pool(e3)))
        enc = {"1_0": e1, "2_0": e2, "3_0": e3, "4_0": e4}
        return self.decoder(enc, x)


class _ResBasicBlock(nn.Module):
    """Bloco basico ResNet18/34 (2 convs 3x3 + shortcut).

    Nomes iguais aos do torchvision (conv1/bn1/conv2/bn2/downsample)
    para o load_state_dict dos pesos ImageNet casar."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c))

    def forward(self, x):
        idt = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            idt = self.downsample(idt)
        return torch.relu(out + idt)


def _resnet_stage(in_c, out_c, n_blocos, stride=1):
    blocos = [_ResBasicBlock(in_c, out_c, stride)]
    blocos += [_ResBasicBlock(out_c, out_c) for _ in range(n_blocos - 1)]
    return nn.Sequential(*blocos)


RESNET_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
}
RESNET_LAYERS = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3]}


class UNetPPResNet(nn.Module):
    """U-Net++ com encoder ResNet18/34 pre-treinado (ImageNet).

    Niveis do encoder (stride entre parenteses):
      X^{1,0}: conv1 7x7 stride 2 (s2)   — pre-treinado (adaptado p/ 4 canais)
      X^{2,0}: maxpool + layer1 (s4)     — pre-treinado
      X^{3,0}: layer2 (s8)               — pre-treinado
      X^{4,0}: layer3 (s16)              — pre-treinado
    Decoder U-Net++ com deep supervision (X^{0,1..4}).

    ATENCAO (honestidade): os pesos ImageNet sao de RGB; nossa entrada e
    PAN + features (GLCM etc.), nao RGB. A 1a conv recebe 4 canais
    (4o canal = media dos 3 originais) e o transfer e PARCIAL — ajuda via
    deteccao de bordas/texturas de baixo nivel, nao via semantica de cena.

    Os pesos sao baixados do servidor oficial do PyTorch (torch.hub) na
    1a execucao (cache em ~/.cache/torch). Se o download falhar, inicializa
    aleatoriamente com aviso (o treino funciona, sem transfer).
    """

    def __init__(self, n_bandas=config.N_BANDAS, n_classes=config.N_CLASSES,
                 backbone="resnet34", pretrained=True, se=True, sup=True):
        super().__init__()
        assert backbone in RESNET_URLS, "backbone deve ser resnet18|resnet34"
        layers = RESNET_LAYERS[backbone]
        c = [64, 64, 128, 256]

        self.conv1 = nn.Conv2d(n_bandas, 64, 7, 2, 3, bias=False)  # s2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = _resnet_stage(64, 64, layers[0])
        self.layer2 = _resnet_stage(64, 128, layers[1], stride=2)   # s8
        self.layer3 = _resnet_stage(128, 256, layers[2], stride=2)  # s16
        self.se3 = _SE(c[2]) if se else nn.Identity()
        self.se4 = _SE(c[3]) if se else nn.Identity()
        self.decoder = _UNetPPDecoder(n_bandas, n_classes, c, se, sup)

        if pretrained:
            self._carregar_pretrained(backbone)

    def _carregar_pretrained(self, backbone):
        try:
            import torch.hub
            sd = torch.hub.load_state_dict_from_url(
                RESNET_URLS[backbone], map_location="cpu", weights_only=True)
        except Exception as e:
            print("[aviso] falha ao baixar pesos pre-treinados %s (%s); "
                  "inicializacao aleatoria" % (backbone, e))
            return
        sd.pop("fc.weight", None)
        sd.pop("fc.bias", None)
        w = sd["conv1.weight"]  # (64,3,7,7) -> 4 canais
        sd["conv1.weight"] = torch.cat([w, w.mean(dim=1, keepdim=True)], dim=1)
        self.load_state_dict(sd, strict=False)
        print("[modelo] encoder %s com pesos ImageNet (4o canal conv1 = "
              "media dos RGB)" % backbone)

    def forward(self, x):
        e1 = self.relu(self.bn1(self.conv1(x)))       # s2 (64)
        e2 = self.layer1(self.maxpool(e1))            # s4 (64)
        e3 = self.se3(self.layer2(e2))                # s8 (128)
        e4 = self.se4(self.layer3(e3))                # s16 (256)
        enc = {"1_0": e1, "2_0": e2, "3_0": e3, "4_0": e4}
        return self.decoder(enc, x)