"""U-Net CNN para os patches SPOT5 (256x256, 3 bandas, 4 classes).

Arquitetura clássica: encoder com 4 níveis (2x Conv+BN+ReLU + MaxPool) que
reduz 256 -> 16 px, bottleneck, e decoder simétrico com ConvTranspose (2x)
concatenando as skip connections do encoder. Base 32 -> canais 32/64/128/256.

Serve para comparar com o ViT (seg_mae_v3_*) e o XGBoost no mesmo dataset:
mesmo treino, mesma augmentation e mesmas métricas (interior).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from atencao import AtencaoCanais, CBAM

NOS_ENCODER = ("e0", "e1", "e2", "e3")
NOS_DECODER = ("x01", "x11", "x21", "x02", "x12", "x03")
NOS_CBAM = NOS_ENCODER + NOS_DECODER


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
    """in_chans=3 (PAN, GLCM, rugosidade), n_classes=4 (CLASS_NOMES)."""

    def __init__(self, in_chans=3, n_classes=4, base=32):
        super().__init__()
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
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b = self.bot(e3)
        x = self.d0[1](torch.cat([self.d0[0](b), e3], dim=1))
        x = self.d1[1](torch.cat([self.d1[0](x), e2], dim=1))
        x = self.d2[1](torch.cat([self.d2[0](x), e1], dim=1))
        x = self.d3[1](torch.cat([self.d3[0](x), e0], dim=1))
        return self.cabeca(x)


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
                 atencao_entrada=False, cbam_decoder=False,
                 cbam_encoder=False, cbam_nos=None):
        super().__init__()
        c = [base * (2 ** i) for i in range(5)]          # 32,64,128,256,512 (base 32)
        selecionados = set(cbam_nos or ())
        desconhecidos = selecionados.difference(NOS_CBAM)
        if desconhecidos:
            raise ValueError("nós CBAM desconhecidos: %s" % sorted(desconhecidos))
        if cbam_encoder:
            selecionados.update(NOS_ENCODER)
        if cbam_decoder:
            selecionados.update(NOS_DECODER)
        self.atencao_entrada = (AtencaoCanais(in_chans) if atencao_entrada
                                else nn.Identity())
        canais_nos = {"e0": c[0], "e1": c[1], "e2": c[2], "e3": c[3],
                      "x01": c[0], "x11": c[1], "x21": c[2],
                      "x02": c[0], "x12": c[1], "x03": c[0]}
        self.cbam_encoder = nn.ModuleDict({n: CBAM(canais_nos[n])
                                           for n in NOS_ENCODER if n in selecionados})
        self.cbam_decoder = nn.ModuleDict({n: CBAM(canais_nos[n])
                                           for n in NOS_DECODER if n in selecionados})
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
        x = self.atencao_entrada(x)
        x00 = self._refinar("e0", self.e0(x))
        x10 = self._refinar("e1", self.e1(x00))
        x20 = self._refinar("e2", self.e2(x10))
        x30 = self._refinar("e3", self.e3(x20))
        b = self.bot(x30)

        def up(t):
            return F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)

        x01 = self._refinar("x01", self.x01(torch.cat([x00, up(x10)], dim=1)))
        x11 = self._refinar("x11", self.x11(torch.cat([x10, up(x20)], dim=1)))
        x21 = self._refinar("x21", self.x21(torch.cat([x20, up(x30)], dim=1)))
        x02 = self._refinar("x02", self.x02(torch.cat([x00, x01, up(x11)], dim=1)))
        x12 = self._refinar("x12", self.x12(torch.cat([x10, x11, up(x21)], dim=1)))
        x03 = self._refinar("x03", self.x03(torch.cat([x00, x01, x02, up(x12)], dim=1)))
        return self.cabeca(x03)

    def _refinar(self, nome, x):
        if nome in self.cbam_encoder:
            return self.cbam_encoder[nome](x)
        return self.cbam_decoder[nome](x) if nome in self.cbam_decoder else x

    def pesos_entrada(self, x):
        """Pesos B,C da entrada; não altera o estado nem guarda ativações."""
        if not isinstance(self.atencao_entrada, AtencaoCanais):
            raise RuntimeError("este modelo não tem atenção na entrada")
        return self.atencao_entrada.pesos(x).flatten(1)


def criar_modelo(arquitetura, in_chans, n_classes, base=32,
                 atencao_entrada=False, cbam_decoder=False,
                 cbam_encoder=False, cbam_nos=None):
    """Fábrica única para treino, avaliação e inferência de checkpoints."""
    if arquitetura == "unetpp":
        return UNetPP(in_chans, n_classes, base, atencao_entrada, cbam_decoder,
                      cbam_encoder, cbam_nos)
    if arquitetura == "unet" and not (atencao_entrada or cbam_decoder or
                                      cbam_encoder or cbam_nos):
        return UNet(in_chans, n_classes, base)
    raise ValueError("atenção só é suportada em --arquitetura unetpp")


def modelo_do_checkpoint(ck, n_classes):
    """Reconstitui a arquitetura com as opções registradas pelo train.py."""
    a = ck.get("args", {})
    canais = ck["state_dict"]["e0.net.0.weight"].shape[1]
    return criar_modelo(a.get("arquitetura", "unetpp"), canais, n_classes,
                        a.get("base", 32), a.get("atencao_entrada", False),
                        a.get("cbam_decoder", False), a.get("cbam_encoder", False),
                        a.get("cbam_nos"))
