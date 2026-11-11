"""Segmentação: encoder ViT (timm) + decoder estilo UPerNet (PPM + fusão densa).

O ViT-S/16 vê a imagem em tokens de 16x16 px (16x16 tokens num patch de 256).
Todas as features intermediárias do encoder ficam em 16x16 px; o decoder faz
uma fusão densa no estilo UPerNet: um PPM (Pyramid Pooling Module) captura o
contexto global no nível mais fundo (bloco 12) e os níveis mais rasos (blocos
2, 5, 8) são concatenados de cima para baixo; o mapa resultante sobe para 256
e é somado a um ramo convolucional raso que devolve os detalhes de borda que
os tokens de 16 px perdem.

criar_encoder(): "dino" (pesos DINO do timm com a mesma adaptação de canais do
MAE: PAN = soma RGB, texturas = 0), "aleatorio" (sem pré-treino) ou o caminho
de um encoder.pth exportado pelo MAE.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def criar_encoder(fonte, arquitetura="vit_small_patch16_224.dino", img_size=224, in_chans=3):
    """-> encoder timm (dynamic_img_size=True) e descrição da origem."""
    if fonte == "dino":
        from mae import MAE
        return MAE(arquitetura, pesos=True, img_size=img_size, in_chans=in_chans).encoder, \
            "%s (pesos DINO, canais adaptados)" % arquitetura
    if fonte == "aleatorio":
        return timm.create_model(arquitetura, pretrained=False, num_classes=0, img_size=img_size,
                                 in_chans=in_chans, dynamic_img_size=True), \
            "%s (sem pré-treino)" % arquitetura
    ck = torch.load(fonte, map_location="cpu", weights_only=False)
    enc = timm.create_model(ck["arquitetura"], pretrained=False, num_classes=0,
                            img_size=tuple(ck["img_size"]), in_chans=ck["in_chans"],
                            dynamic_img_size=True)
    enc.load_state_dict(ck["state_dict"])
    return enc, "%s (%s, epoca %s)" % (ck["arquitetura"], fonte, ck.get("epoca"))


class ConvBloco(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SegViT(nn.Module):
    """Segmentador: encoder ViT (timm) + decoder estilo UPerNet (PPM + FPN).

    O decoder é o UPerHead do UPerNet (MMSegmentation): as features de 4
    profundidades do encoder (blocos 2, 5, 8, 11, todas em 16x16 px) passam por
    convs 1x1 laterais, o nível mais fundo (z12) vira o "P5" e é processado por
    um PPM (Pyramid Pooling Module) que captura contexto global em 4 escalas; o
    top-down do FPN funde isso de volta às resoluções menores. No fim o mapa em
    16x16 sobe para 256 e é somado a um ramo convolucional raso (256) que
    devolve os detalhes de borda que os tokens de 16 px perdem."""

    def __init__(self, encoder, n_classes, in_chans=3, blocos=(2, 5, 8, 11),
                 canais=(256, 128, 64, 32, 16), ppm_ramos=(1, 2, 3, 6)):
        super().__init__()
        self.encoder = encoder
        self.encoder_congelado = False
        self.blocos = list(blocos)
        D = encoder.embed_dim
        c256, c128, c64, c32, c16 = canais   # do nível mais fundo ao mais raso
        self.f12 = nn.Sequential(nn.Conv2d(D, c256, 1, bias=False), nn.BatchNorm2d(c256),
                                 nn.ReLU(inplace=True))
        self.f9 = nn.Sequential(nn.Conv2d(D, c128, 1, bias=False), nn.BatchNorm2d(c128),
                                nn.ReLU(inplace=True))
        self.f6 = nn.Sequential(nn.Conv2d(D, c64, 1, bias=False), nn.BatchNorm2d(c64),
                                nn.ReLU(inplace=True))
        self.f3 = nn.Sequential(nn.Conv2d(D, c32, 1, bias=False), nn.BatchNorm2d(c32),
                                nn.ReLU(inplace=True))
        # PPM sobre o nível mais fundo (z12): contexto global multiescala.
        # Todos os níveis do ViT ficam em 16x16, então o PPM age no z12.
        self.ppm = nn.ModuleList()
        for b in ppm_ramos:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(b),
                                          nn.Conv2d(D, c256, 1, bias=False),
                                          nn.BatchNorm2d(c256), nn.ReLU(inplace=True)))
        self.ppm_fusao = nn.Sequential(nn.Conv2d(c256 * (len(ppm_ramos) + 1), c256, 1, bias=False),
                                       nn.BatchNorm2d(c256), nn.ReLU(inplace=True))
        self.raso = ConvBloco(in_chans, c16)
        # fusão densa em 16x16 (UPerNet): cada nível soma o nível anterior
        self.d3 = ConvBloco(c256 + c128, c128)
        self.d6 = ConvBloco(c128 + c64, c64)
        self.d9 = ConvBloco(c64 + c32, c32)
        # subida para 256 + ramo raso (bordas)
        self.d256 = ConvBloco(c32 + c16, c16)
        self.cabeca = nn.Conv2d(c16, n_classes, 1)

    def congelar_encoder(self):
        """Encoder fixo: sem gradiente e sempre em modo eval (só o decoder aprende)."""
        self.encoder_congelado = True
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def train(self, modo=True):
        super().train(modo)
        if self.encoder_congelado:
            self.encoder.eval()
        return self

    def forward(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.encoder_congelado):
            z3, z6, z9, z12 = self.encoder.forward_intermediates(
                x, indices=self.blocos, output_fmt="NCHW", intermediates_only=True)

        _, _, H, W = z12.shape
        # PPM no mais fundo: cada ramo volta à resolução original antes da fusão
        ramos = [F.interpolate(r(z12), (H, W), mode="bilinear", align_corners=False)
                 for r in self.ppm]
        y = self.ppm_fusao(torch.cat(ramos + [self.f12(z12)], dim=1))   # c256 @16x16
        y = self.d3(torch.cat([y, self.f9(z9)], dim=1))                 # c128 @16x16
        y = self.d6(torch.cat([y, self.f6(z6)], dim=1))                 # c64  @16x16
        y = self.d9(torch.cat([y, self.f3(z3)], dim=1))                 # c32  @16x16
        y = F.interpolate(y, scale_factor=16, mode="bilinear", align_corners=False)  # 256x256
        y = self.d256(torch.cat([y, self.raso(x)], dim=1))              # c16  @256x256
        return self.cabeca(y)
