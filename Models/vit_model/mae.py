"""Masked Autoencoder (MAE) com encoder ViT do timm.

- Encoder: ViT do timm (padrão ViT-S/16 com pesos DINO), dynamic_img_size=True
  para rodar em qualquer tamanho múltiplo do patch (treino em 224, segmentação
  depois em 256: o pos_embed é reamostrado).
- Canais: os pesos pré-treinados são RGB. As nossas bandas são PAN, GLCM e
  rugosidade; o canal PAN recebe a soma dos filtros RGB (responde à PAN como o
  modelo original responderia a uma imagem em tons de cinza) e os canais de
  textura começam em zero (o treino aprende a usá-los).
- Decoder: leve (MAE), só existe para o pré-treino. exportar_encoder() grava
  apenas o encoder, no formato do timm — o decoder é descartado.
- Loss: MSE só nos patches mascarados, alvo normalizado por patch e por banda
  (norm_pix_loss), reportada também por banda.
"""

import math

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


def sincos_2d(dim, gh, gw):
    """pos-embed senoidal 2D fixo -> (gh*gw, dim)."""
    def eixo(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2))
        out = pos.reshape(-1, 1).double() * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)
    yy, xx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    return torch.cat([eixo(dim // 2, yy), eixo(dim // 2, xx)], dim=1).float()


class MAE(nn.Module):
    def __init__(self, arquitetura="vit_small_patch16_224.dino", pesos=True, img_size=224,
                 in_chans=3, dec_dim=256, dec_prof=4, dec_cabecas=8, norm_pix=True):
        super().__init__()
        self.arquitetura = arquitetura
        self.encoder = timm.create_model(arquitetura, pretrained=pesos, num_classes=0,
                                         img_size=img_size, in_chans=3 if pesos else in_chans,
                                         dynamic_img_size=True)
        if pesos:
            self._adaptar_canais(in_chans)
        enc = self.encoder
        self.p = enc.patch_embed.patch_size[0]
        self.in_chans = in_chans
        self.n_prefixo = enc.num_prefix_tokens
        self.norm_pix = norm_pix
        dim = enc.embed_dim

        self.dec_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_blocos = nn.Sequential(*[
            Block(dec_dim, dec_cabecas, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(dec_prof)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, self.p * self.p * in_chans)
        self._init_decoder()

    # ------------------------------------------------------------ init
    @torch.no_grad()
    def _adaptar_canais(self, in_chans):
        proj = self.encoder.patch_embed.proj
        w = proj.weight                                   # (D, 3, p, p) RGB
        novo = torch.zeros(w.shape[0], in_chans, *w.shape[2:], dtype=w.dtype)
        novo[:, 0] = w.sum(dim=1)                         # PAN = soma RGB
        nova = nn.Conv2d(in_chans, w.shape[0], proj.kernel_size, proj.stride,
                         bias=proj.bias is not None)
        nova.weight.copy_(novo)
        if proj.bias is not None:
            nova.bias.copy_(proj.bias)
        self.encoder.patch_embed.proj = nova

    def _init_decoder(self):
        nn.init.normal_(self.mask_token, std=0.02)
        for m in [self.dec_embed, self.dec_pred, *self.dec_blocos.modules(), self.dec_norm]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------- patches
    def patchify(self, x):
        """(B,C,H,W) -> (B, L, p*p, C)."""
        B, C, H, W = x.shape
        p = self.p
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, (H // p) * (W // p), p * p, C)

    def unpatchify(self, t, gh, gw):
        """(B, L, p*p, C) -> (B,C,H,W)."""
        B, _, _, C = t.shape
        p = self.p
        t = t.reshape(B, gh, gw, p, p, C).permute(0, 5, 1, 3, 2, 4)
        return t.reshape(B, C, gh * p, gw * p)

    @staticmethod
    def mascarar(x, razao, gerador=None):
        """Mantém (1-razao) dos tokens, sorteados por amostra.
        -> tokens visíveis, mask (B,L) 1=escondido, ids_restore."""
        B, L, D = x.shape
        manter = max(1, int(round(L * (1 - razao))))
        ruido = torch.rand(B, L, device=x.device, generator=gerador)
        ids_emb = ruido.argsort(dim=1)
        ids_restore = ids_emb.argsort(dim=1)
        vis = x.gather(1, ids_emb[:, :manter, None].expand(-1, -1, D))
        mask = torch.ones(B, L, device=x.device)
        mask[:, :manter] = 0
        return vis, mask.gather(1, ids_restore), ids_restore

    # --------------------------------------------------------- forward
    def codificar(self, imgs, razao=0.0, gerador=None):
        enc = self.encoder
        x = enc._pos_embed(enc.patch_embed(imgs))        # (B, prefixo + L, D)
        prefixo, x = x[:, :self.n_prefixo], x[:, self.n_prefixo:]
        mask = ids_restore = None
        if razao > 0:
            x, mask, ids_restore = self.mascarar(x, razao, gerador)
        x = enc.norm_pre(torch.cat([prefixo, x], dim=1))
        x = enc.norm(enc.blocks(x))
        return x, mask, ids_restore

    def decodificar(self, lat, ids_restore, gh, gw):
        x = self.dec_embed(lat)
        B, _, D = x.shape
        P = self.n_prefixo
        n_mask = ids_restore.shape[1] - (x.shape[1] - P)
        x_ = torch.cat([x[:, P:], self.mask_token.expand(B, n_mask, -1)], dim=1)
        x_ = x_.gather(1, ids_restore[..., None].expand(-1, -1, D))
        pos = sincos_2d(D, gh, gw).to(x_.device, x_.dtype)
        x = torch.cat([x[:, :P], x_ + pos], dim=1)
        x = self.dec_pred(self.dec_norm(self.dec_blocos(x)))[:, P:]
        return x.reshape(B, gh * gw, self.p * self.p, self.in_chans)

    def alvo(self, imgs):
        """Patches alvo (normalizados se norm_pix) + média/desvio por patch e banda."""
        t = self.patchify(imgs.float())
        if not self.norm_pix:
            return t, None, None
        media = t.mean(dim=2, keepdim=True)
        desvio = (t.var(dim=2, keepdim=True) + 1e-6).sqrt()
        return (t - media) / desvio, media, desvio

    def forward(self, imgs, razao=0.75, gerador=None):
        """-> loss, loss por banda (C,), pred (B,L,p*p,C), mask (B,L)."""
        gh, gw = imgs.shape[2] // self.p, imgs.shape[3] // self.p
        lat, mask, ids_restore = self.codificar(imgs, razao, gerador)
        pred = self.decodificar(lat, ids_restore, gh, gw)
        alvo, _, _ = self.alvo(imgs)
        err = (pred.float() - alvo) ** 2
        por_banda = (err.mean(dim=2) * mask[..., None]).sum(dim=(0, 1)) / mask.sum()
        return por_banda.mean(), por_banda, pred, mask

    @torch.no_grad()
    def reconstruir(self, imgs, pred, mask):
        """Reconstrução na escala original: patches visíveis copiados da
        entrada, escondidos vindos da predição. -> (recon, mascarada)."""
        gh, gw = imgs.shape[2] // self.p, imgs.shape[3] // self.p
        orig, media, desvio = self.alvo(imgs)
        if self.norm_pix:
            orig = orig * desvio + media
            pred = pred.float() * desvio + media
        m = mask[..., None, None].bool()
        recon = torch.where(m, pred.float(), orig)
        mascarada = torch.where(m, torch.full_like(orig, float("nan")), orig)
        return self.unpatchify(recon, gh, gw), self.unpatchify(mascarada, gh, gw)

    @torch.no_grad()
    def tokens(self, imgs):
        """Features por patch (sem máscara): (B, L, D)."""
        return self.codificar(imgs)[0][:, self.n_prefixo:]

    def exportar_encoder(self, caminho, **meta):
        """Grava só o encoder (formato timm). Para carregar:
        enc = timm.create_model(arquitetura, pretrained=False, num_classes=0,
                                img_size=img_size, in_chans=in_chans,
                                dynamic_img_size=True)
        enc.load_state_dict(torch.load(caminho)["state_dict"])"""
        torch.save({"arquitetura": self.arquitetura, "in_chans": self.in_chans,
                    "img_size": list(self.encoder.patch_embed.img_size),
                    "state_dict": self.encoder.state_dict(), **meta}, caminho)


def grupos_parametros(mae, lr, lr_encoder_mult, wd):
    """AdamW: sem weight decay em bias/norm/tokens/pos_embed; encoder com lr menor."""
    grupos = {}
    for nome, p in mae.named_parameters():
        parte = "encoder" if nome.startswith("encoder.") else "decoder"
        sem_wd = p.ndim <= 1 or nome.endswith(("pos_embed", "cls_token", "mask_token"))
        chave = (parte, sem_wd)
        if chave not in grupos:
            grupos[chave] = {"params": [], "weight_decay": 0.0 if sem_wd else wd,
                             "lr_mult": lr_encoder_mult if parte == "encoder" else 1.0,
                             "parte": parte, "lr": lr}
        grupos[chave]["params"].append(p)
    return list(grupos.values())


def lr_cosseno(passo, total, aquecimento, lr):
    if passo < aquecimento:
        return lr * (passo + 1) / aquecimento
    t = (passo - aquecimento) / max(1, total - aquecimento)
    return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
