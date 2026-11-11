#!/usr/bin/env python3
"""Texturas da cena inteira, na grade NATIVA do PAN, em DOIS GeoTIFFs de 1 banda
gravados ao lado do PAN (em original/):

  <pan>_glcm.tif         GLCM entropia (janela 7, 8 níveis, offset (0,1)), z-score
  <pan>_rugosidade.tif   média de z(LocalSTD 7x7) e z(Gradiente Sobel), z-score

São as bandas 2 e 3 que o dataset usa (a 1 é a PAN). As fórmulas e as
estatísticas por cena (p2/p98 do esticamento, média/desvio do z-score) são as
MESMAS do gerar_dataset_refinado.py: quando a cena está na receita do dataset
e não foi reamostrada, os valores saem iguais aos dos patches de treino.

Roda na GPU por padrão (torch); `--cpu` usa o caminho antigo (scipy). Medido na
013-014 (29488 px de largura, bloco de 1024 linhas): GPU 8,3 s por bloco contra
139 s na CPU, 17x mais rápido, com 1,9 GB de VRAM. Diferença GPU x CPU:
gradiente idêntico, GLCM ~1e-6, LocalSTD ~6e-3 (arredondamento float32).

Processa em blocos de linhas com margem refletida (memória controlada; o bloco
é ajustado à largura da cena para caber na VRAM). Saída Float32, BIGTIFF,
nodata -9999 onde o PAN é nodata.

Cada bloco gravado fecha o diretorio do TIFF (flush do dataset) e anota a
linha em <pan>_features_progresso.json, entao uma rodada interrompida (p.ex.
morta por falta de memoria) deixa o arquivo legivel ate onde chegou e
`--retomar` continua de la em vez de refazer tudo.

Uso:
  .venv/bin/python scripts/spot5/gerar_features_cena.py --pan <PAN.tif>
  .venv/bin/python scripts/spot5/gerar_features_cena.py --todos
  .venv/bin/python scripts/spot5/gerar_features_cena.py --todos --cpu
  .venv/bin/python scripts/spot5/gerar_features_cena.py --pan <PAN.tif> --retomar
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.ndimage as ndi
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wv1"))
import caminhos  # noqa: E402
from gerar_features_pan import features_block  # noqa: E402

gdal.UseExceptions()

WINDOW = 7
LEVELS = 8
LADO_EST = 512      # lado do bloco das estatísticas
N_EST = 64
SEED = 42
NODATA = -9999.0
VRAM_ALVO = 2.5e9   # bytes de VRAM mirados por bloco
# --byte: quantizacao linear p/ Byte (codigo 0 = sem imagem, 1..255 = valor).
# offset/escala vao no proprio GeoTIFF (SetOffset/SetScale), entao QGIS e
# gdal_translate -unscale mostram o valor real sem conta nenhuma.
# Faixas medidas nas cenas ja geradas: glcm -1,7..3,8 | rugosidade -1,2..12,5.
QUANT = {"glcm": (-6.0, 0.05), "rug": (-4.0, 0.15)}   # (offset, escala)
# Qual dataset fornece as estatisticas por cena. IMPORTA: o GLCM quantiza a
# PAN esticada em 8 niveis, entao um p2/p98 diferente joga pixels para outro
# nivel e a entropia muda. Medido em colinas: texturas geradas com o v15
# (p2=20, p98=165) contra os patches do v18 (p2=24, p98=167) dao correlacao
# 0,87 no GLCM -- parecem certas e nao sao. A rugosidade sobrevive (0,9998)
# porque o z-score absorve mudanca de escala continua; a quantizacao discreta
# do GLCM nao. Use --receita para apontar o dataset do modelo que vai consumir
# as bandas, e --sufixo para nao sobrescrever as de outra versao.
RECEITA = "dados/datasets/spot5_refinado_v15/receita.json"
SUFIXO = ""


# --------------------------------------------------------------- GPU (torch)
def _box_gpu(x, w, F):
    """Média em janela w x w com borda refletida (= ndi.uniform_filter)."""
    import torch
    p = w // 2
    k = torch.ones(1, 1, 1, w, device=x.device, dtype=x.dtype) / w
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), k)
    return F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), k.view(1, 1, w, 1))


def features_gpu(pad_strip, rows, janela, niveis, dev):
    """Mesmas saídas do features_block (std, ent, grad), calculadas na GPU."""
    import torch
    import torch.nn.functional as F
    half = janela // 2
    im = torch.from_numpy(pad_strip).to(dev).float()[None, None]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=dev)[None, None]
    imp = F.pad(im, (1, 1, 1, 1), mode="reflect")
    grad = torch.hypot(F.conv2d(imp, kx), F.conv2d(imp, kx.transpose(2, 3)))
    del imp
    m = _box_gpu(im, janela, F)
    std = torch.sqrt(torch.clamp(_box_gpu(im * im, janela, F) - m * m, min=0))
    del m
    q = (im.to(torch.int16) * niveis // 256).clamp(0, niveis - 1)
    qh = torch.empty_like(q)
    qh[..., :-1] = q[..., 1:]
    qh[..., -1:] = q[..., -1:]
    ent = torch.zeros_like(im)
    tot = torch.zeros_like(im)
    for i in range(niveis):
        Ai = q == i
        for j in range(niveis):
            cnt = _box_gpu((Ai & (qh == j)).to(im.dtype), janela, F) * (janela * janela)
            tot += cnt
            ent += torch.where(cnt > 0, cnt * torch.log(cnt), torch.zeros_like(cnt))
    ent = torch.where(tot > 0, torch.log(tot) - ent / tot, torch.zeros_like(tot))
    corte = slice(half, half + rows)
    saida = tuple(a[0, 0, corte].cpu().numpy() for a in (std, ent, grad))
    del im, q, qh, ent, tot, std, grad
    torch.cuda.empty_cache()
    return saida


def texturas(pad_strip, rows, dev):
    """(std, ent, grad) do bloco — GPU se dev, senão CPU."""
    if dev:
        return features_gpu(pad_strip, rows, WINDOW, LEVELS, dev)
    _, std, ent, grad = features_block(pad_strip, rows, WINDOW, LEVELS)
    return std, ent, grad


# ------------------------------------------------------- estatísticas da cena
def stats_receita(pan_path, W=None, H=None, gt=None):
    """Estatísticas gravadas no receita.json do dataset (valores idênticos aos
    dos patches de treino). None se a cena não está na receita, foi reamostrada
    (aí a grade do dataset não é a nativa) ou se a GRADE do PAN não é a mesma
    que a receita registrou — a receita é indexada pelo nome da pasta da cena,
    e uma pasta pode ter mais de um PAN (jatai: W052S18 e jatai_sirgas, com
    extensões diferentes). Sem essa conferência as estatísticas de um PAN
    entrariam caladas no outro."""
    try:
        cena = os.path.basename(caminhos.pasta_cena(pan_path))
        rec = json.load(open(os.path.join(caminhos.RAIZ, RECEITA), encoding="utf-8"))
        info = rec["resultado"]["por_cena"].get(cena)
        if info is None or info["grade"]["reamostrada"]:
            return None
        g = info["grade"]
        if W is not None and (g["largura"], g["altura"]) != (W, H):
            print("   receita é de outro PAN desta cena (%dx%d na receita, %dx%d aqui)"
                  " — estatísticas próprias" % (g["largura"], g["altura"], W, H), flush=True)
            return None
        if gt is not None and max(abs(a - b) for a, b in zip(g["geotransform"], gt)) > 1e-3:
            print("   receita é de outra grade desta cena — estatísticas próprias", flush=True)
            return None
        e = info["estatisticas"]
        return {"p2": e["p2"], "p98": e["p98"],
                "mu_glcm": e["media"]["glcm"], "sd_glcm": e["desvio"]["glcm"],
                "mu_std": e["media"]["std"], "sd_std": e["desvio"]["std"],
                "mu_grad": e["media"]["grad"], "sd_grad": e["desvio"]["grad"],
                "mu_rug": e["media"]["rugosidade"], "sd_rug": e["desvio"]["rugosidade"],
                "mu_pan": e["media"]["pan"], "sd_pan": e["desvio"]["pan"],
                "fonte": os.path.basename(os.path.dirname(RECEITA))}
    except Exception:
        return None


def ler(band, x0, y0, w, h, W, H):
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    a = band.ReadAsArray(xa, ya, xb - xa, yb - ya).astype(np.uint8)
    pad = ((ya - y0, y0 + h - yb), (xa - x0, x0 + w - xb))
    if any(v for p in pad for v in p):
        a = np.pad(a, pad, mode="reflect")
    return a


def estimar_stats(band, W, H, nd, dev):
    """p2/p98 e média/desvio em N_EST blocos sorteados (mesma receita do dataset)."""
    L = LADO_EST
    pos = [(y, x) for y in range(0, H - L + 1, L) for x in range(0, W - L + 1, L)]
    np.random.default_rng(SEED).shuffle(pos)
    blocos, pans = [], []
    for y, x in pos:
        if len(blocos) == N_EST:
            break
        p = ler(band, x, y, L, L, W, H)
        ok = np.ones((L, L), bool) if nd is None else (p != nd)
        if ok.mean() >= 0.999:
            blocos.append((x, y))
            pans.append(p[ok])
    if not blocos:
        raise SystemExit("nenhum bloco totalmente válido p/ estatística")
    p2, p98 = np.percentile(np.concatenate(pans), [2, 98])
    acc = {"pan": [], "glcm": [], "std": [], "grad": []}
    half = WINDOW // 2
    for x, y in blocos:
        bruto = ler(band, x, y, L, L, W, H)
        s = esticar(bruto, p2, p98)
        std, ent, grad = texturas(np.pad(s, ((half, half), (0, 0)), mode="reflect"), L, dev)
        acc["pan"].append(bruto.ravel().astype(np.float32))   # CRUA (DN), nao a esticada
        acc["glcm"].append(ent.ravel())
        acc["std"].append(std.ravel())
        acc["grad"].append(grad.ravel())
    out = {}
    for k, v in acc.items():
        acc[k] = v = np.concatenate(v)
        out["mu_" + k] = float(v.mean())
        out["sd_" + k] = float(v.std())
    # rugosidade leva um 2o z-score, igual ao gerar_dataset_refinado.py
    rug = ((acc["std"] - out["mu_std"]) / out["sd_std"]
           + (acc["grad"] - out["mu_grad"]) / out["sd_grad"]) / 2
    out["mu_rug"], out["sd_rug"] = float(rug.mean()), float(rug.std())
    return {"p2": float(p2), "p98": float(p98), **out, "fonte": "%d blocos" % len(blocos)}


def esticar(a, p2, p98):
    return (np.clip((a.astype(np.float32) - p2) / (p98 - p2), 0, 1) * 255 + 0.5).astype(np.uint8)


def criar(path, W, H, gt, wkt, descricao, retomar=False, quant=None):
    if retomar and os.path.exists(path):
        d = gdal.Open(path, gdal.GA_Update)
        if (d.RasterXSize, d.RasterYSize) != (W, H):
            raise SystemExit("ERRO: %s e %dx%d, esperado %dx%d"
                             % (path, d.RasterXSize, d.RasterYSize, W, H))
        return d, d.GetRasterBand(1)
    if quant is None:
        tipo, opts = gdal.GDT_Float32, ["PREDICTOR=3"]
    else:
        tipo, opts = gdal.GDT_Byte, ["PREDICTOR=2"]
    d = gdal.GetDriverByName("GTiff").Create(
        path, W, H, 1, tipo,
        ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"] + opts)
    d.SetGeoTransform(gt)
    d.SetProjection(wkt)
    b = d.GetRasterBand(1)
    b.SetDescription(descricao)
    if quant is None:
        b.SetNoDataValue(NODATA)
    else:
        off, esc = quant
        b.SetNoDataValue(0)        # 0 = sem imagem; o valor util vai de 1 a 255
        b.SetOffset(off)
        b.SetScale(esc)
    return d, b


def quantizar(a, fora, quant):
    """Float -> Byte: codigo = (valor - offset)/escala, 1..255; 0 = sem imagem."""
    off, esc = quant
    c = np.clip(np.rint((a - off) / esc), 1, 255).astype(np.uint8)
    c[fora] = 0
    return c


def gerar(pan_path, dev, block=0, retomar=False, byte=False):
    t0 = time.time()
    gdal.SetCacheMax(256 * 1024 * 1024)   # trava a RAM do cache de blocos do GDAL
    ds = gdal.Open(pan_path)
    band = ds.GetRasterBand(1)
    W, H = ds.RasterXSize, ds.RasterYSize
    gt, wkt = ds.GetGeoTransform(), ds.GetProjection()
    nd = band.GetNoDataValue()
    nd = 0 if nd is None else nd        # mosaico sem nodata declarado: 0 = fundo
    if not block:                        # bloco que cabe na VRAM alvo (~16 arrays)
        block = int(np.clip(VRAM_ALVO / (W * 4 * 16) // 128 * 128, 256, 2048)) if dev else 1024
    st = stats_receita(pan_path, W, H, gt) or estimar_stats(band, W, H, nd, dev)
    stem = os.path.splitext(pan_path)[0]
    saidas = [stem + "_glcm" + SUFIXO + ".tif", stem + "_rugosidade" + SUFIXO + ".tif"]
    print("[%s] %dx%d | %s | bloco %d linhas | estatísticas: %s"
          % (os.path.basename(pan_path), W, H, "GPU" if dev else "CPU", block, st["fonte"]), flush=True)
    print("   p2=%.1f p98=%.1f | glcm %.3f/%.3f std %.2f/%.2f grad %.2f/%.2f"
          " rug %.3f/%.3f | pan %.2f/%.2f"
          % (st["p2"], st["p98"], st["mu_glcm"], st["sd_glcm"], st["mu_std"],
             st["sd_std"], st["mu_grad"], st["sd_grad"], st["mu_rug"], st["sd_rug"],
             st["mu_pan"], st["sd_pan"]), flush=True)

    prog = stem + "_features_progresso.json"
    y_inicio = 0
    if retomar:
        if os.path.exists(prog) and all(os.path.exists(s) for s in saidas):
            pr = json.load(open(prog))
            if (pr.get("W"), pr.get("H"), pr.get("block")) == (W, H, block):
                y_inicio = int(pr["y"])
                print("   retomando da linha %d (%.0f%% pronto)"
                      % (y_inicio, 100.0 * y_inicio / H), flush=True)
            else:
                print("   AVISO: progresso de outra cena/bloco — recomecando do zero",
                      flush=True)
        else:
            print("   sem progresso gravado — comecando do zero", flush=True)

    q1 = QUANT["glcm"] if byte else None
    q2 = QUANT["rug"] if byte else None
    d1, b1 = criar(saidas[0], W, H, gt, wkt, "GLCM_Entropia", y_inicio > 0, q1)
    d2, b2 = criar(saidas[1], W, H, gt, wkt, "Rugosidade_STD_Gradiente", y_inicio > 0, q2)
    meta = {k: repr(v) for k, v in st.items()}
    if byte:
        meta["quantizacao"] = ("Byte: valor = codigo*escala + offset (SetScale/SetOffset "
                               "no proprio arquivo); codigo 0 = sem imagem. "
                               "glcm off/esc %s | rugosidade off/esc %s"
                               % (QUANT["glcm"], QUANT["rug"]))
    meta["banda_pan"] = ("(pan_CRUA_em_DN - mu_pan) / sd_pan  -- a banda 1 do dataset "
                         "e o z-score da PAN crua; o esticamento p2-p98 so alimenta o "
                         "calculo das texturas, nao a banda 1")
    d1.SetMetadata(meta)
    d2.SetMetadata(meta)
    half = WINDOW // 2
    vazios = 0
    for n, y0 in enumerate(range(0, H, block), 1):
        y1 = min(y0 + block, H)
        rows = y1 - y0
        if y1 <= y_inicio:
            continue
        top, bot = y0 - half, y1 + half
        ta, tb = max(0, top), min(H, bot)
        strip = band.ReadAsArray(0, ta, W, tb - ta).astype(np.uint8)
        if ta > top:
            strip = np.concatenate([strip[ta - top:0:-1], strip], axis=0)
        if tb < bot:
            k = bot - tb
            strip = np.concatenate([strip, strip[-2:-2 - k:-1]], axis=0)
        pan_c = strip[half:half + rows, :]
        fora = pan_c == nd
        if fora.all():      # bloco 100% fundo: nao calcula textura (so grava vazio)
            vazio = (np.zeros((rows, W), np.uint8) if byte
                     else np.full((rows, W), NODATA, np.float32))
            b1.WriteArray(vazio, 0, y0)
            b2.WriteArray(vazio, 0, y0)
            d1.FlushCache(); d2.FlushCache()
            with open(prog, "w") as f:
                json.dump({"y": y1, "block": block, "W": W, "H": H}, f)
            vazios += 1
            continue
        std, ent, grad = texturas(esticar(strip, st["p2"], st["p98"]), rows, dev)
        glcm = (ent - st["mu_glcm"]) / (st["sd_glcm"] + 1e-9)
        rug = ((std - st["mu_std"]) / (st["sd_std"] + 1e-9) +
               (grad - st["mu_grad"]) / (st["sd_grad"] + 1e-9)) / 2
        rug = (rug - st["mu_rug"]) / (st["sd_rug"] + 1e-9)   # 2o z-score (= dataset)
        if byte:
            b1.WriteArray(quantizar(glcm, fora, q1), 0, y0)
            b2.WriteArray(quantizar(rug, fora, q2), 0, y0)
        else:
            glcm[fora] = NODATA
            rug[fora] = NODATA
            b1.WriteArray(glcm.astype(np.float32), 0, y0)
            b2.WriteArray(rug.astype(np.float32), 0, y0)
        d1.FlushCache()     # dataset, nao banda: fecha o diretorio do TIFF
        d2.FlushCache()
        with open(prog, "w") as f:
            json.dump({"y": y1, "block": block, "W": W, "H": H}, f)
        print("   bloco %2d/%d  linhas %6d-%6d  (%.0fs)"
              % (n, -(-H // block), y0, y1 - 1, time.time() - t0), flush=True)
    d1 = d2 = ds = None
    if os.path.exists(prog):
        os.remove(prog)
    if vazios:
        print("   %d blocos 100%% fundo: textura nao calculada" % vazios, flush=True)
    for p in saidas:
        print("OK %s (%.1f MB)" % (p, os.path.getsize(p) / 1e6))
    print("   %.0fs\n" % (time.time() - t0), flush=True)


def main():
    global RECEITA, SUFIXO
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("Uso:"):])
    ap.add_argument("--pan", help="PAN da cena")
    ap.add_argument("--receita", default=None,
                    help="receita.json do dataset de onde vem as estatisticas da cena "
                         "(padrao: %s)" % RECEITA)
    ap.add_argument("--sufixo", default="",
                    help="sufixo nos arquivos de saida, p.ex. '_v18' gera "
                         "<pan>_glcm_v18.tif; vazio sobrescreve os atuais")
    ap.add_argument("--todos", action="store_true",
                    help="as 3 cenas SPOT5 em UTM 2,5 m (013-014, Colinas, Jataí)")
    ap.add_argument("--cpu", action="store_true", help="força o caminho antigo (scipy)")
    ap.add_argument("--block", type=int, default=0, help="linhas por bloco (0 = automático)")
    ap.add_argument("--byte", action="store_true",
                    help="grava Byte quantizado (offset/escala no arquivo) em vez de "
                         "Float32: ~4x menor, passo 0,05 (glcm) e 0,15 (rugosidade)")
    ap.add_argument("--retomar", action="store_true",
                    help="continua de onde parou (usa <pan>_features_progresso.json)")
    args = ap.parse_args()
    if args.receita:
        RECEITA = os.path.relpath(os.path.abspath(args.receita), caminhos.RAIZ)             if os.path.isabs(args.receita) else args.receita
    SUFIXO = args.sufixo
    print("[stats] %s%s" % (RECEITA, "  | sufixo '%s'" % SUFIXO if SUFIXO else ""),
          flush=True)

    dev = None
    if not args.cpu:
        try:
            import torch
            if torch.cuda.is_available():
                dev = "cuda"
                print("GPU: %s (%.1f GB)" % (torch.cuda.get_device_properties(0).name,
                                             torch.cuda.get_device_properties(0).total_memory / 1e9))
            else:
                print("AVISO: CUDA indisponível — usando CPU")
        except ImportError:
            print("AVISO: torch não instalado — usando CPU")

    if args.todos:
        pans = [caminhos.SPOT5_CENA_013_IMAGEM, caminhos.SPOT5_COLINAS_IMAGEM,
                caminhos.SPOT5_JATAI_IMAGEM]
    elif args.pan:
        pans = [args.pan]
    else:
        ap.error("informe --pan ou --todos")
    for p in pans:
        gerar(p, dev, args.block, args.retomar, args.byte)


if __name__ == "__main__":
    main()
