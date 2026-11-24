#!/usr/bin/env python3
"""Baixa dados do S3, treina U-Net++, avalia e infere uma cena no GPU Pod.

Tudo acontece no Pod. Cada etapa envia seus resultados ao Network Volume antes
da seguinte; o PC pode se desconectar depois que este processo for iniciado.
Os scripts locais de treino e inferência são chamados sem modificações.

Uso (no WSL, a partir de Models/; também vale dentro do Pod):
  ../Data/.venv/bin/python -u ../runpod/rodada_unet_s3.py \
    --dataset spot5_jatai_v27 --nome unetpp_spot5_jatai_v27 \
    --imagem Data/dados/spot5/jatai/original/jatai_sirgas.tif \
    --ndvi Data/dados/spot5/jatai/original/NDVI_Temporal_Landsat5_Jatai_2007_2008.tif \
    --cena jatai

Credenciais S3 vêm do perfil runpod em ~/.aws/credentials ou das variáveis
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY. RUNPOD_VOLUME_ID e RUNPOD_DATACENTER
selecionam o volume; não precisam coincidir com a região da GPU.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from botocore.exceptions import ClientError
from subir import baixar, cliente, enviar


RAIZ = Path(__file__).resolve().parent.parent
MODELS = RAIZ / "Models"
PY = str(RAIZ / "Data" / ".venv" / "bin" / "python")


def relativo(valor):
    p = Path(valor)
    if not p.parts or p.is_absolute() or ".." in p.parts:
        raise argparse.ArgumentTypeError("use caminho relativo à raiz do repositório")
    return p.as_posix()


def comando(args, log, ao_ler=None):
    print("[comando] " + " ".join(map(str, args)), flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n[comando] " + " ".join(map(str, args)) + "\n")
        f.flush()
        p = subprocess.Popen(args, cwd=MODELS, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for linha in p.stdout:
            print(linha, end="", flush=True)
            f.write(linha)
            f.flush()
            if ao_ler is not None:
                ao_ler(linha)
        codigo = p.wait()
        f.write("[exit] %d\n" % codigo)
    if codigo:
        raise RuntimeError("comando falhou com código %d: %s" % (codigo, args[0]))


def sha256(caminho):
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--nome", required=True)
    ap.add_argument("--imagem", required=True, type=relativo)
    ap.add_argument("--ndvi", required=True, type=relativo)
    ap.add_argument("--cena", required=True)
    ap.add_argument("--volume", default=os.environ.get("RUNPOD_VOLUME_ID"))
    ap.add_argument("--datacenter", default=os.environ.get("RUNPOD_DATACENTER"))
    ap.add_argument("--lote", type=int, default=20)
    ap.add_argument("--lote-val", type=int, default=16)
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--paciencia", type=int, default=5)
    ap.add_argument("--aquecimento", type=int, default=5)
    ap.add_argument("--peso-classe", nargs=3, type=float, default=None,
                    metavar=("VEG", "NAO_VEG", "AGUA"))
    ap.add_argument("--lote-inferencia", type=int, default=24)
    ap.add_argument("--ladrilho", type=int, default=4096)
    ap.add_argument("--passo", type=int, default=192)
    ap.add_argument("--piloto-x", type=int, default=11008)
    ap.add_argument("--piloto-y", type=int, default=17216)
    ap.add_argument("--retomar", action="store_true",
                    help="pula treino se checkpoint e relatório já estão no S3")
    a = ap.parse_args()
    if not a.volume or not a.datacenter:
        ap.error("faltou --volume/--datacenter ou RUNPOD_VOLUME_ID/RUNPOD_DATACENTER")
    if any("/" in x or "\\" in x or x in ("", ".", "..")
           for x in (a.dataset, a.nome, a.cena)):
        ap.error("dataset, nome e cena devem ser nomes simples")

    cli = cliente(a.datacenter)
    dataset_key = "Data/dados/datasets/" + a.dataset
    dataset = RAIZ / dataset_key
    run_key = "Models/unet_cnn/runs/" + a.nome
    run = RAIZ / run_key
    run.mkdir(parents=True, exist_ok=True)
    log = run / "rodada_s3.log"
    mapa = run / ("cena_" + a.cena)
    enviados = {}

    def baixar_entrada(key, esperar_s=0):
        if esperar_s:
            limite = time.monotonic() + esperar_s
            while True:
                try:
                    cli.head_object(Bucket=a.volume, Key=key)
                    break
                except ClientError as erro:
                    codigo = erro.response.get("Error", {}).get("Code")
                    if codigo not in ("404", "NoSuchKey", "NotFound"):
                        raise
                    if time.monotonic() >= limite:
                        raise TimeoutError("entrada nao chegou ao S3: " + key) from erro
                    print("[s3] aguardando entrada " + key + " (30 s)", flush=True)
                    time.sleep(30)
        print("[s3] baixando " + key, flush=True)
        baixar(cli, a.volume, key, str(RAIZ / key))

    def salvar(caminho):
        if not caminho.is_file():
            return False
        estado = (caminho.stat().st_size, caminho.stat().st_mtime_ns)
        if enviados.get(caminho) == estado:
            return True
        key = caminho.relative_to(RAIZ).as_posix()
        print("[s3] enviando " + key, flush=True)
        enviar(cli, a.volume, str(caminho), key, forcar=True, tudo=True)
        remoto = cli.head_object(Bucket=a.volume, Key=key)["ContentLength"]
        if remoto != caminho.stat().st_size:
            raise RuntimeError("tamanho S3 diferente: " + key)
        enviados[caminho] = estado
        return True

    def salvar_etapa():
        for nome in ("modelo_melhor.pth", "ultimo.pth", "relatorio.json",
                     "relatorio_completo.json", "log.csv", "rodada_s3.log"):
            salvar(run / nome)
        for nome in ("classes.tif", "classes.png", "info.json"):
            salvar(mapa / nome)

    try:
        t0 = time.monotonic()
        baixar_entrada(dataset_key)
        receita = json.loads((dataset / "receita.json").read_text(encoding="utf-8"))
        esperadas = ["PAN", "GLCM_Entropia", "Rugosidade_STD_Gradiente",
                     "NDVI_median", "NDVI_min", "NDVI_max", "NDVI_stdDev"]
        if receita["bandas"] != esperadas:
            raise ValueError("dataset não tem as sete bandas na ordem esperada")
        if not (dataset / "splits" / "treino.txt").is_file():
            raise FileNotFoundError("dataset sem splits/treino.txt")
        n_imagens = len(list((dataset / "images").glob("*.tif")))
        n_mascaras = len(list((dataset / "masks").glob("*.tif")))
        esperado = sum(receita["resultado"]["total_por_split"].values())
        if (n_imagens, n_mascaras) != (esperado, esperado):
            raise ValueError("dataset incompleto: %d imagens, %d máscaras; esperado %d"
                             % (n_imagens, n_mascaras, esperado))
        print("[dataset] sete bandas; %d pares" % esperado, flush=True)

        if a.retomar:
            for nome in ("modelo_melhor.pth", "relatorio.json"):
                baixar_entrada(run_key + "/" + nome)
        else:
            def salvar_a_cada_cinco_epocas(linha):
                if not linha.startswith("epoca "):
                    return
                try:
                    epoca = int(linha.split()[1])
                except (IndexError, ValueError):
                    return
                if epoca % 5 == 0:
                    salvar_etapa()

            treino = [PY, "-u", "unet_cnn/train.py", "--nome", a.nome,
                     "--dataset", "../Data/dados/datasets/" + a.dataset,
                     "--arquitetura", "unetpp", "--base", "32",
                     "--lote", str(a.lote), "--lote-val", str(a.lote_val),
                     "--epocas", str(a.epocas), "--paciencia", str(a.paciencia),
                     "--aquecimento", str(a.aquecimento),
                     "--reamostrar", "--reamostrar-k", "3", "--dados-na-ram"]
            if a.peso_classe:
                treino += ["--peso-classe"] + [str(p) for p in a.peso_classe]
            comando(treino, log, ao_ler=salvar_a_cada_cinco_epocas)
            if not (run / "modelo_melhor.pth").is_file():
                raise FileNotFoundError("treino terminou sem modelo_melhor.pth")
            salvar_etapa()

        comando([PY, "-u", "unet_cnn/avaliar.py", "--nome", a.nome,
                 "--dataset", "../Data/dados/datasets/" + a.dataset,
                 "--splits", "validacao", "teste", "--lote", str(a.lote_val)], log)
        salvar_etapa()

        baixar_entrada(a.imagem, esperar_s=1800)
        baixar_entrada(a.ndvi)
        mapa.mkdir(parents=True, exist_ok=True)
        piloto = run / ("cena_" + a.cena + "_piloto")
        piloto.mkdir(exist_ok=True)
        recorte = piloto / "pan_recorte.tif"
        if not recorte.is_file():
            comando(["gdal_translate", "-srcwin", str(a.piloto_x), str(a.piloto_y),
                     "1024", "1024", str(RAIZ / a.imagem), str(recorte)], log)
        base_infer = [PY, "-u", "unet_cnn/inferencia_cena.py",
                      "--modelo", str(run / "modelo_melhor.pth"),
                      "--dataset", str(dataset), "--cena", a.cena,
                      "--ndvi-temporal", str(RAIZ / a.ndvi),
                      "--passo", str(a.passo), "--lote", str(a.lote_inferencia)]
        if not (piloto / "classes.tif").is_file():
            comando(base_infer + ["--imagem", str(recorte), "--saida", str(piloto),
                                 "--ladrilho", "1024"], log)
        if not (piloto / "classes.tif").is_file():
            raise FileNotFoundError("piloto sem classes.tif")
        salvar(run / "rodada_s3.log")

        comando(base_infer + ["--imagem", str(RAIZ / a.imagem),
                             "--saida", str(mapa), "--ladrilho", str(a.ladrilho)], log)
        if not (mapa / "classes.tif").is_file():
            raise FileNotFoundError("inferência sem classes.tif")
        comando([PY, "-u", "unet_cnn/gerar_previa.py",
                 "--imagem", str(mapa / "classes.tif"),
                 "--saida", str(mapa / "classes.png"), "--largura", "4096"], log)
        salvar_etapa()

        artefatos = [run / "modelo_melhor.pth", mapa / "classes.tif",
                     mapa / "classes.png"]
        manifesto = {"dataset": a.dataset, "run": a.nome, "cena": a.cena,
                     "volume": a.volume, "datacenter": a.datacenter,
                     "duracao_s": round(time.monotonic() - t0),
                     "artefatos": {p.relative_to(RAIZ).as_posix():
                                   {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                   for p in artefatos}}
        caminho_manifesto = run / "s3_manifesto.json"
        caminho_manifesto.write_text(json.dumps(manifesto, indent=2) + "\n",
                                     encoding="utf-8")
        salvar(caminho_manifesto)
        print("[concluído] treino, avaliação e mapa salvos no S3", flush=True)
    finally:
        salvar_etapa()


if __name__ == "__main__":
    main()
