#!/usr/bin/env python3
"""Envia e baixa arquivos de um Network Volume do RunPod pela API S3.

POR QUE boto3 E NÃO runpodctl. O `runpodctl network-volume upload` só existe da
v3.1.0 em diante; a versão que temos é a 2.1.6 e nem resolve mais o endpoint
dela (`rest.runpod.io` não existe no DNS). A API S3 é o caminho documentado e o
boto3 precisa estar instalado no Python que executa este script.

RETOMÁVEL. Antes de enviar, compara nome e tamanho com o que já está lá e pula
o que confere. Se a conexão cair, rode de novo: continua de onde parou. Não há
verificação de conteúdo -- se um arquivo mudou mantendo o mesmo tamanho, use
`--forcar`.

O VOLUME espelha o repositório, para os caminhos relativos dos scripts
continuarem válidos lá dentro:

    /vol/Models/...                 código
    /vol/Data/dados/datasets/...    datasets
    /vol/Data/dados/wv1/...         rasters das cenas

Credenciais: perfil `runpod` em ~/.aws/credentials no PC, ou variáveis
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY no Pod. São chaves S3, diferentes da
chave da API RunPod.

Uso (a partir da raiz do repositório):

    python runpod/subir.py --listar
    python runpod/subir.py Models                     --para Models
    python runpod/subir.py Data/dados/datasets/wv_jatai_v4 \\
                           --para Data/dados/datasets/wv_jatai_v4
    python runpod/subir.py ... --baixar               # traz de volta
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

DATACENTERS = [
    "EU-CZ-1", "EU-RO-1", "EUR-IS-1", "EUR-NO-1", "US-CA-2", "US-GA-2",
    "US-IL-1", "US-KS-2", "US-MD-1", "US-MO-1", "US-MO-2", "US-NC-1",
    "US-NC-2", "US-NE-1", "US-WA-1",
]
# arquivos que não fazem sentido no volume
IGNORAR = (".git", "__pycache__", ".cache", ".venv", "runs", "inferencias",
           ".pyc", ".tar.gz", ".log")


def cliente(datacenter):
    if datacenter not in DATACENTERS:
        sys.exit("datacenter %s nao suporta a API S3. Suportados: %s"
                 % (datacenter, ", ".join(DATACENTERS)))
    acesso = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("RUNPOD_S3_ACCESS_KEY")
    segredo = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("RUNPOD_S3_SECRET_KEY")
    if acesso and segredo:
        s = boto3.Session(aws_access_key_id=acesso, aws_secret_access_key=segredo)
    else:
        s = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "runpod"))
    return s.client(
        "s3",
        endpoint_url="https://s3api-%s.runpod.io/" % datacenter.lower(),
        region_name=datacenter,
        config=Config(retries={"mode": "standard", "max_attempts": 10},
                      read_timeout=7200, connect_timeout=60),
    )


def _pular(caminho):
    return any(p in caminho for p in IGNORAR)


def remotos(cli, bucket, prefixo):
    """{chave: tamanho} do que ja esta no volume sob o prefixo."""
    fora = {}
    tok = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefixo)
        if tok:
            kw["ContinuationToken"] = tok
        r = cli.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            fora[o["Key"]] = o["Size"]
        if not r.get("IsTruncated"):
            return fora
        tok = r.get("NextContinuationToken")


def enviar(cli, bucket, origem, destino, forcar, tudo=False, workers=1):
    origem = os.path.abspath(origem)
    if not os.path.exists(origem):
        sys.exit("nao existe: " + origem)

    arquivos = []
    if os.path.isfile(origem):
        arquivos.append((origem, destino))
    else:
        for raiz, dirs, nomes in os.walk(origem):
            dirs[:] = [d for d in dirs if tudo or not _pular(d)]
            for n in nomes:
                p = os.path.join(raiz, n)
                if not tudo and _pular(p):
                    continue
                rel = os.path.relpath(p, origem).replace(os.sep, "/")
                arquivos.append((p, destino.rstrip("/") + "/" + rel))

    ja = {} if forcar else remotos(cli, bucket, destino.rstrip("/") + "/")
    pendentes = [(p, k) for p, k in arquivos
                 if forcar or ja.get(k) != os.path.getsize(p)]
    total = sum(os.path.getsize(p) for p, _ in pendentes)
    print("%d arquivos | %d a enviar (%.2f GB) | %d ja la"
          % (len(arquivos), len(pendentes), total / 2**30,
             len(arquivos) - len(pendentes)), flush=True)
    if not pendentes:
        return

    t0, feito = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tarefas = {pool.submit(cli.upload_file, p, bucket, k): os.path.getsize(p)
                   for p, k in pendentes}
        for i, tarefa in enumerate(as_completed(tarefas), 1):
            tarefa.result()
            feito += tarefas[tarefa]
            dt = time.time() - t0
            if i % 25 == 0 or i == len(pendentes):
                print("  %d/%d | %.2f/%.2f GB | %.1f MB/s | faltam %s"
                      % (i, len(pendentes), feito / 2**30, total / 2**30,
                         feito / 2**20 / max(dt, 1e-9),
                         time.strftime("%H:%M:%S",
                                       time.gmtime((total - feito) / max(feito / dt, 1)))),
                      flush=True)
    print("enviado em %.0f s" % (time.time() - t0))


def baixar(cli, bucket, prefixo, destino):
    prefixo = prefixo.strip("/")
    if prefixo:
        try:
            cabecalho = cli.head_object(Bucket=bucket, Key=prefixo)
        except ClientError as erro:
            if erro.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
        else:
            if os.path.isdir(destino):
                destino = os.path.join(destino, os.path.basename(prefixo))
            os.makedirs(os.path.dirname(os.path.abspath(destino)), exist_ok=True)
            if os.path.isfile(destino) and os.path.getsize(destino) == cabecalho["ContentLength"]:
                print("ja existe: " + destino, flush=True)
                return
            temporario = destino + ".part"
            cli.download_file(bucket, prefixo, temporario)
            os.replace(temporario, destino)
            print("-> " + destino, flush=True)
            return
    prefixo = prefixo.rstrip("/") + "/" if prefixo else ""
    os.makedirs(destino, exist_ok=True)
    itens = remotos(cli, bucket, prefixo)
    if not itens:
        raise FileNotFoundError("nenhum arquivo no S3 sob " + prefixo)
    print("%d arquivos sob %s" % (len(itens), prefixo), flush=True)
    for i, (k, tam) in enumerate(sorted(itens.items()), 1):
        if k.endswith("/"):
            continue
        rel = k[len(prefixo):].lstrip("/")
        alvo = os.path.join(destino, rel.replace("/", os.sep))
        if os.path.exists(alvo) and os.path.getsize(alvo) == tam:
            continue
        os.makedirs(os.path.dirname(alvo), exist_ok=True)
        temporario = alvo + ".part"
        cli.download_file(bucket, k, temporario)
        os.replace(temporario, alvo)
        if i % 25 == 0:
            print("  %d/%d" % (i, len(itens)), flush=True)
    print("-> " + destino)


def main():
    ap = argparse.ArgumentParser(description="envia/traz arquivos do volume RunPod")
    ap.add_argument("origem", nargs="?", help="arquivo ou pasta local (ou prefixo, com --baixar)")
    ap.add_argument("--para", help="caminho dentro do volume")
    ap.add_argument("--volume", default=os.environ.get("RUNPOD_VOLUME_ID"),
                    help="ID do volume (= bucket). Ou RUNPOD_VOLUME_ID no ambiente")
    ap.add_argument("--datacenter", default=os.environ.get("RUNPOD_DATACENTER"),
                    help="ex.: US-GA-2. Ou RUNPOD_DATACENTER no ambiente")
    ap.add_argument("--listar", action="store_true", help="lista o que ha no volume")
    ap.add_argument("--baixar", action="store_true", help="traz do volume para cá")
    ap.add_argument("--forcar", action="store_true", help="reenvia mesmo o que ja esta la")
    ap.add_argument("--tudo", action="store_true",
                    help="inclui runs, logs e outros artefatos ao enviar resultados")
    ap.add_argument("--workers", type=int, default=1,
                    help="uploads paralelos; use 8 para milhares de patches pequenos")
    a = ap.parse_args()
    if a.workers < 1:
        ap.error("--workers precisa ser >= 1")

    if not a.volume or not a.datacenter:
        sys.exit("faltou --volume e/ou --datacenter (ver runpod/LEIAME.md)")
    cli = cliente(a.datacenter)

    if a.listar:
        itens = remotos(cli, a.volume, a.origem or "")
        tam = sum(itens.values())
        print("%d arquivos | %.2f GB" % (len(itens), tam / 2**30))
        pastas = {}
        for k, v in itens.items():
            pastas.setdefault("/".join(k.split("/")[:3]), [0, 0])
            pastas["/".join(k.split("/")[:3])][0] += 1
            pastas["/".join(k.split("/")[:3])][1] += v
        for p, (n, v) in sorted(pastas.items()):
            print("  %-48s %5d arq  %8.2f GB" % (p, n, v / 2**30))
    elif a.baixar:
        baixar(cli, a.volume, a.origem or "", a.para or ".")
    else:
        if not a.origem or not a.para:
            sys.exit("informe a origem local e --para <caminho no volume>")
        enviar(cli, a.volume, a.origem, a.para, a.forcar, a.tudo, a.workers)


if __name__ == "__main__":
    main()
