#!/usr/bin/env python3
"""Seleciona treino com 90% de patches mistos e conserva classes raras puras.

Um patch e misto quando pelo menos duas classes ocupam 1% da area do patch.
Mantem todos os mistos e os puros de nao vegetacao/agua. Os poucos puros de
vegetacao restantes preservam blocos espaciais e priorizam falsos positivos de
nao vegetacao de um modelo BASE treinado apenas no split de treino.
Validacao e teste nao participam da selecao.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/selecionar_treino_misto.py \
    --dataset ../Data/dados/datasets/spot5_colinas_v30 \
    --escores unet_cnn/runs/unetpp_spot5_colinas_v28/diagnostico_erros/escores_patches_treino.csv \
    --saida unet_cnn/selecoes/spot5_colinas_v31
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import ambiente  # noqa: F401  (padrao do projeto)


def ler_nomes(caminho):
    return [s.strip() for s in caminho.read_text(encoding="utf-8").splitlines()
            if s.strip()]


def sha256(caminho):
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--escores", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--min-fracao", type=float, default=0.01)
    ap.add_argument("--alvo-misto", type=float, default=0.90)
    args = ap.parse_args()
    if not (0 < args.min_fracao < 0.5 and 0 < args.alvo_misto <= 1):
        ap.error("fracao minima e alvo misto invalidos")
    dataset = Path(args.dataset).resolve(strict=True)
    escores = Path(args.escores).resolve(strict=True)
    saida = Path(args.saida).absolute()
    if saida.exists():
        raise SystemExit("saida ja existe: %s" % saida)
    treino = ler_nomes(dataset / "splits/treino.txt")
    receita = json.loads((dataset / "receita.json").read_text(encoding="utf-8"))
    bloco = int(receita["parametros"]["bloco"])
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest = {r["arquivo"]: r for r in csv.DictReader(f) if r["split"] == "treino"}
    with escores.open(newline="", encoding="utf-8") as f:
        scores = {r["patch"]: r for r in csv.DictReader(f)}
    if set(manifest) != set(treino) or set(scores) != set(treino):
        raise SystemExit("manifest/escores nao correspondem ao split de treino")

    def frac(nome):
        r = manifest[nome]
        return tuple(float(r[k]) for k in ("frac_veg", "frac_nao_veg", "frac_agua"))

    def bloco_de(nome):
        r = manifest[nome]
        return (int(r["x"]) // bloco, int(r["y"]) // bloco)

    def dificil(nome):
        r = scores[nome]
        veg = max(1, int(r["pixels_veg"]))
        return (int(r["fp_nao_veg"]) / veg,
                float(r["prob_nao_veg_p95_em_veg"]), nome)

    mistos, raros_puros, veg_puros = [], [], []
    for nome in treino:
        v, n, a = frac(nome)
        presentes = sum(f >= args.min_fracao for f in (v, n, a))
        if presentes >= 2:
            mistos.append(nome)
        elif presentes == 1 and (n >= args.min_fracao or a >= args.min_fracao):
            raros_puros.append(nome)
        elif presentes == 1 and v >= args.min_fracao:
            veg_puros.append(nome)
        else:
            raise RuntimeError("patch sem classe principal: %s" % nome)

    # maximo de puros que ainda deixa a proporcao mista >= alvo.
    max_puros = math.floor(len(mistos) / args.alvo_misto - len(mistos) + 1e-9)
    vagas_veg = max_puros - len(raros_puros)
    if vagas_veg < 0:
        raise RuntimeError("puros raros excedem o teto; nao cabe o alvo misto")
    blocos_todos = {bloco_de(n) for n in treino}
    blocos_retidos = {bloco_de(n) for n in mistos + raros_puros}
    faltantes = sorted(blocos_todos - blocos_retidos)
    if len(faltantes) > vagas_veg:
        raise RuntimeError("alvo misto nao permite preservar todos os blocos")
    candidatos = {}
    for nome in veg_puros:
        candidatos.setdefault(bloco_de(nome), []).append(nome)
    escolhidos = []
    for b in faltantes:
        escolhidos.append(max(candidatos[b], key=dificil))
    ja = set(escolhidos)
    # Diversifica os exemplos negativos antes de repetir blocos.
    usados = {bloco_de(n) for n in escolhidos}
    por_bloco = []
    for b, nomes in candidatos.items():
        if b not in usados:
            por_bloco.append(max(nomes, key=dificil))
    por_bloco.sort(key=dificil, reverse=True)
    for nome in por_bloco:
        if len(escolhidos) >= vagas_veg:
            break
        escolhidos.append(nome)
        ja.add(nome)
    for nome in sorted(veg_puros, key=dificil, reverse=True):
        if len(escolhidos) >= vagas_veg:
            break
        if nome not in ja:
            escolhidos.append(nome)
            ja.add(nome)

    selecionados = set(mistos + raros_puros + escolhidos)
    lista = [nome for nome in treino if nome in selecionados]
    if len(lista) != len(selecionados) or len(escolhidos) != vagas_veg:
        raise RuntimeError("selecao incompleta ou duplicada")
    if {bloco_de(n) for n in lista} != blocos_todos:
        raise RuntimeError("selecao perdeu cobertura espacial")
    proporcao = len(mistos) / len(lista)
    if proporcao + 1e-12 < args.alvo_misto:
        raise RuntimeError("alvo misto nao atingido")
    somas = [sum(frac(nome)[i] for nome in lista) for i in range(3)]
    total = sum(somas)
    rel = {
        "dataset": dataset.name,
        "criterio_classe_presente": args.min_fracao,
        "alvo_misto": args.alvo_misto,
        "treino_original": len(treino), "treino_selecionado": len(lista),
        "mistos": len(mistos), "puros_raros": len(raros_puros),
        "puros_vegetacao": len(escolhidos),
        "proporcao_mista": proporcao,
        "classes_treino_selecionado": dict(zip(("veg", "nao_veg", "agua"),
                                              (s / total for s in somas))),
        "blocos_originais": len(blocos_todos),
        "blocos_preservados": len({bloco_de(n) for n in lista}),
        "blocos_resgatados_por_veg_puro": len(faltantes),
        "criterio_veg_puro": "maior taxa de FP nao vegetacao, depois p95; "
                              "prioridade para blocos ainda nao representados",
        "escores_modelo": "unetpp_spot5_colinas_v28 (treino in-sample)",
        "sha256_manifest": sha256(dataset / "manifest.csv"),
        "sha256_escores": sha256(escores),
        "validacao_teste": "identicos ao v30; nao usados na selecao",
    }
    saida.mkdir(parents=True)
    (saida / "treino_selecionado.txt").write_text("\n".join(lista) + "\n",
                                                   encoding="utf-8")
    (saida / "relatorio.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[selecao] %d/%d patches; %.2f%% mistos; %d/%d blocos; "
          "classes %.2f/%.2f/%.2f%%" %
          (len(lista), len(treino), 100 * proporcao, rel["blocos_preservados"],
           rel["blocos_originais"], *(100 * s / total for s in somas)), flush=True)


if __name__ == "__main__":
    main()
