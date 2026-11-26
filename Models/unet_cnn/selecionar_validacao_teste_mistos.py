#!/usr/bin/env python3
"""Filtra validacao/teste para >=90% mistos, sem consultar predicoes.

Mantem todos os patches com duas classes >=1% e os puros de classes raras.
Completa com poucos puros de vegetacao, escolhidos por bloco espacial e hash
deterministico. O treino ja selecionado e conferido, mas nao e modificado.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/selecionar_validacao_teste_mistos.py \
    --dataset ../Data/dados/datasets/spot5_colinas_v30 \
    --treino unet_cnn/selecoes/spot5_colinas_v31/treino_selecionado.txt \
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
    return [x.strip() for x in caminho.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--treino", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--min-fracao", type=float, default=0.01)
    ap.add_argument("--alvo-misto", type=float, default=0.90)
    a = ap.parse_args()
    if not (0 < a.min_fracao < 0.5 and 0 < a.alvo_misto <= 1):
        ap.error("parametros de selecao invalidos")
    dataset = Path(a.dataset).resolve(strict=True)
    saida = Path(a.saida).resolve(strict=True)
    treino = ler_nomes(Path(a.treino).resolve(strict=True))
    receita = json.loads((dataset / "receita.json").read_text(encoding="utf-8"))
    tamanho_bloco = int(receita["parametros"]["bloco"])
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = {r["arquivo"]: r for r in csv.DictReader(f)}

    def frac(nome):
        r = rows[nome]
        return tuple(float(r[k]) for k in ("frac_veg", "frac_nao_veg", "frac_agua"))

    def bloco(nome):
        r = rows[nome]
        return (int(r["x"]) // tamanho_bloco,
                int(r["y"]) // tamanho_bloco)

    def aleatorio_estavel(nome):
        return hashlib.sha256(nome.encode("utf-8")).hexdigest()

    def relatar(nomes, original, split):
        mistos = sum(sum(f >= a.min_fracao for f in frac(n)) >= 2 for n in nomes)
        somas = [sum(frac(n)[i] for n in nomes) for i in range(3)]
        total = sum(somas)
        return {"original": len(original), "selecionado": len(nomes),
                "mistos": mistos, "proporcao_mista": mistos / len(nomes),
                "classes": dict(zip(("veg", "nao_veg", "agua"),
                                    (s / total for s in somas))),
                "blocos_originais": len({bloco(n) for n in original}),
                "blocos_preservados": len({bloco(n) for n in nomes}),
                "split": split}

    orig_treino = ler_nomes(dataset / "splits/treino.txt")
    if not set(treino).issubset(orig_treino) or len(treino) != len(set(treino)):
        raise RuntimeError("lista de treino nao corresponde ao dataset")
    rel = {"dataset": dataset.name, "min_fracao": a.min_fracao,
           "alvo_misto": a.alvo_misto,
           "selecao_validacao_teste": "rotulos + cobertura espacial + hash; "
                                     "sem previsoes de modelo",
           "splits": {"treino": relatar(treino, orig_treino, "treino")}}
    for split in ("validacao", "teste"):
        original = ler_nomes(dataset / "splits" / (split + ".txt"))
        mistos, raros, veg_puros = [], [], []
        for nome in original:
            v, n, agua = frac(nome)
            presentes = sum(f >= a.min_fracao for f in (v, n, agua))
            if presentes >= 2:
                mistos.append(nome)
            elif presentes == 1 and (n >= a.min_fracao or agua >= a.min_fracao):
                raros.append(nome)
            elif presentes == 1 and v >= a.min_fracao:
                veg_puros.append(nome)
            else:
                raise RuntimeError("patch sem classe principal: %s" % nome)
        max_puros = math.floor(len(mistos) / a.alvo_misto - len(mistos) + 1e-9)
        vagas_veg = max_puros - len(raros)
        if vagas_veg < 0:
            raise RuntimeError("nao cabem todos os patches puros raros de %s" % split)
        todos_blocos = {bloco(n) for n in original}
        retidos = {bloco(n) for n in mistos + raros}
        faltantes = sorted(todos_blocos - retidos)
        if len(faltantes) > vagas_veg:
            raise RuntimeError("nao ha vagas para preservar blocos de %s" % split)
        candidatos = {}
        for nome in veg_puros:
            candidatos.setdefault(bloco(nome), []).append(nome)
        escolhidos = [min(candidatos[b], key=aleatorio_estavel) for b in faltantes]
        usados = set(escolhidos)
        # Entre blocos ja cobertos, prefere os menos representados no split.
        contagem = {b: sum(bloco(n) == b for n in mistos + raros + escolhidos)
                    for b in todos_blocos}
        restantes = sorted((n for n in veg_puros if n not in usados),
                           key=lambda n: (contagem[bloco(n)], aleatorio_estavel(n)))
        escolhidos.extend(restantes[:vagas_veg - len(escolhidos)])
        selecionados = set(mistos + raros + escolhidos)
        lista = [n for n in original if n in selecionados]
        if len(lista) != len(selecionados) or len(escolhidos) != vagas_veg:
            raise RuntimeError("selecao incompleta: %s" % split)
        info = relatar(lista, original, split)
        if info["proporcao_mista"] + 1e-12 < a.alvo_misto or \
                info["blocos_preservados"] != info["blocos_originais"]:
            raise RuntimeError("alvo misto/espacial nao atingido: %s" % split)
        rel["splits"][split] = info
        (saida / (split + "_selecionado.txt")).write_text(
            "\n".join(lista) + "\n", encoding="utf-8")
        print("[%s] %d/%d; %.2f%% mistos; %d/%d blocos" %
              (split, len(lista), len(original), 100 * info["proporcao_mista"],
               info["blocos_preservados"], info["blocos_originais"]), flush=True)
    (saida / "relatorio_splits.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
