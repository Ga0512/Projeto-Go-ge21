#!/usr/bin/env python3
"""Cria dataset treinavel com splits selecionados a partir de uma origem.

Usa hard links no mesmo volume: nao copia os GeoTIFFs nem altera a origem.
Os links sao arquivos comuns para upload posterior ao S3. A saida nova nao
pode existir, para impedir sobrescrita acidental de datasets ou resultados.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/montar_dataset_selecionado.py \
      --origem ../Data/dados/datasets/spot5_colinas_v30 \
      --lista unet_cnn/selecoes/spot5_colinas_v31/treino_selecionado.txt \
      --lista-validacao unet_cnn/selecoes/spot5_colinas_v31/validacao_selecionado.txt \
      --lista-teste unet_cnn/selecoes/spot5_colinas_v31/teste_selecionado.txt \
      --saida ../Data/dados/datasets/spot5_colinas_v31
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import shutil
import tempfile
from pathlib import Path

import ambiente  # noqa: F401  (convenção do projeto)


def nomes(path):
    with open(path, encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origem", required=True)
    ap.add_argument("--lista", required=True, help="novo split treino.txt")
    ap.add_argument("--lista-validacao", default=None,
                    help="novo split validacao.txt; padrao: identico a origem")
    ap.add_argument("--lista-teste", default=None,
                    help="novo split teste.txt; padrao: identico a origem")
    ap.add_argument("--saida", required=True)
    args = ap.parse_args()
    origem = Path(args.origem).resolve(strict=True)
    saida = Path(args.saida).absolute()
    if saida.exists():
        raise SystemExit("saida ja existe: %s" % saida)
    treino_original = nomes(origem / "splits" / "treino.txt")
    treino_novo = nomes(args.lista)
    originais = {s: nomes(origem / "splits" / (s + ".txt"))
                 for s in ("validacao", "teste")}
    listas = {"treino": treino_novo,
              "validacao": nomes(args.lista_validacao) if args.lista_validacao
              else originais["validacao"],
              "teste": nomes(args.lista_teste) if args.lista_teste
              else originais["teste"]}
    for split, selecionados in listas.items():
        fonte = treino_original if split == "treino" else originais[split]
        if not selecionados or len(set(selecionados)) != len(selecionados):
            raise SystemExit("lista vazia ou duplicada: %s" % split)
        if not set(selecionados).issubset(fonte):
            raise SystemExit("lista %s contem patch fora do split original" % split)
    if sum(len(v) for v in listas.values()) != len(set().union(*(set(v) for v in listas.values()))):
        raise SystemExit("vazamento entre splits")
    with open(origem / "manifest.csv", newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        campos = leitor.fieldnames
        manifest = list(leitor)
    requeridos = set().union(*(set(v) for v in listas.values()))
    linhas = [r for r in manifest if r["arquivo"] in requeridos]
    if {r["arquivo"] for r in linhas} != requeridos:
        raise SystemExit("manifest nao cobre todos os patches selecionados")
    with open(origem / "receita.json", encoding="utf-8") as f:
        receita = json.load(f)
    receita["script"] = "Models/unet_cnn/montar_dataset_selecionado.py"
    receita["gerado_em"] = datetime.now(timezone.utc).isoformat()
    resultado = receita.get("resultado", {})
    resultado.setdefault("total_por_split", {}).update(
        {split: len(nomes_split) for split, nomes_split in listas.items()})
    for cena, info in resultado.get("por_cena", {}).items():
        info.setdefault("selecionadas_por_split", {}).update({
            split: sum(r["cena"] == cena and r["split"] == split for r in linhas)
            for split in listas})
        # Esta metrica de sobreposicao foi calculada para o dataset fonte.
        info.pop("repeticao", None)
    if "classes_por_split" in resultado and all(
            k in campos for k in ("frac_veg", "frac_nao_veg", "frac_agua")):
        for split in listas:
            amostras = [r for r in linhas if r["split"] == split]
            soma = [sum(float(r[c]) for r in amostras)
                    for c in ("frac_veg", "frac_nao_veg", "frac_agua")]
            total = sum(soma)
            if total <= 0:
                raise RuntimeError("split sem pixels rotulados: %s" % split)
            resultado["classes_por_split"][split] = {
                codigo: round(valor / total, 6)
                for codigo, valor in zip(("3", "15", "33"), soma)}
    receita["selecao_splits"] = {
        "origem": str(origem),
        "listas": {"treino": str(Path(args.lista).resolve(strict=True)),
                   "validacao": str(Path(args.lista_validacao).resolve(strict=True))
                   if args.lista_validacao else None,
                   "teste": str(Path(args.lista_teste).resolve(strict=True))
                   if args.lista_teste else None},
        "original": {"treino": len(treino_original),
                     **{s: len(v) for s, v in originais.items()}},
        "selecionado": {s: len(v) for s, v in listas.items()},
        "validacao_teste_identicos": not (args.lista_validacao or args.lista_teste),
        "imagens_mascaras": "hard links para o dataset de origem no mesmo volume",
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=saida.name + ".incompleto_", dir=saida.parent))
    for tipo in ("images", "masks"):
        (tmp / tipo).mkdir()
    (tmp / "splits").mkdir()
    for n in requeridos:
        for tipo in ("images", "masks"):
            fonte = origem / tipo / (n + ".tif")
            if not fonte.is_file():
                raise SystemExit("arquivo ausente: %s" % fonte)
            os.link(fonte, tmp / tipo / fonte.name)
    for split, nomes_split in listas.items():
        (tmp / "splits" / (split + ".txt")).write_text(
            "\n".join(nomes_split) + "\n", encoding="utf-8")
    with open(tmp / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    (tmp / "receita.json").write_text(
        json.dumps(receita, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if len(list((tmp / "images").iterdir())) != len(requeridos) or \
            len(list((tmp / "masks").iterdir())) != len(requeridos):
        raise RuntimeError("contagem de arquivos diverge da selecao")
    os.rename(tmp, saida)
    print("[feito] %s: treino %d, val %d, teste %d, %d bandas" %
          (saida, len(listas["treino"]), len(listas["validacao"]),
           len(listas["teste"]),
           len(receita["bandas"])), flush=True)


if __name__ == "__main__":
    main()
