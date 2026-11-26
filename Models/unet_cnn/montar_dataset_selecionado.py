#!/usr/bin/env python3
"""Cria dataset treinavel com novo treino e validacao/teste identicos.

Usa hard links no mesmo volume: nao copia os GeoTIFFs nem altera a origem.
Os links sao arquivos comuns para upload posterior ao S3. A saida nova nao
pode existir, para impedir sobrescrita acidental de datasets ou resultados.

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python unet_cnn/montar_dataset_selecionado.py \
      --origem ../Data/dados/datasets/spot5_colinas_v28 \
      --lista unet_cnn/runs/unetpp_spot5_colinas_v28/diagnostico_erros/selecao_v29/treino_selecionado.txt \
      --saida ../Data/dados/datasets/spot5_colinas_v29
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
    ap.add_argument("--saida", required=True)
    args = ap.parse_args()
    origem = Path(args.origem).resolve(strict=True)
    saida = Path(args.saida).absolute()
    if saida.exists():
        raise SystemExit("saida ja existe: %s" % saida)
    treino_original = nomes(origem / "splits" / "treino.txt")
    treino_novo = nomes(args.lista)
    validacao = nomes(origem / "splits" / "validacao.txt")
    teste = nomes(origem / "splits" / "teste.txt")
    if len(set(treino_novo)) != len(treino_novo) or not treino_novo:
        raise SystemExit("lista de treino vazia ou duplicada")
    if not set(treino_novo).issubset(treino_original):
        raise SystemExit("lista contem patch fora do treino original")
    if set(treino_novo) & (set(validacao) | set(teste)):
        raise SystemExit("vazamento entre treino e validacao/teste")
    with open(origem / "manifest.csv", newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        campos = leitor.fieldnames
        manifest = list(leitor)
    requeridos = set(treino_novo) | set(validacao) | set(teste)
    linhas = [r for r in manifest if r["arquivo"] in requeridos]
    if {r["arquivo"] for r in linhas} != requeridos:
        raise SystemExit("manifest nao cobre todos os patches selecionados")
    with open(origem / "receita.json", encoding="utf-8") as f:
        receita = json.load(f)
    receita["script"] = "Models/unet_cnn/montar_dataset_selecionado.py"
    receita["gerado_em"] = datetime.now(timezone.utc).isoformat()
    resultado = receita.get("resultado", {})
    resultado.get("total_por_split", {})["treino"] = len(treino_novo)
    for cena, info in resultado.get("por_cena", {}).items():
        n_cena = sum(r["cena"] == cena and r["split"] == "treino"
                     for r in linhas)
        info.get("selecionadas_por_split", {})["treino"] = n_cena
        # Esta metrica de sobreposicao foi calculada para o dataset fonte.
        info.pop("repeticao", None)
    if "classes_por_split" in resultado and all(
            k in campos for k in ("frac_veg", "frac_nao_veg", "frac_agua")):
        amostras_treino = [r for r in linhas if r["split"] == "treino"]
        soma = [sum(float(r[c]) for r in amostras_treino)
                for c in ("frac_veg", "frac_nao_veg", "frac_agua")]
        total = sum(soma)
        if total <= 0:
            raise RuntimeError("treino selecionado sem pixels rotulados")
        resultado["classes_por_split"]["treino"] = {
            codigo: round(valor / total, 6)
            for codigo, valor in zip(("3", "15", "33"), soma)}
    receita["selecao_treino"] = {
        "origem": str(origem), "lista": str(Path(args.lista).resolve(strict=True)),
        "treino_original": len(treino_original), "treino_novo": len(treino_novo),
        "validacao": len(validacao), "teste": len(teste),
        "validacao_teste_identicos": True,
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
    (tmp / "splits" / "treino.txt").write_text(
        "\n".join(treino_novo) + "\n", encoding="utf-8")
    for split in ("validacao", "teste"):
        shutil.copy2(origem / "splits" / (split + ".txt"),
                     tmp / "splits" / (split + ".txt"))
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
          (saida, len(treino_novo), len(validacao), len(teste),
           len(receita["bandas"])), flush=True)


if __name__ == "__main__":
    main()
