#!/usr/bin/env python3
"""Tabela de comparação no TESTE (interior, sem 15 px da borda) entre as três
famílias: vit_model/runs/seg_*, unet_cnn/runs/* e xgboost/runs/* (que inclui os
runs de Random Forest do rf_patches.py).

Uso (no WSL, a partir de Models/):
  ../Data/.venv/bin/python vit_model/comparar_seg.py [--split validacao]
"""

import argparse
import glob
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(BASE)


def fmt(v):
    return "  -  " if v is None else "%.3f" % v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="teste", choices=["teste", "validacao"])
    args = ap.parse_args()

    linhas = []
    for rel in sorted(glob.glob(os.path.join(MODELS, "xgboost", "runs", "*", "relatorio.json"))):
        r = json.load(open(rel, encoding="utf-8"))
        linhas.append(("xgb:" + os.path.basename(os.path.dirname(rel)), r[args.split], r["classes"]))
    for rel in sorted(glob.glob(os.path.join(BASE, "runs", "seg_*", "relatorio.json"))):
        r = json.load(open(rel, encoding="utf-8"))
        linhas.append((os.path.basename(os.path.dirname(rel)), r[args.split]["interior"],
                       r["classes"]))
    # A família CNN mora em unet_cnn/runs/ e ficou de fora deste glob desde
    # sempre: mesmo schema de relatorio.json, mesmo nível "interior".
    for rel in sorted(glob.glob(os.path.join(MODELS, "unet_cnn", "runs", "*", "relatorio.json"))):
        r = json.load(open(rel, encoding="utf-8"))
        linhas.append(("cnn:" + os.path.basename(os.path.dirname(rel)),
                       r[args.split]["interior"], r["classes"]))
    if not linhas:
        print("nenhum relatorio encontrado")
        return

    # runs com classes/cenas diferentes (ex.: dataset de 3 classes x v4 de 4) não
    # cabem na mesma tabela: uma tabela por combinação
    grupos = {}
    for nome, r, classes in linhas:
        grupos.setdefault((tuple(classes), tuple(r["por_cena"])), []).append((nome, r))
    for (classes, cenas), itens in grupos.items():
        abrev = [abreviar(c) for c in classes]
        print("\n%s (interior) — IoU por classe: %s" % (args.split.upper(), ", ".join(
            "%s=%s" % (a, c) for a, c in zip(abrev, classes))))
        cab = "%-26s %6s %6s" % ("modelo", "OA", "mIoU")
        for c in cenas:
            cab += " | %-*s" % (6 + 6 * len(classes), c[:6 + 6 * len(classes)])
        print(cab)
        sub = "%-26s %6s %6s" % ("", "", "")
        for _ in cenas:
            sub += " | %5s %s" % ("OA", " ".join("%5s" % a for a in abrev))
        print(sub)
        for nome, r in itens:
            s = "%-26s %6.3f %6.3f" % (nome[:26], r["oa"], r["miou"])
            for c in cenas:
                pc = r["por_cena"].get(c)
                s += " | %5.3f %s" % (pc["oa"], " ".join(fmt(v) for v in pc["iou"])) if pc else " | -"
            print(s)


def abreviar(classe):
    """'Vegetação não densa' -> 'VnDen', 'Água' -> 'Água' (5 letras, sem repetir)."""
    p = classe.replace("/", " ").split()
    if len(p) == 1:
        return p[0][:5]
    return "".join(w[0] for w in p[:-1]) + p[-1][:5 - (len(p) - 1)].capitalize()


if __name__ == "__main__":
    main()
