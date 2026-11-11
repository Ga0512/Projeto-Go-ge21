#!/usr/bin/env python3
"""Presets por sensor: o que muda entre SPOT5 e WorldView é ESCALA FÍSICA.

Os modelos deste repositório foram escritos para a pancromática SPOT5 de 2,5 m.
O WorldView-1 de jataí tem 0,486 m/px nativos -- 5x mais fino -- e isso muda o
significado de todo parâmetro medido em PIXELS:

  janelas [5,15,31] a 2,5 m   = 12,5 / 37,5 / 77,5 m de chão
  janelas [5,15,31] a 0,5 m   =  2,5 /  7,5 / 15,5 m de chão

Ou seja, rodar o preset errado não quebra nada: produz silenciosamente um modelo
que enxerga 5x menos contexto, e nenhum log denuncia. Daí este arquivo, e daí a
`verificar()`.

Por que um arquivo só, e não uma cópia por pasta: o CLAUDE.md registra que
"editar um e esquecer os outros é o erro mais fácil de cometer aqui". Parâmetro
de escala duplicado em seis scripts é exatamente esse erro esperando acontecer.
Os scripts importam daqui com um sys.path.insert no diretório Models/, como o
dados.py já faz com ../Data/scripts/modelos/config.py.

O QUE NÃO ENTRA AQUI: loss, augmentation, métricas e margem de avaliação de
comparação. O CLAUDE.md exige que mexer nisso seja replicado em todas as
famílias, senão os runs antigos deixam de ser comparáveis; se virarem preset,
dois sensores passam a ser incomparáveis sem que nada no log avise.

Uso:
    import sensores
    s = sensores.carregar("wv")               # ou "spot5"
    sensores.verificar(s, args.dataset)       # aborta se a receita discordar
    janelas, margem = s["janelas"], s["margem"]
"""

import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# Resolução de referência: as janelas do SPOT5 são a definição do contexto
# espacial que os modelos usam, e os outros sensores são escalados para cobrir
# o MESMO chão.
_RES_REF = 2.5
_JANELAS_REF = [5, 15, 31]


def _escalar(janelas_ref, res_ref, res):
    """Janelas (ímpares) que cobrem o mesmo chão das de referência."""
    out = []
    for j in janelas_ref:
        n = round(j * res_ref / res)
        out.append(n + 1 if n % 2 == 0 else n)   # janela par não tem centro
    return out


SENSORES = {
    "spot5": {
        "nome": "SPOT5 pancromática",
        "resolucao": 2.5,
        "patch": 256,
        "janelas": [5, 15, 31],
        "lote": 16,                 # U-Net++ base 32 em patch 256: ~5,9 GB VRAM
        "lote_features": 32,        # patches por lote no cálculo das features
        "dataset": "../Data/dados/datasets/spot5_refinado_v18",
    },
    "wv": {
        "nome": "WorldView-1 pancromática",
        # Nativo é 0,485844 m/px. Arredondado para 0,5 por decisão de projeto:
        # a razão para o SPOT5 fica exatamente 5x e as janelas saem inteiras e
        # ímpares. A diferença geométrica é de 2,9% e NÃO há reamostragem dos
        # pixels -- as imagens seguem as nativas.
        "resolucao": 0.5,
        "resolucao_nativa": 0.485844,
        "patch": 512,
        "janelas": [25, 75, 155],   # = [5,15,31] x 5: o mesmo chão do SPOT5
        "lote": 4,                  # patch 512 tem 4x a área de 256
        "lote_features": 8,
        "dataset": "../Data/dados/datasets/wv_jatai_v2",
    },
}

for _s in SENSORES.values():
    # A margem é derivada, nunca digitada: é o raio da maior janela de features,
    # o mínimo para ela caber inteira (ver features.py).
    _s["margem"] = max(_s["janelas"]) // 2
    _s["margem_m"] = _s["margem"] * _s["resolucao"]


def carregar(sensor):
    """-> dict do preset. Erro explícito em nome desconhecido."""
    if sensor not in SENSORES:
        raise SystemExit("sensor '%s' desconhecido; use um de: %s"
                         % (sensor, ", ".join(sorted(SENSORES))))
    return dict(SENSORES[sensor])


def verificar(preset, dataset, tolerancia=0.1):
    """Aborta se a resolução da receita.json do dataset não bater com o preset.

    É a proteção que justifica o `--sensor` ser explícito: escolher na mão, mas
    não conseguir rodar o preset do SPOT5 num dataset WorldView (margem 15 px =
    7,5 m em vez de 77,5 m, sem nenhum sintoma no log).
    """
    cam = os.path.join(dataset, "receita.json")
    if not os.path.exists(cam):
        print("[sensor] %s sem receita.json: resolução não verificada" % dataset,
              flush=True)
        return
    with open(cam, encoding="utf-8") as f:
        p = json.load(f).get("parametros", {})
    res = p.get("resolucao")
    if res is None:
        print("[sensor] receita.json sem 'resolucao': não verificada", flush=True)
        return
    esperada = preset.get("resolucao_nativa", preset["resolucao"])
    if abs(res - esperada) / esperada > tolerancia:
        raise SystemExit(
            "[sensor] preset '%s' espera ~%.3f m/px, mas %s tem %.3f m/px.\n"
            "         Rodar assim daria janelas %s = %.1f m de chão em vez de "
            "%.1f m.\n         Use o preset do sensor certo."
            % (preset["nome"], esperada, dataset, res, preset["janelas"],
               max(preset["janelas"]) * res, max(preset["janelas"]) * esperada))
    patch = p.get("patch")
    if patch and patch != preset["patch"]:
        print("[sensor] aviso: preset com patch %d, dataset com patch %d"
              % (preset["patch"], patch), flush=True)
    print("[sensor] %s | %.3f m/px | patch %d | janelas %s (%.1f m) | margem %d px (%.1f m)"
          % (preset["nome"], res, preset["patch"], preset["janelas"],
             max(preset["janelas"]) * res, preset["margem"], preset["margem_m"]),
          flush=True)


def resumo():
    """Tabela dos presets, para conferência."""
    linhas = ["%-7s %-28s %8s %6s %-16s %8s %6s"
              % ("sensor", "nome", "m/px", "patch", "janelas", "margem", "lote")]
    for k, s in sorted(SENSORES.items()):
        linhas.append("%-7s %-28s %8.3f %6d %-16s %4d px %6d"
                      % (k, s["nome"], s["resolucao"], s["patch"],
                         ",".join(map(str, s["janelas"])), s["margem"], s["lote"]))
        linhas.append("%-7s %-28s %8s %6s janelas cobrem %.1f / %.1f / %.1f m"
                      % ("", "", "", "", *[j * s["resolucao"] for j in s["janelas"]]))
    return "\n".join(linhas)


if __name__ == "__main__":
    print(resumo())
    print("\nconferência: janelas do WV escaladas a partir do SPOT5 =",
          _escalar(_JANELAS_REF, _RES_REF, SENSORES["wv"]["resolucao"]))
