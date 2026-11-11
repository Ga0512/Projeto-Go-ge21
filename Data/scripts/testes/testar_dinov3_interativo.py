#!/usr/bin/env python3
"""Editor interactivo de agrupamento de patches com DINOv3.

A ideia é diferente de classificar o tile inteiro: você cria GRUPOS com o
nome que quiser (ex.: "água", "sombra", "floresta"), clica em alguns patches
de cada grupo, e o grupo cresce apenas pelos patches SEMELHANTES aos clicados
(similaridade de cosseno ao centroide do grupo acima de um limiar). O resto
da imagem permanece sem cor — nada vira nada sem a sua escolha.

Uso:
  - Campo "novo grupo" + botão Criar (ou Enter) define um grupo colorido.
  - Selecione o grupo ativo no combo; clique na imagem p/ marcar seeds dele.
  - A cada clique, o overlay mostra os patches similares ao grupo ativo.
  - U desfaz o último seed do grupo ativo · R limpa o grupo ativo.
  - S salva a máscara (índice do grupo) + names.txt em experimentos/.
Saída em experimentos/auto_label_dinov3/ — nada dos dados reais é tocado.

Uso:
  .venv/bin/python scripts/testes/testar_dinov3_interativo.py [--tile r09_c11]
"""
import argparse
import os
import sys

import numpy as np
import torch
from osgeo import gdal

sys.path.insert(0, "/tmp/dinov3")
CKPT = "/mnt/d/Projects/DinoV3/models/dinov3_vits16.pth"
OUT = "experimentos/auto_label_dinov3"
BLOCO = 512
PATCH = 16
EMB = 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
RAIO_SEED = 1
LIMIAR = 0.82             # similaridade mínima ao centroide p/ entrar no grupo
CORES = [(238, 80, 80), (120, 200, 80), (60, 120, 240), (230, 200, 120),
         (200, 100, 220), (80, 200, 240), (240, 150, 60), (150, 150, 150),
         (250, 220, 80), (0, 200, 180)]


def carregar_modelo():
    from dinov3.hub.backbones import dinov3_vits16
    torch.hub.set_dir("/tmp/torchhub")
    m = dinov3_vits16(pretrained=True, weights=CKPT)
    m.eval().cuda()
    return m


def features_tile(model, pan):
    H, W = pan.shape
    f = np.zeros((H // PATCH, W // PATCH, EMB), np.float32)
    for r in range(0, H, BLOCO):
        for c in range(0, W, BLOCO):
            bloco = pan[r:r + BLOCO, c:c + BLOCO]
            rgb = np.stack([bloco] * 3, -1)
            x = (rgb / 255.0 - MEAN) / STD
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().cuda()
            with torch.no_grad():
                o = model.forward_features(x)
            fv = o["x_norm_patchtokens"][0].cpu().numpy()
            f[r // PATCH:(r + BLOCO) // PATCH,
              c // PATCH:(c + BLOCO) // PATCH] = fv.reshape(BLOCO // PATCH,
                                                            BLOCO // PATCH, EMB)
    return f


def normalizar(f):
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", default="r09_c11")
    ap.add_argument("--cena", default="cena_013-014_2007-06-14")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    td = os.path.join("dados/spot5", args.cena, "tiles", args.tile)
    cache = os.path.join(OUT, "feats_%s.npy" % args.tile)

    pds = gdal.Open(os.path.join(td, "imagem.tif"))
    pan = pds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    gt, wkt = pds.GetGeoTransform(), pds.GetProjection()
    pds = None

    if os.path.exists(cache):
        feats = np.load(cache)
        print("features (cache):", feats.shape)
    else:
        model = carregar_modelo()
        print("extraindo features DINOv3 do tile...")
        feats = features_tile(model, pan)
        np.save(cache, feats)
        print("features:", feats.shape, "->", cache)

    H, W = pan.shape
    hp, wp = H // PATCH, W // PATCH
    fn = normalizar(feats.reshape(-1, EMB))

    p2, p98 = np.percentile(pan, [2, 98])
    pg = np.clip((pan - p2) / (p98 - p2), 0, 1)
    g = np.uint8(255 * pg)
    base_gray = np.stack([g, g, g], -1)

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                                   QGraphicsView, QGraphicsScene,
                                   QVBoxLayout, QHBoxLayout, QLabel,
                                   QLineEdit, QPushButton, QComboBox)

    def para_qimage(arr):
        return QImage(arr.data, arr.shape[1], arr.shape[0], 3 * arr.shape[1],
                      QImage.Format_RGB888).copy()

    def overlay_grupos(grupos):
        """Para cada patch: grupo vencedor se similaridade >= LIMIAR. Devolve
        (grade hp,wp com índice do grupo 1..n, 0 = nenhum)."""
        if not grupos:
            return None
        nomes = list(grupos)
        mat = np.stack([grupos[n]["cent"] for n in nomes])      # (n, EMB)
        sim = fn @ mat.T                                        # (N, n)
        best = sim.max(1)
        venc = sim.argmax(1)
        grade = np.where(best >= LIMIAR, venc + 1, 0).astype(np.uint8)
        return grade.reshape(hp, wp)

    class Visao(QGraphicsView):
        def mousePressEvent(self, e):
            pos = self.mapToScene(e.position().toPoint())
            x, y = int(pos.x()), int(pos.y())
            j = self.janela
            px, py = x // PATCH, y // PATCH
            if 0 <= px < wp and 0 <= py < hp:
                nome = j.grupo_ativo
                if nome is None:
                    j.status.setText("crie um grupo antes de clicar")
                    return
                for yy in range(max(0, py - RAIO_SEED), min(hp, py + RAIO_SEED + 1)):
                    for xx in range(max(0, px - RAIO_SEED), min(wp, px + RAIO_SEED + 1)):
                        j.grupos[nome]["seeds"].append(yy * wp + xx)
                j._atualizar_centroide(nome)
                j.status.setText("grupo '%s': %d seeds" % (nome,
                                                           len(j.grupos[nome]["seeds"])))
                j._redraw()
            super().mousePressEvent(e)

    class Janela(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("DINOv3 agrupamento — %s (%s)"
                                % (args.cena, args.tile))
            self.grupos = {}          # nome -> {"seeds": [], "cent": ndarray}
            self.grupo_ativo = None
            self.item = None
            self.view = Visao(self)
            self.view.janela = self
            self.scene = QGraphicsScene()
            self.view.setScene(self.scene)

            central = QWidget()
            lay = QVBoxLayout()
            lay.addWidget(self.view)

            barra = QHBoxLayout()
            self.ed_nome = QLineEdit()
            self.ed_nome.setPlaceholderText("nome do novo grupo… (Enter cria)")
            self.ed_nome.returnPressed.connect(self._criar_grupo)
            self.btn_criar = QPushButton("Criar grupo")
            self.btn_criar.clicked.connect(self._criar_grupo)
            self.cb_grupo = QComboBox()
            self.cb_grupo.currentIndexChanged.connect(self._grupo_ativado)
            self.btn_limpar = QPushButton("Limpar grupo (R)")
            self.btn_limpar.clicked.connect(lambda: self._limpar_grupo())
            self.btn_desfazer = QPushButton("Desfazer seed (U)")
            self.btn_desfazer.clicked.connect(lambda: self._desfazer_seed())
            self.btn_salvar = QPushButton("Salvar (S)")
            self.btn_salvar.clicked.connect(lambda: self._salvar())
            for w in (self.ed_nome, self.btn_criar, self.cb_grupo,
                      self.btn_limpar, self.btn_desfazer, self.btn_salvar):
                barra.addWidget(w)
            barra.addStretch(1)
            lay.addLayout(barra)

            self.status = QLabel("crie um grupo (nome) e clique nos patches dele")
            lay.addWidget(self.status)
            central.setLayout(lay)
            self.setCentralWidget(central)
            self.resize(900, 900)
            self._redraw()

        def _criar_grupo(self):
            nome = self.ed_nome.text().strip()
            if not nome or nome in self.grupos:
                return
            cor = CORES[len(self.grupos) % len(CORES)]
            self.grupos[nome] = {"seeds": [], "cent": None,
                                 "cor": cor, "limiar": LIMIAR}
            self.cb_grupo.addItem(nome)
            self.cb_grupo.setCurrentIndex(self.cb_grupo.count() - 1)
            self.ed_nome.clear()
            self.status.setText("grupo '%s' criado — clique nos patches" % nome)

        def _grupo_ativado(self, *_):
            self.grupo_ativo = self.cb_grupo.currentText() or None

        def _atualizar_centroide(self, nome):
            s = np.array(self.grupos[nome]["seeds"])
            if s.size:
                v = fn[s].mean(0)
                self.grupos[nome]["cent"] = v / (np.linalg.norm(v) + 1e-9)
            else:
                self.grupos[nome]["cent"] = None

        def _desfazer_seed(self):
            if self.grupo_ativo and self.grupos[self.grupo_ativo]["seeds"]:
                self.grupos[self.grupo_ativo]["seeds"].pop()
                self._atualizar_centroide(self.grupo_ativo)
                self._redraw()

        def _limpar_grupo(self):
            if self.grupo_ativo:
                self.grupos[self.grupo_ativo]["seeds"] = []
                self._atualizar_centroide(self.grupo_ativo)
                self._redraw()

        def _redraw(self):
            grade = overlay_grupos(self.grupos)
            arr = base_gray.copy()
            if grade is not None:
                ovb = np.kron(grade, np.ones((PATCH, PATCH), np.uint8))
                for i, nome in enumerate(self.grupos, 1):
                    sel = ovb == i
                    arr[sel] = np.uint8(0.4 * arr[sel] +
                                        0.6 * np.array(self.grupos[nome]["cor"]))
            img = para_qimage(arr)
            if self.item is None:
                self.item = self.scene.addPixmap(QPixmap.fromImage(img))
            else:
                self.item.setPixmap(QPixmap.fromImage(img))
            self.view.fitInView(self.item, Qt.KeepAspectRatio)
            self.view.update()

        def keyPressEvent(self, e):
            k = e.key()
            if k == Qt.Key_S:
                self._salvar()
            elif k == Qt.Key_U:
                self._desfazer_seed()
            elif k == Qt.Key_R:
                self._limpar_grupo()

        def _salvar(self):
            grade = overlay_grupos(self.grupos)
            if grade is None:
                self.status.setText("sem grupos/seeds ainda")
                return
            out = np.kron(grade, np.ones((PATCH, PATCH), np.uint8))
            out = out[:H, :W]
            path = os.path.join(OUT, "grupos_%s.tif" % args.tile)
            d = gdal.GetDriverByName("GTiff").Create(
                path, out.shape[1], out.shape[0], 1, gdal.GDT_Byte,
                ["COMPRESS=DEFLATE", "TILED=YES"])
            d.SetGeoTransform(gt); d.SetProjection(wkt)
            b = d.GetRasterBand(1); b.SetNoDataValue(0); b.WriteArray(out); d = None
            with open(os.path.join(OUT, "names_%s.txt" % args.tile), "w") as fp:
                for i, nome in enumerate(self.grupos, 1):
                    fp.write("%d %s (%d seeds, limiar %.2f)\n"
                             % (i, nome, len(self.grupos[nome]["seeds"]),
                                self.grupos[nome]["limiar"]))
            self.status.setText("salvo: %s" % path)
            print("salvo:", path)

    app = QApplication(sys.argv)
    janela = Janela()
    janela.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()