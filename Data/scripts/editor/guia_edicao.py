#!/usr/bin/env python3
"""GUI (PySide6) para corrigir as bordas das classes LULC.

A fonte da verdade é a MÁSCARA RASTER (grade da imagem, códigos MapBiomas,
0 = vazio/sem classe). Os polígonos na tela são só a vista dela: após cada
edição a máscara é revetorizada (gdal.Polygonize + shapely.coverage_simplify),
então as classes formam sempre uma partição 2D — sem sobreposição, sem
fresta, vizinhos da mesma classe se fundem sozinhos.

Editar uma borda empurra o vizinho junto: a área que o polígono invade sai do
vizinho, a área que ele libera vai para a classe vizinha mais próxima.

Correção automática (A, e ao carregar se marcado):
  1. limpar ilhas menores que N ha (gdal.SieveFilter, absorvidas pelo vizinho)
  2. ajustar bordas à imagem: watershed no gradiente da PAN, só numa faixa de
     ±N px em volta das bordas originais (tira o degrau do pixel 30 m)
     + filtro de moda 3x3 categórico
  3. limpar ilhas de novo (o ajuste pode criar farelo)
  4. suavizar escadinhas: simplificação de cobertura (bordas compartilhadas
     simplificadas uma vez só, para os dois lados) rasterizada de volta

Uso:
  .venv/bin/python scripts/editor/guia_edicao.py

Mouse (modo seleção):
  arrastar perto de um vértice   move o vértice (de qualquer polígono, sem
                                 precisar selecionar antes; o vértice
                                 compartilhado com o vizinho anda junto)
  arrastar a borda do selecionado  cria vértice ali e já arrasta
  clique / Ctrl+clique           seleciona / seleção múltipla
  Shift+arrastar                 move o polígono inteiro (ou a seleção)
  direito no vértice             remove o vértice
  roda / meio                    zoom / pan (ou Espaço + esquerdo)
Modos:
  balde (B)     clique pinta a região com a classe ativa
  desenho (D)   forma livre: cliques = vértices, duplo clique/Enter fecha,
                direito/Backspace desfaz ponto; círculo / retângulo /
                polígono regular (N lados): clique no centro (canto) e clique
                de novo, ou arraste. Cria com a classe ativa por cima dos
                vizinhos. Esc cancela.
  colar (Ctrl+V)  a cópia (Ctrl+C) segue o cursor; cada clique cola uma vez

Carregar: se existir mascara_refinada.tif salva, pergunta se continua dela
(sem correção automática). "Abrir máscara…" abre qualquer máscara salva.
Ao fechar com edições não salvas, pergunta se quer salvar.

Atalhos:
  1..9         classe ativa (e aplica na seleção, se houver)
  C            escolher qualquer código (classe ativa + seleção)
  B            liga/desliga o balde de preenchimento
  D            liga/desliga o desenho de polígono
  Ctrl+C/V     copiar seleção / colar
  A            correção automática (ilhas + ajuste à imagem + suavização)
  N            ir para o próximo vazio (seleciona e centraliza)
  Del          apagar seleção (vira vazio)
  Esc          cancelar desenho / sair do modo / limpar seleção
  F            ajustar janela ao conteúdo
  T            alternar opacidade do overlay de polígonos (transparente)
  G            liga/desliga o fundo PAN
  R            reverter para a máscara em disco
  Ctrl+Z/Y     desfazer / refazer
  Ctrl+S       salvar máscara refinada (GeoTIFF Byte, grade original)
"""

import json
import math
import os
import sys
import time

import numpy as np

try:
    from osgeo import gdal, ogr, osr
except ImportError:
    print("osgeo indisponivel (use o venv do projeto)", file=sys.stderr)
    sys.exit(1)

import shapely
import shapely.affinity
from scipy import ndimage
from skimage.segmentation import watershed

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (QImage, QColor, QBrush, QPen, QPainter, QPainterPath,
                           QPolygonF, QPixmap)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPathItem, QGraphicsRectItem, QGraphicsItem, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QCheckBox, QComboBox, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QInputDialog,
    QScrollArea, QDialog)

gdal.UseExceptions()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import caminhos  # noqa: E402

# cenas editáveis. `mascara` = MapBiomas original na grade da imagem;
# `refinada` = máscara de trabalho (cena inteira, 1 banda) — criada no 1º
# salvar como cópia da original; cada salvar grava só o recorte editado.
CENAS = [
    {"nome": "SPOT5 10/09/2007 (recorte)",
     "imagem": caminhos.SPOT5_RECORTE_IMAGEM,
     "mascara": os.path.join(caminhos.SPOT5_RECORTE_MB_2007,
                             "PAR_SPOT5_PAN_MASK_2007.tif"),
     "mascara_banda": 2,
     "refinada": caminhos.SPOT5_RECORTE_REFINADA},
    # cena grande cortada em tiles (scripts/spot5/tiles_cena.py): cada
    # tile tem imagem/mascara_mapbiomas/mascara_refinada na sua pasta; os
    # .vrt da pasta dão a cena inteira (contexto para a correção automática)
    {"nome": "SPOT5 013-014 14/06/2007 (tiles)",
     "tiles": caminhos.SPOT5_CENA_013_TILES,
     "imagem": os.path.join(caminhos.SPOT5_CENA_013_TILES, "imagem.vrt"),
     "mascara": os.path.join(caminhos.SPOT5_CENA_013_TILES,
                             "mascara_mapbiomas.vrt"),
     "mascara_banda": 1,
     "refinada": None},
    {"nome": "SPOT5 Colinas do Sul (tiles)",
     "tiles": caminhos.SPOT5_COLINAS_TILES,
     "imagem": os.path.join(caminhos.SPOT5_COLINAS_TILES, "imagem.vrt"),
     "mascara": os.path.join(caminhos.SPOT5_COLINAS_TILES,
                             "mascara_mapbiomas.vrt"),
     "mascara_banda": 1,
     "refinada": None},
    {"nome": "SPOT5 Jataí (tiles)",
     "tiles": caminhos.SPOT5_JATAI_TILES,
     "imagem": os.path.join(caminhos.SPOT5_JATAI_TILES, "imagem.vrt"),
     "mascara": os.path.join(caminhos.SPOT5_JATAI_TILES,
                             "mascara_mapbiomas.vrt"),
     "mascara_banda": 1,
     "refinada": None},
]

RECORTE_PX = 3072          # lado do recorte de edição em cenas grandes
RECORTE_MAX_PX = 16_000_000  # cena até esse nº de px abre inteira
MARGEM_AUTO = 96           # px de vizinhança que a correção automática enxerga
FORA = 255                 # fora da imagem (nodata): travado, gravado como 0

# código MapBiomas -> classe final: o agrupamento do modelo (scripts/modelos/config.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelos"))
import config  # noqa: E402
CLASS_MAP = config.CLASS_MAP
CLASS_NOMES = config.CLASS_NOMES

# legenda oficial MapBiomas (código -> nome, cor hex)
LEGEND = {
    3: ("Formação Florestal", "#1f8d49"),
    4: ("Formação Savânica", "#7dc975"),
    9: ("Silvicultura", "#7a5900"),
    11: ("Campo Alagado e Área Pantanosa", "#519799"),
    12: ("Formação Campestre", "#d6bc74"),
    15: ("Pastagem", "#edde8e"),
    19: ("Lavoura Temporária", "#c27ba0"),
    20: ("Cana", "#db7093"),
    21: ("Mosaico de Usos", "#ffefc3"),
    24: ("Área Urbanizada", "#d4271e"),
    25: ("Outras Áreas não Vegetadas", "#db4d4f"),
    29: ("Afloramento Rochoso", "#ffaa5f"),
    30: ("Mineração", "#9c0027"),
    33: ("Rio, Lago e Oceano", "#2532e4"),
    36: ("Lavoura Perene", "#d082de"),
    39: ("Soja", "#f5b3c8"),
    40: ("Arroz", "#c71585"),
    41: ("Outras Lavouras Temporárias", "#f54ca9"),
    62: ("Algodão beta", "#ff69b4"),
}

# teclas 1..9 -> código MapBiomas
TECLAS_RAPIDAS = {1: 3, 2: 4, 3: 11, 4: 12, 5: 15, 6: 19, 7: 21, 8: 25, 9: 33}

# defaults da correção automática
ILHA_HA = 1.0         # ilhas menores que isso são absorvidas pelo vizinho
FAIXA_PX = 3          # ±px em volta da borda que o watershed pode redecidir
SUAVIZA_PX = 3        # tolerância da simplificação de cobertura (≈ raiz da área
                      # do triângulo removido; 6 px de degrau some com ~3)
TOL_TELA = 1.0        # simplificação da vista: só tira o serrilhado de 1 px;
                      # maior que isso engole os ajustes finos de vértice
VAZIO_FILL_PX = 36    # ao salvar, vazios < isso herdam a classe do vizinho
MAX_UNDO = 20

COR_VAZIO = QColor(255, 0, 255)
ID_GT = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # rasters de trabalho em coords de pixel


# ------------------------------------------------------------------ raster

def _mem(arr):
    """Dataset MEM em coordenadas de pixel (y para baixo) com `arr`."""
    ds = gdal.GetDriverByName("MEM").Create("", arr.shape[1], arr.shape[0], 1,
                                            gdal.GDT_Byte)
    ds.SetGeoTransform(ID_GT)
    ds.GetRasterBand(1).WriteArray(arr)
    return ds


def limpar_ilhas(R, n_px):
    """Regiões conexas (8) menores que n_px são absorvidas pelo vizinho."""
    if n_px < 2:
        return R
    ds = _mem(R)
    b = ds.GetRasterBand(1)
    gdal.SieveFilter(b, None, b, int(n_px), 8)
    return b.ReadAsArray()


def moda3(R):
    """Filtro de moda 3x3 categórico (nunca inventa código); vazio fica."""
    out = R.copy()
    best = np.zeros(R.shape, np.float32)
    for c in np.unique(R):
        if c == 0:
            continue
        f = ndimage.uniform_filter((R == c).astype(np.float32), 3)
        s = f > best + 1e-6
        out[s] = c
        best[s] = f[s]
    out[R == 0] = 0
    return out


def gradiente_pan(pan):
    """Gradiente da PAN (uint8, 0..255 no p99) para o watershed."""
    g = ndimage.gaussian_gradient_magnitude(pan.astype(np.float32), 1.0)
    p = np.percentile(g, 99)
    return np.clip(g / (p if p > 0 else 1) * 255, 0, 255).astype(np.uint8)


def ajustar_bordas(R, grad, faixa):
    """Watershed no gradiente da PAN. Sementes = interior de cada região
    (erosão de `faixa` px); regiões finas demais para sobreviver à erosão
    viram semente inteiras (não somem). Só a faixa em volta das bordas é
    redecidida, então a borda anda no máximo ~2*faixa px."""
    if faixa < 1:
        return R
    mk = np.zeros(R.shape, np.int32)
    est = np.ones((3, 3), bool)
    for c in np.unique(R):
        if c == 0:
            continue
        m = R == c
        er = ndimage.binary_erosion(m, iterations=int(faixa))
        lab, n = ndimage.label(m, est)
        tem = np.zeros(n + 1, bool)
        tem[np.unique(lab[er])] = True
        tem[0] = False
        mk[er | (m & ~tem[lab])] = int(c)
    # vazio e fora da imagem ficam fora do watershed (não avançam sobre a imagem)
    livre = (R != 0) & (R != FORA)
    mk[~livre] = 0
    out = watershed(grad, markers=mk, mask=livre).astype(np.uint8)
    out[~livre] = R[~livre]
    return moda3(out)


def vetorizar(R, tol):
    """Máscara -> (códigos, polígonos shapely em px). Com tol > 0 aplica
    simplificação de COBERTURA: cada borda compartilhada é simplificada uma
    vez, igual para os dois lados — continua partição, sem fresta."""
    ds = _mem(R)
    vds = ogr.GetDriverByName("MEMORY").CreateDataSource("vet")
    lyr = vds.CreateLayer("p", None, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))
    gdal.Polygonize(ds.GetRasterBand(1), None, lyr, 0, ["8CONNECTED=8"])
    cods, wkbs = [], []
    for f in lyr:
        cods.append(f.GetField("DN"))
        wkbs.append(bytes(f.GetGeometryRef().ExportToWkb()))
    geoms = shapely.from_wkb(wkbs)
    if tol > 0 and len(geoms):
        geoms = shapely.coverage_simplify(geoms, tol)
    return np.array(cods, dtype=np.int32), geoms


def rasterizar(cods, geoms, shape):
    """Polígonos -> máscara (0 onde nada cobre); o último vence."""
    ds = _mem(np.zeros(shape, np.uint8))
    vds = ogr.GetDriverByName("MEMORY").CreateDataSource("ras")
    lyr = vds.CreateLayer("p", None, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("c", ogr.OFTInteger))
    for c, w in zip(cods, shapely.to_wkb(geoms)):
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetField("c", int(c))
        f.SetGeometry(ogr.CreateGeometryFromWkb(w))
        lyr.CreateFeature(f)
    gdal.RasterizeLayer(ds, [1], lyr, options=["ATTRIBUTE=c"])
    return ds.GetRasterBand(1).ReadAsArray()


def mascara_geom(geom, shape):
    """Pixels (bool) cujo centro cai dentro de `geom`."""
    return rasterizar([1], [geom], shape).astype(bool)


def suavizar(R, tol):
    if tol <= 0:
        return R
    return rasterizar(*vetorizar(R, tol), R.shape)


def auto_corrigir(R, grad, ilha_px, faixa, tol):
    R = limpar_ilhas(R, ilha_px)
    if grad is not None and faixa > 0:
        R = limpar_ilhas(ajustar_bordas(R, grad, faixa), ilha_px)
    return suavizar(R, tol)


def componente(R, geom, cod, folga):
    """Região conexa de `cod` em R que o polígono `geom` representa."""
    h, w = R.shape
    x0, y0, x1, y1 = geom.bounds
    ys = slice(max(0, int(y0) - folga), min(h, int(math.ceil(y1)) + folga))
    xs = slice(max(0, int(x0) - folga), min(w, int(math.ceil(x1)) + folga))
    pix = mascara_geom(geom, R.shape)[ys, xs]
    lab, _ = ndimage.label(R[ys, xs] == cod, np.ones((3, 3), bool))
    comp = np.zeros(R.shape, bool)
    cont = np.bincount(lab[pix])
    cont[0] = 0
    if cont.any():
        # um polígono = uma região; a borda suavizada pode encostar numa
        # região vizinha da mesma classe — fica só a de maior sobreposição
        comp[ys, xs] = lab == int(np.argmax(cont))
    return comp


def aplicar_forma(R, velho, novo, cod):
    """O polígono passa a ocupar `novo`: invade vizinhos; o que ele liberou
    (velho & ~novo) vai para a classe vizinha mais próxima."""
    R = R.copy()
    R[novo] = cod
    lib = velho & ~novo
    if not lib.any():
        return R
    yy, xx = np.nonzero(lib)
    m = 64
    ys = slice(max(0, yy.min() - m), yy.max() + m + 1)
    xs = slice(max(0, xx.min() - m), xx.max() + m + 1)
    lw, Rw = lib[ys, xs], R[ys, xs]
    fonte = (Rw != 0) & (Rw != FORA) & ~lw & ~novo[ys, xs]
    if fonte.any():
        iy, ix = ndimage.distance_transform_edt(
            ~fonte, return_distances=False, return_indices=True)
        Rw[lw] = Rw[iy[lw], ix[lw]]
    else:
        Rw[lw] = 0
    return R


def preencher_vazios_pequenos(R, n_px):
    """Vazios < n_px herdam a classe do vizinho (só mexe nos zeros)."""
    vazio = R == 0
    if not vazio.any():
        return R
    return np.where(vazio, limpar_ilhas(R, n_px), R).astype(np.uint8)


def m2_por_pixel(gt, proj, h):
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    if srs.IsGeographic():
        lat = math.radians(gt[3] + gt[5] * h / 2)
        return abs(gt[1]) * 111320 * math.cos(lat) * abs(gt[5]) * 110574
    return abs(gt[1] * gt[5])


def _valido(arr, nodata):
    """Pixels com imagem. Só o nodata DECLARADO no arquivo conta como fora."""
    if nodata is None:
        return np.ones(arr.shape, bool)
    return arr != nodata


def _qimage_pan(arr, valido):
    """PAN como QImage esticada p2-p98 (só pixels válidos entram na conta)."""
    v = arr[valido]
    mn, mx = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)
    g = np.clip((arr - mn) / (mx - mn), 0, 1) * 255 if mx > mn else arr
    g = np.where(valido, g, 0).astype(np.uint8)
    h, w = g.shape
    return QImage(g.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()


# ------------------------------------------------------------------ vista

def _pen(cor, largura=1.0, estilo=Qt.SolidLine):
    """Pen cosmética: espessura constante na tela, qualquer zoom."""
    p = QPen(cor, largura, estilo)
    p.setCosmetic(True)
    return p


def _cor_do_codigo(codigo):
    return QColor(LEGEND.get(codigo, ("", "#888888"))[1])


def _nome_codigo(codigo):
    return LEGEND.get(codigo, ("Cod. %d" % codigo, ""))[0]


def _anel(coords):
    return QPolygonF([QPointF(x, y) for x, y in coords])


PEGA_VERTICE_PX = 8   # raio (na tela) para pegar um vértice com o mouse
PEGA_BORDA_PX = 6     # raio (na tela) para pegar a borda do selecionado
FORMAS = [("livre", "Livre (N vértices)"), ("circulo", "Círculo"),
          ("retangulo", "Retângulo"), ("regular", "Polígono regular")]


def forma_geom(forma, p0, p1, n_lados=6):
    """Geometria de um desenho de 2 pontos (centro/canto -> cursor)."""
    (x0, y0), (x1, y1) = p0, p1
    r = math.hypot(x1 - x0, y1 - y0)
    if forma == "retangulo":
        return shapely.box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if r == 0:
        return shapely.Polygon()
    if forma == "circulo":
        return shapely.Point(x0, y0).buffer(r, quad_segs=16)
    a0 = math.atan2(y1 - y0, x1 - x0)
    return shapely.Polygon([(x0 + r * math.cos(a0 + 2 * math.pi * k / n_lados),
                             y0 + r * math.sin(a0 + 2 * math.pi * k / n_lados))
                            for k in range(n_lados)])


def _path_geoms(geoms):
    """QPainterPath (com buracos) de uma lista de polígonos shapely."""
    path = QPainterPath()
    path.setFillRule(Qt.OddEvenFill)
    for g in geoms:
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon" or p.is_empty:
                continue
            path.addPolygon(_anel(p.exterior.coords))
            for r in p.interiors:
                path.addPolygon(_anel(r.coords))
    return path


class ItemRegiao(QGraphicsPathItem):
    """Um polígono da vista. Não é arrastável pelo Qt: a Cena trata vértice,
    borda e Shift+arrasto do corpo. Marcadores de vértice aparecem quando
    selecionado. Edições marcam `sujo`; a Janela as aplica na máscara."""

    def __init__(self, cod, geom, alpha):
        super().__init__()
        self.cod = int(cod)
        self.geom = geom
        self.vazio = self.cod == 0
        self.sujo = False
        self._ext = np.asarray(geom.exterior.coords, dtype=np.float64)[:-1]
        self._holes = [np.asarray(r.coords) for r in geom.interiors]
        self._marcas = []
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self._montar()
        self.estilo(alpha)

    def _montar(self):
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)   # buracos ficam vazados (e clicáveis)
        pts = _anel(self._ext)
        pts.append(pts[0])
        path.addPolygon(pts)
        for h in self._holes:
            path.addPolygon(_anel(h))
        self.setPath(path)

    def estilo(self, alpha=None):
        if alpha is not None:
            self._alpha = alpha
        sel = self.isSelected()
        if self.vazio:
            self.setZValue(0.5)
            self.setPen(_pen(COR_VAZIO, 2 if sel else 1, Qt.DashLine))
            self.setBrush(QBrush(COR_VAZIO, Qt.BDiagPattern))
        else:
            self.setZValue(2 if sel else 1)
            self.setPen(_pen(QColor(255, 0, 0), 2) if sel
                        else _pen(QColor(0, 0, 0), 1))
            cor = _cor_do_codigo(self.cod)
            cor.setAlphaF(self._alpha)
            self.setBrush(QBrush(cor))

    # ---- marcadores de vértice (só visuais) ----
    def _criar_marcas(self):
        self._remover_marcas()
        for x, y in self._ext:
            self._marcas.append(MarcaVertice(x, y, self))

    def _remover_marcas(self):
        for m in self._marcas:
            m.setParentItem(None)
            if m.scene() is not None:
                m.scene().removeItem(m)
        self._marcas = []

    # ---- edição ----
    def vertice_movido(self, i, x, y):
        self._ext[i] = (x, y)
        self._montar()
        if i < len(self._marcas):
            self._marcas[i].setPos(QPointF(x, y))
        self.sujo = True

    def aresta_proxima(self, q):
        """(índice da aresta, distância, ponto projetado) mais próximos de q."""
        a = self._ext
        d = np.roll(a, -1, axis=0) - a
        l2 = (d ** 2).sum(1)
        t = np.clip(((q - a) * d).sum(1) / np.where(l2 == 0, 1, l2), 0, 1)
        proj = a + t[:, None] * d
        dist = np.hypot(*(proj - q).T)
        i = int(np.argmin(dist))
        return i, float(dist[i]), proj[i]

    def inserir_vertice(self, q, tol):
        """Insere vértice na aresta mais próxima de q (se <= tol). Devolve o
        índice do vértice novo ou None."""
        i, dist, _ = self.aresta_proxima(np.asarray(q, float))
        if dist > tol:
            return None
        self._ext = np.insert(self._ext, i + 1, q, axis=0)
        self._montar()
        if self.isSelected():
            self._criar_marcas()
        self.sujo = True
        return i + 1

    def remover_vertice(self, i):
        if len(self._ext) <= 3:
            return False
        self._ext = np.delete(self._ext, i, axis=0)
        self._montar()
        if self.isSelected():
            self._criar_marcas()
        self.sujo = True
        return True

    def geom_atual(self):
        """Forma editada, em coords da cena (válida)."""
        o = np.array([self.pos().x(), self.pos().y()])
        g = shapely.Polygon(self._ext + o, [h + o for h in self._holes])
        return shapely.make_valid(g)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.estilo()
            if self.isSelected() and not self.vazio:
                self._criar_marcas()
            else:
                self._remover_marcas()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        cena = self.scene()
        if isinstance(cena, Cena):
            cena.janela.mostrar_info(self)
        super().hoverEnterEvent(event)


class MarcaVertice(QGraphicsRectItem):
    """Marcador de vértice (tamanho fixo na tela). Não recebe mouse: quem
    pega o vértice é a Cena, por proximidade."""

    LADO = 7.0

    def __init__(self, x, y, parent):
        s = self.LADO
        super().__init__(-s / 2, -s / 2, s, s, parent)
        self.setPos(QPointF(float(x), float(y)))
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(_pen(QColor(255, 0, 0), 1.5))
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(10)


class Cena(QGraphicsScene):
    def __init__(self, janela, pan_img, w, h):
        super().__init__()
        self.janela = janela
        self.setSceneRect(QRectF(0, 0, w, h))
        self.pix = self.addPixmap(QPixmap.fromImage(pan_img))
        self.pix.setZValue(-1)
        self.itens = []
        self.vazios = []
        # desenho: pontos clicados + prévia; colar: prévia segue o cursor
        self.pontos = []
        self._cursor = None
        self.rascunho = QGraphicsPathItem()
        self.rascunho.setPen(_pen(QColor(255, 230, 0), 2, Qt.DashLine))
        self.rascunho.setZValue(20)
        self.rascunho.setAcceptedMouseButtons(Qt.NoButton)
        self.addItem(self.rascunho)
        # arrasto em andamento: ("vertice", [(item, i), ...]) ou
        # ("corpo", itens, ponto_inicial)
        self._arrasto = None

    # ---- utilidades ----
    def _transform(self):
        v = self.views()
        return v[0].transform() if v else None

    def _escala(self):
        t = self._transform()
        return t.m11() if t is not None else 1.0

    def regiao_em(self, pos):
        for it in self.items(pos, Qt.IntersectsItemShape, Qt.DescendingOrder,
                             self._transform()):
            if isinstance(it, ItemRegiao):
                return it
        return None

    def selecionados(self):
        return [i for i in self.selectedItems() if isinstance(i, ItemRegiao)]

    def vertice_em(self, pos):
        """Vértice mais próximo de pos (até PEGA_VERTICE_PX na tela) entre os
        polígonos editáveis ali perto. Devolve [(item, i), ...] com TODOS os
        polígonos que têm um vértice nesse mesmo ponto (borda compartilhada
        anda junto), o selecionado/de cima primeiro; ou []."""
        tol = PEGA_VERTICE_PX / self._escala()
        caixa = QRectF(pos.x() - tol, pos.y() - tol, 2 * tol, 2 * tol)
        q = np.array([pos.x(), pos.y()])
        cand = [it for it in self.items(caixa, Qt.IntersectsItemBoundingRect,
                                        Qt.DescendingOrder)
                if isinstance(it, ItemRegiao) and not it.vazio]
        cand.sort(key=lambda it: not it.isSelected())
        melhor = None
        for it in cand:
            d = np.hypot(*(it._ext - q).T)
            i = int(np.argmin(d))
            if d[i] <= tol and (melhor is None or d[i] < melhor[2] - 1e-9):
                melhor = (it, i, d[i])
        if melhor is None:
            return []
        it0, i0, _ = melhor
        alvo = it0._ext[i0].copy()
        grupo = [(it0, i0)]
        for it in cand:
            if it is it0:
                continue
            d = np.hypot(*(it._ext - alvo).T)
            i = int(np.argmin(d))
            if d[i] < 0.5:
                grupo.append((it, i))
        return grupo

    # ---- rascunho (desenho / colar) ----
    def _atualizar_rascunho(self):
        j = self.janela
        path = QPainterPath()
        cor = _cor_do_codigo(j.classe_ativa())
        if j.modo == "colar" and self._cursor is not None and j.clipboard:
            path = _path_geoms(j.geoms_colados(self._cursor))
            cor = _cor_do_codigo(j.clipboard[0][0])
        elif j.modo == "desenho" and self.pontos:
            forma = j.forma()
            if forma == "livre":
                pts = [QPointF(x, y) for x, y in self.pontos]
                if self._cursor is not None:
                    pts.append(self._cursor)
                path.addPolygon(QPolygonF(pts + ([pts[0]] if len(pts) > 2
                                                 else [])))
            elif self._cursor is not None:
                c = self._cursor
                path = _path_geoms([forma_geom(forma, self.pontos[0],
                                               (c.x(), c.y()), j.n_lados())])
            r = 3.0 / self._escala()
            for x, y in self.pontos:
                path.addEllipse(QPointF(x, y), r, r)
        cor.setAlphaF(0.4)
        self.rascunho.setBrush(QBrush(cor))
        self.rascunho.setPath(path)

    def cancelar_desenho(self):
        self.pontos, self._cursor = [], None
        self._atualizar_rascunho()

    def desfazer_ponto(self):
        if self.pontos:
            self.pontos.pop()
            self._atualizar_rascunho()

    def terminar_desenho(self, fim=None):
        """Fecha o desenho. Forma livre: usa os pontos clicados. Outras
        formas: 1º ponto + `fim` (2º clique ou soltar do arrasto)."""
        j = self.janela
        pts, self.pontos = self.pontos, []
        self._atualizar_rascunho()
        forma = j.forma()
        if forma == "livre":
            if len(pts) >= 3:
                j.criar_geom(shapely.Polygon(pts))
            else:
                j.status.setText("Desenho precisa de pelo menos 3 pontos")
        elif pts and fim is not None:
            j.criar_geom(forma_geom(forma, pts[0], fim, j.n_lados()))

    # ---- mouse ----
    def mousePressEvent(self, event):
        j = self.janela
        pos = event.scenePos()
        bt = event.button()
        self._press_tela = event.screenPos()
        if j.modo == "colar":
            if bt == Qt.LeftButton:
                j.colar_em(pos)
            elif bt == Qt.RightButton:
                j.set_modo(None)
            event.accept()
            return
        if j.modo == "desenho":
            if bt == Qt.LeftButton:
                if j.forma() != "livre" and self.pontos:
                    self.terminar_desenho((pos.x(), pos.y()))
                else:
                    self.pontos.append((pos.x(), pos.y()))
            elif bt == Qt.RightButton:
                self.desfazer_ponto()
            self._atualizar_rascunho()
            event.accept()
            return
        if j.modo == "balde":
            if bt == Qt.LeftButton:
                it = self.regiao_em(pos)
                if it is not None:
                    j.pintar([it], j.classe_ativa())
            event.accept()
            return

        # ---- modo seleção/edição ----
        grupo = self.vertice_em(pos)
        if bt == Qt.RightButton:
            if grupo:
                it, i = grupo[0]
                if it.remover_vertice(i):
                    j.aplicar_edicoes([it])
            event.accept()
            return
        if bt != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        ctrl = bool(event.modifiers() & Qt.ControlModifier)
        if grupo and not shift:
            dono = grupo[0][0]
            if not dono.isSelected() and not ctrl:
                self.clearSelection()
            dono.setSelected(True)
            self._arrasto = ("vertice", grupo)
            event.accept()
            return
        # borda do polígono selecionado: cria vértice e já arrasta
        tol = PEGA_BORDA_PX / self._escala()
        q = np.array([pos.x(), pos.y()])
        for it in self.selecionados():
            if it.vazio or shift:
                continue
            i = it.inserir_vertice(q, tol)
            if i is not None:
                self._arrasto = ("vertice", [(it, i)])
                event.accept()
                return
        it = self.regiao_em(pos)
        if shift and it is not None and not it.vazio:
            if not it.isSelected():
                if not ctrl:
                    self.clearSelection()
                it.setSelected(True)
            corpos = [s for s in self.selecionados() if not s.vazio]
            self._arrasto = ("corpo", corpos, pos,
                             {s: s.pos() for s in corpos})
            event.accept()
            return
        super().mousePressEvent(event)   # seleção normal (clique/Ctrl+clique)

    def mouseMoveEvent(self, event):
        j = self.janela
        pos = event.scenePos()
        if j.modo in ("desenho", "colar"):
            self._cursor = pos
            self._atualizar_rascunho()
        elif self._arrasto is not None:
            if self._arrasto[0] == "vertice":
                for it, i in self._arrasto[1]:
                    it.vertice_movido(i, pos.x(), pos.y())
            else:
                _, corpos, p0, orig = self._arrasto
                d = pos - p0
                for s in corpos:
                    s.setPos(orig[s] + d)
                    s.sujo = True
            event.accept()
            return
        elif j.modo is None and not event.buttons():
            v = self.views()
            if v:
                v[0].viewport().setCursor(
                    Qt.SizeAllCursor if self.vertice_em(pos) else Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        j = self.janela
        if j.modo == "desenho" and event.button() == Qt.LeftButton and \
                j.forma() != "livre" and self.pontos:
            # arrastar-e-soltar também fecha círculo/retângulo/regular
            d = event.screenPos() - self._press_tela
            if abs(d.x()) + abs(d.y()) > 6:
                p = event.scenePos()
                self.terminar_desenho((p.x(), p.y()))
            event.accept()
            return
        if self._arrasto is not None:
            self._arrasto = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        sujos = [i for i in self.itens if i.sujo]
        if sujos:
            j.aplicar_edicoes(sujos)

    def mouseDoubleClickEvent(self, event):
        j = self.janela
        if j.modo == "desenho":
            # o 1º clique do duplo já marcou o último ponto
            if event.button() == Qt.LeftButton and j.forma() == "livre":
                self.terminar_desenho()
            event.accept()
            return
        if j.modo is not None:
            event.accept()
            return
        sel = [i for i in self.selecionados() if not i.vazio]
        if event.button() == Qt.LeftButton and len(sel) == 1:
            p = event.scenePos()
            if sel[0].inserir_vertice((p.x(), p.y()),
                                      10.0 / self._escala()) is not None:
                j.aplicar_edicoes(sel)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def popular(self, cods, geoms, alpha):
        self._arrasto = None
        self.clearSelection()
        for it in self.itens:
            it._remover_marcas()
            self.removeItem(it)
        self.itens, self.vazios = [], []
        areas = shapely.area(geoms)
        for k in np.argsort(-areas):
            if cods[k] == FORA:       # fora da imagem: não desenha nem edita
                continue
            for g in getattr(geoms[k], "geoms", [geoms[k]]):
                if g.geom_type != "Polygon" or g.is_empty:
                    continue
                it = ItemRegiao(cods[k], g, alpha)
                self.addItem(it)
                self.itens.append(it)
                if it.vazio:
                    self.vazios.append(it)


class ViewZoom(QGraphicsView):
    """QGraphicsView com zoom pela roda (no cursor) e pan pelo mouse do meio
    (ou espaço + botão esquerdo; o esquerdo sozinho edita polígonos)."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._fator = 1.25
        self._pan_modo = False
        self._espaco = False

    def wheelEvent(self, event):
        f = self._fator if event.angleDelta().y() > 0 else 1 / self._fator
        self.scale(f, f)

    def _ativa_pan(self):
        if not self._pan_modo:
            self._pan_modo = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setInteractive(False)

    def _desativa_pan(self):
        if self._pan_modo:
            self._pan_modo = False
            self.setDragMode(QGraphicsView.NoDrag)
            self.setInteractive(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._espaco = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._espaco = False
            if self._pan_modo and not QApplication.mouseButtons() & Qt.MiddleButton:
                self._desativa_pan()
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or \
                (event.button() == Qt.LeftButton and self._espaco):
            self._ativa_pan()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MiddleButton:
            self._desativa_pan()
        elif event.button() == Qt.LeftButton and self._espaco:
            if not QApplication.mouseButtons() & Qt.MiddleButton:
                self._desativa_pan()


AJUDA = (
    "<b>Atalhos</b><br>"
    "1–9 classe ativa (+ aplica na seleção) · C outro código<br>"
    "B balde: clique pinta a região · Esc sai<br>"
    "D desenhar (forma no seletor) · Esc cancela<br>"
    "Ctrl+C copiar · Ctrl+V colar (clique posiciona)<br>"
    "Arrastar vértice: move · arrastar borda: + vértice<br>"
    "Direito no vértice: − vértice · Shift+arrastar: move polígono<br>"
    "Ctrl+clique: multi · Del apagar<br>"
    "A correção automática · N próximo vazio<br>"
    "Roda: zoom · Meio/Espaço+arrastar: pan<br>"
    "T overlay · G fundo PAN · F ajustar<br>"
    "Ctrl+Z/Y desfazer/refazer · R reverter · Ctrl+S salvar"
)


class _CenaRecortes(QGraphicsScene):
    def __init__(self, dialogo):
        super().__init__(dialogo)
        self.dialogo = dialogo

    def mousePressEvent(self, event):
        for it in self.items(event.scenePos()):
            g = it.data(0)
            if g:
                self.dialogo.escolher(g)
                return


class _VistaAjustada(QGraphicsView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)


class DialogoRecortes(QDialog):
    """Mapa da cena reduzida com a grade de recortes; clique abre um."""

    def __init__(self, janela, grade, editados, atual):
        super().__init__(janela)
        self.setWindowTitle("Recortes — %s" % janela._cena()["nome"])
        self.escolhido = None
        ov = janela._overview()
        s = ov["escala"]
        cena = _CenaRecortes(self)
        cena.addPixmap(QPixmap.fromImage(ov["img"]))
        todos = "*" in editados
        n_ed = 0
        for k, g in enumerate(grade):
            chave, x0, y0, w, h = g
            r = QRectF(x0 / s, y0 / s, w / s, h / s)
            item = QGraphicsRectItem(r)
            if not janela._ov_valido(g):
                item.setPen(_pen(QColor(120, 120, 120), 1, Qt.DotLine))
                cena.addItem(item)
                continue
            ed = todos or chave in editados
            n_ed += ed
            cor = QColor(0, 200, 0) if ed else QColor(255, 210, 0)
            fundo = QColor(40, 120, 255, 110) if chave == atual else \
                QColor(cor.red(), cor.green(), cor.blue(), 55 if ed else 0)
            item.setPen(_pen(cor, 2))
            item.setBrush(QBrush(fundo))
            item.setData(0, g)
            item.setToolTip("recorte %s — %s" % (chave, "salvo" if ed
                                                  else "não editado"))
            cena.addItem(item)
            txt = cena.addSimpleText(str(k + 1))
            txt.setBrush(QBrush(QColor(255, 255, 255)))
            txt.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            txt.setPos(r.topLeft() + QPointF(2, 2))
        vista = _VistaAjustada(cena)
        vista.setRenderHint(QPainter.SmoothPixmapTransform)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Clique num recorte para editar. <b style='color:#0a0'>verde</b> = "
            "já salvo (%d) · <b style='color:#c90'>amarelo</b> = não editado "
            "(abre com a correção automática, se marcada) · "
            "<b style='color:#27f'>azul</b> = atual · pontilhado = sem imagem"
            % n_ed))
        lay.addWidget(vista, 1)
        self.resize(1000, 900)

    def escolher(self, g):
        self.escolhido = g
        self.accept()


class Janela(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Corretor de bordas LULC")
        self.setMinimumSize(1100, 720)
        self.R = None            # máscara de trabalho (h, w) uint8
        self.gt = self.proj = None
        self._pan_img = None
        self._salvo = None       # R do último salvar/abrir (p/ avisar ao sair)
        self._modo = None
        self.clipboard = []      # [(cod, geom centrada na origem)]
        self.recorte = None      # (chave, x0, y0, w, h) em px da cena
        self.valido = None       # pixels do recorte com imagem
        self._undo, self._redo = [], []
        self._op_ant = 55
        self._i_vazio = -1
        self._m2px = 25.0

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)

        self.scene = QGraphicsScene()
        self.view = ViewZoom(self.scene)

        painel = QVBoxLayout()

        # --- classe ativa + balde
        painel.addWidget(QLabel("<b>Classe ativa</b>"))
        self.cb_classe = QComboBox()
        for c in sorted(LEGEND):
            self.cb_classe.addItem("%d - %s" % (c, LEGEND[c][0]), c)
            self.cb_classe.setItemData(self.cb_classe.count() - 1,
                                       _cor_do_codigo(c), Qt.DecorationRole)
        self.cb_classe.setCurrentIndex(sorted(LEGEND).index(15))
        self.cb_classe.setFocusPolicy(Qt.NoFocus)
        painel.addWidget(self.cb_classe)
        self.btn_balde = QPushButton("Balde de preenchimento (B)")
        self.btn_desenho = QPushButton("Desenhar polígono (D)")
        self.btn_colar = QPushButton("Colar (Ctrl+V)")
        for b, m in ((self.btn_balde, "balde"), (self.btn_desenho, "desenho"),
                     (self.btn_colar, "colar")):
            b.setCheckable(True)
            b.toggled.connect(lambda on, m=m: self._botao_modo(m, on))
        painel.addWidget(self.btn_balde)
        painel.addWidget(self.btn_desenho)
        linha = QHBoxLayout()
        linha.addWidget(QLabel("Forma"))
        self.cb_forma = QComboBox()
        for chave, nome in FORMAS:
            self.cb_forma.addItem(nome, chave)
        self.cb_forma.setFocusPolicy(Qt.NoFocus)
        self.cb_forma.currentIndexChanged.connect(
            lambda *_: self.set_modo("desenho"))
        linha.addWidget(self.cb_forma, 1)
        self.sp_lados = QSpinBox()
        self.sp_lados.setRange(3, 64)
        self.sp_lados.setValue(6)
        self.sp_lados.setSuffix(" lados")
        self.sp_lados.setToolTip("Nº de lados do polígono regular")
        linha.addWidget(self.sp_lados)
        painel.addLayout(linha)
        linha = QHBoxLayout()
        b = QPushButton("Copiar (Ctrl+C)")
        b.clicked.connect(self.copiar)
        linha.addWidget(b)
        linha.addWidget(self.btn_colar)
        painel.addLayout(linha)

        # --- correção automática
        painel.addWidget(QLabel("<b>Correção automática</b>"))
        grade = QGridLayout()
        grade.addWidget(QLabel("Ilhas <"), 0, 0)
        self.sp_ilha = QDoubleSpinBox()
        self.sp_ilha.setRange(0, 1000)
        self.sp_ilha.setSingleStep(0.25)
        self.sp_ilha.setValue(ILHA_HA)
        self.sp_ilha.setSuffix(" ha")
        self.sp_ilha.valueChanged.connect(self._ilha_mudou)
        grade.addWidget(self.sp_ilha, 0, 1)
        self.lb_ilha_px = QLabel("")
        grade.addWidget(self.lb_ilha_px, 0, 2)
        grade.addWidget(QLabel("Ajuste à imagem ±"), 1, 0)
        self.sp_faixa = QSpinBox()
        self.sp_faixa.setRange(0, 30)
        self.sp_faixa.setValue(FAIXA_PX)
        self.sp_faixa.setSuffix(" px")
        self.sp_faixa.setToolTip("Quanto a borda pode andar para encaixar no "
                                 "gradiente da imagem (0 = desliga)")
        grade.addWidget(self.sp_faixa, 1, 1)
        grade.addWidget(QLabel("Suavizar escadas"), 2, 0)
        self.sp_suav = QSpinBox()
        self.sp_suav.setRange(0, 20)
        self.sp_suav.setValue(SUAVIZA_PX)
        self.sp_suav.setSuffix(" px")
        self.sp_suav.setToolTip("Simplificação das bordas (compartilhada entre "
                                "vizinhos; 0 = bordas em pixel)")
        grade.addWidget(self.sp_suav, 2, 1)
        painel.addLayout(grade)
        self.chk_auto = QCheckBox("Aplicar ao carregar")
        self.chk_auto.setChecked(True)
        painel.addWidget(self.chk_auto)
        self.btn_auto = QPushButton("Corrigir agora (A)")
        self.btn_auto.clicked.connect(self.auto)
        painel.addWidget(self.btn_auto)

        # --- legenda / info
        painel.addWidget(QLabel("<b>Classes</b>"))
        self.legenda = QLabel("—")
        self.legenda.setTextFormat(Qt.RichText)
        self.legenda.setWordWrap(True)
        self.legenda.setStyleSheet(
            "background:#f8f8f8; border:1px solid #ccc; padding:6px;")
        painel.addWidget(self.legenda)
        painel.addWidget(QLabel("<b>Sob o cursor</b>"))
        self.info = QLabel("—")
        self.info.setWordWrap(True)
        painel.addWidget(self.info)

        # --- ações
        botoes = [
            ("Ajustar (F)", self._fit),
            ("Próximo vazio (N)", self.proximo_vazio),
            ("Apagar selecionado (Del)", self.apagar_selecionado),
            ("Desfazer (Ctrl+Z)", self.desfazer),
            ("Refazer (Ctrl+Y)", self.refazer),
            ("Recomeçar recorte do MapBiomas", self.recomecar_recorte),
        ]
        for texto, fn in botoes:
            b = QPushButton(texto)
            b.clicked.connect(fn)
            painel.addWidget(b)
        ajuda = QLabel(AJUDA)
        ajuda.setTextFormat(Qt.RichText)
        ajuda.setWordWrap(True)
        ajuda.setStyleSheet("color:#444; font-size:11px;")
        painel.addWidget(ajuda)
        painel.addStretch(1)

        cont = QWidget()
        cont.setLayout(painel)
        for w in cont.findChildren(QWidget):
            if isinstance(w, (QPushButton, QCheckBox)):
                w.setFocusPolicy(Qt.NoFocus)  # atalhos continuam na view
        rolagem = QScrollArea()   # painel alto: rola em telas baixas
        rolagem.setWidget(cont)
        rolagem.setWidgetResizable(True)
        rolagem.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        rolagem.setFixedWidth(360)
        rolagem.setFocusPolicy(Qt.NoFocus)
        hmain = QHBoxLayout()
        hmain.addWidget(self.view, 1)
        hmain.addWidget(rolagem)
        lay.addLayout(hmain)

        barra = QHBoxLayout()
        self.cb_cena = QComboBox()
        for c in CENAS:
            self.cb_cena.addItem(c["nome"])
        self._cena_idx = 0
        self.cb_cena.currentIndexChanged.connect(self._cena_mudou)
        self.cb_fonte = QComboBox()
        self.cb_fonte.addItem("MapBiomas 30 m (2007)")
        self.cb_fonte.addItem("MapBiomas 10 m (2024)")
        self.cb_fonte.setToolTip("Fonte da máscara MapBiomas original usada na edição")
        self.cb_fonte.currentIndexChanged.connect(self._fonte_mudou)
        self.btn_carregar = QPushButton("Carregar / Recortes… (PgUp/PgDn)")
        self.btn_carregar.clicked.connect(self.carregar)
        self.btn_abrir = QPushButton("Abrir máscara…")
        self.btn_abrir.clicked.connect(self.abrir_mascara)
        self.op = QSlider(Qt.Horizontal)
        self.op.setRange(0, 100)
        self.op.setValue(55)
        self.op.setMaximumWidth(160)
        self.op.valueChanged.connect(self._op_mudou)
        self.chk_pan = QCheckBox("Fundo PAN")
        self.chk_pan.setChecked(True)
        self.chk_pan.toggled.connect(self._pan_toggle)
        self.btn_salvar = QPushButton("Salvar (Ctrl+S)")
        self.btn_salvar.clicked.connect(self.salvar)
        self.status = QLabel("Nada carregado")
        for w in (self.cb_cena, self.cb_fonte, self.btn_carregar, self.btn_abrir,
                  QLabel("Opacidade"), self.op,
                  self.chk_pan, self.btn_salvar):
            w.setFocusPolicy(Qt.NoFocus)
            barra.addWidget(w)
        barra.addStretch(1)
        lay.addLayout(barra)
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self._ilha_mudou()
        self._cena_mudou()

    # ------------------------------------------------------------ estado
    def _ilha_px(self):
        return int(round(self.sp_ilha.value() * 10000 / self._m2px))

    def _ilha_mudou(self, *_):
        self.lb_ilha_px.setText("≈%d px" % self._ilha_px())

    def classe_ativa(self):
        return self.cb_classe.currentData()

    def set_classe_ativa(self, cod):
        if cod in LEGEND:
            self.cb_classe.setCurrentIndex(sorted(LEGEND).index(cod))

    @property
    def modo(self):
        """None (seleção/edição), 'balde', 'desenho' ou 'colar'."""
        return self._modo

    def balde_ativo(self):
        return self._modo == "balde"

    def forma(self):
        return self.cb_forma.currentData()

    def n_lados(self):
        return self.sp_lados.value()

    def set_modo(self, modo):
        """Modos são exclusivos; os botões só refletem o estado."""
        self._modo = modo
        for b, m in ((self.btn_balde, "balde"), (self.btn_desenho, "desenho"),
                     (self.btn_colar, "colar")):
            b.blockSignals(True)
            b.setChecked(modo == m)
            b.blockSignals(False)
        if isinstance(self.scene, Cena):
            self.scene.cancelar_desenho()
            if modo is not None:
                self.scene.clearSelection()
        self.view.viewport().setCursor(Qt.CrossCursor if modo
                                       else Qt.ArrowCursor)
        c = self.classe_ativa()
        if modo == "balde":
            self.status.setText("Balde: clique numa região para pintar com "
                                "%d (%s)" % (c, _nome_codigo(c)))
        elif modo == "desenho":
            if self.forma() == "livre":
                dica = ("clique os vértices · duplo clique/Enter fecha · "
                        "direito/Backspace desfaz ponto")
            else:
                dica = ("clique no %s e clique de novo (ou arraste) para o "
                        "tamanho" % ("canto" if self.forma() == "retangulo"
                                     else "centro"))
            self.status.setText("Desenho %s (%d %s): %s · Esc cancela"
                                % (dict(FORMAS)[self.forma()], c,
                                   _nome_codigo(c), dica))
        elif modo == "colar":
            self.status.setText("Colar: clique para posicionar (%d polígonos) "
                                "· pode colar várias vezes · Esc/direito sai"
                                % len(self.clipboard))
        else:
            self.status.setText("Modo seleção")

    def _botao_modo(self, modo, on):
        if on and modo == "colar" and not self.clipboard:
            self.set_modo(self._modo)   # desmarca o botão
            self.status.setText("Nada copiado (selecione e Ctrl+C)")
        elif on:
            self.set_modo(modo)
        elif self._modo == modo:
            self.set_modo(None)

    # ---- copiar / colar ----
    def copiar(self):
        if not isinstance(self.scene, Cena):
            return
        sel = [i for i in self.scene.selecionados() if not i.vazio]
        if not sel:
            self.status.setText("Selecione polígonos para copiar")
            return
        geoms = [i.geom_atual() for i in sel]
        c = shapely.union_all(geoms).centroid
        self.clipboard = [(i.cod, shapely.affinity.translate(g, -c.x, -c.y))
                          for i, g in zip(sel, geoms)]
        self.status.setText("Copiados %d polígonos (Ctrl+V para colar)"
                            % len(sel))

    def colar(self):
        if not self.clipboard:
            self.status.setText("Nada copiado (selecione e Ctrl+C)")
            return
        self.set_modo("colar")

    def geoms_colados(self, pos):
        return [shapely.affinity.translate(g, pos.x(), pos.y())
                for _, g in self.clipboard]

    def colar_em(self, pos):
        R = self.R.copy()
        n = 0
        for (cod, _), g in sorted(zip(self.clipboard, self.geoms_colados(pos)),
                                  key=lambda t: -t[1].area):
            m = mascara_geom(g, R.shape)
            R[m] = cod
            n += int(m.sum())
        self.mudar_R(R, "Colados %d polígonos (%.2f ha) — clique de novo para "
                     "mais uma cópia, Esc sai"
                     % (len(self.clipboard), n * self._m2px / 10000))
        self.scene._atualizar_rascunho()

    def _ocupado(self, msg):
        self.status.setText(msg)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

    def _livre(self):
        QApplication.restoreOverrideCursor()

    def _alpha(self):
        return self.op.value() / 100.0

    def mudar_R(self, novo, msg=None, selecionar=None):
        """Troca a máscara de trabalho (com undo) e redesenha. Fora da
        imagem (nodata) é sempre restaurado: nenhuma ferramenta pinta lá."""
        novo[~self.valido] = FORA
        novo[self.valido & (novo == FORA)] = 0
        if self.R is not None and np.array_equal(novo, self.R):
            return False
        if self.R is not None:
            self._undo.append(self.R)
            del self._undo[:-MAX_UNDO]
            self._redo.clear()
        self.R = novo
        self.reconstruir(selecionar)
        if msg:
            self.status.setText(msg)
        return True

    def reconstruir(self, selecionar=None):
        """Revetoriza R e repopula a cena (mantém zoom/posição)."""
        cods, geoms = vetorizar(self.R, TOL_TELA)
        self._cods = cods
        self.scene.popular(cods, geoms, self._alpha())
        self._i_vazio = -1
        if selecionar:
            for p in selecionar:
                it = self.scene.regiao_em(QPointF(*p))
                if it is not None and not it.vazio:
                    it.setSelected(True)
        self.atualizar_legenda()

    # ------------------------------------------------------------ carregar
    def _fit(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _cena(self):
        return CENAS[self.cb_cena.currentIndex()]

    def _info_cena(self):
        """Grade da cena atual (cache): w, h, gt, wkt, nodata, m²/px."""
        c = self._cena()
        if c.get("_info") is None:
            ds = gdal.Open(c["imagem"])
            b = ds.GetRasterBand(1)
            nd = b.GetNoDataValue()
            # nd None = imagem sem nodata declarado: tudo é válido (pixel 0
            # pode ser sombra/água escura, não "fora da imagem")
            c["_info"] = {"w": ds.RasterXSize, "h": ds.RasterYSize,
                          "gt": ds.GetGeoTransform(),
                          "wkt": ds.GetProjection(), "nd": nd}
            ds = None
            i = c["_info"]
            i["m2px"] = m2_por_pixel(i["gt"], i["wkt"], i["h"])
        return c["_info"]

    def _refinada(self):
        c = self._cena()
        return c.get("_refinada_usuario") or c["refinada"]

    def _em_tiles(self):
        return bool(self._cena().get("tiles"))

    def _tiles_meta(self):
        c = self._cena()
        if c.get("_meta") is None:
            with open(os.path.join(c["tiles"], "tiles.json"),
                      encoding="utf-8") as fp:
                c["_meta"] = json.load(fp)
        return c["_meta"]

    def _arq_tile(self, tid, nome):
        return os.path.join(self._cena()["tiles"], tid, nome)

    def _vrt_mascara_atual(self):
        """Mosaico virtual (em memória) da máscara atual de cada tile, na ordem:
        refinada (edição manual) > base (MapBiomas + água da PAN, gerada por
        scripts/spot5/agua_pan.py) > MapBiomas original (30m ou 10m conforme
        o seletor cb_fonte)."""
        m = self._tiles_meta()
        ori = "mascara_mapbiomas_10m.tif" if self._fonte_10m() \
            else "mascara_mapbiomas.tif"
        arqs = []
        for t in m["tiles"]:
            cands = [self._arq_tile(t["id"], n) for n in
                     ("mascara_refinada.tif", "mascara_base.tif", ori)]
            arqs.append(next(p for p in cands if os.path.exists(p)))
        gt = m["gt"]
        gdal.BuildVRT("/vsimem/mascara_atual.vrt", arqs,
                      outputBounds=(gt[0], gt[3] + m["h"] * gt[5],
                                    gt[0] + m["w"] * gt[1], gt[3])
                      ).FlushCache()
        return "/vsimem/mascara_atual.vrt"

    def _editados(self):
        """Recortes já salvos (não recebem correção automática ao abrir).
        Refinada antiga sem registro = tudo editado."""
        if self._em_tiles():
            return {t["id"] for t in self._tiles_meta()["tiles"]
                    if os.path.exists(self._arq_tile(t["id"],
                                                     "mascara_refinada.tif"))}
        ref = self._refinada()
        js = ref + ".recortes.json"
        if os.path.exists(js):
            with open(js, encoding="utf-8") as fp:
                return set(json.load(fp)["recortes"])
        return {"*"} if os.path.exists(ref) else set()

    def _marcar_editado(self, chave):
        ed = self._editados() - {"*"}
        ed.add(chave)
        with open(self._refinada() + ".recortes.json", "w",
                  encoding="utf-8") as fp:
            json.dump({"recortes": sorted(ed)}, fp, indent=1)

    def _fonte_mudou(self, *_):
        """Trocou 30m/10m: recarrega o recorte atual da nova máscara original."""
        if not self.recorte:
            return
        if self.R is not None and self._modificado() and not self._perguntar_salvar():
            self.cb_fonte.blockSignals(True)
            self.cb_fonte.setCurrentIndex(0 if self._fonte_10m() else 1)
            self.cb_fonte.blockSignals(False)
            return
        self._carregar_recorte(*self.recorte, fonte_original=True)

    def _cena_mudou(self, *_):
        """Parâmetros da correção pela resolução: faixa/suavização = meio
        pixel do MapBiomas (15 m)."""
        if self.R is not None and self._modificado() and not self._perguntar_salvar():
            self.cb_cena.blockSignals(True)
            self.cb_cena.setCurrentIndex(self._cena_idx)
            self.cb_cena.blockSignals(False)
            return
        self._cena_idx = self.cb_cena.currentIndex()
        if not os.path.exists(self._cena()["imagem"]):
            self.status.setText("Imagem não encontrada: %s"
                                % self._cena()["imagem"])
            return
        i = self._info_cena()
        px_m = math.sqrt(i["m2px"])
        n = max(1, int(round(15.0 / px_m)))
        self.sp_faixa.setValue(n)
        self.sp_suav.setValue(n)
        self._m2px = i["m2px"]
        self._ilha_mudou()
        self.recorte = None
        self.status.setText("%s: %dx%d px de %.1f m — Carregar"
                            % (self._cena()["nome"], i["w"], i["h"], px_m))

    def _perguntar_salvar(self):
        """Edições pendentes: salvar/descartar/cancelar. True = pode seguir."""
        r = QMessageBox.question(
            self, "Edições não salvas", "Salvar o recorte atual antes?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        return r == QMessageBox.Discard or self.salvar()

    def _grade(self):
        """Recortes da cena: [(chave, x0, y0, w, h)] (cena pequena = 1)."""
        if self._em_tiles():
            return [(t["id"], t["x0"], t["y0"], t["w"], t["h"])
                    for t in self._tiles_meta()["tiles"]]
        i = self._info_cena()
        if i["w"] * i["h"] <= RECORTE_MAX_PX:
            return [("0_0", 0, 0, i["w"], i["h"])]
        t = RECORTE_PX
        return [("%d_%d" % (x, y), x, y, min(t, i["w"] - x), min(t, i["h"] - y))
                for y in range(0, i["h"], t) for x in range(0, i["w"], t)]

    def carregar(self):
        """Cena pequena: abre inteira. Cena grande: escolher o recorte."""
        if not os.path.exists(self._cena()["imagem"]):
            QMessageBox.warning(self, "Erro", "Imagem não encontrada")
            return
        if self.R is not None and self._modificado() and not self._perguntar_salvar():
            return
        grade = self._grade()
        if len(grade) == 1:
            self._carregar_recorte(*grade[0])
            return
        dlg = DialogoRecortes(self, grade, self._editados(),
                              self.recorte[0] if self.recorte else None)
        if dlg.exec() and dlg.escolhido is not None:
            self._carregar_recorte(*dlg.escolhido)

    def recorte_vizinho(self, passo):
        """PgDn/PgUp: próximo/anterior recorte com imagem."""
        grade = [g for g in self._grade() if self._ov_valido(g)]
        if len(grade) < 2 or self.recorte is None:
            return
        if self._modificado() and not self._perguntar_salvar():
            return
        chaves = [g[0] for g in grade]
        i = chaves.index(self.recorte[0]) if self.recorte[0] in chaves else -1
        self._carregar_recorte(*grade[(i + passo) % len(grade)])

    def _ov_valido(self, g):
        if self._em_tiles():
            return True                 # tiles só-nodata nem foram gerados
        ov = self._overview()
        s = ov["escala"]
        _, x0, y0, w, h = g
        return ov["valido"][y0 // s:(y0 + h - 1) // s + 1,
                            x0 // s:(x0 + w - 1) // s + 1].any()

    def _overview(self, lado=1400):
        """Imagem reduzida da cena (para o mapa de recortes), com cache."""
        c = self._cena()
        cache = os.path.join(c["tiles"], "overview.png") if c.get("tiles") \
            else None
        if c.get("_ov") is None and cache and os.path.exists(cache):
            i = self._info_cena()
            img = QImage(cache)
            c["_ov"] = {"img": img, "valido": None,
                        "escala": max(1, int(math.ceil(max(i["w"], i["h"])
                                                       / max(img.width(),
                                                             img.height()))))}
        if c.get("_ov") is None:
            i = self._info_cena()
            s = max(1, int(math.ceil(max(i["w"], i["h"]) / lado)))
            ds = gdal.Open(c["imagem"])
            a = ds.GetRasterBand(1).ReadAsArray(
                buf_xsize=-(-i["w"] // s), buf_ysize=-(-i["h"] // s),
                resample_alg=gdal.GRIORA_NearestNeighbour).astype(np.float32)
            ds = None
            valido = _valido(a, i["nd"])
            v = a[valido]
            lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size \
                else (0, 1)
            g8 = (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)
            h, w = g8.shape
            c["_ov"] = {"img": QImage(g8.tobytes(), w, h, w,
                                      QImage.Format_Grayscale8).copy(),
                        "escala": s, "valido": valido}
            if cache:
                c["_ov"]["img"].save(cache)   # ler a cena toda é lento
        return c["_ov"]

    def _fonte_10m(self):
        """Seletor 30 m (2007) / 10 m (2024): True = usar a máscara MapBiomas
        10 m (coleção 3, mais recente) como original da edição."""
        return bool(getattr(self, "cb_fonte", None)
                    and self.cb_fonte.currentIndex() == 1)

    def _mascara_original(self):
        """(caminho, banda) da máscara MapBiomas original conforme o seletor."""
        c = self._cena()
        if self._em_tiles():
            nome = "mascara_mapbiomas_10m.vrt" if self._fonte_10m() \
                else "mascara_mapbiomas.vrt"
            return os.path.join(c["tiles"], nome), 1
        return c["mascara"], c["mascara_banda"]

    def _ler_janela(self, x0, y0, w, h, fonte_original=False):
        """(pan float32, R uint8, válido) de uma janela da cena. R vem da
        refinada (se existir) ou da máscara MapBiomas original."""
        c = self._cena()
        i = self._info_cena()
        ds = gdal.Open(c["imagem"])
        pan = ds.GetRasterBand(1).ReadAsArray(x0, y0, w, h).astype(np.float32)
        ds = None
        valido = _valido(pan, i["nd"])
        ref = self._refinada()
        if fonte_original:
            caminho, banda = self._mascara_original()
        elif self._em_tiles():
            caminho, banda = self._vrt_mascara_atual(), 1
        elif os.path.exists(ref):
            caminho, banda = ref, 1
        else:
            caminho, banda = self._mascara_original()
        ds = gdal.Open(caminho)
        if (ds.RasterXSize, ds.RasterYSize) != (i["w"], i["h"]):
            raise ValueError("%s (%dx%d) não está na grade da imagem (%dx%d)"
                             % (caminho, ds.RasterXSize, ds.RasterYSize,
                                i["w"], i["h"]))
        R = ds.GetRasterBand(min(banda, ds.RasterCount)).ReadAsArray(
            x0, y0, w, h).astype(np.uint8)
        ds = None
        R[~np.isin(R, list(LEGEND))] = 0
        R[~valido] = FORA
        return pan, R, valido

    def _carregar_recorte(self, chave, x0, y0, w, h, fonte_original=False):
        c = self._cena()
        i = self._info_cena()
        self._ocupado("Carregando recorte…")
        try:
            t = time.time()
            pan, R, valido = self._ler_janela(x0, y0, w, h, fonte_original)
            if not valido.any():
                self.status.setText("Recorte sem imagem (só nodata)")
                return
            self.recorte = (chave, x0, y0, w, h)
            self.valido = valido
            self.gt, self.proj = i["gt"], i["wkt"]
            self._m2px = i["m2px"]
            self._pan_img = _qimage_pan(pan, valido)
            auto = self.chk_auto.isChecked() and (
                fonte_original or not ({"*", chave} & self._editados()))
            if auto:
                R = self._auto_com_margem(R)
            self._undo, self._redo = [], []
            self.R = None
            self.scene = Cena(self, self._pan_img, w, h)
            self.view.setScene(self.scene)
            self._pan_toggle(self.chk_pan.isChecked())
            self.mudar_R(R)
            self._salvo = None if (auto or fonte_original) else self.R
            self._fit()
            self.view.setFocus()
            nrec = len(self._grade())
            onde = "" if nrec == 1 else " — recorte %s (%d de %d)" % (
                chave, [g[0] for g in self._grade()].index(chave) + 1, nrec)
            self.setWindowTitle("Corretor de bordas LULC — %s%s"
                                % (c["nome"], onde))
            self.status.setText(
                "%s%s: %d polígonos, %d vazios (%.1fs%s)"
                % (c["nome"], onde,
                   len(self.scene.itens) - len(self.scene.vazios),
                   len(self.scene.vazios), time.time() - t,
                   ", corrigido" if auto else ""))
        except Exception as e:           # arquivo fora da grade etc.
            QMessageBox.warning(self, "Erro ao carregar", str(e))
        finally:
            self._livre()

    def _auto_com_margem(self, R):
        """Correção automática do recorte olhando MARGEM px da vizinhança
        (evita ilha falsa/borda torta no corte); só o recorte volta."""
        _, x0, y0, w, h = self.recorte
        i = self._info_cena()
        m = MARGEM_AUTO
        ax0, ay0 = max(0, x0 - m), max(0, y0 - m)
        ax1, ay1 = min(i["w"], x0 + w + m), min(i["h"], y0 + h + m)
        pan, Rg, valido = self._ler_janela(ax0, ay0, ax1 - ax0, ay1 - ay0)
        dy, dx = y0 - ay0, x0 - ax0
        Rg[dy:dy + h, dx:dx + w] = R
        grad = gradiente_pan(np.where(valido, pan, np.median(pan[valido])
                                      if valido.any() else 0))
        out = auto_corrigir(Rg, grad, self._ilha_px(), self.sp_faixa.value(),
                            self.sp_suav.value())
        # sieve/suavização podem empurrar a borda do "fora" alguns px para
        # dentro da imagem: esses px (e vazios que não existiam) pegam a
        # classe válida mais próxima
        ruim = valido & ((out == FORA) | ((out == 0) & (Rg != 0)))
        fonte = valido & (out != 0) & (out != FORA)
        if ruim.any() and fonte.any():
            iy, ix = ndimage.distance_transform_edt(
                ~fonte, return_distances=False, return_indices=True)
            out[ruim] = out[iy[ruim], ix[ruim]]
        out[~valido] = FORA
        return out[dy:dy + h, dx:dx + w].copy()

    def abrir_mascara(self):
        """Usar outra máscara (na grade da cena) como a refinada: é dela que
        os recortes são lidos e para ela que são salvos."""
        if self._em_tiles():
            self.status.setText("Cena em tiles: cada tile salva a sua própria "
                                "mascara_refinada.tif na pasta do tile")
            return
        caminho, _ = QFileDialog.getOpenFileName(
            self, "Abrir máscara (grade da cena)",
            os.path.dirname(self._refinada()), "GeoTIFF (*.tif *.tiff)")
        if not caminho:
            return
        self._cena()["_refinada_usuario"] = caminho
        if self.recorte:
            self._carregar_recorte(*self.recorte)
        else:
            self.carregar()

    def recomecar_recorte(self):
        """Recarrega o recorte da máscara MapBiomas ORIGINAL (+ correção
        automática, se marcada). Só vai para o disco ao salvar."""
        if not self.recorte:
            return
        r = QMessageBox.question(
            self, "Recomeçar recorte",
            "Descartar as edições deste recorte e voltar ao MapBiomas "
            "original?\n(o arquivo só muda quando você salvar)")
        if r == QMessageBox.Yes:
            self._carregar_recorte(*self.recorte, fonte_original=True)

    # ------------------------------------------------------------ edições
    def _folga(self):
        return int(TOL_TELA) + 3

    def aplicar_edicoes(self, itens):
        """Aplica na máscara as formas editadas (vértices/arrasto). Só a
        DIFERENÇA entre a forma antiga e a nova vai para o raster: o resto do
        contorno fica intacto (a vista é simplificada e não bate pixel a
        pixel com a máscara)."""
        R = self.R
        pts = []
        for it in itens:
            nova = it.geom_atual()
            g_velha = mascara_geom(it.geom, R.shape)
            g_nova = mascara_geom(nova, R.shape)
            comp = componente(R, it.geom, it.cod, self._folga())
            ganho = g_nova & ~g_velha
            perda = g_velha & ~g_nova & comp
            R = aplicar_forma(R, perda, ganho, it.cod)
            if it.isSelected() and not nova.is_empty:
                # só reseleciona quem estava selecionado (o vizinho que
                # andou junto pelo vértice compartilhado não)
                p = nova.representative_point()
                pts.append((p.x, p.y))
        if not self.mudar_R(R, "Borda editada (%d polígonos)" % len(itens),
                            selecionar=pts):
            self.reconstruir(pts)

    def criar_geom(self, geom):
        """Forma desenhada vira a classe ativa por cima do que houver
        (os vizinhos perdem a área coberta)."""
        geom = shapely.make_valid(geom)
        novo = mascara_geom(geom, self.R.shape)
        if not novo.any():
            self.status.setText("Polígono pequeno demais (não cobre pixel)")
            return
        cod = self.classe_ativa()
        R = self.R.copy()
        R[novo] = cod
        self.mudar_R(R, "Criado polígono %d (%s), %.2f ha — continue "
                     "desenhando ou D para sair"
                     % (cod, _nome_codigo(cod),
                        novo.sum() * self._m2px / 10000))

    def pintar(self, itens, cod):
        """Balde / classificar: a região inteira de cada item vira `cod`."""
        R = self.R.copy()
        pts = []
        for it in itens:
            R[componente(self.R, it.geom, it.cod, self._folga())] = cod
            p = it.geom.representative_point()
            pts.append((p.x, p.y))
        sel = pts if not self.balde_ativo() else None
        self.mudar_R(R, "Pintado %d (%s) em %d regiões"
                     % (cod, _nome_codigo(cod), len(itens)), selecionar=sel)

    def classificar(self, cod):
        self.set_classe_ativa(cod)
        if not isinstance(self.scene, Cena):
            return
        sel = self.scene.selecionados()
        if sel:
            self.pintar(sel, cod)
        else:
            self.status.setText("Classe ativa: %d (%s) — selecione algo ou use "
                                "o balde (B)" % (cod, _nome_codigo(cod)))

    def escolher_classe(self):
        items = ["%d - %s" % (c, LEGEND[c][0]) for c in sorted(LEGEND)]
        escolha, ok = QInputDialog.getItem(
            self, "Classe", "Código MapBiomas:", items,
            self.cb_classe.currentIndex(), False)
        if ok:
            self.classificar(int(escolha.split(" - ")[0]))

    def apagar_selecionado(self):
        if not isinstance(self.scene, Cena):
            return
        sel = [i for i in self.scene.selecionados() if not i.vazio]
        if sel:
            self.pintar(sel, 0)
            self.status.setText("Apagados %d polígonos (viraram vazio)" % len(sel))

    def auto(self):
        if self.R is None:
            return
        self._ocupado("Corrigindo…")
        try:
            t = time.time()
            R = self._auto_com_margem(self.R)
            mud = (R != self.R)[self.valido].mean() * 100
            self.mudar_R(R, "Correção automática: %.2f%% dos pixels mudaram "
                         "(%.1fs)" % (mud, time.time() - t))
        finally:
            self._livre()

    # ------------------------------------------------------------ vista
    def _op_mudou(self, v):
        if isinstance(self.scene, Cena):
            for it in self.scene.itens:
                it.estilo(v / 100.0)

    def _pan_toggle(self, on):
        if isinstance(self.scene, Cena):
            self.scene.pix.setVisible(on)

    def mostrar_info(self, it):
        area = shapely.area(it.geom) * self._m2px / 10000
        if it.vazio:
            txt = ("<b style='color:#c0c'>Vazio (sem classe)</b><br>"
                   "selecione e tecle 1–9, ou use o balde")
        else:
            cl = CLASS_MAP.get(it.cod)
            txt = "<b>%d</b> %s<br>%s" % (it.cod, _nome_codigo(it.cod),
                                          CLASS_NOMES[cl] if cl is not None else "?")
        self.info.setText("%s<br>%.2f ha · %d vértices"
                          % (txt, area, len(it._ext)))

    def atualizar_legenda(self):
        if self.R is None:
            self.legenda.setText("—")
            return
        rapidas = {c: k for k, c in TECLAS_RAPIDAS.items()}
        n_por = {}
        for it in self.scene.itens:
            n_por[it.cod] = n_por.get(it.cod, 0) + 1
        codigos = sorted((set(n_por) - {0, FORA}) |
                         set(TECLAS_RAPIDAS.values()))
        html = ["<table cellspacing='2'>"]
        for c in codigos:
            nome, hexc = LEGEND.get(c, ("Cod. %d" % c, "#888888"))
            tecla = ("[%d] " % rapidas[c]) if c in rapidas else ""
            html.append(
                "<tr><td style='background:%s;width:18px'>&nbsp;</td>"
                "<td><b>%s%d</b> %s <span style='color:#666'>(%d)</span></td>"
                "</tr>" % (hexc, tecla, c, nome, n_por.get(c, 0)))
        html.append(
            "<tr><td style='background:#f0f;width:18px'>&nbsp;</td>"
            "<td><b>Vazios</b> <span style='color:#666'>(%d, N navega)</span>"
            "</td></tr></table>" % n_por.get(0, 0))
        self.legenda.setText("".join(html))

    def proximo_vazio(self):
        if not isinstance(self.scene, Cena):
            return
        vaz = self.scene.vazios
        if not vaz:
            self.status.setText("Nenhum vazio restante")
            return
        self._i_vazio = (self._i_vazio + 1) % len(vaz)
        it = vaz[self._i_vazio]
        self.scene.clearSelection()
        it.setSelected(True)
        r = it.sceneBoundingRect()
        m = max(r.width(), r.height(), 20)
        self.view.fitInView(r.adjusted(-m, -m, m, m), Qt.KeepAspectRatio)
        self.mostrar_info(it)
        self.status.setText("Vazio %d/%d" % (self._i_vazio + 1, len(vaz)))

    def alternar_overlay(self):
        if self.op.value() > 0:
            self._op_ant = self.op.value()
            self.op.setValue(0)
        else:
            self.op.setValue(self._op_ant or 55)

    def reverter(self):
        """Volta o recorte ao que está salvo em disco."""
        if self.R is None or not self.recorte:
            return
        r = QMessageBox.question(self, "Reverter",
                                 "Descartar as edições não salvas deste "
                                 "recorte?")
        if r == QMessageBox.Yes:
            self._salvo = self.R          # evita a pergunta de salvar
            self._carregar_recorte(*self.recorte)

    def _modificado(self):
        return self.R is not None and (self._salvo is None or
                                       not np.array_equal(self.R, self._salvo))

    def closeEvent(self, event):
        if self._modificado():
            r = QMessageBox.question(
                self, "Sair", "Há edições não salvas. Salvar antes de sair?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Cancel or \
                    (r == QMessageBox.Save and not self.salvar()):
                event.ignore()
                return
        event.accept()

    def keyPressEvent(self, event):
        k, mod = event.key(), event.modifiers()
        ctrl = bool(mod & Qt.ControlModifier)
        cena = isinstance(self.scene, Cena)
        if k == Qt.Key_Z and ctrl:
            self.refazer() if mod & Qt.ShiftModifier else self.desfazer()
        elif k == Qt.Key_Y and ctrl:
            self.refazer()
        elif k == Qt.Key_S and ctrl:
            self.salvar()
        elif k == Qt.Key_C and ctrl:
            self.copiar()
        elif k == Qt.Key_V and ctrl:
            self.colar()
        elif self.modo == "desenho" and k in (Qt.Key_Return, Qt.Key_Enter):
            self.scene.terminar_desenho()
        elif self.modo == "desenho" and k == Qt.Key_Backspace:
            self.scene.desfazer_ponto()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.apagar_selecionado()
        elif k == Qt.Key_D:
            self.set_modo(None if self._modo == "desenho" else "desenho")
        elif Qt.Key_1 <= k <= Qt.Key_9 and not ctrl:
            self.classificar(TECLAS_RAPIDAS[k - Qt.Key_0])
        elif k == Qt.Key_C:
            self.escolher_classe()
        elif k == Qt.Key_B:
            self.set_modo(None if self._modo == "balde" else "balde")
        elif k == Qt.Key_A:
            self.auto()
        elif k == Qt.Key_N:
            self.proximo_vazio()
        elif k == Qt.Key_Escape:
            if self.modo == "desenho" and self.scene.pontos:
                self.scene.cancelar_desenho()
            elif self.modo is not None:
                self.set_modo(None)
            elif cena:
                self.scene.clearSelection()
        elif k == Qt.Key_F:
            self._fit()
        elif k == Qt.Key_T:
            self.alternar_overlay()
        elif k == Qt.Key_G:
            self.chk_pan.toggle()
        elif k == Qt.Key_R:
            self.reverter()
        elif k == Qt.Key_PageDown:
            self.recorte_vizinho(+1)
        elif k == Qt.Key_PageUp:
            self.recorte_vizinho(-1)
        else:
            super().keyPressEvent(event)

    def desfazer(self):
        if not self._undo:
            return
        self._redo.append(self.R)
        self.R = self._undo.pop()
        self.reconstruir()
        self.status.setText("Desfeito (%d restantes)" % len(self._undo))

    def refazer(self):
        if not self._redo:
            return
        self._undo.append(self.R)
        self.R = self._redo.pop()
        self.reconstruir()
        self.status.setText("Refeito")

    def salvar(self):
        """Grava o recorte na máscara refinada da cena (cena inteira, 1
        banda). No 1º salvar a refinada nasce como cópia da máscara MapBiomas
        original. Fora da imagem vira 0. Devolve True se gravou."""
        if self.R is None or not self.recorte:
            return False
        arr = preencher_vazios_pequenos(self.R, VAZIO_FILL_PX)
        arr[~self.valido] = FORA
        sem = int(((arr == 0) & self.valido).sum())
        if sem:
            r = QMessageBox.question(
                self, "Vazios",
                "%d px continuam sem classe (serão gravados como 0 = nodata)."
                "\nSalvar mesmo assim?" % sem)
            if r != QMessageBox.Yes:
                return False
        arr[arr == FORA] = 0
        c = self._cena()
        if self._em_tiles():
            return self._salvar_tile(arr)
        ref = self._refinada()
        self._ocupado("Salvando…")
        try:
            if not os.path.exists(ref):
                gdal.Translate(ref, c["mascara"], bandList=[c["mascara_banda"]],
                               creationOptions=["COMPRESS=DEFLATE", "TILED=YES",
                                                "BIGTIFF=IF_SAFER"])
            ds = gdal.Open(ref, gdal.GA_Update)
            i = self._info_cena()
            if (ds.RasterXSize, ds.RasterYSize) != (i["w"], i["h"]):
                raise ValueError("%s não está na grade da cena" % ref)
            b = ds.GetRasterBand(1)
            b.SetNoDataValue(0)
            chave, x0, y0, _, _ = self.recorte
            b.WriteArray(arr, x0, y0)
            b.FlushCache()
            ds = None
            self._marcar_editado(chave)
        except Exception as e:
            QMessageBox.warning(self, "Erro ao salvar", str(e))
            return False
        finally:
            self._livre()
        self._salvo = self.R
        self.status.setText("Recorte %s salvo em %s" % (self.recorte[0], ref))
        return True


    def _salvar_tile(self, arr):
        """Cena em tiles: grava <tile>/mascara_refinada.tif (grade do tile,
        copiada da mascara_mapbiomas.tif do próprio tile)."""
        tid = self.recorte[0]
        base = self._arq_tile(tid, "mascara_mapbiomas.tif")
        dest = self._arq_tile(tid, "mascara_refinada.tif")
        try:
            src = gdal.Open(base)
            if (src.RasterXSize, src.RasterYSize) != arr.shape[::-1]:
                raise ValueError("tile %s: recorte %s ≠ arquivo %s"
                                 % (tid, arr.shape[::-1],
                                    (src.RasterXSize, src.RasterYSize)))
            ds = gdal.GetDriverByName("GTiff").Create(
                dest, arr.shape[1], arr.shape[0], 1, gdal.GDT_Byte,
                ["COMPRESS=DEFLATE", "TILED=YES"])
            ds.SetGeoTransform(src.GetGeoTransform())
            ds.SetProjection(src.GetProjection())
            src = None
            b = ds.GetRasterBand(1)
            b.SetNoDataValue(0)
            b.WriteArray(arr)
            ds = None
        except Exception as e:
            QMessageBox.warning(self, "Erro ao salvar", str(e))
            return False
        self._salvo = self.R
        self.status.setText("Tile %s salvo em %s" % (tid, dest))
        return True


def main():
    app = QApplication(sys.argv)
    j = Janela()
    j.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
