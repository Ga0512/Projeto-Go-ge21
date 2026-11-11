"""Smoke test headless do editor (guia_edicao.py) — DESATUALIZADO, NÃO RODAR.

Foi escrito para o modo antigo em que a cena grande era editada em recortes de
uma máscara única. Hoje essa cena é editada em tiles: o salvar() abaixo pinta
um polígono e grava num dados/spot5/cena_013-014_2007-06-14/tiles/rRR_cCC/
mascara_refinada.tif REAL, sobrescrevendo edições do usuário. Mantido só como
referência; reescrever antes de usar.
"""
import sys
sys.exit(__doc__)

import os, time, json, hashlib, warnings; warnings.filterwarnings("ignore")
os.environ["QT_QPA_PLATFORM"]="offscreen"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "editor"))
import numpy as np, guia_edicao as g
from osgeo import gdal
from PySide6.QtWidgets import QApplication
app=QApplication([])
REF_P=g.caminhos.SPOT5_RECORTE_REFINADA
h0=hashlib.md5(open(REF_P,"rb").read()).hexdigest()
j=g.Janela(); j.resize(1400,900); j.show()
# ---- cena pequena: continua da refinada do usuario, sem auto
j._carregar_recorte(*j._grade()[0]); print("pequena:", j.status.text(), "| faixa", j.sp_faixa.value(), "| modificado", j._modificado())
# ---- cena grande
j.cb_cena.setCurrentIndex(1); print("grande:", j.status.text(), "| faixa/suav", j.sp_faixa.value(), j.sp_suav.value(), "| ilha px", j._ilha_px())
TMP="/tmp/_refinada_teste.tif"
for f in (TMP, TMP+".recortes.json"):
    if os.path.exists(f): os.remove(f)
j._cena()["_refinada_usuario"]=TMP
t=time.time(); ov=j._overview(); grade=j._grade()
val=[x for x in grade if j._ov_valido(x)]; print("overview %.1fs | recortes %d, com imagem %d"%(time.time()-t, len(grade), len(val)))
dlg=g.DialogoRecortes(j, grade, j._editados(), None); dlg.show(); app.processEvents(); dlg.grab().save("/tmp/_dlg.png"); dlg.close()
interior=grade[len(grade)//2+ int(np.sqrt(len(grade)))//2]; borda=val[0]
for nome,rec in (("interior",interior),("borda",borda)):
    t=time.time(); j._carregar_recorte(*rec); dt=time.time()-t
    R=j.R; print(f"{nome} {rec[0]}: {dt:.1f}s | {j.status.text()}")
    print(f"   fora={int((R==g.FORA).sum())} vazio={int(((R==0)&j.valido).sum())} codigos={sorted(set(np.unique(R).tolist())-{g.FORA})}")
# edita no recorte de borda: pinta o maior poligono de 33 e salva
it=max([i for i in j.scene.itens if not i.vazio], key=lambda i:i.geom.area); j.pintar([it], 29)
t=time.time(); ok=j.salvar(); print("salvar:", ok, "%.1fs"%(time.time()-t), j.status.text())
print("json:", json.load(open(TMP+".recortes.json")))
chave,x0,y0,w,h=j.recorte
ds=gdal.Open(TMP); A=ds.GetRasterBand(1).ReadAsArray(x0,y0,w,h); ds=None
Rs=j.R.copy(); Rs[Rs==g.FORA]=0; print("recorte gravado == R:", bool((A==Rs).all()))
ds=gdal.Open(j._cena()["mascara"]); O=ds.GetRasterBand(1).ReadAsArray(x0+w, y0, 500, 500) if x0+w+500<=ds.RasterXSize else None
ds2=gdal.Open(TMP); O2=ds2.GetRasterBand(1).ReadAsArray(x0+w, y0, 500, 500) if O is not None else None; ds=ds2=None
print("fora do recorte intacto:", None if O is None else bool((O==O2).all()))
# reabre: nao roda auto, igual ao salvo
j._salvo=j.R; j._carregar_recorte(*j.recorte); print("reaberto:", j.status.text(), "| igual:", bool(((j.R==g.FORA)|(j.R==Rs)).all()))
j._salvo=j.R; j.recorte_vizinho(+1); print("PgDn ->", j.recorte[0])
print("refinada do usuario (cena pequena) intacta:", hashlib.md5(open(REF_P,"rb").read()).hexdigest()==h0)
j.grab().save("/tmp/_big.png")
