"""App de predicción de partidos de LaLiga.

Ejecutar desde la raíz del proyecto:   streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import futbol as fb
import modelo as m

st.set_page_config(page_title="Predicción de LaLiga", page_icon="⚽", layout="wide")
fb.estilo()


@st.cache_resource
def cargar_modelo():
    # Parámetros ya ajustados (ver scripts/exportar_modelo.py): la app no necesita los datos originales
    ruta = Path(__file__).resolve().parent / "modelo_actual.json"
    return m.ModeloActual.desde_json(json.loads(ruta.read_text(encoding="utf-8")))


modelo = cargar_modelo()
actuales = sorted(modelo.actuales, key=fb.nombre_completo)

st.title("⚽ ¿Quién gana? Predicción de partidos de LaLiga")
st.caption(f"Modelo estadístico de goles (Poisson) con ataque, defensa y ventaja local, ajustado con "
           f"{fb.miles(modelo.n_partidos)} partidos hasta el {modelo.ultima_fecha:%d/%m/%Y}.")

c1, c2 = st.columns(2)
por_defecto = {"local": actuales.index("Real Madrid") if "Real Madrid" in actuales else 0,
               "visitante": actuales.index("Barcelona") if "Barcelona" in actuales else 1}
local = c1.selectbox("Equipo local", actuales, index=por_defecto["local"], format_func=fb.nombre_completo)
visitante = c2.selectbox("Equipo visitante", actuales, index=por_defecto["visitante"], format_func=fb.nombre_completo)

if local == visitante:
    st.warning("Elige dos equipos distintos.")
    st.stop()

lam, mu, p, matriz = modelo.predecir(local, visitante)
n_local, n_visit = fb.nombre_completo(local), fb.nombre_completo(visitante)

# --- Probabilidades ---------------------------------------------------------
st.subheader("Probabilidad de cada resultado")
m1, m2, m3 = st.columns(3)
m1.metric(f"Gana {n_local}", f"{p[0] * 100:.1f} %", help=f"Cuota justa equivalente: {1 / p[0]:.2f}")
m2.metric("Empate", f"{p[1] * 100:.1f} %", help=f"Cuota justa equivalente: {1 / p[1]:.2f}")
m3.metric(f"Gana {n_visit}", f"{p[2] * 100:.1f} %", help=f"Cuota justa equivalente: {1 / p[2]:.2f}")

fig, ax = plt.subplots(figsize=(10, 1.1))
izq = 0
for valor, color, etiqueta in [(p[0], fb.COLOR_LOCAL, n_local), (p[1], fb.COLOR_EMPATE, "Empate"),
                               (p[2], fb.COLOR_VISITANTE, n_visit)]:
    ax.barh(0, valor * 100, left=izq, color=color, height=0.6, edgecolor=fb.SUPERFICIE, linewidth=2)
    if valor > 0.09:
        ax.text(izq + valor * 50, 0, f"{valor * 100:.0f} %", ha="center", va="center", color="white", fontweight="bold")
    izq += valor * 100
ax.set_xlim(0, 100)
ax.axis("off")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# --- Goles y marcadores -----------------------------------------------------
izq_col, der_col = st.columns([1, 1])
with izq_col:
    st.subheader("Marcadores más probables")
    st.write(f"Goles esperados: **{n_local} {lam:.2f}** · **{n_visit} {mu:.2f}**")
    flat = [(x, y, matriz[x, y]) for x in range(matriz.shape[0]) for y in range(matriz.shape[1])]
    top = sorted(flat, key=lambda t: -t[2])[:6]
    st.table(pd.DataFrame({"Marcador": [f"{x} - {y}" for x, y, _ in top],
                           "Probabilidad": [f"{pr * 100:.1f} %" for _, _, pr in top]}).set_index("Marcador"))

with der_col:
    st.subheader("Mapa de marcadores")
    k = 6
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(matriz[:k, :k] * 100, cmap="Blues")
    for x in range(k):
        for y in range(k):
            v = matriz[x, y] * 100
            ax.text(y, x, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > matriz[:k, :k].max() * 100 * 0.55 else fb.TINTA_SECUNDARIA)
    ax.set_xlabel(f"Goles de {n_visit}")
    ax.set_ylabel(f"Goles de {n_local}")
    ax.grid(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# --- Ataque y defensa -------------------------------------------------------
st.subheader("Ataque y defensa de cada equipo")
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.axhline(0, color=fb.TINTA_SUAVE, lw=0.8)
ax.axvline(0, color=fb.TINTA_SUAVE, lw=0.8)
otros = [e for e in actuales if e not in (local, visitante)]
ax.scatter(modelo.ataque[otros], modelo.defensa[otros], s=45, color="#c3c2b7", edgecolor=fb.SUPERFICIE, linewidth=1.2, zorder=2)
for e in otros:
    ax.annotate(fb.nombre_completo(e), (modelo.ataque[e], modelo.defensa[e]), xytext=(4, 3),
                textcoords="offset points", fontsize=7.5, color=fb.TINTA_SUAVE)
for e, color in [(local, fb.COLOR_LOCAL), (visitante, fb.COLOR_VISITANTE)]:
    ax.scatter([modelo.ataque[e]], [modelo.defensa[e]], s=110, color=color, edgecolor=fb.SUPERFICIE, linewidth=1.5, zorder=3)
    ax.annotate(fb.nombre_completo(e), (modelo.ataque[e], modelo.defensa[e]), xytext=(7, 6),
                textcoords="offset points", fontsize=10, fontweight="bold", color=color)
ax.grid(False)
ax.set_xlabel("Ataque (más a la derecha = marca más)")
ax.set_ylabel("Defensa (más arriba = concede menos)")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# --- Cómo funciona y límites ------------------------------------------------
with st.expander("¿Cómo funciona y cuánto fiarse?"):
    st.markdown(
        f"""
**El modelo.** Los goles de cada equipo se modelan con una distribución de Poisson cuya media depende del **ataque**
del equipo, de la **defensa** del rival y de la **ventaja de jugar en casa** (hoy equivale a multiplicar los goles
por {np.exp(modelo.ventaja):.2f}). Se ajusta con todos los partidos desde 2005, dando a cada partido la mitad de peso
por cada año que pasa.

**Cuánto acierta.** Sobre partidos que no se usaron para diseñarlo (2019/20 en adelante), el modelo cierra el 82 % del
hueco entre "repartir según las frecuencias históricas" y las cuotas de cierre de Pinnacle, una de las casas más
precisas. Queda algo por detrás de las casas de apuestas (pérdida logarítmica 0,987 frente a 0,968) y sus
probabilidades están bien calibradas: cuando dice un 25 %, ocurre en torno al 25 %.

**Lo que no sabe.** Solo conoce los resultados pasados: no tiene en cuenta lesiones, alineaciones, sanciones, el
calendario ni el estado de forma reciente más allá del peso de los últimos partidos. Los equipos recién ascendidos
empiezan con un nivel medio.

*Proyecto educativo. Los datos son de [football-data.co.uk](https://www.football-data.co.uk). No es una recomendación
de apuestas.*
"""
    )
