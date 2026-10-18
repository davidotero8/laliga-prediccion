"""Utilidades del proyecto: carga y limpieza de los datos de LaLiga, conversión de cuotas
a probabilidades y estilo de gráficas.

Los datos vienen de football-data.co.uk (un CSV por temporada, ver `scripts/descargar_datos.py`).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PARTIDOS_TEMPORADA_COMPLETA = 380   # 20 equipos x 19 rivales x 2 (ida y vuelta)

# Columnas del CSV original -> nombre que usamos (solo las que necesita el proyecto)
COLUMNAS = {
    "Date": "fecha",
    "HomeTeam": "local",
    "AwayTeam": "visitante",
    "FTHG": "goles_local",
    "FTAG": "goles_visitante",
    "FTR": "resultado",            # H = gana el local, D = empate, A = gana el visitante
    "HS": "tiros_local", "AS": "tiros_visitante",
    "HST": "tiros_puerta_local", "AST": "tiros_puerta_visitante",
    "HC": "corners_local", "AC": "corners_visitante",
    "HF": "faltas_local", "AF": "faltas_visitante",
    "HY": "amarillas_local", "AY": "amarillas_visitante",
    "HR": "rojas_local", "AR": "rojas_visitante",
    # Cuotas (decimales europeas: 2,50 significa que apostando 1 € ganas 2,50 € si aciertas)
    "B365H": "b365_local", "B365D": "b365_empate", "B365A": "b365_visitante",        # Bet365, antes del partido
    "PSCH": "pinnacle_cierre_local", "PSCD": "pinnacle_cierre_empate",               # Pinnacle, al cierre
    "PSCA": "pinnacle_cierre_visitante",
    "AvgCH": "media_cierre_local", "AvgCD": "media_cierre_empate",                   # media del mercado, al cierre
    "AvgCA": "media_cierre_visitante",
}


def cargar_laliga(desde=None):
    """Devuelve un DataFrame con todos los partidos de LaLiga descargados, ordenados por fecha.

    Añade `temporada` ('2005-06'), `anio_inicio` (2005) y `temporada_completa`
    (True si se han jugado ya los 380 partidos).
    """
    trozos = []
    for f in sorted(RAW_DIR.glob("laliga_*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig")
        d = d.reindex(columns=list(COLUMNAS))            # las columnas que falten quedan en NaN
        d = d.dropna(subset=["HomeTeam", "AwayTeam"])   # filas vacías al final del CSV, si las hay
        d["temporada"] = f.stem.replace("laliga_", "")
        trozos.append(d)
    df = pd.concat(trozos, ignore_index=True).rename(columns=COLUMNAS)
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, format="mixed")   # admite '27/08/05' y '15/08/2024'
    df["anio_inicio"] = df["temporada"].str[:4].astype(int)
    df["temporada_completa"] = df.groupby("temporada")["local"].transform("size") == PARTIDOS_TEMPORADA_COMPLETA
    if desde is not None:
        df = df[df["anio_inicio"] >= desde]
    return df.sort_values("fecha").reset_index(drop=True)


def probabilidades_implicitas(cuota_local, cuota_empate, cuota_visitante):
    """Convierte cuotas en probabilidades.

    Una cuota de 2,00 equivale a una probabilidad de 1/2,00 = 50 %. Pero las tres
    probabilidades `1/cuota` suman más de 1: el exceso es el margen (la comisión) de la casa.
    Para obtener probabilidades que sumen 1 se reparte ese margen de forma proporcional.

    Devuelve (probabilidades con forma (n, 3) en el orden local-empate-visitante, margen).
    """
    inversas = 1.0 / np.column_stack([cuota_local, cuota_empate, cuota_visitante])
    suma = inversas.sum(axis=1)
    return inversas / suma[:, None], suma - 1.0


# Nombres con los que aparecen en los datos -> nombre completo para mostrar
NOMBRE_COMPLETO = {
    "Alaves": "Deportivo Alavés", "Almeria": "UD Almería", "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atlético de Madrid", "Barcelona": "FC Barcelona", "Betis": "Real Betis", "Cadiz": "Cádiz CF",
    "Celta": "Celta de Vigo", "Cordoba": "Córdoba CF", "Eibar": "SD Eibar", "Elche": "Elche CF",
    "Espanol": "RCD Espanyol", "Getafe": "Getafe CF", "Gimnastic": "Gimnàstic de Tarragona", "Girona": "Girona FC",
    "Granada": "Granada CF", "Hercules": "Hércules CF", "Huesca": "SD Huesca", "La Coruna": "Deportivo de La Coruña",
    "Las Palmas": "UD Las Palmas", "Leganes": "CD Leganés", "Levante": "Levante UD", "Malaga": "Málaga CF",
    "Mallorca": "RCD Mallorca", "Murcia": "Real Murcia", "Numancia": "CD Numancia", "Osasuna": "CA Osasuna",
    "Oviedo": "Real Oviedo", "Real Madrid": "Real Madrid", "Recreativo": "Recreativo de Huelva",
    "Santander": "Racing de Santander", "Sevilla": "Sevilla FC", "Sociedad": "Real Sociedad",
    "Sp Gijon": "Sporting de Gijón", "Tenerife": "CD Tenerife", "Valencia": "Valencia CF",
    "Valladolid": "Real Valladolid", "Vallecano": "Rayo Vallecano", "Villarreal": "Villarreal CF",
    "Xerez": "Xerez CD", "Zaragoza": "Real Zaragoza",
}


def nombre_completo(equipo):
    return NOMBRE_COMPLETO.get(equipo, equipo)


def miles(n):
    """5127 -> '5.127' (separador de miles en español)."""
    return f"{n:,}".replace(",", ".")


# --------------------------------------------------------------------------
# Estilo de gráficas (sobrio: marcas finas, cuadrícula casi invisible, texto en tinta neutra)
# --------------------------------------------------------------------------
COLOR_LOCAL = "#2a78d6"       # azul
COLOR_EMPATE = "#898781"      # gris: el empate es el resultado "neutro"
COLOR_VISITANTE = "#eb6834"   # naranja
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_SUAVE = "#898781"
CUADRICULA = "#e1e0d9"
SUPERFICIE = "#fcfcfb"


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
        "text.color": TINTA, "axes.labelcolor": TINTA_SECUNDARIA,
        "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 1,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": CUADRICULA, "grid.linewidth": 0.8,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": TINTA,
        "legend.frameon": False, "figure.dpi": 110, "lines.linewidth": 2,
    })
