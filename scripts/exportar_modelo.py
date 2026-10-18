"""Ajusta el modelo con todos los partidos descargados y guarda sus parámetros en `app/modelo_actual.json`.

La app lee ese archivo (ataque y defensa de cada equipo, ventaja local...) y no necesita los datos originales.
Volver a ejecutarlo tras cada jornada, después de `scripts/descargar_datos.py`, para actualizar las predicciones.

Uso (desde la raíz del repositorio)::

    python scripts/exportar_modelo.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import futbol as fb
import modelo as m

if __name__ == "__main__":
    df = fb.cargar_laliga()
    modelo = m.ModeloActual.desde_datos(df, semivida_anios=1.0, ridge=2.0)
    destino = RAIZ / "app" / "modelo_actual.json"
    destino.write_text(json.dumps(modelo.a_json(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Modelo guardado en {destino} ({fb.miles(modelo.n_partidos)} partidos hasta el {modelo.ultima_fecha:%d/%m/%Y})")
