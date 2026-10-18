"""Descarga los resultados y cuotas de LaLiga (Primera División) de football-data.co.uk.

Cada temporada es un CSV (`SP1.csv`; SP1 = Primera División de España). Se guardan
sin modificar en `data/raw/` con el nombre `laliga_AAAA-AA.csv`.

Uso (desde la raíz del repositorio)::

    python scripts/descargar_datos.py
    python scripts/descargar_datos.py --desde 2010   # solo desde la temporada 2010/11

Fuente: https://www.football-data.co.uk/spainm.php  (consultar allí sus condiciones de uso).
"""
import argparse
import datetime
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
URL = "https://www.football-data.co.uk/mmz4281/{codigo}/SP1.csv"


def codigo_temporada(anio_inicio):
    """2005 -> '0506' (temporada 2005/06)."""
    return f"{anio_inicio % 100:02d}{(anio_inicio + 1) % 100:02d}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--desde", type=int, default=2005, help="año de inicio de la primera temporada (por defecto 2005)")
    args = p.parse_args()

    hoy = datetime.date.today()
    ultimo = hoy.year if hoy.month >= 7 else hoy.year - 1   # la temporada empieza en verano
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for anio in range(args.desde, ultimo + 1):
        codigo = codigo_temporada(anio)
        destino = RAW_DIR / f"laliga_{anio}-{(anio + 1) % 100:02d}.csv"
        r = requests.get(URL.format(codigo=codigo), timeout=60)
        r.raise_for_status()
        destino.write_bytes(r.content)
        print(f"{destino.name}: {len(r.content) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
