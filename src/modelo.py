"""Modelo de goles de Poisson con ataque, defensa y ventaja local (tipo Maher / Dixon-Coles).

Idea. Los goles que marca cada equipo en un partido siguen una distribución de Poisson cuya media
(los "goles esperados") depende de:

    goles esperados del local     = exp( intercepto + ventaja_local + ataque[local]     - defensa[visitante] )
    goles esperados del visitante = exp( intercepto                 + ataque[visitante] - defensa[local]     )

* ``ataque``: cuánto marca un equipo (valores altos = marca más).
* ``defensa``: lo sólida que es su defensa (valores altos = concede menos).
* ``ventaja_local``: el efecto de jugar en casa, igual para todos los equipos.

Se ajusta por máxima verosimilitud con dos refinamientos: los partidos antiguos pesan menos
(decaimiento exponencial) y los parámetros se encogen hacia 0, la media (regularización "ridge"), para
que un equipo con pocos partidos no tenga valores extremos.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOLES = 10      # al construir la tabla de marcadores se ignoran los resultados con más goles


# --------------------------------------------------------------------------
# Ajuste
# --------------------------------------------------------------------------
def _menos_logverosimilitud(theta, li, vi, gl, gv, w, n, ridge):
    """Devuelve (-log verosimilitud penalizada, gradiente). `li`/`vi`: índice del equipo local/visitante."""
    mu, h = theta[0], theta[1]
    ataque, defensa = theta[2:2 + n], theta[2 + n:]
    log_l = mu + h + ataque[li] - defensa[vi]
    log_v = mu + ataque[vi] - defensa[li]
    el, ev = np.exp(log_l), np.exp(log_v)

    # log P(y | media) = y*log(media) - media  (se omite el término constante log y!)
    nll = -(w * (gl * log_l - el + gv * log_v - ev)).sum() + ridge * (ataque @ ataque + defensa @ defensa)

    rl, rv = w * (gl - el), w * (gv - ev)
    g = np.empty_like(theta)
    g[0] = -(rl.sum() + rv.sum())
    g[1] = -rl.sum()
    g[2:2 + n] = -(np.bincount(li, rl, n) + np.bincount(vi, rv, n)) + 2 * ridge * ataque
    g[2 + n:] = (np.bincount(vi, rl, n) + np.bincount(li, rv, n)) + 2 * ridge * defensa
    return nll, g


def ajustar(li, vi, gl, gv, w, n_equipos, ridge, theta0=None):
    """Ajusta los parámetros. Devuelve el vector theta = [intercepto, ventaja, ataque (n), defensa (n)]."""
    if theta0 is None:
        theta0 = np.zeros(2 + 2 * n_equipos)
        theta0[0] = np.log(max(np.average(np.r_[gl, gv], weights=np.r_[w, w]), 0.1))
    res = minimize(_menos_logverosimilitud, theta0, args=(li, vi, gl, gv, w, n_equipos, ridge),
                   jac=True, method="L-BFGS-B")
    return res.x


def goles_esperados(theta, n, li, vi):
    """Goles esperados (local, visitante) para partidos entre los equipos `li` y `vi`."""
    mu, h = theta[0], theta[1]
    ataque, defensa = theta[2:2 + n], theta[2 + n:]
    return np.exp(mu + h + ataque[li] - defensa[vi]), np.exp(mu + ataque[vi] - defensa[li])


# --------------------------------------------------------------------------
# De goles esperados a probabilidades 1-X-2
# --------------------------------------------------------------------------
def _tau(x, y, lam, mu, rho):
    """Corrección de Dixon-Coles para marcadores bajos: el 0-0 y el 1-1 ocurren algo más (o menos) de lo
    que daría el producto de dos Poisson independientes, y el 1-0 / 0-1 al revés."""
    t = np.ones_like(lam * mu, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def matriz_marcadores(lam, mu, rho=0.0):
    """Tabla P[x, y] = probabilidad de que el local marque x goles y el visitante y, para un partido."""
    k = np.arange(MAX_GOLES + 1)
    p = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))
    if rho != 0.0:
        p = p * _tau(k[:, None], k[None, :], lam, mu, rho)
    return p / p.sum()


def probs_1x2(lam, mu, rho=0.0):
    """Probabilidades (local, empate, visitante) para muchos partidos a la vez.

    `lam`, `mu`: arrays con los goles esperados de local y visitante de cada partido.
    """
    lam, mu = np.atleast_1d(lam), np.atleast_1d(mu)
    k = np.arange(MAX_GOLES + 1)
    pl, pv = poisson.pmf(k[None, :], lam[:, None]), poisson.pmf(k[None, :], mu[:, None])   # (n, k)
    joint = pl[:, :, None] * pv[:, None, :]                                               # (n, k, k)
    if rho != 0.0:
        joint = joint * _tau(k[None, :, None], k[None, None, :], lam[:, None, None], mu[:, None, None], rho)
    joint = joint / joint.sum(axis=(1, 2), keepdims=True)
    gana_local = np.tril(np.ones((len(k), len(k))), -1)           # x > y
    gana_visit = np.triu(np.ones((len(k), len(k))), 1)            # x < y
    p_l = (joint * gana_local).sum(axis=(1, 2))
    p_v = (joint * gana_visit).sum(axis=(1, 2))
    return np.column_stack([p_l, 1 - p_l - p_v, p_v])


def estimar_rho(gl, gv, lam, mu, w):
    """Valor de la corrección de Dixon-Coles que maximiza la verosimilitud de los marcadores bajos."""
    bajo = (gl <= 1) & (gv <= 1)
    gl, gv, lam, mu, w = gl[bajo], gv[bajo], lam[bajo], mu[bajo], w[bajo]

    def nll(rho):
        t = _tau(gl, gv, lam, mu, rho)
        return -(w * np.log(np.clip(t, 1e-9, None))).sum()

    return minimize_scalar(nll, bounds=(-0.3, 0.1), method="bounded").x


# --------------------------------------------------------------------------
# Validación temporal (walk-forward)
# --------------------------------------------------------------------------
def indices_equipos(df):
    equipos = sorted(set(df["local"]) | set(df["visitante"]))
    return {e: i for i, e in enumerate(equipos)}


def walk_forward(df, desde, hasta, semividas_anios=None, ridge=2.0, dixon_coles=False, dias_bloque=7,
                 indice=None, col_local="goles_local", col_visitante="goles_visitante"):
    """Predice cada partido entre `desde` y `hasta` usando SOLO los partidos anteriores a su bloque.

    El tiempo se divide en bloques de `dias_bloque` días. Antes de cada bloque se vuelve a ajustar el
    modelo con todo el pasado y se predicen los partidos del bloque. Así nunca se "mira el futuro".

    semividas_anios: un partido pesa la mitad cada `semividas_anios` años (None = todos pesan igual).
    col_local, col_visitante: columnas con los "goles" con los que se entrena el modelo. Por defecto los goles
        reales, pero puede ser cualquier cantidad no negativa (p. ej. goles esperados estimados a partir de
        los tiros). Las probabilidades 1-X-2 se evalúan siempre contra el resultado real (`resultado`).
    Devuelve un DataFrame con los partidos predichos: goles esperados y probabilidades 1-X-2.
    """
    df = df.sort_values("fecha").reset_index(drop=True)
    idx = indice or indices_equipos(df)
    n = len(idx)
    li_all = df["local"].map(idx).to_numpy()
    vi_all = df["visitante"].map(idx).to_numpy()
    gl_all = df[col_local].to_numpy(dtype=float)
    gv_all = df[col_visitante].to_numpy(dtype=float)
    dias = (df["fecha"] - df["fecha"].min()).dt.days.to_numpy(dtype=float)

    en_periodo = (df["fecha"] >= pd.Timestamp(desde)) & (df["fecha"] <= pd.Timestamp(hasta))
    inicio_bloque = df["fecha"].min() + pd.to_timedelta(
        ((df["fecha"] - df["fecha"].min()).dt.days // dias_bloque) * dias_bloque, unit="D")

    salidas, theta = [], None
    for ini in sorted(inicio_bloque[en_periodo].unique()):
        pred = en_periodo & (inicio_bloque == ini)
        entreno = (df["fecha"] < ini).to_numpy()
        if entreno.sum() < 200:
            continue
        if semividas_anios is None:
            w = np.ones(entreno.sum())
        else:
            antig = (dias[entreno].max() + 1 - dias[entreno]) / 365.25        # años hasta el último partido
            w = 0.5 ** (antig / semividas_anios)
        theta = ajustar(li_all[entreno], vi_all[entreno], gl_all[entreno], gv_all[entreno], w, n, ridge, theta)

        rho = 0.0
        if dixon_coles:
            lam_e, mu_e = goles_esperados(theta, n, li_all[entreno], vi_all[entreno])
            rho = estimar_rho(gl_all[entreno], gv_all[entreno], lam_e, mu_e, w)

        lam, mu = goles_esperados(theta, n, li_all[pred], vi_all[pred])
        p = probs_1x2(lam, mu, rho)
        bloque = df.loc[pred, ["fecha", "temporada", "local", "visitante", "resultado"]].copy()
        bloque["lam"], bloque["mu"], bloque["rho"] = lam, mu, rho
        bloque[["p_local", "p_empate", "p_visitante"]] = p
        salidas.append(bloque)
    return pd.concat(salidas)


class ModeloActual:
    """Modelo ajustado con todos los partidos disponibles hasta hoy (el que usa la app).

    Usa la configuración elegida en los cuadernos: goles reales, semivida de 1 año y ridge = 2.
    """

    def __init__(self, theta, equipos, ultima_fecha, n_partidos, actuales=()):
        self.indice = {e: i for i, e in enumerate(equipos)}
        self.nombres = {i: e for e, i in self.indice.items()}
        self.n = len(equipos)
        self.theta = np.asarray(theta, dtype=float)
        self.ultima_fecha = pd.Timestamp(ultima_fecha)
        self.n_partidos = int(n_partidos)
        self.actuales = list(actuales)          # equipos de la temporada en curso
        self.ventaja = float(self.theta[1])
        self.ataque = pd.Series(self.theta[2:2 + self.n], index=equipos)
        self.defensa = pd.Series(self.theta[2 + self.n:], index=equipos)

    @classmethod
    def desde_datos(cls, df, semivida_anios=1.0, ridge=2.0):
        """Ajusta el modelo con todos los partidos de `df`."""
        df = df.sort_values("fecha")
        indice = indices_equipos(df)
        ultima = df["fecha"].max()
        li = df["local"].map(indice).to_numpy()
        vi = df["visitante"].map(indice).to_numpy()
        antig = (ultima - df["fecha"]).dt.days.to_numpy() / 365.25 + 1 / 365.25
        w = 0.5 ** (antig / semivida_anios)
        theta = ajustar(li, vi, df["goles_local"].to_numpy(float), df["goles_visitante"].to_numpy(float),
                        w, len(indice), ridge)
        actuales = sorted(df[df["temporada"] == df["temporada"].iloc[-1]]["local"].unique())
        return cls(theta, sorted(indice, key=indice.get), ultima, len(df), actuales)

    def a_json(self):
        """Los parámetros ajustados (no los datos originales), en un diccionario listo para guardar."""
        return {"ultima_fecha": str(self.ultima_fecha.date()), "n_partidos": self.n_partidos,
                "equipos": [self.nombres[i] for i in range(self.n)], "actuales": self.actuales,
                "theta": self.theta.tolist()}

    @classmethod
    def desde_json(cls, d):
        return cls(d["theta"], d["equipos"], d["ultima_fecha"], d["n_partidos"], d["actuales"])

    def predecir(self, local, visitante):
        """Devuelve (goles esperados local, visitante, [p_local, p_empate, p_visitante], matriz de marcadores)."""
        li, vi = np.array([self.indice[local]]), np.array([self.indice[visitante]])
        lam, mu = goles_esperados(self.theta, self.n, li, vi)
        lam, mu = float(lam[0]), float(mu[0])
        return lam, mu, probs_1x2(np.array([lam]), np.array([mu]))[0], matriz_marcadores(lam, mu)


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def indice_resultado(resultado):
    """'H','D','A' -> 0,1,2"""
    return resultado.map({"H": 0, "D": 1, "A": 2}).to_numpy()


def log_loss(probs, y):
    """Pérdida logarítmica media. `probs`: (n, 3); `y`: 0/1/2."""
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()


def hueco_cerrado(perdida_modelo, perdida_base, perdida_referencia):
    """Fracción del hueco entre la base (p. ej. frecuencias históricas) y la referencia (p. ej. Pinnacle)
    que consigue cerrar el modelo. 0 = igual que la base; 1 = igual que la referencia."""
    return (perdida_base - perdida_modelo) / (perdida_base - perdida_referencia)
