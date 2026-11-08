"""
CÓRNERS, TARJETAS Y BACKTEST DE "AMBOS MARCAN"
==============================================

1. CÓRNERS Y TARJETAS
   Mismo motor que los goles (fuerza "a favor" y "en contra" de cada equipo, con
   decaimiento temporal), pero aplicado a los córners y a las tarjetas de cada equipo.
   El total se modela con una binomial negativa: los córners varían más de lo que
   permite una Poisson. Se evalúa contra la referencia más simple posible (la
   frecuencia de la liga en las dos temporadas anteriores) para saber si aporta algo.
   football-data no publica cuotas de córners ni de tarjetas: no hay comparación con el mercado.

2. AMBOS MARCAN
   Reproduce la regla "apuesto a Ambos marcan si local y visitante tienen más de un X %
   de marcar" partido a partido, usando solo información anterior a cada partido.
   football-data no publica cuotas de Ambos marcan, así que no se puede calcular el
   beneficio. Se calcula la CUOTA MÍNIMA que habría hecho falta para no perder
   (1 / % de acierto). Si la cuota que te ofrecen es menor, la regla pierde dinero.
"""

import numpy as np
import pandas as pd
from scipy.stats import nbinom

import motor_futbol as mf

# ============================================================
# 1. CÓRNERS Y TARJETAS
# ============================================================

CONTEOS = {
    "corners": {"nombre": "Córners", "vida_media": 365.0, "k_prior": 3.0,
                "lineas_total": [7.5, 8.5, 9.5, 10.5, 11.5], "lineas_equipo": [3.5, 4.5, 5.5, 6.5]},
    "tarjetas": {"nombre": "Tarjetas", "vida_media": 365.0, "k_prior": 10.0,
                 "lineas_total": [2.5, 3.5, 4.5, 5.5, 6.5], "lineas_equipo": [0.5, 1.5, 2.5, 3.5]},
}
TOPE_CONTEO = 15.0
K_MAX = 1e4                       # dispersión "infinita" = Poisson


def columnas_conteo(hist, tipo):
    """Serie (local, visitante) del conteo. Tarjetas = amarillas + rojas."""
    if tipo == "corners":
        return hist["hc"].astype(float), hist["ac"].astype(float)
    return (hist["hy"] + hist["hr"].fillna(0)).astype(float), (hist["ay"] + hist["ar"].fillna(0)).astype(float)


def datos_conteo(sub, tipo):
    cl, cv = columnas_conteo(sub, tipo)
    d = mf.construir_datos(sub)
    return mf.Datos(f=d.f, il=d.il, iv=d.iv, gl=cl.to_numpy(float), gv=cv.to_numpy(float),
                    n_eq=d.n_eq, indice=d.indice)


def k_momentos(y, mu):
    """Dispersión de la binomial negativa (Var = mu + mu²/k) por el método de momentos."""
    exceso = ((y - mu) ** 2).sum() - mu.sum()
    return float(np.clip((mu ** 2).sum() / exceso, 1.0, K_MAX)) if exceso > 0 else K_MAX


def p_mas_de(mu, k, linea):
    mu = np.asarray(mu, float)
    return nbinom.sf(np.floor(linea), k, k / (k + mu))


def _predicciones_conteo(hist, tipo, temporadas):
    cfg = CONTEOS[tipo]
    trozos = []
    for liga in hist["liga"].unique():
        sub = hist[hist["liga"] == liga].reset_index(drop=True)
        d = datos_conteo(sub, tipo)
        idx = np.where(sub["temporada"].isin(temporadas) & np.isfinite(d.gl) & np.isfinite(d.gv))[0]
        if len(idx) == 0:
            continue
        ml, mv = mf.lambdas_walkforward(d, idx, cfg["vida_media"], cfg["k_prior"], tope=TOPE_CONTEO)
        t = sub.iloc[idx][["liga", "temporada", "fecha"]].copy()
        t["yl"], t["yv"], t["ml"], t["mv"] = d.gl[idx], d.gv[idx], ml, mv
        trozos.append(t)
    if not trozos:
        return pd.DataFrame()
    t = pd.concat(trozos, ignore_index=True)
    return t[np.isfinite(t["ml"]) & np.isfinite(t["mv"])].reset_index(drop=True)


def _frecuencia_previa(hist, tipo, liga, temporada, linea):
    """Referencia ingenua: % de partidos por encima de la línea en las 2 temporadas anteriores de la liga."""
    temps = sorted(t for t in hist.loc[hist["liga"] == liga, "temporada"].unique() if t < temporada)[-2:]
    h = hist[(hist["liga"] == liga) & hist["temporada"].isin(temps)]
    cl, cv = columnas_conteo(h, tipo)
    tot = (cl + cv).dropna()
    if len(tot) < 50:
        return np.nan
    return (float((tot > linea).sum()) + 1) / (len(tot) + 2)


def evaluar_conteos(hist, temporadas_dev, temporadas_test):
    """Ajusta la dispersión en 'desarrollo' y mide en 'test' frente a la frecuencia de la liga."""
    salida = {}
    for tipo, cfg in CONTEOS.items():
        t = _predicciones_conteo(hist, tipo, list(temporadas_dev) + list(temporadas_test))
        if t.empty:
            continue
        dev = t[t["temporada"].isin(temporadas_dev)]
        te = t[t["temporada"].isin(temporadas_test)].copy()
        if len(dev) < 200 or len(te) < 100:
            continue
        k_tot = k_momentos((dev["yl"] + dev["yv"]).to_numpy(), (dev["ml"] + dev["mv"]).to_numpy())
        tot, mu = (te["yl"] + te["yv"]).to_numpy(), (te["ml"] + te["mv"]).to_numpy()
        lineas = []
        for linea in cfg["lineas_total"]:
            y = (tot > linea).astype(float)
            pm = np.clip(p_mas_de(mu, k_tot, linea), 1e-4, 1 - 1e-4)
            cache = {c: _frecuencia_previa(hist, tipo, c[0], c[1], linea)
                     for c in set(zip(te["liga"], te["temporada"]))}
            pb = np.array([cache[c] for c in zip(te["liga"], te["temporada"])])
            ok = np.isfinite(pb)
            pb = np.clip(pb, 1e-4, 1 - 1e-4)
            ll = lambda p: float(-(y[ok] * np.log(p[ok]) + (1 - y[ok]) * np.log(1 - p[ok])).mean())
            lineas.append({"linea": linea, "n": int(ok.sum()), "frecuencia": float(y[ok].mean()),
                           "p_media_modelo": float(pm[ok].mean()),
                           "brier_modelo": float(((pm[ok] - y[ok]) ** 2).mean()),
                           "brier_liga": float(((pb[ok] - y[ok]) ** 2).mean()),
                           "ll_modelo": ll(pm), "ll_liga": ll(pb)})
        # todo el histórico para la dispersión que usará la app
        k_tot_app = k_momentos(np.r_[dev["yl"] + dev["yv"], te["yl"] + te["yv"]], np.r_[dev["ml"] + dev["mv"], mu])
        k_lado_app = k_momentos(np.r_[dev["yl"], dev["yv"], te["yl"], te["yv"]],
                                np.r_[dev["ml"], dev["mv"], te["ml"], te["mv"]])
        salida[tipo] = {"nombre": cfg["nombre"], "k_total": round(k_tot_app, 2), "k_lado": round(k_lado_app, 2),
                        "lineas_total": cfg["lineas_total"], "lineas_equipo": cfg["lineas_equipo"],
                        "test": list(temporadas_test), "n_test": int(len(te)),
                        "corr_total": float(np.corrcoef(mu, tot)[0, 1]), "evaluacion": lineas}
    return salida


def fuerzas_conteo(sub, tipo, corte):
    cfg = CONTEOS[tipo]
    d = datos_conteo(sub, tipo)
    return mf.ajustar_fuerzas(d, corte, cfg["vida_media"], cfg["k_prior"])


# ============================================================
# 2. AMBOS MARCAN
# ============================================================

UMBRAL_USUARIO = 0.62
UMBRALES_EQUIPO = [round(x, 2) for x in np.arange(0.50, 0.761, 0.02)]
UMBRALES_BTTS = [round(x, 2) for x in np.arange(0.48, 0.661, 0.02)]
MIN_PARTIDOS_EMP = 4


def _wilson(aciertos, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = aciertos / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - m, c + m


def _resumen(sel, btts):
    n, a = int(sel.sum()), int(btts[sel].sum())
    if n == 0:
        return {"n": 0, "acierto": None, "cuota_min": None, "cuota_min_lo": None, "cuota_min_hi": None}
    lo, hi = _wilson(a, n)
    return {"n": n, "acierto": a / n, "cuota_min": n / a if a else None,
            "cuota_min_lo": 1 / hi, "cuota_min_hi": 1 / lo if lo > 0 else None}


def porcentaje_marcaje_previo(v):
    """% de partidos de la temporada (antes de esa fecha) en que el local marcó EN CASA y el visitante FUERA."""
    emp_l = np.full(len(v), np.nan)
    emp_v = np.full(len(v), np.nan)
    pos = {i: k for k, i in enumerate(v.index)}
    for _, g in v.sort_values("fecha").groupby(["liga", "temporada"]):
        casa, fuera = {}, {}
        for _, gd in g.groupby("fecha"):
            for r in gd.itertuples():
                h, a = casa.get(r.local, []), fuera.get(r.visitante, [])
                if len(h) >= MIN_PARTIDOS_EMP:
                    emp_l[pos[r.Index]] = np.mean(h)
                if len(a) >= MIN_PARTIDOS_EMP:
                    emp_v[pos[r.Index]] = np.mean(a)
            for r in gd.itertuples():
                casa.setdefault(r.local, []).append(int(r.gl > 0))
                fuera.setdefault(r.visitante, []).append(int(r.gv > 0))
    return emp_l, emp_v


def prob_marcar(lam_l, lam_v, rho=0.0):
    M = mf.matriz_goles(np.asarray(lam_l, float), np.asarray(lam_v, float), rho)
    return 1 - M[:, 0, :].sum(1), 1 - M[:, :, 0].sum(1), M[:, 1:, 1:].sum((1, 2))


def backtest_btts(v, temporadas_dev, temporadas_test):
    """
    v: variables walk-forward de ModeloMercado (rat_l, rat_v = goles esperados del rating de mercado).
    Dos versiones de la regla:
      - 'modelo': probabilidad de marcar según el rating de mercado (la que enseña la app);
      - 'casa_fuera': % de partidos de la temporada en que el local marcó en casa y el visitante fuera.
    """
    v = v[np.isfinite(v["rat_l"]) & np.isfinite(v["rat_v"])].copy()
    pl, pv, pb = prob_marcar(v["rat_l"].to_numpy(), v["rat_v"].to_numpy())
    el, ev = porcentaje_marcaje_previo(v)
    btts = ((v["gl"] > 0) & (v["gv"] > 0)).to_numpy()
    temp = v["temporada"].to_numpy()
    bloques = {"desarrollo": np.isin(temp, temporadas_dev), "prueba": np.isin(temp, temporadas_test)}

    por_temporada = []
    for t in sorted(set(temp)):
        m = temp == t
        fila = {"temporada": t, "partidos": int(m.sum()), "base": float(btts[m].mean())}
        fila["modelo"] = _resumen(m & (pl > UMBRAL_USUARIO) & (pv > UMBRAL_USUARIO), btts)
        fila["casa_fuera"] = _resumen(m & (el > UMBRAL_USUARIO) & (ev > UMBRAL_USUARIO), btts)
        fila["modelo"]["pct_partidos"] = fila["modelo"]["n"] / max(int(m.sum()), 1)
        fila["casa_fuera"]["pct_partidos"] = fila["casa_fuera"]["n"] / max(int(m.sum()), 1)
        por_temporada.append(fila)

    barrido_equipo, barrido_btts = [], []
    for u in UMBRALES_EQUIPO:
        fila = {"umbral": u}
        for nb, m in bloques.items():
            fila[nb] = _resumen(m & (pl > u) & (pv > u), btts)
            fila[nb + "_casa_fuera"] = _resumen(m & (el > u) & (ev > u), btts)
        barrido_equipo.append(fila)
    for u in UMBRALES_BTTS:
        fila = {"umbral": u}
        for nb, m in bloques.items():
            fila[nb] = _resumen(m & (pb >= u), btts)
        barrido_btts.append(fila)

    # calibración de P(ambos marcan) del modelo en la prueba
    mp = bloques["prueba"]
    cal = []
    if mp.any():
        t = pd.DataFrame({"p": pb[mp], "y": btts[mp].astype(float)})
        t["tramo"] = pd.cut(t["p"], [0, .40, .45, .50, .55, .60, .65, 1])
        for tramo, g in t.groupby("tramo", observed=True):
            cal.append({"tramo": f"{100 * tramo.left:.0f}–{100 * tramo.right:.0f} %", "n": int(len(g)),
                        "p_media": float(g["p"].mean()), "frecuencia": float(g["y"].mean())})
    return {"umbral_usuario": UMBRAL_USUARIO, "desarrollo": list(temporadas_dev), "prueba": list(temporadas_test),
            "base_prueba": float(btts[mp].mean()) if mp.any() else None,
            "por_temporada": por_temporada, "barrido_equipo": barrido_equipo, "barrido_btts": barrido_btts,
            "calibracion": cal}
