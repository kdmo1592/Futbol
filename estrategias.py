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


# ============================================================
# 3. MERCADOS DERIVADOS Y PRUEBA DE VARIABLES (informe por partido)
# ============================================================
# Idea: el 1X2 y el Más/Menos 2.5 del propio partido (sin margen) se convierten en goles
# esperados, y de ahí salen los demás mercados (Ambos marcan, 1.5, 3.5, doble oportunidad).
# Se mide en las temporadas de prueba frente al rating de mercado y al modelo de goles,
# y se comprueba si la forma, el descanso o los tiros a puerta añaden algo a esa base.

DERIVADOS = {
    "btts": ("Ambos marcan", lambda i, j: (i > 0) & (j > 0)),
    "o15": ("Más de 1.5 goles", lambda i, j: i + j >= 2),
    "o25": ("Más de 2.5 goles", lambda i, j: i + j >= 3),
    "o35": ("Más de 3.5 goles", lambda i, j: i + j >= 4),
    "1x": ("Local o empate (1X)", lambda i, j: i >= j),
    "x2": ("Empate o visitante (X2)", lambda i, j: i <= j),
}
RHOS = [-0.15, -0.10, -0.05, 0.0]
TRAMOS_CAL = [0, .3, .4, .5, .6, .7, .8, .9, 1]


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def probs_derivadas(lam_l, lam_v, rho):
    lam_l, lam_v = np.asarray(lam_l, float), np.asarray(lam_v, float)
    out = {k: np.full(len(lam_l), np.nan) for k in DERIVADOS}
    ok = np.isfinite(lam_l) & np.isfinite(lam_v)
    if ok.any():
        M = mf.matriz_goles(lam_l[ok], lam_v[ok], rho)
        I, J = np.indices(M.shape[1:])
        for k, (_, f) in DERIVADOS.items():
            out[k][ok] = (M * f(I, J)).sum(axis=(1, 2))
    return out


def _resultados(v):
    gl, gv = v["gl"].to_numpy(), v["gv"].to_numpy()
    return {k: f(gl, gv).astype(float) for k, (_, f) in DERIVADOS.items()}


def anadir_implicitas(v, inversor):
    """Goles esperados que explican las cuotas del propio partido (antes del cierre y al cierre)."""
    import modelo_mercado as mm
    v = v.reset_index(drop=True).copy()
    for nombre, cierre in (("ape", False), ("cie", True)):
        p1, pou = mm.probs_mercado(v, cierre=cierre)
        lam = inversor.invertir(p1, pou)
        v[f"{nombre}_l"], v[f"{nombre}_v"] = lam[:, 0], lam[:, 1]
    return v


def evaluar_derivados(v, temporadas_test, rho_goles):
    """v: variables walk-forward con ape_*, cie_*, rat_*, gol_*. Devuelve tabla, calibración y rho elegido."""
    test = v["temporada"].isin(temporadas_test).to_numpy()
    Y = _resultados(v)
    ape_l, ape_v = v["ape_l"].to_numpy(float), v["ape_v"].to_numpy(float)
    dev = ~test & np.isfinite(ape_l)
    # rho del ajuste de empates: se elige fuera de la prueba
    mejor, rho_m = np.inf, 0.0
    for rho in RHOS:
        P = probs_derivadas(ape_l, ape_v, rho)
        s = np.mean([_ll(P[k][dev], Y[k][dev]).mean() for k in ("btts", "o15", "o35")]) if dev.sum() > 200 else 0.0
        if s < mejor - 1e-9:
            mejor, rho_m = s, rho
    fuentes = {"Cuotas del partido": probs_derivadas(ape_l, ape_v, rho_m),
               "Cierre": probs_derivadas(v["cie_l"], v["cie_v"], rho_m),
               "Rating de mercado": probs_derivadas(v["rat_l"], v["rat_v"], 0.0),
               "Modelo de goles": probs_derivadas(v["gol_l"], v["gol_v"], rho_goles)}
    comun = test.copy()
    for P in fuentes.values():
        comun &= np.isfinite(P["btts"])
    tabla, cal = [], {}
    if comun.sum() < 100:
        return {"rho_mercado": rho_m, "n": int(comun.sum()), "tabla": [], "calibracion": {}, "test": list(temporadas_test)}
    for k, (nombre, _) in DERIVADOS.items():
        y = Y[k]
        const = _ll(np.full(comun.sum(), y[~test].mean()), y[comun]).mean()
        fila = {"clave": k, "mercado": nombre, "base": float(y[comun].mean()), "ll_const": float(const), "fuentes": {}}
        for n, P in fuentes.items():
            ll = float(_ll(P[k][comun], y[comun]).mean())
            fila["fuentes"][n] = {"ll": ll, "mejora": float(1 - ll / const)}
        tabla.append(fila)
        t = pd.DataFrame({"p": fuentes["Cuotas del partido"][k][comun], "y": y[comun]})
        t["tramo"] = pd.cut(t["p"], TRAMOS_CAL)
        cal[k] = [{"tramo": f"{100 * tr.left:.0f}–{100 * tr.right:.0f} %", "n": int(len(g)),
                   "p": float(g["p"].mean()), "y": float(g["y"].mean())}
                  for tr, g in t.groupby("tramo", observed=True) if len(g) >= 20]
    return {"rho_mercado": rho_m, "n": int(comun.sum()), "tabla": tabla, "calibracion": cal, "test": list(temporadas_test)}


GRUPOS_VARIABLES = {
    "Forma de los últimos 5 (goles frente a lo esperado)": ["Lr_dgf", "Lr_dgc", "Vr_dgf", "Vr_dgc"],
    "Puntos de los últimos 5": ["Lr_pts", "Vr_pts"],
    "% de Ambos marcan y de Más de 2.5 recientes": ["Lr_bt", "Vr_bt", "Lr_ov", "Vr_ov"],
    "Tiros a puerta (fuerza y últimos 5)": ["sot_l", "sot_v", "Lr_sf", "Lr_sc", "Vr_sf", "Vr_sc"],
    "Días de descanso": ["desc_l", "desc_v"],
}


def _variables_forma(v, hist):
    """Variables previas a cada partido: medias de los 5 anteriores de cada equipo y fuerza en tiros a puerta."""
    largo = pd.concat([
        pd.DataFrame({"i": v.index, "eq": v["local"], "f": v["fecha"], "gf": v["gl"], "gc": v["gv"],
                      "xf": v["cie_l"], "xc": v["cie_v"], "sf": v["hst"], "sc": v["ast"], "lado": "L"}),
        pd.DataFrame({"i": v.index, "eq": v["visitante"], "f": v["fecha"], "gf": v["gv"], "gc": v["gl"],
                      "xf": v["cie_v"], "xc": v["cie_l"], "sf": v["ast"], "sc": v["hst"], "lado": "V"})])
    largo = largo.sort_values(["eq", "f"], kind="stable")
    largo["pts"] = np.where(largo["gf"] > largo["gc"], 3.0, np.where(largo["gf"] == largo["gc"], 1.0, 0.0))
    largo["bt"] = ((largo["gf"] > 0) & (largo["gc"] > 0)).astype(float)
    largo["ov"] = (largo["gf"] + largo["gc"] >= 3).astype(float)
    largo["dgf"], largo["dgc"] = largo["gf"] - largo["xf"], largo["gc"] - largo["xc"]
    F = pd.DataFrame(index=v.index)
    g = largo.groupby("eq")
    for c in ("dgf", "dgc", "pts", "bt", "ov", "sf", "sc"):
        largo["r_" + c] = g[c].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        for lado in ("L", "V"):
            F[f"{lado}r_{c}"] = largo[largo["lado"] == lado].set_index("i")["r_" + c].reindex(v.index)
    F["desc_l"], F["desc_v"] = np.log(v["desc_l"].to_numpy(float)), np.log(v["desc_v"].to_numpy(float))
    # fuerza en tiros a puerta (mismo motor, walk-forward)
    F["sot_l"], F["sot_v"] = np.nan, np.nan
    temporadas = sorted(v["temporada"].unique())
    for liga in v["liga"].unique():
        sub = hist[hist["liga"] == liga].reset_index(drop=True)
        if sub["hst"].notna().sum() < 200:
            continue
        d = mf.construir_datos(sub)
        d = mf.Datos(f=d.f, il=d.il, iv=d.iv, gl=sub["hst"].to_numpy(float), gv=sub["ast"].to_numpy(float),
                     n_eq=d.n_eq, indice=d.indice)
        idx = np.where(sub["temporada"].isin(temporadas))[0]
        a, b = mf.lambdas_walkforward(d, idx, 365.0, 3.0, tope=TOPE_CONTEO)
        t = sub.iloc[idx][["fecha", "local", "visitante"]].assign(a=a, b=b)
        m = v.loc[v["liga"] == liga, ["fecha", "local", "visitante"]].reset_index().merge(
            t, on=["fecha", "local", "visitante"], how="left").set_index("index")
        F.loc[m.index, "sot_l"], F.loc[m.index, "sot_v"] = m["a"], m["b"]
    return F


def probar_variables(v, hist, temporadas_test, rho_mercado, n_boot=1000, seed=7):
    """
    ¿Mejora la predicción si a la probabilidad derivada de las cuotas se le añaden otras variables?
    Regresión logística entrenada fuera de la prueba; diferencia de log loss en la prueba con IC 95 %.
    Negativa = la variable ayuda. Si el intervalo incluye el 0, no se distingue de la suerte.
    """
    import modelo_mercado as mm
    F = _variables_forma(v, hist)
    P = probs_derivadas(v["ape_l"], v["ape_v"], rho_mercado)
    Y = _resultados(v)
    test = v["temporada"].isin(temporadas_test).to_numpy()
    rng = np.random.default_rng(seed)
    filas = []
    for k in ("btts", "o25", "1x"):
        y, base = Y[k], _logit(P[k])
        for nombre, cols in GRUPOS_VARIABLES.items():
            X = np.column_stack([base] + [F[c].to_numpy(float) for c in cols])
            ok = np.isfinite(X).all(1)
            tr, te = ok & ~test, ok & test
            if tr.sum() < 1000 or te.sum() < 300:
                continue
            m = mm.Softmax().fit(X[tr], np.column_stack([y[tr], 1 - y[tr]]), 1e-3)
            dif = _ll(m.predict(X[te])[:, 0], y[te]) - _ll(P[k][te], y[te])
            bs = dif[rng.integers(0, len(dif), (n_boot, len(dif)))].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            filas.append({"mercado": DERIVADOS[k][0], "variables": nombre, "n": int(te.sum()), "dif": float(dif.mean()),
                          "ic_lo": float(lo), "ic_hi": float(hi),
                          "veredicto": "ayuda" if hi < 0 else "empeora" if lo > 0 else "no se distingue"})
    return filas


# ============================================================
# 4. AJUSTE DE TARJETAS POR EL CONTEXTO DEL PARTIDO
# ============================================================
# Medido en las temporadas de prueba: un partido igualado tiene MÁS tarjetas que uno con
# favorito claro, aunque los dos equipos tengan el mismo rating. El árbitro, en cambio, no
# mejoró la predicción de forma medible (solo hay árbitro en la Premier).
#
#   tarjetas esperadas = rating x exp(b0 + b1*log(goles esperados) + b2*|diferencia de goles|)
#
# Los coeficientes se ajustan fuera de la prueba y se comprueban dentro. Si no mejoran,
# el robot no los publica y la app sigue con el rating a secas.

LINEAS_EVAL_TAR = [2.5, 3.5, 4.5, 5.5, 6.5]


def _poisson_log(X, y, offset):
    """Poisson con log-enlace y desplazamiento fijo: mu = offset * exp(X b)."""
    from scipy.optimize import minimize
    off = np.log(np.clip(offset, 1e-6, None))

    def f(b):
        eta = off + X @ b
        mu = np.exp(np.clip(eta, -20, 20))
        return float((mu - y * eta).sum() / len(y)), (X * (mu - y)[:, None]).sum(0) / len(y)

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def _brier_lineas(y, mu, k, lineas):
    out = np.zeros(len(y))
    for linea in lineas:
        p = np.clip(p_mas_de(mu, k, linea), 1e-4, 1 - 1e-4)
        out += (p - (y > linea).astype(float)) ** 2
    return out / len(lineas)


def ajuste_tarjetas(v, hist, temporadas_test, n_boot=2000, seed=11):
    """Devuelve los coeficientes si mejoran de verdad en la prueba; si no, None."""
    tl, tv = columnas_conteo(hist, "tarjetas")
    h = hist.assign(y_tar=(tl + tv))
    clave = ["liga", "fecha", "local", "visitante"]
    datos = v.merge(h[clave + ["y_tar"]], on=clave, how="inner")
    # rating walk-forward de tarjetas
    trozos = []
    for liga in sorted(datos["liga"].unique()):
        sub = hist[hist["liga"] == liga].reset_index(drop=True)
        if sub["hy"].notna().sum() < 200:
            continue
        d = datos_conteo(sub, "tarjetas")
        idx = np.where(sub["temporada"].isin(sorted(datos["temporada"].unique())))[0]
        if len(idx) == 0:
            continue
        cfg = CONTEOS["tarjetas"]
        a, b = mf.lambdas_walkforward(d, idx, cfg["vida_media"], cfg["k_prior"], tope=TOPE_CONTEO)
        trozos.append(sub.iloc[idx][clave].assign(rat_tar=a + b))
    if not trozos:
        return None
    datos = datos.merge(pd.concat(trozos, ignore_index=True), on=clave, how="inner")
    y = datos["y_tar"].to_numpy(float)
    base = datos["rat_tar"].to_numpy(float)
    tot = (datos["ape_l"] + datos["ape_v"]).to_numpy(float)
    dif = np.abs(datos["ape_l"] - datos["ape_v"]).to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(base) & np.isfinite(tot) & np.isfinite(dif) & (base > 0)
    test = datos["temporada"].isin(temporadas_test).to_numpy()
    dev, te = ok & ~test, ok & test
    if dev.sum() < 1500 or te.sum() < 400:
        return None
    X = np.column_stack([np.ones(len(datos)), np.log(np.clip(tot, 0.1, None)), dif])
    b = _poisson_log(X[dev], y[dev], base[dev])
    mu = base * np.exp(np.clip(X @ b, -5, 5))
    k0, k1 = k_momentos(y[dev], base[dev]), k_momentos(y[dev], mu[dev])
    b0 = _brier_lineas(y[te], base[te], k0, LINEAS_EVAL_TAR)
    b1 = _brier_lineas(y[te], mu[te], k1, LINEAS_EVAL_TAR)
    dif_b = b1 - b0
    rng = np.random.default_rng(seed)
    boot = dif_b[rng.integers(0, len(dif_b), (n_boot, len(dif_b)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    res = {"coef": [float(x) for x in b], "k": round(float(k1), 2), "k_sin": round(float(k0), 2),
           "n_test": int(te.sum()), "brier_sin": float(b0.mean()), "brier_con": float(b1.mean()),
           "dif": float(dif_b.mean()), "ic_lo": float(lo), "ic_hi": float(hi), "usar": bool(hi < 0),
           "ejemplos": [[d_, round(float(np.exp(b[0] + b[1] * np.log(2.7) + b[2] * d_)), 3)] for d_ in (0.2, 1.0, 2.0)]}
    return res
