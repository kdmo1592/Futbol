"""
MOTOR DE LA APP (sin Streamlit): datos, modelo, mercados, valor y paper trading.

Se puede probar y ejecutar por separado; la interfaz (app_futbol.py) solo lo llama.
El modelo es el mismo motor validado en el backtest v4: Poisson multiplicativo
iterado por liga, con decaimiento temporal, shrinkage y corrección Dixon-Coles.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

LIGAS = {
    "SP1": "LaLiga",
    "SP2": "LaLiga 2 (Segunda)",
    "E0": "Premier League",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

TEMPORADAS = ["2122", "2223", "2324", "2425", "2526", "2627"]
TEMPORADAS_BACKTEST = ["2324", "2425", "2526", "2627"]     # las anteriores calientan el modelo

URL_HISTORICO = "https://www.football-data.co.uk/mmz4281/{t}/{liga}.csv"
URL_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"

MAX_GOLES = 10
MIN_PARTIDOS_EQUIPO = 5
N_ITER_MAX = 60

# Hiperparámetros por defecto: los del backtest v3 (LaLiga). Reajústalos con v4.
VIDA_MEDIA_DEFECTO = 730.0
K_PRIOR_DEFECTO = 3.0
RHO_DEFECTO = -0.10

# Grupos de cuotas: nombre -> lista (una por resultado) de columnas candidatas
GRUPOS_CUOTAS = {
    "b365": [["B365H"], ["B365D"], ["B365A"]],
    "b365ou": [["B365>2.5", "B365O2.5"], ["B365<2.5", "B365U2.5"]],
    "avg": [["AvgH"], ["AvgD"], ["AvgA"]],
    "avgou": [["Avg>2.5"], ["Avg<2.5"]],
    "cierre": [["AvgCH"], ["AvgCD"], ["AvgCA"]],
    "cierreou": [["AvgC>2.5"], ["AvgC<2.5"]],
    "ps": [["PSH"], ["PSD"], ["PSA"]],
    "psou": [["P>2.5"], ["P<2.5"]],
    "psc": [["PSCH"], ["PSCD"], ["PSCA"]],
    "pscou": [["PC>2.5"], ["PC<2.5"]],
    "b365c": [["B365CH"], ["B365CD"], ["B365CA"]],
    "b365cou": [["B365C>2.5"], ["B365C<2.5"]],
}

# Mercados con cuotas históricas disponibles en football-data (1X2 y Más/Menos 2.5).
MERCADOS = {
    "1X2": {
        "k": 3, "selecciones": ["Local", "Empate", "Visitante"],
        "p": ["p1", "px", "p2"], "cuota": "b365", "avg": "avg",
        "cierre": "cierre", "b365c": "b365c", "y": "y",
    },
    "Más/Menos 2.5": {
        "k": 2, "selecciones": ["Más de 2.5", "Menos de 2.5"],
        "p": ["p_over", "p_under"], "cuota": "b365ou", "avg": "avgou",
        "cierre": "cierreou", "b365c": "b365cou", "y": "y_ou",
    },
}


# ============================================================
# 2. DESCARGA Y NORMALIZACIÓN
# ============================================================

def descargar_csv(url, timeout=30):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    try:
        texto = r.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = r.content.decode("latin1")
    return pd.read_csv(io.StringIO(texto), on_bad_lines="skip")


def _primera_col(raw, candidatas):
    for c in candidatas:
        if c in raw.columns:
            v = pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=float)
            return np.where(v > 1.0, v, np.nan)          # una cuota válida es > 1
    return np.full(len(raw), np.nan)


def _anadir_cuotas(df, raw):
    for grupo, listas in GRUPOS_CUOTAS.items():
        for k, candidatas in enumerate(listas):
            df[f"{grupo}_{k}"] = _primera_col(raw, candidatas)
    return df


def normalizar_historico(raw):
    necesarias = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "liga", "temporada"]
    faltan = [c for c in necesarias if c not in raw.columns]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias: {faltan}")

    raw = raw.reset_index(drop=True)
    df = pd.DataFrame({
        "fecha": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce"),
        "liga": raw["liga"].astype(str),
        "temporada": raw["temporada"].astype(str),
        "local_n": raw["HomeTeam"].astype(str).str.strip(),
        "visitante_n": raw["AwayTeam"].astype(str).str.strip(),
        "gl": pd.to_numeric(raw["FTHG"], errors="coerce"),
        "gv": pd.to_numeric(raw["FTAG"], errors="coerce"),
        "y": raw["FTR"].map({"H": 0, "D": 1, "A": 2}),
    })
    for col, nuevo in (("HS", "hs"), ("AS", "as_"), ("HST", "hst"), ("AST", "ast"), ("HC", "hc"), ("AC", "ac"),
                       ("HY", "hy"), ("AY", "ay"), ("HR", "hr"), ("AR", "ar"), ("HTHG", "htgl"), ("HTAG", "htgv")):
        df[nuevo] = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else np.nan
    df = _anadir_cuotas(df, raw)
    df = df.dropna(subset=["fecha", "gl", "gv", "y"]).copy()
    df[["gl", "gv", "y"]] = df[["gl", "gv"]].astype(int).join(df["y"].astype(int))

    # Coherencia: FTR debe cuadrar con los goles
    y_goles = np.where(df["gl"] > df["gv"], 0, np.where(df["gl"] == df["gv"], 1, 2))
    df = df[y_goles == df["y"]].copy()

    df["local"] = df["liga"] + ":" + df["local_n"]
    df["visitante"] = df["liga"] + ":" + df["visitante_n"]
    df["y_ou"] = np.where(df["gl"] + df["gv"] >= 3, 0, 1)          # 0 = Más de 2.5
    df = df.sort_values(["liga", "fecha", "local"], kind="stable").reset_index(drop=True)

    # Días de descanso de cada equipo desde su partido anterior
    largo = pd.concat([
        pd.DataFrame({"eq": df["local"], "fecha": df["fecha"], "fila": df.index, "lado": "l"}),
        pd.DataFrame({"eq": df["visitante"], "fecha": df["fecha"], "fila": df.index, "lado": "v"}),
    ]).sort_values(["eq", "fecha"], kind="stable")
    largo["desc"] = largo.groupby("eq")["fecha"].diff().dt.days
    for lado in ("l", "v"):
        serie = largo[largo["lado"] == lado].set_index("fila")["desc"]
        df[f"desc_{lado}"] = serie.reindex(df.index).fillna(7).clip(2, 21).to_numpy()
    return df


def dias_descanso(hist, liga, equipo, fecha):
    """Días desde el último partido del equipo antes de `fecha` (7 por defecto)."""
    h = hist[(hist["liga"] == liga) & ((hist["local_n"] == equipo) | (hist["visitante_n"] == equipo))
             & (hist["fecha"] < pd.Timestamp(fecha))]
    if h.empty:
        return 7.0
    return float(np.clip((pd.Timestamp(fecha) - h["fecha"].max()).days, 2, 21))


def cargar_historico(descargador=descargar_csv, ligas=None, temporadas=None):
    ligas = ligas or list(LIGAS)
    temporadas = temporadas or TEMPORADAS
    frames, errores = [], []
    for liga in ligas:
        for t in temporadas:
            url = URL_HISTORICO.format(t=t, liga=liga)
            try:
                df = descargador(url)
                if "HomeTeam" not in df.columns:
                    errores.append(f"{liga} {t}: sin columna HomeTeam")
                    continue
                df = df.copy()
                df["liga"], df["temporada"] = liga, t
                frames.append(df)
            except Exception as exc:                       # noqa: BLE001
                errores.append(f"{liga} {t}: {exc}")
    if not frames:
        raise RuntimeError("No se pudo descargar ningún histórico.")
    return normalizar_historico(pd.concat(frames, ignore_index=True)), errores


def normalizar_fixtures(raw):
    """Próximos partidos (formato fixtures.csv de football-data)."""
    for c in ("Div", "Date", "HomeTeam", "AwayTeam"):
        if c not in raw.columns:
            raise ValueError(f"fixtures: falta la columna {c}")
    raw = raw.reset_index(drop=True)
    df = pd.DataFrame({
        "liga": raw["Div"].astype(str),
        "fecha": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce"),
        "hora": raw["Time"].astype(str).replace("nan", "") if "Time" in raw.columns else "",
        "local_n": raw["HomeTeam"].astype(str).str.strip(),
        "visitante_n": raw["AwayTeam"].astype(str).str.strip(),
    })
    df = _anadir_cuotas(df, raw)
    df = df[df["liga"].isin(LIGAS)].dropna(subset=["fecha"]).copy()
    df["local"] = df["liga"] + ":" + df["local_n"]
    df["visitante"] = df["liga"] + ":" + df["visitante_n"]
    return df.sort_values(["fecha", "hora"]).reset_index(drop=True)


# ============================================================
# 3. MODELO: POISSON MULTIPLICATIVO ITERADO POR LIGA
# ============================================================
#   lambda_local     = mu_l * ataque[local]     * defensa[visitante]
#   lambda_visitante = mu_v * ataque[visitante] * defensa[local]
# La ventaja de campo va en mu_l vs mu_v (una fuerza por equipo, sin partir casa/fuera).

@dataclass
class Datos:
    f: np.ndarray
    il: np.ndarray
    iv: np.ndarray
    gl: np.ndarray
    gv: np.ndarray
    n_eq: int
    indice: dict


def construir_datos(sub):
    equipos = sorted(set(sub["local"]) | set(sub["visitante"]))
    indice = {e: i for i, e in enumerate(equipos)}
    return Datos(
        f=sub["fecha"].values.astype("datetime64[D]").astype(np.int64),
        il=sub["local"].map(indice).values.astype(int),
        iv=sub["visitante"].map(indice).values.astype(int),
        gl=sub["gl"].values.astype(float),
        gv=sub["gv"].values.astype(float),
        n_eq=len(equipos),
        indice=indice,
    )


def ajustar_fuerzas(d, corte, vida_media, k_prior, n_iter=N_ITER_MAX, tol=1e-6):
    m = (d.f < corte) & np.isfinite(d.gl) & np.isfinite(d.gv)
    if m.sum() < 30:
        return None
    il, iv, gl, gv = d.il[m], d.iv[m], d.gl[m], d.gv[m]
    w = np.ones(m.sum()) if not vida_media else 0.5 ** ((corte - d.f[m]) / vida_media)
    wgl, wgv = w * gl, w * gv
    n = d.n_eq

    n_juegos = np.bincount(il, minlength=n) + np.bincount(iv, minlength=n)
    activos = n_juegos > 0
    prior = k_prior * (gl.mean() + gv.mean()) / 2

    def bc(idx, vals):
        return np.bincount(idx, weights=vals, minlength=n)

    att, dfn = np.ones(n), np.ones(n)
    for _ in range(n_iter):
        att_ant, dfn_ant = att, dfn
        mu_l = wgl.sum() / (w * att[il] * dfn[iv]).sum()
        mu_v = wgv.sum() / (w * att[iv] * dfn[il]).sum()

        num = bc(il, wgl) + bc(iv, wgv)
        den = bc(il, w * mu_l * dfn[iv]) + bc(iv, w * mu_v * dfn[il])
        att = np.where(den + prior > 0, (num + prior) / (den + prior + 1e-12), 1.0)
        att = att / att[activos].mean()

        num = bc(iv, wgl) + bc(il, wgv)
        den = bc(iv, w * mu_l * att[il]) + bc(il, w * mu_v * att[iv])
        dfn = np.where(den + prior > 0, (num + prior) / (den + prior + 1e-12), 1.0)
        dfn = dfn / dfn[activos].mean()

        if max(np.abs(att - att_ant).max(), np.abs(dfn - dfn_ant).max()) < tol:
            break

    mu_l = wgl.sum() / (w * att[il] * dfn[iv]).sum()
    mu_v = wgv.sum() / (w * att[iv] * dfn[il]).sum()
    return att, dfn, mu_l, mu_v, n_juegos


def _lambdas(att, dfn, mu_l, mu_v, i, j):
    return (np.clip(mu_l * att[i] * dfn[j], 0.05, 6.0),
            np.clip(mu_v * att[j] * dfn[i], 0.05, 6.0))


def lambdas_walkforward(d, idx_eval, vida_media, k_prior):
    """Cada partido usa solo partidos con fecha ANTERIOR. Se ajusta una vez por fecha."""
    lam_l = np.full(len(idx_eval), np.nan)
    lam_v = np.full(len(idx_eval), np.nan)
    fechas = d.f[idx_eval]
    for corte in np.unique(fechas):
        sel = np.where(fechas == corte)[0]
        ajuste = ajustar_fuerzas(d, corte, vida_media, k_prior)
        if ajuste is None:
            continue
        att, dfn, mu_l, mu_v, n_juegos = ajuste
        i, j = d.il[idx_eval[sel]], d.iv[idx_eval[sel]]
        ok = (n_juegos[i] >= MIN_PARTIDOS_EQUIPO) & (n_juegos[j] >= MIN_PARTIDOS_EQUIPO)
        ll, lv = _lambdas(att, dfn, mu_l, mu_v, i, j)
        lam_l[sel[ok]], lam_v[sel[ok]] = ll[ok], lv[ok]
    return lam_l, lam_v


# ============================================================
# 4. PROBABILIDADES DE MERCADO
# ============================================================

_G = MAX_GOLES + 1
_I, _J = np.indices((_G, _G))
_MASK = {
    "1": _I > _J, "X": _I == _J, "2": _I < _J,
    "over": (_I + _J) >= 3, "btts": (_I >= 1) & (_J >= 1),
}
COLS_PROB = ["p1", "px", "p2", "p_over", "p_under", "p_btts_si", "p_btts_no"]


def matriz_goles(lam_l, lam_v, rho=0.0):
    k = np.arange(_G)
    M = poisson.pmf(k[None, :], lam_l[:, None])[:, :, None] * \
        poisson.pmf(k[None, :], lam_v[:, None])[:, None, :]
    if rho:
        M[:, 0, 0] *= 1 - lam_l * lam_v * rho          # corrección Dixon-Coles
        M[:, 0, 1] *= 1 + lam_l * rho
        M[:, 1, 0] *= 1 + lam_v * rho
        M[:, 1, 1] *= 1 - rho
    return M / M.sum(axis=(1, 2), keepdims=True)


def anadir_probs(df, rho=RHO_DEFECTO):
    df = df.copy()
    P = np.full((len(df), len(COLS_PROB)), np.nan)
    ok = np.isfinite(df["lam_l"].to_numpy()) & np.isfinite(df["lam_v"].to_numpy())
    if ok.any():
        M = matriz_goles(df["lam_l"].to_numpy()[ok], df["lam_v"].to_numpy()[ok], rho)
        s = {k: (M * v).sum(axis=(1, 2)) for k, v in _MASK.items()}
        P[ok] = np.column_stack([s["1"], s["X"], s["2"], s["over"], 1 - s["over"],
                                 s["btts"], 1 - s["btts"]])
    for c, col in zip(COLS_PROB, P.T):
        df[c] = col
    return df


def backtest(hist, temporadas_eval=None, vida_media=VIDA_MEDIA_DEFECTO,
             k_prior=K_PRIOR_DEFECTO, rho=RHO_DEFECTO):
    """Predicciones walk-forward de todas las ligas. Rápido: ajusta una vez por fecha."""
    temporadas_eval = temporadas_eval or TEMPORADAS_BACKTEST
    trozos = []
    for cod in LIGAS:
        sub = hist[hist["liga"] == cod].reset_index(drop=True)
        idx_eval = np.where(sub["temporada"].isin(temporadas_eval))[0]
        if sub.empty or len(idx_eval) == 0:
            continue
        d = construir_datos(sub)
        ll, lv = lambdas_walkforward(d, idx_eval, vida_media, k_prior)
        out = sub.iloc[idx_eval].copy()
        out["lam_l"], out["lam_v"] = ll, lv
        trozos.append(out)
    if not trozos:
        return pd.DataFrame()
    bt = pd.concat(trozos, ignore_index=True)
    bt = bt[np.isfinite(bt["lam_l"])].reset_index(drop=True)
    return anadir_probs(bt, rho)


def predecir_fixtures(hist, fx, vida_media=VIDA_MEDIA_DEFECTO, k_prior=K_PRIOR_DEFECTO,
                      rho=RHO_DEFECTO, hoy=None):
    """Predice próximos partidos con TODO el histórico disponible. Devuelve (pred, sin_historial)."""
    hoy = pd.Timestamp(hoy) if hoy is not None else pd.Timestamp.today().normalize()
    filas, sin_hist = [], []
    for cod, g in fx.groupby("liga"):
        sub = hist[hist["liga"] == cod].reset_index(drop=True)
        if sub.empty:
            sin_hist += [f"{cod}: sin histórico"]
            continue
        d = construir_datos(sub)
        corte = max(int(d.f.max()) + 1, int(hoy.to_datetime64().astype("datetime64[D]").astype(np.int64)))
        ajuste = ajustar_fuerzas(d, corte, vida_media, k_prior)
        if ajuste is None:
            sin_hist += [f"{cod}: histórico insuficiente"]
            continue
        att, dfn, mu_l, mu_v, n_juegos = ajuste
        for _, p in g.iterrows():
            i, j = d.indice.get(p["local"]), d.indice.get(p["visitante"])
            if i is None or j is None or n_juegos[i] < MIN_PARTIDOS_EQUIPO or n_juegos[j] < MIN_PARTIDOS_EQUIPO:
                falta = [n for n, ix in ((p["local_n"], i), (p["visitante_n"], j)) if ix is None]
                sin_hist.append(f"{cod}: {p['local_n']} - {p['visitante_n']}"
                                + (f" (nombre no encontrado: {', '.join(falta)})" if falta else " (pocos partidos)"))
                continue
            ll, lv = _lambdas(att, dfn, mu_l, mu_v, np.array([i]), np.array([j]))
            fila = p.to_dict()
            fila["lam_l"], fila["lam_v"] = float(ll[0]), float(lv[0])
            filas.append(fila)
    if not filas:
        return pd.DataFrame(), sin_hist
    return anadir_probs(pd.DataFrame(filas), rho), sin_hist


# ============================================================
# 5. MERCADO, VALOR Y STAKING
# ============================================================

def probs_sin_margen(cuotas):
    """Normalización proporcional (referencia simple; Shin / potencia son más finos)."""
    inv = 1.0 / cuotas
    return inv / inv.sum(axis=1, keepdims=True)


def mezcla_logit(P_modelo, P_ref, alpha):
    """p ∝ p_modelo^alpha * p_ref^(1-alpha): encoge el modelo hacia el mercado."""
    L = alpha * np.log(np.clip(P_modelo, 1e-12, 1)) + (1 - alpha) * np.log(np.clip(P_ref, 1e-12, 1))
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def _mat(df, prefijo, k):
    return np.column_stack([df[f"{prefijo}_{i}"].to_numpy(float) for i in range(k)])


def _cuotas_validas(M):
    return np.isfinite(M).all(axis=1) & (np.nan_to_num(M, nan=0.0) > 1).all(axis=1)


def preparar_mercado(df, nombre, alpha):
    """Matrices del mercado: prob. del modelo, cuotas B365, referencia de mercado y mezcla."""
    cfg = MERCADOS[nombre]
    k = cfg["k"]
    P = df[cfg["p"]].to_numpy(float)
    O = _mat(df, cfg["cuota"], k)          # cuota ejecutable (Bet365)
    A = _mat(df, cfg["avg"], k)            # consenso previo al cierre (si existe)
    C = _mat(df, cfg["cierre"], k)         # cierre (solo como benchmark ex post)
    Cb = _mat(df, cfg["b365c"], k)

    ok_o, ok_a, ok_c = _cuotas_validas(O), _cuotas_validas(A), _cuotas_validas(C)
    ref = np.full_like(P, np.nan)
    if ok_a.any():
        ref[ok_a] = probs_sin_margen(A[ok_a])
    solo_b = ok_o & ~ok_a                  # sin consenso: la referencia es la propia B365 sin margen
    if solo_b.any():
        ref[solo_b] = probs_sin_margen(O[solo_b])
    Pc = np.full_like(P, np.nan)
    if ok_c.any():
        Pc[ok_c] = probs_sin_margen(C[ok_c])

    valido = np.isfinite(P).all(axis=1) & ok_o & np.isfinite(ref).all(axis=1)
    Pf = np.full_like(P, np.nan)
    if valido.any():
        Pf[valido] = mezcla_logit(P[valido], ref[valido], alpha)
    return {"P": P, "O": O, "ref": ref, "Pf": Pf, "Pc": Pc, "Cb": Cb, "valido": valido}


def construir_oportunidades(df, alpha):
    """Mejor selección (mayor EV) de cada mercado y partido. `fila` apunta a df."""
    trozos = []
    for nombre, cfg in MERCADOS.items():
        m = preparar_mercado(df, nombre, alpha)
        filas = np.where(m["valido"])[0]
        if len(filas) == 0:
            continue
        ev = m["Pf"][filas] * m["O"][filas] - 1
        sel = ev.argmax(axis=1)
        r = np.arange(len(filas))
        out = pd.DataFrame({
            "fila": filas, "mercado": nombre, "sel_idx": sel,
            "seleccion": np.array(cfg["selecciones"])[sel],
            "p_modelo": m["P"][filas][r, sel], "p_ref": m["ref"][filas][r, sel],
            "p_final": m["Pf"][filas][r, sel], "cuota": m["O"][filas][r, sel],
            "ev": ev[r, sel], "p_cierre": m["Pc"][filas][r, sel],
            "cuota_cierre": m["Cb"][filas][r, sel],
        })
        out["edge_pts"] = out["p_final"] - 1 / out["cuota"]
        trozos.append(out)
    return pd.concat(trozos, ignore_index=True) if trozos else pd.DataFrame()


def stake_kelly(p, cuota, bankroll, fraccion=0.25, tope=0.02):
    """Kelly fraccionado con tope (fracción del bankroll). 0 si no hay ventaja."""
    if not (np.isfinite(p) and np.isfinite(cuota)) or cuota <= 1:
        return 0.0
    f = (p * cuota - 1) / (cuota - 1)
    if f <= 0:
        return 0.0
    return round(float(bankroll * min(fraccion * f, tope)), 2)


# ============================================================
# 6. EVALUACIÓN DEL BACKTEST
# ============================================================

def _metricas(P, y):
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    n = len(y)
    O = np.eye(P.shape[1])[y]
    ll = -np.log(P[np.arange(n), y])
    return ll, ((P - O) ** 2).sum(axis=1), (P.argmax(axis=1) == y).astype(float)


def bootstrap_diff(a, b, n_boot=3000, seed=42):
    dif = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    medias = dif[rng.integers(0, len(dif), (n_boot, len(dif)))].mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return dif.mean(), lo, hi


def comparar_prediccion(bt):
    """Modelo vs mercado (cierre sin margen) vs uniforme, sobre los mismos partidos."""
    filas, dif = [], []
    for nombre, cfg in MERCADOS.items():
        k = cfg["k"]
        P = bt[cfg["p"]].to_numpy(float)
        C = _mat(bt, cfg["cierre"], k)
        ok = np.isfinite(P).all(axis=1) & _cuotas_validas(C)
        if not ok.any():
            continue
        y = bt[cfg["y"]].to_numpy()[ok].astype(int)
        refs = {"Modelo": P[ok], "Mercado (cierre, sin margen)": probs_sin_margen(C[ok]),
                "Uniforme": np.full((ok.sum(), k), 1.0 / k)}
        por = {}
        for ref_nombre, PP in refs.items():
            ll, br, ac = _metricas(PP, y)
            por[ref_nombre] = ll
            filas.append({"mercado": nombre, "modelo": ref_nombre, "n": int(ok.sum()),
                          "log_loss": ll.mean(), "brier": br.mean(), "accuracy": ac.mean()})
        d, lo, hi = bootstrap_diff(por["Modelo"], por["Mercado (cierre, sin margen)"])
        dif.append({"mercado": nombre, "log_loss modelo - mercado": d, "IC95_lo": lo, "IC95_hi": hi,
                    "distinguible de 0": "SÍ" if (lo > 0 or hi < 0) else "no"})
    # BTTS: no hay cuotas históricas en football-data; solo se compara con la constante 50 %
    P = bt[["p_btts_si", "p_btts_no"]].to_numpy(float)
    ok = np.isfinite(P).all(axis=1)
    if ok.any():
        y = ((bt["gl"] > 0) & (bt["gv"] > 0)).to_numpy()[ok]
        y = np.where(y, 0, 1).astype(int)
        for nom, PP in (("Modelo", P[ok]), ("Uniforme", np.full((ok.sum(), 2), 0.5))):
            ll, br, ac = _metricas(PP, y)
            filas.append({"mercado": "Ambos marcan (sin cuotas)", "modelo": nom, "n": int(ok.sum()),
                          "log_loss": ll.mean(), "brier": br.mean(), "accuracy": ac.mean()})
    return pd.DataFrame(filas), pd.DataFrame(dif)


def tabla_calibracion(P, y, n_bins=8):
    O = np.eye(P.shape[1])[y]
    t = pd.DataFrame({"p": P.ravel(), "o": O.ravel()})
    t["tramo"] = pd.qcut(t["p"], n_bins, duplicates="drop").astype(str)
    g = t.groupby("tramo", observed=True).agg(prob_media=("p", "mean"), frec_obs=("o", "mean"), n=("o", "size"))
    ece = float((g["n"] * (g["prob_media"] - g["frec_obs"]).abs()).sum() / g["n"].sum())
    return g.reset_index(), ece


def evaluar_estrategia(bt, alpha, min_ev, cuota_max, n_sim=3000, seed=42):
    """
    Apuesta plana de 1 unidad a la mejor selección de cada mercado/partido si EV >= min_ev.
    Devuelve (resumen por mercado, curva de beneficio, detalle, drawdown).
    """
    ops = construir_oportunidades(bt, alpha)
    if ops.empty:
        return pd.DataFrame(), pd.DataFrame(), ops, 0.0
    fila = ops["fila"].to_numpy()
    ops["y_mkt"] = np.where(ops["mercado"] == "1X2", bt["y"].to_numpy()[fila], bt["y_ou"].to_numpy()[fila])
    ops["gana"] = ops["sel_idx"] == ops["y_mkt"]
    ops["fecha"] = bt["fecha"].to_numpy()[fila]
    ops["liga"] = bt["liga"].to_numpy()[fila]

    sel = ops[(ops["ev"] >= min_ev) & (ops["cuota"] <= cuota_max)].copy()
    sel["beneficio"] = np.where(sel["gana"], sel["cuota"] - 1.0, -1.0)
    sel["ev_cierre"] = sel["cuota"] * sel["p_cierre"] - 1.0
    sel["clv"] = sel["cuota"] / sel["cuota_cierre"] - 1.0

    rng = np.random.default_rng(seed)
    filas = []
    for nombre in list(MERCADOS) + ["TOTAL"]:
        s = sel if nombre == "TOTAL" else sel[sel["mercado"] == nombre]
        if s.empty:
            continue
        ben = s["beneficio"].to_numpy()
        boot = ben[rng.integers(0, len(ben), (n_sim, len(ben)))].mean(axis=1)
        fila_res = {
            "mercado": nombre, "apuestas": len(s), "acierto_%": 100 * s["gana"].mean(),
            "cuota_media": s["cuota"].mean(), "profit_u": ben.sum(), "ROI_%": 100 * ben.mean(),
            "ROI_IC95_lo_%": 100 * np.percentile(boot, 2.5), "ROI_IC95_hi_%": 100 * np.percentile(boot, 97.5),
            "EV_vs_cierre_%": 100 * np.nanmean(s["ev_cierre"]) if s["ev_cierre"].notna().any() else np.nan,
            "CLV_%": 100 * np.nanmean(s["clv"]) if s["clv"].notna().any() else np.nan,
            "suelo_sin_habilidad_%": np.nan, "p_valor_EV": np.nan,
        }
        if nombre != "TOTAL":
            # suelo 'sin habilidad' y baseline aleatorio (mismo nº de apuestas) contra el cierre
            m = preparar_mercado(bt, nombre, alpha)
            v = m["valido"] & np.isfinite(m["Pc"]).all(axis=1)
            if v.any() and s["ev_cierre"].notna().any():
                O, Pc = m["O"][v], m["Pc"][v]
                fila_res["suelo_sin_habilidad_%"] = 100 * ((O * Pc).mean() - 1)
                i = rng.integers(0, len(O), (n_sim, len(s)))
                kk = rng.integers(0, O.shape[1], (n_sim, len(s)))
                ev_null = (O[i, kk] * Pc[i, kk] - 1).mean(axis=1)
                fila_res["p_valor_EV"] = float((ev_null >= np.nanmean(s["ev_cierre"])).mean())
        filas.append(fila_res)

    curva = sel.sort_values("fecha")[["fecha", "beneficio"]].copy()
    curva["acumulado"] = curva["beneficio"].cumsum()
    dd = float((curva["acumulado"].cummax() - curva["acumulado"]).max()) if len(curva) else 0.0
    return pd.DataFrame(filas), curva.reset_index(drop=True), sel, dd


# ============================================================
# 7. PAPER TRADING
# ============================================================

COLS_APUESTAS = ["id", "fecha_registro", "fecha_partido", "liga", "local", "visitante", "mercado",
                 "seleccion", "cuota", "prob_modelo", "ev_pct", "stake", "estado", "beneficio",
                 "cuota_cierre", "clv_pct"]
_IDX_SEL = {"1X2": {"Local": 0, "Empate": 1, "Visitante": 2},
            "Más/Menos 2.5": {"Más de 2.5": 0, "Menos de 2.5": 1}}


def cargar_bankroll(path, defecto=1000.0):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return float(data.get("bankroll_inicial", data.get("bankroll", defecto)))
    except Exception:                                      # noqa: BLE001
        return float(defecto)


def guardar_bankroll(path, valor):
    Path(path).write_text(json.dumps({"bankroll_inicial": float(valor)}, indent=2), encoding="utf-8")


def cargar_apuestas(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLS_APUESTAS)
    df = pd.read_csv(p)
    for c in COLS_APUESTAS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLS_APUESTAS]


def guardar_apuestas(path, df):
    df.to_csv(path, index=False)


def id_apuesta(fecha, liga, local, visitante, mercado):
    return f"{pd.Timestamp(fecha):%Y%m%d}|{liga}|{local}|{visitante}|{mercado}"


def registrar_apuestas(path, nuevas):
    """Añade apuestas simuladas evitando duplicados (una por partido y mercado)."""
    actuales = cargar_apuestas(path)
    ya = set(actuales["id"].astype(str))
    filas, dup = [], 0
    ahora = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    for _, r in nuevas.iterrows():
        i = id_apuesta(r["fecha"], r["liga"], r["local_n"], r["visitante_n"], r["mercado"])
        if i in ya:
            dup += 1
            continue
        ya.add(i)
        filas.append({"id": i, "fecha_registro": ahora, "fecha_partido": pd.Timestamp(r["fecha"]).strftime("%Y-%m-%d"),
                      "liga": r["liga"], "local": r["local_n"], "visitante": r["visitante_n"],
                      "mercado": r["mercado"], "seleccion": r["seleccion"], "cuota": float(r["cuota"]),
                      "prob_modelo": float(r["p_final"]), "ev_pct": 100 * float(r["ev"]),
                      "stake": float(r["stake"]), "estado": "PENDIENTE", "beneficio": np.nan,
                      "cuota_cierre": np.nan, "clv_pct": np.nan})
    if filas:
        guardar_apuestas(path, pd.concat([actuales, pd.DataFrame(filas)], ignore_index=True))
    return len(filas), dup


def liquidar(bets, hist):
    """Liquida las apuestas PENDIENTES cuyo resultado ya aparece en el histórico."""
    bets = bets.copy()
    pend = bets.index[bets["estado"] == "PENDIENTE"]
    n = 0
    for idx in pend:
        b = bets.loc[idx]
        cand = hist[(hist["liga"] == b["liga"]) & (hist["local_n"] == b["local"])
                    & (hist["visitante_n"] == b["visitante"])]
        cand = cand[(cand["fecha"] - pd.Timestamp(b["fecha_partido"])).abs() <= pd.Timedelta(days=3)]
        if cand.empty:
            continue
        r = cand.iloc[0]
        sel_i = _IDX_SEL.get(b["mercado"], {}).get(b["seleccion"])
        if sel_i is None:
            continue
        y_mkt = int(r["y"]) if b["mercado"] == "1X2" else int(r["y_ou"])
        gana = sel_i == y_mkt
        bets.loc[idx, "estado"] = "GANADA" if gana else "PERDIDA"
        bets.loc[idx, "beneficio"] = b["stake"] * (b["cuota"] - 1) if gana else -b["stake"]
        col = f"b365c_{sel_i}" if b["mercado"] == "1X2" else f"b365cou_{sel_i}"
        cierre = r.get(col, np.nan)
        if pd.notna(cierre) and cierre > 1:
            bets.loc[idx, "cuota_cierre"] = cierre
            bets.loc[idx, "clv_pct"] = 100 * (b["cuota"] / cierre - 1)
        n += 1
    return bets, n


def resumen_apuestas(bets, bankroll_inicial):
    liq = bets[bets["estado"].isin(["GANADA", "PERDIDA"])].copy()
    pend = bets[bets["estado"] == "PENDIENTE"]
    liq["beneficio"] = pd.to_numeric(liq["beneficio"], errors="coerce")
    profit = float(liq["beneficio"].sum()) if len(liq) else 0.0
    apostado = float(liq["stake"].sum()) if len(liq) else 0.0
    curva = pd.DataFrame(columns=["fecha", "bankroll"])
    dd = 0.0
    if len(liq):
        liq = liq.sort_values("fecha_partido")
        cum = liq["beneficio"].cumsum().to_numpy()
        curva = pd.DataFrame({"fecha": pd.to_datetime(liq["fecha_partido"]).to_numpy(),
                              "bankroll": bankroll_inicial + cum})
        dd = float((np.maximum.accumulate(np.r_[0.0, cum])[1:] - cum).max())
    clv = pd.to_numeric(liq["clv_pct"], errors="coerce") if len(liq) else pd.Series(dtype=float)
    return {
        "apuestas_liquidadas": len(liq), "apuestas_pendientes": len(pend),
        "stake_pendiente": float(pend["stake"].sum()) if len(pend) else 0.0,
        "profit": profit, "apostado": apostado,
        "roi_pct": 100 * profit / apostado if apostado > 0 else np.nan,
        "acierto_pct": 100 * float((liq["estado"] == "GANADA").mean()) if len(liq) else np.nan,
        "bankroll_actual": bankroll_inicial + profit, "drawdown": dd,
        "clv_medio_pct": float(clv.mean()) if clv.notna().any() else np.nan,
        "clv_n": int(clv.notna().sum()),
    }, curva
