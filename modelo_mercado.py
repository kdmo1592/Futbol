"""
MODELO DE MERCADO
=================

Objetivo: acercarse a las probabilidades que maneja el mercado usando SOLO información
anterior a cada partido (nunca la cuota del propio partido).

Tres piezas:

  1. INVERSOR: convierte unas cuotas 1X2 (+ Más/Menos 2.5) en los goles esperados que
     las explican. Sirve para leer el mercado "en goles" y para la pieza 2.

  2. RATING DE MERCADO: ajusta las fuerzas de cada equipo con los goles esperados que el
     mercado asignó a sus partidos ANTERIORES (cierre), en vez de con los goles reales.
     El mercado ya filtró la suerte, así que es mucho menos ruidoso que los goles.

  3. MODELO DE MERCADO: una regresión softmax entrenada con temporadas pasadas para
     predecir la PROBABILIDAD DE CIERRE a partir del modelo de goles + el rating de
     mercado + liga y descanso. Es un "mercado sintético": lo que diría la línea de
     cierre si solo mirara la fuerza de los equipos.

Referencia (backtest V5, 5 ligas, 3 temporadas de test, log loss del 1X2; menor = mejor):
     mercado de cierre 0.964 · modelo tipo-mercado 0.966 · apertura 0.967
     rating de mercado 0.980 · modelo de goles 0.984
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import motor_futbol as mf

VIDA_MEDIA_MKT, K_PRIOR_MKT = 150.0, 1.0        # el mercado es preciso: memoria más corta
L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
MIN_TEMPORADAS_ENTRENO = 2


# ============================================================
# 1. INVERSOR DE CUOTAS -> GOLES ESPERADOS
# ============================================================

class Inversor:
    """Busca el par (goles esperados local, visitante) que reproduce unas probabilidades."""

    def __init__(self, paso=0.02, maximo=4.5):
        lam = np.arange(0.15, maximo + paso, paso)
        LL, LV = (a.ravel() for a in np.meshgrid(lam, lam, indexing="ij"))
        M = mf.matriz_goles(LL, LV, 0.0)
        I, J = np.indices(M.shape[1:])
        p1 = (M * (I > J)).sum(axis=(1, 2))
        p2 = (M * (I < J)).sum(axis=(1, 2))
        pov = (M * ((I + J) >= 3)).sum(axis=(1, 2))
        self.lam = np.column_stack([LL, LV])
        self.arbol_ou = cKDTree(np.column_stack([p1 - p2, pov]))
        self.arbol_12 = cKDTree(np.column_stack([p1, p2]))

    def invertir(self, P1x2, Pou=None):
        P1x2 = np.atleast_2d(P1x2)
        out = np.full((len(P1x2), 2), np.nan)
        ok12 = np.isfinite(P1x2).all(1)
        if Pou is None:
            Pou = np.full((len(P1x2), 2), np.nan)
        Pou = np.atleast_2d(Pou)
        con = ok12 & np.isfinite(Pou).all(1)
        sin = ok12 & ~con
        if con.any():
            out[con] = self.lam[self.arbol_ou.query(
                np.column_stack([P1x2[con, 0] - P1x2[con, 2], Pou[con, 0]]))[1]]
        if sin.any():
            out[sin] = self.lam[self.arbol_12.query(P1x2[sin][:, [0, 2]])[1]]
        return out


def justas(O):
    P = np.full(np.shape(O), np.nan)
    O = np.atleast_2d(O)
    ok = np.isfinite(O).all(axis=1) & (np.nan_to_num(O, nan=0) > 1).all(axis=1)
    if ok.any():
        inv = 1.0 / O[ok]
        P = np.full(O.shape, np.nan)
        P[ok] = inv / inv.sum(axis=1, keepdims=True)
    return P


def cuotas(df, grupo, K):
    return np.column_stack([pd.to_numeric(df[f"{grupo}_{k}"], errors="coerce").to_numpy(float)
                            if f"{grupo}_{k}" in df.columns else np.full(len(df), np.nan) for k in range(K)])


def preferir(*matrices):
    out = np.full(matrices[0].shape, np.nan)
    for M in matrices:
        hueco = ~np.isfinite(out).all(axis=1) & np.isfinite(M).all(axis=1)
        out[hueco] = M[hueco]
    return out


def probs_mercado(df, cierre=True):
    """Probabilidades sin margen del mercado: Pinnacle si está, si no la media de casas."""
    if cierre:
        p1 = preferir(justas(cuotas(df, "psc", 3)), justas(cuotas(df, "cierre", 3)))
        pou = preferir(justas(cuotas(df, "pscou", 2)), justas(cuotas(df, "cierreou", 2)))
    else:
        p1 = preferir(justas(cuotas(df, "ps", 3)), justas(cuotas(df, "avg", 3)), justas(cuotas(df, "b365", 3)))
        pou = preferir(justas(cuotas(df, "psou", 2)), justas(cuotas(df, "avgou", 2)), justas(cuotas(df, "b365ou", 2)))
    return p1, pou


# ============================================================
# 2. SOFTMAX CON L2 Y ETIQUETAS SUAVES
# ============================================================

class Softmax:
    def fit(self, X, T, l2):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Xb = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        n, d, K = len(X), Xb.shape[1], T.shape[1]
        pen = np.ones((d, 1))
        pen[-1] = 0.0

        def f(theta):
            W = theta.reshape(d, K)
            Z = Xb @ W
            Z -= Z.max(1, keepdims=True)
            logP = Z - np.log(np.exp(Z).sum(1, keepdims=True))
            loss = -(T * logP).sum() / n + l2 * (pen * W ** 2).sum()
            G = Xb.T @ (np.exp(logP) - T) / n + 2 * l2 * pen * W
            return loss, G.ravel()

        r = minimize(f, np.zeros(d * K), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.W = r.x.reshape(d, K)
        return self

    def predict(self, X):
        Xb = np.hstack([(np.atleast_2d(X) - self.mu) / self.sd, np.ones((len(np.atleast_2d(X)), 1))])
        Z = Xb @ self.W
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def a_dict(self):
        return {"W": self.W.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def de_dict(cls, d):
        m = cls()
        m.W, m.mu, m.sd = np.array(d["W"]), np.array(d["mu"]), np.array(d["sd"])
        return m


def _log(P):
    return np.log(np.clip(P, 1e-6, 1.0))


def entropia(T, P):
    return -(T * _log(P)).sum(axis=1)


# ============================================================
# 3. MODELO DE MERCADO
# ============================================================

class ModeloMercado:
    """Entrena un 'mercado sintético' y calcula los ratings de mercado por liga."""

    def __init__(self, hist, params_goles):
        self.hist = hist
        self.params_goles = params_goles           # (vida_media, k_prior, rho) del modelo de goles
        self.inv = Inversor()
        self.ligas = [c for c in mf.LIGAS if (hist["liga"] == c).any()]
        self.temporadas = sorted(hist["temporada"].unique())
        self._datos_mkt = {}
        self.modelos = {}
        self.metricas = None
        self.info = {}

    # ---- goles esperados implícitos del cierre de cada partido del histórico
    def implicitas(self):
        if "imp_l" not in self.hist.columns:
            p1, pou = probs_mercado(self.hist, cierre=True)
            lam = self.inv.invertir(p1, pou)
            self.hist["imp_l"], self.hist["imp_v"] = lam[:, 0], lam[:, 1]
        return self.hist

    def datos_mercado(self, liga):
        """Estructura de partidos de la liga con los goles implícitos del mercado."""
        if liga not in self._datos_mkt:
            self.implicitas()
            sub = self.hist[self.hist["liga"] == liga].reset_index(drop=True)
            d = mf.construir_datos(sub)
            self._datos_mkt[liga] = (mf.Datos(f=d.f, il=d.il, iv=d.iv, gl=sub["imp_l"].to_numpy(float),
                                              gv=sub["imp_v"].to_numpy(float), n_eq=d.n_eq, indice=d.indice), sub)
        return self._datos_mkt[liga]

    # ---- variables de cada partido (walk-forward)
    def construir_variables(self, temporadas, progreso=None):
        vm_g, k_g, rho_g = self.params_goles
        trozos = []
        for n, liga in enumerate(self.ligas, 1):
            if progreso:
                progreso(f"  Calculando variables de {mf.LIGAS[liga]} ({n}/{len(self.ligas)})…")
            d_mkt, sub = self.datos_mercado(liga)
            idx = np.where(sub["temporada"].isin(temporadas))[0]
            if len(idx) == 0:
                continue
            d_gol = mf.construir_datos(sub)
            gl, gv = mf.lambdas_walkforward(d_gol, idx, vm_g, k_g)
            ml, mv = mf.lambdas_walkforward(d_mkt, idx, VIDA_MEDIA_MKT, K_PRIOR_MKT)
            t = sub.iloc[idx].copy()
            t["gol_l"], t["gol_v"], t["rat_l"], t["rat_v"] = gl, gv, ml, mv
            trozos.append(t)
        v = pd.concat(trozos, ignore_index=True)
        pg = mf.matriz_goles(v["gol_l"].to_numpy(float), v["gol_v"].to_numpy(float), rho_g)
        pr = mf.matriz_goles(v["rat_l"].to_numpy(float), v["rat_v"].to_numpy(float), 0.0)
        for nombre, M in (("gol", pg), ("rat", pr)):
            p = self.desglosar(M)
            for k in range(3):
                v[f"{nombre}_1x2_{k}"] = p["1x2"][:, k]
            v[f"{nombre}_ou_0"], v[f"{nombre}_ou_1"] = p["ou"][:, 0], p["ou"][:, 1]
        return v

    @staticmethod
    def desglosar(M):
        I, J = np.indices(M.shape[1:])
        p1 = (M * (I > J)).sum(axis=(1, 2))
        px = (M * (I == J)).sum(axis=(1, 2))
        p2 = (M * (I < J)).sum(axis=(1, 2))
        pov = (M * ((I + J) >= 3)).sum(axis=(1, 2))
        return {"1x2": np.column_stack([p1, px, p2]), "ou": np.column_stack([pov, 1 - pov])}

    def variables_fila(self, liga, p_gol, p_rat, desc_l, desc_v, tag):
        """Vector de variables de un partido futuro (mismo orden que en el entrenamiento)."""
        ligas = np.array([1.0 if liga == c else 0.0 for c in self.ligas])
        return np.hstack([_log(np.atleast_2d(p_gol))[0], _log(np.atleast_2d(p_rat))[0],
                          [np.log(max(desc_l, 1)), np.log(max(desc_v, 1))], ligas])

    def _X(self, v, tag):
        K = 3 if tag == "1x2" else 2
        ligas = np.column_stack([(v["liga"] == c).to_numpy(float) for c in self.ligas])
        return np.hstack([_log(v[[f"gol_{tag}_{k}" for k in range(K)]].to_numpy(float)),
                          _log(v[[f"rat_{tag}_{k}" for k in range(K)]].to_numpy(float)),
                          np.log(np.clip(v[["desc_l", "desc_v"]].to_numpy(float), 1, None)), ligas])

    # ---- entrenamiento
    def entrenar(self, progreso=None):
        temporadas = self.temporadas[2:]                    # las dos primeras calientan los ratings
        if len(temporadas) < MIN_TEMPORADAS_ENTRENO + 1:
            raise ValueError("Hacen falta al menos 5 temporadas de histórico.")
        if progreso:
            progreso(f"Preparando variables de {len(temporadas)} temporadas (1-2 minutos)…")
        v = self.construir_variables(temporadas, progreso)
        test = temporadas[-1]
        val = temporadas[-2]
        entreno = temporadas[:-2]
        p1c, pouc = probs_mercado(v, cierre=True)
        p1a, poua = probs_mercado(v, cierre=False)
        metricas = []
        for tag, K, T_all, A_all in (("1x2", 3, p1c, p1a), ("ou", 2, pouc, poua)):
            X = self._X(v, tag)
            ok = np.isfinite(X).all(1) & np.isfinite(T_all).all(1)
            temp = v["temporada"].to_numpy()
            tr, va, te = (ok & np.isin(temp, entreno)), (ok & (temp == val)), (ok & (temp == test))
            if tr.sum() < 500 or va.sum() < 100:
                raise ValueError("Histórico insuficiente para entrenar el modelo de mercado.")
            mejor, mejor_l2 = np.inf, L2_GRID[-1]
            for l2 in L2_GRID:
                m = Softmax().fit(X[tr], T_all[tr], l2)
                p = entropia(T_all[va], m.predict(X[va])).mean()
                if p < mejor:
                    mejor, mejor_l2 = p, l2
            modelo = Softmax().fit(X[tr | va], T_all[tr | va], mejor_l2)
            self.modelos[tag] = modelo
            # ¿cuánto nos acercamos al mercado en la temporada de test?
            y = (v["y"] if tag == "1x2" else v["y_ou"]).to_numpy()[te].astype(int)
            O = np.eye(K)[y]
            fuentes = {"Mercado (cierre)": T_all[te], "Mercado (apertura)": A_all[te],
                       "Modelo de mercado": modelo.predict(X[te]),
                       "Rating de mercado": v[[f"rat_{tag}_{k}" for k in range(K)]].to_numpy(float)[te],
                       "Modelo de goles": v[[f"gol_{tag}_{k}" for k in range(K)]].to_numpy(float)[te],
                       "Uniforme": np.full((int(te.sum()), K), 1.0 / K)}
            for nombre, P in fuentes.items():
                m_ok = np.isfinite(P).all(1)
                if m_ok.sum() == 0:
                    continue
                metricas.append({"mercado": "1X2" if tag == "1x2" else "Más/Menos 2.5", "fuente": nombre,
                                 "n": int(m_ok.sum()), "log_loss": entropia(O[m_ok], P[m_ok]).mean(),
                                 "brier": ((P[m_ok] - O[m_ok]) ** 2).sum(1).mean(),
                                 "acierto": (P[m_ok].argmax(1) == y[m_ok]).mean(),
                                 "l2": mejor_l2 if nombre == "Modelo de mercado" else np.nan})
        self.metricas = pd.DataFrame(metricas)
        self.info = {"entreno": entreno, "validacion": val, "test": test,
                     "fecha_datos": str(self.hist["fecha"].max().date()), "partidos": int(len(self.hist))}
        return self.metricas

    def predecir(self, liga, p_gol, p_rat, desc_l, desc_v, tag):
        if tag not in self.modelos:
            return None
        x = self.variables_fila(liga, p_gol, p_rat, desc_l, desc_v, tag)
        return self.modelos[tag].predict(x[None, :])[0]

    # ---- guardar / cargar
    def guardar(self, path):
        Path(path).write_text(json.dumps({
            "modelos": {k: m.a_dict() for k, m in self.modelos.items()},
            "ligas": self.ligas, "info": self.info,
            "metricas": self.metricas.to_dict("records") if self.metricas is not None else [],
        }), encoding="utf-8")

    def cargar(self, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("ligas") != self.ligas:
            return False
        self.modelos = {k: Softmax.de_dict(v) for k, v in d["modelos"].items()}
        self.info = d.get("info", {})
        self.metricas = pd.DataFrame(d.get("metricas", []))
        return bool(self.modelos)
