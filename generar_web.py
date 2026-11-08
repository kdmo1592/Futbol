"""
GENERADOR DE DATOS PARA LA APP WEB
==================================

Lo ejecuta el robot de GitHub (GitHub Actions) automáticamente cada día.
No hace falta que lo ejecutes tú.

Qué hace:
  1. Descarga resultados y cuotas de football-data.co.uk
     (LaLiga, Segunda, Premier, Bundesliga, Serie A, Ligue 1 y Allsvenskan).
  2. Descarga los próximos partidos con sus cuotas.
  3. Calcula las fuerzas de cada equipo (goles, rating de mercado, córners y tarjetas)
     y entrena el modelo de mercado.
  4. Mide cada modelo en temporadas que no ha visto (pestaña Fiabilidad) y hace el
     backtest de la regla de "Ambos marcan".
  5. Escribe data.json, que es lo que lee la app (index.html).
"""

import io
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import estrategias as es
import modelo_mercado as mm
import motor_futbol as mf

# ---------------------------------------------------------------- configuración
VIDA_MEDIA, K_PRIOR, RHO = 365.0, 3.0, -0.10          # modelo de goles
TEMPORADAS_PARTIDOS = 3                                # temporadas que viajan a la app (forma, cara a cara)
TEMPORADAS_ARBITROS = 3                                # temporadas para la media de tarjetas de cada árbitro

# Ligas "extra" de football-data (formato distinto: un archivo con todas las temporadas).
# Solo existen primeras divisiones. Para añadir otra, añade una línea:
#   "NOR": ("Eliteserien (Noruega)", "Norway"),
#   "DNK": ("Superliga (Dinamarca)", "Denmark"),
#   "FIN": ("Veikkausliiga (Finlandia)", "Finland"),
LIGAS_EXTRA = {
    "SWE": ("Allsvenskan (Suecia)", "Sweden"),
}
URL_EXTRA = "https://www.football-data.co.uk/new/{cod}.csv"
URL_FIXTURES_EXTRA = "https://www.football-data.co.uk/new_league_fixtures.csv"

SALIDA = Path(__file__).resolve().parent / "data.json"


def aviso(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- ligas extra
def temporada_codigo(valor):
    """'2025' -> '2526' (año natural) · '2012/2013' -> '1213'."""
    s = str(valor).strip()
    if "/" in s:
        a, b = s.split("/")[:2]
        return f"{int(a) % 100:02d}{int(b) % 100:02d}"
    y = int(float(s))
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def cargar_extra(descargador, cod):
    """Convierte el formato de las ligas extra al formato de las ligas principales."""
    raw = descargador(URL_EXTRA.format(cod=cod))
    ren = {"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG", "Res": "FTR"}
    faltan = [c for c in ("Season", "Date", "Home", "Away", "HG", "AG", "Res") if c not in raw.columns]
    if faltan:
        raise ValueError(f"formato inesperado, faltan columnas {faltan}")
    df = raw.rename(columns=ren).copy()
    df["temporada"] = df["Season"].map(temporada_codigo)
    df = df[df["temporada"].isin(mf.TEMPORADAS)].copy()
    df["liga"] = cod
    return df


def cargar_fixtures_extra(descargador):
    raw = descargador(URL_FIXTURES_EXTRA)
    pais_a_cod = {pais: cod for cod, (_, pais) in LIGAS_EXTRA.items()}
    if "Country" not in raw.columns:
        raise ValueError("formato inesperado (sin columna Country)")
    df = raw[raw["Country"].isin(pais_a_cod)].copy()
    if df.empty:
        return pd.DataFrame()
    df["Div"] = df["Country"].map(pais_a_cod)
    df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
    return mf.normalizar_fixtures(df)


# ---------------------------------------------------------------- carga
def cargar_todo(descargador=mf.descargar_csv):
    for cod, (nombre, _) in LIGAS_EXTRA.items():
        mf.LIGAS[cod] = nombre
    trozos, errores = [], []
    for liga in [c for c in mf.LIGAS if c not in LIGAS_EXTRA]:
        for t in mf.TEMPORADAS:
            try:
                raw = descargador(mf.URL_HISTORICO.format(t=t, liga=liga))
                if "HomeTeam" not in raw.columns:
                    errores.append(f"{liga} {t}: sin columna HomeTeam")
                    continue
                raw = raw.copy()
                raw["liga"], raw["temporada"] = liga, t
                trozos.append(raw)
            except Exception as e:                      # noqa: BLE001
                errores.append(f"{liga} {t}: {e}")
    for cod in LIGAS_EXTRA:
        try:
            trozos.append(cargar_extra(descargador, cod))
        except Exception as e:                          # noqa: BLE001
            errores.append(f"{cod}: {e}")
    if not trozos:
        raise RuntimeError("No se pudo descargar ningún histórico.")
    hist = mf.normalizar_historico(pd.concat(trozos, ignore_index=True))

    fx = []
    try:
        fx.append(mf.normalizar_fixtures(descargador(mf.URL_FIXTURES)))
    except Exception as e:                              # noqa: BLE001
        errores.append(f"próximos partidos: {e}")
    try:
        f2 = cargar_fixtures_extra(descargador)
        if len(f2):
            fx.append(f2)
    except Exception as e:                              # noqa: BLE001
        errores.append(f"próximos partidos ligas extra: {e}")
    fx = pd.concat(fx, ignore_index=True) if fx else pd.DataFrame()
    return hist, fx, errores


# ---------------------------------------------------------------- exportación
def _num(x, dec=4):
    x = float(x)
    return round(x, dec) if np.isfinite(x) else None


def texto(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else str(x)


def _lista(a, dec=4):
    return [_num(v, dec) for v in np.asarray(a, float)]


def fuerzas_json(d, aj):
    if aj is None:
        return None
    att, dfn, mu_l, mu_v, nj = aj
    return {"att": _lista(att), "dfn": _lista(dfn), "mu_l": _num(mu_l), "mu_v": _num(mu_v),
            "n": [int(v) for v in nj]}


def exportar(hist, fx, modelo, errores, hoy, conteos=None, btts=None):
    hoy_d = int(np.datetime64(hoy.to_datetime64(), "D").astype(np.int64))
    ligas, fuerzas, ultimo = [], {}, {}
    for cod in mf.LIGAS:
        sub = hist[hist["liga"] == cod].reset_index(drop=True)
        if sub.empty:
            continue
        d = mf.construir_datos(sub)
        corte = max(int(d.f.max()) + 1, hoy_d)
        aj_g = mf.ajustar_fuerzas(d, corte, VIDA_MEDIA, K_PRIOR)
        aj_m = None
        if modelo is not None:
            d_mkt, _ = modelo.datos_mercado(cod)
            aj_m = mf.ajustar_fuerzas(d_mkt, corte, mm.VIDA_MEDIA_MKT, mm.K_PRIOR_MKT)
        equipos = [None] * d.n_eq
        for k, i in d.indice.items():
            equipos[i] = k.split(":", 1)[1]
        fuerzas[cod] = {"equipos": equipos, "gol": fuerzas_json(d, aj_g), "mkt": fuerzas_json(d, aj_m),
                        "cor": fuerzas_json(d, es.fuerzas_conteo(sub, "corners", corte)),
                        "tar": fuerzas_json(d, es.fuerzas_conteo(sub, "tarjetas", corte))}
        temp = sub["temporada"].max()
        nombre_t = f"20{temp[:2]}" if cod in LIGAS_EXTRA else f"{temp[:2]}/{temp[2:]}"
        ligas.append({"cod": cod, "nombre": mf.LIGAS[cod], "temporada": temp, "temporada_nombre": nombre_t,
                      "equipos_temporada": sorted(set(sub.loc[sub["temporada"] == temp, "local_n"])
                                                  | set(sub.loc[sub["temporada"] == temp, "visitante_n"]))})
        largo = pd.concat([sub[["local_n", "fecha"]].rename(columns={"local_n": "e"}),
                           sub[["visitante_n", "fecha"]].rename(columns={"visitante_n": "e"})])
        ultimo[cod] = {e: f"{f:%Y-%m-%d}" for e, f in largo.groupby("e")["fecha"].max().items()}

    temps = sorted(hist["temporada"].unique())[-TEMPORADAS_PARTIDOS:]
    rec = hist[hist["temporada"].isin(temps)].sort_values("fecha")
    tar_l, tar_v = es.columnas_conteo(rec, "tarjetas")
    rec = rec.assign(tar_l=tar_l, tar_v=tar_v)
    ent = lambda x: None if pd.isna(x) else int(x)
    # [liga, fecha, local, visitante, gl, gv, tiros a puerta l, v, temporada, córners l, v, tarjetas l, v]
    partidos = [[r.liga, f"{r.fecha:%Y-%m-%d}", r.local_n, r.visitante_n, int(r.gl), int(r.gv),
                 ent(r.hst), ent(r.ast), r.temporada, ent(r.hc), ent(r.ac), ent(r.tar_l), ent(r.tar_v)]
                for r in rec.itertuples()]

    # media de tarjetas por árbitro (solo ligas en las que football-data publica el árbitro)
    arbitros = {}
    t_arb = sorted(hist["temporada"].unique())[-TEMPORADAS_ARBITROS:]
    h_arb = hist[hist["temporada"].isin(t_arb) & hist["arbitro"].notna()]
    if len(h_arb):
        tl, tv = es.columnas_conteo(h_arb, "tarjetas")
        h_arb = h_arb.assign(tot=tl + tv).dropna(subset=["tot"])
        for cod, g in h_arb.groupby("liga"):
            s_ = g.groupby("arbitro")["tot"].agg(["size", "mean"])
            arbitros[cod] = {"media_liga": _num(g["tot"].mean(), 2),
                             "arbitros": {a: [int(r["size"]), _num(r["mean"], 2)] for a, r in s_.iterrows()}}

    proximos = []
    if len(fx):
        f = fx[fx["fecha"] >= hoy - pd.Timedelta(days=1)].sort_values(["fecha", "hora"])
        for _, r in f.iterrows():
            d = pd.DataFrame([r])
            p1, pou = mm.probs_mercado(d, cierre=False)
            b1, bou = mm.cuotas(d, "b365", 3)[0], mm.cuotas(d, "b365ou", 2)[0]
            proximos.append({
                "liga": r["liga"], "fecha": f"{r['fecha']:%Y-%m-%d}", "hora": texto(r.get("hora")),
                "local": r["local_n"], "visitante": r["visitante_n"], "arbitro": texto(r.get("arbitro")),
                "mer1x2": _lista(p1[0]) if np.isfinite(p1[0]).all() else None,
                "merou": _lista(pou[0]) if np.isfinite(pou[0]).all() else None,
                "b365": _lista(b1, 2) if np.isfinite(b1).all() else None,
                "b365ou": _lista(bou, 2) if np.isfinite(bou).all() else None,
            })

    mod = None
    if modelo is not None and modelo.modelos:
        mod = {"ligas": modelo.ligas, "principal": modelo.principal,
               "softmax": {k: {"W": np.round(m.W, 6).tolist(), "mu": np.round(m.mu, 6).tolist(),
                               "sd": np.round(m.sd, 6).tolist()} for k, m in modelo.modelos.items()},
               "metricas": [{k: (_num(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
                            for r in modelo.metricas.to_dict("records")],
               "info": modelo.info}

    return {"version": 2, "generado": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "hoy": f"{hoy:%Y-%m-%d}", "datos_hasta": f"{hist['fecha'].max():%Y-%m-%d}",
            "cobertura_proximos": ([f"{fx['fecha'].min():%Y-%m-%d}", f"{fx['fecha'].max():%Y-%m-%d}"]
                                   if len(fx) else None),
            "parametros": {"rho_goles": RHO, "vida_media": VIDA_MEDIA, "k_prior": K_PRIOR,
                           "min_partidos": mf.MIN_PARTIDOS_EQUIPO, "tope_conteo": es.TOPE_CONTEO},
            "ligas": ligas, "fuerzas": fuerzas, "ultimo_partido": ultimo, "modelo": mod,
            "conteos": limpiar(conteos), "estrategias": {"btts": limpiar(btts)}, "arbitros": arbitros,
            "partidos": partidos, "proximos": proximos, "avisos": errores}


def limpiar(x):
    """Convierte numpy/NaN a tipos JSON (NaN -> null)."""
    if isinstance(x, dict):
        return {k: limpiar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [limpiar(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _num(x, 4)
    return x


def main(descargador=mf.descargar_csv, salida=SALIDA, hoy=None):
    t0 = time.time()
    hoy = pd.Timestamp(hoy) if hoy is not None else pd.Timestamp.today().normalize()
    hist, fx, errores = cargar_todo(descargador)
    aviso(f"Histórico: {len(hist):,} partidos · próximos: {len(fx)} · avisos: {len(errores)}")
    for e in errores:
        aviso(f"  aviso: {e}")
    modelo = None
    try:
        modelo = mm.ModeloMercado(hist, (VIDA_MEDIA, K_PRIOR, RHO))
        modelo.entrenar(progreso=aviso)
        aviso(modelo.metricas.round(4).to_string(index=False))
    except Exception as e:                              # noqa: BLE001
        errores.append(f"modelo de mercado no entrenado: {e}")
        traceback.print_exc()
        modelo = None if modelo is None or not modelo.modelos else modelo
    conteos, btts = None, None
    if modelo is not None and modelo.info:
        dev = list(modelo.info["entreno"]) + [modelo.info["validacion"]]
        test = list(modelo.info["test"])
        try:
            conteos = es.evaluar_conteos(hist, dev[-2:], test)
            for k, c in conteos.items():
                aviso(f"{c['nombre']}: dispersión total {c['k_total']}, por equipo {c['k_lado']}")
        except Exception as e:                          # noqa: BLE001
            errores.append(f"córners/tarjetas sin evaluar: {e}")
            traceback.print_exc()
        try:
            btts = es.backtest_btts(modelo.variables, dev, test)
        except Exception as e:                          # noqa: BLE001
            errores.append(f"backtest de Ambos marcan no disponible: {e}")
            traceback.print_exc()
    datos = exportar(hist, fx, modelo, errores, hoy, conteos, btts)
    Path(salida).write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    aviso(f"Escrito {salida} ({Path(salida).stat().st_size / 1024:.0f} KB) en {time.time() - t0:.0f} s")
    return datos


if __name__ == "__main__":
    try:
        main()
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
