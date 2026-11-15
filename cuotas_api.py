"""
CUOTAS DE "AMBOS MARCAN" (The Odds API)
=======================================

football-data.co.uk no publica cuotas de Ambos marcan, así que sin ellas solo se puede
calcular la cuota mínima teórica. Este módulo las trae de the-odds-api.com, que en el plan
gratuito da 500 créditos al mes.

Costes medidos (1 crédito = 1 mercado x 1 región):
  - lista de partidos de una liga ..... 0 créditos (gratis)
  - cuotas de un partido (btts, 1 reg.) 1 crédito

Por eso se pide la lista gratis, se empareja con los partidos de football-data y solo se
gasta un crédito por partido. Cada partido se consulta una vez; se repite solo si la cuota
guardada es vieja y el partido está cerca. Si quedan pocos créditos, el robot para solo.

La API NO tiene mercados de córners ni de tarjetas para estas ligas: eso se queda como está.
"""

from __future__ import annotations

import json
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import requests

BASE = "https://api.the-odds-api.com/v4"
REGIONES = "uk"                      # donde más casas publican Ambos marcan
MERCADO = "btts"
RESERVA = 60                         # nunca bajar de aquí: deja margen para el resto del mes
MAX_POR_EJECUCION = 40               # tope de partidos nuevos por ejecución
HORAS_REFRESCO = 20                  # repetir una cuota guardada solo si es más vieja que esto
HORAS_CERCA = 36                     # ...y el partido empieza dentro de este plazo
DIAS_VISTA = 6                       # partidos a cubrir hacia delante

LIGAS_API = {
    "SP1": "soccer_spain_la_liga",
    "SP2": "soccer_spain_segunda_division",
    "E0": "soccer_epl",
    "D1": "soccer_germany_bundesliga",
    "I1": "soccer_italy_serie_a",
    "F1": "soccer_france_ligue_one",
    "SWE": "soccer_sweden_allsvenskan",
}

# palabras que sobran al comparar nombres de equipos entre las dos fuentes
RUIDO = {"fc", "cf", "ad", "cd", "ud", "sd", "sc", "ac", "as", "ssc", "afc", "bk", "if", "ff",
         "club", "calcio", "futbol", "football", "balompie", "deportivo", "de", "the", "1899",
         "1900", "04", "05", "96", "united", "utd"}
ALIAS = {                             # casos que la similitud no resuelve
    "ath madrid": "atletico madrid", "ath bilbao": "athletic bilbao", "sp gijon": "sporting gijon",
    "espanol": "espanyol", "la coruna": "deportivo la coruna", "vallecano": "rayo vallecano",
    "betis": "real betis", "sociedad": "real sociedad", "celta": "celta vigo",
    "man city": "manchester city", "man united": "manchester united", "newcastle": "newcastle united",
    "nottm forest": "nottingham forest", "wolves": "wolverhampton wanderers", "tottenham": "tottenham hotspur",
    "west ham": "west ham united", "leeds": "leeds united", "brighton": "brighton and hove albion",
    "ein frankfurt": "eintracht frankfurt", "m gladbach": "borussia monchengladbach",
    "celta b": "celta fortuna", "bilbao ath": "bilbao athletic", "osasuna b": "osasuna promesas",
    "dortmund": "borussia dortmund", "leverkusen": "bayer leverkusen", "bayern munich": "bayern munich",
    "fc koln": "koln", "st pauli": "st pauli", "paris sg": "paris saint germain",
    "inter": "inter milan", "milan": "ac milan", "verona": "hellas verona", "roma": "as roma",
}


def normalizar(nombre):
    s = unicodedata.normalize("NFKD", str(nombre).lower()).encode("ascii", "ignore").decode()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    s = " ".join(s.split())
    s = ALIAS.get(s, s)
    palabras = [p for p in s.split() if p not in RUIDO]
    return " ".join(palabras) if palabras else s


def parecido(a, b):
    na, nb = normalizar(a), normalizar(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    base = SequenceMatcher(None, na, nb).ratio()
    # un nombre contenido en el otro ("girona" dentro de "girona fc") también vale
    if na in nb or nb in na:
        base = max(base, 0.92)
    return base


class ClienteCuotas:
    """Cliente con control de presupuesto. Nunca lanza: devuelve lo que puede y anota avisos."""

    def __init__(self, clave, estado=None, sesion=None):
        self.clave = clave
        self.s = sesion or requests.Session()
        self.restantes = None
        self.usados = 0
        self.avisos = []
        self.estado = estado if isinstance(estado, dict) else {}

    def _get(self, ruta, **params):
        params["apiKey"] = self.clave
        r = self.s.get(f"{BASE}/{ruta}", params=params, timeout=30)
        rest = r.headers.get("x-requests-remaining")
        if rest is not None:
            self.restantes = int(float(rest))
        self.usados += int(float(r.headers.get("x-requests-last", 0) or 0))
        if r.status_code != 200:
            self.avisos.append(f"cuotas {ruta}: HTTP {r.status_code} {r.text[:120]}")
            return None
        return r.json()

    def hay_presupuesto(self, coste=1):
        return self.restantes is None or self.restantes - coste >= RESERVA

    def eventos(self, liga_api):
        """Lista de partidos de una liga. No gasta créditos."""
        d = self._get(f"sports/{liga_api}/events")
        return d if isinstance(d, list) else []

    def btts(self, liga_api, evento_id):
        d = self._get(f"sports/{liga_api}/events/{evento_id}/odds",
                      regions=REGIONES, markets=MERCADO, oddsFormat="decimal")
        if not isinstance(d, dict):
            return None
        casas = []
        for b in d.get("bookmakers", []):
            for m in b.get("markets", []):
                if m.get("key") != MERCADO:
                    continue
                precios = {o.get("name"): o.get("price") for o in m.get("outcomes", [])}
                si, no = precios.get("Yes"), precios.get("No")
                if si and no and si > 1 and no > 1:
                    casas.append({"casa": b.get("title") or b.get("key"), "si": float(si), "no": float(no)})
        return casas or None


def consenso(casas):
    """Probabilidad sin margen (media de las casas) y mejor precio disponible para 'Sí' y 'No'."""
    if not casas:
        return None
    ps = []
    for c in casas:
        inv_si, inv_no = 1 / c["si"], 1 / c["no"]
        ps.append(inv_si / (inv_si + inv_no))
    p = sum(ps) / len(ps)
    mejor_si = max(casas, key=lambda c: c["si"])
    mejor_no = max(casas, key=lambda c: c["no"])
    margen = sum(1 / c["si"] + 1 / c["no"] for c in casas) / len(casas) - 1
    return {"p": round(p, 4), "n": len(casas), "margen": round(margen, 4),
            "mejor_si": [mejor_si["casa"], round(mejor_si["si"], 2)],
            "mejor_no": [mejor_no["casa"], round(mejor_no["no"], 2)],
            "casas": sorted(({"casa": c["casa"], "si": round(c["si"], 2), "no": round(c["no"], 2)}
                             for c in casas), key=lambda c: -c["si"])}


def clave_partido(liga, fecha, local, visitante):
    return f"{liga}|{fecha}|{local}|{visitante}"


def emparejar(eventos, fixtures_liga, umbral=0.78):
    """Empareja partidos de la API con los de football-data por fecha (±1 día) y nombre."""
    pares, usados = [], set()
    for fx in fixtures_liga:
        mejor, mejor_s = None, 0.0
        for k, ev in enumerate(eventos):
            if k in usados:
                continue
            f_ev = str(ev.get("commence_time", ""))[:10]
            if not f_ev or abs((_fecha(f_ev) - _fecha(fx["fecha"])).days) > 1:
                continue
            s = min(parecido(fx["local"], ev.get("home_team")), parecido(fx["visitante"], ev.get("away_team")))
            if s > mejor_s:
                mejor, mejor_s = k, s
        if mejor is not None and mejor_s >= umbral:
            usados.add(mejor)
            pares.append((fx, eventos[mejor], round(mejor_s, 3)))
    return pares


def _fecha(s):
    import datetime
    return datetime.date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def _ahora():
    return time.time()


def recoger(clave, proximos, estado, hoy, log=print):
    """
    proximos: lista de dicts del data.json (liga, fecha, local, visitante).
    estado: dict guardado entre ejecuciones {clave_partido: {ts, cuotas}}.
    Devuelve (btts_por_partido, estado_nuevo, avisos, resumen).
    """
    cli = ClienteCuotas(clave, estado)
    salida, nuevo = {}, dict(estado or {})
    limite = _fecha(hoy)
    por_liga = {}
    for p in proximos:
        if p["liga"] in LIGAS_API and 0 <= (_fecha(p["fecha"]) - limite).days <= DIAS_VISTA:
            por_liga.setdefault(p["liga"], []).append(p)
    pedidos = 0
    for liga, fixtures in sorted(por_liga.items()):
        if pedidos >= MAX_POR_EJECUCION or not cli.hay_presupuesto():
            break
        liga_api = LIGAS_API[liga]
        evs = cli.eventos(liga_api)                      # gratis
        if not evs:
            continue
        pares = emparejar(evs, fixtures)
        log(f"  {liga}: {len(fixtures)} partidos, {len(evs)} en la API, {len(pares)} emparejados")
        for fx, ev, _s in pares:
            k = clave_partido(fx["liga"], fx["fecha"], fx["local"], fx["visitante"])
            guardado = nuevo.get(k)
            if guardado and guardado.get("cuotas"):
                horas = (_ahora() - guardado.get("ts", 0)) / 3600
                faltan = (_fecha(fx["fecha"]) - limite).days * 24
                if horas < HORAS_REFRESCO or faltan > HORAS_CERCA:
                    salida[k] = guardado["cuotas"]
                    continue
            if pedidos >= MAX_POR_EJECUCION or not cli.hay_presupuesto():
                break
            casas = cli.btts(liga_api, ev.get("id"))
            pedidos += 1
            c = consenso(casas)
            if c:
                c["actualizado"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
                salida[k] = c
                nuevo[k] = {"ts": _ahora(), "cuotas": c}
    # limpieza: olvidar partidos de hace más de 60 días
    corte = _ahora() - 60 * 86400
    nuevo = {k: v for k, v in nuevo.items() if v.get("ts", 0) > corte}
    resumen = {"consultas": pedidos, "creditos": cli.usados, "restantes": cli.restantes,
               "partidos_con_cuota": len(salida)}
    return salida, nuevo, cli.avisos, resumen


def cargar_estado(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}


def guardar_estado(path, estado):
    Path(path).write_text(json.dumps(estado, separators=(",", ":")), encoding="utf-8")
