import json, os, sys, requests
sys.path.insert(0, ".")
import cuotas_api as ca
K = os.environ["K"]; s = requests.Session()
cli = ca.ClienteCuotas(K, {}, s)
evs = cli.eventos("soccer_spain_la_liga")
print("eventos LaLiga:", len(evs))
ev = evs[0]
print("partido:", ev["home_team"], "-", ev["away_team"], ev["commence_time"][:16])
casas, corners = cli.cuotas("soccer_spain_la_liga", ev["id"], con_corners=True)
print("creditos usados:", cli.usados, "restantes:", cli.restantes)
print("btts:", json.dumps(ca.consenso(casas))[:200] if casas else None)
cc = ca.consenso_corners(corners)
print("corners crudo (1 linea):", json.dumps({k: v for k, v in list((corners or {}).items())[:1]})[:300])
print("corners consenso:", json.dumps(cc, indent=1)[:700] if cc else "NINGUNO")
