# ⚽ App de análisis de fútbol

App web para el iPhone que se actualiza sola cada día. No hay que ejecutar nada.

- **La web** (`index.html`) se abre desde un icono en tu pantalla de inicio.
- **El robot** (GitHub Actions) descarga los datos de football-data.co.uk cada día,
  entrena el modelo y deja el resultado en `data.json`.
- **Tu diario** se guarda en tu iPhone, no en internet.

Ligas: LaLiga, LaLiga 2, Premier League, Bundesliga, Serie A, Ligue 1 y Allsvenskan (Suecia).

## Si algo falla
- Pestaña **Actions** del repositorio: una ejecución en rojo muestra el error. Pulsa en ella para ver el detalle.
- En la app, pestaña **✅ Fiabilidad → Avisos**: descargas que no funcionaron.
- Si GitHub te escribe diciendo que ha desactivado el robot por inactividad: entra en **Actions** y pulsa **Enable workflow**.

## Añadir otra liga nórdica
En `generar_web.py`, dentro de `LIGAS_EXTRA`, añade una línea:
`"NOR": ("Eliteserien (Noruega)", "Norway"),`
(también disponibles `DNK` Dinamarca y `FIN` Finlandia; solo existen primeras divisiones).
