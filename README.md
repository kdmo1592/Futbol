# ⚽ App de análisis de fútbol

App web para el iPhone que se actualiza sola cada día. No hay que ejecutar nada.
Web: https://kdmo1592.github.io/Futbol/

- **La web** (`index.html`) se abre desde un icono en la pantalla de inicio
  (Safari → Compartir → Añadir a pantalla de inicio). Ábrela siempre desde el icono:
  dentro de Safari, el navegador puede borrar el diario si pasas 7 días sin entrar.
- **El robot** (GitHub Actions) descarga los datos de football-data.co.uk cada día,
  entrena los modelos, comprueba que la app funciona con ellos y deja el resultado en `data.json`.
- **Tu diario y tus apuestas** se guardan en tu iPhone, no en internet. Haz copia con
  📝 Diario → Exportar de vez en cuando.

Ligas: LaLiga, LaLiga 2, Premier League, Bundesliga, Serie A, Ligue 1 y Allsvenskan (Suecia).

## Pestañas

| Pestaña | Para qué |
|---|---|
| 📅 Jornada | Próximos partidos con el modelo y el mercado. Δ grande = partido para estudiar. |
| 🔍 Partido | Probabilidades de todos los mercados, marcadores, córners, tarjetas, forma y cara a cara. |
| 📊 Variables | Todo para decidir un partido: forma reciente frente a su media, dificultad de los últimos rivales, rendimiento frente a lo esperado, descanso, mercado, Ambos marcan (con tu regla del 62 %), córners, tarjetas y árbitro. Desde aquí registras apuestas con la cuota real. |
| 🎛️ Lab | Mueve los goles esperados a mano y guarda tu pronóstico. |
| 🏆 Liga | Clasificación y mapa de fuerzas. |
| 📝 Diario | Tus pronósticos (¿predices mejor que el modelo y el mercado?) y tus apuestas (beneficio y ROI). |
| ✅ Fiable | Cuánto se acerca cada modelo al mercado, si el modelo de córners y tarjetas aporta algo y el backtest de Ambos marcan. |

## Archivos

| Archivo | Qué es |
|---|---|
| `index.html` | La app (motor de cálculo + interfaz) |
| `generar_web.py` | Lo ejecuta el robot: descarga, entrena y escribe `data.json` |
| `motor_futbol.py` | Descarga de datos y modelo de goles (Poisson + Dixon-Coles) |
| `modelo_mercado.py` | Rating de mercado y modelo de mercado; mide todo en temporadas de prueba |
| `estrategias.py` | Córners, tarjetas y backtest de Ambos marcan |
| `comprobar.js` | Prueba la app con los datos nuevos antes de publicarlos |
| `data.json` | Datos que lee la app (lo genera el robot; no tocar) |

## Cómo se mide (para no engañarse)

- Ningún modelo usa información posterior al partido que predice.
- Las dos primeras temporadas solo "calientan"; se entrena con las siguientes, se elige la
  configuración con una temporada de validación y se mide en las **dos últimas temporadas**, que no se usan para entrenar.
- Todas las fuentes se comparan con el mercado de cierre **en los mismos partidos**, con intervalo de confianza.
- La estimación principal de la app es la que mejor sale en esa prueba; la elige el robot cada día.
- football-data no tiene cuotas de Ambos marcan, córners ni tarjetas. Para esos mercados
  solo se puede medir el acierto y la cuota mínima necesaria. El beneficio real sale de tus apuestas registradas.

## Si algo falla

- Pestaña **Actions** del repositorio: una ejecución en rojo muestra el error. Pulsa en ella para ver el detalle.
  Si falla el paso "Comprobar…", el robot no publica los datos y la app sigue con los del día anterior.
- En la app, la cabecera avisa si los datos llevan 2 días o más sin actualizarse.
- En la app, pestaña **✅ Fiable → Avisos**: descargas que no funcionaron.
- Si GitHub te escribe diciendo que ha desactivado el robot por inactividad: entra en **Actions** y pulsa **Enable workflow**.

## Añadir otra liga nórdica

En `generar_web.py`, dentro de `LIGAS_EXTRA`, añade una línea:
`"NOR": ("Eliteserien (Noruega)", "Norway"),`
(también hay `DNK` Dinamarca y `FIN` Finlandia; solo primeras divisiones y sin córners ni tarjetas).
