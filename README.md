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
| 📅 Jornada | Próximos partidos y las selecciones más probables. Toca un partido para abrir su informe. |
| 📋 Informe | El asesor de cada partido: resumen, probabilidad y **cuota mínima** de cada mercado (resultado, goles, ambos marcan, córners, tarjetas), factores, lista de comprobaciones y tu apuesta (simple o combinada) con importe recomendado según tu presupuesto. |
| 🔍 Detalle | Nuestras estimaciones comparadas entre sí y con el mercado, marcadores, forma y cara a cara. |
| 🎛️ Lab | Mueve los goles esperados a mano (por ejemplo, por una baja) y mira cómo cambian las cuotas justas. |
| 🏆 Liga | Clasificación y mapa de fuerzas. |
| 📝 Diario | Presupuesto, apuestas (se liquidan solas; beneficio y ROI) y pronósticos. |
| ✅ Fiable | Las pruebas: qué estimación predice mejor, de dónde salen las probabilidades del informe, si la forma o el descanso añaden algo, córners, tarjetas y el backtest de Ambos marcan. |

## Cómo decide el informe

- Si el partido tiene cuotas (jornada en curso), la base son esas cuotas sin el margen de la casa, convertidas en goles
  esperados. De ahí salen todos los mercados. En las temporadas de prueba es lo que mejor predice, por delante de cualquier modelo propio.
- Si no tiene cuotas, usa el mejor modelo propio y sube el margen de seguridad.
- "Pide ≥" es la cuota justa más un margen de seguridad (mayor en los mercados menos predecibles). Por debajo, no hay valor.
- La forma, el descanso y los tiros a puerta se muestran como contexto: el robot comprueba cada día si mejoran la
  predicción y, hasta ahora, no lo hacen. Lo que el mercado puede no haber recogido aún son las noticias: bajas y alineaciones.
- **Ambos marcan tiene precio real**: football-data no publica ese mercado, así que el robot lo pide a
  the-odds-api.com (plan gratuito, 500 consultas al mes; la clave está en el secreto `ODDSPAPI_KEY` del repositorio).
  La lista de partidos es gratis y cada partido cuesta una sola consulta, así que sobra presupuesto.
  Cuando hay precio real, el informe usa el consenso del mercado en vez del modelo y enseña la mejor cuota disponible.
  En LaLiga y Segunda también hay **cuotas reales de córners** (`alternate_totals_corners`); donde las hay, el
  informe usa esas líneas y precios en vez del modelo. No existe mercado de tarjetas en ninguna liga.
- Cada cuota recogida se archiva en `historico_btts.json` junto con el resultado del partido. Con el tiempo eso
  permite medir de verdad si apostar a Ambos marcan sale rentable (pestaña ✅ Fiable).
- **Tarjetas**: se corrigen por el contexto del partido (un partido igualado tiene más tarjetas que uno con
  favorito claro: ×1.06 frente a ×0.85). Es el único añadido que mejoró de verdad en las temporadas de prueba.
  Probé también faltas, tiros y el árbitro: ninguno mejoró de forma medible, así que no están en el modelo
  (el árbitro se sigue enseñando como contexto).
- **Córners**: probé faltas, tiros y contexto del partido; ninguno mejora al modelo actual. Por eso lo que vale
  de verdad son las cuotas reales donde existen.
- Fuentes de xG gratuitas (Understat) no están disponibles de forma fiable; por eso no se usan.

## Archivos

| Archivo | Qué es |
|---|---|
| `index.html` | La app (motor de cálculo + interfaz) |
| `generar_web.py` | Lo ejecuta el robot: descarga, entrena y escribe `data.json` |
| `motor_futbol.py` | Descarga de datos y modelo de goles (Poisson + Dixon-Coles) |
| `modelo_mercado.py` | Rating de mercado y modelo de mercado; mide todo en temporadas de prueba |
| `estrategias.py` | Córners, tarjetas y backtest de Ambos marcan |
| `cuotas_api.py` | Cuotas reales de Ambos marcan (the-odds-api.com) y emparejado de nombres de equipos |
| `comprobar.js` | Prueba la app con los datos nuevos antes de publicarlos |
| `data.json` | Datos que lee la app (lo genera el robot; no tocar) |
| `cuotas_btts.json` | Cuotas ya pedidas, para no pagarlas dos veces (lo genera el robot) |
| `historico_btts.json` | Cuotas de Ambos marcan con su resultado, para medir la regla (lo genera el robot) |

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
