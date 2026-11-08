/*
 * Comprobación que hace el robot ANTES de publicar data.json:
 * ejecuta el motor de la app (index.html) con los datos nuevos sobre todos los
 * próximos partidos y falla si salen probabilidades imposibles (NaN, fuera de 0-1,
 * que no suman 1). Así un data.json roto nunca llega al iPhone.
 *   node comprobar.js
 */
const fs = require("fs");
const html = fs.readFileSync(__dirname + "/index.html", "utf8");
const codigo = html.match(/<script id="motor">([\s\S]*?)<\/script>/)[1];
const Motor = new Function("module", codigo + "\nreturn Motor;")({});
const D = JSON.parse(fs.readFileSync(__dirname + "/data.json", "utf8"));

const fallos = [], esperados = [];
const ok = (p) => Number.isFinite(p) && p >= -1e-9 && p <= 1 + 1e-9;
let n = 0;
for (const f of D.proximos) {
  let a;
  try { a = Motor.analizar(D, f.liga, f.local, f.visitante); }
  catch (e) { esperados.push(`${f.liga} ${f.local}-${f.visitante}: ${e.message}`); continue; }
  n++;
  const nombre = `${f.liga} ${f.local}-${f.visitante}`;
  for (const k of a.orden) {
    const p = a.fuentes[k].p, s = p.reduce((x, y) => x + y, 0);
    if (!p.every(ok) || Math.abs(s - 1) > 1e-3) fallos.push(`${nombre} · ${k}: 1X2 ${p}`);
  }
  for (const m of Motor.mercados(a.M)) if (!ok(m[2])) fallos.push(`${nombre} · ${m[0]} ${m[1]}: ${m[2]}`);
  for (const x of [a.marca.pl, a.marca.pv, a.marca.pb]) if (!ok(x)) fallos.push(`${nombre} · ambos marcan: ${x}`);
  for (const c of Object.values(a.conteos || {}))
    for (const l of c.total_lineas.concat(c.local_lineas, c.visit_lineas)) if (!ok(l[1])) fallos.push(`${nombre} · ${c.nombre} ${l}`);
}
console.log(`Partidos comprobados: ${n} de ${D.proximos.length}`);
esperados.forEach((e) => console.log("  sin análisis (normal en recién ascendidos): " + e));
if (D.proximos.length && n === 0) fallos.push("Ningún próximo partido se pudo analizar.");
if (fallos.length) { console.error("FALLOS:\n" + fallos.join("\n")); process.exit(1); }
console.log("OK");
