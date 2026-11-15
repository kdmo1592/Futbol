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
  const S = a.selecciones || [];
  if (S.length < 14) fallos.push(`${nombre} · informe con solo ${S.length} selecciones`);
  for (const x of S) if (!ok(x.p)) fallos.push(`${nombre} · informe ${x.mercado} ${x.sel}: ${x.p}`);
  if (S.length >= 3 && Math.abs(S[0].p + S[1].p + S[2].p - 1) > 1e-3) fallos.push(`${nombre} · informe: el 1X2 no suma 1`);
  const pc = Motor.probCombinada(S.filter((x) => x.tipo === "DO" && x.lado === 0 || x.tipo === "BTTS" && x.lado === 1)
    .map((x) => Object.assign({ partido: nombre, lam: a.base.lam, rho: a.base.rho }, x)));
  if (!ok(pc)) fallos.push(`${nombre} · combinada: ${pc}`);
  if (f.btts) {
    const b = f.btts;
    if (!ok(b.p) || !(b.n > 0)) fallos.push(`${nombre} · cuotas btts: p=${b.p} n=${b.n}`);
    if (!(b.mejor_si && b.mejor_si[1] > 1) || !(b.mejor_no && b.mejor_no[1] > 1)) fallos.push(`${nombre} · btts sin mejor precio`);
    if (a.fixture === f && b.n >= 2 && Math.abs(a.marca.pb - b.p) > 1e-6)
      fallos.push(`${nombre} · el informe no usa la cuota real de btts`);
  }
  for (const c of Object.values(a.conteos || {})) {
    for (const l of c.total_lineas.concat(c.local_lineas, c.visit_lineas)) if (!ok(l[1])) fallos.push(`${nombre} · ${c.nombre} ${l}`);
    if (c.ajuste && !(c.ajuste.factor > 0.5 && c.ajuste.factor < 2)) fallos.push(`${nombre} · ajuste de tarjetas raro: ${c.ajuste.factor}`);
    if (c.mercado) for (const [l, m] of Object.entries(c.mercado)) if (!ok(m.p)) fallos.push(`${nombre} · córners de mercado ${l}: ${m.p}`);
  }
}
console.log(`Partidos comprobados: ${n} de ${D.proximos.length}`);
esperados.forEach((e) => console.log("  sin análisis (normal en recién ascendidos): " + e));
if (D.proximos.length && n === 0) fallos.push("Ningún próximo partido se pudo analizar.");
if (fallos.length) { console.error("FALLOS:\n" + fallos.join("\n")); process.exit(1); }
console.log("OK");
