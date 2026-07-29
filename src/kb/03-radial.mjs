// Stage A3 — rotation-invariant radial readout + the stability demonstration:
//   distinctiveness-radius (from fixed high-D data) is EXACTLY reproducible;
//   a projection's own 2D-radius is only approximately reproducible across seeds.
//
// Run: node src/kb/03-radial.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pca2 } from "../lib/projection/pca.mjs";
import { smacof } from "../lib/projection/mds.mjs";
import { centeredNorms, percentileRank, angleOf, radar } from "../lib/radial.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const emb = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "kb", "permanent-embeddings.json"), "utf8"));
const V = emb.vectors.map((v) => Float64Array.from(v));
const n = V.length, dim = emb.dim;
const c = new Float64Array(dim);
for (const v of V) for (let d = 0; d < dim; d++) c[d] += v[d] / n;
const Xc = V.map((v) => Float64Array.from(v, (x, d) => x - c[d]));

// radial coordinate from fixed high-D distinctiveness
const distinct = centeredNorms(Xc);
const radiusPct = percentileRank(distinct);
// angle from PCA (stable) projection
const pcaXY = pca2(Xc);
const angle = angleOf(pcaXY);
const R = radar(radiusPct, angle, 1);

function pearson(a, b) {
  const m = a.length; let ma = 0, mb = 0;
  for (let i = 0; i < m; i++) { ma += a[i] / m; mb += b[i] / m; }
  let sab = 0, sa = 0, sb = 0;
  for (let i = 0; i < m; i++) { const da = a[i] - ma, db = b[i] - mb; sab += da * db; sa += da * da; sb += db * db; }
  return sab / Math.sqrt(sa * sb);
}
// per-point 2D radius (distance from centroid) of a projection
function radius2D(Y) {
  let cx = 0, cy = 0; for (const p of Y) { cx += p[0] / n; cy += p[1] / n; }
  return Y.map((p) => Math.hypot(p[0] - cx, p[1] - cy));
}

// stability: high-D distinctiveness vs a projection's own 2D-radius across seeds
const Dhi = Array.from({ length: n }, () => new Float64Array(n));
for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
  let s = 0; for (let k = 0; k < dim; k++) { const d = Xc[i][k] - Xc[j][k]; s += d * d; }
  Dhi[i][j] = Dhi[j][i] = Math.sqrt(s);
}
const r2d_s1 = radius2D(smacof(Dhi, 1));
const r2d_s2 = radius2D(smacof(Dhi, 2));

console.log("=== Stage A3: rotation-invariant radial readout ===\n");
console.log("radius source: high-D distinctiveness ‖centered‖, mapped by percentile rank");
console.log("angle source : PCA projection (stable)\n");

console.log("stability of the RADIAL coordinate:");
console.log("  distinctiveness (high-D, fixed): identical across any projection/seed  -> corr = 1.000");
console.log(`  a projection's own 2D-radius, MDS seed1 vs seed2: corr = ${pearson(r2d_s1, r2d_s2).toFixed(3)}`);
console.log(`  (2D-radius vs true distinctiveness, seed1): corr = ${pearson(r2d_s1, distinct).toFixed(3)}\n`);

const order = [...distinct.keys()].sort((a, b) => distinct[b] - distinct[a]);
console.log("periphery (most distinctive → outer ring):");
for (const i of order.slice(0, 3)) console.log(`   pct ${(radiusPct[i] * 100).toFixed(0)}%  ${emb.labels[i].slice(0, 40)}`);
console.log("center (most mainstream → inner):");
for (const i of order.slice(-3)) console.log(`   pct ${(radiusPct[i] * 100).toFixed(0)}%  ${emb.labels[i].slice(0, 40)}`);

fs.writeFileSync(
  path.join(ROOT, "out", "kb", "radial.json"),
  JSON.stringify({ labels: emb.labels, distinct, radiusPct, angle, radar: R }),
);
console.log("\nsaved: out/kb/radial.json (gitignored)");
