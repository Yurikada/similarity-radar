// Stage A4 — KDE density surface + the two traps:
//   (1) which theme is "hot" (most-written) — read from tags of dense notes;
//   (2) density on the percentile-RADAR is partly a 1/r geometric artifact, so
//       KDE for "competition/whitespace" must run on the distance-preserving
//       projection (PCA), not on the percentile-radar.
//
// Run: node src/kb/04-density.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pca2 } from "../lib/projection/pca.mjs";
import { pointDensities, scottBandwidth } from "../lib/kde.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const emb = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "kb", "permanent-embeddings.json"), "utf8"));
const rad = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "kb", "radial.json"), "utf8"));
const V = emb.vectors.map((v) => Float64Array.from(v));
const n = V.length, dim = emb.dim;
const c = new Float64Array(dim);
for (const v of V) for (let d = 0; d < dim; d++) c[d] += v[d] / n;
const Xc = V.map((v) => Float64Array.from(v, (x, d) => x - c[d]));

// density on the distance-preserving PCA projection (meaningful)
const pca = pca2(Xc);
const hP = scottBandwidth(pca);
const densP = pointDensities(pca, hP);

function topTags(indices) {
  const cnt = new Map();
  for (const i of indices) for (const t of emb.tags[i]) if (t !== "permanent") cnt.set(t, (cnt.get(t) ?? 0) + 1);
  return [...cnt.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([t, c]) => `${t}:${c}`).join("  ");
}
const orderP = [...densP.keys()].sort((a, b) => densP[b] - densP[a]);

console.log("=== Stage A4: KDE density (on PCA, distance-preserving) ===\n");
console.log("HOT (densest 12) tags :", topTags(orderP.slice(0, 12)));
console.log("COLD (sparsest 12) tags:", topTags(orderP.slice(-12)));
console.log("\ndensest notes:");
for (const i of orderP.slice(0, 3)) console.log("   ", emb.labels[i].slice(0, 40));
console.log("sparsest notes (whitespace edges):");
for (const i of orderP.slice(-3)) console.log("   ", emb.labels[i].slice(0, 40));

// trap: does density correlate with the theme cluster (angle) or with center (radius)?
// correlation of PCA-density with radar-radius percentile:
function pearson(a, b) { const m = a.length; let ma = 0, mb = 0; for (let i = 0; i < m; i++) { ma += a[i] / m; mb += b[i] / m; } let s = 0, sa = 0, sb = 0; for (let i = 0; i < m; i++) { const da = a[i] - ma, db = b[i] - mb; s += da * db; sa += da * da; sb += db * db; } return s / Math.sqrt(sa * sb); }
console.log("\ncorr(PCA-density, radar-radius pct) =", pearson(densP, rad.radiusPct).toFixed(3),
  "  (negative => dense notes are central/mainstream, sparse => distinctive)");

// artifact demonstration: radial density profile on the percentile-RADAR.
// points-per-shell / shell-area. Percentile radius is uniform, so this ~ 1/r.
const R = rad.radar;
const rr = R.map((p) => Math.hypot(p[0], p[1]));
const bins = 5;
console.log("\n[trap] area-density by radius on the percentile-radar (uniform radius => ~1/r):");
for (let b = 0; b < bins; b++) {
  const lo = b / bins, hi = (b + 1) / bins;
  const cnt = rr.filter((r) => r >= lo && r < hi + (b === bins - 1 ? 1e-9 : 0)).length;
  const area = Math.PI * (hi * hi - lo * lo);
  console.log(`  r ${lo.toFixed(1)}-${hi.toFixed(1)}: ${cnt} pts / area ${area.toFixed(2)} = density ${(cnt / area).toFixed(1)}`);
}

fs.writeFileSync(path.join(ROOT, "out", "kb", "density.json"),
  JSON.stringify({ densityP: densP, bandwidth: hP, pca }));
console.log("\nsaved: out/kb/density.json (gitignored)");
