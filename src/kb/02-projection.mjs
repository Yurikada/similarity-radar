// Stage A2 — Projection comparison: PCA vs metric MDS (SMACOF) vs t-SNE.
//   Experiment I  (rotation): run each twice; Procrustes-align the two runs and
//                  report rotation angle + residual disparity (structure change).
//   Experiment II (global fidelity / Shepard): correlate high-dim distances with
//                  2D distances (Pearson + Spearman) over all pairs.
//
// Run: node src/kb/02-projection.mjs   (after npm run kb:matrix)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pca2 } from "../lib/projection/pca.mjs";
import { smacof } from "../lib/projection/mds.mjs";
import { tsne } from "../lib/projection/tsne.mjs";
import { procrustes } from "../lib/procrustes.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const payload = JSON.parse(
  fs.readFileSync(path.join(ROOT, "out", "kb", "permanent-embeddings.json"), "utf8"),
);
const V = payload.vectors.map((v) => Float64Array.from(v));
const n = V.length,
  dim = payload.dim;

// center (A1 decision)
const c = new Float64Array(dim);
for (const v of V) for (let d = 0; d < dim; d++) c[d] += v[d] / n;
const Xc = V.map((v) => Float64Array.from(v, (x, d) => x - c[d]));

// high-dim Euclidean distance (target for MDS + reference for Shepard)
function hiDist() {
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = 0; k < dim; k++) {
        const d = Xc[i][k] - Xc[j][k];
        s += d * d;
      }
      const d = Math.sqrt(s);
      D[i][j] = d;
      D[j][i] = d;
    }
  return D;
}
const Dhi = hiDist();

function pairs2D(Y) {
  const out = [];
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      out.push(Math.hypot(Y[i][0] - Y[j][0], Y[i][1] - Y[j][1]));
  return out;
}
const hiPairs = [];
for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) hiPairs.push(Dhi[i][j]);

function pearson(a, b) {
  const m = a.length;
  let ma = 0,
    mb = 0;
  for (let i = 0; i < m; i++) {
    ma += a[i] / m;
    mb += b[i] / m;
  }
  let sab = 0,
    sa = 0,
    sb = 0;
  for (let i = 0; i < m; i++) {
    const da = a[i] - ma,
      db = b[i] - mb;
    sab += da * db;
    sa += da * da;
    sb += db * db;
  }
  return sab / Math.sqrt(sa * sb);
}
function ranks(a) {
  const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]);
  const r = new Array(a.length);
  for (let k = 0; k < idx.length; k++) r[idx[k][1]] = k;
  return r;
}
const spearman = (a, b) => pearson(ranks(a), ranks(b));

// build projections
const proj = {
  PCA: { s1: pca2(Xc), s2: pca2(Xc) },
  MDS: { s1: smacof(Dhi, 1), s2: smacof(Dhi, 2) },
  "t-SNE": { s1: tsne(Xc, { seed: 1 }), s2: tsne(Xc, { seed: 2 }) },
};

console.log("=== Stage A2: projection comparison (n=" + n + ") ===\n");
console.log("Exp I — rotation between two runs (Procrustes):");
console.log("  method   rotation(deg)   disparity(structure Δ)");
for (const [name, p] of Object.entries(proj)) {
  const { angleDeg, disparity } = procrustes(p.s1, p.s2);
  console.log(
    `  ${name.padEnd(7)}  ${Math.abs(angleDeg).toFixed(1).padStart(8)}       ${disparity.toExponential(2)}`,
  );
}

console.log("\nExp II — global fidelity: corr(high-dim dist, 2D dist) over 3655 pairs:");
console.log("  method   Pearson r   Spearman rho");
const shepard = {};
for (const [name, p] of Object.entries(proj)) {
  const d2 = pairs2D(p.s1);
  const pe = pearson(hiPairs, d2),
    sp = spearman(hiPairs, d2);
  shepard[name] = { pearson: pe, spearman: sp };
  console.log(`  ${name.padEnd(7)}  ${pe.toFixed(3).padStart(7)}     ${sp.toFixed(3).padStart(7)}`);
}

// persist coords + shepard pairs for later visualization (gitignored)
const outDir = path.join(ROOT, "out", "kb");
const dump = { labels: payload.labels, tags: payload.tags, shepard, hiPairs, proj: {} };
for (const [name, p] of Object.entries(proj)) dump.proj[name] = { s1: p.s1, s2: p.s2, d2: pairs2D(p.s1) };
fs.writeFileSync(path.join(outDir, "projections.json"), JSON.stringify(dump));
console.log("\nsaved: out/kb/projections.json (gitignored)");
