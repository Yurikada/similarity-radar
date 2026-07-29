// Stage B1 — whiten (remove feature correlation = stock analog of KB centering)
// then compare projections (PCA / MDS / t-SNE), mirroring A2, to test whether the
// KB findings transfer to a different domain.
//
// Run: node src/stocks/01-project.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { jacobiEigsym } from "../lib/eig.mjs";
import { pca2 } from "../lib/projection/pca.mjs";
import { smacof } from "../lib/projection/mds.mjs";
import { tsne } from "../lib/projection/tsne.mjs";
import { procrustes } from "../lib/procrustes.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "features.json"), "utf8"));
const X = S.matrix, n = X.length, d = S.features.length;

// covariance (X already z-scored -> this is the correlation matrix)
const C = Array.from({ length: d }, () => new Float64Array(d));
for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) {
  let s = 0; for (let i = 0; i < n; i++) s += X[i][a] * X[i][b];
  C[a][b] = s / n;
}
const { values, vectors } = jacobiEigsym(C);
const floor = values[0] * 1e-3;
// PCA-whitening: w_i = Λ^{-1/2} Vᵀ x_i
const W = X.map((x) => vectors.map((vec, k) => {
  let dot = 0; for (let j = 0; j < d; j++) dot += vec[j] * x[j];
  return dot / Math.sqrt(Math.max(values[k], floor));
}));

// verify decorrelation: max |corr| among whitened columns
function maxAbsCorr(M) {
  const cols = M[0].length; let mx = 0;
  for (let a = 0; a < cols; a++) for (let b = a + 1; b < cols; b++) {
    let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += M[i][a] / n; mb += M[i][b] / n; }
    let s = 0, sa = 0, sb = 0; for (let i = 0; i < n; i++) { const u = M[i][a] - ma, v = M[i][b] - mb; s += u * v; sa += u * u; sb += v * v; }
    mx = Math.max(mx, Math.abs(s / Math.sqrt(sa * sb)));
  }
  return mx;
}
console.log("=== Stage B1: whitening + projection (n=" + n + ", d=" + d + ") ===\n");
console.log("max |feature corr|  before whitening:", maxAbsCorr(X).toFixed(2), " after:", maxAbsCorr(W).toFixed(2));

// distances on whitened space (= Mahalanobis on original)
const Dhi = Array.from({ length: n }, () => new Float64Array(n));
for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
  let s = 0; for (let k = 0; k < d; k++) { const dd = W[i][k] - W[j][k]; s += dd * dd; }
  Dhi[i][j] = Dhi[j][i] = Math.sqrt(s);
}
const hiPairs = [];
for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) hiPairs.push(Dhi[i][j]);
function pairs2D(Y) { const o = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) o.push(Math.hypot(Y[i][0] - Y[j][0], Y[i][1] - Y[j][1])); return o; }
function pearson(a, b) { const m = a.length; let ma = 0, mb = 0; for (let i = 0; i < m; i++) { ma += a[i] / m; mb += b[i] / m; } let s = 0, sa = 0, sb = 0; for (let i = 0; i < m; i++) { const u = a[i] - ma, v = b[i] - mb; s += u * v; sa += u * u; sb += v * v; } return s / Math.sqrt(sa * sb); }

const proj = {
  PCA: { s1: pca2(W.map((v) => Float64Array.from(v))), s2: pca2(W.map((v) => Float64Array.from(v))) },
  MDS: { s1: smacof(Dhi, 1), s2: smacof(Dhi, 2) },
  "t-SNE": { s1: tsne(W, { seed: 1, perplexity: 12 }), s2: tsne(W, { seed: 2, perplexity: 12 }) },
};
console.log("\ntransfer check (does A2 replicate on stocks?):");
console.log("  method   rotation(2seed)   corr(hiD,2D)");
for (const [name, p] of Object.entries(proj)) {
  const { angleDeg } = procrustes(p.s1, p.s2);
  const r = pearson(hiPairs, pairs2D(p.s1));
  console.log(`  ${name.padEnd(7)} ${Math.abs(angleDeg).toFixed(1).padStart(7)}°        ${r.toFixed(3)}`);
}

fs.writeFileSync(path.join(ROOT, "out", "stocks", "whitened.json"),
  JSON.stringify({ W, Dhi_saved: false, tickers: S.tickers, names: S.names, sectors: S.sectors, features: S.features, raw: S.raw }));
console.log("\nsaved: out/stocks/whitened.json");
