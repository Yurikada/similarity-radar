// The two domains Part C compares, loaded identically for every stage so C0's
// pre-registration and C1's comparison provably share one input.
//
// stocks: whitened features -> Euclidean = Mahalanobis in the original space (B1)
// kb:     centered embeddings -> Euclidean (A1/A2). Centering shifts every vector
//         by the same amount, so pairwise distances are unchanged; done anyway to
//         stay byte-identical with 02-projection.mjs.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

export function euclid(X) {
  const n = X.length,
    dim = X[0].length;
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = 0; k < dim; k++) {
        const d = X[i][k] - X[j][k];
        s += d * d;
      }
      D[i][j] = D[j][i] = Math.sqrt(s);
    }
  return D;
}

export function loadDomains() {
  const SW = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "whitened.json"), "utf8"));
  const KBP = JSON.parse(
    fs.readFileSync(path.join(ROOT, "out", "kb", "permanent-embeddings.json"), "utf8"),
  );
  const dim = KBP.dim,
    nk = KBP.vectors.length;
  const c = new Float64Array(dim);
  for (const v of KBP.vectors) for (let d = 0; d < dim; d++) c[d] += v[d] / nk;

  return [
    {
      key: "stocks",
      label: `stocks (n=${SW.W.length}, whitened Mahalanobis)`,
      D: euclid(SW.W),
      labels: SW.names,
      tickers: SW.tickers,
      sectors: SW.sectors,
      // B1/C0: whitening makes the covariance isotropic, so the classical-MDS
      // warm start sits in a near-degenerate eigen-subspace (144.98 vs 144.23).
      // The init is deterministic but NOT canonical -> read shape, never angle.
      canonicalFrame: false,
    },
    {
      key: "kb",
      label: `kb (n=${nk}, centered ${dim}d embeddings)`,
      D: euclid(KBP.vectors.map((v) => Float64Array.from(v, (x, d) => x - c[d]))),
      labels: KBP.labels,
      canonicalFrame: true, // eigenvalues 1.56 vs 1.20 are separated
    },
  ];
}

export function pairStats(D) {
  const n = D.length,
    v = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) v.push(D[i][j]);
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length);
  v.sort((a, b) => a - b);
  const q = (p) => v[Math.min(v.length - 1, Math.floor(p * (v.length - 1)))];
  return { n, pairs: v.length, mean, sd, cv: sd / mean, min: q(0), p50: q(0.5), max: q(1) };
}
