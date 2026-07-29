// k-means with k-means++ seeded init (Euclidean). Works in any dimension.

import { mulberry32 } from "./rng.mjs";

function d2(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; }
  return s;
}

export function kmeans(X, k, { seed = 1, iters = 100 } = {}) {
  const n = X.length, dim = X[0].length;
  const rng = mulberry32(seed);
  // k-means++ init
  const centers = [X[Math.floor(rng() * n)].slice()];
  while (centers.length < k) {
    const dmin = X.map((x) => Math.min(...centers.map((c) => d2(x, c))));
    const tot = dmin.reduce((a, b) => a + b, 0) || 1;
    let r = rng() * tot, idx = 0;
    for (let i = 0; i < n; i++) { r -= dmin[i]; if (r <= 0) { idx = i; break; } }
    centers.push(X[idx].slice());
  }
  let labels = new Array(n).fill(0);
  for (let t = 0; t < iters; t++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) { const d = d2(X[i], centers[c]); if (d < bd) { bd = d; best = c; } }
      if (labels[i] !== best) { labels[i] = best; moved = true; }
    }
    const sum = Array.from({ length: k }, () => new Float64Array(dim));
    const cnt = new Array(k).fill(0);
    for (let i = 0; i < n; i++) { cnt[labels[i]]++; for (let d = 0; d < dim; d++) sum[labels[i]][d] += X[i][d]; }
    for (let c = 0; c < k; c++) if (cnt[c]) for (let d = 0; d < dim; d++) centers[c][d] = sum[c][d] / cnt[c];
    if (!moved && t > 0) break;
  }
  return { labels, centers };
}

/** Adjusted Rand Index between two labelings. */
export function adjustedRand(a, b) {
  const n = a.length;
  const A = [...new Set(a)], B = [...new Set(b)];
  const idxA = new Map(A.map((v, i) => [v, i])), idxB = new Map(B.map((v, i) => [v, i]));
  const M = Array.from({ length: A.length }, () => new Array(B.length).fill(0));
  for (let i = 0; i < n; i++) M[idxA.get(a[i])][idxB.get(b[i])]++;
  const comb2 = (x) => (x * (x - 1)) / 2;
  let sumIJ = 0, sumA = 0, sumB = 0;
  const ai = M.map((row) => row.reduce((s, v) => s + v, 0));
  const bj = B.map((_, j) => M.reduce((s, row) => s + row[j], 0));
  for (let i = 0; i < A.length; i++) for (let j = 0; j < B.length; j++) sumIJ += comb2(M[i][j]);
  for (const x of ai) sumA += comb2(x);
  for (const x of bj) sumB += comb2(x);
  const tot = comb2(n);
  const exp = (sumA * sumB) / tot;
  const maxi = (sumA + sumB) / 2;
  return maxi - exp === 0 ? 1 : (sumIJ - exp) / (maxi - exp);
}
