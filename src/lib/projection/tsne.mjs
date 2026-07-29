// Compact t-SNE. Preserves LOCAL neighborhoods (perplexity-scaled Gaussian
// affinities in high-D, matched to a heavy-tailed t-distribution in 2D). Random
// (seeded) init + non-convex SGD → both orientation AND fine structure vary
// between seeds. Global inter-cluster distances are not preserved by design.

import { mulberry32, gaussian } from "../rng.mjs";

function sqdistHighD(X) {
  const n = X.length;
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      const a = X[i],
        b = X[j];
      for (let k = 0; k < a.length; k++) {
        const dd = a[k] - b[k];
        s += dd * dd;
      }
      D[i][j] = s;
      D[j][i] = s;
    }
  return D;
}

// Row-wise affinities via binary search on beta to hit target perplexity.
function computeP(D2, perplexity) {
  const n = D2.length;
  const P = Array.from({ length: n }, () => new Float64Array(n));
  const logU = Math.log(perplexity);
  for (let i = 0; i < n; i++) {
    let betaMin = -Infinity,
      betaMax = Infinity,
      beta = 1;
    let row = new Float64Array(n);
    for (let tries = 0; tries < 60; tries++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        row[j] = i === j ? 0 : Math.exp(-D2[i][j] * beta);
        sum += row[j];
      }
      if (sum < 1e-12) sum = 1e-12;
      let H = 0;
      for (let j = 0; j < n; j++) {
        const p = row[j] / sum;
        if (p > 1e-12) H += -p * Math.log(p);
      }
      const diff = H - logU;
      if (Math.abs(diff) < 1e-5) break;
      if (diff > 0) {
        betaMin = beta;
        beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2;
      } else {
        betaMax = beta;
        beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2;
      }
    }
    let sum = 0;
    for (let j = 0; j < n; j++) sum += row[j];
    for (let j = 0; j < n; j++) P[i][j] = row[j] / (sum || 1);
  }
  // symmetrize + normalize
  let tot = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      P[i][j] = (P[i][j] + P[j][i]) / 2;
      tot += P[i][j];
    }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i][j] = Math.max(P[i][j] / tot, 1e-12);
  return P;
}

export function tsne(X, { seed = 1, perplexity = 15, iters = 600, lr = 120 } = {}) {
  const n = X.length;
  const rng = mulberry32(seed);
  const P = computeP(sqdistHighD(X), perplexity);
  let Y = Array.from({ length: n }, () => [gaussian(rng) * 1e-2, gaussian(rng) * 1e-2]);
  const gains = Array.from({ length: n }, () => [1, 1]);
  const inc = Array.from({ length: n }, () => [0, 0]);

  for (let t = 0; t < iters; t++) {
    const exag = t < 100 ? 4 : 1;
    const momentum = t < 250 ? 0.5 : 0.8;
    // Q (unnormalized) and normalizer
    const num = Array.from({ length: n }, () => new Float64Array(n));
    let qsum = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const dx = Y[i][0] - Y[j][0],
          dy = Y[i][1] - Y[j][1];
        const q = 1 / (1 + dx * dx + dy * dy);
        num[i][j] = q;
        num[j][i] = q;
        qsum += 2 * q;
      }
    qsum = Math.max(qsum, 1e-12);
    // gradient
    const grad = Array.from({ length: n }, () => [0, 0]);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const q = num[i][j] / qsum;
        const mult = (exag * P[i][j] - q) * num[i][j];
        grad[i][0] += mult * (Y[i][0] - Y[j][0]);
        grad[i][1] += mult * (Y[i][1] - Y[j][1]);
      }
    for (let i = 0; i < n; i++)
      for (let d = 0; d < 2; d++) {
        const g = 4 * grad[i][d];
        gains[i][d] =
          Math.sign(g) === Math.sign(inc[i][d]) ? gains[i][d] * 0.8 : gains[i][d] + 0.2;
        if (gains[i][d] < 0.01) gains[i][d] = 0.01;
        inc[i][d] = momentum * inc[i][d] - lr * gains[i][d] * g;
        Y[i][d] += inc[i][d];
      }
    // recenter
    let mx = 0,
      my = 0;
    for (let i = 0; i < n; i++) {
      mx += Y[i][0] / n;
      my += Y[i][1] / n;
    }
    for (let i = 0; i < n; i++) {
      Y[i][0] -= mx;
      Y[i][1] -= my;
    }
  }
  return Y;
}
