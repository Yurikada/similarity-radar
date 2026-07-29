// Sammon mapping: MDS variant weighting stress by 1/δ_ij, so it preserves SMALL
// (local) distances better than plain metric MDS at the cost of global ones.
// Uses Sammon's original diagonal pseudo-Newton step (magic factor), which
// converges far better than plain gradient descent, from a classical-MDS init.
//
//   E = (1/Σδ) Σ_{i<j} (d_ij − δ_ij)² / δ_ij

export function sammon(D, init, { iters = 120, mf = 0.3 } = {}) {
  const n = D.length;
  let Y = init.map((p) => [p[0], p[1]]);
  let c = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) c += D[i][j];
  c = (c || 1) * 2; // 2·Σ_{i<j} δ = Σ_{i≠j} δ

  for (let t = 0; t < iters; t++) {
    const Yn = Y.map((p) => [p[0], p[1]]);
    for (let p = 0; p < n; p++) {
      for (let q = 0; q < 2; q++) {
        let g1 = 0, g2 = 0;
        for (let j = 0; j < n; j++) {
          if (j === p) continue;
          const dpj = D[p][j];
          if (dpj < 1e-9) continue;
          const diff = Y[p][q] - Y[j][q];
          const d = Math.hypot(Y[p][0] - Y[j][0], Y[p][1] - Y[j][1]) || 1e-9;
          const inv = 1 / (dpj * d);
          g1 += -inv * (dpj - d) * diff;
          g2 += -inv * ((dpj - d) - (diff * diff / d) * (1 + (dpj - d) / d));
        }
        g1 *= 2 / c; g2 *= 2 / c;
        Yn[p][q] = Y[p][q] - mf * (g1 / (Math.abs(g2) || 1e-9));
      }
    }
    Y = Yn;
  }
  return Y;
}
