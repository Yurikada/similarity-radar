// Classical (Torgerson) MDS from a distance matrix: double-center D², then take
// the top-2 eigenpairs of the resulting Gram matrix B = -½ J D² J.
//
// Used here as the WARM START shared by every stress objective in Part C. Sharing
// one deterministic init is what makes the objectives comparable: any difference
// between the resulting maps is then attributable to the objective, not to where
// the optimizer happened to start.
//
// Note: B is PSD only when D is Euclidean. To stay correct for non-Euclidean D
// (e.g. disparities from monotone regression later), the power iteration runs on
// a shifted matrix B + cI so every eigenvalue is non-negative, then unshifts.

import { topKEigSym } from "./pca.mjs";

/** D: n×n distance matrix. Returns Y (n×2) and the two eigenvalues. */
export function classicalMDS(D) {
  const n = D.length;

  // squared distances
  const S = Array.from({ length: n }, (_, i) =>
    Float64Array.from({ length: n }, (_, j) => D[i][j] * D[i][j]),
  );

  // double-centering: B = -½ (S - rowMean - colMean + grandMean)
  const rowMean = new Float64Array(n);
  let grand = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += S[i][j];
    rowMean[i] = s / n;
    grand += s / (n * n);
  }
  const B = Array.from({ length: n }, (_, i) =>
    Float64Array.from({ length: n }, (_, j) => -0.5 * (S[i][j] - rowMean[i] - rowMean[j] + grand)),
  );

  // shift so the spectrum is non-negative (power iteration tracks largest |λ|)
  let shift = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += Math.abs(B[i][j]);
    shift = Math.max(shift, s);
  }
  for (let i = 0; i < n; i++) B[i][i] += shift;

  const [e1, e2] = topKEigSym(B, 2);
  const l1 = Math.max(0, e1.val - shift);
  const l2 = Math.max(0, e2.val - shift);
  const s1 = Math.sqrt(l1),
    s2 = Math.sqrt(l2);

  // sign convention identical to pca2, so re-runs are byte-identical
  const fix = (vec) => {
    let m = 0,
      mi = 0;
    for (let i = 0; i < n; i++)
      if (Math.abs(vec[i]) > m) {
        m = Math.abs(vec[i]);
        mi = i;
      }
    return vec[mi] < 0 ? -1 : 1;
  };
  const f1 = fix(e1.vec),
    f2 = fix(e2.vec);

  return {
    Y: Array.from({ length: n }, (_, i) => [f1 * s1 * e1.vec[i], f2 * s2 * e2.vec[i]]),
    eigenvalues: [l1, l2],
  };
}
