// Full symmetric eigendecomposition via the cyclic Jacobi method. For the small
// dense covariance matrices (d≈9) used in whitening. Returns eigenvalues (desc)
// and the corresponding orthonormal eigenvectors as columns.

export function jacobiEigsym(Ain, { sweeps = 100, tol = 1e-12 } = {}) {
  const n = Ain.length;
  const A = Ain.map((r) => Float64Array.from(r));
  const V = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < tol) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - sn * akq;
          A[k][q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - sn * aqk;
          A[q][k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - sn * vkq;
          V[k][q] = sn * vkp + c * vkq;
        }
      }
    }
  }
  const vals = A.map((r, i) => r[i]);
  const order = [...vals.keys()].sort((a, b) => vals[b] - vals[a]);
  return {
    values: order.map((i) => vals[i]),
    vectors: order.map((i) => V.map((row) => row[i])), // vectors[k] = k-th eigenvector
  };
}
