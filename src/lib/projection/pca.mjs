// PCA (top-2) via the dual/Gram route: for n < d it is cheaper to eigen-decompose
// the n×n Gram matrix of the centered data than the d×d covariance. The top
// eigenvectors of G = Xc·Xcᵀ, scaled by sqrt(eigenvalue), ARE the PCA scores.
// (This is also exactly classical MDS on Euclidean distances — the two coincide.)
//
// Deterministic: no random seed. Sign of each axis is fixed by a convention so
// re-runs are byte-identical (used to demonstrate rotation-stability).

function gram(Xc) {
  const n = Xc.length;
  const G = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = i; j < n; j++) {
      let s = 0;
      const a = Xc[i],
        b = Xc[j];
      for (let k = 0; k < a.length; k++) s += a[k] * b[k];
      G[i][j] = s;
      G[j][i] = s;
    }
  return G;
}

function matVec(M, v) {
  const n = M.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const row = M[i];
    for (let j = 0; j < n; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}

// Top eigenpair of symmetric M by power iteration. Deterministic fixed init so
// the whole pipeline is reproducible (classical-MDS init must be stable).
function topEig(M, iters = 500) {
  const n = M.length;
  let v = new Float64Array(n).map((_, i) => Math.sin(i + 1) * 0.5 + 0.3);
  let norm = Math.hypot(...v);
  v = v.map((x) => x / norm);
  let lambda = 0;
  for (let t = 0; t < iters; t++) {
    const Mv = matVec(M, v);
    norm = Math.hypot(...Mv);
    if (norm < 1e-12) break;
    const vn = Mv.map((x) => x / norm);
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(vn[i] - v[i]);
    v = vn;
    lambda = norm;
    if (diff < 1e-11) break;
  }
  return { vec: v, val: lambda };
}

function deflate(M, vec, val) {
  const n = M.length;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) M[i][j] -= val * vec[i] * vec[j];
}

/** Top-k eigenpairs of a symmetric PSD matrix, by deflated power iteration.
 * Copies M so callers keep their matrix. Same deterministic init as pca2, so
 * classical MDS and PCA share one reproducible eigen-routine. */
export function topKEigSym(M, k) {
  const A = M.map((r) => Float64Array.from(r));
  const out = [];
  for (let t = 0; t < k; t++) {
    const e = topEig(A);
    out.push(e);
    deflate(A, e.vec, e.val);
  }
  return out;
}

/** centered vectors Xc (n×d) -> scores (n×2). */
export function pca2(Xc) {
  const G = gram(Xc);
  const e1 = topEig(G);
  deflate(G, e1.vec, e1.val);
  const e2 = topEig(G);
  const s1 = Math.sqrt(Math.max(0, e1.val));
  const s2 = Math.sqrt(Math.max(0, e2.val));
  const n = Xc.length;
  // sign convention: make the largest-|coord| entry positive on each axis
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
  return Array.from({ length: n }, (_, i) => [f1 * s1 * e1.vec[i], f2 * s2 * e2.vec[i]]);
}
