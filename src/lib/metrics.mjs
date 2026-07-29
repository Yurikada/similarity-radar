// Projection quality metrics: how much high-dimensional information a 2D layout
// keeps. Different metrics capture different notions (global distance vs local
// neighborhoods), so we report several — there is no single "information kept".

function pairsFrom(D) {
  const n = D.length, out = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) out.push(D[i][j]);
  return out;
}
function dist2D(Y) {
  const n = Y.length, D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const d = Math.hypot(Y[i][0] - Y[j][0], Y[i][1] - Y[j][1]);
    D[i][j] = D[j][i] = d;
  }
  return D;
}
function pearson(a, b) {
  const m = a.length; let ma = 0, mb = 0;
  for (let i = 0; i < m; i++) { ma += a[i] / m; mb += b[i] / m; }
  let s = 0, sa = 0, sb = 0;
  for (let i = 0; i < m; i++) { const u = a[i] - ma, v = b[i] - mb; s += u * v; sa += u * u; sb += v * v; }
  return s / Math.sqrt(sa * sb);
}
// rank of every other point from i (1 = nearest), given a distance matrix
function rankRows(D) {
  const n = D.length, R = Array.from({ length: n }, () => new Int32Array(n));
  for (let i = 0; i < n; i++) {
    const order = [...Array(n).keys()].filter((j) => j !== i).sort((a, b) => D[i][a] - D[i][b]);
    order.forEach((j, r) => { R[i][j] = r + 1; });
  }
  return R;
}
function knn(D, k) {
  const n = D.length, sets = [];
  for (let i = 0; i < n; i++) {
    const order = [...Array(n).keys()].filter((j) => j !== i).sort((a, b) => D[i][a] - D[i][b]);
    sets.push(new Set(order.slice(0, k)));
  }
  return sets;
}

/** all metrics for a 2D layout Y against the high-D distance matrix Dhi. */
export function projectionMetrics(Dhi, Y, k = 10) {
  const n = Y.length;
  const D2 = dist2D(Y);
  const dh = pairsFrom(Dhi), d2 = pairsFrom(D2);

  // stress-1 (Kruskal): scale-fit d2 to dh first, then normalized residual
  let num = 0, den = 0;
  for (let p = 0; p < dh.length; p++) { den += dh[p] * dh[p]; }
  // optimal isotropic scale a = <dh,d2>/<d2,d2>
  let dot = 0, dd = 0;
  for (let p = 0; p < dh.length; p++) { dot += dh[p] * d2[p]; dd += d2[p] * d2[p]; }
  const a = dd > 0 ? dot / dd : 1;
  for (let p = 0; p < dh.length; p++) { const e = a * d2[p] - dh[p]; num += e * e; }
  const stress1 = Math.sqrt(num / den);

  const corr = pearson(dh, d2);

  // trustworthiness & continuity (Venna & Kaski) at neighborhood size k
  const rHi = rankRows(Dhi), r2 = rankRows(D2);
  const nnHi = knn(Dhi, k), nn2 = knn(D2, k);
  let tSum = 0, cSum = 0;
  for (let i = 0; i < n; i++) {
    for (const j of nn2[i]) if (!nnHi[i].has(j)) tSum += rHi[i][j] - k;   // false neighbors in 2D
    for (const j of nnHi[i]) if (!nn2[i].has(j)) cSum += r2[i][j] - k;    // missed neighbors in 2D
  }
  const norm = (2 / (n * k * (2 * n - 3 * k - 1)));
  const trust = 1 - norm * tSum;
  const cont = 1 - norm * cSum;

  return { stress1, corr, trust, cont, k };
}
