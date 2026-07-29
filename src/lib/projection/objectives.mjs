// Stress objectives for Part C — "what does the map try to preserve?"
//
// The four points of the comparison were fixed before any of them was run
// (out/objectives/preregistration.json). Each answers a different question about
// what a distance is allowed to mean:
//
//   raw stress   absolute distance      Σ (d̂ − d)²
//   log-distance relative distance      Σ (log d̂ − log d)²
//   kNN-weighted neighbourhood set      Σ w (d̂ − d)²,  w = 1 inside kNN, γ outside
//   non-metric   rank order only        min over monotone δ of  Σ(d̂−δ)² / Σd̂²
//
// A continuous d^{-p} sweep between them was deliberately rejected: p's effect
// size is governed by the coefficient of variation of the distance distribution,
// so "p=1" does not denote the same operation in two domains — fatal for a study
// whose subject is transplanting a mechanism across domains. These four have
// domain-independent verbal definitions instead.
//
// Interface, identical for all of them:
//   { name, label, prepare?(D) -> ctx, valueAndGrad(Y, D, ctx) -> {value, grad} }

const EPS = 1e-12;

function dist2D(Y) {
  const n = Y.length;
  const d = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const dx = Y[i][0] - Y[j][0],
        dy = Y[i][1] - Y[j][1];
      const v = Math.hypot(dx, dy);
      d[i][j] = v;
      d[j][i] = v;
    }
  return d;
}
const zeroGrad = (n) => Array.from({ length: n }, () => [0, 0]);

/** Accumulate g = Σ_pairs coef_ij · (y_i − y_j) into grad. */
function spread(grad, Y, i, j, coef) {
  const gx = coef * (Y[i][0] - Y[j][0]),
    gy = coef * (Y[i][1] - Y[j][1]);
  grad[i][0] += gx;
  grad[i][1] += gy;
  grad[j][0] -= gx;
  grad[j][1] -= gy;
}

// ------------------------------------------------------------- 1. raw stress

/** σ = Σ_{i<j} (‖yi−yj‖ − Dij)².  Absolute error: large distances dominate.
 *  Exactly what SMACOF majorizes — C0 uses this to calibrate the optimizer. */
export const rawStress = {
  name: "raw",
  label: "raw stress (絶対距離)",
  status: "valid",
  valueAndGrad(Y, D) {
    const n = Y.length,
      d = dist2D(Y),
      grad = zeroGrad(n);
    let value = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const dij = d[i][j],
          r = dij - D[i][j];
        value += r * r;
        spread(grad, Y, i, j, dij > EPS ? (2 * r) / dij : 0);
      }
    return { value, grad };
  },
};

// ---------------------------------------------------------- 2. log-distance

/** σ = Σ_{i<j} (log‖yi−yj‖ − log Dij)².  Uniform RELATIVE error at every scale,
 *  and invariant under D→cD, Y→cY. This is the exact object that Sammon's 1/d
 *  weighting only approximates to first order, with no ε-style weight blow-up. */
export const logDistance = {
  name: "log",
  label: "log距離 (相対距離)",
  status: "valid",
  valueAndGrad(Y, D) {
    const n = Y.length,
      d = dist2D(Y),
      grad = zeroGrad(n);
    let value = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const dij = Math.max(d[i][j], EPS);
        const r = Math.log(dij) - Math.log(D[i][j]);
        value += r * r;
        // ∂/∂y_i (log d̂ − log d)² = 2r · (1/d̂) · (y_i−y_j)/d̂
        spread(grad, Y, i, j, (2 * r) / (dij * dij));
      }
    return { value, grad };
  },
};

// --------------------------------------------------------- 3. kNN-weighted

/** Symmetric weights: 1 if j is among i's k nearest (or vice versa), else γ. */
export function knnWeights(D, k, gamma) {
  const n = D.length;
  const W = Array.from({ length: n }, () => new Float64Array(n).fill(gamma));
  for (let i = 0; i < n; i++) {
    const order = [...Array(n).keys()].filter((j) => j !== i).sort((a, b) => D[i][a] - D[i][b]);
    for (const j of order.slice(0, k)) {
      W[i][j] = 1;
      W[j][i] = 1;
    }
  }
  return W;
}

/** σ = Σ w_ij (d̂ − d)² with w from the kNN graph.
 *  Rank-based, so the weight means the same thing under any monotone rescaling
 *  of the distances — unlike d^{-p}. But k and γ are themselves hidden knobs of
 *  the same species as Sammon's ε, which is why C1 sweeps them.
 *
 *  DISQUALIFIED in C1, kept in the comparison as a documented negative result.
 *  It fails twice, and the second failure is the interesting one:
 *
 *   (1) unstable to its own knobs — the k×γ grid moves the map more (disparity
 *       0.662 stocks / 0.843 kb) than the choice of objective does (0.618/0.695).
 *       The ε criticism levelled at d^{-p} applies here unchanged.
 *
 *   (2) LOSS OF GEOMETRIC SCAFFOLDING — the objective built to protect
 *       neighbourhoods scores LAST on neighbourhood preservation
 *       (trust 0.679/0.684, cont 0.814/0.755 against ≥0.800/≥0.848 elsewhere).
 *       Down-weighting far pairs to γ removes the struts that hold unrelated
 *       clusters apart, so they drift together and manufacture false neighbours
 *       in 2D. Locality cannot be imposed by weighting alone: preserving a
 *       neighbourhood requires the distant pairs that define where it is NOT. */
export function knnWeighted(k, gamma) {
  return {
    name: `knn-k${k}-g${gamma}`,
    label: `kNN重み (近傍集合, k=${k}, γ=${gamma})`,
    status: "disqualified",
    disqualification: {
      stage: "C1",
      reasons: ["unstable to own hyperparameters (k, γ)", "loss of geometric scaffolding"],
      retained: "reported alongside the valid three; excluding it would be the H10 error",
    },
    prepare: (D) => ({ W: knnWeights(D, k, gamma) }),
    valueAndGrad(Y, D, ctx) {
      const n = Y.length,
        d = dist2D(Y),
        grad = zeroGrad(n),
        W = ctx.W;
      let value = 0;
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
          const w = W[i][j],
            dij = d[i][j],
            r = dij - D[i][j];
          value += w * r * r;
          spread(grad, Y, i, j, dij > EPS ? (2 * w * r) / dij : 0);
        }
      return { value, grad };
    },
  };
}

// ------------------------------------------------- 4. non-metric (monotone)

/** Pool-adjacent-violators: least-squares fit of a non-decreasing sequence. */
export function pava(v) {
  const m = v.length;
  const val = new Float64Array(m),
    wt = new Float64Array(m),
    idx = new Int32Array(m);
  let top = -1;
  for (let i = 0; i < m; i++) {
    top++;
    val[top] = v[i];
    wt[top] = 1;
    idx[top] = i;
    while (top > 0 && val[top - 1] > val[top]) {
      const w = wt[top - 1] + wt[top];
      val[top - 1] = (wt[top - 1] * val[top - 1] + wt[top] * val[top]) / w;
      wt[top - 1] = w;
      top--;
    }
  }
  const out = new Float64Array(m);
  let pos = 0;
  for (let b = 0; b <= top; b++) for (let c = 0; c < wt[b]; c++) out[pos++] = val[b];
  return out;
}

/** Kruskal stress-1 with a monotone transform:  σ = √( Σ(d̂−δ)² / Σd̂² ),
 *  δ = isotonic regression of d̂ against the rank order of D.
 *
 *  The Σd̂² denominator is not cosmetic — without it the objective is minimised
 *  by collapsing every point together, the classic non-metric degeneracy. C1
 *  still checks the spread of the fitted δ explicitly rather than trusting it.
 *
 *  δ is re-fitted at every evaluation, so `value` really is the minimum over
 *  monotone transforms. Holding δ fixed inside the gradient is exact, not an
 *  approximation: δ is optimal given the ordering, so ∂σ/∂δ = 0 (envelope). */
export const nonMetric = {
  name: "nonmetric",
  label: "非計量 (順位のみ・単調回帰)",
  status: "valid",
  prepare(D) {
    const n = D.length,
      pairs = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
    // fixed ordering by target distance; ties broken deterministically by index
    pairs.sort((a, b) => D[a[0]][a[1]] - D[b[0]][b[1]] || a[0] - b[0] || a[1] - b[1]);
    return { order: pairs };
  },
  valueAndGrad(Y, D, ctx) {
    const n = Y.length,
      d = dist2D(Y),
      grad = zeroGrad(n);
    const ord = ctx.order,
      m = ord.length;
    const dh = new Float64Array(m);
    for (let p = 0; p < m; p++) dh[p] = d[ord[p][0]][ord[p][1]];

    const delta = pava(dh);
    let S = 0,
      T = 0;
    for (let p = 0; p < m; p++) {
      const r = dh[p] - delta[p];
      S += r * r;
      T += dh[p] * dh[p];
    }
    const sigma = T > EPS ? Math.sqrt(S / T) : 0;
    if (sigma < EPS) return { value: sigma, grad };

    // ∂σ/∂d̂_p = ( (d̂−δ) − σ² d̂ ) / (σ T)
    for (let p = 0; p < m; p++) {
      const [i, j] = ord[p],
        dij = Math.max(dh[p], EPS);
      const dsig = (dh[p] - delta[p] - sigma * sigma * dh[p]) / (sigma * T);
      spread(grad, Y, i, j, dsig / dij);
    }
    return { value: sigma, grad, _fit: { delta, dh, S, T } };
  },
};

export const OBJECTIVES = { raw: rawStress, log: logDistance, nonmetric: nonMetric };
