// Generic gradient-descent MDS: minimizes ANY objective from objectives.mjs.
//
// Why a generic optimizer instead of a majorization per objective: log-distance
// fitting, kNN weighting and monotone regression would each need their own
// SMACOF derivation. Deriving four different majorizations and then comparing the
// resulting maps would confound "the objective changed" with "the update rule
// changed". One optimizer + one init isolates the objective as the only variable.
//
// Step size: Armijo backtracking line search, NOT a fixed learning rate. A fixed
// rate would have to be tuned per objective (their gradients differ in scale by
// orders of magnitude), reintroducing exactly the kind of hidden tuning knob this
// stage exists to eliminate. Backtracking removes it: the same rule serves all.

const C1 = 1e-4; // Armijo sufficient-decrease constant

function axpy(Y, g, t) {
  return Y.map((p, i) => [p[0] - t * g[i][0], p[1] - t * g[i][1]]);
}
function sqNorm(g) {
  let s = 0;
  for (const [a, b] of g) s += a * a + b * b;
  return s;
}

/**
 * @param objective  one of objectives.mjs (valueAndGrad)
 * @param D          n×n target distance matrix
 * @param init       n×2 starting coordinates (required — share it across objectives)
 * @returns { Y, value, iters, gradNorm, converged, evals }
 */
export function gdMDS(objective, D, init, opts = {}) {
  const {
    maxIter = 5000,
    gradTol = 1e-10, // stop when ‖∇‖ / max(1,|σ|) falls below this
    relTol = 1e-14, // stop when the relative objective decrease stalls
    maxBacktrack = 60,
  } = opts;

  // objective-specific precomputation (kNN weight matrix, pair ordering, …).
  // Depends only on D, so it is built once and reused across every evaluation.
  const ctx = opts.ctx ?? objective.prepare?.(D);

  let Y = init.map((p) => [p[0], p[1]]);
  let { value, grad } = objective.valueAndGrad(Y, D, ctx);
  let evals = 1;
  let t = 1 / Math.max(1, Math.sqrt(sqNorm(grad))); // scale-free first trial step
  let iters = 0;
  let converged = false;

  for (; iters < maxIter; iters++) {
    const gg = sqNorm(grad);
    if (Math.sqrt(gg) / Math.max(1, Math.abs(value)) < gradTol) {
      converged = true;
      break;
    }

    // backtracking: accept the first t with σ(Y − t∇) ≤ σ(Y) − c₁ t ‖∇‖²
    let step = t * 2; // try to grow again each iteration
    let next = null,
      nextVal = Infinity,
      nextGrad = null;
    let ok = false;
    for (let b = 0; b < maxBacktrack; b++) {
      const cand = axpy(Y, grad, step);
      const r = objective.valueAndGrad(cand, D, ctx);
      evals++;
      if (r.value <= value - C1 * step * gg) {
        next = cand;
        nextVal = r.value;
        nextGrad = r.grad;
        ok = true;
        break;
      }
      step *= 0.5;
    }
    if (!ok) {
      converged = true; // no admissible step: at (or numerically at) a minimum
      break;
    }

    const rel = (value - nextVal) / Math.max(1e-300, Math.abs(value));
    Y = next;
    value = nextVal;
    grad = nextGrad;
    t = step;
    if (rel < relTol) {
      converged = true;
      break;
    }
  }

  return { Y, value, iters, gradNorm: Math.sqrt(sqNorm(grad)), converged, evals };
}
