// Metric MDS via SMACOF (stress majorization). Minimizes raw stress
// Σ (‖yi−yj‖ − Dij)². Fits the *numeric* target distances Dij, so it preserves
// global distance structure. Uses a random (seeded) init → the converged layout
// is correct only up to rotation/reflection/translation, which is exactly the
// point: distance-preserving does NOT fix orientation.

import { mulberry32, gaussian } from "../rng.mjs";

function dist2D(Y) {
  const n = Y.length;
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const dx = Y[i][0] - Y[j][0],
        dy = Y[i][1] - Y[j][1];
      const d = Math.hypot(dx, dy);
      D[i][j] = d;
      D[j][i] = d;
    }
  return D;
}

/** target: n×n distance matrix. Returns Y (n×2).
 * `init` (optional n×2 coords, e.g. classical MDS) makes the result deterministic
 * and lowers stress vs random init; otherwise a seeded random start is used. */
export function smacof(target, seed = 1, iters = 300, init = null) {
  const n = target.length;
  const rng = mulberry32(seed);
  let Y = init
    ? init.map((p) => [p[0], p[1]])
    : Array.from({ length: n }, () => [gaussian(rng), gaussian(rng)]);

  for (let t = 0; t < iters; t++) {
    const d = dist2D(Y);
    const Yn = Array.from({ length: n }, () => [0, 0]);
    for (let i = 0; i < n; i++) {
      let bii = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const bij = d[i][j] > 1e-9 ? -target[i][j] / d[i][j] : 0;
        bii -= bij;
        Yn[i][0] += bij * Y[j][0];
        Yn[i][1] += bij * Y[j][1];
      }
      Yn[i][0] += bii * Y[i][0];
      Yn[i][1] += bii * Y[i][1];
      Yn[i][0] /= n;
      Yn[i][1] /= n;
    }
    Y = Yn;
  }
  return Y;
}
