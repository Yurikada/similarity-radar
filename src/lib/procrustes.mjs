// 2D Procrustes: best similarity transform (rotation + optional reflection +
// scale + translation) aligning B onto A. Returns the rotation angle and the
// residual "disparity" (normalized leftover after the best alignment).
//
// disparity ≈ 0  → same structure, differing only by orientation/scale (rotation-only difference)
// disparity large → the layouts genuinely differ (structure changed)

function center(P) {
  const n = P.length;
  let cx = 0,
    cy = 0;
  for (const p of P) {
    cx += p[0] / n;
    cy += p[1] / n;
  }
  const Q = P.map((p) => [p[0] - cx, p[1] - cy]);
  let s = 0;
  for (const p of Q) s += p[0] * p[0] + p[1] * p[1];
  return { Q, scale: Math.sqrt(s) };
}

/** Same alignment, but returns B actually mapped onto A's frame (centered and
 * unit-scaled like A). Use this when you need the aligned coordinates — e.g. to
 * measure per-point drift after removing the frame — instead of hand-rolling the
 * rotation at the call site and re-introducing the reflected-branch bug. */
export function procrustesAlign(A, B) {
  const { angleDeg, disparity, reflected } = procrustes(A, B);
  const n = B.length;
  const { Q: Bc, scale: sB } = center(B);
  const a = (angleDeg * Math.PI) / 180,
    c = Math.cos(a),
    s = Math.sin(a);
  const Y = new Array(n);
  for (let i = 0; i < n; i++) {
    const bx = Bc[i][0] / sB,
      by = (reflected ? -Bc[i][1] : Bc[i][1]) / sB;
    Y[i] = [c * bx - s * by, s * bx + c * by];
  }
  return { Y, angleDeg, disparity, reflected };
}

export function procrustes(A, B) {
  const n = A.length;
  const { Q: Ac, scale: sA } = center(A);
  const { Q: Bc, scale: sB } = center(B);
  const An = Ac.map((p) => [p[0] / sA, p[1] / sA]);
  const Bn = Bc.map((p) => [p[0] / sB, p[1] / sB]);

  // cross-covariance M = Anᵀ Bn (2×2)
  let m00 = 0,
    m01 = 0,
    m10 = 0,
    m11 = 0;
  for (let i = 0; i < n; i++) {
    m00 += An[i][0] * Bn[i][0];
    m01 += An[i][0] * Bn[i][1];
    m10 += An[i][1] * Bn[i][0];
    m11 += An[i][1] * Bn[i][1];
  }
  // Optimal rotation to map Bn -> An (2D Kabsch). Reflection is a SEPARATE branch:
  // flipping y changes the cross-covariance (m01,m11 -> -m01,-m11), so the reflected
  // branch has its own optimal angle. Reusing the unreflected angle here would
  // overstate the reflected residual and could pick the wrong branch.
  const angleFor = (flip) =>
    flip ? Math.atan2(m10 + m01, m00 - m11) : Math.atan2(m10 - m01, m00 + m11);
  const resid = (flip, angle) => {
    const c = Math.cos(angle),
      s = Math.sin(angle);
    let e = 0;
    for (let i = 0; i < n; i++) {
      const bx = Bn[i][0],
        by = flip ? -Bn[i][1] : Bn[i][1];
      const rx = c * bx - s * by,
        ry = s * bx + c * by;
      e += (rx - An[i][0]) ** 2 + (ry - An[i][1]) ** 2;
    }
    return e;
  };
  const a0 = angleFor(false),
    a1 = angleFor(true);
  const e0 = resid(false, a0),
    e1 = resid(true, a1);
  const flipped = e1 < e0;
  return {
    angleDeg: ((flipped ? a1 : a0) * 180) / Math.PI,
    disparity: flipped ? e1 : e0,
    reflected: flipped,
    // rotation-only disparity: what the previous implementation reported when the
    // two layouts differ by a pure rotation. Kept so A2/A6 numbers stay checkable.
    disparityNoReflection: e0,
  };
}
