// Rotation-invariant radial readout.
//
// distinctiveness(i) = ‖centered_i‖ = distance from the centroid in HIGH-D space.
// This is invariant to any rotation/reflection of the configuration AND does not
// depend on the (wobbly) 2D projection at all — it is computed from fixed data.
// We map it to a radius by PERCENTILE RANK so the rings are evenly populated and
// the compressed raw magnitudes (A1) don't collapse everything to one radius.
//
// The projection supplies only the ANGLE (which direction / neighborhood), which
// is arbitrary but neighborhood-preserving.

export function centeredNorms(Xc) {
  return Xc.map((v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)));
}

/** percentile rank in [0,1] of each value (ties broken by order). */
export function percentileRank(vals) {
  const idx = vals.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(vals.length);
  for (let k = 0; k < idx.length; k++) r[idx[k][1]] = idx.length === 1 ? 0.5 : k / (idx.length - 1);
  return r;
}

export function angleOf(coords2d) {
  return coords2d.map((p) => Math.atan2(p[1], p[0]));
}

/** radar (x,y): radius = percentile(distinctiveness), angle = projection angle. */
export function radar(radiusPct, angle, R = 1) {
  return radiusPct.map((pct, i) => [R * pct * Math.cos(angle[i]), R * pct * Math.sin(angle[i])]);
}
