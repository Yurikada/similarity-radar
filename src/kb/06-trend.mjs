// Stage A6 — temporal trend + why alignment is needed even for PCA.
//   Split by date: T1 = older half, T2 = all notes (a growing corpus).
//   Project each snapshot INDEPENDENTLY with PCA. For the common notes (T1),
//   compare their positions across the two projections:
//     raw displacement (no align)  = dominated by the arbitrary frame
//     Procrustes-aligned residual  = the real change (small)
//   The alignment rotation angle ≠ 0 proves PCA's frame shifts across populations.
//
// Run: node src/kb/06-trend.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pca2 } from "../lib/projection/pca.mjs";
import { smacof } from "../lib/projection/mds.mjs";
import { procrustes } from "../lib/procrustes.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const emb = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "kb", "permanent-embeddings.json"), "utf8"));
const V = emb.vectors.map((v) => Float64Array.from(v));
const n = V.length, dim = emb.dim;

function centeredSub(indices) {
  const sub = indices.map((i) => V[i]);
  const c = new Float64Array(dim);
  for (const v of sub) for (let d = 0; d < dim; d++) c[d] += v[d] / sub.length;
  return sub.map((v) => Float64Array.from(v, (x, d) => x - c[d]));
}
const centerAndPca = (indices) => pca2(centeredSub(indices));
function centerAndMds(indices, seed) {
  const Xc = centeredSub(indices), m = Xc.length;
  const D = Array.from({ length: m }, () => new Float64Array(m));
  for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
    let s = 0; for (let k = 0; k < dim; k++) { const d = Xc[i][k] - Xc[j][k]; s += d * d; }
    D[i][j] = D[j][i] = Math.sqrt(s);
  }
  return smacof(D, seed); // independent seeds per snapshot = realistic (no shared init)
}
function normalize(P) {
  const m = P.length; let cx = 0, cy = 0;
  for (const p of P) { cx += p[0] / m; cy += p[1] / m; }
  const Q = P.map((p) => [p[0] - cx, p[1] - cy]);
  let s = 0; for (const p of Q) s += p[0] * p[0] + p[1] * p[1];
  s = Math.sqrt(s);
  return Q.map((p) => [p[0] / s, p[1] / s]);
}
function meanDist(A, B) { let e = 0; for (let i = 0; i < A.length; i++) e += Math.hypot(A[i][0] - B[i][0], A[i][1] - B[i][1]); return e / A.length; }
function bestRotResidual(A, B) {
  // rotate B onto A by optimal angle (allow reflection), return residual meanDist
  let m00 = 0, m01 = 0, m10 = 0, m11 = 0;
  for (let i = 0; i < A.length; i++) { m00 += A[i][0] * B[i][0]; m01 += A[i][0] * B[i][1]; m10 += A[i][1] * B[i][0]; m11 += A[i][1] * B[i][1]; }
  const ang = Math.atan2(m10 - m01, m00 + m11), c = Math.cos(ang), s = Math.sin(ang);
  const rot = (flip) => B.map((p) => { const by = flip ? -p[1] : p[1]; return [c * p[0] - s * by, s * p[0] + c * by]; });
  const r1 = meanDist(A, rot(false)), r2 = meanDist(A, rot(true));
  return { residual: Math.min(r1, r2), angleDeg: (ang * 180) / Math.PI };
}

// split by date (older half vs all)
const order = [...Array(n).keys()].sort((a, b) => (emb.dates[a] < emb.dates[b] ? -1 : 1));
const half = Math.floor(n / 2);
const T1 = order.slice(0, half);          // older notes
const newOnly = order.slice(half);        // newer notes
const isT1 = new Set(T1);

const orderPos = new Map(order.map((idx, k) => [idx, k]));
console.log("=== Stage A6: temporal trend + alignment ===\n");
console.log(`split by date: T1(older)=${T1.length}  newer=${newOnly.length}  (dates ${emb.dates[order[0]]} .. ${emb.dates[order[n - 1]]})\n`);
console.log("common notes (T1) across two INDEPENDENT projections (older-snapshot vs full-snapshot):");
console.log("  method  raw disp.  aligned resid.  frame×   rotation");
const results = {};
for (const [name, projT1, projAll] of [
  ["PCA", centerAndPca(T1), centerAndPca(order)],
  ["MDS", centerAndMds(T1, 11), centerAndMds(order, 22)],
]) {
  const A = normalize(projT1);
  const B = normalize(T1.map((idx) => projAll[orderPos.get(idx)]));
  const raw = meanDist(A, B);
  const al = bestRotResidual(A, B);
  results[name] = { raw, aligned: al.residual, angle: Math.abs(al.angleDeg), A, B };
  console.log(`  ${name.padEnd(6)} ${raw.toFixed(3).padStart(8)}  ${al.residual.toFixed(3).padStart(13)}  ${(raw / al.residual).toFixed(1).padStart(5)}x  ${Math.abs(al.angleDeg).toFixed(1).padStart(7)}°`);
}
console.log("\n  PCA: frame shifts only a little across populations (variance structure stable) but ≠0 → Procrustes still needed for correctness.");
console.log("  MDS: frame is arbitrary (random init) → alignment is essential; raw displacement is mostly frame.\n");
const P2all = centerAndPca(order);

// real trend in the aligned full-snapshot frame: centroid of old vs new notes
const oldXY = order.map((idx, k) => ({ idx, xy: P2all[k] })).filter((o) => isT1.has(o.idx)).map((o) => o.xy);
const newXY = order.map((idx, k) => ({ idx, xy: P2all[k] })).filter((o) => !isT1.has(o.idx)).map((o) => o.xy);
const cen = (P) => { let x = 0, y = 0; for (const p of P) { x += p[0] / P.length; y += p[1] / P.length; } return [x, y]; };
const co = cen(oldXY), cn = cen(newXY);
console.log(`corpus drift (old→new centroid, full-snapshot frame): |Δ| = ${Math.hypot(cn[0] - co[0], cn[1] - co[1]).toFixed(3)}`);

fs.writeFileSync(path.join(ROOT, "out", "kb", "trend.json"),
  JSON.stringify({ results, drift: Math.hypot(cn[0] - co[0], cn[1] - co[1]), T1n: T1.length, newN: newOnly.length }));
console.log("\nsaved: out/kb/trend.json (gitignored)");
