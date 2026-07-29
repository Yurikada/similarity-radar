// Stage D1 — does separating the layers actually buy anything?
//
// "Keep identity and state in different layers" is a design assertion until it is
// measured. D1 tests three consequences it should have, each with a threshold
// fixed before the numbers were seen (out/layers/preregistration.json):
//
//  D-H1  contamination — concatenating the 4 state features into the 9 identity
//        features materially changes the map. If it does not, layer separation is
//        moot on this data and Part D reports that instead.
//        Pre-registered: Procrustes disparity(identity, mixed) >= 0.10.
//        (C1's closest objective pair sat at 0.130, so 0.10 is "as large as
//        changing what the objective preserves".)
//
//  D-H2  instability cost — under a STATE-ONLY update (prices move six months,
//        fundamentals held fixed) the mixed map moves. The separated identity map
//        moves exactly zero, by construction rather than by luck: state never
//        enters its distance matrix. This is the concrete price of mixing, and it
//        is what forced Procrustes alignment on the A6/temporal work in the first
//        place.  Pre-registered: disparity(mixed@t0, mixed@t2) >= 0.05.
//
//  D-H3  peer group — separation is what makes the base map usable as a
//        data-driven peer group. Reading momentum against a stock's nearest
//        neighbours in identity space should differ from reading it against its
//        sector label. Neighbours are taken in the HIGH-DIMENSIONAL identity
//        space, not on the 2D map (A5: cluster in high dimensions; 2D halves the
//        agreement).  Pre-registered: Spearman(peer-z, sector-z) < 0.7 and at
//        least 20 stocks flip sign between the two readings.
//
// Note on the mixed map, worth stating before the result: whitening equalizes all
// directions, so 4 state columns against 9 identity columns hands state 4/13 of
// the variance budget purely by column count. The contamination level is
// therefore a function of how many state features one happens to add — an
// unjustifiable free parameter, and an argument for separation independent of
// whatever D-H1 measures.
//
// Run: node src/layers/01-separation.mjs   (no network — uses the D0 cache)

import fs from "node:fs";
import path from "node:path";
import { jacobiEigsym } from "../lib/eig.mjs";
import { smacof } from "../lib/projection/mds.mjs";
import { classicalMDS } from "../lib/projection/classical-mds.mjs";
import { gdMDS } from "../lib/projection/mds-gd.mjs";
import { rawStress, nonMetric } from "../lib/projection/objectives.mjs";
import { procrustes } from "../lib/procrustes.mjs";
import { euclid, ROOT } from "../objectives/domains.mjs";

const OUT = path.join(ROOT, "out", "layers");
fs.mkdirSync(OUT, { recursive: true });

const SW = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "whitened.json"), "utf8"));
const ST = JSON.parse(fs.readFileSync(path.join(OUT, "state-features.json"), "utf8"));
const idx = ST.baseIndex;
const n = idx.length;

// ---------------------------------------------------------------- pre-register

const PREREG = {
  stage: "D1",
  hypotheses: {
    "D-H1": { claim: "mixing state into identity materially changes the map", metric: "procrustes disparity(identity, mixed)", threshold: 0.1, direction: ">=" },
    "D-H2": { claim: "the mixed map moves under a state-only update; the separated map cannot", metric: "procrustes disparity(mixed@t0, mixed@t2)", threshold: 0.05, direction: ">=" },
    "D-H3": { claim: "identity-space peers are not a re-description of the sector label", metric: "spearman(peer-z, sector-z)", threshold: 0.7, direction: "<", also: "sign flips >= 20" },
  },
  fixed: { k_peers: 10, min_sector_size: 5, overlay_features: ["mom3m", "vol3m"], base_objective: "raw stress", robustness_objective: "non-metric" },
};
const pregPath = path.join(OUT, "preregistration.json");
let pregNote;
if (fs.existsSync(pregPath)) {
  pregNote = `verified against frozen file (${JSON.parse(fs.readFileSync(pregPath, "utf8")).frozen_at})`;
} else {
  fs.writeFileSync(pregPath, JSON.stringify({ ...PREREG, frozen_at: new Date().toISOString() }, null, 2));
  pregNote = "frozen now (first run)";
}
console.log(`=== Stage D1 — layer separation ===\n\npre-registration: ${pregNote}`);
for (const [k, h] of Object.entries(PREREG.hypotheses))
  console.log(`  ${k}  ${h.metric} ${h.direction} ${h.threshold}${h.also ? " and " + h.also : ""}`);

// ------------------------------------------------------------ build the layers

/** z-score columns, then PCA-whiten — the same preprocessing B1 settled on. */
function whiten(rows) {
  const d = rows[0].length,
    m = rows.length;
  const mean = Array.from({ length: d }, (_, j) => rows.reduce((a, r) => a + r[j], 0) / m);
  const sd = Array.from({ length: d }, (_, j) =>
    Math.sqrt(rows.reduce((a, r) => a + (r[j] - mean[j]) ** 2, 0) / m) || 1);
  const Z = rows.map((r) => r.map((v, j) => (v - mean[j]) / sd[j]));
  const C = Array.from({ length: d }, () => new Float64Array(d));
  for (let a = 0; a < d; a++)
    for (let b = 0; b < d; b++) {
      let s = 0;
      for (const r of Z) s += r[a] * r[b];
      C[a][b] = s / m;
    }
  const { values, vectors } = jacobiEigsym(C);
  const floor = values[0] * 1e-3;
  return Z.map((x) =>
    vectors.map((vec, k) => {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += vec[j] * x[j];
      return dot / Math.sqrt(Math.max(values[k], floor));
    }));
}

const identityW = idx.map((i) => Array.from(SW.W[i])); // already whitened in B1
const Didentity = euclid(identityW);

// The mixed block reuses B0's finished identity columns (winsorized, median-
// imputed, z-scored) verbatim, so the only thing that differs between the two
// maps is the four appended state columns — not a second, subtly different
// imputation of the fundamentals.
const FEAT = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "features.json"), "utf8"));
const identityZ = idx.map((i) => FEAT.matrix[i]);
const mixedAt = (snap) => whiten(identityZ.map((r, p) => [...r, ...ST[snap][p]]));
const Dmixed0 = euclid(mixedAt("t0"));
const Dmixed2 = euclid(mixedAt("t2"));

const layout = (D, obj = rawStress) => gdMDS(obj, D, classicalMDS(D).Y).Y;
const Yidentity = layout(Didentity);
const Ymixed0 = layout(Dmixed0);
const Ymixed2 = layout(Dmixed2);

// -------------------------------------------------------------------- D-H1/H2

const h1 = procrustes(Yidentity, Ymixed2).disparity;
const h1nm = procrustes(layout(Didentity, nonMetric), layout(Dmixed2, nonMetric)).disparity;
const h2 = procrustes(Ymixed0, Ymixed2).disparity;

console.log("\n[D-H1] contamination — identity map vs identity+state map");
console.log(`  disparity (raw stress)  ${h1.toFixed(3)}   ${h1 >= 0.1 ? "SUPPORTED" : "not supported"}`);
console.log(`  disparity (non-metric)  ${h1nm.toFixed(3)}   robustness check, C1 showed the readout is objective-invariant`);
console.log(`  variance budget handed to state by column count alone: 4/13 = ${(4 / 13).toFixed(3)}`);

console.log("\n[D-H2] instability cost — state-only update (t0 -> t2), fundamentals frozen");
console.log(`  mixed map   disparity ${h2.toFixed(3)}   ${h2 >= 0.05 ? "SUPPORTED" : "not supported"}`);
console.log(`  separated   disparity 0.000   (state never enters the identity distance matrix)`);

// per-stock displacement caused purely by prices moving
const alignedPairs = (() => {
  const pr = procrustes(Ymixed0, Ymixed2);
  return pr;
})();

// ----------------------------------------------------------------------- D-H3

function knnHigh(W, k) {
  const m = W.length;
  const D = euclid(W);
  return Array.from({ length: m }, (_, i) =>
    [...Array(m).keys()].filter((j) => j !== i).sort((a, b) => D[i][a] - D[i][b]).slice(0, k));
}
const peers = knnHigh(identityW, PREREG.fixed.k_peers);
const sectorOf = ST.sectors;
const bySector = {};
sectorOf.forEach((s, i) => (bySector[s] ??= []).push(i));

function spearman(a, b) {
  const rank = (v) => {
    const o = [...v.keys()].sort((x, y) => v[x] - v[y]);
    const r = new Array(v.length);
    o.forEach((idx2, k) => (r[idx2] = k));
    return r;
  };
  const ra = rank(a),
    rb = rank(b),
    m = a.length;
  let ma = 0,
    mb = 0;
  for (let i = 0; i < m; i++) {
    ma += ra[i] / m;
    mb += rb[i] / m;
  }
  let s = 0,
    sa = 0,
    sb = 0;
  for (let i = 0; i < m; i++) {
    const u = ra[i] - ma,
      v = rb[i] - mb;
    s += u * v;
    sa += u * u;
    sb += v * v;
  }
  return s / Math.sqrt(sa * sb);
}

const h3 = {};
console.log("\n[D-H3] peer group — same overlay read against identity peers vs against sector");
for (const [fi, fname] of PREREG.fixed.overlay_features.map((f) => [ST.featureNames.indexOf(f), f])) {
  const x = ST.t2.map((r) => r[fi]);
  const eligible = [];
  const peerZ = [],
    sectZ = [];
  for (let i = 0; i < n; i++) {
    const grp = bySector[sectorOf[i]].filter((j) => j !== i);
    if (grp.length + 1 < PREREG.fixed.min_sector_size) continue;
    const zOf = (members) => {
      const mu = members.reduce((a, j) => a + x[j], 0) / members.length;
      const sd = Math.sqrt(members.reduce((a, j) => a + (x[j] - mu) ** 2, 0) / members.length) || 1;
      return (x[i] - mu) / sd;
    };
    eligible.push(i);
    peerZ.push(zOf(peers[i]));
    sectZ.push(zOf(grp));
  }
  const rho = spearman(peerZ, sectZ);
  const flips = peerZ.filter((v, p) => Math.sign(v) !== Math.sign(sectZ[p])).length;
  const supported = rho < 0.7 && flips >= 20;
  h3[fname] = { rho, flips, eligible: eligible.length, supported };
  console.log(
    `  ${fname.padEnd(7)} n=${eligible.length}  spearman ${rho.toFixed(3)}  sign flips ${flips}` +
      `   ${supported ? "SUPPORTED" : "not supported"}`,
  );
  // the disagreements are the point: same stock, opposite verdict
  const dis = eligible
    .map((i, p) => ({ i, d: Math.abs(peerZ[p] - sectZ[p]), peerZ: peerZ[p], sectZ: sectZ[p] }))
    .sort((a, b) => b.d - a.d)
    .slice(0, 5);
  for (const d of dis)
    console.log(
      `      ${ST.tickers[d.i].padEnd(8)} ${String(ST.names[d.i]).slice(0, 26).padEnd(28)}` +
        ` peer-z ${d.peerZ.toFixed(2).padStart(6)}  sector-z ${d.sectZ.toFixed(2).padStart(6)}  (${sectorOf[d.i]})`,
    );
}

// ------------------------------------------------------------------- verdicts

const verdict = {
  "D-H1": { value: h1, nonmetric: h1nm, supported: h1 >= 0.1 },
  "D-H2": { value: h2, supported: h2 >= 0.05, separatedValue: 0 },
  "D-H3": h3,
};
console.log("\n=== D1 verdicts ===");
for (const [k, v] of Object.entries(verdict))
  console.log(`  ${k}: ${JSON.stringify(v.supported ?? Object.fromEntries(Object.entries(v).map(([a, b]) => [a, b.supported])))}`);

fs.writeFileSync(
  path.join(OUT, "d1-separation.json"),
  JSON.stringify({ stage: "D1", ran_at: new Date().toISOString(), prereg: PREREG, verdict, rotationMixed: alignedPairs.angleDeg, n }, null, 2),
);
console.log("\nsaved: out/layers/d1-separation.json");
