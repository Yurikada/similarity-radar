// Stage C0 — calibration + pre-registration for the objective-function comparison.
//
// Two things happen here, both BEFORE any alternative objective is implemented.
//
// (1) PRE-REGISTRATION. Measure the coefficient of variation CV = sd(d)/mean(d)
//     of the distance matrix each domain actually feeds to MDS, and freeze the
//     hypothesis "the effect size of choosing an objective grows with CV".
//     Recording CV now means the C1 result cannot be re-explained after the fact.
//     Re-runs never overwrite the frozen file; they re-verify it.
//
// (2) CALIBRATION. Solve raw stress two ways — the existing SMACOF majorization
//     and the new generic gradient descent — and check they land on the same map.
//     Pass thresholds were fixed before running:
//         Procrustes disparity < 0.01
//         |Δ stress-1|         < 0.005
//         |Δ trustworthiness|  < 0.01
//         |Δ continuity|       < 0.01
//     If this fails, C1 does not run: any difference between objectives could
//     then be the optimizer rather than the objective.
//
// Run: node src/objectives/00-calibration.mjs   (no network)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { smacof } from "../lib/projection/mds.mjs";
import { gdMDS } from "../lib/projection/mds-gd.mjs";
import { rawStress } from "../lib/projection/objectives.mjs";
import { classicalMDS } from "../lib/projection/classical-mds.mjs";
import { procrustes } from "../lib/procrustes.mjs";
import { projectionMetrics } from "../lib/metrics.mjs";
import { mulberry32, gaussian } from "../lib/rng.mjs";
import { loadDomains, pairStats, ROOT } from "./domains.mjs";

const OUT = path.join(ROOT, "out", "objectives");
fs.mkdirSync(OUT, { recursive: true });

const SMACOF_ITERS = 3000; // well past the 300 used in A2 — convergence, not budget
const K_NN = 10; // neighborhood size for trust/continuity (same as elsewhere)

// ---------------------------------------------------------------- load domains

const DOMAINS = loadDomains();

// ------------------------------------------------------- (1) pre-registration

const prereg = {
  stage: "C0",
  frozen_at: new Date().toISOString(),
  hypothesis:
    "The effect size of choosing a stress objective (max pairwise Procrustes disparity across " +
    "the four C1 objectives) is a monotone increasing function of CV = sd(d)/mean(d) of the " +
    "distance matrix fed to MDS. Prediction on record: CV(kb) < CV(stocks), therefore " +
    "effect(kb) < effect(stocks). Not to be revised after seeing C1 maps.",
  objectives_frozen: ["raw stress", "log-distance", "kNN-weighted", "non-metric (monotone)"],
  sensitivity_frozen: { knn_gamma: [0.01, 0.05, 0.1], knn_k: [5, 10, 20] },
  pass_thresholds: { disparity: 0.01, stress1: 0.005, trust: 0.01, cont: 0.01 },
  distance_stats: Object.fromEntries(DOMAINS.map((d) => [d.key, pairStats(d.D)])),
};

const preregPath = path.join(OUT, "preregistration.json");
let preregNote;
if (fs.existsSync(preregPath)) {
  const old = JSON.parse(fs.readFileSync(preregPath, "utf8"));
  const same = DOMAINS.every(
    (d) =>
      Math.abs(old.distance_stats[d.key].cv - prereg.distance_stats[d.key].cv) < 1e-9,
  );
  preregNote = same
    ? `verified against frozen file (${old.frozen_at}) — CV unchanged`
    : `!! MISMATCH against frozen file (${old.frozen_at}) — inputs changed since pre-registration`;
} else {
  fs.writeFileSync(preregPath, JSON.stringify(prereg, null, 2));
  preregNote = "frozen now (first run)";
}

console.log("=== Stage C0 — pre-registration ===\n");
console.log(`  ${preregNote}\n`);
console.log("  domain   n    pairs     mean      sd       CV      min     median    max");
for (const d of DOMAINS) {
  const s = prereg.distance_stats[d.key];
  console.log(
    `  ${d.key.padEnd(7)} ${String(s.n).padStart(3)} ${String(s.pairs).padStart(6)}` +
      `  ${s.mean.toFixed(4).padStart(8)} ${s.sd.toFixed(4).padStart(8)}` +
      ` ${s.cv.toFixed(4).padStart(7)} ${s.min.toFixed(3).padStart(7)}` +
      ` ${s.p50.toFixed(3).padStart(8)} ${s.max.toFixed(3).padStart(7)}`,
  );
}
const cvStocks = prereg.distance_stats.stocks.cv,
  cvKb = prereg.distance_stats.kb.cv;
console.log(
  `\n  CV(kb)=${cvKb.toFixed(4)} vs CV(stocks)=${cvStocks.toFixed(4)}  ->  ` +
    (cvKb < cvStocks
      ? "prediction-2 direction HOLDS on the input side"
      : "prediction-2 direction is REVERSED on the input side"),
);

// ------------------------------------------------------------ (2) calibration

function randomInit(n, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => [gaussian(rng), gaussian(rng)]);
}

const T = prereg.pass_thresholds;
const results = [];

console.log("\n\n=== Stage C0 — SMACOF vs generic GD on raw stress ===");
for (const dom of DOMAINS) {
  const n = dom.D.length;
  const inits = [
    { name: "classical-MDS warm start", Y: classicalMDS(dom.D).Y },
    { name: "random (seed 1)", Y: randomInit(n, 1) },
  ];

  console.log(`\n${dom.label}`);
  console.log(
    "  init                       rawStress(SMACOF)  rawStress(GD)   disparity  Δstress1   Δtrust    Δcont   verdict",
  );

  for (const init of inits) {
    const Ys = smacof(dom.D, 1, SMACOF_ITERS, init.Y);
    const g = gdMDS(rawStress, dom.D, init.Y);

    const vs = rawStress.valueAndGrad(Ys, dom.D).value;
    const vg = g.value;

    const pr = procrustes(Ys, g.Y);
    const ms = projectionMetrics(dom.D, Ys, K_NN);
    const mg = projectionMetrics(dom.D, g.Y, K_NN);

    const d = {
      disparity: pr.disparity,
      stress1: Math.abs(ms.stress1 - mg.stress1),
      trust: Math.abs(ms.trust - mg.trust),
      cont: Math.abs(ms.cont - mg.cont),
      corr: Math.abs(ms.corr - mg.corr),
    };
    const pass =
      d.disparity < T.disparity &&
      d.stress1 < T.stress1 &&
      d.trust < T.trust &&
      d.cont < T.cont;

    console.log(
      `  ${init.name.padEnd(26)} ${vs.toFixed(4).padStart(16)} ${vg.toFixed(4).padStart(14)}` +
        `  ${d.disparity.toExponential(2).padStart(9)} ${d.stress1.toExponential(2).padStart(9)}` +
        ` ${d.trust.toExponential(2).padStart(9)} ${d.cont.toExponential(2).padStart(9)}` +
        `   ${pass ? "PASS" : "FAIL"}`,
    );

    results.push({
      domain: dom.key,
      init: init.name,
      smacof: { rawStress: vs, ...ms },
      gd: { rawStress: vg, ...mg, iters: g.iters, evals: g.evals, converged: g.converged, gradNorm: g.gradNorm },
      procrustes: { disparity: pr.disparity, angleDeg: pr.angleDeg, reflected: pr.reflected },
      deltas: d,
      pass,
    });
  }
}

// warm start should reproduce ORIENTATION too (shared init, no randomness left)
const warm = results.filter((r) => r.init.startsWith("classical"));
console.log("\n  orientation check (warm start shares one init -> rotation should be ~0°):");
for (const r of warm)
  console.log(
    `    ${r.domain.padEnd(7)} rotation ${Math.abs(r.procrustes.angleDeg).toFixed(3)}°` +
      `  reflected=${r.procrustes.reflected}  GD iters=${r.gd.iters} evals=${r.gd.evals} converged=${r.gd.converged}`,
  );

const allPass = results.every((r) => r.pass);
fs.writeFileSync(
  path.join(OUT, "c0-calibration.json"),
  JSON.stringify({ stage: "C0", ran_at: new Date().toISOString(), thresholds: T, results, allPass }, null, 2),
);

console.log(
  `\n=== C0 verdict: ${allPass ? "PASS — differences in C1 are attributable to the objective" : "FAIL — do not proceed to C1"} ===`,
);
console.log("saved: out/objectives/preregistration.json, out/objectives/c0-calibration.json");
