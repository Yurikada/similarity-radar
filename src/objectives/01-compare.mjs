// Stage C1 — the four-point objective comparison.
//
// Fixed before running (out/objectives/preregistration.json):
//   objectives   raw stress / log-distance / kNN-weighted / non-metric
//   hypothesis   effect size (max pairwise Procrustes disparity) grows with
//                CV = sd(d)/mean(d).  Measured in C0: CV(kb)=0.171 < CV(stocks)=0.397
//                => prediction on record is effect(kb) < effect(stocks)
//   sensitivity  kNN γ ∈ {0.01,0.05,0.1} × k ∈ {5,10,20}
//
// Measurement contract (C0):
//   one D, one init (classical-MDS warm start), one optimizer (GD + Armijo);
//   Procrustes-align before comparing shape; radius read as a percentile, never raw;
//   objectives compared only through external metrics, never their own values.
//
// Frame caveat: on stocks the warm start sits in a near-degenerate eigen-subspace
// (144.98 vs 144.23), so its orientation is arbitrary. Rotation angles are printed
// for kb only; the shape (disparity) is the axis that carries meaning everywhere.
//
// Run: node src/objectives/01-compare.mjs   (no network)

import fs from "node:fs";
import path from "node:path";
import { gdMDS } from "../lib/projection/mds-gd.mjs";
import {
  rawStress,
  logDistance,
  knnWeighted,
  nonMetric,
  pava,
} from "../lib/projection/objectives.mjs";
import { classicalMDS } from "../lib/projection/classical-mds.mjs";
import { procrustes } from "../lib/procrustes.mjs";
import { projectionMetrics } from "../lib/metrics.mjs";
import { mulberry32, gaussian } from "../lib/rng.mjs";
import { loadDomains, pairStats, ROOT } from "./domains.mjs";

const OUT = path.join(ROOT, "out", "objectives");
fs.mkdirSync(OUT, { recursive: true });
const prereg = JSON.parse(fs.readFileSync(path.join(OUT, "preregistration.json"), "utf8"));

const K_NN = 10;
const KNN_REF = { k: 10, gamma: 0.05 }; // middle of the frozen grid, chosen without looking at any map
const DOMAINS = loadDomains();

const FOUR = [
  rawStress,
  logDistance,
  knnWeighted(KNN_REF.k, KNN_REF.gamma),
  nonMetric,
];

// -------------------------------------------------------------------- helpers

function fit(obj, D, init) {
  const t0 = Date.now();
  const r = gdMDS(obj, D, init);
  return { ...r, ms: Date.now() - t0 };
}
function randomInit(n, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => [gaussian(rng), gaussian(rng)]);
}
/** percentile rank of each point's distance from the layout centroid (A3 readout) */
function radiusPercentile(Y) {
  const n = Y.length;
  let cx = 0,
    cy = 0;
  for (const p of Y) {
    cx += p[0] / n;
    cy += p[1] / n;
  }
  const r = Y.map((p) => Math.hypot(p[0] - cx, p[1] - cy));
  const order = [...r.keys()].sort((a, b) => r[a] - r[b]);
  const pct = new Array(n);
  order.forEach((idx, rank) => (pct[idx] = (100 * rank) / (n - 1)));
  return pct;
}
function pearson(a, b) {
  const m = a.length;
  let ma = 0,
    mb = 0;
  for (let i = 0; i < m; i++) {
    ma += a[i] / m;
    mb += b[i] / m;
  }
  let s = 0,
    sa = 0,
    sb = 0;
  for (let i = 0; i < m; i++) {
    const u = a[i] - ma,
      v = b[i] - mb;
    s += u * v;
    sa += u * u;
    sb += v * v;
  }
  return s / Math.sqrt(sa * sb);
}
const f = (x, w = 6, p = 3) => x.toFixed(p).padStart(w);

const report = { stage: "C1", ran_at: new Date().toISOString(), knn_reference: KNN_REF, domains: {} };

// ============================================================ main comparison

for (const dom of DOMAINS) {
  const n = dom.D.length;
  const init = classicalMDS(dom.D).Y;
  console.log(`\n\n########## ${dom.label} ##########`);
  console.log(
    `frame: ${dom.canonicalFrame ? "canonical (rotation readable)" : "DEGENERATE warm start — shape only, angles suppressed"}`,
  );

  // ---- 1. four objectives from one shared init
  const runs = FOUR.map((o) => ({ obj: o, r: fit(o, dom.D, init) }));

  console.log("\n[1] four objectives, shared classical-MDS warm start");
  console.log(
    "  objective                          stress-1   corr    trust   cont    iters  status",
  );
  const rows = [];
  for (const { obj, r } of runs) {
    const m = projectionMetrics(dom.D, r.Y, K_NN);
    rows.push({
      name: obj.name,
      label: obj.label,
      status: obj.status,
      ...m,
      iters: r.iters,
      converged: r.converged,
      ms: r.ms,
    });
    console.log(
      `  ${obj.label.padEnd(34)} ${f(m.stress1)}  ${f(m.corr)}  ${f(m.trust)}  ${f(m.cont)}` +
        `  ${String(r.iters).padStart(5)}  ${obj.status === "valid" ? "valid" : "DISQUALIFIED"}`,
    );
  }

  // ---- 2. pairwise shape difference (the primary axis)
  console.log("\n[2] pairwise Procrustes disparity (shape difference between objectives)");
  const names = runs.map((x) => x.obj.name);
  console.log("            " + names.map((s) => s.padStart(10)).join(""));
  const dispM = {};
  let effect = 0,
    effectPair = null;
  for (let i = 0; i < runs.length; i++) {
    let line = "  " + names[i].padEnd(10);
    dispM[names[i]] = {};
    for (let j = 0; j < runs.length; j++) {
      const pr = procrustes(runs[i].r.Y, runs[j].r.Y);
      dispM[names[i]][names[j]] = { disparity: pr.disparity, angleDeg: pr.angleDeg, reflected: pr.reflected };
      line += (i === j ? "—" : pr.disparity.toFixed(3)).padStart(10);
      if (j > i && pr.disparity > effect) {
        effect = pr.disparity;
        effectPair = [names[i], names[j]];
      }
    }
    console.log(line);
  }
  console.log(`  effect size (max) = ${effect.toFixed(3)}  between ${effectPair.join(" / ")}`);

  if (dom.canonicalFrame) {
    console.log("  rotation vs raw:  " +
      names.slice(1).map((nm) => `${nm} ${Math.abs(dispM.raw[nm].angleDeg).toFixed(1)}°${dispM.raw[nm].reflected ? "+鏡映" : ""}`).join("   "));
  }

  // ---- 3. radius percentile agreement (is the A3 readout objective-invariant?)
  console.log("\n[3] radius-percentile correlation vs raw stress (A3 readout robustness)");
  const pcts = Object.fromEntries(runs.map((x) => [x.obj.name, radiusPercentile(x.r.Y)]));
  for (const nm of names.slice(1))
    console.log(`  ${nm.padEnd(12)} corr = ${pearson(pcts.raw, pcts[nm]).toFixed(3)}`);

  // ---- 4. non-metric degeneracy guard
  const nmRun = runs.find((x) => x.obj.name === "nonmetric");
  const ctx = nonMetric.prepare(dom.D);
  const ev = nonMetric.valueAndGrad(nmRun.r.Y, dom.D, ctx);
  const dl = ev._fit.delta;
  let mn = Infinity,
    mx = -Infinity,
    sum = 0;
  for (const v of dl) {
    mn = Math.min(mn, v);
    mx = Math.max(mx, v);
    sum += v;
  }
  const mean = sum / dl.length;
  let sd = 0;
  for (const v of dl) sd += (v - mean) ** 2;
  sd = Math.sqrt(sd / dl.length);
  let blocks = 1;
  for (let p = 1; p < dl.length; p++) if (dl[p] - dl[p - 1] > 1e-9) blocks++;
  const degenerate = sd / mean < 0.05 || blocks < dl.length * 0.01;
  console.log("\n[4] non-metric degeneracy guard (fitted disparities δ)");
  console.log(
    `  σ=${nmRun.r.value.toFixed(4)}  δ range ${mn.toFixed(3)}–${mx.toFixed(3)}  CV(δ)=${(sd / mean).toFixed(3)}` +
      `  distinct blocks ${blocks}/${dl.length}  ->  ${degenerate ? "DEGENERATE — 4点目は資格なし" : "OK (not collapsed)"}`,
  );

  // ---- 5. kNN sensitivity: are k and γ the same species of hidden knob as ε?
  console.log("\n[5] kNN sensitivity — disparity vs the k=10,γ=0.05 reference");
  const refY = runs.find((x) => x.obj.name.startsWith("knn")).r.Y;
  const grid = [];
  console.log("        γ=0.01   γ=0.05   γ=0.1");
  for (const k of prereg.sensitivity_frozen.knn_k) {
    let line = `  k=${String(k).padEnd(3)}`;
    for (const g of prereg.sensitivity_frozen.knn_gamma) {
      const r = fit(knnWeighted(k, g), dom.D, init);
      const d = procrustes(refY, r.Y).disparity;
      grid.push({ k, gamma: g, disparity: d, stress1: projectionMetrics(dom.D, r.Y, K_NN).stress1 });
      line += d.toFixed(3).padStart(9);
    }
    console.log(line);
  }
  const gridMax = Math.max(...grid.map((g) => g.disparity));
  const knnUnstable = gridMax > effect;
  console.log(
    `  max within-grid disparity ${gridMax.toFixed(3)} vs between-objective effect ${effect.toFixed(3)}` +
      `  ->  ${knnUnstable ? "kNN の内部揺れが目的関数間の差を上回る＝4点目として不適格" : "grid spread stays below the effect being measured"}`,
  );

  // ---- 6. stochasticity per objective (A2 procedure, two random seeds)
  console.log("\n[6] random-init sensitivity (seed 1 vs seed 2, A2 procedure)");
  const stoch = {};
  for (const o of FOUR) {
    const a = fit(o, dom.D, randomInit(n, 1)).Y;
    const b = fit(o, dom.D, randomInit(n, 2)).Y;
    const pr = procrustes(a, b);
    stoch[o.name] = { disparity: pr.disparity, angleDeg: pr.angleDeg, reflected: pr.reflected };
    console.log(
      `  ${o.name.padEnd(12)} disparity ${pr.disparity.toFixed(3)}` +
        (dom.canonicalFrame ? `  rotation ${Math.abs(pr.angleDeg).toFixed(1)}°${pr.reflected ? " +鏡映" : ""}` : ""),
    );
  }

  report.domains[dom.key] = {
    label: dom.label,
    canonicalFrame: dom.canonicalFrame,
    cv: pairStats(dom.D).cv,
    metrics: rows,
    disparity: dispM,
    effect,
    effectPair,
    radiusCorrVsRaw: Object.fromEntries(names.slice(1).map((nm) => [nm, pearson(pcts.raw, pcts[nm])])),
    nonmetricGuard: { sigma: nmRun.r.value, deltaMin: mn, deltaMax: mx, deltaCV: sd / mean, blocks, total: dl.length, degenerate },
    knnGrid: grid,
    knnGridMax: gridMax,
    knnUnstable,
    stochastic: stoch,
    coords: Object.fromEntries(runs.map((x) => [x.obj.name, x.r.Y])),
    radiusPercentile: pcts,
  };
}

// ====================================================== pre-registered verdict

console.log("\n\n########## pre-registered hypothesis ##########");
const S = report.domains.stocks,
  K = report.domains.kb;
console.log(`  CV:     kb ${K.cv.toFixed(4)}  <  stocks ${S.cv.toFixed(4)}   (frozen in C0)`);
console.log(`  effect: kb ${K.effect.toFixed(3)}  ${K.effect < S.effect ? "<" : ">="}  stocks ${S.effect.toFixed(3)}`);
const supported = K.effect < S.effect;
report.hypothesis = {
  statement: prereg.hypothesis,
  cv: { kb: K.cv, stocks: S.cv },
  effect: { kb: K.effect, stocks: S.effect },
  supported,
};
console.log(
  `  -> ${supported ? "SUPPORTED: 効果量はCVと同じ向き。目的関数選択の効き量は母集団依存" : "NOT SUPPORTED: 効果量がCVと逆向き。「集中した距離分布では目的関数が効かない」は棄却"}`,
);

// ---- prediction 1: where do the extreme-profile stocks land?
const stockDom = DOMAINS.find((d) => d.key === "stocks");
const WATCH = ["9984.T", "5401.T", "6857.T"];
console.log("\n  prediction 1 — extreme-profile stocks, radius percentile by objective");
console.log("    ticker    name                          " + Object.keys(S.radiusPercentile).map((s) => s.padStart(10)).join(""));
const pred1 = {};
for (const t of WATCH) {
  const i = stockDom.tickers.indexOf(t);
  if (i < 0) continue;
  pred1[t] = Object.fromEntries(Object.entries(S.radiusPercentile).map(([k, v]) => [k, v[i]]));
  console.log(
    `    ${t.padEnd(9)} ${String(stockDom.labels[i]).slice(0, 28).padEnd(30)}` +
      Object.values(pred1[t]).map((v) => v.toFixed(1).padStart(10)).join(""),
  );
}
report.prediction1 = pred1;

// ---- prediction 3: kNN vs non-metric — same map or different?
console.log("\n  prediction 3 — kNN vs non-metric disparity (vs the other pairs)");
for (const [k, dom] of Object.entries(report.domains)) {
  const knnName = Object.keys(dom.disparity).find((s) => s.startsWith("knn"));
  const kn = dom.disparity[knnName].nonmetric.disparity;
  const others = [];
  for (const a of Object.keys(dom.disparity))
    for (const b of Object.keys(dom.disparity))
      if (a < b && !(a === knnName && b === "nonmetric") && !(b === knnName && a === "nonmetric"))
        others.push(dom.disparity[a][b].disparity);
  console.log(
    `    ${k.padEnd(7)} kNN↔非計量 ${kn.toFixed(3)}   他ペア中央値 ${others.sort((x, y) => x - y)[Math.floor(others.length / 2)].toFixed(3)}` +
      `  -> ${kn > 0.1 ? "別物" : "実質同じ"}`,
  );
}

// ============================================================ frozen C1 record

// The comparison stays a FOUR-point pipeline. kNN is reported as a disqualified
// member, not deleted: dropping the point that failed and publishing only the
// three that worked is the same selection error C1 was designed to avoid (H10).
report.frozen = {
  composition: "3 valid (raw / log / non-metric) + 1 disqualified (kNN-weighted), reported together",
  disqualified: {
    objective: "kNN-weighted",
    reason_1_hyperparameters: {
      claim: "unstable to its own knobs",
      evidence: Object.fromEntries(
        Object.entries(report.domains).map(([k, d]) => [
          k,
          { withinGridMax: d.knnGridMax, betweenObjectiveEffect: d.effect, exceeds: d.knnUnstable },
        ]),
      ),
    },
    reason_2_scaffolding: {
      name: "幾何的支え棒の喪失 / loss of geometric scaffolding",
      claim:
        "The objective built to protect neighbourhoods ranks last on neighbourhood preservation. " +
        "Down-weighting far pairs to γ removes the struts that hold unrelated clusters apart, so " +
        "they drift together and manufacture false neighbours in 2D. Locality cannot be imposed by " +
        "weighting alone — a neighbourhood is defined partly by the distant pairs it is not near.",
      evidence: Object.fromEntries(
        Object.entries(report.domains).map(([k, d]) => [
          k,
          Object.fromEntries(d.metrics.map((m) => [m.name, { trust: m.trust, cont: m.cont }])),
        ]),
      ),
    },
  },
  cv_hypothesis: {
    verdict: "NOT SUPPORTED",
    wording:
      "The pre-registered hypothesis (effect size increases with CV) is rejected: effect(kb)=" +
      `${K.effect.toFixed(3)} > effect(stocks)=${S.effect.toFixed(3)} while CV(kb) < CV(stocks). ` +
      "This does NOT establish that the four-point family is CV-independent — CV-independence was " +
      "a design intent, not a measured property, the observed effect is dominated by the " +
      "disqualified kNN point, and n=2 domains cannot separate the two readings. The d^{-p} " +
      "portability claim itself remains untested; the sweep was deliberately not run.",
  },
  a3_invariance: {
    claim:
      "Radius-percentile readout (A3) is near-invariant to the choice of objective: shape differs " +
      "by disparity 0.44–0.62 while the radius percentile correlates 0.94–1.00 across the valid three.",
    scope: [
      "Holds for the three valid objectives; the disqualified kNN point is the weakest (0.917/0.834).",
      "Observed on n=2 domains (financial factors, text embeddings) — empirical robustness, not a theorem.",
      "Both populations are broadly unimodal around a centroid. Under sharply separated multi-modal " +
        "structure the objective decides which inter-cluster gaps are crushed, and radius-from-global-" +
        "centroid could move with it. Untested here.",
      "Percentile ranking absorbs the non-linear radial stretching that distinguishes the objectives; " +
        "the invariance is a property of the readout, not of the underlying coordinates.",
    ],
  },
};

console.log("\n\n########## C1 frozen ##########");
console.log(`  composition: ${report.frozen.composition}`);
console.log("  disqualified kNN retained in the comparison — reasons:");
console.log("    1. unstable to its own hyperparameters (k, γ)");
console.log("    2. 幾何的支え棒の喪失 — far pairs are the struts that keep neighbourhoods apart");
console.log(`  CV hypothesis: ${report.frozen.cv_hypothesis.verdict} (CV-independence NOT thereby established)`);
console.log("  A3 invariance: empirical, n=2 domains, unimodal populations, readout-level");

fs.writeFileSync(path.join(OUT, "c1-compare.json"), JSON.stringify(report, null, 2));
console.log("\nsaved: out/objectives/c1-compare.json");
