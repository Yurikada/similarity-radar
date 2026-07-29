// Stage D5 — does the peer-relative reading predict anything?
//
// Pre-registered in source before running. The composed radar makes a screening
// claim implicitly: "extreme against its identity peers" is supposed to mean
// something. This stage checks, and is written to make a null result reportable
// rather than embarrassing. The honest prior is that three-month momentum in a
// 145-name large-cap universe carries little or no forward information, and the
// point of measuring is that the radar should not imply otherwise.
//
//   F-H1  peer-relative momentum z ranks forward 63-day returns:
//         |mean Spearman across snapshots| >= 0.05 AND the sign is consistent in
//         at least 2/3 of snapshots.
//   F-H2  it adds something over the free covariate. Raw momentum is available
//         without any of this machinery, so peer-z must beat it:
//         |mean rho(peer-z)| > |mean rho(raw mom3m)|.   (H7b: always keep the
//         cheap covariate in the control column.)
//   F-H3  the cohesion guard is doing work: |mean rho| on unflagged stocks
//         exceeds |mean rho| on flagged ones.
//
// Deliberately NOT claimed: any p-value. Snapshots step 21 trading days while the
// forward window is 63, so consecutive observations overlap threefold and the
// effective sample is far smaller than 26 x 144. Sign consistency across
// snapshots is reported instead, and the weights are frozen: nothing here is
// re-tuned after seeing the outcome (H10).
//
// Run: node src/layers/05-forward.mjs   (no network)

import fs from "node:fs";
import path from "node:path";
import { lsa, TOKENIZERS } from "../lib/text/lsa.mjs";
import { peerGroups, peerCohesion, peerReading } from "../lib/peer.mjs";
import { euclid, ROOT } from "../objectives/domains.mjs";

export const F_PREREG = {
  stage: "D5",
  horizon_primary: 63,
  horizons: [21, 63],
  signal: "peer-relative momentum z (mom3m against 10 identity-space peers)",
  controls: ["sector-relative z", "raw mom3m"],
  hypotheses: {
    "F-H1": { metric: "|mean spearman(peer-z, fwd63)|", threshold: 0.05, plus: "sign consistent in >= 2/3 of snapshots" },
    "F-H2": { metric: "|mean rho(peer-z)| > |mean rho(raw mom3m)|" },
    "F-H3": { metric: "|mean rho| unflagged > |mean rho| flagged" },
  },
  not_claimed: "no p-value: 21-day steps with a 63-day window overlap threefold",
};

const K_PEERS = 10;
const C = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "text", "corpus.json"), "utf8"));
const SS = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "layers", "state-series.json"), "utf8"));

const posInCorpus = new Map(C.docs.map((d, p) => [d.base, p]));
const joint = SS.baseIndex.map((b, p) => ({ state: p, text: posInCorpus.get(b) })).filter((r) => r.text !== undefined);
const n = joint.length;

// identity layer — identical construction to D2
const full = lsa(C.docs.map((d) => d.text), TOKENIZERS.word);
const X = joint.map((r) => full.X[r.text]);
const D = euclid(X);
const peers = peerGroups(D, K_PEERS);
const cohesion = peerCohesion(D, peers);

const fi = SS.featureNames.indexOf("mom3m");
const sectors = joint.map((r) => SS.sectors[r.state]);
const bySector = {};
sectors.forEach((s, i) => (bySector[s] ??= []).push(i));

function spearman(a, b) {
  const m = a.length;
  if (m < 8) return null;
  const rank = (v) => {
    const o = [...v.keys()].sort((x, y) => v[x] - v[y]);
    const r = new Array(v.length);
    o.forEach((i, k) => (r[i] = k));
    return r;
  };
  const ra = rank(a), rb = rank(b);
  let ma = 0, mb = 0;
  for (let i = 0; i < m; i++) { ma += ra[i] / m; mb += rb[i] / m; }
  let s = 0, sa = 0, sb = 0;
  for (let i = 0; i < m; i++) {
    const u = ra[i] - ma, v = rb[i] - mb;
    s += u * v; sa += u * u; sb += v * v;
  }
  return s / Math.sqrt(sa * sb);
}
const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
const pick = (v, ix) => ix.map((i) => v[i]);

const rows = [];
for (const snap of SS.snapshots) {
  const mom = joint.map((r) => snap.features[r.state][fi]);
  const reading = peerReading(mom, peers, cohesion);
  const sectorZ = mom.map((_, i) => {
    const g = bySector[sectors[i]].filter((j) => j !== i);
    if (g.length < 4) return 0;
    const mu = g.reduce((a, j) => a + mom[j], 0) / g.length;
    const sd = Math.sqrt(g.reduce((a, j) => a + (mom[j] - mu) ** 2, 0) / g.length) || 1;
    return (mom[i] - mu) / sd;
  });

  const row = { date: snap.date, flagged: reading.filter((r) => r.unreliable).length };
  for (const h of F_PREREG.horizons) {
    const fwdAll = snap.forward[h];
    if (!fwdAll) continue;
    const fwd = joint.map((r) => fwdAll[r.state]);
    const ok = [...Array(n).keys()].filter((i) => !reading[i].unreliable);
    const bad = [...Array(n).keys()].filter((i) => reading[i].unreliable);
    row[h] = {
      peer: spearman(pick(reading.map((r) => r.z), ok), pick(fwd, ok)),
      sector: spearman(pick(sectorZ, ok), pick(fwd, ok)),
      raw: spearman(pick(mom, ok), pick(fwd, ok)),
      peerFlagged: spearman(pick(reading.map((r) => r.z), bad), pick(fwd, bad)),
      nOk: ok.length,
    };
  }
  rows.push(row);
}

const H = F_PREREG.horizon_primary;
const usable = rows.filter((r) => r[H]);
const col = (k) => usable.map((r) => r[k === "peerFlagged" ? H : H][k]).filter((v) => v !== null);

console.log("=== Stage D5 — forward-return check (pre-registered) ===\n");
console.log(`snapshots with a full ${H}-day forward window: ${usable.length}   stocks per snapshot: ~${usable[0][H].nOk} unflagged`);
console.log(F_PREREG.not_claimed + "\n");

console.log("date         rho(peer-z)  rho(sector-z)  rho(raw mom)   flagged");
for (const r of usable.filter((_, k) => k % 3 === 0))
  console.log(
    `  ${r.date}  ${r[H].peer.toFixed(3).padStart(10)}  ${r[H].sector.toFixed(3).padStart(12)}` +
      `  ${r[H].raw.toFixed(3).padStart(11)}  ${String(r.flagged).padStart(8)}`,
  );

const summary = {};
for (const k of ["peer", "sector", "raw", "peerFlagged"]) {
  const v = col(k);
  const m = mean(v);
  const consistent = v.filter((x) => Math.sign(x) === Math.sign(m)).length;
  summary[k] = { mean: m, consistent, total: v.length, frac: consistent / v.length };
}

console.log(`\nacross ${usable.length} snapshots, ${H}-day horizon:`);
for (const [k, s] of Object.entries(summary))
  console.log(
    `  ${k.padEnd(12)} mean rho ${s.mean.toFixed(3).padStart(7)}   same sign as the mean in ${s.consistent}/${s.total} (${(100 * s.frac).toFixed(0)}%)`,
  );

const h1 = Math.abs(summary.peer.mean) >= 0.05 && summary.peer.frac >= 2 / 3;
const h2 = Math.abs(summary.peer.mean) > Math.abs(summary.raw.mean);
const h3 = Math.abs(summary.peer.mean) > Math.abs(summary.peerFlagged.mean);

console.log("\nverdicts:");
console.log(`  F-H1  peer-z ranks forward returns          ${h1 ? "SUPPORTED" : "NOT SUPPORTED"}` +
  `   (|${summary.peer.mean.toFixed(3)}| vs 0.05, sign ${(100 * summary.peer.frac).toFixed(0)}% vs 67%)`);
console.log(`  F-H2  peer-z beats the free covariate       ${h2 ? "SUPPORTED" : "NOT SUPPORTED"}` +
  `   (|${summary.peer.mean.toFixed(3)}| vs |${summary.raw.mean.toFixed(3)}| raw momentum)`);
console.log(`  F-H3  the cohesion guard is doing work      ${h3 ? "SUPPORTED" : "NOT SUPPORTED"}` +
  `   (|${summary.peer.mean.toFixed(3)}| unflagged vs |${summary.peerFlagged.mean.toFixed(3)}| flagged)`);

// secondary horizon, reported without a verdict attached
const H2h = 21;
const u2 = rows.filter((r) => r[H2h]);
console.log(`\n${H2h}-day horizon (secondary, no pre-registered threshold):`);
for (const k of ["peer", "sector", "raw"])
  console.log(`  ${k.padEnd(8)} mean rho ${mean(u2.map((r) => r[H2h][k])).toFixed(3)}`);

fs.writeFileSync(
  path.join(ROOT, "out", "layers", "d5-forward.json"),
  JSON.stringify({ stage: "D5", ran_at: new Date().toISOString(), prereg: F_PREREG, rows, summary, verdicts: { "F-H1": h1, "F-H2": h2, "F-H3": h3 } }, null, 2),
);
console.log("\nsaved: out/layers/d5-forward.json");
