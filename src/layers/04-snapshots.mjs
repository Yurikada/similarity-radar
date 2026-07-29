// Stage D4 — derive the state layer at a series of dates, plus the forward
// returns that follow each one. Offline: everything comes from the D3 cache.
//
// Two consumers:
//   D2  steps the overlay through the series with the identity map held fixed —
//       the separation argument shown over 28 dates instead of two.
//   D5  measures what happened AFTER each snapshot, which is only meaningful
//       because the signal at date s uses no price beyond s.
//
// Run: node src/layers/04-snapshots.mjs   (no network)

import fs from "node:fs";
import path from "node:path";
import { ROOT } from "../objectives/domains.mjs";

const LOOKBACK = 126; // longest trailing window (6m momentum)
const STEP = 21; // one trading month between snapshots
const FORWARD = [21, 63]; // horizons measured after each snapshot

const P = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "layers", "price-cache.json"), "utf8"));
const { closes, dates, n } = P;
const T = dates.length;

/** state at close index s — every window trails, so nothing after s is used */
const stateAt = (c, s) => {
  const w = c.slice(s - 63, s + 1).map((x, k, a) => (k ? x / a[k - 1] - 1 : 0)).slice(1);
  const mu = w.reduce((a, b) => a + b, 0) / w.length;
  const sd = Math.sqrt(w.reduce((a, b) => a + (b - mu) ** 2, 0) / w.length);
  return [c[s] / c[s - 21] - 1, c[s] / c[s - 63] - 1, c[s] / c[s - 126] - 1, sd * Math.sqrt(252)];
};

const idx = [];
for (let s = LOOKBACK; s < T; s += STEP) idx.push(s);

const snapshots = idx.map((s) => ({
  index: s,
  date: dates[s],
  features: closes.map((c) => stateAt(c, s)),
  forward: Object.fromEntries(
    FORWARD.map((h) => [h, s + h < T ? closes.map((c) => c[s + h] / c[s] - 1) : null]),
  ),
}));

const out = {
  built_at: new Date().toISOString(),
  featureNames: ["mom1m", "mom3m", "mom6m", "vol3m"],
  forwardHorizons: FORWARD,
  step: STEP,
  lookback: LOOKBACK,
  n,
  baseIndex: P.baseIndex,
  tickers: P.tickers,
  names: P.names,
  sectors: P.sectors,
  snapshots,
};
fs.writeFileSync(path.join(ROOT, "out", "layers", "state-series.json"), JSON.stringify(out));

const withFwd = snapshots.filter((s) => s.forward[63] !== null).length;
console.log(`=== Stage D4 — state series ===\n`);
console.log(`snapshots: ${snapshots.length}  (${snapshots[0].date} .. ${snapshots[snapshots.length - 1].date}), one per ${STEP} trading days`);
console.log(`with a full 63-day forward window: ${withFwd}`);
console.log(`stocks: ${n}\n`);
const mom = (s) => {
  const v = s.features.map((f) => f[1]).sort((a, b) => a - b);
  return v[Math.floor(0.5 * (v.length - 1))];
};
console.log("date        median mom3m   median vol3m");
for (const s of snapshots.filter((_, k) => k % 4 === 0)) {
  const v = s.features.map((f) => f[3]).sort((a, b) => a - b);
  console.log(`  ${s.date}  ${(100 * mom(s)).toFixed(1).padStart(7)}%  ${(100 * v[Math.floor(0.5 * (v.length - 1))]).toFixed(1).padStart(11)}%`);
}
console.log(`\nsaved: out/layers/state-series.json`);
