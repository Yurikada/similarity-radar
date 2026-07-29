// Stage D0 — build the STATE layer and cache it.
//
// Part D separates two things Part B had no reason to distinguish, and the
// temporal experiment actively conflated:
//
//   identity layer  what a company IS   — fundamentals (value/quality/growth/size)
//   state layer     how it is BEHAVING  — momentum and volatility, from prices
//
// The two differ in kind (structure vs condition), in update frequency (annual
// vs daily) and in what proximity means. Putting them in one distance matrix
// makes "these two are close" unreadable: you cannot tell whether it means the
// same sort of business or the same recent price path. Part D keeps them apart
// and composes them only at display time.
//
// This stage does nothing but fetch and cache, so every measurement downstream
// runs offline and is reproducible without hitting the network again. Two
// snapshots ~6 months apart are stored, which is what makes the stability test
// in D1 possible: the state layer can be updated while identity is held fixed.
//
// Run: node src/layers/00-state-features.mjs   (network required, ~2 min)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YahooFinance from "yahoo-finance2";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const yf = new YahooFinance({ suppressNotices: ["yahooSurvey"] });
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "features.json"), "utf8"));

const STATE_FEATURES = ["mom1m", "mom3m", "mom6m", "vol3m"];

/** price-derived state at close index s (all windows trailing, no look-ahead) */
const stateAt = (c, s) => [
  c[s] / c[s - 21] - 1,
  c[s] / c[s - 63] - 1,
  c[s] / c[s - 126] - 1,
  (() => {
    const w = c
      .slice(s - 63, s)
      .map((x, k, a) => (k ? x / a[k - 1] - 1 : 0))
      .slice(1);
    let m = 0,
      v = 0;
    for (const r of w) m += r / w.length;
    for (const r of w) v += (r - m) ** 2 / w.length;
    return Math.sqrt(v) * Math.sqrt(252);
  })(),
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const period1 = new Date(Date.now() - 430 * 864e5);
const series = [];
process.stdout.write("fetching prices");
for (let i = 0; i < S.tickers.length; i++) {
  try {
    const r = await yf.chart(S.tickers[i], { period1, interval: "1d" });
    const closes = (r.quotes ?? []).map((q) => q.close).filter((c) => Number.isFinite(c));
    if (closes.length >= 260) series.push({ base: i, ticker: S.tickers[i], closes });
    process.stdout.write(".");
  } catch {
    process.stdout.write("x");
  }
  await sleep(90);
}
console.log(`\nusable: ${series.length}/${S.tickers.length}`);

// snapshots placed off the shortest series so both have a full 126d lookback
const minLen = Math.min(...series.map((s) => s.closes.length));
const at = (s, off) => stateAt(s.closes, s.closes.length - (minLen - off));
const t0 = minLen - 127,
  t2 = minLen - 1;

const out = {
  built_at: new Date().toISOString(),
  featureNames: STATE_FEATURES,
  snapshots: { t0: "~6 months ago", t2: "latest" },
  // index into features.json, so identity and state stay joinable
  baseIndex: series.map((s) => s.base),
  tickers: series.map((s) => s.ticker),
  names: series.map((s) => S.names[s.base]),
  sectors: series.map((s) => S.sectors[s.base]),
  t0: series.map((s) => at(s, t0)),
  t2: series.map((s) => at(s, t2)),
};

fs.mkdirSync(path.join(ROOT, "out", "layers"), { recursive: true });
fs.writeFileSync(path.join(ROOT, "out", "layers", "state-features.json"), JSON.stringify(out));

console.log("\nstate layer summary (t2, latest snapshot):");
for (let j = 0; j < STATE_FEATURES.length; j++) {
  const v = out.t2.map((r) => r[j]).sort((a, b) => a - b);
  const q = (p) => v[Math.min(v.length - 1, Math.floor(p * (v.length - 1)))];
  console.log(
    `  ${STATE_FEATURES[j].padEnd(7)} min ${q(0).toFixed(3)}  p25 ${q(0.25).toFixed(3)}` +
      `  median ${q(0.5).toFixed(3)}  p75 ${q(0.75).toFixed(3)}  max ${q(1).toFixed(3)}`,
  );
}
console.log(`\nsaved: out/layers/state-features.json (n=${series.length}, d=${STATE_FEATURES.length}, 2 snapshots)`);
