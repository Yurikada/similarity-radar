// Stage D3 — cache raw daily closes, so the state layer can be rebuilt at any
// date offline and forward returns can be measured without refetching.
//
// D0 stored only the four derived features at two snapshots. That was enough to
// prove the map does not move when state changes, but not enough to (a) step the
// overlay through a series of dates or (b) measure what happened AFTER a signal.
// Both need the underlying series.
//
// Written to a SEPARATE file. out/layers/state-features.json is the input to D1's
// pre-registered result and to E1's baseline; refetching over it with a different
// history window would silently move numbers that are already frozen.
//
// Run: node src/layers/03-price-cache.mjs   (network required, ~2 min)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YahooFinance from "yahoo-finance2";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const yf = new YahooFinance({ suppressNotices: ["yahooSurvey"] });
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "features.json"), "utf8"));

const YEARS = 3;
const MIN_BARS = 500; // need 126d lookback + room for forward windows

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const period1 = new Date(Date.now() - Math.round(YEARS * 365.25) * 864e5);
const rows = [];
process.stdout.write(`fetching ${YEARS}y of closes`);
for (let i = 0; i < S.tickers.length; i++) {
  try {
    const r = await yf.chart(S.tickers[i], { period1, interval: "1d" });
    const qs = (r.quotes ?? []).filter((q) => Number.isFinite(q.close));
    if (qs.length >= MIN_BARS) {
      rows.push({
        base: i,
        ticker: S.tickers[i],
        name: S.names[i],
        sector: S.sectors[i],
        dates: qs.map((q) => new Date(q.date).toISOString().slice(0, 10)),
        closes: qs.map((q) => q.close),
      });
      process.stdout.write(".");
    } else process.stdout.write("-");
  } catch {
    process.stdout.write("x");
  }
  await sleep(90);
}
console.log(`\nusable: ${rows.length}/${S.tickers.length} (>= ${MIN_BARS} bars)`);

// Align every series on the trading days common to all of them, so a snapshot
// index means the same calendar date for every stock. Without this, "index 400"
// silently refers to different dates for names with different halt histories.
const common = rows
  .map((r) => new Set(r.dates))
  .reduce((acc, s) => new Set([...acc].filter((d) => s.has(d))));
const dates = [...common].sort();
const closes = rows.map((r) => {
  const m = new Map(r.dates.map((d, k) => [d, r.closes[k]]));
  return dates.map((d) => m.get(d));
});

console.log(`common trading days: ${dates.length}  (${dates[0]} .. ${dates[dates.length - 1]})`);

fs.writeFileSync(
  path.join(ROOT, "out", "layers", "price-cache.json"),
  JSON.stringify({
    built_at: new Date().toISOString(),
    years: YEARS,
    n: rows.length,
    baseIndex: rows.map((r) => r.base),
    tickers: rows.map((r) => r.ticker),
    names: rows.map((r) => r.name),
    sectors: rows.map((r) => r.sector),
    dates,
    closes,
  }),
);
console.log(`\nsaved: out/layers/price-cache.json (${rows.length} x ${dates.length})`);
