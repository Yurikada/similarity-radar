// Stage B0 — fetch fundamentals for the JP universe and build a standardized
// feature matrix. Features span value / quality / growth / size. Reports coverage,
// drops sparse rows, winsorizes, imputes cross-sectional medians, then z-scores.
//
// Run: node src/stocks/00-features.mjs   (network required)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YahooFinance from "yahoo-finance2";
import { UNIVERSE } from "./universe.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const yf = new YahooFinance({ suppressNotices: ["yahooSurvey"] });

// feature name -> extractor from the quoteSummary modules
const FEATURES = {
  per: (q) => q.summaryDetail?.trailingPE,
  pbr: (q) => q.defaultKeyStatistics?.priceToBook,
  divYield: (q) => q.summaryDetail?.dividendYield ?? 0,
  roe: (q) => q.financialData?.returnOnEquity,
  roa: (q) => q.financialData?.returnOnAssets,
  opMargin: (q) => q.financialData?.operatingMargins,
  revGrowth: (q) => q.financialData?.revenueGrowth,
  earnGrowth: (q) => q.financialData?.earningsGrowth,
  logMktCap: (q) => (q.price?.marketCap ? Math.log10(q.price.marketCap) : undefined),
};
const FNAMES = Object.keys(FEATURES);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const rows = [];
process.stdout.write("fetching");
for (const ticker of UNIVERSE) {
  try {
    const q = await yf.quoteSummary(ticker, {
      modules: ["price", "summaryDetail", "defaultKeyStatistics", "financialData", "assetProfile"],
    }, { validateResult: false });
    const feat = {};
    for (const f of FNAMES) { const v = FEATURES[f](q); feat[f] = Number.isFinite(v) ? v : null; }
    rows.push({ ticker, name: q.price?.shortName ?? q.price?.longName ?? ticker, sector: q.assetProfile?.sector ?? "Unknown", feat });
    process.stdout.write(".");
  } catch (e) {
    process.stdout.write("x");
  }
  await sleep(120);
}
console.log("\nfetched:", rows.length, "/", UNIVERSE.length);

// coverage
console.log("\nfeature coverage:");
for (const f of FNAMES) {
  const have = rows.filter((r) => r.feat[f] !== null).length;
  console.log(`  ${f.padEnd(11)} ${have}/${rows.length}`);
}

// drop rows missing > 40% of features
const kept = rows.filter((r) => FNAMES.filter((f) => r.feat[f] !== null).length >= FNAMES.length * 0.6);
console.log(`\nkept ${kept.length} rows (dropped ${rows.length - kept.length} too-sparse)`);

// winsorize [2%,98%] per feature, then impute median, then z-score
const quantile = (arr, q) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]; };
const stats = {};
for (const f of FNAMES) {
  const vals = kept.map((r) => r.feat[f]).filter((v) => v !== null);
  const lo = quantile(vals, 0.02), hi = quantile(vals, 0.98);
  const med = quantile(vals, 0.5);
  const clipped = vals.map((v) => Math.max(lo, Math.min(hi, v)));
  const mean = clipped.reduce((a, b) => a + b, 0) / clipped.length;
  const sd = Math.sqrt(clipped.reduce((a, b) => a + (b - mean) ** 2, 0) / clipped.length) || 1;
  stats[f] = { lo, hi, med, mean, sd };
}
const matrix = kept.map((r) => FNAMES.map((f) => {
  const s = stats[f];
  const raw = r.feat[f] === null ? s.med : r.feat[f];
  return (Math.max(s.lo, Math.min(s.hi, raw)) - s.mean) / s.sd;
}));

const out = {
  features: FNAMES,
  tickers: kept.map((r) => r.ticker),
  names: kept.map((r) => r.name),
  sectors: kept.map((r) => r.sector),
  raw: kept.map((r) => r.feat),
  matrix,
};
fs.mkdirSync(path.join(ROOT, "out", "stocks"), { recursive: true });
fs.writeFileSync(path.join(ROOT, "out", "stocks", "features.json"), JSON.stringify(out));

const sectorCount = {};
for (const s of out.sectors) sectorCount[s] = (sectorCount[s] ?? 0) + 1;
console.log("\nsectors:", Object.entries(sectorCount).sort((a, b) => b[1] - a[1]).map(([s, c]) => `${s}:${c}`).join("  "));
console.log("\nsaved: out/stocks/features.json (n=" + kept.length + ", d=" + FNAMES.length + ")");
