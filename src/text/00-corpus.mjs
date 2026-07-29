// Stage E0 — fetch the business-description corpus and audit it before use.
//
// Part E replaces the identity substrate. D1 showed why it is needed: peers taken
// in fundamentals space correlate 0.72-0.73 with the vendor sector label, because
// margins, ROE and capital intensity are themselves industry artefacts. Comparing
// those peers against the sector was comparing a proxy for sector against sector.
// Text is the substrate on which "what is this company" is actually written.
//
// This is the PILOT: Yahoo's `assetProfile.longBusinessSummary`, already covered
// by the modules B0 requests, so the plumbing can be validated end to end before
// committing to EDINET (which needs a subscription key, ZIP/XBRL handling and
// section extraction from the filing HTML).
//
// Nothing is vectorized here. The stage exists to answer the questions that
// decide whether the corpus is usable at all: coverage, length, language, near
// duplicates, and how much of every document is shared boilerplate. A text map
// built on boilerplate clusters by writing style, not by business.
//
// Run: node src/text/00-corpus.mjs   (network required)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YahooFinance from "yahoo-finance2";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const yf = new YahooFinance({ suppressNotices: ["yahooSurvey"] });
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "features.json"), "utf8"));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const docs = [];
process.stdout.write("fetching profiles");
for (let i = 0; i < S.tickers.length; i++) {
  try {
    const q = await yf.quoteSummary(S.tickers[i], { modules: ["assetProfile"] }, { validateResult: false });
    const text = q.assetProfile?.longBusinessSummary ?? null;
    docs.push({ base: i, ticker: S.tickers[i], name: S.names[i], sector: S.sectors[i], text });
    process.stdout.write(text ? "." : "-");
  } catch {
    docs.push({ base: i, ticker: S.tickers[i], name: S.names[i], sector: S.sectors[i], text: null });
    process.stdout.write("x");
  }
  await sleep(120);
}

const have = docs.filter((d) => d.text);
console.log(`\n\ncoverage: ${have.length}/${docs.length} have a business summary`);
if (!have.length) throw new Error("no documents fetched");

// ---- length
const lens = have.map((d) => d.text.length).sort((a, b) => a - b);
const q = (p) => lens[Math.min(lens.length - 1, Math.floor(p * (lens.length - 1)))];
console.log(
  `length (chars): min ${q(0)}  p25 ${q(0.25)}  median ${q(0.5)}  p75 ${q(0.75)}  max ${q(1)}`,
);

// ---- language: CJK share decides whether character n-grams are the right unit
const cjk = (s) => (s.match(/[぀-ヿ一-鿿]/g) ?? []).length / s.length;
const cjkShare = have.map((d) => cjk(d.text));
const meanCjk = cjkShare.reduce((a, b) => a + b, 0) / cjkShare.length;
console.log(`CJK character share: mean ${(100 * meanCjk).toFixed(1)}%  (max ${(100 * Math.max(...cjkShare)).toFixed(1)}%)`);

// ---- near duplicates on 3-gram Jaccard: identical filings would fake similarity
const grams = (s, n) => {
  const t = s.toLowerCase().replace(/\s+/g, " ");
  const g = new Set();
  for (let i = 0; i + n <= t.length; i++) g.add(t.slice(i, i + n));
  return g;
};
const G = have.map((d) => grams(d.text, 3));
let dupes = 0;
const worst = [];
for (let i = 0; i < G.length; i++)
  for (let j = i + 1; j < G.length; j++) {
    let inter = 0;
    for (const g of G[i]) if (G[j].has(g)) inter++;
    const jac = inter / (G[i].size + G[j].size - inter);
    if (jac > 0.6) {
      dupes++;
      worst.push({ a: have[i].ticker, b: have[j].ticker, jac });
    }
  }
worst.sort((x, y) => y.jac - x.jac);
console.log(`near-duplicate pairs (3-gram Jaccard > 0.6): ${dupes}`);
for (const w of worst.slice(0, 5)) console.log(`   ${w.a} ~ ${w.b}  ${w.jac.toFixed(3)}`);

// ---- boilerplate: n-grams present in most documents carry no discriminating signal
const df = new Map();
for (const g of G) for (const t of g) df.set(t, (df.get(t) ?? 0) + 1);
const N = G.length;
const ubiquitous = [...df.entries()].filter(([, c]) => c >= 0.9 * N).length;
const top = [...df.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12);
console.log(`\n3-grams appearing in >=90% of documents: ${ubiquitous} of ${df.size} distinct`);
console.log("most ubiquitous:", top.map(([t, c]) => `"${t}"(${Math.round((100 * c) / N)}%)`).join(" "));
// share of each document made of near-ubiquitous grams
const ubiSet = new Set([...df.entries()].filter(([, c]) => c >= 0.9 * N).map(([t]) => t));
const shares = G.map((g) => {
  let hit = 0;
  for (const t of g) if (ubiSet.has(t)) hit++;
  return hit / g.size;
}).sort((a, b) => a - b);
const qs = (p) => shares[Math.min(shares.length - 1, Math.floor(p * (shares.length - 1)))];
console.log(
  `per-document boilerplate share: p25 ${(100 * qs(0.25)).toFixed(1)}%  median ${(100 * qs(0.5)).toFixed(1)}%  p75 ${(100 * qs(0.75)).toFixed(1)}%`,
);

// ---- sector coverage after dropping missing docs
const bySector = {};
for (const d of have) bySector[d.sector] = (bySector[d.sector] ?? 0) + 1;
console.log("\nsectors retained:", Object.entries(bySector).sort((a, b) => b[1] - a[1]).map(([s, c]) => `${s}:${c}`).join("  "));

fs.mkdirSync(path.join(ROOT, "out", "text"), { recursive: true });
fs.writeFileSync(
  path.join(ROOT, "out", "text", "corpus.json"),
  JSON.stringify({
    built_at: new Date().toISOString(),
    source: "yahoo assetProfile.longBusinessSummary (pilot; EDINET 事業の内容 is the intended source)",
    n: have.length,
    requested: docs.length,
    docs: have,
  }),
);
console.log(`\nsaved: out/text/corpus.json (n=${have.length})`);
console.log("sample:", have[0].ticker, "-", have[0].text.slice(0, 180).replace(/\s+/g, " "), "...");
