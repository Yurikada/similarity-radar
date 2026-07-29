// Stage: temporal drift — the A6 trend line on real time series.
// Fetch ~14 months of daily prices; compute price-based factors (momentum 1/3/6m,
// 3m volatility) at two snapshots (~6 months apart); pool-standardize + pool-whiten;
// MDS each snapshot INDEPENDENTLY; Procrustes-align t2 onto t0; the residual arrow
// per stock is its real drift in factor space (frame removed). Emits an arrow map.
//
// Run: node src/stocks/02-temporal.mjs   (network required)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YahooFinance from "yahoo-finance2";
import { jacobiEigsym } from "../lib/eig.mjs";
import { smacof } from "../lib/projection/mds.mjs";
import { procrustesAlign } from "../lib/procrustes.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const yf = new YahooFinance({ suppressNotices: ["yahooSurvey"] });
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "features.json"), "utf8"));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const period1 = new Date(Date.now() - 430 * 864e5);
const series = [];
process.stdout.write("fetching prices");
for (let i = 0; i < S.tickers.length; i++) {
  const t = S.tickers[i];
  try {
    const r = await yf.chart(t, { period1, interval: "1d" });
    const closes = (r.quotes ?? []).map((q) => q.close).filter((c) => Number.isFinite(c));
    if (closes.length >= 260) series.push({ i, ticker: t, name: S.names[i], sector: S.sectors[i], closes });
    process.stdout.write(".");
  } catch { process.stdout.write("x"); }
  await sleep(90);
}
console.log(`\nusable: ${series.length}/${S.tickers.length}`);

// features at snapshot index s (end of trailing windows)
const feat = (c, s) => [
  c[s] / c[s - 21] - 1,             // 1m momentum
  c[s] / c[s - 63] - 1,             // 3m momentum
  c[s] / c[s - 126] - 1,            // 6m momentum
  (() => { let m = 0, v = 0; const w = c.slice(s - 63, s).map((x, k, a) => k ? x / a[k - 1] - 1 : 0).slice(1); for (const r of w) m += r / w.length; for (const r of w) v += (r - m) ** 2 / w.length; return Math.sqrt(v) * Math.sqrt(252); })(), // 3m vol
];
// use the shortest series length to place snapshots consistently
const minLen = Math.min(...series.map((s) => s.closes.length));
const t2 = minLen - 1, t0 = minLen - 127; // ~6 months apart, both with 126d lookback
const rows0 = series.map((s) => feat(s.closes, s.closes.length - (minLen - t0)));
const rows2 = series.map((s) => feat(s.closes, s.closes.length - (minLen - t2)));
const n = series.length, d = 4;

// pooled standardize then pooled whiten (consistent transform across time)
const pool = [...rows0, ...rows2];
const mean = Array.from({ length: d }, (_, j) => pool.reduce((a, r) => a + r[j], 0) / pool.length);
const sd = Array.from({ length: d }, (_, j) => Math.sqrt(pool.reduce((a, r) => a + (r[j] - mean[j]) ** 2, 0) / pool.length) || 1);
const z = (rows) => rows.map((r) => r.map((v, j) => (v - mean[j]) / sd[j]));
const Z0 = z(rows0), Z2 = z(rows2), Zp = [...Z0, ...Z2];
const C = Array.from({ length: d }, () => new Float64Array(d));
for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) { let s = 0; for (const r of Zp) s += r[a] * r[b]; C[a][b] = s / Zp.length; }
const { values, vectors } = jacobiEigsym(C);
const floor = values[0] * 1e-3;
const whiten = (rows) => rows.map((x) => vectors.map((vec, k) => { let dot = 0; for (let j = 0; j < d; j++) dot += vec[j] * x[j]; return dot / Math.sqrt(Math.max(values[k], floor)); }));
const W0 = whiten(Z0), W2 = whiten(Z2);

const distMat = (W) => { const D = Array.from({ length: n }, () => new Float64Array(n)); for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { let s = 0; for (let k = 0; k < d; k++) { const dd = W[i][k] - W[j][k]; s += dd * dd; } D[i][j] = D[j][i] = Math.sqrt(s); } return D; };
let P0 = smacof(distMat(W0), 1, 300), P2 = smacof(distMat(W2), 2, 300);

// normalize (center + unit scale), then Procrustes-align P2 onto P0. Uses the
// shared implementation: the reflected branch needs its own optimal angle, which
// the previous inline version did not recompute.
const norm = (P) => { let cx = 0, cy = 0; for (const p of P) { cx += p[0] / n; cy += p[1] / n; } const Q = P.map((p) => [p[0] - cx, p[1] - cy]); let s = 0; for (const p of Q) s += p[0] ** 2 + p[1] ** 2; s = Math.sqrt(s); return Q.map((p) => [p[0] / s, p[1] / s]); };
P0 = norm(P0);
const { Y: P2a, angleDeg, reflected } = procrustesAlign(P0, P2);

const drift = series.map((s, i) => ({ ...s, x0: P0[i][0], y0: P0[i][1], x2: P2a[i][0], y2: P2a[i][1], mag: Math.hypot(P2a[i][0] - P0[i][0], P2a[i][1] - P0[i][1]) }));
drift.sort((a, b) => b.mag - a.mag);
const ang = (angleDeg * Math.PI) / 180;
console.log(`\nrotation removed by Procrustes: ${Math.abs(angleDeg).toFixed(1)}°${reflected ? " (+reflection)" : ""}`);
console.log("biggest factor-space movers (6 months):");
for (const m of drift.slice(0, 6)) console.log(`   |Δ|=${m.mag.toFixed(3)}  ${m.name} (${m.sector})`);

fs.writeFileSync(path.join(ROOT, "out", "stocks", "temporal.json"),
  JSON.stringify({ points: drift.map(({ closes, ...r }) => r), rotationDeg: Math.abs(ang) * 180 / Math.PI, n }));
console.log("\nsaved: out/stocks/temporal.json");
