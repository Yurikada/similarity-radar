// Build a self-contained, theme-aware HTML case-study document from the computed
// artifacts in out/kb/. Regenerate as stages advance. Output contains private KB
// titles → written to out/kb/ (gitignored). Publish separately as an Artifact.
//
// Run: node src/kb/build-doc.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { mulberry32 } from "../lib/rng.mjs";
import { gridDensity } from "../lib/kde.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const rd = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, "out", "kb", f), "utf8"));
const emb = rd("permanent-embeddings.json");
const fig = rd("_figdata.json");
const proj = rd("projections.json");
const rad = rd("radial.json");
const den = rd("density.json");
const clu = rd("clusters.json");
const trend = rd("trend.json");

// ---------- figure builders (self-contained SVG, colors via CSS vars) ----------
const T = (x, y, s, cls, anchor = "middle") =>
  `<text x="${x}" y="${y}" font-size="${s}" fill="var(--${cls})" text-anchor="${anchor}" font-family="var(--mono)">`;

function figPipeline() {
  const rows = [
    ["A0", "特徴行列", "done"],
    ["A1", "距離・異方性", "done"],
    ["A2", "投影", "done"],
    ["A3", "半径", "done"],
    ["A4", "密度KDE", "done"],
    ["A5", "クラスタ", "done"],
  ];
  const col = { done: "fig-teal", next: "fig-amber", wait: "fig-muted" };
  let s = `<svg viewBox="0 0 680 120" width="100%" role="img" aria-label="pipeline progress A0 to A5">`;
  rows.forEach((r, i) => {
    const x = 20 + i * 110;
    s += `<rect x="${x}" y="34" width="100" height="54" rx="9" fill="var(--surface)" stroke="var(--hairline)"/>`;
    s += `<circle cx="${x + 16}" cy="52" r="4" fill="var(--${col[r[2]]})"/>`;
    s += T(x + 50, 58, 15, "fig-text") + r[0] + `</text>`;
    s += T(x + 50, 76, 11, "fig-muted") + r[1] + `</text>`;
  });
  return s + `</svg>`;
}

function figCone() {
  const cx = 200, cy = 180, r = 120, phi = -22 * Math.PI / 180, spread = 20 * Math.PI / 180;
  const rng = mulberry32(42);
  let dots = "";
  for (let i = 0; i < 34; i++) {
    const a = phi + (rng() - 0.5) * 2 * spread;
    const rr = r * (0.985 + (rng() - 0.5) * 0.03);
    dots += `<circle cx="${(cx + rr * Math.cos(a)).toFixed(1)}" cy="${(cy + rr * Math.sin(a)).toFixed(1)}" r="3" fill="var(--fig-blue)"/>`;
  }
  let s = `<svg viewBox="0 0 680 330" width="100%" role="img" aria-label="unit vectors clustered in a narrow cone">`;
  s += `<defs><marker id="ca" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="var(--fig-amber)" stroke-width="1.6" stroke-linecap="round"/></marker></defs>`;
  s += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--fig-grid)"/>`;
  s += dots;
  s += `<line x1="${cx}" y1="${cy}" x2="${(cx + 0.9 * r * Math.cos(phi)).toFixed(1)}" y2="${(cy + 0.9 * r * Math.sin(phi)).toFixed(1)}" stroke="var(--fig-amber)" stroke-width="2.4" marker-end="url(#ca)"/>`;
  s += `<circle cx="${cx}" cy="${cy}" r="2.6" fill="var(--fig-muted)"/>`;
  s += T(cx, 325, 12, "fig-muted") + `単位球（ノルム = 1）</text>`;
  s += T(250, 200, 12, "fig-amber", "start") + `‖平均‖ = 0.942</text>`;
  s += T(392, 108, 14, "fig-text", "start") + `異方性 = cone effect</text>`;
  s += T(392, 136, 12, "fig-muted", "start") + `86個の単位ベクトルが狭い円錐に密集</text>`;
  s += T(392, 158, 12, "fig-muted", "start") + `→ 各ベクトルの約89%が</text>`;
  s += T(392, 176, 12, "fig-muted", "start") + `　 内容と無関係な共通方向</text>`;
  s += T(392, 210, 12, "fig-muted", "start") + `だから無関係な文でも cos ≈ 0.88</text>`;
  return s + `</svg>`;
}

function figHist() {
  const MAX = 1992, base = 300, H = 230, x0 = 70, bw = 23.75, barW = 21;
  const bars = (arr, v) =>
    arr.map((c, i) => c ? `<rect x="${(x0 + i * bw + 1).toFixed(1)}" y="${(base - c / MAX * H).toFixed(1)}" width="${barW}" height="${(c / MAX * H).toFixed(1)}" rx="1" fill="var(--fig-${v})" fill-opacity="0.82"/>` : "").join("");
  const mx = (val) => (x0 + (val + 0.5) / 1.5 * 570).toFixed(1);
  let s = `<svg viewBox="0 0 680 356" width="100%" role="img" aria-label="cosine similarity histogram before and after centering">`;
  s += `<rect x="90" y="36" width="14" height="14" rx="3" fill="var(--fig-amber)"/>` + T(112, 47, 12, "fig-muted", "start") + `中心化後（全域に展開）</text>`;
  s += `<rect x="290" y="36" width="14" height="14" rx="3" fill="var(--fig-blue)"/>` + T(312, 47, 12, "fig-muted", "start") + `生 raw（0.78–0.99 に圧縮）</text>`;
  s += `<line x1="70" y1="300" x2="640" y2="300" stroke="var(--fig-grid)"/>`;
  s += `<line x1="${mx(-0.048)}" y1="92" x2="${mx(-0.048)}" y2="300" stroke="var(--fig-amber)" stroke-width="1.4" stroke-dasharray="4 3"/>` + T(mx(-0.048), 84, 12, "fig-amber") + `中心化 median −0.05</text>`;
  s += `<line x1="${mx(0.888)}" y1="78" x2="${mx(0.888)}" y2="300" stroke="var(--fig-blue)" stroke-width="1.4" stroke-dasharray="4 3"/>` + T(mx(0.888), 70, 12, "fig-blue") + `生 median 0.89</text>`;
  s += bars(fig.centered.h, "amber") + bars(fig.raw.h, "blue");
  for (const [v, x] of [[-0.5, 70], [0, 260], [0.5, 450], [1.0, 640]]) s += T(x, 318, 12, "fig-muted") + v + `</text>`;
  s += T(355, 344, 12, "fig-muted") + `コサイン類似度（全 3655 ペア）</text>`;
  return s + `</svg>`;
}

function figRank() {
  const x0 = 80, W = 540, yb = 260, H = 200;
  const px = (c) => x0 + (c + 1) / 2 * W;
  const py = (d) => yb - d / 2 * H;
  let path = "";
  for (let i = 0; i <= 80; i++) { const c = -1 + 2 * i / 80; const d = Math.sqrt(Math.max(0, 2 - 2 * c)); path += (i ? "L" : "M") + px(c).toFixed(1) + " " + py(d).toFixed(1); }
  let s = `<svg viewBox="0 0 680 300" width="100%" role="img" aria-label="euclidean distance is a monotone function of cosine similarity">`;
  s += `<rect x="${px(0.78)}" y="60" width="${(px(0.99) - px(0.78)).toFixed(1)}" height="200" fill="var(--fig-blue)" fill-opacity="0.12"/>`;
  s += T(px(0.885), 52, 12, "fig-blue") + `データ領域</text>` + T(px(0.885), 288, 12, "fig-blue") + `0.78–0.99</text>`;
  s += `<line x1="80" y1="260" x2="620" y2="260" stroke="var(--fig-grid)"/><line x1="80" y1="60" x2="80" y2="260" stroke="var(--fig-grid)"/>`;
  s += `<path d="${path}" fill="none" stroke="var(--fig-teal)" stroke-width="2.4"/>`;
  s += T(80, 278, 12, "fig-muted") + `cos −1</text>` + T(350, 278, 12, "fig-muted") + `0</text>` + T(620, 278, 12, "fig-muted") + `1</text>`;
  s += T(70, 64, 12, "fig-muted", "end") + `2</text>` + T(70, 164, 12, "fig-muted", "end") + `1</text>` + T(70, 262, 12, "fig-muted", "end") + `0</text>`;
  s += T(150, 120, 12, "fig-muted", "start") + `d = √(2 − 2cos)</text>`;
  s += T(150, 148, 14, "fig-text", "start") + `単調減少</text>`;
  s += T(150, 172, 12, "fig-muted", "start") + `→ 近傍の順位は全域で不変</text>`;
  s += T(150, 192, 12, "fig-muted", "start") + `→ 86/86 ノートで順序完全一致</text>`;
  return s + `</svg>`;
}

function figScatter() {
  const PW = 180, PH = 140;
  const meta = [
    ["PCA", "fig-teal", "回転 0°", "相関 0.69"],
    ["MDS", "fig-blue", "回転 159°+鏡映", "相関 0.73"],
    ["t-SNE", "fig-amber", "回転 61°", "相関 0.51"],
  ];
  const key = { PCA: "PCA", MDS: "MDS", "t-SNE": "t-SNE" };
  const scaled = (Y, ox, oy, color) => {
    let mx = 0; for (const p of Y) mx = Math.max(mx, Math.abs(p[0]), Math.abs(p[1]));
    const sc = (Math.min(PW, PH) / 2 - 8) / mx, cx = ox + PW / 2, cy = oy + PH / 2;
    return Y.map((p) => `<circle cx="${(cx + p[0] * sc).toFixed(1)}" cy="${(cy + p[1] * sc).toFixed(1)}" r="2.2" fill="var(--${color})"/>`).join("");
  };
  let s = `<svg viewBox="0 0 680 560" width="100%" role="img" aria-label="two-seed projections PCA MDS t-SNE">`;
  s += T(210, 44, 12, "fig-muted") + `seed 1</text>` + T(470, 44, 12, "fig-muted") + `seed 2</text>`;
  const yTop = { PCA: 60, MDS: 230, "t-SNE": 400 };
  meta.forEach((m) => {
    const y = yTop[m[0]], cy = y + 64;
    s += T(30, cy, 14, "fig-text", "start") + m[0] + `</text>`;
    s += T(30, cy + 20, 12, "fig-muted", "start") + m[2] + `</text>`;
    s += T(30, cy + 38, 12, "fig-muted", "start") + m[3] + `</text>`;
    s += `<rect x="120" y="${y}" width="180" height="140" rx="6" fill="none" stroke="var(--fig-grid)"/><rect x="380" y="${y}" width="180" height="140" rx="6" fill="none" stroke="var(--fig-grid)"/>`;
    s += scaled(proj.proj[key[m[0]]].s1, 120, y, m[1]) + scaled(proj.proj[key[m[0]]].s2, 380, y, m[1]);
  });
  s += T(340, 544, 12, "fig-muted") + `PCAは2 seedで完全一致（回転しない）。MDS・t-SNEは向きも構造も毎回変わる</text>`;
  return s + `</svg>`;
}

function figRadar() {
  const cx = 250, cy = 210, maxR = 175;
  let s = `<svg viewBox="0 0 680 430" width="100%" role="img" aria-label="polar radar: radius is distinctiveness percentile, angle from projection">`;
  for (const pct of [0.3, 0.6, 0.9, 1.0]) {
    s += `<circle cx="${cx}" cy="${cy}" r="${(pct * maxR).toFixed(1)}" fill="none" stroke="var(--fig-grid)" stroke-dasharray="${pct === 1 ? "0" : "3 3"}"/>`;
    if (pct < 1) s += T(cx, (cy - pct * maxR - 4).toFixed(1), 11, "fig-muted") + (pct * 100) + `%</text>`;
  }
  const band = (p) => (p > 0.66 ? "fig-amber" : p > 0.33 ? "fig-blue" : "fig-teal");
  rad.radiusPct.forEach((p, i) => {
    const x = cx + p * maxR * Math.cos(rad.angle[i]);
    const y = cy + p * maxR * Math.sin(rad.angle[i]);
    s += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="var(--${band(p)})" fill-opacity="0.85"/>`;
  });
  s += `<circle cx="${cx}" cy="${cy}" r="2.6" fill="var(--fig-muted)"/>`;
  s += T(475, 150, 13, "fig-text", "start") + `半径 = 際立ち度の順位</text>`;
  s += T(475, 172, 12, "fig-muted", "start") + `外周ほど際立つ / 未踏</text>`;
  s += T(475, 190, 12, "fig-muted", "start") + `中心ほど中心的テーマ</text>`;
  s += T(475, 224, 13, "fig-text", "start") + `角度 = 投影（PCA）</text>`;
  s += T(475, 246, 12, "fig-muted", "start") + `任意だが近傍を保存</text>`;
  s += T(475, 280, 12, "fig-teal", "start") + `● 内側 30%</text>` + T(475, 300, 12, "fig-blue", "start") + `● 中間</text>` + T(475, 320, 12, "fig-amber", "start") + `● 外側 33%</text>`;
  return s + `</svg>`;
}

function figKDE() {
  const W = 50, H = 36;
  const { grid, max, bbox } = gridDensity(den.pca, den.bandwidth, W, H);
  const [x0, y0, x1, y1] = bbox;
  const ax = 60, ay = 40, aw = 500, ah = 300;
  const sx = (x) => ax + (x - x0) / (x1 - x0) * aw;
  const sy = (y) => ay + ah - (y - y0) / (y1 - y0) * ah;
  const cw = aw / (W - 1), ch = ah / (H - 1);
  let cells = "";
  for (let gy = 0; gy < H; gy++)
    for (let gx = 0; gx < W; gx++) {
      const o = Math.pow(grid[gy][gx] / max, 0.75);
      if (o < 0.04) continue;
      const px = x0 + gx / (W - 1) * (x1 - x0), py = y0 + gy / (H - 1) * (y1 - y0);
      cells += `<rect x="${(sx(px) - cw / 2).toFixed(1)}" y="${(sy(py) - ch / 2).toFixed(1)}" width="${(cw + 0.6).toFixed(1)}" height="${(ch + 0.6).toFixed(1)}" fill="var(--fig-amber)" fill-opacity="${o.toFixed(2)}"/>`;
    }
  let dots = "";
  for (const p of den.pca) dots += `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="1.9" fill="var(--fig-text)" fill-opacity="0.55"/>`;
  let s = `<svg viewBox="0 0 680 380" width="100%" role="img" aria-label="KDE density over the PCA projection">`;
  s += `<rect x="${ax}" y="${ay}" width="${aw}" height="${ah}" fill="none" stroke="var(--fig-grid)"/>`;
  s += cells + dots;
  s += T(585, 150, 12, "fig-amber", "start") + `● hot（密）</text>` + T(585, 170, 12, "fig-muted", "start") + `justice / value /</text>` + T(585, 186, 12, "fig-muted", "start") + `self-management</text>`;
  s += T(585, 220, 12, "fig-text", "start") + `疎（cold）</text>` + T(585, 240, 12, "fig-muted", "start") + `ai-human / writing /</text>` + T(585, 256, 12, "fig-muted", "start") + `trading（多様で散在）</text>`;
  s += T(310, 368, 12, "fig-muted") + `PCA投影上のKDE（距離保存空間で計算）</text>`;
  return s + `</svg>`;
}

function figCluster() {
  const x0 = 250, w = 330, rows = [
    ["高次元クラスタ vs 人手タグ", clu.ari.hiHuman, "fig-teal"],
    ["2Dクラスタ vs 人手タグ", clu.ari.loHuman, "fig-amber"],
    ["高次元 vs 2Dクラスタ", clu.ari.hiLo, "fig-blue"],
  ];
  let s = `<svg viewBox="0 0 680 220" width="100%" role="img" aria-label="adjusted Rand index comparison">`;
  rows.forEach((r, i) => {
    const y = 44 + i * 46;
    s += T(x0 - 12, y + 17, 12, "fig-muted", "end") + r[0] + `</text>`;
    s += `<rect x="${x0}" y="${y}" width="${w}" height="26" rx="4" fill="var(--fig-grid)" fill-opacity="0.25"/>`;
    s += `<rect x="${x0}" y="${y}" width="${(r[1] * w).toFixed(1)}" height="26" rx="4" fill="var(--${r[2]})"/>`;
    s += T(x0 + r[1] * w + 8, y + 17, 13, "fig-text", "start") + r[1].toFixed(3) + `</text>`;
  });
  s += `<line x1="${x0}" y1="188" x2="${x0 + w}" y2="188" stroke="var(--fig-grid)"/>`;
  for (const v of [0, 0.5, 1]) s += T(x0 + v * w, 204, 11, "fig-muted") + v + `</text>`;
  s += T(x0 + w / 2, 24, 12, "fig-muted") + `ARI（1 = 完全一致, 0 = ランダム）</text>`;
  return s + `</svg>`;
}

function figTrend() {
  const A = trend.results.MDS.A, B = trend.results.MDS.B;
  // optimal rotation of B onto A (Kabsch 2D, allow reflection)
  let m00 = 0, m01 = 0, m10 = 0, m11 = 0;
  for (let i = 0; i < A.length; i++) { m00 += A[i][0] * B[i][0]; m01 += A[i][0] * B[i][1]; m10 += A[i][1] * B[i][0]; m11 += A[i][1] * B[i][1]; }
  const ang = Math.atan2(m10 - m01, m00 + m11), cc = Math.cos(ang), ss = Math.sin(ang);
  const rot = (flip) => B.map((p) => { const by = flip ? -p[1] : p[1]; return [cc * p[0] - ss * by, ss * p[0] + cc * by]; });
  const resid = (P) => { let e = 0; for (let i = 0; i < A.length; i++) e += Math.hypot(A[i][0] - P[i][0], A[i][1] - P[i][1]); return e; };
  const Br = resid(rot(false)) < resid(rot(true)) ? rot(false) : rot(true);
  let mx = 0; for (const p of [...A, ...B, ...Br]) mx = Math.max(mx, Math.abs(p[0]), Math.abs(p[1]));
  const panel = (Bset, ox, title) => {
    const cx = ox + 140, cy = 200, sc = 120 / mx;
    let g = `<rect x="${ox + 20}" y="80" width="240" height="240" rx="8" fill="none" stroke="var(--fig-grid)"/>`;
    g += T(ox + 140, 68, 13, "fig-text") + title + `</text>`;
    for (let i = 0; i < A.length; i++) {
      const ax = cx + A[i][0] * sc, ay = cy + A[i][1] * sc, bx = cx + Bset[i][0] * sc, by = cy + Bset[i][1] * sc;
      g += `<line x1="${ax.toFixed(1)}" y1="${ay.toFixed(1)}" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}" stroke="var(--fig-grid)" stroke-width="0.7"/>`;
    }
    for (const p of A) g += `<circle cx="${(cx + p[0] * sc).toFixed(1)}" cy="${(cy + p[1] * sc).toFixed(1)}" r="2.2" fill="var(--fig-blue)"/>`;
    for (const p of Bset) g += `<circle cx="${(cx + p[0] * sc).toFixed(1)}" cy="${(cy + p[1] * sc).toFixed(1)}" r="2.2" fill="var(--fig-amber)"/>`;
    return g;
  };
  let s = `<svg viewBox="0 0 680 360" width="100%" role="img" aria-label="MDS snapshots before and after Procrustes alignment">`;
  s += panel(B, 40, "アライン無し（回転107°）") + panel(Br, 360, "Procrustes後（残差だけ）");
  s += `<rect x="300" y="335" width="12" height="12" rx="3" fill="var(--fig-blue)"/>` + T(318, 345, 12, "fig-muted", "start") + `古い群</text>`;
  s += `<rect x="400" y="335" width="12" height="12" rx="3" fill="var(--fig-amber)"/>` + T(418, 345, 12, "fig-muted", "start") + `全体（同じノート）</text>`;
  return s + `</svg>`;
}

// ---------- page assembly ----------
const CSS = `
:root{--bg:#f5f7f8;--surface:#ffffff;--tile:#eef2f3;--ink:#161a1e;--muted:#59636d;--hairline:#e2e7ea;--accent:#12876a;--rule:#d3dade;
--fig-text:#2a3138;--fig-muted:#6b757e;--fig-grid:#c4ccd2;--fig-teal:#12876a;--fig-blue:#2f6fb0;--fig-amber:#b26a12;
--mono:ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif;}
@media (prefers-color-scheme:dark){:root{--bg:#13161a;--surface:#1b2126;--tile:#20272d;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#43c39c;--rule:#2a323a;
--fig-text:#dfe5ea;--fig-muted:#8b96a0;--fig-grid:#3a434c;--fig-teal:#43c39c;--fig-blue:#5aa0e0;--fig-amber:#e0a44e;}}
:root[data-theme="dark"]{--bg:#13161a;--surface:#1b2126;--tile:#20272d;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#43c39c;--rule:#2a323a;--fig-text:#dfe5ea;--fig-muted:#8b96a0;--fig-grid:#3a434c;--fig-teal:#43c39c;--fig-blue:#5aa0e0;--fig-amber:#e0a44e;}
:root[data-theme="light"]{--bg:#f5f7f8;--surface:#ffffff;--tile:#eef2f3;--ink:#161a1e;--muted:#59636d;--hairline:#e2e7ea;--accent:#12876a;--rule:#d3dade;--fig-text:#2a3138;--fig-muted:#6b757e;--fig-grid:#c4ccd2;--fig-teal:#12876a;--fig-blue:#2f6fb0;--fig-amber:#b26a12;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.7;-webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto;padding:56px 24px 80px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 12px}
h1{font-size:34px;line-height:1.15;margin:0 0 10px;letter-spacing:-.02em;text-wrap:balance;font-weight:600}
h2{font-size:15px;font-family:var(--mono);letter-spacing:.02em;margin:0;font-weight:600}
.lead{font-size:18px;color:var(--muted);margin:0 0 28px;max-width:64ch}
.meta{display:flex;flex-wrap:wrap;gap:18px;font-family:var(--mono);font-size:12.5px;color:var(--muted);border-top:1px solid var(--hairline);border-bottom:1px solid var(--hairline);padding:12px 0;margin:0 0 40px}
.meta b{color:var(--ink);font-weight:600}
.stage{border-top:2px solid var(--rule);padding-top:22px;margin:44px 0 0}
.stage-head{display:flex;align-items:center;gap:12px;margin:0 0 6px}
.code{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--surface);background:var(--accent);padding:3px 9px;border-radius:6px;letter-spacing:.04em}
.chip{font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--hairline);color:var(--muted)}
.chip.done{color:var(--accent);border-color:var(--accent)}
p{margin:14px 0;max-width:66ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0}
.tile{background:var(--tile);border-radius:10px;padding:14px 16px}
.tile .k{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin:0 0 6px;letter-spacing:.03em}
.tile .v{font-family:var(--mono);font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
figure{margin:26px 0;background:var(--surface);border:1px solid var(--hairline);border-radius:14px;padding:18px 18px 10px;overflow-x:auto}
figcaption{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:6px;padding:0 4px}
.callout{border-left:3px solid var(--accent);background:var(--tile);border-radius:0 8px 8px 0;padding:12px 16px;margin:20px 0;font-size:15.5px}
.callout b{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--hairline)}
th{font-family:var(--mono);font-size:12px;color:var(--muted);font-weight:600}
td.mono,.mono{font-family:var(--mono)}
.verdict{font-family:var(--mono);font-weight:600}
.ok{color:var(--accent)} .no{color:#c0562f} .mid{color:var(--fig-amber)}
.pending{opacity:.62}
.pending li{font-family:var(--mono);font-size:13px;color:var(--muted);margin:5px 0;list-style:none}
.pending li::before{content:"○ ";color:var(--muted)}
.foot{margin-top:56px;border-top:1px solid var(--hairline);padding-top:18px;font-family:var(--mono);font-size:12px;color:var(--muted)}
`;

const html = `<style>${CSS}</style>
<div class="wrap">
<p class="eyebrow">dimensionality reduction · case study</p>
<h1>similarity-radar</h1>
<p class="lead">高次元類似度から2Dの「地形／Radar」を自作する学習ケーススタディ。公開情報から推定した商用技術ランドスケープ製品の構成を一般化し、6段パイプラインとして自分の手で再構築して、各段で予測と実測を突き合わせた記録。</p>
<div class="meta"><span>母集団 <b>20_Permanent 86ノート</b></span><span>埋め込み <b>bge-micro-v2 / 384d</b></span><span>実行系 <b>Node.js</b></span><span>更新 <b>2026-07-22</b></span></div>

<p>読み方: 各段は <span class="mono">実装 → 予測 → 実測 → 精緻化</span> の順で進む。図はすべて自分の86ノートの実データ。結論だけでなく「なぜその手法を選ぶか」を即答できる状態を目的にしている。</p>
<figure>${figPipeline()}<figcaption>図0 — 6段パイプラインの現在地。Part A（KB）は A0–A6 完了。</figcaption></figure>

<div class="callout" style="border-left-color:var(--accent)">
<p style="margin:0 0 8px"><b>Part A まとめ（要点）</b></p>
<ul style="margin:0;padding-left:1.1em;font-size:14.5px;line-height:1.9">
<li>A1 類似度が0.78–0.99に潰れる主因は内容でなく<b>cone effect</b>（‖平均‖=0.94）。中心化で全域に展開。</li>
<li>A2 回転不定性は<b>初期化の問題</b>（MDSも159°回る＋<b>鏡映</b>／PCAはスペクトル法で0°）。t-SNEは大域距離が最下位(0.51)。</li>
<li>A3 意味は<b>回転不変量（半径＝際立ち度の順位）にのみ</b>載せ、角度は捨てる。半径は再現性corr 1.0。</li>
<li>A4 hot＝<b>集中度であって件数ではない</b>。whitespace＝未踏とは限らない。罠: Radar上の密度は1/rアーティファクト。</li>
<li>A5 クラスタは<b>高次元で</b>（2Dだと人手一致が0.45→0.25に半減）。人手との<b>食い違いに価値</b>。</li>
<li>A6 経年比較は<b>PCA＋Procrustes</b>（PCAでも母集団依存で枠が3.7°ずれる／MDSは107°で不適）。</li>
</ul>
<p style="margin:10px 0 0;font-size:14.5px">→ 3つの主張をPermanentに結晶: 回転不定性の正体／回転不変量にのみ意味を載せる／クラスタは高次元で。</p>
</div>

<section class="stage"><div class="stage-head"><span class="code">A0</span><h2>特徴行列</h2><span class="chip done">done</span></div>
<p>Smart Connectionsの <span class="mono">.ajson</span>（追記型JSON、last-write-wins）から、20_Permanent配下でベクトルを持つノートを抽出し行列 X を構築。</p>
<div class="tiles"><div class="tile"><p class="k">notes (n)</p><p class="v">86</p></div><div class="tile"><p class="k">dim</p><p class="v">384</p></div><div class="tile"><p class="k">L2 norm</p><p class="v">1.000</p></div></div>
<div class="callout"><b>発見</b> ベクトルはL2正規化済み（単位球上）。よって <span class="mono">‖a−b‖² = 2−2cos</span>。Euclidと cosine は同順位になり、距離選択は「順位」ではなく「投影のストレス関数が何を最適化するか」の問題になる。</div>
</section>

<section class="stage"><div class="stage-head"><span class="code">A1</span><h2>距離・異方性</h2><span class="chip done">done</span></div>
<p>全3655ペアのコサイン類似度は <b>0.78–0.99</b> に圧縮（中央値0.888）。原因は内容ではなく、埋め込みが狭い円錐に密集する<b>異方性（cone effect）</b>。平均ベクトルのノルムが0.942＝各ベクトルの約89%が内容と無関係な共通方向を向いている。</p>
<figure>${figCone()}<figcaption>図1 — 単位球上の86ベクトルは狭い円錐に密集。共通方向が類似度を高く潰す。</figcaption></figure>
<p>共通方向を引く（中心化する）だけで、類似度は全域に展開する。圧縮は内容ではなく cone だった、の決定的証拠。</p>
<figure>${figHist()}<figcaption>図2 — 中心化前（青、右端の塔）と後（橙、0付近を中心に展開）。中央値 0.89 → −0.05。</figcaption></figure>
<figure>${figRank()}<figcaption>図3 — d=√(2−2cos) は単調減少。近傍の順位は全域で不変（86/86実証）。</figcaption></figure>
<div class="callout"><b>決定</b> 投影の前に中心化する。共通方向は情報ゼロの1軸を占めているだけ。中心化後のノルム（0.22–0.43）は「重心からの距離＝際立ち度」で、A3の半径座標の生データになる。</div>
</section>

<section class="stage"><div class="stage-head"><span class="code">A2</span><h2>投影の三つ巴</h2><span class="chip done">done</span></div>
<p>中心化した行列に PCA・計量MDS(SMACOF)・t-SNE を掛け、(I) 2 seed間の回転／構造変化、(II) 高次元距離との相関（Shepard）を測る。</p>
<figure>${figScatter()}<figcaption>図4 — 3手法 × 2 seed。PCAは左右完全一致。MDS・t-SNEは向きも構造も毎回変わる。</figcaption></figure>
<table><thead><tr><th>観点</th><th>予測</th><th>実測</th><th>判定</th></tr></thead><tbody>
<tr><td>回転(2 seed)</td><td class="mono">t-SNE&gt;MDS&gt;PCA</td><td class="mono">PCA 0° / MDS 159°+鏡映 / t-SNE 61°</td><td class="verdict mid">△</td></tr>
<tr><td>構造変化(disparity)</td><td class="mono">t-SNE&gt;MDS&gt;PCA</td><td class="mono">0.96 &gt; 0.62 &gt; ~0</td><td class="verdict ok">✓</td></tr>
<tr><td>大域相関(hiD,2D)</td><td class="mono">t-SNE&gt;MDS&gt;PCA</td><td class="mono">MDS 0.73 &gt; PCA 0.69 &gt; t-SNE 0.51</td><td class="verdict no">✗ 逆転</td></tr>
</tbody></table>
<div class="callout"><b>精緻化</b> 回転不定性は手法の線形/非線形では決まらない。距離を保存する目的関数はどれも回転不変なので、乱数初期化なら向きは任意になり<b>計量MDSも回る</b>。PCAが安定なのは線形だからではなく、<b>スペクトル法で軸が決定的に定まる</b>から。t-SNEの綺麗なクラスタは大域距離が最下位＝「見栄え vs 距離忠実度」。距離保存型ランドスケープはMDS系を候補とし、回転は半径など回転不変量で別途無害化する（→ A3）。</div>
<div class="callout" style="border-left-color:var(--accent)"><b>精緻化（Part C・後日）</b> raw stress が不変なのは回転だけではない。<b>鏡映</b>も距離を保存するので、乱数初期化のMDSは向きに加えて<b>裏返り</b>も生む。実測でも MDS の2 seed差は回転 159° <b>＋鏡映</b>で、鏡映を含めて整列すると構造Δは 0.62（t-SNE 0.96 は鏡映なし）。つまり2D座標の非同一性は「回転だけ無害化すれば済む」問題ではなく、<b>直交群 O(2) 全体</b>（回転＋鏡映）に対する不変量を読むしかない。半径はその両方に対して不変なので、A3の設計はここでも生き残る。</div>
<div class="callout" style="border-left-color:var(--muted)"><b>訂正（Part C）</b> 本節は当初 <span class="mono">MDS 171° / disparity 0.81</span> と記載していた。自作Procrustesが鏡映側の分岐で最適回転角を再計算せず、非鏡映の角度を流用していたため、鏡映を含む差の残差を過大に見積もっていた（合成テスト: 鏡映コピーの disparity が 1.34 → 2.8e-32 に修正）。修正後の値は <span class="mono">159° / 0.62</span>。順序 t-SNE &gt; MDS &gt; PCA、大域相関（0.726 / 0.690 / 0.511）、およびA2の結論はいずれも不変。A6（PCA 3.7° / MDS 107°）にも影響なし。</div>
</section>

<section class="stage"><div class="stage-head"><span class="code">A3</span><h2>回転不変な読出し</h2><span class="chip done">done</span></div>
<p>A2で「向きは任意」と分かった。ならば意味を<b>回転不変な量にだけ</b>載せる。半径 = 高次元の際立ち度 <span class="mono">‖centered‖</span>（＝重心からの距離）をパーセンタイル順位に変換。角度だけを投影から取る。半径は固定データ由来なので投影に一切依存しない。</p>
<figure>${figRadar()}<figcaption>図5 — Radar。半径＝際立ち度の順位（外周ほど際立つ／未踏）、角度＝PCA。同心円は30/60/90%。</figcaption></figure>
<div class="tiles"><div class="tile"><p class="k">半径 corr(seed/手法)</p><p class="v">1.000</p></div><div class="tile"><p class="k">投影2D半径 corr(seed)</p><p class="v">0.960</p></div><div class="tile"><p class="k">2D半径 vs 際立ち度</p><p class="v">0.958</p></div></div>
<div class="callout"><b>発見</b> 回転と多峰性が壊すのは主に<b>角度</b>で、<b>半径（中心性の順位）は頑健</b>（MDSはdisparity 0.62でも半径corr 0.96）。しかも A2 の精緻化どおり、seed間の差は回転だけでなく<b>鏡映</b>を含む。鏡映は角度の符号を反転させる一方、原点からの距離は一切変えない ―― つまり角度を壊し半径を保つ変換がもう一つ増えたことになり、「半径に意味・角度は捨てる」設計は<b>三重に</b>正当化される。座標系の任意性は O(2)（回転＋鏡映）全体であり、半径はその不変量。高次元の際立ち度を半径にすれば再現性は完全（corr 1）。パーセンタイル順位により、圧縮した生値（0.22–0.43）でも同心円が均等に埋まる。</div>
<p>配置の妥当性: 外周＝<span class="mono">TSVF援用の方法論的注意</span>(100%)・<span class="mono">間違いはクリエイティビティの裏返し</span>(99%) のメタ的外れ値。中心＝<span class="mono">価値は関係性のなかで生じる</span>(1%)・<span class="mono">主体依存性はまず劣化要因</span>(0%) の中心的テーマ。</p>
</section>

<section class="stage"><div class="stage-head"><span class="code">A4</span><h2>密度KDE</h2><span class="chip done">done</span></div>
<p>投影後の点群にKDEをかけ、密集（hot）と疎（whitespace）を出す。ただし<b>KDEは距離保存投影（PCA）上で計算する</b>——パーセンタイルRadar上では半径が一様なため面積密度が <span class="mono">1/r</span> になり、中心に嘘のホットスポットが出る（実測 中心135 vs 外周15）。</p>
<figure>${figKDE()}<figcaption>図6 — PCA投影上のKDE。hot＝倫理/価値/自己管理の塊。ai-human等は最大タグだが多様で散在＝疎。</figcaption></figure>
<div class="callout"><b>訂正1（hot）</b> 密度は「ノート数」ではなく「集中度」。最大タグ ai-human(26) は最密ではなく<b>疎</b>——内容が多様で散らばるから。密なのは justice/value/self-management の凝集した塊。density と半径の相関は −0.25（中心的テーマがやや密）。</div>
<div class="callout"><b>訂正2（whitespace）</b> 疎＝「未踏の次テーマ」とは限らない。ここでの疎領域は ai-human 等の<b>よく書いたが多様な</b>領域で、未踏ではない。無関係クラスタ間の隙間は<b>埋めても無意味</b>（トレード↔因果の橋渡しはナンセンス）。商用技術ランドスケープで使われる「whitespace＝機会」という解釈は特許landscape固有で、個人KBでは解釈が割れる。</div>
</section>

<section class="stage"><div class="stage-head"><span class="code">A5</span><h2>クラスタと人手の食い違い</h2><span class="chip done">done</span></div>
<p>点群を k-means でクラスタリング。<b>高次元(384d)</b>と<b>2D投影</b>で別々にやり、人手タグ（primary topic tag, k=9）との一致をARIで測る。</p>
<figure>${figCluster()}<figcaption>図7 — ARI比較。2D投影してからクラスタリングすると、人手構造との一致が0.45→0.25にほぼ半減。</figcaption></figure>
<div class="callout"><b>Q1 高次元 vs 2D</b> 投影してからクラスタリングは人手タグとの一致を<b>約半分に劣化</b>させる（0.448→0.247）。次元圧縮で失われた情報がそのままクラスタ品質の劣化になる。<b>クラスタは高次元で、投影は表示だけ。</b></div>
<p><b>Q2 アルゴリズムと人手の食い違い＝価値。</b> 高次元クラスタの多数派タグが、そのノート自身のタグと違う＝異テーマを跨ぐ架橋。全${clu.bridges.length}件のうち代表例:</p>
<table><thead><tr><th>ノートのタグ</th><th>クラスタの多数派</th><th>ノート</th></tr></thead><tbody>
${clu.bridges.slice(0, 5).map((b) => `<tr><td class="mono">${b.tag}</td><td class="mono">${b.dom}</td><td>${b.label}</td></tr>`).join("")}
</tbody></table>
<div class="callout"><b>発見</b> 一致は退屈、<b>食い違いに情報</b>。trading×ai-human、value×causality、engineering×self-management——これらは自分のノート「価値はリンクと埋め込みの食い違いにある」「価値ある架橋は異Indexクラスタを跨ぐ」の実証。人手Indexとアルゴリズムの差分こそが創造的架橋のカタログ。</div>
</section>

<section class="stage"><div class="stage-head"><span class="code">A6</span><h2>経年トレンドとアライメント</h2><span class="chip done">done</span></div>
<p>日付で古い群(43)と全体(86)に分け、各スナップショットを<b>独立に</b>投影。共通ノートが2枚の間でどれだけ「動いて見えるか」を、アライン前後で測る。ここが回転不定性の最難所（経年比較）。</p>
<figure>${figTrend()}<figcaption>図8 — MDSの2スナップショット。左＝独立投影は枠が107°回り大きくズレて見える。右＝Procrustes後、同じノートはほぼ重なる（動いていない）。</figcaption></figure>
<table><thead><tr><th>手法</th><th>raw変位</th><th>アライン後</th><th>枠の寄与</th><th>回転角</th></tr></thead><tbody>
<tr><td class="mono">PCA</td><td class="mono">0.016</td><td class="mono">0.013</td><td class="mono">1.2×</td><td class="mono">3.7°</td></tr>
<tr><td class="mono">MDS</td><td class="mono">0.223</td><td class="mono">0.066</td><td class="mono">3.4×</td><td class="mono">106.7°</td></tr>
</tbody></table>
<div class="callout"><b>発見</b> 独立投影の見かけの移動は大半が<b>枠（回転・鏡映・スケール）</b>。Procrustesがそれを消して残差＝真の変化を出す。<b>PCAでも枠は3.7°ずれる（≠0）</b>——母集団が変われば主軸も変わるから。ただし分散構造が安定なので小さい。よって経年比較は<b>PCA（枠ほぼ安定）＋正確性のためProcrustes</b>が正解。MDSは枠が任意（107°）で経年比較に不向き。これは一般にトレンドラインを実装するときの要請になる。</div>
</section>

<section class="stage"><div class="stage-head"><span class="code">A7</span><h2>結晶（Part A 完了）</h2><span class="chip done">done</span></div>
<p>Part Aの発見から、自分の言葉で3つのPermanentノートを結晶させた（20_Permanent）:</p>
<ul style="font-size:14.5px;line-height:1.9">
<li>回転不定性は手法の線形非線形ではなく初期化の乱数性と軸の決定性で決まる</li>
<li>距離保存の2Dマップでは意味を回転不変量にのみ載せるべきである</li>
<li>クラスタリングは次元圧縮後の2Dではなく高次元で行うべきである</li>
</ul>
<p>いずれも自分の86ノートの実データで検証済み。そして全レイヤーを1つに統合した <a href="https://claude.ai/code/artifact/db538c92-0312-4223-9360-4e0849151642"><b>Knowledge Base Radar</b></a>（MDS土台＋同心円＋KDE＋高次元クラスタ色＋架橋強調・インタラクティブ）を作成。次はこのパイプラインを全く別ドメイン（日本株）へ機序移植し、転移で定着を確認する。</p>
</section>

<section class="stage"><div class="stage-head"><span class="code">B0–B6</span><h2>Part B: 日本株へ機序移植</h2><span class="chip done">done</span></div>
<p>同じパイプラインを全く別ドメイン（日経225大型68銘柄）へ移植。特徴＝9ファクター（value/quality/growth/size）。成果物: <a href="https://claude.ai/code/artifact/7301c539-f87c-4909-ab30-855869448195"><b>日本株ファクター Radar</b></a>（MDS土台・適応KDE等高線・セクター色・食い違い強調）。</p>
<div class="callout"><b>B0/B1 転移の要</b> KBの異方性(cone)＝株の<b>特徴量相関</b>（pbr×roa 0.85）。KBは中心化、株は<b>白色化</b>で除去（相関0.85→0.00）。転移した結論: 大域忠実度は <b>MDS 0.79 &gt; PCA 0.57 &gt; t-SNE 0.54</b>（KBと同順）。</div>
<div class="callout" style="border-left-color:var(--accent)"><b>非自明な発見（機序の相互作用）</b> <b>白色化は共分散を単位行列（等方）にするため、PCAの分散最大軸が縮退し、PCAが回り出す（56.8°、KBでは0°）</b>。KBでPCAが安定だったのは n≪d で白色化できず中心化止まり＝分散構造を温存したから。<b>機序は移植できても、他の選択との相互作用は移植できない</b>——だから株Radarは（PCAでなく）MDS土台一択。</div>
<p><b>B6 スクリーニング解釈:</b> 外周＝際立つファクター profile（SoftBank G: ROE33%×PER6.5／日本製鉄: PER182×ROE1%＝シクリカル底／Advantest: PBR27×ROE58%＝極端グロース）、中心＝平凡（KDDI・ENEOS）。食い違い${'37'}件＝「セクターは違うがファクター profile が似ている」銘柄＝分散・ペア候補。hot＝過密ファクター領域、whitespace＝どの銘柄も居ないファクター組合せ。</p>
<div class="callout"><b>B7 転移の総括</b> <b>同じ:</b> 冗長方向を潰して独立変動だけを距離にする／距離忠実はMDS・t-SNEは大域が弱い／回転は同心円（回転不変な半径）で無害化。<b>変えた:</b> KBは中心化(n≪d)・株は白色化(n&gt;d)／KBはPCAで安定な角度が取れたが株は白色化でPCAが不安定化しMDS必須。<b>機序移植の教科書例</b>——[機序の一致で正当化]の実証。</div>
<p><b>時系列ドリフト（A6の実時系列版・否定的結果）:</b> 株価からモメンタム＋ボラを2時点で算出し独立MDS→<b>Procrustesで枠 176° を除去</b>（→ <a href="https://claude.ai/code/artifact/c915a8f5-97ba-42ca-9810-e7c4a7f00b44">ドリフト矢印マップ</a>）。だが<b>示唆は薄かった</b>。これ自体が学び。</p>
<div class="callout" style="border-left-color:var(--accent)"><b>締めの発見（問いの型は移植できない）</b> 道具（ドリフトマップ）は移植できたが、問いの型が移植できていなかった。KBの時系列＝「思考は新領域へ広がったか」＝<b>幾何の問い</b>（マップが正解）。株の時系列＝「今どのファクターが効くレジームか」＝<b>リターンの問い</b>（マップ不適）。相対順位は安定でマップが動かず、信号はファクターリターンに宿る。しかもProcrustesはレジーム転換（ほぼ剛体運動）を枠として消す。B1の「機序は移植できても相互作用は移植できない」の一段上：<b>問いの型がドメインで変われば正しい道具も変わる</b>（結晶: 20_Permanent「道具だけでなく問いの型の一致まで確かめるべき」）。株で示唆があるのは横断面のRadar（際立つprofile・whitespace）のほう。</div>
</section>

<p class="foot">code: Projects/similarity-radar（MIT, 公開可）／ このドキュメントとKBデータは非公開（gitignore）。図はすべて自分の86 Permanentノートの実データから生成。</p>
</div>`;

const out = path.join(ROOT, "out", "kb", "casestudy.html");
fs.writeFileSync(out, html);
console.log("wrote", path.relative(ROOT, out), html.length, "bytes");
