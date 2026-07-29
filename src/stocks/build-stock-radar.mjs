// Build the JP-equity Radar (interactive HTML) — the same layered pipeline as the
// KB radar, transferred: MDS base on Mahalanobis (whitened) distance, percentile
// rings (rotation-invariant radius = factor-profile distinctiveness), adaptive-KDE
// contour terrain, color = sector, black ring = "bridge" (factor cluster != sector).
//
// Run: node src/stocks/build-stock-radar.mjs   (after 01-project.mjs)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { smacof } from "../lib/projection/mds.mjs";
import { kmeans } from "../lib/kmeans.mjs";
import { adaptiveGridDensity, contourSegments, scottBandwidth } from "../lib/kde.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "whitened.json"), "utf8"));
const W = S.W, n = W.length, d = S.features.length;

// Mahalanobis (= Euclidean on whitened) distance matrix
const D = Array.from({ length: n }, () => new Float64Array(n));
for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
  let s = 0; for (let k = 0; k < d; k++) { const dd = W[i][k] - W[j][k]; s += dd * dd; }
  D[i][j] = D[j][i] = Math.sqrt(s);
}
let Y = smacof(D, 1, 400);
let mx = 0, my = 0; for (const p of Y) { mx += p[0] / n; my += p[1] / n; }
Y = Y.map((p) => [p[0] - mx, p[1] - my]);
let Sxx = 0, Syy = 0, Sxy = 0;
for (const p of Y) { Sxx += p[0] * p[0]; Syy += p[1] * p[1]; Sxy += p[0] * p[1]; }
const ang = 0.5 * Math.atan2(2 * Sxy, Sxx - Syy), ca = Math.cos(ang), sa = Math.sin(ang);
Y = Y.map((p) => [ca * p[0] + sa * p[1], -sa * p[0] + ca * p[1]]);

const rDist = Y.map((p) => Math.hypot(p[0], p[1]));
const sortedR = [...rDist].sort((a, b) => a - b);
const ringR = [0.3, 0.6, 0.9].map((q) => sortedR[Math.floor(q * (n - 1))]);
const maxR = sortedR[n - 1];

// clusters in whitened (factor) space; human label = sector
const sector = S.sectors;
const k = new Set(sector).size;
const { labels: cl } = kmeans(W, k, { seed: 3 });
const dom = {};
for (let c2 = 0; c2 < k; c2++) {
  const mem = [...cl.keys()].filter((i) => cl[i] === c2);
  const cnt = {}; for (const i of mem) cnt[sector[i]] = (cnt[sector[i]] ?? 0) + 1;
  dom[c2] = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0]?.[0];
}

const bw = scottBandwidth(Y);
const kde = adaptiveGridDensity(Y, bw, 80, 60);
const levelFracs = [0.08, 0.16, 0.27, 0.4, 0.55, 0.72, 0.88];
const contours = levelFracs.map((f, i) => ({
  opacity: +(0.16 + (0.5 - 0.16) * (i / (levelFracs.length - 1))).toFixed(3),
  segs: contourSegments(kde.grid, kde.bbox, f * kde.max),
}));

const palette = {
  "Industrials": ["#4a6274", "#8ba3b5"], "Consumer Cyclical": ["#c0562f", "#e0895a"],
  "Technology": ["#2f6fb0", "#5aa0e0"], "Communication Services": ["#7d54c9", "#b38ce8"],
  "Healthcare": ["#12876a", "#43c39c"], "Consumer Defensive": ["#78871a", "#bcc74e"],
  "Financial Services": ["#b0357f", "#e06ab0"], "Basic Materials": ["#b26a12", "#e0a44e"],
  "Real Estate": ["#0e8a9c", "#4ec3d4"], "Energy": ["#8a2f2f", "#d06a6a"], "Unknown": ["#888", "#aaa"],
};
const fmt = (x) => (x === null || x === undefined ? "—" : (Math.abs(x) < 1 ? x.toFixed(2) : x.toFixed(1)));
const points = Y.map((p, i) => ({
  x: p[0], y: p[1], sector: sector[i], name: S.names[i], ticker: S.tickers[i],
  bridge: sector[i] !== dom[cl[i]], domSector: dom[cl[i]],
  per: fmt(S.raw[i].per), pbr: fmt(S.raw[i].pbr), roe: S.raw[i].roe === null ? "—" : (S.raw[i].roe * 100).toFixed(1) + "%",
}));
const sectorsUsed = [...new Set(sector)].sort((a, b) => sector.filter((s) => s === b).length - sector.filter((s) => s === a).length);

const bundle = { points, contours, ringR, maxR, palette, sectors: sectorsUsed, bridgeCount: points.filter((p) => p.bridge).length, n };

const html = `<style>
:root{--bg:#f5f7f8;--surface:#fff;--ink:#161a1e;--muted:#59636d;--hairline:#e2e7ea;--accent:#2f6fb0;--ring:#c4ccd2;--terrain:#4a6274;
--mono:ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif;}
@media (prefers-color-scheme:dark){:root{--bg:#13161a;--surface:#1b2126;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#5aa0e0;--ring:#3a434c;--terrain:#8ba3b5;}}
:root[data-theme="dark"]{--bg:#13161a;--surface:#1b2126;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#5aa0e0;--ring:#3a434c;--terrain:#8ba3b5;}
:root[data-theme="light"]{--bg:#f5f7f8;--surface:#fff;--ink:#161a1e;--muted:#59636d;--hairline:#e2e7ea;--accent:#2f6fb0;--ring:#c4ccd2;--terrain:#4a6274;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6}
.wrap{max-width:900px;margin:0 auto;padding:32px 20px 60px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}
h1{font-size:26px;margin:0 0 6px;font-weight:600}.sub{color:var(--muted);margin:0 0 18px;font-size:15px}
.controls{display:flex;flex-wrap:wrap;gap:14px;font-family:var(--mono);font-size:13px;margin:0 0 14px;padding:12px 14px;background:var(--surface);border:1px solid var(--hairline);border-radius:10px}
.controls label{display:flex;align-items:center;gap:6px;cursor:pointer;color:var(--muted)}
.stage{position:relative;background:var(--surface);border:1px solid var(--hairline);border-radius:14px;overflow:hidden}
canvas{display:block;width:100%;height:auto}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12.5px;padding:6px 9px;border-radius:7px;max-width:280px;opacity:0;transition:opacity .1s;line-height:1.45}
.legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin:16px 0 0;font-family:var(--mono);font-size:12.5px}
.legend span{display:flex;align-items:center;gap:6px;color:var(--muted)}.sw{width:11px;height:11px;border-radius:3px}
.note{font-family:var(--mono);font-size:12px;color:var(--muted);margin:16px 0 0;line-height:1.7}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
</style>
<div class="wrap">
<h2 class="sr-only">Interactive Japanese equity radar of ${n} large/mid caps on an MDS projection of whitened factors, with density contours, percentile rings, and sector colors.</h2>
<p class="eyebrow">jp equity radar · MDS on Mahalanobis</p>
<h1>日本株ファクター Radar</h1>
<p class="sub">日経225主要${n}銘柄を、白色化した9ファクター（value/quality/growth/size）のマハラノビス距離でMDS配置。同心円＝際立ち度（平凡な profile ほど中心・特異なほど外周）、等高線＝適応KDE、色＝セクター、黒枠＝ファクター上のクラスタが本来のセクターと食い違う銘柄。軸に意味はない。</p>
<div class="controls">
<label><input type="checkbox" id="t-kde" checked> 等高線 (適応KDE)</label>
<label><input type="checkbox" id="t-rings" checked> 同心円 (際立ち度%)</label>
<label><input type="checkbox" id="t-bridge" checked> 食い違い銘柄を強調</label>
<label><input type="checkbox" id="t-labels"> セクター名</label>
</div>
<div class="stage"><canvas id="radar" role="img" aria-label="jp equity factor radar"></canvas><div class="tip" id="tip"></div></div>
<div class="legend" id="legend"></div>
<p class="note">土台=MDS（マハラノビス距離＝相関を除いたファクター距離。B1: 白色化でPCAは等方化して回るためMDS一択）。密度はMDS座標上で計算＝中心1/r罠なし。クラスタは白色化ファクター空間で算出。食い違い${bundle.bridgeCount}件＝「セクターは違うがファクター profile が似ている」銘柄＝分散やペア候補。KB Radarと同一パイプラインの機序移植。</p>
</div>
<script>
const DATA=${JSON.stringify(bundle)};
const cv=document.getElementById("radar"),tip=document.getElementById("tip");
const opt={kde:1,rings:1,bridge:1,labels:0};
function theme(){const r=document.documentElement.getAttribute("data-theme");if(r)return r;return matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}
function col(s){const c=DATA.palette[s]||DATA.palette.Unknown;return c[theme()==="dark"?1:0];}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
let W,H,cx,cy,scale;
function layout(){const cssW=cv.clientWidth,dpr=devicePixelRatio||1;W=cssW;H=Math.round(cssW*0.72);cv.width=W*dpr;cv.height=H*dpr;cv.getContext("2d").setTransform(dpr,0,0,dpr,0,0);const pad=46;scale=(Math.min(W,H)/2-pad)/DATA.maxR;cx=W/2;cy=H/2;}
function draw(){const ctx=cv.getContext("2d");ctx.clearRect(0,0,W,H);
 if(opt.kde){const t=css("--terrain");ctx.lineWidth=1;ctx.lineJoin="round";DATA.contours.forEach(c=>{ctx.globalAlpha=c.opacity;ctx.strokeStyle=t;ctx.beginPath();for(const s of c.segs){ctx.moveTo(cx+s[0]*scale,cy-s[1]*scale);ctx.lineTo(cx+s[2]*scale,cy-s[3]*scale);}ctx.stroke();});ctx.globalAlpha=1;}
 if(opt.rings){ctx.strokeStyle=css("--ring");ctx.fillStyle=css("--muted");ctx.font="11px "+css("--mono");ctx.textAlign="center";const labs=["30%","60%","90%"];DATA.ringR.forEach((r,i)=>{ctx.setLineDash(i<2?[3,3]:[]);ctx.beginPath();ctx.arc(cx,cy,r*scale,0,7);ctx.stroke();ctx.fillText(labs[i],cx,cy-r*scale-4);});ctx.setLineDash([]);}
 DATA.points.forEach(p=>{const sx=cx+p.x*scale,sy=cy-p.y*scale;ctx.beginPath();ctx.arc(sx,sy,5,0,7);ctx.fillStyle=col(p.sector);ctx.fill();if(opt.bridge&&p.bridge){ctx.lineWidth=1.6;ctx.strokeStyle=theme()==="dark"?"#fff":"#111";ctx.stroke();}});
 if(opt.labels){ctx.fillStyle=css("--ink");ctx.font="12px "+css("--sans");ctx.textAlign="center";const seen={};DATA.points.forEach(p=>{if(seen[p.sector])return;seen[p.sector]=1;ctx.fillText(p.sector,cx+p.x*scale,cy-p.y*scale-9);});}
}
function pick(mx,my){let best=-1,bd=1e9;DATA.points.forEach((p,i)=>{const sx=cx+p.x*scale,sy=cy-p.y*scale,dd=(sx-mx)**2+(sy-my)**2;if(dd<bd){bd=dd;best=i;}});return bd<220?best:-1;}
cv.addEventListener("mousemove",e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,i=pick(mx,my);if(i<0){tip.style.opacity=0;return;}const p=DATA.points[i];tip.innerHTML='<b>'+p.name+'</b> ('+p.ticker+')<br>'+p.sector+' · PER '+p.per+' · PBR '+p.pbr+' · ROE '+p.roe+(p.bridge?'<br><span style="opacity:.7">食い違い: profile は '+p.domSector+' に近い</span>':'');tip.style.opacity=1;tip.style.left=Math.min(mx+12,W-270)+"px";tip.style.top=(my+12)+"px";});
cv.addEventListener("mouseleave",()=>tip.style.opacity=0);
function legend(){document.getElementById("legend").innerHTML=DATA.sectors.map(s=>'<span><i class="sw" style="background:'+col(s)+'"></i>'+s+'</span>').join("")+'<span><i class="sw" style="border:1.6px solid '+(theme()==="dark"?"#fff":"#111")+';background:transparent"></i>食い違い</span>';}
for(const key of ["kde","rings","bridge","labels"])document.getElementById("t-"+key).addEventListener("change",e=>{opt[key]=e.target.checked?1:0;draw();});
function all(){layout();draw();legend();}
addEventListener("resize",all);matchMedia("(prefers-color-scheme: dark)").addEventListener("change",all);
new MutationObserver(all).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});all();
</script>`;

const out = path.join(ROOT, "out", "stocks", "stock-radar.html");
fs.writeFileSync(out, html);
console.log("wrote", path.relative(ROOT, out), html.length, "bytes;", `${bundle.bridgeCount} bridges, ${bundle.sectors.length} sectors, ${n} stocks`);
