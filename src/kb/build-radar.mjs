// Build the integrated Knowledge Base Radar (single interactive HTML) from all
// Part A layers, plus: (A) commercial-landscape-style cluster view
// (plot = cluster, size = count), (B) deterministic classical-MDS init + retention metrics (stress-1,
// distance corr, trustworthiness, continuity), (C) Sammon vs MDS toggle.
// Output HTML has private titles -> out/kb/ (gitignored). Publish as an Artifact.
//
// Run: node src/kb/build-radar.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pca2 } from "../lib/projection/pca.mjs";
import { smacof } from "../lib/projection/mds.mjs";
import { sammon } from "../lib/projection/sammon.mjs";
import { kmeans } from "../lib/kmeans.mjs";
import { adaptiveGridDensity, contourSegments, scottBandwidth } from "../lib/kde.mjs";
import { projectionMetrics } from "../lib/metrics.mjs";
import { jacobiEigsym } from "../lib/eig.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const emb = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "kb", "permanent-embeddings.json"), "utf8"));
const V = emb.vectors.map((v) => Float64Array.from(v));
const n = V.length, dim = emb.dim;
const c = new Float64Array(dim);
for (const v of V) for (let d = 0; d < dim; d++) c[d] += v[d] / n;
const Xc = V.map((v) => Float64Array.from(v, (x, d) => x - c[d]));

const D = Array.from({ length: n }, () => new Float64Array(n));
for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
  let s = 0; for (let k = 0; k < dim; k++) { const dd = Xc[i][k] - Xc[j][k]; s += dd * dd; }
  D[i][j] = D[j][i] = Math.sqrt(s);
}

// (B) classical-MDS init = PCA scores (deterministic). MDS from it + Sammon from it.
const init = pca2(Xc);
const Y_mds = smacof(D, 1, 400, init);
const Y_sam = sammon(D, init, { iters: 250, mf: 0.3 });

// high-D clusters + human tags + bridges (shared across projections)
const human = emb.tags.map((ts) => ts.find((t) => t !== "permanent") ?? "untagged");
const k = new Set(human).size;
const { labels: cl } = kmeans(Xc.map((v) => Array.from(v)), k, { seed: 3 });
const dom = {};
for (let c2 = 0; c2 < k; c2++) {
  const mem = [...cl.keys()].filter((i) => cl[i] === c2);
  const cnt = {}; for (const i of mem) cnt[human[i]] = (cnt[human[i]] ?? 0) + 1;
  dom[c2] = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0]?.[0];
}

// (micro) per-cluster local PCA — axes ARE meaningful within a homogeneous topic.
// Reflect ONLY rotation-invariant summaries up: anisotropy λ1/λ2 and effective
// dimensionality (participation ratio). Also bake micro-KDE contours for the drill.
const clusterMeta = {};
const clusters = [];
for (let c2 = 0; c2 < k; c2++) {
  const mem = [...cl.keys()].filter((i) => cl[i] === c2);
  if (!mem.length) continue;
  const sub = mem.map((i) => Xc[i]); const mm = sub.length;
  const cc = new Float64Array(dim);
  for (const v of sub) for (let d = 0; d < dim; d++) cc[d] += v[d] / mm;
  const subc = sub.map((v) => Float64Array.from(v, (x, d) => x - cc[d]));
  let loc, aniso = 1, effDim = 1;
  if (mm >= 3) {
    const G = Array.from({ length: mm }, () => new Float64Array(mm));
    for (let i = 0; i < mm; i++) for (let j = i; j < mm; j++) { let s = 0; for (let d = 0; d < dim; d++) s += subc[i][d] * subc[j][d]; G[i][j] = G[j][i] = s; }
    const { values, vectors } = jacobiEigsym(G);
    const pos = values.map((v) => Math.max(v, 0));
    const sum = pos.reduce((a, b) => a + b, 0) || 1, sum2 = pos.reduce((a, b) => a + b * b, 0) || 1;
    effDim = (sum * sum) / sum2; aniso = pos[1] > 1e-9 ? pos[0] / pos[1] : 999;
    const s1 = Math.sqrt(pos[0]), s2 = Math.sqrt(pos[1]);
    loc = mem.map((_, r) => [s1 * vectors[0][r], s2 * vectors[1][r]]);
  } else loc = sub.map(() => [0, 0]);
  // micro-KDE contours in the PCA frame (recursive radar, interpretable axes)
  let mContours = [];
  if (mm >= 6) {
    const bw = scottBandwidth(loc); const kde = adaptiveGridDensity(loc, bw, 50, 38);
    mContours = [0.1, 0.22, 0.38, 0.55, 0.75].map((f, i, a) => ({ opacity: +(0.15 + 0.3 * (i / (a.length - 1))).toFixed(3), segs: contourSegments(kde.grid, kde.bbox, f * kde.max) }));
  }
  clusterMeta[c2] = { aniso, effDim };
  clusters.push({ id: c2, tag: dom[c2], aniso, effDim, contours: mContours, members: mem.map((i, r) => ({ title: emb.labels[i], path: emb.labels[i], lx: loc[r][0], ly: loc[r][1], ownTag: human[i], bridge: human[i] !== dom[c2] })) });
}

// orient, center; ref (optional) = align to a reference layout via Procrustes
// (rotation+reflection) so switching projections does not rotate the map.
function buildProjection(Yin, ref) {
  let Y = Yin.map((p) => [p[0], p[1]]);
  let mx = 0, my = 0; for (const p of Y) { mx += p[0] / n; my += p[1] / n; }
  Y = Y.map((p) => [p[0] - mx, p[1] - my]);
  if (ref) {
    // 2D orthogonal Procrustes: optimal rotation per reflection state, keep the better.
    const align = (flip) => {
      let sc = 0, sd = 0;
      for (let i = 0; i < n; i++) { const yx = Y[i][0], yy = flip ? -Y[i][1] : Y[i][1]; sc += yx * ref[i][1] - yy * ref[i][0]; sd += yx * ref[i][0] + yy * ref[i][1]; }
      const a = Math.atan2(sc, sd), ca = Math.cos(a), sa = Math.sin(a);
      const out = Y.map((p) => { const x = p[0], y = flip ? -p[1] : p[1]; return [ca * x - sa * y, sa * x + ca * y]; });
      let e = 0; for (let i = 0; i < n; i++) e += (out[i][0] - ref[i][0]) ** 2 + (out[i][1] - ref[i][1]) ** 2;
      return { out, e };
    };
    const A = align(false), B = align(true);
    Y = (A.e <= B.e ? A : B).out;
  } else {
    let Sxx = 0, Syy = 0, Sxy = 0;
    for (const p of Y) { Sxx += p[0] * p[0]; Syy += p[1] * p[1]; Sxy += p[0] * p[1]; }
    const ang = 0.5 * Math.atan2(2 * Sxy, Sxx - Syy), ca = Math.cos(ang), sa = Math.sin(ang);
    Y = Y.map((p) => [ca * p[0] + sa * p[1], -sa * p[0] + ca * p[1]]);
  }

  const rDist = Y.map((p) => Math.hypot(p[0], p[1]));
  const sortedR = [...rDist].sort((a, b) => a - b);
  const ringR = [0.3, 0.6, 0.9].map((q) => sortedR[Math.floor(q * (n - 1))]);
  const maxR = sortedR[n - 1];

  const bw = scottBandwidth(Y);
  const kde = adaptiveGridDensity(Y, bw, 80, 60);
  const levelFracs = [0.08, 0.16, 0.27, 0.4, 0.55, 0.72, 0.88];
  const contours = levelFracs.map((f, i) => ({
    opacity: +(0.16 + 0.34 * (i / (levelFracs.length - 1))).toFixed(3),
    segs: contourSegments(kde.grid, kde.bbox, f * kde.max),
  }));

  // (A) cluster glyphs as ELLIPSES: aspect = high-D internal anisotropy (rotation-
  // invariant), orientation = 2D footprint (placement only), fill opacity = coherence
  // (effective dim), dashed edge = low macro fidelity.
  const glyphs = [];
  for (let c2 = 0; c2 < k; c2++) {
    const mem = [...cl.keys()].filter((i) => cl[i] === c2); if (!mem.length) continue;
    const mm = mem.length; let gx = 0, gy = 0; for (const i of mem) { gx += Y[i][0] / mm; gy += Y[i][1] / mm; }
    let sxx = 0, syy = 0, sxy = 0; for (const i of mem) { const dx = Y[i][0] - gx, dy = Y[i][1] - gy; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const aniso = clusterMeta[c2]?.aniso ?? 1, aspect = Math.min(2.6, Math.max(1, Math.sqrt(aniso))), r0 = 6 + Math.sqrt(mm) * 3.4;
    const hd = [], td = [];
    for (let a = 0; a < mm; a++) for (let b = a + 1; b < mm; b++) { hd.push(D[mem[a]][mem[b]]); td.push(Math.hypot(Y[mem[a]][0] - Y[mem[b]][0], Y[mem[a]][1] - Y[mem[b]][1])); }
    let fidelity = 1;
    if (hd.length > 1) { let ma = 0, mb = 0; for (let t = 0; t < hd.length; t++) { ma += hd[t] / hd.length; mb += td[t] / td.length; } let s = 0, sa = 0, sb = 0; for (let t = 0; t < hd.length; t++) { const u = hd[t] - ma, v = td[t] - mb; s += u * v; sa += u * u; sb += v * v; } fidelity = sa && sb ? s / Math.sqrt(sa * sb) : 1; }
    glyphs.push({ x: gx, y: gy, count: mm, tag: dom[c2], cid: c2, angle, ax: r0 * Math.sqrt(aspect), ay: r0 / Math.sqrt(aspect), fidelity, effDim: clusterMeta[c2]?.effDim ?? 1, aniso });
  }

  const m = projectionMetrics(D, Y, 10);
  return { xy: Y, ringR, maxR, contours, glyphs, metrics: m };
}
const projMDS = buildProjection(Y_mds);
const proj = { MDS: projMDS, Sammon: buildProjection(Y_sam, projMDS.xy) }; // Sammon aligned to MDS (no rotation on toggle)
console.log("MDS   metrics:", JSON.stringify(proj.MDS.metrics));
console.log("Sammon metrics:", JSON.stringify(proj.Sammon.metrics));

const palette = {
  "ai-human": ["#2f6fb0", "#5aa0e0"], "self-management": ["#12876a", "#43c39c"],
  "causality": ["#7d54c9", "#b38ce8"], "trading": ["#c0562f", "#e0895a"],
  "writing": ["#b26a12", "#e0a44e"], "justice": ["#b0357f", "#e06ab0"],
  "engineering": ["#4a6274", "#8ba3b5"], "value": ["#78871a", "#bcc74e"],
  "machine-learning": ["#0e8a9c", "#4ec3d4"], "untagged": ["#888", "#aaa"],
};
const points = emb.labels.map((title, i) => ({ title, tag: human[i], bridge: human[i] !== dom[cl[i]], domTag: dom[cl[i]], cid: cl[i] }));
const tagsUsed = [...new Set(human)].sort((a, b) => human.filter((h) => h === b).length - human.filter((h) => h === a).length);
const bundle = { points, proj, clusters, palette, tags: tagsUsed, bridgeCount: points.filter((p) => p.bridge).length, n };

const html = `<style>
:root{--bg:#f5f7f8;--surface:#fff;--tile:#eef2f3;--ink:#161a1e;--muted:#59636d;--hairline:#e2e7ea;--accent:#12876a;--ring:#c4ccd2;--terrain:#b26a12;
--mono:ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif;}
@media (prefers-color-scheme:dark){:root{--bg:#13161a;--surface:#1b2126;--tile:#20272d;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#43c39c;--ring:#3a434c;--terrain:#e0a44e;}}
:root[data-theme="dark"]{--bg:#13161a;--surface:#1b2126;--tile:#20272d;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#43c39c;--ring:#3a434c;--terrain:#e0a44e;}
:root[data-theme="light"]{--bg:#f5f7f8;--surface:#fff;--tile:#eef2f3;--ink:#161a1e;--muted:#59636d;--hairline:#e2e7ea;--accent:#12876a;--ring:#c4ccd2;--terrain:#b26a12;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6}
.wrap{max-width:900px;margin:0 auto;padding:32px 20px 60px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}
h1{font-size:26px;margin:0 0 6px;font-weight:600}.sub{color:var(--muted);margin:0 0 16px;font-size:15px}
.controls{display:flex;flex-wrap:wrap;gap:14px;font-family:var(--mono);font-size:13px;margin:0 0 12px;padding:12px 14px;background:var(--surface);border:1px solid var(--hairline);border-radius:10px;align-items:center}
.controls label{display:flex;align-items:center;gap:6px;cursor:pointer;color:var(--muted)}
.controls select{font-family:var(--mono);font-size:13px;padding:2px 6px;border-radius:6px;border:1px solid var(--hairline);background:var(--surface);color:var(--ink)}
.metrics{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 12px}
.metric{background:var(--tile);border-radius:9px;padding:8px 12px;font-family:var(--mono)}
.metric .k{font-size:11px;color:var(--muted)}.metric .v{font-size:17px;font-weight:600}
.stage{position:relative;background:var(--surface);border:1px solid var(--hairline);border-radius:14px;overflow:hidden}
canvas{display:block;width:100%;height:auto}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12.5px;padding:6px 9px;border-radius:7px;max-width:280px;opacity:0;transition:opacity .1s;line-height:1.45}
.legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin:16px 0 0;font-family:var(--mono);font-size:12.5px}
.legend span{display:flex;align-items:center;gap:6px;color:var(--muted)}.sw{width:11px;height:11px;border-radius:3px}
.note{font-family:var(--mono);font-size:12px;color:var(--muted);margin:16px 0 0;line-height:1.7}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
</style>
<div class="wrap">
<h2 class="sr-only">Interactive knowledge base radar: 86 notes on an MDS or Sammon projection with density contours, percentile rings, cluster colors, cluster aggregation, and retention metrics.</h2>
<p class="eyebrow">knowledge base radar · MDS / Sammon</p>
<h1>Knowledge Base Radar</h1>
<p class="sub">86 Permanentノートを距離忠実な投影に配置。同心円＝際立ち度の順位、等高線＝適応KDE、色＝テーマ、黒枠＝架橋（人手タグ≠高次元クラスタ）。<b>マクロ地図の軸に意味はない</b>（MDS）。投影(MDS/Sammon)・表示(個体/クラスタ)を切替可、Sammonは<b>MDSに整列済みで切替時に回転しない</b>。クラスタ表示は<b>楕円＝内部構造</b>（縦横比＝異方性・濃さ＝コヒーレンス・点線＝低忠実度）。クリックで<b>ミクロ地図（PCA軸に意味＋内部密度等高線）</b>。</p>
<div class="controls">
<label>投影 <select id="s-proj"><option value="MDS">MDS（大域距離）</option><option value="Sammon">Sammon（局所優先）</option></select></label>
<label>表示 <select id="s-view"><option value="ind">個体（1点=1ノート）</option><option value="cluster">クラスタ（1点=1クラスタ, サイズ=件数）</option></select></label>
<label><input type="checkbox" id="t-kde" checked> 等高線</label>
<label><input type="checkbox" id="t-rings" checked> 同心円</label>
<label><input type="checkbox" id="t-bridge" checked> 架橋</label>
</div>
<div class="metrics" id="metrics"></div>
<div class="stage"><canvas id="radar" role="img" aria-label="knowledge base radar"></canvas><div class="tip" id="tip"></div></div>
<div class="legend" id="legend"></div>
<p class="note" style="margin-top:10px">クラスタ（点・glyph）を<b>クリック</b>すると、そのクラスタ内だけの<b>ミクロ地図</b>が下に開く。</p>
<div id="drill"></div>
<p class="note">投影は古典MDS(PCA)で初期化＝決定的・再現可能。保持指標: stress-1(低いほど良)・距離相関(大域)・trustworthiness(2Dの隣は本物か)・continuity(本物の隣を保てたか)。<b>このデータではMDSが全指標で優位</b>——Sammonは自分の目的(1/δ重み＝極小距離)では勝つが、cone構造では「極小距離」と「k近傍」が別物でk=10のtrust/continuityは負ける。Sammonが効くのはタイトな局所クラスタを持つデータ。<b>だから測って選ぶ(＝指標の意義)</b>。クラスタ表示は商用ランドスケープ型(プロット=クラスタ・サイズ=件数)。架橋${bundle.bridgeCount}件。</p>
</div>
<script>
const DATA=${JSON.stringify(bundle)};
const cv=document.getElementById("radar"),tip=document.getElementById("tip");
let curProj="MDS",view="ind";
const opt={kde:1,rings:1,bridge:1};
function theme(){const r=document.documentElement.getAttribute("data-theme");if(r)return r;return matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}
function col(tag){const c=DATA.palette[tag]||DATA.palette.untagged;return c[theme()==="dark"?1:0];}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
let W,H,cx,cy,scale;
function P(){return DATA.proj[curProj];}
function layout(){const cssW=cv.clientWidth,dpr=devicePixelRatio||1;W=cssW;H=Math.round(cssW*0.72);cv.width=W*dpr;cv.height=H*dpr;cv.getContext("2d").setTransform(dpr,0,0,dpr,0,0);scale=(Math.min(W,H)/2-46)/P().maxR;cx=W/2;cy=H/2;}
function draw(){const ctx=cv.getContext("2d");ctx.clearRect(0,0,W,H);const p=P();
 if(opt.kde){const t=css("--terrain");ctx.lineWidth=1;ctx.lineJoin="round";p.contours.forEach(c=>{ctx.globalAlpha=c.opacity;ctx.strokeStyle=t;ctx.beginPath();for(const s of c.segs){ctx.moveTo(cx+s[0]*scale,cy-s[1]*scale);ctx.lineTo(cx+s[2]*scale,cy-s[3]*scale);}ctx.stroke();});ctx.globalAlpha=1;}
 if(opt.rings){ctx.strokeStyle=css("--ring");ctx.fillStyle=css("--muted");ctx.font="11px "+css("--mono");ctx.textAlign="center";const labs=["30%","60%","90%"];p.ringR.forEach((r,i)=>{ctx.setLineDash(i<2?[3,3]:[]);ctx.beginPath();ctx.arc(cx,cy,r*scale,0,7);ctx.stroke();ctx.fillText(labs[i],cx,cy-r*scale-4);});ctx.setLineDash([]);}
 if(view==="ind"){DATA.points.forEach((pt,i)=>{const xy=p.xy[i],sx=cx+xy[0]*scale,sy=cy-xy[1]*scale;ctx.beginPath();ctx.arc(sx,sy,4.5,0,7);ctx.fillStyle=col(pt.tag);ctx.fill();if(opt.bridge&&pt.bridge){ctx.lineWidth=1.6;ctx.strokeStyle=theme()==="dark"?"#fff":"#111";ctx.stroke();}});}
 else{const brd=theme()==="dark"?"#fff":"#111";p.glyphs.forEach(g=>{const sx=cx+g.x*scale,sy=cy-g.y*scale,alpha=0.22+0.38*Math.max(0,Math.min(1,1-(g.effDim-1)/8));ctx.save();ctx.translate(sx,sy);ctx.rotate(-g.angle);ctx.beginPath();ctx.ellipse(0,0,g.ax,g.ay,0,0,7);ctx.fillStyle=col(g.tag);ctx.globalAlpha=alpha;ctx.fill();ctx.globalAlpha=1;ctx.lineWidth=1.4;ctx.strokeStyle=col(g.tag);if(g.fidelity<0.6)ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);ctx.restore();ctx.fillStyle=brd;ctx.font="11px "+css("--mono");ctx.textAlign="center";ctx.fillText(g.count,sx,sy+4);});}
}
function metrics(){const m=P().metrics;const cell=(k,v)=>'<div class="metric"><div class="k">'+k+'</div><div class="v">'+v+'</div></div>';
 document.getElementById("metrics").innerHTML=cell("stress-1 ↓",m.stress1.toFixed(3))+cell("距離相関 ↑",m.corr.toFixed(3))+cell("trustworthiness ↑",m.trust.toFixed(3))+cell("continuity ↑",m.cont.toFixed(3));}
function pick(mx,my){const p=P();let best=-1,bd=1e9;if(view==="ind"){DATA.points.forEach((pt,i)=>{const xy=p.xy[i],sx=cx+xy[0]*scale,sy=cy-xy[1]*scale,d=(sx-mx)**2+(sy-my)**2;if(d<bd){bd=d;best=i;}});return bd<200?["p",best]:null;}
 p.glyphs.forEach((g,i)=>{const sx=cx+g.x*scale,sy=cy-g.y*scale,d=(sx-mx)**2+(sy-my)**2;if(d<bd){bd=d;best=i;}});return bd<900?["g",best]:null;}
cv.addEventListener("mousemove",e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,h=pick(mx,my);if(!h){tip.style.opacity=0;return;}
 if(h[0]==="p"){const pt=DATA.points[h[1]];tip.innerHTML=pt.title+(pt.bridge?'<br><span style="opacity:.7">架橋: '+pt.tag+' → クラスタは'+pt.domTag+'</span>':'');}
 else{const g=P().glyphs[h[1]];const shape=g.aniso>2.2?'スペクトル的(1D)':g.aniso<1.4?'均質(等方)':'中間';tip.innerHTML='<b>'+g.tag+'</b> クラスタ · '+g.count+'ノート<br><span style="opacity:.75">内部='+shape+' · 有効次元'+g.effDim.toFixed(1)+' · 忠実度'+g.fidelity.toFixed(2)+' · クリックで内訳</span>';}
 tip.style.opacity=1;tip.style.left=Math.min(mx+12,W-260)+"px";tip.style.top=(my+12)+"px";});
cv.addEventListener("mouseleave",()=>tip.style.opacity=0);
let drillCid=null;
cv.addEventListener("click",e=>{const r=cv.getBoundingClientRect(),h=pick(e.clientX-r.left,e.clientY-r.top);if(!h)return;drillCid=h[0]==="g"?P().glyphs[h[1]].cid:DATA.points[h[1]].cid;renderDrill();});
function renderDrill(){const cl=DATA.clusters.find(c=>c.id===drillCid);const el=document.getElementById("drill");if(!cl){el.innerHTML="";return;}
 const m=cl.members;let mx=0;for(const p of m)mx=Math.max(mx,Math.abs(p.lx),Math.abs(p.ly));mx=mx||1;const cx=340,cy=175,sc=120/mx;
 const mut=css("--muted"),grid=css("--ring"),brd=theme()==="dark"?"#fff":"#111";
 const byX=[...m].sort((a,b)=>a.lx-b.lx),byY=[...m].sort((a,b)=>a.ly-b.ly);
 const esc=t=>t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/"/g,"&quot;");
 const short=t=>esc(t.length>20?t.slice(0,19)+"…":t);
 let s='<svg id="dsvg" width="100%" viewBox="0 0 680 350" role="img" aria-label="cluster micro map (PCA)">';
 if(opt.kde&&cl.contours){const terr=css("--terrain");for(const c of cl.contours){let d='';for(const g of c.segs){d+='M'+(cx+g[0]*sc).toFixed(1)+' '+(cy-g[1]*sc).toFixed(1)+'L'+(cx+g[2]*sc).toFixed(1)+' '+(cy-g[3]*sc).toFixed(1);}if(d)s+='<path d="'+d+'" fill="none" stroke="'+terr+'" stroke-width="1" opacity="'+c.opacity+'"/>';}}
 s+='<line x1="40" y1="'+cy+'" x2="640" y2="'+cy+'" stroke="'+grid+'"/><line x1="'+cx+'" y1="34" x2="'+cx+'" y2="316" stroke="'+grid+'"/>';
 s+='<text x="638" y="'+(cy-7)+'" text-anchor="end" font-size="10.5" fill="'+mut+'">PC1+ → '+short(byX[byX.length-1].title)+'</text>';
 s+='<text x="42" y="'+(cy-7)+'" text-anchor="start" font-size="10.5" fill="'+mut+'">← PC1− '+short(byX[0].title)+'</text>';
 s+='<text x="'+(cx+7)+'" y="44" font-size="10.5" fill="'+mut+'">PC2+ ↑ '+short(byY[byY.length-1].title)+'</text>';
 s+='<text x="'+(cx+7)+'" y="312" font-size="10.5" fill="'+mut+'">PC2− ↓ '+short(byY[0].title)+'</text>';
 for(const p of m){const x=cx+p.lx*sc,y=cy-p.ly*sc;s+='<circle class="dpt" data-t="'+esc(p.title)+(p.bridge?"（架橋:"+p.ownTag+"）":"")+'" cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="5" fill="'+col(p.ownTag)+'"'+(p.bridge?' stroke="'+brd+'" stroke-width="1.4"':'')+'></circle>';}
 s+='</svg>';
 const shape=cl.aniso>2.2?'スペクトル的':cl.aniso<1.4?'均質':'中間';
 const gg=P().glyphs.find(x=>x.cid===cl.id);
 el.innerHTML='<div style="position:relative;background:var(--surface);border:1px solid var(--hairline);border-radius:14px;padding:14px 16px;margin-top:6px"><div style="font-family:var(--mono);font-size:13px;margin-bottom:4px"><b>クラスタ「'+cl.tag+'」のミクロ地図</b> <span style="color:var(--muted)">n='+m.length+' ／ 内部='+shape+'（異方性'+cl.aniso.toFixed(1)+'・有効次元'+cl.effDim.toFixed(1)+'・忠実度'+(gg?gg.fidelity.toFixed(2):'-')+'）／ PCA軸=主変動・等高線=内部密度</span> <span id="drill-x" style="float:right;cursor:pointer;color:var(--muted)">✕ 閉じる</span></div>'+s+'<div id="dtip" class="tip"></div><div style="font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:4px">マクロ(上)=MDSで軸は無意味。ミクロ=PCAで軸＝この話題内の主変動＝解釈可能（両端のノートが軸の意味）。等高線=クラスタ内の密度。点にホバーでノート名。</div></div>';
 document.getElementById("drill-x").addEventListener("click",()=>{drillCid=null;el.innerHTML="";});
 const svg=document.getElementById("dsvg"),dtip=document.getElementById("dtip"),pts=[...svg.querySelectorAll(".dpt")];
 svg.addEventListener("mousemove",ev=>{const rc=svg.getBoundingClientRect(),vx=(ev.clientX-rc.left)/rc.width*680,vy=(ev.clientY-rc.top)/rc.height*350;let best=null,bd=1e9;for(const c of pts){const dx=+c.getAttribute("cx")-vx,dy=+c.getAttribute("cy")-vy,d=dx*dx+dy*dy;if(d<bd){bd=d;best=c;}}if(best&&bd<400){dtip.innerHTML=best.getAttribute("data-t");dtip.style.opacity=1;dtip.style.left=Math.min(ev.clientX-rc.left+12,rc.width-210)+"px";dtip.style.top=(ev.clientY-rc.top+12)+"px";}else dtip.style.opacity=0;});
 svg.addEventListener("mouseleave",()=>dtip.style.opacity=0);}
function legend(){document.getElementById("legend").innerHTML=DATA.tags.map(t=>'<span><i class="sw" style="background:'+col(t)+'"></i>'+t+'</span>').join("")+'<span><i class="sw" style="border:1.6px solid '+(theme()==="dark"?"#fff":"#111")+';background:transparent"></i>架橋</span>';}
document.getElementById("s-proj").addEventListener("change",e=>{curProj=e.target.value;all();});
document.getElementById("s-view").addEventListener("change",e=>{view=e.target.value;draw();});
for(const key of ["kde","rings","bridge"])document.getElementById("t-"+key).addEventListener("change",e=>{opt[key]=e.target.checked?1:0;draw();});
function all(){layout();draw();legend();metrics();if(drillCid!=null)renderDrill();}
addEventListener("resize",all);matchMedia("(prefers-color-scheme: dark)").addEventListener("change",all);
new MutationObserver(all).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});all();
</script>`;

const out = path.join(ROOT, "out", "kb", "kb-radar.html");
fs.writeFileSync(out, html);
console.log("wrote", path.relative(ROOT, out), html.length, "bytes;", `${bundle.bridgeCount} bridges, ${bundle.tags.length} tags`);
