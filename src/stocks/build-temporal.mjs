// Build the temporal drift arrow-map (interactive HTML): each stock's movement in
// whitened factor space over ~6 months, after Procrustes removes the arbitrary
// frame between the two independent MDS snapshots. This is the A6 trend line on
// real time series. Output -> out/stocks/ (public data, still starts private).
//
// Run: node src/stocks/build-temporal.mjs   (after 02-temporal.mjs)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const T = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "stocks", "temporal.json"), "utf8"));
const P = T.points;
const palette = {
  "Industrials": ["#4a6274", "#8ba3b5"], "Consumer Cyclical": ["#c0562f", "#e0895a"],
  "Technology": ["#2f6fb0", "#5aa0e0"], "Communication Services": ["#7d54c9", "#b38ce8"],
  "Healthcare": ["#12876a", "#43c39c"], "Consumer Defensive": ["#78871a", "#bcc74e"],
  "Financial Services": ["#b0357f", "#e06ab0"], "Basic Materials": ["#b26a12", "#e0a44e"],
  "Real Estate": ["#0e8a9c", "#4ec3d4"], "Energy": ["#8a2f2f", "#d06a6a"], "Utilities": ["#556b2f", "#9caf6a"], "Unknown": ["#888", "#aaa"],
};
const sectors = [...new Set(P.map((p) => p.sector))].sort((a, b) => P.filter((x) => x.sector === b).length - P.filter((x) => x.sector === a).length);
let mx = 0; for (const p of P) mx = Math.max(mx, Math.abs(p.x0), Math.abs(p.y0), Math.abs(p.x2), Math.abs(p.y2));
const bundle = { points: P, palette, sectors, maxR: mx, rotationDeg: T.rotationDeg, n: T.n, topN: 8 };

const html = `<style>
:root{--bg:#f5f7f8;--surface:#fff;--ink:#161a1e;--muted:#59636d;--hairline:#e2e7ea;--accent:#2f6fb0;--faint:#c4ccd2;
--mono:ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif;}
@media (prefers-color-scheme:dark){:root{--bg:#13161a;--surface:#1b2126;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#5aa0e0;--faint:#3a434c;}}
:root[data-theme="dark"]{--bg:#13161a;--surface:#1b2126;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#5aa0e0;--faint:#3a434c;}
:root[data-theme="light"]{--bg:#f5f7f8;--surface:#fff;--ink:#161a1e;--muted:#59636d;--hairline:#e2e7ea;--accent:#2f6fb0;--faint:#c4ccd2;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6}
.wrap{max-width:900px;margin:0 auto;padding:32px 20px 60px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}
h1{font-size:26px;margin:0 0 6px;font-weight:600}.sub{color:var(--muted);margin:0 0 18px;font-size:15px}
.controls{display:flex;flex-wrap:wrap;gap:14px;font-family:var(--mono);font-size:13px;margin:0 0 14px;padding:12px 14px;background:var(--surface);border:1px solid var(--hairline);border-radius:10px}
.controls label{display:flex;align-items:center;gap:6px;cursor:pointer;color:var(--muted)}
.stage{position:relative;background:var(--surface);border:1px solid var(--hairline);border-radius:14px;overflow:hidden}
canvas{display:block;width:100%;height:auto}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12.5px;padding:6px 9px;border-radius:7px;opacity:0;transition:opacity .1s;line-height:1.45}
.legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin:16px 0 0;font-family:var(--mono);font-size:12.5px}
.legend span{display:flex;align-items:center;gap:6px;color:var(--muted)}.sw{width:11px;height:11px;border-radius:3px}
.note{font-family:var(--mono);font-size:12px;color:var(--muted);margin:16px 0 0;line-height:1.7}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
</style>
<div class="wrap">
<h2 class="sr-only">Interactive map of ${bundle.n} Japanese stocks' six-month drift in whitened price-factor space after Procrustes alignment.</h2>
<p class="eyebrow">temporal drift · A6 trend line</p>
<h1>ファクター空間の6か月ドリフト</h1>
<p class="sub">株価履歴からモメンタム(1/3/6m)＋3mボラを2時点で算出し、各時点を独立にMDS→Procrustesで枠（回転${bundle.rotationDeg.toFixed(0)}°）を除去。矢印＝各銘柄の"真の"移動（○=6か月前→●=現在）。色＝セクター、太い矢印＝移動上位。</p>
<div class="controls">
<label><input type="checkbox" id="t-arrows" checked> 全銘柄の矢印</label>
<label><input type="checkbox" id="t-labels" checked> 上位移動をラベル</label>
</div>
<div class="stage"><canvas id="cv" role="img" aria-label="factor drift arrow map"></canvas><div class="tip" id="tip"></div></div>
<div class="legend" id="legend"></div>
<p class="note">独立に投影した2枚は枠が任意（ここでは${bundle.rotationDeg.toFixed(0)}°回転）。Procrustesで枠を消して初めて、矢印＝本当の移動になる（A6の実証）。大きく動いた銘柄＝この6か月でモメンタム/ボラのファクター性格が変わった銘柄。KBの経年比較と同一機序を実時系列へ移植。</p>
</div>
<script>
const DATA=${JSON.stringify(bundle)};
const cv=document.getElementById("cv"),tip=document.getElementById("tip");
const opt={arrows:1,labels:1};
function theme(){const r=document.documentElement.getAttribute("data-theme");if(r)return r;return matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}
function col(s){const c=DATA.palette[s]||DATA.palette.Unknown;return c[theme()==="dark"?1:0];}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
let W,H,cx,cy,scale;
function layout(){const cssW=cv.clientWidth,dpr=devicePixelRatio||1;W=cssW;H=Math.round(cssW*0.72);cv.width=W*dpr;cv.height=H*dpr;cv.getContext("2d").setTransform(dpr,0,0,dpr,0,0);scale=(Math.min(W,H)/2-40)/DATA.maxR;cx=W/2;cy=H/2;}
function arrow(ctx,x1,y1,x2,y2,w){ctx.lineWidth=w;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();const a=Math.atan2(y2-y1,x2-x1),h=5+w;ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-h*Math.cos(a-0.4),y2-h*Math.sin(a-0.4));ctx.lineTo(x2-h*Math.cos(a+0.4),y2-h*Math.sin(a+0.4));ctx.closePath();ctx.fill();}
function draw(){const ctx=cv.getContext("2d");ctx.clearRect(0,0,W,H);
 DATA.points.forEach((p,idx)=>{const top=idx<DATA.topN;if(!opt.arrows&&!top)return;const x1=cx+p.x0*scale,y1=cy-p.y0*scale,x2=cx+p.x2*scale,y2=cy-p.y2*scale,cc=col(p.sector);
  ctx.globalAlpha=top?0.95:0.4;ctx.strokeStyle=cc;ctx.fillStyle=cc;arrow(ctx,x1,y1,x2,y2,top?2.2:1);
  ctx.globalAlpha=top?0.9:0.4;ctx.beginPath();ctx.arc(x1,y1,2,0,7);ctx.strokeStyle=cc;ctx.lineWidth=1;ctx.stroke();ctx.beginPath();ctx.arc(x2,y2,top?4:3,0,7);ctx.fillStyle=cc;ctx.fill();});
 ctx.globalAlpha=1;
 if(opt.labels){ctx.fillStyle=css("--ink");ctx.font="11px "+css("--sans");ctx.textAlign="left";DATA.points.slice(0,DATA.topN).forEach(p=>{const x2=cx+p.x2*scale,y2=cy-p.y2*scale;ctx.fillText(p.name.split(" ")[0],x2+6,y2+3);});}
}
function pick(mx,my){let best=-1,bd=1e9;DATA.points.forEach((p,i)=>{const x2=cx+p.x2*scale,y2=cy-p.y2*scale,dd=(x2-mx)**2+(y2-my)**2;if(dd<bd){bd=dd;best=i;}});return bd<220?best:-1;}
cv.addEventListener("mousemove",e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,i=pick(mx,my);if(i<0){tip.style.opacity=0;return;}const p=DATA.points[i];tip.innerHTML='<b>'+p.name+'</b><br>'+p.sector+' · 移動量 '+p.mag.toFixed(3);tip.style.opacity=1;tip.style.left=Math.min(mx+12,W-200)+"px";tip.style.top=(my+12)+"px";});
cv.addEventListener("mouseleave",()=>tip.style.opacity=0);
function legend(){document.getElementById("legend").innerHTML=DATA.sectors.map(s=>'<span><i class="sw" style="background:'+col(s)+'"></i>'+s+'</span>').join("");}
for(const key of ["arrows","labels"])document.getElementById("t-"+key).addEventListener("change",e=>{opt[key]=e.target.checked?1:0;draw();});
function all(){layout();draw();legend();}
addEventListener("resize",all);matchMedia("(prefers-color-scheme: dark)").addEventListener("change",all);
new MutationObserver(all).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});all();
</script>`;

const out = path.join(ROOT, "out", "stocks", "temporal-drift.html");
fs.writeFileSync(out, html);
console.log("wrote", path.relative(ROOT, out), html.length, "bytes;", `${P.length} stocks, rotation ${T.rotationDeg.toFixed(1)}deg removed`);
