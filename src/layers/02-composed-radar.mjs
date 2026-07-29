// Stage D2 — the composed radar: position from identity, overlay from state.
//
// This is where the three doubts that opened Part C converge.
//
//   position   text identity (E1 LSA space), because PER/PBR describe how a
//              company is priced, not what it is (Part E)
//   overlay    momentum and volatility, kept OUT of the distance matrix and
//              applied only at draw time (Part D: mixing them in moves the map
//              0.932, and a routine six-month price update moves it 0.819)
//   objective  raw stress, and the choice is defensible precisely because C1
//              showed the readout is near-invariant to it (radius percentile
//              correlates 0.94-1.00 across the valid objectives)
//
// Two conventions carried over rather than reinvented:
//   A4  density contours are computed on the MDS coordinates, never on the
//       percentile radar, where 1/r manufactures a hot centre.
//   A3  the rotation-invariant reading is the high-dimensional distinctiveness
//       percentile; it is in the tooltip, since angle carries no meaning.
//
// The date slider is D-H2 made visible over 29 dates: every colour and every
// radius changes, and no point moves, because state never entered the geometry.
// Peer-relative readings carry the cohesion guard from src/lib/peer.mjs.
//
// Run: node src/layers/02-composed-radar.mjs   (no network)

import fs from "node:fs";
import path from "node:path";
import { lsa, TOKENIZERS } from "../lib/text/lsa.mjs";
import { classicalMDS } from "../lib/projection/classical-mds.mjs";
import { gdMDS } from "../lib/projection/mds-gd.mjs";
import { rawStress } from "../lib/projection/objectives.mjs";
import { kmeans } from "../lib/kmeans.mjs";
import { adaptiveGridDensity, contourSegments, scottBandwidth } from "../lib/kde.mjs";
import { peerGroups, peerCohesion, peerReading, ALPHA, FLAG } from "../lib/peer.mjs";
import { euclid, ROOT } from "../objectives/domains.mjs";

const TOKENIZER = "word"; // E-H2: char45 agrees at ARI 0.228; word is readable in the English pilot
const K_PEERS = 10;
const K_CLUSTERS = 11;
const r4 = (x) => (x === null || x === undefined ? null : Math.round(x * 1e4) / 1e4);

const C = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "text", "corpus.json"), "utf8"));
const SS = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "layers", "state-series.json"), "utf8"));

const posInCorpus = new Map(C.docs.map((d, p) => [d.base, p]));
const joint = SS.baseIndex
  .map((b, p) => ({ state: p, text: posInCorpus.get(b) }))
  .filter((r) => r.text !== undefined);
const n = joint.length;

// ---------------------------------------------------------------- identity layer

const full = lsa(C.docs.map((d) => d.text), TOKENIZERS[TOKENIZER]);
const X = joint.map((r) => full.X[r.text]);
const D = euclid(X);
const Y = gdMDS(rawStress, D, classicalMDS(D).Y).Y;

let cx = 0, cy = 0;
for (const p of Y) { cx += p[0] / n; cy += p[1] / n; }
const P = Y.map((p) => [p[0] - cx, p[1] - cy]);

// A3: distinctiveness measured in the HIGH-dimensional identity space
const centroid = X[0].map((_, k) => X.reduce((a, x) => a + x[k], 0) / n);
const distinct = X.map((x) => Math.hypot(...x.map((v, k) => v - centroid[k])));
const dOrder = [...distinct.keys()].sort((a, b) => distinct[a] - distinct[b]);
const distinctPct = new Array(n);
dOrder.forEach((i, r) => (distinctPct[i] = (100 * r) / (n - 1)));

const clusters = kmeans(X, K_CLUSTERS, { seed: 1 }).labels;
const peers = peerGroups(D, K_PEERS);
const cohesion = peerCohesion(D, peers);

// ------------------------------------------------------------------ state layer

const fi = { mom: SS.featureNames.indexOf("mom3m"), vol: SS.featureNames.indexOf("vol3m") };
const sectors = joint.map((r) => SS.sectors[r.state]);
const bySector = {};
sectors.forEach((s, i) => (bySector[s] ??= []).push(i));

const pctRank = (v) => {
  const o = [...v.keys()].sort((a, b) => v[a] - v[b]);
  const r = new Array(v.length);
  o.forEach((i, k) => (r[i] = k / (v.length - 1)));
  return r;
};

function overlayAt(snapshot) {
  const mom = joint.map((r) => snapshot.features[r.state][fi.mom]);
  const vol = joint.map((r) => snapshot.features[r.state][fi.vol]);
  const sectorZ = mom.map((_, i) => {
    const g = bySector[sectors[i]].filter((j) => j !== i);
    if (g.length < 4) return null;
    const mu = g.reduce((a, j) => a + mom[j], 0) / g.length;
    const sd = Math.sqrt(g.reduce((a, j) => a + (mom[j] - mu) ** 2, 0) / g.length) || 1;
    return (mom[i] - mu) / sd;
  });
  return {
    date: snapshot.date,
    mom, vol,
    volPct: pctRank(vol),
    momPeer: peerReading(mom, peers, cohesion),
    volPeer: peerReading(vol, peers, cohesion),
    sectorZ,
  };
}

const series = SS.snapshots.map(overlayAt);

// symmetric colour clamp shared by every date, so colours are comparable in time
const allMom = series.flatMap((s) => s.mom).map(Math.abs).sort((a, b) => a - b);
const momClamp = allMom[Math.floor(0.9 * (allMom.length - 1))];

// --------------------------------------------------------------------- terrain

const bw = scottBandwidth(P);
const kde = adaptiveGridDensity(P, bw, 80, 60);
const contours = [0.08, 0.16, 0.27, 0.4, 0.55, 0.72, 0.88].map((f, i) => ({
  opacity: 0.16 + 0.08 * i,
  segs: contourSegments(kde.grid, kde.bbox, f * kde.max).map((s) => s.map(r4)),
}));

const rDist = P.map((p) => Math.hypot(p[0], p[1]));
const sortedR = [...rDist].sort((a, b) => a - b);
const ringR = [0.3, 0.6, 0.9].map((q) => sortedR[Math.floor(q * (n - 1))]);
const maxR = sortedR[n - 1];

// ---------------------------------------------------------------------- bundle

const points = joint.map((r, i) => ({
  t: SS.tickers[r.state],
  nm: SS.names[r.state],
  sec: sectors[i],
  cl: clusters[i],
  x: r4(P[i][0]), y: r4(P[i][1]),
  dp: Math.round(distinctPct[i]),
  coh: Math.round(cohesion.toPeersPct[i]),
  peerNames: peers[i].slice(0, 4).map((j) => String(SS.names[joint[j].state]).split(/[ ,]/)[0]),
}));

// overlay stored column-wise per date: [mom, volPct, peerZ, shrunkZ, sectorZ, flag]
const frames = series.map((s) => ({
  d: s.date,
  m: s.mom.map(r4),
  vp: s.volPct.map(r4),
  vv: s.vol.map(r4),
  pz: s.momPeer.map((x) => r4(x.z)),
  ps: s.momPeer.map((x) => r4(x.zShrunk)),
  sr: s.momPeer.map((x) => r4(x.sdRatio)),
  bad: s.momPeer.map((x) => (x.unreliable ? 1 : 0)),
  vz: s.volPeer.map((x) => r4(x.z)),
  sz: s.sectorZ.map(r4),
}));

const bundle = { points, frames, contours, ringR: ringR.map(r4), maxR: r4(maxR), momClamp: r4(momClamp), n, rank: full.rank, flag: FLAG, alpha: ALPHA };

// ------------------------------------------------------------------------ html

const html = `<style>
:root{--bg:#fbfbfa;--surface:#fff;--ink:#1d2226;--muted:#6b7680;--hairline:#e3e7ea;--accent:#2f6fb0;--ring:#d7dde2;--terrain:#9fb2c0;--pos:#1f7a5a;--neg:#b4453a;}
@media (prefers-color-scheme:dark){:root{--bg:#13161a;--surface:#1b2126;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#5aa0e0;--ring:#3a434c;--terrain:#8ba3b5;--pos:#4cc79b;--neg:#e0736a;}}
:root[data-theme=light]{--bg:#fbfbfa;--surface:#fff;--ink:#1d2226;--muted:#6b7680;--hairline:#e3e7ea;--accent:#2f6fb0;--ring:#d7dde2;--terrain:#9fb2c0;--pos:#1f7a5a;--neg:#b4453a;}
:root[data-theme=dark]{--bg:#13161a;--surface:#1b2126;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#5aa0e0;--ring:#3a434c;--terrain:#8ba3b5;--pos:#4cc79b;--neg:#e0736a;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 system-ui,-apple-system,"Hiragino Kaku Gothic ProN","Noto Sans JP",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:21px;margin:0 0 6px;letter-spacing:.01em}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 18px}
.panel{background:var(--surface);border:1px solid var(--hairline);border-radius:10px;padding:14px}
.ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:13px;color:var(--muted);margin-bottom:10px}
.ctl label{display:flex;gap:5px;align-items:center;cursor:pointer}
button{font:inherit;font-size:13px;padding:3px 12px;border-radius:6px;border:1px solid var(--hairline);background:var(--bg);color:var(--ink);cursor:pointer}
#date{font-variant-numeric:tabular-nums;color:var(--ink);min-width:92px;display:inline-block}
input[type=range]{flex:1;min-width:200px;accent-color:var(--accent)}
canvas{width:100%;display:block;border-radius:8px}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--surface);border:1px solid var(--hairline);border-radius:8px;padding:9px 11px;font-size:12.5px;max-width:300px;box-shadow:0 6px 24px rgba(0,0,0,.14);z-index:9}
#tip b{font-size:13px}.kv{color:var(--muted)}.mono{font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:18px;font-size:12.5px;color:var(--muted);margin-top:12px}
.swatch{display:inline-block;width:34px;height:9px;border-radius:2px;vertical-align:middle;margin-right:5px;background:linear-gradient(90deg,var(--neg),var(--hairline),var(--pos))}
.note{font-size:12.5px;color:var(--muted);margin-top:16px;border-left:2px solid var(--hairline);padding-left:12px}
</style>
<div class="wrap">
<h1>Composed radar — 位置＝事業内容、色と大きさ＝値動き</h1>
<p class="sub">位置は事業記述テキスト（TF-IDF→LSA ${full.rank}次元）の距離をMDSで配置。色＝3ヶ月モメンタム、大きさ＝3ヶ月ボラティリティ。<b>値動きは距離行列に一切入っていない</b>ので、日付を${frames.length}時点動かしても点は1つも動かない。等高線はMDS座標上の適応KDE、同心円は2D半径のパーセンタイル。軸に意味はない。</p>
<div class="panel">
<div class="ctl">
<button id="play">▶ 再生</button>
<span id="date"></span>
<input type="range" id="slider" min="0" max="${frames.length - 1}" value="${frames.length - 1}">
</div>
<div class="ctl">
<label><input type="checkbox" id="t-kde" checked> 等高線</label>
<label><input type="checkbox" id="t-ring" checked> 同心円</label>
<label><input type="checkbox" id="t-lab"> ラベル</label>
<label><input type="checkbox" id="t-guard" checked> 非同質ピアを破線表示</label>
</div>
<canvas id="radar"></canvas>
<div class="legend">
<span><i class="swatch"></i>モメンタム 負 → 正</span>
<span>◦ → ● ボラティリティ 低 → 高</span>
<span>同心円 = 際立ち度 30/60/90%</span>
<span>破線 = ピア基準 z が信用できない銘柄</span>
</div>
</div>
<p class="note">この図はPart C–Eの結論の合成物。目的関数にraw stressを使えるのは、C1で読み出し（半径パーセンタイル）が目的関数にほぼ不変（相関0.94–1.00）と実測されたため。値動きを別レイヤーにしたのは、混ぜると地図が0.932ずれ、6ヶ月の価格更新だけで0.819動くと実測されたため（D-H1/D-H2）。位置をテキストにしたのは、ファンダメンタルズ近傍がセクターの代理変数でしかなかったため（D-H3）。等高線をパーセンタイル半径ではなくMDS座標で計算しているのは、1/rが中心に偽の高密度を作るため（A4）。破線は、最近傍がそもそも近くない銘柄（凝集度&gt;${FLAG.cohesionPct}%）またはピア群のsdが全体の${FLAG.sdRatio}倍未満の銘柄で、ピア基準zを信用しない印。</p>
</div>
<div id="tip"></div>
<script>
const DATA=${JSON.stringify(bundle)};
const cv=document.getElementById("radar"),tip=document.getElementById("tip"),slider=document.getElementById("slider"),dateEl=document.getElementById("date"),playBtn=document.getElementById("play");
let fi=DATA.frames.length-1,opt={kde:1,ring:1,lab:0,guard:1},W=0,H=0,scale=1,ox=0,oy=0,timer=null;
const css=(v)=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function mix(a,b,t){const p=(h)=>{h=h.replace('#','');return[parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];};const A=p(a),B=p(b);return"rgb("+A.map((v,i)=>Math.round(v+(B[i]-v)*t)).join(",")+")";}
function colour(m){const t=Math.max(-1,Math.min(1,m/DATA.momClamp));const base=css("--hairline");return t>=0?mix(base,css("--pos"),t):mix(base,css("--neg"),-t);}
const radius=(vp)=>2.6+5.2*vp;
function layout(){const w=cv.clientWidth,dpr=devicePixelRatio||1;W=w;H=Math.round(w*0.74);cv.width=W*dpr;cv.height=H*dpr;cv.getContext("2d").setTransform(dpr,0,0,dpr,0,0);scale=(Math.min(W,H)/2-40)/DATA.maxR;ox=W/2;oy=H/2;}
const px=(p)=>[ox+p.x*scale,oy-p.y*scale];
function draw(){const f=DATA.frames[fi],c=cv.getContext("2d");c.clearRect(0,0,W,H);dateEl.textContent=f.d;
 if(opt.kde){c.lineWidth=1;c.lineJoin="round";c.strokeStyle=css("--terrain");DATA.contours.forEach(k=>{c.globalAlpha=k.opacity;c.beginPath();for(const s of k.segs){c.moveTo(ox+s[0]*scale,oy-s[1]*scale);c.lineTo(ox+s[2]*scale,oy-s[3]*scale);}c.stroke();});c.globalAlpha=1;}
 if(opt.ring){c.strokeStyle=css("--ring");c.setLineDash([3,4]);c.lineWidth=1;DATA.ringR.forEach(r=>{c.beginPath();c.arc(ox,oy,r*scale,0,7);c.stroke();});c.setLineDash([]);}
 DATA.points.forEach((p,i)=>{const[a,b]=px(p),r=radius(f.vp[i]);
  c.beginPath();c.arc(a,b,r,0,7);c.fillStyle=colour(f.m[i]);c.fill();
  if(opt.guard&&f.bad[i]){c.setLineDash([2,2]);c.strokeStyle=css("--muted");c.globalAlpha=.95;c.lineWidth=1.4;c.stroke();c.setLineDash([]);}
  else{c.strokeStyle=css("--ink");c.globalAlpha=.35;c.lineWidth=.8;c.stroke();}
  c.globalAlpha=1;});
 if(opt.lab){c.fillStyle=css("--muted");c.font="10px system-ui";c.textAlign="center";DATA.points.forEach((p,i)=>{const[a,b]=px(p);c.fillText(p.t.replace(".T",""),a,b-radius(f.vp[i])-3);});}
}
function hit(mx,my){let best=-1,bd=1e9;DATA.points.forEach((p,i)=>{const[a,b]=px(p),d=Math.hypot(a-mx,b-my);if(d<bd&&d<14){bd=d;best=i;}});return best;}
cv.addEventListener("mousemove",(e)=>{const r=cv.getBoundingClientRect(),i=hit(e.clientX-r.left,e.clientY-r.top);
 if(i<0){tip.style.opacity=0;return;}const p=DATA.points[i],f=DATA.frames[fi];
 const g=(v,d=2)=>v===null||v===undefined?"—":v.toFixed(d);
 tip.innerHTML="<b>"+p.nm+"</b><br><span class='kv'>"+p.t+" / "+p.sec+" / テキストクラスタ c"+p.cl+"</span><hr style='border:0;border-top:1px solid var(--hairline);margin:6px 0'>"
  +"<span class='kv'>"+f.d+"</span><br>"
  +"<span class='kv'>際立ち度</span> <span class='mono'>"+p.dp+"%</span>　<span class='kv'>ピア凝集度</span> <span class='mono'>"+p.coh+"%</span><span class='kv'>（小さいほど密）</span><br>"
  +"<span class='kv'>モメンタム3M</span> <span class='mono'>"+(f.m[i]*100).toFixed(1)+"%</span><br>"
  +"<span class='kv'>　ピア基準 z</span> <span class='mono'>"+g(f.pz[i])+"</span> <span class='kv'>／縮小 z</span> <span class='mono'>"+g(f.ps[i])+"</span> <span class='kv'>／セクター基準 z</span> <span class='mono'>"+g(f.sz[i])+"</span><br>"
  +"<span class='kv'>ボラ3M</span> <span class='mono'>"+(f.vv[i]*100).toFixed(0)+"%</span> <span class='kv'>ピア基準 z</span> <span class='mono'>"+g(f.vz[i])+"</span><br>"
  +"<span class='kv'>テキスト近傍</span> "+p.peerNames.join(", ")
  +(f.bad[i]?"<br><span style='color:var(--neg)'>⚠ ピア群が非同質（凝集度 "+p.coh+"% / sd比 "+g(f.sr[i])+"）— このピア基準 z は信用しない</span>":"");
 tip.style.opacity=1;tip.style.left=Math.min(innerWidth-310,e.clientX+14)+"px";tip.style.top=(e.clientY+14)+"px";});
cv.addEventListener("mouseleave",()=>tip.style.opacity=0);
slider.addEventListener("input",(e)=>{fi=+e.target.value;draw();});
function stop(){clearInterval(timer);timer=null;playBtn.textContent="▶ 再生";}
playBtn.addEventListener("click",()=>{if(timer){stop();return;}playBtn.textContent="⏸ 停止";
 timer=setInterval(()=>{fi=(fi+1)%DATA.frames.length;slider.value=fi;draw();},420);});
for(const[id,k]of[["t-kde","kde"],["t-ring","ring"],["t-lab","lab"],["t-guard","guard"]])document.getElementById(id).addEventListener("change",(e)=>{opt[k]=e.target.checked?1:0;draw();});
addEventListener("resize",()=>{layout();draw();});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
layout();draw();
</script>`;

fs.writeFileSync(path.join(ROOT, "out", "layers", "composed-radar.html"), html);

console.log(`=== Stage D2 — composed radar ===\n`);
console.log(`identity: text LSA (${TOKENIZER}, rank ${full.rank}) on ${n} stocks — one map, ${frames.length} dates`);
console.log(`overlay : mom3m (colour, clamp ±${(100 * momClamp).toFixed(1)}%), vol3m (size), ${series[0].date} .. ${series[series.length - 1].date}`);

const last = series[series.length - 1];
const flagged = last.momPeer.filter((r) => r.unreliable).length;
console.log(`\npeer guard at ${last.date}: ${flagged}/${n} readings flagged (cohesion > ${FLAG.cohesionPct}% or sd ratio < ${FLAG.sdRatio})`);

// The cohesion criterion is fixed by the identity space and flags a constant ~25%
// (it is a percentile). Everything above that comes from the sd criterion, which
// moves with the market: when peers happen to move together, the denominator
// shrinks and the peer-relative reading genuinely is fragile that month.
const cohOnly = points.filter((p) => p.coh > FLAG.cohesionPct).length;
console.log(`  of which cohesion-driven: ${cohOnly} at every date (percentile, fixed by the identity space)`);
const counts = series.map((s) => s.momPeer.filter((r) => r.unreliable).length);
console.log(`  flagged over the ${counts.length} dates: min ${Math.min(...counts)}  median ${[...counts].sort((a, b) => a - b)[Math.floor(counts.length / 2)]}  max ${Math.max(...counts)}`);
console.log(`  -> the guard is regime-dependent: the sd criterion fires when peer momentum is tightly clustered`);
const kept = [...Array(n).keys()].filter((i) => !last.momPeer[i].unreliable)
  .sort((a, b) => last.momPeer[b].z - last.momPeer[a].z);
console.log("\nmomentum vs identity peers, guard applied:");
for (const i of [...kept.slice(0, 3), ...kept.slice(-3)])
  console.log(
    `  ${last.momPeer[i].z.toFixed(2).padStart(6)}  coh ${String(points[i].coh).padStart(3)}%  ` +
      `${String(points[i].nm).slice(0, 26).padEnd(28)} ${points[i].peerNames.join(", ")}`,
  );

console.log(`\nsaved: out/layers/composed-radar.html (${(html.length / 1024).toFixed(0)} KB, self-contained)`);
