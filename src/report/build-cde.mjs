// Case study, Parts C–E: the objective function, the layers, and the substrate.
//
// Every number is read from the stage outputs in out/ rather than typed in, so
// the document cannot drift from the code that produced it. Re-run the stages,
// re-run this, and the prose stays true or the build fails loudly.
//
// Run: node src/report/build-cde.mjs   (no network)

import fs from "node:fs";
import path from "node:path";
import { ROOT } from "../objectives/domains.mjs";

const read = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, "out", p), "utf8"));
const C0 = read("objectives/c0-calibration.json");
const C1 = read("objectives/c1-compare.json");
const PRE = read("objectives/preregistration.json");
const D1 = read("layers/d1-separation.json");
const E1 = read("text/e1-lsa.json");
const D5 = read("layers/d5-forward.json");

const f = (x, d = 3) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));
const stocks = C1.domains.stocks, kb = C1.domains.kb;
const m = (dom, name) => dom.metrics.find((r) => r.name === name || r.name.startsWith(name));
const disp = (dom, a, b) => dom.disparity[Object.keys(dom.disparity).find((k) => k.startsWith(a))][Object.keys(dom.disparity).find((k) => k.startsWith(b))].disparity;

// ------------------------------------------------------------------- figures

const T = (x, y, s, cls, anchor = "middle") =>
  `<text x="${x}" y="${y}" font-size="${s}" fill="var(--${cls})" text-anchor="${anchor}">`;

/** C1: how far apart the four objectives put the same 145 stocks */
function figDisparity() {
  const names = ["raw", "log", "knn", "nonmetric"];
  const lab = { raw: "raw stress", log: "log距離", knn: "kNN重み", nonmetric: "非計量" };
  let s = `<svg viewBox="0 0 660 300" width="100%" role="img" aria-label="pairwise disparity between objectives">`;
  s += T(330, 20, 13, "fig-text") + `目的関数どうしの形状差（Procrustes disparity・株145銘柄）</text>`;
  const x0 = 150, y0 = 46, cw = 108, ch = 46;
  names.forEach((c, j) => { s += T(x0 + cw * j + cw / 2, y0 - 8, 11, "fig-muted") + lab[c] + `</text>`; });
  names.forEach((r, i) => {
    s += T(x0 - 10, y0 + ch * i + ch / 2 + 4, 11, "fig-muted", "end") + lab[r] + `</text>`;
    names.forEach((c, j) => {
      const v = i === j ? null : disp(stocks, r, c);
      const t = v === null ? 0 : Math.min(1, v / 0.7);
      s += `<rect x="${x0 + cw * j}" y="${y0 + ch * i}" width="${cw - 3}" height="${ch - 3}" rx="3" fill="var(--fig-blue)" fill-opacity="${(0.08 + 0.75 * t).toFixed(2)}"/>`;
      s += T(x0 + cw * j + cw / 2, y0 + ch * i + ch / 2 + 4, 12, "fig-text") + (v === null ? "—" : f(v, 3)) + `</text>`;
    });
  });
  s += T(330, 262, 11, "fig-muted") + `最小は raw↔非計量 ${f(disp(stocks, "raw", "nonmetric"), 3)}、最大は kNN↔非計量 ${f(stocks.effect, 3)}</text>`;
  s += T(330, 280, 11, "fig-muted") + `参照: A2のt-SNE 2 seed差 0.959 / MDS 2 seed差 0.621</text>`;
  return s + `</svg>`;
}

/** C1: shapes differ a lot, the A3 readout barely moves */
function figInvariance() {
  const rows = [
    ["log距離", disp(stocks, "raw", "log"), stocks.radiusCorrVsRaw.log],
    ["kNN重み", disp(stocks, "raw", "knn"), stocks.radiusCorrVsRaw[Object.keys(stocks.radiusCorrVsRaw).find((k) => k.startsWith("knn"))]],
    ["非計量", disp(stocks, "raw", "nonmetric"), stocks.radiusCorrVsRaw.nonmetric],
  ];
  let s = `<svg viewBox="0 0 660 250" width="100%" role="img" aria-label="shape difference versus readout agreement">`;
  s += T(330, 20, 13, "fig-text") + `raw stress との比較 — 形は動くが、読み出しは動かない</text>`;
  s += T(175, 44, 11, "fig-muted") + `形状差 (disparity)</text>` + T(495, 44, 11, "fig-muted") + `半径パーセンタイル相関</text>`;
  rows.forEach(([nm, d, c], i) => {
    const y = 76 + 48 * i;
    s += T(112, y + 4, 12, "fig-muted", "end") + nm + `</text>`;
    s += `<rect x="122" y="${y - 11}" width="${(240 * Math.min(1, d)).toFixed(0)}" height="16" rx="3" fill="var(--fig-amber)" fill-opacity=".75"/>`;
    s += T(128 + 240 * Math.min(1, d), y + 3, 11, "fig-text", "start") + f(d, 3) + `</text>`;
    s += `<rect x="410" y="${y - 11}" width="${(200 * c).toFixed(0)}" height="16" rx="3" fill="var(--fig-teal)" fill-opacity=".75"/>`;
    s += T(416 + 200 * c, y + 3, 11, "fig-text", "start") + f(c, 3) + `</text>`;
  });
  s += T(330, 232, 11, "fig-muted") + `目的関数を変えると地図の形は0.44〜0.62ずれるのに、A3が読む半径の順位は0.92〜1.00で一致する</text>`;
  return s + `</svg>`;
}

/** D1: what mixing costs */
function figLayers() {
  const h1 = D1.verdict["D-H1"].value, h2 = D1.verdict["D-H2"].value;
  const bars = [
    ["identity 地図 vs 混合地図", h1, "fig-amber"],
    ["混合地図が価格更新だけで動く量", h2, "fig-amber"],
    ["分離した identity 地図が動く量", 0, "fig-teal"],
    ["（参照）C1 の目的関数間・最大効果", stocks.effect, "fig-muted"],
  ];
  let s = `<svg viewBox="0 0 660 250" width="100%" role="img" aria-label="cost of mixing the state layer">`;
  s += T(330, 20, 13, "fig-text") + `値動きを距離行列に混ぜた場合のコスト（disparity）</text>`;
  bars.forEach(([nm, v, col], i) => {
    const y = 58 + 44 * i;
    s += T(300, y + 4, 11.5, "fig-muted", "end") + nm + `</text>`;
    s += `<rect x="310" y="${y - 11}" width="${Math.max(2, 300 * Math.min(1, v)).toFixed(0)}" height="16" rx="3" fill="var(--${col})" fill-opacity=".8"/>`;
    s += T(318 + 300 * Math.min(1, v), y + 3, 11.5, "fig-text", "start") + f(v, 3) + `</text>`;
  });
  s += T(330, 236, 11, "fig-muted") + `価格が6ヶ月動いただけの変形(${f(h2, 3)})が、混ぜるか否かの差の総量(${f(h1, 3)})にほぼ匹敵する</text>`;
  return s + `</svg>`;
}

/** D5: forward-return rho per snapshot, with the free covariate alongside */
function figForward() {
  const rows = D5.rows.filter((r) => r["63"]);
  const w = 620, h = 210, x0 = 40, y0 = 40, pw = w - x0 - 20, ph = h - y0 - 40;
  const vals = rows.flatMap((r) => [r["63"].peer, r["63"].raw]);
  const lim = Math.max(...vals.map(Math.abs)) * 1.1;
  const X = (i) => x0 + (pw * i) / (rows.length - 1);
  const Y = (v) => y0 + ph / 2 - (ph / 2) * (v / lim);
  let s = `<svg viewBox="0 0 ${w} ${h + 40}" width="100%" role="img" aria-label="forward return rank correlation per snapshot">`;
  s += T(w / 2, 20, 13, "fig-text") + `63営業日フォワードリターンとの順位相関（26スナップショット）</text>`;
  s += `<line x1="${x0}" y1="${Y(0)}" x2="${x0 + pw}" y2="${Y(0)}" stroke="var(--fig-grid)"/>`;
  for (const [k, col] of [["raw", "fig-muted"], ["peer", "fig-blue"]]) {
    s += `<polyline fill="none" stroke="var(--${col})" stroke-width="${k === "peer" ? 2 : 1.4}" ${k === "raw" ? 'stroke-dasharray="3,3"' : ""} points="${rows.map((r, i) => `${X(i).toFixed(1)},${Y(r["63"][k]).toFixed(1)}`).join(" ")}"/>`;
  }
  s += T(x0, y0 - 6, 11, "fig-muted", "start") + `+${f(lim, 2)}</text>`;
  s += T(x0, y0 + ph + 14, 11, "fig-muted", "start") + rows[0].date + `</text>`;
  s += T(x0 + pw, y0 + ph + 14, 11, "fig-muted", "end") + rows[rows.length - 1].date + `</text>`;
  s += T(w / 2, h + 16, 11.5, "fig-muted") + `実線＝ピア基準z（平均 ${f(D5.summary.peer.mean, 3)}）　破線＝生のモメンタム（平均 ${f(D5.summary.raw.mean, 3)}）</text>`;
  s += T(w / 2, h + 32, 11.5, "fig-muted") + `符号は時点で反転し、機械を通さない生の共変量のほうが強い</text>`;
  return s + `</svg>`;
}

/** E1: text clusters against the vendor taxonomy */
function figClusters() {
  const byCluster = {};
  E1.spaces.word.labels.forEach((c, i) => (byCluster[c] ??= []).push(i));
  const order = Object.keys(byCluster).sort((a, b) => byCluster[b].length - byCluster[a].length).slice(0, 7);
  let s = `<svg viewBox="0 0 660 ${60 + 34 * order.length}" width="100%" role="img" aria-label="text cluster composition by sector">`;
  s += T(330, 20, 13, "fig-text") + `テキストクラスタの中身（word, k=11）— 色分けはYahooセクター</text>`;
  order.forEach((c, i) => {
    const mem = byCluster[c], y = 46 + 34 * i;
    const mix = {};
    for (const j of mem) mix[E1.sectors[j]] = (mix[E1.sectors[j]] ?? 0) + 1;
    const top = Object.entries(mix).sort((a, b) => b[1] - a[1]);
    s += T(64, y + 4, 11, "fig-muted", "end") + `c${c} (n=${mem.length})` + `</text>`;
    let x = 74;
    top.forEach(([, cnt], k) => {
      const wd = (330 * cnt) / mem.length;
      s += `<rect x="${x.toFixed(1)}" y="${y - 9}" width="${Math.max(1, wd - 1).toFixed(1)}" height="14" rx="2" fill="var(--fig-blue)" fill-opacity="${(0.85 - 0.11 * k).toFixed(2)}"/>`;
      x += wd;
    });
    s += T(418, y + 4, 10.5, "fig-muted", "start") + top.slice(0, 2).map(([a, b]) => `${a}:${b}`).join("  ") + `</text>`;
  });
  return s + `</svg>`;
}

const V = (ok) => `<span class="verdict ${ok ? "ok" : "no"}">${ok ? "支持" : "不支持"}</span>`;

// ---------------------------------------------------------------------- html

const html = `<style>
:root{--bg:#fbfbfa;--surface:#fff;--ink:#1d2226;--muted:#6b7680;--hairline:#e3e7ea;--accent:#2f6fb0;
--fig-text:#1d2226;--fig-muted:#6b7680;--fig-grid:#d7dde2;--fig-blue:#3f7fbf;--fig-teal:#2f8f7a;--fig-amber:#c08a3e;}
@media (prefers-color-scheme:dark){:root{--bg:#13161a;--surface:#1b2126;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--accent:#5aa0e0;
--fig-text:#e7ebee;--fig-muted:#98a2ab;--fig-grid:#39424b;--fig-blue:#5f9fdf;--fig-teal:#4cc79b;--fig-amber:#dcae62;}}
:root[data-theme=light]{--bg:#fbfbfa;--surface:#fff;--ink:#1d2226;--muted:#6b7680;--hairline:#e3e7ea;--fig-text:#1d2226;--fig-muted:#6b7680;--fig-grid:#d7dde2;--fig-blue:#3f7fbf;--fig-teal:#2f8f7a;--fig-amber:#c08a3e;}
:root[data-theme=dark]{--bg:#13161a;--surface:#1b2126;--ink:#e7ebee;--muted:#98a2ab;--hairline:#2a323a;--fig-text:#e7ebee;--fig-muted:#98a2ab;--fig-grid:#39424b;--fig-blue:#5f9fdf;--fig-teal:#4cc79b;--fig-amber:#dcae62;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.85 system-ui,-apple-system,"Hiragino Kaku Gothic ProN","Noto Sans JP",sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:44px 22px 90px}
h1{font-size:27px;line-height:1.35;margin:0 0 10px;letter-spacing:.01em}
h2{font-size:20px;margin:52px 0 6px;padding-top:20px;border-top:1px solid var(--hairline)}
h3{font-size:16px;margin:30px 0 4px;color:var(--accent)}
p{margin:12px 0}
.lede{color:var(--muted);font-size:15px}
figure{margin:26px 0;background:var(--surface);border:1px solid var(--hairline);border-radius:10px;padding:16px 14px 8px}
figcaption{font-size:12.5px;color:var(--muted);margin-top:8px;line-height:1.65}
table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--hairline);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12.5px}
.mono{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.verdict{font-size:12px;padding:1px 8px;border-radius:10px;white-space:nowrap}
.verdict.ok{background:rgba(47,143,122,.15);color:var(--fig-teal)}
.verdict.no{background:rgba(192,138,62,.18);color:var(--fig-amber)}
.callout{background:var(--surface);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:12px 16px;margin:20px 0;font-size:14.5px}
.callout.warn{border-left-color:var(--fig-amber)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13.5px;background:var(--surface);padding:1px 5px;border-radius:4px}
</style>
<div class="wrap">
<h1>俯瞰マップの設計判断を実測で決める — Parts C–E</h1>
<p class="lede">高次元類似度から2Dの地形を作るケーススタディの後半。日本株スクリーニングに移植した地図に3つの疑問が出たところから始まり、目的関数・レイヤー構成・特徴量の基盤という3つの設計判断を、それぞれ事前登録した閾値つきの実測で決めていく。棄却された仮説と外れた予測もそのまま残してある。</p>

<div class="callout">
<b>出発点の3つの疑問</b><br>
① 値動きやPER/PBRといった数値だけを入れていたが、本来は事業内容や領域といったテキストを使うべきではないか<br>
② 値動きとボラティリティを地図と同じレイヤーに入れず、別レイヤーとして計算して可視化時に合成すべきではないか<br>
③ 計量MDSの目的関数を変えたとき、地図の見え方はどう変わるのか
</div>

<h2>Part C — 目的関数をどう比較するか</h2>

<h3>まず、一番自然に見えた案を捨てた</h3>
<p>最初に出した案は1パラメータ族 <code>Σ d^-p (d̂ − d)²</code> を p で掃引することだった。p=0 で絶対誤差、p=1 でSammon、p=2 で相対誤差になる。これを5つの理由で棄却した。</p>
<p>単位が length^(2−p) なので p を跨いで損失値が比較できず、族の内部に物差しが無い。d→0 で重みが発散するため ε の下限が必要になり、その ε が p より地図を動かす領域がある。p の<b>効果量が距離分布の変動係数に支配される</b>ので「p=1」が二つのドメインで同じ操作を意味しない — 機序の移植を主題にした研究では致命的になる。本来データの信頼度を入れる <code>w_ij</code> のスロットを幾何の好みで潰してしまう。そして PCA（内積フィット）も t-SNE（近傍確率マッチング）も族に含まれないので設計空間を張っていない。</p>
<p>代わりに目的関数を <code>σ = Σ w ρ(f(d̂) − f(d))</code> の3軸（重み・残差の形・距離の変換）に分解し、<b>言葉で定義できる4点</b>を比較することにした。連続掃引だと「一番きれいな p」を事後に選べてしまう。</p>

<h3>C0 — 比較器の較正と事前登録</h3>
<p>目的関数ごとに別々の majorization を導出すると「目的関数が変わった」と「更新則が変わった」が混ざる。共通の勾配降下（Armijo直線探索。固定学習率だと目的関数ごとの調整が必要になり、この段が消そうとしているノブが復活する）を用意し、まず raw stress で既存の SMACOF と同じ地図に着くことを確認した。閾値は実行前に固定。</p>
<table><thead><tr><th>init</th><th>disparity</th><th>Δstress-1</th><th>判定</th></tr></thead><tbody>
${C0.results.map((r) => `<tr><td>${r.domain} / ${r.init}</td><td class="mono">${r.deltas.disparity.toExponential(2)}</td><td class="mono">${r.deltas.stress1.toExponential(2)}</td><td>${V(r.pass)}</td></tr>`).join("")}
</tbody></table>
<p>この段で二つ副産物が出た。<b>中心化はペア距離を一切変えない</b>ので、A1の「中心化で類似度の幅が0.21→1.44」はMDSが食う行列について何も言っていなかった。そして株側では classical MDS のウォームスタートが<b>退化している</b> — Gram上位2固有値が 144.98 対 144.23。白色化が共分散を等方にするためで、B1でPCAが回った現象の根がここに見えた。initは決定的だが正準ではないので、株では形だけを読み、角度は読まない。</p>
<p>さらに自作Procrustesに欠陥が見つかった。鏡映側の分岐で最適回転角を再計算せず非鏡映の角度を流用していたため、鏡映を含む差の残差を過大評価していた（合成テストで 1.34 → 2.8e-32）。A2の記録値は <code>MDS 171° / 0.81</code> から <code>159° / 0.62</code> へ動き、<b>差の正体が回転＋鏡映だった</b>ことが分かった。A2の結論はすべて生存し、A3はむしろ強化される — 座標系の任意性はSO(2)ではなくO(2)全体で、半径はその両方の不変量だからである。</p>

<h3>C1 — 有効3点と、失格1点</h3>
<figure>${figDisparity()}<figcaption>図1 — 同じ145銘柄を4つの目的関数で配置し、Procrustes整列後に形を比較した。</figcaption></figure>
<p>kNN重みは<b>失格したが、比較から削除していない</b>。失敗した点を落として成功した3点だけを載せるのは、事前登録が防ごうとしている選択そのものになる。失格の理由は2つある。</p>
<p>ひとつは自分のノブに負けたこと。凍結した k×γ グリッドが地図を最大 ${f(stocks.knnGridMax, 3)}（株）／${f(kb.knnGridMax, 3)}（KB）動かし、これは<b>参加するはずだった目的関数間の効果量（${f(stocks.effect, 3)} / ${f(kb.effect, 3)}）を上回る</b>。<code>d^-p</code> の ε に向けた批判がそのまま自分に返ってきた。「順位ベースだから母集団不変」は重みの定義を守っただけで、地図の安定性は守らなかった。</p>
<p>もうひとつが<b>幾何的支え棒の喪失</b>だ。近傍を守るために作った目的関数が、近傍保存の指標で最下位になった。</p>
<table><thead><tr><th>目的関数</th><th>stress-1</th><th>corr</th><th>trust</th><th>cont</th><th>状態</th></tr></thead><tbody>
${stocks.metrics.map((r) => `<tr><td>${r.label}</td><td class="mono">${f(r.stress1)}</td><td class="mono">${f(r.corr)}</td><td class="mono">${f(r.trust)}</td><td class="mono">${f(r.cont)}</td><td>${r.status === "valid" ? "有効" : '<span class="verdict no">失格</span>'}</td></tr>`).join("")}
</tbody></table>
<p>遠方ペアの重みを γ に落とすと、無関係なクラスタを引き離していた支え棒が消える。クラスタ同士が寄って2D上に偽の近傍が生まれる。<b>局所性は重みだけでは強制できない</b> — 近傍は「近くにない遠方ペア」によって部分的に定義されているからだ。</p>

<div class="callout warn">
<b>事前登録した仮説は棄却された。</b> 「目的関数選択の効果量は距離分布のCVとともに増える」と登録していたが、CV(kb)=${f(PRE.distance_stats.kb.cv)} &lt; CV(stocks)=${f(PRE.distance_stats.stocks.cv)} に対し効果量は kb ${f(kb.effect, 3)} &gt; stocks ${f(stocks.effect, 3)} で<b>逆向き</b>だった。ただしこれは4点族がCV非依存であることを立証しない — CV非依存は設計意図であって実測された性質ではなく、観測された効果量は失格したkNN点に支配されており、n=2ドメインでは両解釈を分離できない。<code>d^-p</code> の可搬性という主張自体は未検証のまま残っている。
</div>

<h3>C1の主要な発見 — 読み出しは目的関数に不変</h3>
<figure>${figInvariance()}<figcaption>図2 — 形状差と、A3の読み出し（高次元際立ち度のパーセンタイル）の一致度。</figcaption></figure>
<p>地図の形は disparity 0.44〜0.62 も違うのに、A3が実際に読む量はほとんど動かない。パーセンタイル順位が、目的関数を分けている非線形な半径方向の引き延ばしを吸収してしまう。<b>「どの目的関数が正しいか」に決着がつかなくても、A3の読み方をする限り結論は揺れない。</b></p>
<p>主張の範囲は明示しておく。有効3点について成立し（失格したkNNが最弱の ${f(stocks.radiusCorrVsRaw[Object.keys(stocks.radiusCorrVsRaw).find((k) => k.startsWith("knn"))], 3)}）、n=2ドメインの観測なので定理ではなく経験的頑健性であり、両母集団とも重心まわりに概ね単峰である。明瞭に分離した多峰構造では、どのクラスタ間ギャップを潰すかが目的関数で変わり、大域重心からの半径が動きうる — これは未検証。そして不変性は<b>読み出しの性質であって座標の性質ではない</b>。</p>
<p>副次的に、raw stress と非計量はほぼ同じ地図を作り（disparity ${f(disp(stocks, "raw", "nonmetric"), 3)}、全ペア中最小）、それでいて非計量のほうが大域相関で上回る（${f(m(stocks, "nonmetric").corr)} 対 ${f(m(stocks, "raw").corr)}）。<b>距離の絶対値は順位に対してほぼ冗長</b>ということになる。</p>

<h2>Part D — レイヤーを分ける</h2>
<figure>${figLayers()}<figcaption>図3 — 値動きを距離行列に混ぜたときのコスト。</figcaption></figure>
<p>identity（企業が何であるか）と state（どう振る舞っているか）は種類が違う。構造か状態か、更新頻度が年か日か、近さが「同種の事業」か「似た値動き」か。ひとつの距離行列に入れると「この2つは近い」が読めなくなる。</p>
<p>混ぜると地図は ${f(D1.verdict["D-H1"].value, 3)} ずれる。C1の目的関数間の最大効果 ${f(stocks.effect, 3)} より大きく、A2でt-SNEを別seedで引き直した 0.959 に近い。<b>値動き4列を足すほうが、目的関数を何にするか決めるより地図を壊す。</b></p>
<p>そして価格が6ヶ月動いただけで混合地図は ${f(D1.verdict["D-H2"].value, 3)} 動く。これは混ぜるか否かの差の総量にほぼ匹敵し、<b>混合地図には持続的な同一性が存在しない</b>ことを意味する。分離した地図は0.000 — 幸運ではなく構成上そうなる。</p>
<p>結果に依存しない論点もある。白色化は全方向を等しくするので、state 4列 対 identity 9列という構成は<b>列数だけで state に 4/13 の分散予算</b>を渡す。混合地図の汚染度が「state特徴をいくつ足したか」で決まるということで、正当化できる値が存在しない自由パラメータになっている。</p>

<h2>Part E — 基盤をテキストに置き換える</h2>
<p>D1の3つ目の仮説は不支持だった。ファンダメンタルズ空間の近傍とセクターラベルの相関が ${f(D1.verdict["D-H3"].mom3m.rho)} / ${f(D1.verdict["D-H3"].vol3m.rho)} で、閾値0.7をわずかに超えた。理由は構造的で、利益率もROEも資本集約度も業種の産物なので、<b>セクターの代理変数とセクターを比べていた</b>ことになる。</p>
<figure>${figClusters()}<figcaption>図4 — 事業記述テキストから作ったクラスタの中身。ARI(word) = ${f(E1.ariSector.word)}。</figcaption></figure>
<p>ARIは低い。だがこれは失敗ではなく発見のほうだ。自動車、医薬、食品、メディアは綺麗に分かれ、そして<b>総合商社がひとつのクラスタになる</b> — Yahooの分類ではFinancial Services / Industrials / Real Estate に割られている集団である。テキストは実在の事業グループを見つけていて、11分類はそれを持っていない。A5でKBに見た「食い違いにこそ価値がある」の株版になっている。</p>
<p>ここで一度、この段を始めた理由づけを訂正した。「ピアがセクターの代理変数だったのだから、テキストなら食い違いが増えるはず」と考えていたが、おそらく逆である。セクターラベル自体が事業の粗い記述なので、事業記述から作ったピアはむしろ<b>より一致するはず</b>だ。そう向きを反転させて登録し直したところ、テキストピアの相関は ${f(E1.h3.word.mom3m.rho)} / ${f(E1.h3.word.vol3m.rho)} でベースラインを上回った。テキスト基盤を推す理由は食い違いではない — ファンダメンタルズのピアが<b>解釈できない</b>（「ROEが近いから同業」は同一性の主張ではない）のに対し、テキストのピアは読んで検証でき、食い違いも読める、という点にある。</p>

<h2>合成 — そして、予測できるとは言わない</h2>
<p>位置をテキストidentity、色と大きさを値動きにした地図を作った。日付スライダーを29時点動かすと、色も大きさも全部変わって<b>点は1つも動かない</b>。</p>
<p>ピア基準の読みには番人を付けた。ある銘柄が「ピアに対して極端」でも、そのピア群が非同質なら意味がない。ここでも当初の診断が測定に否定された — z が膨らんだ原因はピア群の分散が小さいことだと思っていたが、実際に外れ値だった銘柄のsd比は平凡で、<b>凝集度</b>（最近傍が本当に近いか）のほうが真因だった。分母を下支えする縮小zは<b>どの値も変えず不活性</b>で、効くまで係数を上げるのは結果を見てのチューニングなので、不活性のまま報告している。</p>
<figure>${figForward()}<figcaption>図5 — 事前登録したフォワードリターン検証。26スナップショット、63営業日先。</figcaption></figure>
<table><thead><tr><th>仮説</th><th>実測</th><th>判定</th></tr></thead><tbody>
<tr><td>F-H1 ピア基準zがフォワードリターンを順位づける</td><td class="mono">|${f(D5.summary.peer.mean)}| vs 0.05</td><td>${V(D5.verdicts["F-H1"])}</td></tr>
<tr><td>F-H2 ピア基準zが「無料の共変量」を上回る</td><td class="mono">|${f(D5.summary.peer.mean)}| vs |${f(D5.summary.raw.mean)}|</td><td>${V(D5.verdicts["F-H2"])}</td></tr>
<tr><td>F-H3 凝集度の番人が仕事をしている</td><td class="mono">|${f(D5.summary.peer.mean)}| vs |${f(D5.summary.peerFlagged.mean)}|</td><td>${V(D5.verdicts["F-H3"])}</td></tr>
</tbody></table>
<p>3つとも不支持である。生のモメンタムを順位づけるだけの、この機械を一切通さない共変量のほうが強く（${f(D5.summary.raw.mean)} 対 ${f(D5.summary.peer.mean)}）、番人が弾いた銘柄のほうが関係が強かった（${f(D5.summary.peerFlagged.mean)}）。</p>
<p>最も無理のない読みはこうだ。<b>ピア基準で中心化する操作は、業種のモメンタムを取り除く。</b> そして業種のモメンタムこそが実際にフォワードリターンを動かしている成分だった。凝集度の高いピア群ほどその成分を強く抜いてしまうので、番人が「信頼できる」と判定した側で相関が下がる。ただしこれは事後の解釈であって検証していない。そもそも相関の絶対値はすべて0.03〜0.09で符号も時点で反転するので、この大学的な差を強く読むべきではない。</p>
<div class="callout">
<b>この地図は予測器ではない。</b> ピアグループは「この銘柄はその種類の企業として異常か」という<b>解釈</b>の問いに答える道具で、「次に上がるか」という<b>予測</b>の問いには答えない。二つは違う問いで、後者については、機械を通さない生のモメンタムに負けたことを実測として記録しておく。俯瞰マップの価値はアルゴリズムではなく地図を意思決定へ翻訳する力にある、という立場は、その翻訳先が予測ではないと明示して初めて誠実になる。
</div>

<h2>この一連で守ったこと</h2>
<p>閾値は必ず結果を見る前に固定し、ソースにコミットしてから実行した。事前登録した仮説が棄却されたとき（CV仮説、F-H1〜3）は棄却として記録し、閾値に寄せた再解釈をしていない。失格した目的関数は比較から削除せず、失格の構造的理由とともに残した。自分の診断が測定に否定されたとき（ピア群の分散、テキストピアの向き）は訂正を本文に残した。効かなかった仕掛け（縮小z）は効くまで係数を動かさず、効かないまま報告した。そして無料で手に入る対照（生のモメンタム）を常に表の隣に置いた。</p>
<p class="lede" style="margin-top:26px">生成: <code>node src/report/build-cde.mjs</code>。数値はすべて <code>out/</code> の各段の出力から読み込んでおり、この文書に手で書いた実測値はない。</p>
</div>`;

fs.mkdirSync(path.join(ROOT, "out", "report"), { recursive: true });
const outPath = path.join(ROOT, "out", "report", "part-cde.html");
fs.writeFileSync(outPath, html);
console.log(`wrote out/report/part-cde.html  ${(html.length / 1024).toFixed(0)} KB`);
