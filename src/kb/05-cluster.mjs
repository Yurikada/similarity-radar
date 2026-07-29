// Stage A5 — clustering + the two questions:
//   (1) does clustering in high-D vs 2D differ? which to trust?
//   (2) where do algorithmic clusters DISAGREE with the human tags/Index? (value)
//
// Run: node src/kb/05-cluster.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pca2 } from "../lib/projection/pca.mjs";
import { kmeans, adjustedRand } from "../lib/kmeans.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const emb = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "kb", "permanent-embeddings.json"), "utf8"));
const V = emb.vectors.map((v) => Float64Array.from(v));
const n = V.length, dim = emb.dim;
const c = new Float64Array(dim);
for (const v of V) for (let d = 0; d < dim; d++) c[d] += v[d] / n;
const Xc = V.map((v) => Float64Array.from(v, (x, d) => x - c[d]));
const pca = pca2(Xc);

// human label = primary topic tag (first non-permanent)
const human = emb.tags.map((ts) => ts.find((t) => t !== "permanent") ?? "untagged");
const tagCounts = {};
for (const h of human) tagCounts[h] = (tagCounts[h] ?? 0) + 1;
const k = new Set(human).size;

// cluster in high-D and in 2D (same k, same seed)
const hi = kmeans(Xc.map((v) => Array.from(v)), k, { seed: 3 });
const lo = kmeans(pca, k, { seed: 3 });

console.log("=== Stage A5: clustering ===\n");
console.log(`k = ${k} (distinct primary tags), n = ${n}`);
console.log("human tags:", Object.entries(tagCounts).sort((a, b) => b[1] - a[1]).map(([t, c]) => `${t}:${c}`).join("  "), "\n");

console.log("Q1 — high-D vs 2D clustering:");
console.log("  ARI(high-D, 2D)   =", adjustedRand(hi.labels, lo.labels).toFixed(3), " (1=identical, 0=random) => they differ");
console.log("  ARI(high-D, human)=", adjustedRand(hi.labels, human).toFixed(3));
console.log("  ARI(2D,     human)=", adjustedRand(lo.labels, human).toFixed(3), "\n");

// label each high-D cluster by its dominant human tag
console.log("Q2 — high-D clusters (size, dominant tag, purity):");
for (let cl = 0; cl < k; cl++) {
  const members = [...hi.labels.keys()].filter((i) => hi.labels[i] === cl);
  if (!members.length) continue;
  const cnt = {};
  for (const i of members) cnt[human[i]] = (cnt[human[i]] ?? 0) + 1;
  const [domTag, domN] = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0];
  console.log(`  cluster ${cl}: ${members.length} notes, dominant=${domTag} (${domN}/${members.length}), mix=${Object.entries(cnt).map(([t, c]) => t + ":" + c).join(",")}`);
}

// the valuable disagreements: notes whose cluster's dominant tag != own tag
console.log("\ndisagreements (note tag ≠ its cluster's dominant tag) = cross-theme bridges:");
const dom = {};
for (let cl = 0; cl < k; cl++) {
  const members = [...hi.labels.keys()].filter((i) => hi.labels[i] === cl);
  const cnt = {};
  for (const i of members) cnt[human[i]] = (cnt[human[i]] ?? 0) + 1;
  dom[cl] = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0]?.[0];
}
const bridges = [];
for (let i = 0; i < n; i++) {
  if (human[i] !== dom[hi.labels[i]]) bridges.push({ tag: human[i], dom: dom[hi.labels[i]], label: emb.labels[i] });
}
for (const b of bridges.slice(0, 6)) console.log(`   [${b.tag} → クラスタは${b.dom}] ${b.label.slice(0, 40)}`);

const ari = {
  hiLo: adjustedRand(hi.labels, lo.labels),
  hiHuman: adjustedRand(hi.labels, human),
  loHuman: adjustedRand(lo.labels, human),
};
fs.writeFileSync(path.join(ROOT, "out", "kb", "clusters.json"),
  JSON.stringify({ labels: emb.labels, human, hi: hi.labels, pca, k, dom, ari, bridges }));
console.log("\nsaved: out/kb/clusters.json (gitignored)");
