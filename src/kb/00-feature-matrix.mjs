// Stage A0 — Build the feature matrix from KB note embeddings and report basic
// shape/health stats. Saves the matrix to out/kb/ (gitignored, private).
//
// Run: npm run kb:matrix

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadConfig, loadKbEmbeddings } from "./load-embeddings.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

function l2(vec) {
  let s = 0;
  for (const x of vec) s += x * x;
  return Math.sqrt(s);
}

function quantiles(arr, qs) {
  const s = [...arr].sort((a, b) => a - b);
  return qs.map((q) => s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]);
}

const cfg = loadConfig(ROOT);
const { labels, tags, links, vectors, dates, dim } = loadKbEmbeddings(cfg);

const norms = vectors.map(l2);
const [nMin, nMed, nMax] = quantiles(norms, [0, 0.5, 1]);

// tag distribution
const tagCount = new Map();
for (const ts of tags) for (const t of ts) tagCount.set(t, (tagCount.get(t) ?? 0) + 1);
const topTags = [...tagCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);

console.log("=== Stage A0: KB feature matrix ===");
console.log("notes (n):", labels.length);
console.log("dim:", dim);
console.log("L2 norm  min/median/max:", nMin.toFixed(4), nMed.toFixed(4), nMax.toFixed(4));
console.log("normalized?:", nMax - nMin < 1e-3 ? "YES (unit vectors)" : "NO (varying norms)");
console.log("top tags:", topTags.map(([t, c]) => `${t}:${c}`).join("  "));
console.log("sample titles:", labels.slice(0, 3));

// Persist matrix (PRIVATE — gitignored). Save vec as plain arrays for portability.
const outDir = path.join(ROOT, "out", "kb");
fs.mkdirSync(outDir, { recursive: true });
const payload = {
  model: cfg.kb.embeddingModel,
  dim,
  n: labels.length,
  labels,
  tags,
  links,
  dates,
  vectors: vectors.map((v) => Array.from(v)),
};
fs.writeFileSync(path.join(outDir, "permanent-embeddings.json"), JSON.stringify(payload));
console.log("saved:", path.relative(ROOT, path.join(outDir, "permanent-embeddings.json")), "(gitignored)");
