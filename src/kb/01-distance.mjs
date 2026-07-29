// Stage A1 — Distance & anisotropy.
//  (1) Distribution of all pairwise cosine similarities (the "compression" test).
//  (2) Empirical proof that cosine vs Euclidean induce the SAME neighbor ranking.
//  (3) An anisotropy read: how far is the mean vector from origin? A large-norm
//      mean vector = all embeddings share a common direction (the "cone effect").
//
// Run: node src/kb/01-distance.mjs   (after npm run kb:matrix)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { cosineSim, euclideanDist } from "../lib/distance.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const payload = JSON.parse(
  fs.readFileSync(path.join(ROOT, "out", "kb", "permanent-embeddings.json"), "utf8"),
);
const V = payload.vectors.map((v) => Float64Array.from(v));
const n = V.length;

// (1) pairwise cosine similarities
const sims = [];
for (let i = 0; i < n; i++)
  for (let j = i + 1; j < n; j++) sims.push(cosineSim(V[i], V[j]));
sims.sort((a, b) => a - b);
const q = (p) => sims[Math.min(sims.length - 1, Math.floor(p * (sims.length - 1)))];
const mean = sims.reduce((a, b) => a + b, 0) / sims.length;

console.log("=== Stage A1: pairwise cosine similarity (n pairs =", sims.length, ") ===");
console.log(
  "min/1%/25%/median/75%/99%/max:",
  [q(0), q(0.01), q(0.25), q(0.5), q(0.75), q(0.99), q(1)].map((x) => x.toFixed(3)).join(" / "),
);
console.log("mean:", mean.toFixed(3), " range width:", (q(1) - q(0)).toFixed(3));

// ASCII histogram over [min,max]
const BINS = 20;
const lo = q(0),
  hi = q(1),
  w = (hi - lo) / BINS;
const hist = new Array(BINS).fill(0);
for (const s of sims) hist[Math.min(BINS - 1, Math.floor((s - lo) / w))]++;
const maxc = Math.max(...hist);
console.log("\nhistogram:");
for (let b = 0; b < BINS; b++) {
  const left = (lo + b * w).toFixed(2);
  const bar = "#".repeat(Math.round((hist[b] / maxc) * 40));
  console.log(`  ${left}  ${bar} ${hist[b]}`);
}

// (2) neighbor-ranking invariance: for each i, is ordering by cosine-desc identical
//     to ordering by euclidean-asc?
let exact = 0;
for (let i = 0; i < n; i++) {
  const others = [];
  for (let j = 0; j < n; j++) if (j !== i) others.push(j);
  const byCos = [...others].sort((a, b) => cosineSim(V[i], V[b]) - cosineSim(V[i], V[a]));
  const byEuc = [...others].sort((a, b) => euclideanDist(V[i], V[a]) - euclideanDist(V[i], V[b]));
  if (byCos.every((v, k) => v === byEuc[k])) exact++;
}
console.log(
  `\nneighbor-ranking invariance (cosine vs euclidean): ${exact}/${n} notes identical ordering`,
);

// (3) anisotropy: norm of the mean (centroid) direction. Vectors are unit norm,
//     so a centroid norm near 1 means they all point the same way (anisotropic);
//     near 0 means they're spread over the sphere (isotropic).
const centroid = new Float64Array(payload.dim);
for (const v of V) for (let d = 0; d < v.length; d++) centroid[d] += v[d] / n;
let cnorm = 0;
for (const x of centroid) cnorm += x * x;
cnorm = Math.sqrt(cnorm);
console.log(
  `\nanisotropy: ‖mean unit-vector‖ = ${cnorm.toFixed(3)}  (0=isotropic sphere, 1=all same direction)`,
);
