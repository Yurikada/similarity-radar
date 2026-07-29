// Stage E1 — text identity space: TF-IDF -> LSA -> clusters, against the vendor sector.
//
// Pre-registered here, in source, before the stage was run (see E1_PREREG below).
//
// A correction to the reasoning that motivated this stage. D-H3 failed with
// fundamentals-space peers correlating 0.732 / 0.713 against the sector label,
// and it was tempting to conclude that text peers would correlate LESS. That is
// probably backwards: a sector label is itself a coarse description of what a
// company does, so a peer group built from business descriptions should agree
// with it MORE, not less. The argument for a text substrate is not disagreement
// with the taxonomy. It is that a fundamentals peer group cannot be interpreted
// -- "these ten are peers because their ROE is similar" is not an identity claim
// -- whereas a text peer group is inspectable, finer-grained than eleven buckets,
// and its disagreements are readable. E-H3 therefore registers the OPPOSITE
// direction to the one first assumed.
//
// Tokenizer: both are implemented and both are run. The corpus turned out to be
// entirely English (E0: CJK share 0.0%), where character 3-grams recover letter
// sequences rather than meaning, while character n-grams are exactly right for
// the Japanese EDINET text this pilot stands in for. Running both measures
// whether the tokenizer choice dominates the signal -- the same check that
// disqualified the kNN objective in C1.
//
// Run: node src/text/01-lsa.mjs   (no network — uses the E0 cache)

import fs from "node:fs";
import path from "node:path";
import { kmeans, adjustedRand } from "../lib/kmeans.mjs";
import { euclid, ROOT } from "../objectives/domains.mjs";
import { TOKENIZERS, lsa, TFIDF } from "../lib/text/lsa.mjs";

const OUT = path.join(ROOT, "out", "text");
const C = JSON.parse(fs.readFileSync(path.join(OUT, "corpus.json"), "utf8"));

export const E1_PREREG = {
  stage: "E1",
  tokenizers: ["word", "char45"],
  tfidf: TFIDF,
  lsa_rank: "smallest k reaching 80% cumulative eigenvalue energy, capped at 100",
  kmeans_k: 11, // = number of Yahoo sectors, so ARI compares like with like
  k_sensitivity: [8, 11, 15],
  peers_k: 10,
  hypotheses: {
    "E-H1": {
      claim: "text clusters carry industry signal without merely restating the taxonomy",
      metric: "ARI(text clusters, Yahoo sector)",
      window: [0, 0.6],
      note: "A5's KB analogue against human tags was 0.448",
    },
    "E-H2": {
      claim: "the tokenizer does not dominate the signal",
      metric: "ARI(word clusters, char45 clusters) > max(ARI(word,sector), ARI(char45,sector))",
    },
    "E-H3": {
      claim:
        "text peers agree with the sector label MORE than fundamentals peers did, " +
        "because a sector label is a coarse identity description and text is a fine one",
      metric: "spearman(text-peer-z, sector-z) vs the D1 baseline",
      baseline: { mom3m: 0.732, vol3m: 0.713 },
      direction: ">",
    },
  },
};

// -------------------------------------------------------------------- run both

const texts = C.docs.map((d) => d.text);
const sectors = C.docs.map((d) => d.sector);
const sectorNames = [...new Set(sectors)];
const sectorLabels = sectors.map((s) => sectorNames.indexOf(s));

console.log(`=== Stage E1 — text identity space (n=${C.n}) ===`);
console.log(`pre-registered: E-H1 ARI in (0,0.6) | E-H2 tokenizer agreement > sector agreement | E-H3 spearman > D1 baseline\n`);

const spaces = {};
for (const [name, tok] of Object.entries(TOKENIZERS)) {
  const r = lsa(texts, tok);
  const km = kmeans(r.X, E1_PREREG.kmeans_k, { seed: 1 });
  spaces[name] = { ...r, labels: km.labels };
  console.log(
    `${name.padEnd(7)} distinct terms ${String(r.distinct).padStart(7)} -> kept ${String(r.vocab).padStart(6)}` +
      `   LSA rank ${String(r.rank).padStart(3)} (${(100 * r.energy).toFixed(1)}% energy)`,
  );
}

// ------------------------------------------------------------------- E-H1/E-H2

console.log("\n[E-H1] ARI against the Yahoo sector taxonomy   (A5's KB analogue: 0.448)");
const ariSector = {};
for (const [name, s] of Object.entries(spaces)) {
  ariSector[name] = adjustedRand(s.labels, sectorLabels);
  const ok = ariSector[name] > 0 && ariSector[name] < 0.6;
  console.log(`  ${name.padEnd(7)} ARI ${ariSector[name].toFixed(3)}   ${ok ? "SUPPORTED" : "not supported"}`);
}

const ariTok = adjustedRand(spaces.word.labels, spaces.char45.labels);
const h2 = ariTok > Math.max(...Object.values(ariSector));
console.log("\n[E-H2] tokenizer robustness");
console.log(
  `  ARI(word, char45) ${ariTok.toFixed(3)}  vs  max ARI(tokenizer, sector) ${Math.max(...Object.values(ariSector)).toFixed(3)}` +
    `   ${h2 ? "SUPPORTED" : "not supported — the tokenizer dominates"}`,
);

console.log("\n  k sensitivity (frozen grid) — ARI(word, char45) at each k");
for (const k of E1_PREREG.k_sensitivity) {
  const a = kmeans(spaces.word.X, k, { seed: 1 }).labels;
  const b = kmeans(spaces.char45.X, k, { seed: 1 }).labels;
  console.log(`    k=${String(k).padEnd(3)} ARI(word,char45) ${adjustedRand(a, b).toFixed(3)}` +
    `   ARI(word,sector) ${adjustedRand(a, sectorLabels).toFixed(3)}`);
}

// --------------------------------------------------------------------- E-H3

const ST = JSON.parse(fs.readFileSync(path.join(ROOT, "out", "layers", "state-features.json"), "utf8"));
// map corpus rows (features.json index) onto the state-layer rows
const posInCorpus = new Map(C.docs.map((d, p) => [d.base, p]));
const joint = ST.baseIndex.map((b, p) => ({ state: p, text: posInCorpus.get(b) })).filter((r) => r.text !== undefined);
console.log(`\n[E-H3] peers taken in TEXT space, overlay read against sector (n=${joint.length})`);
console.log("  registered direction: spearman ABOVE the fundamentals-peer baseline (0.732 / 0.713)");

function spearman(a, b) {
  const rank = (v) => {
    const o = [...v.keys()].sort((x, y) => v[x] - v[y]);
    const r = new Array(v.length);
    o.forEach((i, k) => (r[i] = k));
    return r;
  };
  const ra = rank(a), rb = rank(b), m = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < m; i++) { ma += ra[i] / m; mb += rb[i] / m; }
  let s = 0, sa = 0, sb = 0;
  for (let i = 0; i < m; i++) {
    const u = ra[i] - ma, v = rb[i] - mb;
    s += u * v; sa += u * u; sb += v * v;
  }
  return s / Math.sqrt(sa * sb);
}

const h3 = {};
for (const [tname, sp] of Object.entries(spaces)) {
  const Xj = joint.map((r) => sp.X[r.text]);
  const D = euclid(Xj);
  const peers = Xj.map((_, i) =>
    [...Array(Xj.length).keys()].filter((j) => j !== i).sort((a, b) => D[i][a] - D[i][b]).slice(0, E1_PREREG.peers_k));
  const sec = joint.map((r) => ST.sectors[r.state]);
  const bySector = {};
  sec.forEach((s, i) => (bySector[s] ??= []).push(i));

  h3[tname] = {};
  for (const fname of ["mom3m", "vol3m"]) {
    const fi = ST.featureNames.indexOf(fname);
    const x = joint.map((r) => ST.t2[r.state][fi]);
    const pz = [], sz = [];
    for (let i = 0; i < x.length; i++) {
      const grp = bySector[sec[i]].filter((j) => j !== i);
      if (grp.length + 1 < 5) continue;
      const z = (m) => {
        const mu = m.reduce((a, j) => a + x[j], 0) / m.length;
        const sd = Math.sqrt(m.reduce((a, j) => a + (x[j] - mu) ** 2, 0) / m.length) || 1;
        return (x[i] - mu) / sd;
      };
      pz.push(z(peers[i]));
      sz.push(z(grp));
    }
    const rho = spearman(pz, sz);
    const flips = pz.filter((v, p) => Math.sign(v) !== Math.sign(sz[p])).length;
    const base = E1_PREREG.hypotheses["E-H3"].baseline[fname];
    h3[tname][fname] = { rho, flips, n: pz.length, aboveBaseline: rho > base };
    console.log(
      `  ${tname.padEnd(7)} ${fname.padEnd(7)} spearman ${rho.toFixed(3)} (baseline ${base})  flips ${flips}` +
        `   ${rho > base ? "above" : "BELOW"} baseline`,
    );
  }
}

// -------------------------------------------------------- cluster inspection

console.log("\ncluster composition (word tokenizer, k=11) — sector mix per text cluster");
const byCluster = {};
spaces.word.labels.forEach((c, i) => (byCluster[c] ??= []).push(i));
for (const c of Object.keys(byCluster).sort((a, b) => byCluster[b].length - byCluster[a].length)) {
  const mem = byCluster[c];
  const mix = {};
  for (const i of mem) mix[sectors[i]] = (mix[sectors[i]] ?? 0) + 1;
  const top = Object.entries(mix).sort((a, b) => b[1] - a[1]);
  console.log(
    `  c${String(c).padStart(2)} n=${String(mem.length).padStart(3)}  ` +
      top.slice(0, 3).map(([s, n]) => `${s}:${n}`).join(" ").padEnd(52) +
      `  e.g. ${mem.slice(0, 3).map((i) => C.docs[i].name.split(/[ ,]/)[0]).join(", ")}`,
  );
}

fs.writeFileSync(
  path.join(OUT, "e1-lsa.json"),
  JSON.stringify({
    stage: "E1", ran_at: new Date().toISOString(), prereg: E1_PREREG,
    ariSector, ariTokenizer: ariTok, h2, h3,
    spaces: Object.fromEntries(Object.entries(spaces).map(([k, v]) => [k, { rank: v.rank, vocab: v.vocab, distinct: v.distinct, labels: v.labels }])),
    tickers: C.docs.map((d) => d.ticker), sectors,
  }, null, 2),
);
console.log("\nsaved: out/text/e1-lsa.json");
