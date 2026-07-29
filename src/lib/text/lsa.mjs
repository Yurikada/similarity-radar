// TF-IDF + LSA over a pluggable tokenizer, extracted from E1 so the radar builder
// and the analysis stage compute the identity space from one implementation.
//
// LSA runs through the Gram route: with n≈145 documents the n×n cosine matrix is
// tiny, so document coordinates come from its eigendecomposition and the
// term-document matrix is never materialized.

import { topKEigSym } from "../projection/pca.mjs";

export const TFIDF = { sublinear_tf: true, min_df: 3, max_df_ratio: 0.9, row_norm: "l2" };

const STOP = new Set(
  ("a an and or the of in on for to with as at by from is are was were be been it its" +
    " this that these those which who whom whose has have had also into their they them" +
    " other others such under over through during about between among various including" +
    " company companies corporation limited ltd inc co group holdings japan japanese" +
    " well provides provide offers offer operates operate serves serve products services" +
    " business segment segments").split(/\s+/),
);

export const TOKENIZERS = {
  /** English words. Corporate boilerplate is dropped explicitly rather than left
   *  to IDF, since nearly every filing contains it. */
  word: (s) => s.toLowerCase().split(/[^a-z]+/).filter((w) => w.length >= 3 && !STOP.has(w)),
  /** Character 4- and 5-grams: dictionary-free, the unit intended for Japanese
   *  EDINET text. 3-grams proved too short for English (E0). */
  char45: (s) => {
    const t = s.toLowerCase().replace(/\s+/g, " ");
    const out = [];
    for (const n of [4, 5]) for (let i = 0; i + n <= t.length; i++) out.push(t.slice(i, i + n));
    return out;
  },
};

/** @returns {{X:number[][], rank:number, vocab:number, distinct:number, energy:number}} */
export function lsa(texts, tokenize, { energyTarget = 0.8, maxRank = 100 } = {}) {
  const n = texts.length;
  const tf = texts.map((t) => {
    const m = new Map();
    for (const w of tokenize(t)) m.set(w, (m.get(w) ?? 0) + 1);
    return m;
  });
  const df = new Map();
  for (const m of tf) for (const w of m.keys()) df.set(w, (df.get(w) ?? 0) + 1);

  const keep = new Set();
  for (const [w, c] of df) if (c >= TFIDF.min_df && c <= TFIDF.max_df_ratio * n) keep.add(w);

  const rows = tf.map((m) => {
    const r = new Map();
    let norm = 0;
    for (const [w, c] of m) {
      if (!keep.has(w)) continue;
      const v = (1 + Math.log(c)) * Math.log(n / df.get(w));
      r.set(w, v);
      norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [w, v] of r) r.set(w, v / norm);
    return r;
  });

  const G = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = i; j < n; j++) {
      const [a, b] = rows[i].size < rows[j].size ? [rows[i], rows[j]] : [rows[j], rows[i]];
      let s = 0;
      for (const [w, v] of a) {
        const u = b.get(w);
        if (u !== undefined) s += v * u;
      }
      G[i][j] = G[j][i] = s;
    }

  const eigs = topKEigSym(G, Math.min(maxRank, n - 1));
  const total = eigs.reduce((a, e) => a + Math.max(0, e.val), 0);
  let cum = 0,
    rank = eigs.length;
  for (let k = 0; k < eigs.length; k++) {
    cum += Math.max(0, eigs[k].val);
    if (cum / total >= energyTarget) {
      rank = k + 1;
      break;
    }
  }
  const X = Array.from({ length: n }, (_, i) =>
    eigs.slice(0, rank).map((e) => Math.sqrt(Math.max(0, e.val)) * e.vec[i]));

  return { X, rank, vocab: keep.size, distinct: df.size, energy: cum / total };
}
