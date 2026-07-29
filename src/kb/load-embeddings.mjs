// Load note-level embeddings for a subset of an Obsidian vault (Smart Connections).
//
// Returns { labels, tags, links, vectors, dim } where:
//   labels[i]  = note title (basename without .md)   — PRIVATE, never commit
//   tags[i]    = array of tags for note i
//   links[i]   = array of outlink target titles
//   vectors[i] = Float64Array of length `dim`
//
// Only source-level entries (`smart_sources:`) under includePrefix are kept,
// and only those that actually carry a vector for the configured model.

import fs from "node:fs";
import path from "node:path";
import { iterAjsonEntries } from "../lib/ajson.mjs";

export function loadConfig(root) {
  const cfgPath = path.join(root, "config.json");
  const examplePath = path.join(root, "config.example.json");
  const p = fs.existsSync(cfgPath) ? cfgPath : examplePath;
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

export function loadKbEmbeddings(cfg) {
  const { vaultPath, smartEnvDir, embeddingModel, embeddingDim, includePrefix, excludeTitles } =
    cfg.kb;
  const dir = path.join(vaultPath, smartEnvDir);
  const exclude = new Set(excludeTitles ?? []);

  const labels = [];
  const tags = [];
  const links = [];
  const vectors = [];
  const dates = [];

  const srcPrefix = "smart_sources:" + includePrefix;
  for (const [key, val] of iterAjsonEntries(dir)) {
    if (!key.startsWith(srcPrefix)) continue;
    const rel = key.slice("smart_sources:".length); // e.g. 20_Permanent/Foo.md
    const title = path.basename(rel).replace(/\.md$/, "");
    if (exclude.has(path.basename(rel)) || exclude.has(title)) continue;

    const emb = val?.embeddings?.[embeddingModel];
    if (!emb || !Array.isArray(emb.vec) || emb.vec.length !== embeddingDim) continue;

    labels.push(title);
    tags.push((val.metadata?.tags ?? []).map((t) => String(t).replace(/^#/, "")));
    links.push((val.outlinks ?? []).map((o) => o.target));
    vectors.push(Float64Array.from(emb.vec));
    dates.push((val.metadata?.["date created"] ?? "").slice(0, 10));
  }

  return { labels, tags, links, vectors, dates, dim: embeddingDim };
}
