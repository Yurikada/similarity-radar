// Parser for Smart Connections `.ajson` (append-only JSON) files.
//
// Format: each file is a sequence of `"<key>": {<object>},` fragments — NOT a
// single JSON object. The same key may appear multiple times across the file
// (append-only log); the LAST occurrence is the current value. Wrapping the
// whole content in `{ ... }` and running JSON.parse gives exactly last-write-wins
// semantics, which is what we want.

import fs from "node:fs";
import path from "node:path";

/** Parse one .ajson file into a plain object (last-write-wins on duplicate keys). */
export function parseAjsonFile(filePath) {
  const raw = fs.readFileSync(filePath, "utf8").trim().replace(/,\s*$/, "");
  if (!raw) return {};
  try {
    return JSON.parse("{" + raw + "}");
  } catch {
    return {}; // skip malformed files rather than crashing the whole run
  }
}

/** Iterate every `.ajson` entry across a directory, yielding [key, value]. */
export function* iterAjsonEntries(dir) {
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".ajson"));
  for (const f of files) {
    const obj = parseAjsonFile(path.join(dir, f));
    for (const kv of Object.entries(obj)) yield kv;
  }
}
