# similarity-radar

A landscape / "radar" visualization built from **high-dimensional similarity**.

Given a set of items described by high-dimensional vectors — documents via text
embeddings, or stocks via financial factors — this projects them to 2D and then
reads structure the *right* way: through **rotation-invariant coordinates** and a
**density surface**, not through meaningless Cartesian axes.

It is a hands-on study of the algorithm family behind similarity-landscape tools:
what each dimensionality-reduction method preserves, why the map's rotation is
arbitrary, and how to build a readout that survives that arbitrariness.

## Why this exists

Most 2D "maps" of embeddings are read naively — people point at the x-axis as if
it meant something, or compare two re-runs as if the layout were stable. It isn't.
This project makes the failure modes concrete and then designs around them:

- **Axes carry no meaning; only distances (and radius) do.** Rotation and
  reflection are isometries, so any single snapshot is defined only up to a
  rotation. We encode meaning only into rotation-invariant quantities.
- **Global vs local is a choice.** PCA/MDS preserve global distance (stable,
  faithful large-scale structure); t-SNE/UMAP preserve local neighborhoods
  (crisp clusters, but inter-cluster distance and rotation are not trustworthy).
- **Density is a separate layer.** A kernel-density surface over the 2D points
  reveals crowded regions (competition) and sparse regions (whitespace).

## The pipeline (6 stages)

```
items → feature vectors → similarity/distance
      → 2D projection (PCA / MDS / t-SNE, compared)
      → rotation-invariant readout (radius = percentile from centroid)
      → density surface (2D KDE: hotspots & whitespace)
      → clusters + centroids (labelled)
      → temporal trend (Procrustes-aligned snapshots)
```

Two domains exercise the same pipeline:

- **A. Knowledge base** — 384-d text embeddings of one author's Zettelkasten notes.
  (Familiar data: learn the algorithm where you know the ground truth.)
- **B. Japanese equities** — engineered factor vectors (value / quality / growth /
  momentum / size). (Transfer test: the same pipeline on a different domain.)

## Status

The core case study is complete. See [`docs/scheme.md`](docs/scheme.md) for the
design decisions, pre-registered hypotheses, negative results, and open
questions.

- [x] Part A — knowledge-base pipeline: distance audit, projection comparison,
  radial readout, KDE, clustering, and aligned temporal snapshots
- [x] Part B — transfer test on Japanese equities
- [x] Part C — objective-function comparison with a committed pre-registration
- [x] Part D — separation of stable identity from changing market state,
  composed radar, time slider, and forward-return falsification
- [x] Part E — text-derived identity pilot and peer-group reliability guard
- [ ] Next — repeat the text layer with Japanese EDINET business descriptions
  before deciding whether to split the remaining broad cluster

## Privacy

The knowledge-base domain runs on **private note content**. This repository ships
**code only**. Note text, embeddings, and any computed coordinates live under
`data/` and `out/`, which are git-ignored. To run the KB pipeline on your own
vault, copy `config.example.json` to `config.json` (also git-ignored) and point it
at your vault. The equities domain uses public market data and is fully shareable.

The public repository starts from a clean source snapshot. Private development
context and generated local artifacts are intentionally not included in its
history.

## Run

```bash
# KB domain
cp config.example.json config.json   # then edit vaultPath
npm run kb:matrix                     # Stage A0
```

Requires Node.js ≥ 20. No build step; plain ES modules.

## License

MIT — see [LICENSE](LICENSE). Not affiliated with or derived from any commercial
product; an independent study of standard dimensionality-reduction methods.
