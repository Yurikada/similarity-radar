# Scheme & design decisions

A running log of *why* each stage is built the way it is. Each stage lists the
implementation, the design decision under review, and what was observed. This is
the portfolio-facing narrative — it should read as "I understand the trade-offs",
not "I called a library".

---

## A0 — Feature matrix

**Built:** parse Smart Connections `.ajson` (append-only JSON, last-write-wins),
keep source-level entries under `20_Permanent/` that carry a 384-d vector for
`TaylorAI/bge-micro-v2`, exclude the template. Output: matrix `X ∈ ℝ^{86×384}`.

**Observed:** n=86, dim=384. **All vectors are L2-normalized (‖v‖=1).**

**Consequence (matters downstream):** on the unit sphere,
`‖a−b‖² = 2 − 2·cos(a,b)`. So Euclidean distance is a monotone function of cosine
similarity — cosine and (squared) Euclidean induce the *same* ordering of
neighbors. Distance choice is therefore not about ranking here; it is about what
the projection's stress function optimizes.

**Open:** population choice. We use Permanent-only (86) for interpretability;
Index/structure notes are excluded because they act as artificial hub bridges.

---

## A1 — Distance & anisotropy

**Built:** pairwise cosine similarity distribution; empirical neighbor-ranking
invariance test (cosine vs Euclidean); anisotropy read via centroid norm; and a
centering experiment.

**Observed:**
- Pairwise cosine similarities compress into **[0.783, 0.994]**, median 0.888
  (3655 pairs). Matches the vault's earlier finding.
- **Neighbor ranking is identical under cosine and Euclidean for 86/86 notes.**
  Confirms: on the unit sphere `d_euc = sqrt(2·d_cos)` is a strictly monotone
  transform, so ranking is preserved *for all angles*, not just small ones. The
  distance choice cannot change neighbor order; it only changes the numeric
  values a metric-fitting method (MDS) tries to reproduce.
- **Anisotropy: ‖mean unit-vector‖ = 0.942.** All embeddings sit in a narrow
  cone; ~89% (0.942²) of each vector's energy is in a shared, content-free
  direction (the "cone effect").
- **Centering experiment:** subtract the mean direction and renormalize →
  similarity spread jumps from width 0.211 (median 0.888) to width **1.442**
  (median −0.048, mean ≈ 0), while truly-related notes stay high (max 0.954).

**Decision (resolved):** **center the vectors before projecting.** The common
direction is one wasted coordinate carrying no information; leaving it in lets a
content-free axis dominate the layout. This is the same conclusion the vault
reached via "rank-normalize", arrived at here from the cone-removal angle.

**Consequence for A2:** we will project the *centered* matrix. PCA on centered
data = standard PCA (it centers anyway); MDS/t-SNE will be fed centered-cosine
distances.

---

## A2 — Projection comparison

**Built:** PCA (dual/Gram, deterministic), metric MDS (SMACOF, seeded), t-SNE
(seeded), all on the centered matrix. Procrustes rotation/disparity between two
seeds; Pearson+Spearman between high-dim and 2D distances over all 3655 pairs.

**Observed:**

| method | rotation (2 seeds) | disparity (structure Δ) | corr(hiD, 2D) |
|---|---|---|---|
| PCA | 0° (deterministic) | ~0 | 0.69 |
| metric MDS | 171° (arbitrary) | 0.81 | **0.73** (best) |
| t-SNE | 61° (arbitrary) | 0.96 (worst) | **0.51** (worst) |

- **Rotation is not a "nonlinear/t-SNE" problem — it is an init problem.** Any
  distance-preserving method has a rotation-free objective (distances are
  rotation-invariant), so with random init the orientation is arbitrary. MDS
  rotates just as freely as t-SNE. PCA is stable only because it is *spectral*
  (canonical axes from variance-max), not because it is "linear". This refines
  the vault's claim that rotation instability is t-SNE/UMAP-specific.
- **MDS structure also varies (disparity 0.81), and it is genuine multi-minima,
  not under-convergence:** all seeds reach nearly identical stress (~0.32) yet
  differ structurally. 2D cannot hold 384-dim distances (32% residual stress),
  so the stress surface is rugged with many near-equal minima.
- **t-SNE has the *worst* global fidelity (0.51).** Its crisp clusters are a
  look; inter-cluster distances are the least trustworthy. MDS wins global
  fidelity by construction (it minimizes distance stress directly); PCA close.

**Consequence:** for a *distance-faithful landscape* ("distance = similarity,
axes meaningless"), the method must be MDS-family, and rotation must be tamed
separately (A3) — matching the behavior inferred from a commercial
technology-landscape product. t-SNE is
disqualified for that promise despite its visual appeal.

---

## A3 — Rotation-invariant readout

**Built:** radius = percentile rank of high-D distinctiveness (‖centered‖ =
distance from centroid); angle = PCA projection. Stability comparison of the
radial coordinate across seeds/methods.

**Observed:**
- distinctiveness radius: corr = 1.000 across any projection/seed (it is computed
  from fixed high-D data, independent of the wobbly 2D layout).
- a projection's own 2D-radius (MDS seed1 vs seed2): corr = 0.960.
- 2D-radius vs true distinctiveness: corr = 0.958.

**Insight:** rotation and multi-minima scramble mainly the *angle*; the *radius*
(centrality ordering) is robust (MDS disparity 0.81 yet radius corr 0.96). So
"put meaning in the radius, throw away the angle" is doubly justified — and
taking the radius from high-D distinctiveness makes it exactly reproducible.
Percentile-rank mapping fills the rings evenly despite the compressed raw
magnitudes (0.22–0.43). Periphery = methodological/meta outliers; center =
core themes. This matches the percentile-ring design seen in commercial
landscape tools.

## A4 — KDE density surface

**Built:** Gaussian KDE (Scott bandwidth) on the PCA projection; per-note density;
tag breakdown of dense vs sparse notes; and the radial-density artifact check.

**Observed:**
- HOT (densest) tags: self-management/justice/value — a tight ethics/meaning cluster.
- COLD (sparsest) tags: ai-human/writing/trading — note that ai-human is the
  *largest* tag (26) yet lands in sparse regions: density measures concentration,
  not count. Diverse themes spread out.
- corr(PCA-density, radar-radius) = −0.25 (mainstream ≈ slightly denser).
- **Artifact:** on the percentile-radar, area-density by radius is 135 → 15 from
  center to edge purely because uniform radius ⇒ ~1/r. So KDE must run on the
  distance-preserving projection, not on the percentile-radar.

**Two corrections:** (1) "hot = most-written" is wrong; hot = most *concentrated*.
(2) "whitespace = unexplored opportunity" is domain-specific: here sparse regions
are heavily-written-but-diverse themes, and gaps between unrelated clusters are
meaningless to fill. The "whitespace as opportunity" interpretation used in
commercial tools can hold for patent
landscapes (density = activity), not automatically for a personal KB.

## A5 — Clusters + human disagreement

**Built:** k-means (k=9 = distinct primary tags) in high-D (384d) and in 2D (PCA);
Adjusted Rand Index vs each other and vs human tags; bridge extraction.

**Observed:**
- ARI(high-D, human) = 0.448 vs ARI(2D, human) = 0.247. Clustering *after*
  projecting to 2D nearly halves agreement with the human labels → cluster in
  high-D, use the projection only for display. (ARI(high-D, 2D) = 0.493.)
- 24 "bridge" notes where the note's own tag ≠ its high-D cluster's dominant tag:
  trading×ai-human, value×causality, engineering×self-management, etc.

**Insight:** the algorithm-vs-human *disagreement* is the payload, not the
agreement — it surfaces cross-theme bridges. Direct evidence for the vault's own
claims ("value lives in the link/embedding discrepancy"; "valuable bridges cross
Index clusters"). Agreement is redundant with the Index; disagreement is the
creative-bridge catalog.

## A6 — Temporal trend via Procrustes

**Built:** split by date (older 43 vs all 86); project each snapshot independently
with PCA and with MDS; compare common notes' positions raw vs Procrustes-aligned;
report rotation angle.

**Observed:**
| method | raw disp | aligned | frame× | rotation |
|---|---|---|---|---|
| PCA | 0.016 | 0.013 | 1.2× | 3.7° |
| MDS | 0.223 | 0.066 | 3.4× | 106.7° |

**Insight:** independent projections carry an arbitrary frame (rotation +
reflection + scale + translation); most of the apparent movement is that frame,
and Procrustes removes exactly it so the residual is the real change. **PCA's
frame shifts only 3.7° across the two populations — small (stable variance
structure) but nonzero**, confirming that PCA axes depend on the population, so
Procrustes is still needed for exactness. MDS's frame is arbitrary (106.7°),
making it unfit for temporal comparison. Conclusion for trend lines: PCA (near
frame-stable) + Procrustes. (Reflection/scale note: same seed with correlated
init masks MDS rotation — use independent inits per snapshot.)

## B0 — JP equities: universe & features

**Built:** fetched fundamentals for 68 Nikkei-225 large caps via yahoo-finance2;
9 features (value: per/pbr/divYield; quality: roe/roa/opMargin; growth:
revGrowth/earnGrowth; size: logMktCap). Winsorize [2,98%] → median-impute →
z-score. Coverage high (per 62/68, growth ~56-63/68, rest 68/68). 10 sectors.

**Transfer checkpoint:** the KB's anisotropy (cone) maps to feature CORRELATION
here — pbr×roa 0.85, pbr×roe 0.74, roe×roa 0.71, per×pbr 0.67. Plain Euclidean
on z-scores over-weights the correlated quality/value bundle. Open decision for
B1: decorrelate/whiten (Mahalanobis / PCA-whiten) vs plain Euclidean.

## B1 — Whitening + projection (transfer test)

**Built:** PCA-whiten the z-scored features (remove correlation; Euclidean on
whitened = Mahalanobis on original), then compare PCA/MDS/t-SNE as in A2.

**Observed:** max |feature corr| 0.85 → 0.00 after whitening.
| method | rotation (2 seeds) | corr(hiD, 2D) |
|---|---|---|
| PCA | 56.8° | 0.574 |
| MDS | 22.8° | **0.790** |
| t-SNE | 44.7° | 0.543 |

**Transfers:** MDS best global fidelity, t-SNE worst — same as KB.
**Breaks (key finding):** PCA is NO LONGER rotation-stable (56.8° vs KB's 0°).
Whitening makes the covariance identity (isotropic), so PCA's max-variance axes
are degenerate/undefined → arbitrary orientation. PCA was stable in the KB only
because we centered (not whitened; n≪d made whitening ill-posed) and the
embeddings kept an anisotropic variance structure. Whitening helps the *distance*
but destroys the variance structure PCA relied on for a canonical frame. →
Stock radar must use an MDS base (Mahalanobis distance) with rings for rotation,
not "PCA for a stable angle". A clean example of a mechanism transferring while
its interaction with another choice does not.

## B2–B6 — Stock radar + screening

**Built:** JP equity radar — MDS base on Mahalanobis (whitened) distance,
percentile rings, adaptive-KDE contour terrain, sector color, 37 bridge stocks
(factor cluster ≠ sector). Same generator pattern as the KB radar (transferred).

**B6 screening read:** periphery = extreme factor profiles (SoftBank Group
ROE33%/PER6.5; Nippon Steel PER182/ROE1% cyclical trough; Advantest PBR27/ROE58%
extreme growth); center = ordinary (KDDI, ENEOS). 37 bridges = stocks whose
factor profile resembles a different sector → pair/diversification candidates.

## B7 — Transfer summary

**Same mechanism (transferred):** kill redundant directions so distance reflects
only independent variation; MDS is the distance-faithful base and t-SNE is worst
at global distance; rotation is neutralized by a rotation-invariant radius (rings).

**What changed:** KB centered (n≪d, whitening ill-posed); stocks whitened (n>d).
And the sharp one — **whitening isotropizes the covariance, so PCA loses its
canonical axes and rotates (56.8° vs KB 0°)**; the KB's "PCA for a stable angle"
does not survive whitening, so the stock radar must use MDS. A mechanism can
transfer while its interaction with another design choice does not.

---

# Part C — Which objective function should the map minimize?

Part B left three open doubts about the stock radar: the features were numeric
state (PER/PBR) rather than textual identity; price movement and volatility sat in
the same layer as everything else; and the metric-MDS objective had never been
varied. Part C takes the third.

## C-design — Rejecting the obvious parameterization

**Proposed first:** a one-parameter family `Σ d^{-p} (d̂ − d)²`, sweeping p from
absolute error (p=0) through Sammon (p=1) to relative error (p=2).

**Rejected, for five reasons.** (1) The units are length^(2−p), so loss values are
not comparable across p and the family carries no internal yardstick. (2) The
weight diverges as d→0, forcing an ε floor that can move the map more than p does.
(3) The *effect size* of p is governed by the coefficient of variation of the
distance distribution, so "p=1" denotes a different operation in each domain —
disqualifying for a study whose subject is transplanting a mechanism across
domains. (4) It overloads `w_ij`, whose principled meaning is data confidence.
(5) It does not span the design space: neither PCA (inner-product fitting) nor
t-SNE (neighbourhood-probability matching) is a member, so a single p implies the
space is 1-D when the large jumps measured in A2 are between families.

**Decision:** decompose as `σ = Σ w ρ( f(d̂) − f(d) )` — weight, residual shape,
distance transform — and compare **four discrete points** whose definitions can be
stated in words, rather than sweeping a continuum where the prettiest map could be
picked after the fact:

| point | preserves | objective |
|---|---|---|
| raw stress | absolute distance | `Σ (d̂ − d)²` |
| log-distance | relative distance | `Σ (log d̂ − log d)²` |
| kNN-weighted | neighbourhood set | `Σ w (d̂ − d)²`, w = 1 inside kNN else γ |
| non-metric | rank order | `min` over monotone δ of `Σ(d̂−δ)² / Σd̂²` |

Note on the family: p=2 is the relative-error end, whose exact scale-invariant
form is log-distance fitting; Sammon's 1/d sits between absolute and relative.
Kruskal's stress-1 is a global normalizer, not a member of the weight family.

## C0 — Calibration and pre-registration

**Built:** `classical-mds.mjs` (shared warm start), `mds-gd.mjs` (generic gradient
descent with Armijo backtracking — a fixed learning rate would need per-objective
tuning, reintroducing the hidden knob this stage exists to remove), and a
calibration that solves raw stress by both the existing SMACOF majorization and
the new optimizer. Thresholds were fixed before running: Procrustes disparity
<0.01, |Δstress-1| <0.005, |Δtrust| <0.01, |Δcont| <0.01.

**Observed:** all four conditions pass. Warm start: disparity 1.12e-3 (stocks) /
2.73e-3 (kb), rotation 0.006°/0.011°. Random init: 2.97e-23 / 5.08e-13.
Differences seen in C1 are therefore attributable to the objective, not the
optimizer.

**Pre-registered:** CV = sd(d)/mean(d) of the distance matrix each domain actually
feeds to MDS — stocks 0.3969, kb 0.1713 — with the hypothesis that the effect size
of choosing an objective grows with CV. Frozen to
`out/objectives/preregistration.json` before any alternative objective existed.

**Found on the way:** centering does not change pairwise Euclidean distances at
all, so A1's "similarity width 0.21→1.44 after centering" says nothing about the
matrix MDS consumes. And on stocks the classical-MDS warm start is **degenerate** —
Gram eigenvalues 144.98 vs 144.23, because whitening isotropizes the covariance
(the B1 finding, seen at its root). The init is deterministic but not canonical:
on stocks, read shape, never angle.

**Fixed here:** the in-repo 2-D Procrustes reused the unreflected optimal angle
when scoring the reflected branch, overstating the residual for layouts that
differ by a reflection (synthetic check: 1.34 → 2.8e-32). A2's recorded figures
move from `MDS 171° / disparity 0.81` to `159° / 0.62`, and the difference turns
out to be rotation **plus reflection**. Every A2 conclusion survives; A3 is
strengthened, since reflection is one more transform that destroys angle while
leaving radius untouched. The coordinate ambiguity is O(2), not SO(2).

## C1 — The four-point comparison

**Built:** one distance matrix, one warm start, one optimizer, Procrustes
alignment before any shape comparison, and comparison only through external
metrics (stress-1, corr, trustworthiness, continuity) — never through the
objectives' own values, which are not commensurable.

**Result: 3 valid + 1 disqualified, reported together.**

**kNN-weighted is disqualified, and stays in the report.** Deleting the point
that failed and publishing the three that worked would be the same selection error
the pre-registration exists to prevent. It fails twice:

1. *Unstable to its own knobs.* The frozen k×γ grid moves the map by up to 0.662
   (stocks) / 0.843 (kb) — **more than the between-objective effect** it was meant
   to participate in (0.618 / 0.695). The ε criticism aimed at `d^{-p}` applies
   here unchanged; "rank-based, therefore population-invariant" secured the
   definition of the weight, not the stability of the map.
2. **Loss of geometric scaffolding.** The objective built to protect
   neighbourhoods ranks *last* on neighbourhood preservation: trust 0.679/0.684
   and continuity 0.814/0.755, against ≥0.800 and ≥0.848 for the other three.
   Down-weighting far pairs to γ removes the struts holding unrelated clusters
   apart; they drift together and manufacture false neighbours in 2-D. Locality
   cannot be imposed by weighting alone — a neighbourhood is defined partly by the
   distant pairs it is *not* near.

**The pre-registered CV hypothesis is not supported.** effect(kb)=0.695 >
effect(stocks)=0.618 while CV(kb) < CV(stocks) — the wrong direction. This does
not establish that the four-point family is CV-independent: that was a design
intent, not a measured property; the observed effect is dominated by the
disqualified point; and n=2 domains cannot separate the readings. The `d^{-p}`
portability claim itself stays untested — the sweep was deliberately not run,
since re-confirming that a pathological parameterization behaves pathologically
buys little once objectives that avoid the pathology exist.

**The main positive finding: the A3 readout is near-invariant to the objective.**
Shapes differ by disparity 0.44–0.62, yet radius percentile correlates 0.988/0.943
(log) and 0.998/0.980 (non-metric) against raw stress. Percentile ranking absorbs
the non-linear radial stretching that separates the objectives, so the choice
changes the picture without changing what A3 reads off it. A3's justification now
rests on invariance to the coordinate frame *and* to the objective.

*Scope of that claim:* holds for the three valid objectives (the disqualified kNN
is weakest at 0.917/0.834); observed on n=2 domains, so empirical robustness, not
a theorem; both populations are broadly unimodal about a centroid, and under
sharply separated multi-modal structure the objective decides which inter-cluster
gaps are crushed, which could move radius-from-global-centroid — untested here;
and the invariance belongs to the readout, not to the coordinates.

**Also observed.** Raw stress and non-metric produce nearly the same map
(disparity 0.130/0.139, the smallest pair in both domains), yet non-metric scores
*higher* on global correlation (0.895/0.860 vs 0.843/0.782) with equal or better
trust/continuity — metric magnitudes are close to redundant given the ranks. The
non-metric fit is not degenerate (δ spans 0.109–8.010 with 121 pooled blocks on
stocks, 0.018–0.692 with 96 on kb) though the pooling is coarse. Log-distance has
the worst optimization landscape: most iterations (4615/2071) and the largest
random-init sensitivity (disparity 1.631/0.973), against non-metric's 0.657/0.149.

**Predictions on record:** the extreme-profile stocks do *not* move inward under
log-distance (SoftBank 98.6→97.9, Nippon Steel 97.9→98.6, Advantest 91.0→93.1 —
inconsistent in sign, ±2 percentile points), for the same reason the A3 readout is
invariant. kNN and non-metric are indeed different maps (0.618/0.498 against
0.454/0.384 median for other pairs).
