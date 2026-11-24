# 06 · Do independent fits learn the same microscopic partition?

**No. At β=3.05 the three fits share a symmetry preference, not one partition viewed in different cubic frames.** All three emphasize `T1u`, the vector-like sector that changes sign under inversion. Yet even after choosing among all 48 cubic symmetries and renaming categories on TRAIN, the three pairs give different labels on **33.8–49.6%** of held equilibrium mass. Ordinary site/species indicators carry this symmetry too. This result does not identify a unique order parameter, a new phase, or three independent confirmations of one structure.

### How much agreement survives on independent exteriors?

The entries below use 8,187 TEST parents at β=3.05. NMI is normalized mutual information between two *hard categorical functions*, not the training InfoNCE bound. One spatial map and one category renaming per pair were locked on 8,185 TRAIN parents. Intervals are pointwise 95% across TEST parents, conditional on the frozen fits and choices.

| Seeds | NMI: fixed frame → proper rotations → full cube | Full cube: held NMI [95%] | Held disagreement [95%] | Disagreements on 210 states, unweighted | Full-cube shuffled NMI, mean |
|---|---|---|---|---:|---:|
| 6101 / 6102 | .092 → .305 → .337 | .337 [.326, .348] | 37.8% [36.8, 38.8] | 82 | .064 |
| 6101 / 6103 | .202 → .407 → .407 | .407 [.397, .417] | 33.8% [32.8, 34.8] | 92 | .067 |
| 6102 / 6103 | .304 → .376 → .376 | .376 [.368, .385] | 49.6% [48.5, 50.6] | 107 | .065 |

- **Exact identity:** exhaustive search found no symmetry and category bijection making even one nonconstant pair identical on all 210 states, at any positive β studied. The registered approximate criterion (all pairs' held NMI above .90) was also ruled out at each positive β. At β=0 all codes are constant; their perfect raw match carries no information. At β≈.678 the descriptive full-cube scores were .457, .246 and .527 in table order—not a second confirmatory endpoint.
- **Reflections:** only 6101/6102 gained at the cold endpoint from allowing inversion in addition to the 24 proper rotations: paired NMI gain .0314 [.0171, .0456]. The other two retained their proper rotation. Full-group action IDs were 42, 9 and 5; category renamings B→A were `2301`, `2301`, `3120` (image of labels `0123`). Cold full-group optima were unique to numerical tolerance; 1, 2 and 1 actions respectively were within .01 TRAIN NMI of the maximum. The optimization target was NMI, not minimum disagreement: one pair can therefore improve NMI without improving its best-matched raw accuracy.
- **What remains different:** conditional entropies H(A|B) were .890, .795 and .865 nats; H(B|A) were .924, .820 and .856. Chance-adjusted agreements (κ) were .495, .546 and .339. Held Pearson correlations of the separately frozen scalar contrasts, with sign chosen on TRAIN, were .371, .605 and .459. Similar memory scores in experiment 03 did not imply the same answers state by state.

### What the shared character does—and does not—explain

On the centered category-indicator span, `T1u` accounts for **45.5%, 41.2%, 51.5%** of power in seed order. Its excess over each seed's own shuffle mean was 20.9, 16.9 and 27.2 percentage points (registered requirement: ≥5 points and the same dominant sector in all three). The scalar contrasts also favor `T1u` (51.6%, 44.4%, 68.6%). These percentages are not probabilities of identical labels. A cubic sector cuts out a space of possible functions, not one function: the `T1u` projector has rank 48 on these 210 patterns.

The conventional packing span has **50%** `T1u` power. Projection onto ordinary site/species indicators accounts for 50.9–52.3% of the learned category spans and 68.9–74.6% of the frozen scalars; matched shuffled category spans give about 9.1–9.4%. Center vacancy alone accounts for only 0.07–1.44% of category-span power. A directional distinction between arms of the star is compatible with these results. A particular hidden vector, or an effect beyond ordinary packing, is not established by the symmetry test.

The exact 03 memory spectrum at β=3.05 has **34** nonconstant modes within .01 of the leading eigenvalue. The three category spans place 31.4–32.5% of their power in this band, versus 27.4–28.0% for matched shuffled partitions; a uniformly oriented direction would give 34/209 = 16.3%. The matched comparison matters. Narrowing the band to .005 or .001 leaves ranks 23 or 8 and reduces overlap to 15.6–19.2% or 5.4–6.3%. No individual near-degenerate eigenvector is claimed as the learned variable. At β=0 the flat band spans all 209 nonconstant directions and cannot select an operator.

### Gallery

[Held agreement and disagreement, every pair and β](held-agreement.png) · [Conditional entropies, scalar correlations and inversion gains](entropy-scalar-inversion.png) · [Cold cubic powers and near-leading subspace](symmetry-and-band.png) · [All-temperature symmetry decomposition](symmetry-temperature.png) · [Actual cold assignments before and after alignment](cold-maps.png). The last panel uses the twelve highest TRAIN-mass star states; [all 210 patterns and assignments](cold-example.csv) are supplied, not a curated set of successes.

### Scope, controls and reproducibility

[Prospective decision and input hashes](design.json) were published [before alignment](https://github.com/konradha/RSMI-NE/commit/b9b3ecf1651e0b332d052e0fe5e76236d1c19128). Inputs are the unchanged [03 exact exterior bank and trained maps](../experiment-03-cavity/), inventory (2 vacancies, 2 A, 3 B), all ten β, seeds 6101/6102/6103, final depth 10,257 updates (10,218 at β=3.05). One eligible exterior per independent parent, as in 03. Held parent counts in increasing β order: 8,192 at each of the first eight, then 8,190 and 8,187. These are the original 03 holdouts, **not newly acquired configurations**.

On the raw TRAIN marginal, select the spatial map by maximum NMI, then the category bijection by maximum agreement. No transform or sign is chosen on TEST and no group augmentation is applied to TEST. Each independent TEST parent contributes its exact 210-component conditional probability vector, not 210 independent observations. Score errors propagate the covariance across these parent vectors; contrasts retain pairing. The direct per-parent NMI calculation and covariance calculation agree to <2×10⁻¹⁶ in interval width. The finite-difference check of all nine held-score gradients had maximum directional error <2×10⁻¹¹. These intervals do not quantify optimization-seed uncertainty.

Each of 64 independent null draws per seed shuffles the assignments to the same 210 states, preserving category counts, and receives the **same** TRAIN search and TEST scoring. This does not preserve equilibrium-weighted entropy exactly: cold real marginal entropies are 1.350–1.385 nats, corresponding shuffled means 1.275–1.378. The null-draw range is distinct from parent sampling uncertainty. Only the symmetry decomposition uses the orbit-averaged TRAIN metric, so the group actions preserve the inner product. The centered four-label projector trace is invariant to category renamings. All group products, inverses, preserved star edges, state actions and character projectors were checked; a planted proper transform was recovered by both searches, an improper one only by the full group. See [geometry and planted checks](geometry-audit.json) and [all actions, in order](group-actions.npz).

**Source data:** [held pair scores, choices and intervals](seed-pairs.csv) · [per-temperature decisions](family-by-temperature.csv) · [matched pair nulls](random-pairs.csv) · [symmetry fractions](symmetry-power.csv) and [nulls](random-power.csv) · [subspace overlaps](reference-overlap.csv) and [nulls](random-overlap.csv) · [coverage and band ranks](support.csv) · [TRAIN/TEST state masses](state-masses.npz) · [decision](decision.json) and [execution receipt](receipt.json).

To reproduce from the repository root, extract [the 03 bank](../experiment-03-cavity/exact-exteriors-and-replicas.zip), use a fresh output directory containing a copy of `design.json`, then run `OPENBLAS_NUM_THREADS=1 python examples/stock_glass_cubic_families.py --banks <directory-containing-prepared.json> --out <output>`; see [numerics](../../examples/stock_glass_cubic_families.py), [group conventions](../../examples/stock_glass_cubic.py) and [gallery renderer](../../examples/stock_glass_cubic_figures.py).

Physical and methodological context: [Nishikawa & Hukushima, PRL 125, 065501](https://doi.org/10.1103/PhysRevLett.125.065501); [Gökmen et al., PRL 127, 240603](https://doi.org/10.1103/PhysRevLett.127.240603) and [PRE 104, 064106](https://doi.org/10.1103/PhysRevE.104.064106). Their operator interpretations motivate asking about symmetry; they are not evidence for a glass order parameter here.

**Cost:** $0 for 06; $17.99322425 of the prior authorization remains. No new fits, simulations, or experiment 07 work.
