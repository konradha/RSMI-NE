# Frozen RSMI transfer from β=3.05 to β=3.20 and 3.40

**The β=3.05 observables transfer without retraining to both colder endpoint ensembles. The primary test does not establish that the learned partition has a practically meaningful advantage over arbitrary partitions.** The distinction matters: at β=3.40 even the *entire seven-site microstate*, before compression, has 97.6% conditional replica agreement. A two-bit code can therefore be highly predictable without being uniquely informative.

### Primary 500M results: one eligible star per parent

`T` is the fraction of scalar variance explained by the frozen exterior. `I` is mutual information in bits for this frozen code. `D` is learned minus mean-null normalized categorical agreement; a practical gain was set to +0.01 before scoring. Brackets are pointwise 95% whole-parent bootstrap intervals. Named seeds are distinct functions, not three replicate measurements of one function.

| β | Frozen seed | Scalar `T` [95%] | `I(B;H)` bits [95%] | Learned − random `D` [95%] |
|---|---:|---:|---:|---:|
| 3.20 | 6101 | .9857 [.9692, .9985] | 1.916 [1.813, 1.961] | +.0091 [−.0035, +.0250] |
| 3.20 | 6102 | .9785 [.9524, .9990] | 1.941 [1.857, 1.973] | +.0071 [−.0052, +.0228] |
| 3.20 | 6103 | .9666 [.9327, .9927] | 1.896 [1.790, 1.944] | −.0014 [−.0107, +.0099] |
| 3.40 | 6101 | .9843 [.9553, 1.0000] | 1.896 [1.778, 1.955] | +.0072 [−.0051, +.0233] |
| 3.40 | 6102 | .9918 [.9746, 1.0000] | 1.964 [1.883, 1.989] | +.0144 [−.00001, +.0325] |
| 3.40 | 6103 | .9837 [.9634, .9990] | 1.952 [1.871, 1.980] | +.0030 [−.0080, +.0170] |

1. **Retained static memory: yes, by the registered criterion.** All six lower bounds exceed .90. Total scalar variance was .950–1.017; 87–106 effective parents contributed to it. The frozen site-linear baseline and center vacancy each had `T > .9997` at both targets, so this is **not** superiority over ordinary observables. Only 1–4 effective parents carried the within-cavity variance for some contrasts; do not rank these nearly saturated baselines by their last decimals.
2. **Learned specificity above +.01: unresolved.** Whole-state collision was .9656 [.9410, .9868] at 3.20 and .9764 [.9569, .9921] at 3.40. The primary learned label-disagreement deficit came from only 5.0–7.1 and 2.0–4.0 effective parents respectively (minimum required: 10). None of the six primary comparisons supports the margin. A nominal interval just below it for one seed cannot exclude a population effect under the support rule.
3. **More specificity at 3.40 than 3.20: not established.** `D(3.40) − D(3.20)` in seed order: −.0019 [−.0225, +.0186], +.0073 [−.0153, +.0301], +.0043 [−.0116, +.0213]. The common null assignments are paired across temperatures; the physical parents are not.

**A small effect in the registered sensitivity analysis.** Using all 780 eligible stars at β=3.20 and 764 at β=3.40, still weighting and resampling parents rather than stars, raised the effective support of the learned deficit to 20–23 and 14–16 parents. `D` was +.0062 to +.0064 at 3.20 and +.0063 to +.0081 at 3.40. All six pointwise intervals were above zero, but **all six straddled +.01**. This supports a small reproducible preference for these partitions *in this analysis*, not a large effect, stronger order upon cooling, or a replacement primary endpoint.

### What the bits and forecasts actually say

Across the six primary target fits, marginal label entropy was 1.920–1.986 bits (two-bit maximum); mean conditional entropy was only .016–.055 bits. In seed order, the corresponding null marginal entropies were 1.820/1.967/1.959 bits at 3.20 and 1.813/1.962/1.948 at 3.40. Nulls have exactly matching **state counts**, not matching entropy. The target delete-one-parent jackknife raised estimated `I` by .0166–.0170 bit. At the source, size-128 subsets underestimated full-source `I` by about .016–.017 bit on average, with appreciable spread. No monotonic temperature trend in `I` is claimed.

**A genuine forecast, not enumeration as a predictor:** the frozen β=3.05 directional heads used only exterior species and exterior-only coordination. Against the frozen orientation-blind packing head, they reduced expected log loss by **1.26–1.36 bits/star at 3.20** and **1.14–1.28 at 3.40**; all six paired lower bounds were positive. They picked the exact modal label on 77–85% and 75–85% of exterior samples, respectively. This is transfer of contact-shell predictive skill, not newly learned target correlations.

**Observed interiors provide a separate check.** At 500M the observed center-vacancy fractions were .50394 and .60938 vs .50389 and .60937 from the conditional model. For the `00` label of seed 6101 the observed/model fractions were .3465/.3438 and .3750/.3750. Agreement was not perfect across species: at 3.40 the observed center-A fraction was **.0156 below** its conditional prediction [−.0313, −.0039], with a compensating center-B excess. This is the same center statistic for all three seeds, descriptive and not multiplicity-adjusted. The per-category signed differences, including all seeds and both ages, are in [estimates.csv](estimates.csv). These checks do not certify equilibrium.

### Preparation age and coverage

All **128 independent preparation parents per target temperature** were present at both ages. At 500M one β=3.20 parent (chain 224) had no sector-223 match on the prespecified grid: **127/128** contributed to the primary metric there and **128/128** at 3.40. At 300M all 256 had a match. Across ages the selected location changed for 127/127 complete pairs at 3.20 and 126/128 at 3.40. These are paired *parent* changes, not measurements of the same boundary relaxing.

All six paired age intervals for primary `T` and `D` included zero. Mean native energy/site at 3.20 went .059623 → .059707 (paired change +.000084 [−.000543, +.000722]); at 3.40 it went .050240 → .049693 (−.000547 [−.001063, .000000]). The latter is a possible downward age drift. **The verdict concerns the sampled 500M exteriors under an exact conditional Gibbs model; neither equilibrium of their distribution nor physical decorrelation was demonstrated.** Neither local-only dataset was opened. New target training, new simulations, and Modal spend: zero.

### Gallery, data, and reproduction

[Frozen scalar and categorical transfer](cold-transfer.png) · [learned-minus-random and the temperature contrast](cold-specificity.png) · [MI, entropy, variance and whole-state saturation](cold-information.png) · [paired ages and anchor sensitivity](cold-sensitivity.png) · [frozen forecasts and observed label fractions](cold-forecasts.png).

[Native snapshot and checkpoint receipt](bank-receipt.json) · [source replay](source-replay.csv) · [all estimates](estimates.csv) · [paired contrasts](contrasts.csv) · [per-map null results](partitions.csv) · [variance and deficit support](support.csv) · [paired native energies](age-energy.csv) · [matched source subsets](source-n128.csv) · [jackknife sensitivity](entropy-sensitivity.csv) · [registered decisions](decisions.csv). Four `target-{10,11}-{300,500}.npz` banks retain every target ID, original energy, shifts, chosen stars, observed interior and sufficient exterior data; `frozen.npz` contains the maps, coefficients and matched null bank.

Both native snapshots had zero recomputed energy mismatches, correct low-bit decoding, native mask parity, and fixed composition (2,000 vacancies, 2,400 A, 3,600 B) on every L20 parent. All three checkpoint logits and all 630 state-to-category assignments replayed without a mismatch. Source scalar `T` was .97779/.97571/.97313 and source specificity .01133/.00962/.00771; original 03/05/07 score replays agreed to <7×10⁻¹³. From the repository root, with `BANK` set to the extracted [03 source bank](../experiment-03-cavity/exact-exteriors-and-replicas.zip), reproduce the numerical tables and plots from the published target banks:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 examples/stock_glass_cold_transfer.py --source-banks "$BANK"
python3 examples/stock_glass_cold_figures.py
```

To rebuild bank extraction in a new directory, pass `--prep "$PREP" --out "$OUT"` to the first command and `--out "$OUT"` to the second, where `PREP` contains the original, hash-pinned `plan.json` and `batch-00000000` snapshots.

## What transferred

The three final stock 03 encoders (seeds 6101, 6102, 6103; 10,218 updates) each supply **one** four-category map on the same 210 labeled seven-site stars, and one scalar coefficient per category. Those exact maps and coefficients, the center-vacancy observable, the source TRAIN site-linear comparator, and the 07 exterior forecasts stay fixed at every temperature and age. Labels from different seeds need not have the same meaning. Source values use the original β=3.05 TEST parents; the source is a replay, not a newly blind temperature. [Prospective lock](design.json) ([commit](https://github.com/konradha/RSMI-NE/commit/228b999ca7aed4446881f113117a7a38a897e6eb)).

The primary target is a nonlocally prepared **500-million-sweep endpoint per chain**, not a local trajectory. The 300M endpoint on that very same chain is used only to diagnose age sensitivity. The geometry and conditional energy are the original 03 ones: a center and six ordered nearest neighbors with exactly two vacancies, two A and three B. On a species-blind, seeded periodic grid of 216 disjoint candidate stars the first inventory match is selected. The grid and order stay fixed across ages, but the first eligible star can change. Secondary results use *all* eligible stars and give each parent equal total weight; conditional entropies and collisions are averaged over its stars **after** computing them at each exterior. Neither stars nor ages are additional independent parents.

For a sampled exterior B, the 210-state conditional Gibbs law at the named β gives `q_B(h)` for each frozen label. The scalar memory ratio is `T = 1 − mean_B Var(f|B)/Var(f)`. Categorical agreement compares two hypothetical independent conditional replicas: `A = 1 − (1−mean_B sum(q_B²))/(1−sum(mean_B q_B)²)`. Specificity is `D = A − mean(A_null)` over the **same 64 deterministic state partitions per seed** used in 05. They preserve unweighted category sizes, not equilibrium-weighted entropy. The shared null bank is never resampled as if it contained independent physical samples. `I(B;H) = H(mean_B q_B) − mean_B H(q_B)` in **bits**; this is not InfoNCE, a kinetic relaxation, or an estimate of a point-to-set length. Near-complete pinning can make even arbitrary partitions highly predictable.

Intervals are pointwise percentile bootstrap intervals across whole independent parents, conditional on the frozen maps. The temperatures are resampled independently; within each temperature the same draws are used for both ages, all seeds, all nulls and both anchor rules. Age contrasts use jointly eligible parent identities. Source has 2,048 bootstrap draws; targets 4,096. The finite-size diagnostics are delete-one-parent jackknife MI on the targets and 1,024 size-128 subsamples without replacement from the source. The latter describe sampling spread at the old temperature, not more new data. When fewer than ten effective parents carry the residual pinning deficit, a narrow bootstrap interval cannot exclude unseen rare exteriors; the registration withholds population claims. Scalar total-variance support and rare-deficit support answer different questions.

[Source 03 checkpoint and scalar study](https://github.com/konradha/RSMI-NE/tree/030b46644405cd94b394a843d96767d7c2c38ddf/results/experiment-03-cavity/corrected) · [05 matched partitions](https://github.com/konradha/RSMI-NE/tree/6a06813f2a702a29e97cb2dd62aeb2709cf0ffb4/results/experiment-05-inventory-transfer) · [07 supervised exterior forecasts](../experiment-07-boundary-forecast/) · Gökmen et al., [PRL 127, 240603](https://doi.org/10.1103/PhysRevLett.127.240603) and [PRE 104, 064106](https://doi.org/10.1103/PhysRevE.104.064106).
