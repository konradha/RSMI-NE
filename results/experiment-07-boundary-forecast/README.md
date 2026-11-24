# 07 · Which cavity label does the boundary favor? Which boundaries pin it weakly?

**The oriented contact shell predicts which label is favored. It is much less useful for predicting exceptions to strong pinning.** At β=3.05 a simple exterior-only classifier picked the exact most probable label on 80–82% of 8,187 held exteriors. Removing the boundary’s orientation, while preserving its packing, erased nearly all of that skill. But ordinary orientation-blind packing beat the directional regression at predicting *how strongly* those same boundaries pinned the label, for all three seeds.

These are forecasts on independent parents, not an exact enumeration passed off as a predictor. The frozen stock codes from [03](../experiment-03-cavity/) define three different targets. [06](../experiment-06-cubic-families/) showed they are not rotated copies. No labels were aligned or pooled here.

## Which label? Cold primary endpoint

Expected log loss in **bits per cavity**, scored against each boundary’s exact probability vector. Lower is better. “Rotated” has the same features, parameter count and tuning budget as “directional”, but a different predetermined cube rotation for each parent. Error intervals on gains are paired, pointwise 95% over TEST parents, conditional on these fits.

| Frozen seed | TRAIN prior | Packing | Rotated | Directional | Exact oracle | Packing − directional [95%] | Predicted mode = true mode |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6101 | 1.948 | 1.947 | 1.952 | 0.692 | 0.041 | 1.255 [1.231, 1.279] | 82.5% |
| 6102 | 1.998 | 1.979 | 1.991 | 0.753 | 0.045 | 1.226 [1.202, 1.249] | 79.9% |
| 6103 | 1.983 | 1.959 | 1.969 | 0.691 | 0.047 | 1.267 [1.245, 1.289] | 80.8% |

All three lower limits clear the [preregistered](design.json) 0.01-bit margin. Rotated − directional was 1.260 [1.236, 1.284], 1.237 [1.214, 1.261], and 1.277 [1.256, 1.299] bits. The directional advantage over orbit-averaged packing was present at all nine positive β (descriptive outside the cold endpoint); it was **not monotonic**. At β=0 the codes are constants: the oracle information is zero and perfect prediction is trivial.

For these frozen hard codes the empirical information about the label in the boundary is `H(mean q) − mean H(q)`: **1.907, 1.953, 1.935 bits** at β=3.05. These are ceiling values, not forecast scores or InfoNCE. Using the TRAIN prior instead of the held marginal adds 0.000325, 0.000324 and 0.000020 bits of loss. The directional forecast recovered roughly 64–67% of the available log-loss reduction; there is a real gap to the oracle.

**A microscopic example.** For seed 6102, the TRAIN-selected contrast for label `00` was whether the A indicator at shell site `(1,−1,0)` exceeded that at its opposite. On TEST, mean probability of `00` was 37.2% on the positive side (1,992 parents) versus 13.4% on the negative side (1,991); difference 23.8 percentage points [21.2, 26.4]. Ties were excluded from this comparison only. All 12 cold label contrasts and their coverage are shown, including channels and signs selected on TRAIN. This is an association at contact, not a causal intervention or a distinction beyond orientation-aware packing.

## How strongly? Independent forecast of replica agreement

`q_B` is the true four-label distribution conditional on exterior B. Its collision probability `s_B = Σ q_B²` is the chance that **two independent conditional interiors** agree in label. Each strength head was separately trained on `s_B`. Squaring and summing a coarsened label forecast instead would mix up the predictor’s uncertainty with actual within-cavity uncertainty; the directional plug-in underestimated mean agreement by 0.250–0.275 at cold.

| Seed | TRAIN marginal collision | Mean true `s` (held SD) | MSE: constant → packing → directional | Directional reduction vs constant [95%], ×10⁻⁴ | Spearman ρ: packing / directional |
|---|---:|---:|---:|---:|---:|
| 6101 | .270 | .980 (.097) | .009470 → .009365 → .009428 | 0.42 [−0.13, 0.96] | .284 / .275 |
| 6102 | .251 | .978 (.102) | .010495 → .010333 → .010443 | 0.52 [−0.11, 1.15] | .352 / .247 |
| 6103 | .256 | .977 (.104) | .010846 → .010637 → .010752 | 0.94 [0.26, 1.62] | .376 / .330 |

This is **weak skill, not zero skill**. Packing reduced cold MSE by 1.1–1.9% relative to the constant predictor; the directional head reduced it by 0.4–0.9%. The paired contrast favored packing over directional for every seed. All cold directional strength forecasts were in the highest fixed calibration bin: accurate mean calibration does not imply accurate per-boundary strength. At warmer temperatures the directional head gave larger absolute strength improvements; see the curves rather than extrapolating the cold ranking.

**Finding weakly pinned boundaries:** 321, 347 and 374 of 8,187 had `s < .8` (3.9–4.6%). Flagging strength below the TRAIN tenth percentile, the directional model found 55/321, 70/347 and 80/374 (17–21% recall); only 6.8%, 8.8% and 10.0% of its flags were true events. Packing found 61/321, 89/347 and 87/374 (19–26% recall). There were enough events to assess the tail; neither model found most of it. [Counts and Wilson intervals for every view](ambiguous-tail.csv).

## Gallery

- [Which label? Prior, packing, random frame, directional and oracle; paired gains at all ten β](which-label.png)
- [Modal-label calibration and actual representative boundaries](modal-calibration.png) — the 10th, 50th and 90th TRAIN confidence percentiles specify the examples, including incorrect ones. Exact label masses are distinct from model confidence.
- [Which shell differences distinguish each cold label?](boundary-descriptors.png) — both signs, TEST coverage and parent uncertainty.
- [Pinning strength: calibration, MSE gains and target dispersion](pinning-strength.png) · [Recovery of the ambiguous cold tail](cold-ambiguous-tail.png).

## What was fitted, and what was not

[Design locked before fitting/scoring](https://github.com/konradha/RSMI-NE/commit/0cfd403c6a6c6b924a4878b41056d222094e7d7c). Used all ten temperatures and all eligible original parents: **81,909 TRAIN, 81,913 TEST**. One seven-site star per parent, fixed inventory (2 vacancies, 2 A, 3 B). At the first eight β: 8,192/8,192 per β; at β=2.7111: 8,188/8,190; at β=3.05: 8,185/8,187. TEST is held out from these fits but has been used for other archive questions. No simultaneous bulk configurations were counted as independent conditional replicas.

- Forecast inputs are only the **18 exterior sites** at lattice L1 distance 2 from the center and their **exterior-only** coordination. Features at each site: A, B, coordination, its square, and coordination times A or B. The archived coordination explicitly subtracts every neighbor inside the seven-site cavity. No exact energies/probabilities, observed interior states, labels, anchor IDs, selection counts or target-side frequencies are predictors.
- Packing has 12 features: channel means on axial and face-diagonal shell orbits. Directional adds the 108 position-resolved deviations from those means. The capacity control rotates the whole shell by one of 24 proper rotations chosen by hash of immutable parent ID/β, independently per parent, with fixed original label frame. It retains the shape of each parent’s own exterior; it is not a cross-parent shuffle.
- For each β × seed × view, the label head minimizes soft-label cross entropy; the strength head is separate least-squares ridge, clipped to [.25,1]. Mean and scale are learned inside the fold, intercepts unpenalized. Three deterministic TRAIN folds; identical four λ values per view, independent selection for each head. Labels absent from a frozen map have exact zero target/forecast mass. Nonconstant fits use stable log-softmax. All selected choices were refit on the entire eligible TRAIN cohort. All ten parameter files were hashed in the [fit lock](fit-lock.json) **before** scoring any new TEST targets. Stock encoder weights never changed.
- The fit had no oracle as an input or alternative decoder. Only to form supervisory labels and exact held scores, we aggregate the archived 210-state conditional law by each stock code. The mathematical boundary-information ceiling refers to this fixed inventory, these exteriors and this code. Nothing here measures long-range amorphous order, a PTS length, or symmetry breaking.

**Audit data:** [per-view scores](held-scores.csv), [paired contrasts](paired-gains.csv), [coverage, collisions, entropy and dispersion](support.csv), [calibration bins with denominators](calibration.csv), [signed descriptors](descriptors.csv), [representative forecasts](representatives.csv), [all regularization trials, folds and convergence](training-selection.csv), [decision](decision.json). `fit-b00.npz` … `fit-b09.npz` contain TRAIN IDs, folds, rotations, frozen coefficients, both regularizers, signs and thresholds. `held-b00.npz` … [`held-b09.npz`](held-b09.npz) contain TEST IDs, rotations, exact `q`, per-view four-label forecasts and separate strength forecasts. In every held file the axes are indexed explicitly by `seeds` and `views`; order is seed, view, parent, label for categorical forecasts. The [checksums](held-receipt.json) cover all ten. Recomputing the cold paired log losses directly from saved vectors reproduced all three primary gains.

**Reproduce on CPU:** extract [the original 03 bank](../experiment-03-cavity/exact-exteriors-and-replicas.zip) into `BANK`. From the repository root:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir NEW_DIR
cp results/experiment-07-boundary-forecast/design.json NEW_DIR/
python3 examples/stock_glass_forecast.py --banks BANK --out NEW_DIR
python3 examples/stock_glass_forecast_readout.py --banks BANK --out NEW_DIR
python3 examples/stock_glass_forecast_figures.py --out NEW_DIR
```

**Context:** [Gökmen et al., PRL 127, 240603](https://doi.org/10.1103/PhysRevLett.127.240603) and [PRE 104, 064106](https://doi.org/10.1103/PhysRevE.104.064106) motivate predictive coarse observables; [Biroli et al., Nature Physics](https://doi.org/10.1038/nphys1050) and [Yaida et al.](https://arxiv.org/abs/1511.03573) motivate conditional overlap and observable-dependent point-to-set questions. This contact-shell forecast is our test, not a reproduction of their length measurements.

**Cost:** $0 on Modal; $17.99322425 remains unchanged. No GPU, new glass configurations, new trajectories, or new RSMI fit. Stopped before 08.
