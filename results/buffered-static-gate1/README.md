# Can a buffered shell forecast the actual seven-site glass state?

**STOP — Gate 1 is inconclusive; Gate 2 stays sealed.** Neither the conventional nor the detailed R4 forecaster improved on the source prior on development data. Both therefore returned the **same 210 probabilities for every exterior**, at the source and in true zero-shot transfer. Their held gain of zero with a [0, 0] interval is exact equality of two selected forecasts. **It is not an upper bound on physical information in the shell.** Missing Crystal3/Crystal5 provenance independently prevents a definitive test of specificity to noncrystalline structure.

| R4 assessment, 500M for the two cold endpoints | Eligible independent parents | Stars | Native log score, either model (bits/star) | Detailed minus conventional: native / soft |
|---|---:|---:|---:|---:|
| β=3.05, source confirmation | 4,095 | 28,394 | −7.13151 | 0 / 0 |
| β=2.711111, prespecified zero-shot transfer | 4,094 | 33,473 | −7.19796 | 0 / 0 |
| β=3.20, additional zero-shot transfer | 127 | 780 | −7.10427 | 0 / 0 |
| β=3.40, additional zero-shot transfer | 128 | 764 | −7.07736 | 0 / 0 |

The registered practical gain was +.01 bit/star. Zero parents contributed a nonzero paired increment. The paired soft-minus-native gain discrepancy and source-minus-transfer gain were likewise identically zero, not independent confirmations. R5 and the one-star sensitivity did not rescue the comparison. In total 41 frozen score banks cover **664,153 eligible physical stars from 41,212 independent parents** across 12 temperatures (physical stars counted once, not per mask or control). Further optimization at R4 was not supported by the registered DEV grid: fit loss fell while DEV loss rose at all three independent-parent fractions. Replacing exact training targets with observed states did not resolve the problem: the exact-trained forecast gained +.00032 [−.00023, +.00085] on native source states versus the native-trained forecast. These are results for these models and training budgets, not universal limits.

**Controls, with two-sided 95% whole-parent intervals.**

* **Contact:** the detailed model beat the prior by +3.82434 [3.79531, 3.85341] bits/star on native states; the corresponding exact score gain was +3.82792 [3.79926, 3.85586]. Direct-contact dependence is recoverable, but is not a buffered discovery. The β=0 de novo fit reproduced the same-sector canonical uniform oracle (−log₂210). In the source composition-matched null, 4,084 of 4,095 first-star parents survived derangement across 37 four-count bins; the detailed gain was zero.
* **Weak buffered plant:** the known conditional gained **+.00991 [.00975, .01007]** over the composition-bin law, not over the conventional decoder. Switching on the coupling improved the fitted conventional model's soft score **relative to that law** by +.00617 [.00600, .00633]; the increase in oracle headroom over the fitted model was +.00374 [.00363, .00385]. The much larger absolute oracle-minus-fitted gap (+.12336 on vs +.11962 off) mostly reflects imperfect modeling of the bin-dependent nuisance distribution. Against the conventional decoder the detailed model gained **+.00317 [.00311, .00324]** on soft targets and **+.00367 [.00270, .00455]** on sampled states. Off-control gains were +.00012 and +.00010 respectively, both below the practical margin. Both plant fits selected the 16-epoch cap. The joint bootstrap detection fraction of 1.000 describes precision **conditional on one fitted pair and one held realization**, not prospective power across independent datasets/refits, and not the chance of clearing a +.01 lower bound. No physical exclusion follows.
* **Archive vs exact teacher:** all 210 native categories occurred at source and primary transfer; 14 and 6 respectively had fewer than ten observed anchors. Native/exact marginal total-variation distances were .00820 and .00935. At 3.20 and 3.40 only 154 and 153 categories appeared; their distances were .02385 and .02143. Unseen categories were not dropped from scoring. At β=0 sampling alone produced .01906 distance from the uniform law. These are calibration measurements, not evidence of global equilibration.

**Gallery:** [buffered gains and transfer](inference-buffered-gains.png) · [absolute native and exact log scores](inference-model-scores.png) · [controls and plant](inference-controls.png) · [parent precision and one-star sensitivity](inference-parent-precision.png) · [DEV depth](dev-depth.png) · [held calibration](held-calibration.png) · [masks](masks.png).

**Audit and reusable data:** [signed estimates and parent support](inference-estimates.csv) · [human-readable uncertainty report](inference-summary.txt) · [decision by criterion](decision.json) · [independent transfer contrasts](independent-transfer.csv) · [on/off and native-trained controls](paired-fit-controls.csv) · [all per-parent scores](parent-scores.zip) · [210-state calibration](calibration.csv.gz) · [coverage](coverage.csv.gz) · [predictor mask perturbation](predictor-invariance.json) · [published pre-assessment freeze](https://github.com/konradha/RSMI-NE/tree/a6026bcac9013785716f40e6a2f2e5b7518789d5/results/buffered-static-gate1). No reserve states accessed, no RSMI training and no Modal charges.

[Pre-fit design and split](https://github.com/konradha/RSMI-NE/tree/2e0156681001fc2c406e073d69a5b12ee1dbd53d/results/buffered-static-gate1) · [parent roster receipt](roster.json) · [geometry and template audit](mask-audit.json).

**How redundant are the two shell inputs?** On 2,046 source DEV parents (14,332 stars), a fixed linear ridge reconstruction from the conventional descriptors explained **87.65% [87.60%, 87.71%]** of raw indicator variation and **93.34% [93.11%, 93.57%]** of the composition-centered planted contrast. It never saw native central states, exact targets or TEST. The 933-column FIT Gram matrix had numerical rank 591 at a relative eigenvalue cutoff of 10⁻¹⁰. Predicting the contrast is not predicting the star and an `R²` is not a bit score. The head-to-head test measures **the gain of these specified decoders**, not the information added by independent microscopic coordinates. [Reconstruction and parent uncertainty](descriptor-reconstruction.json) · [all 450 sitewise results](descriptor-sites.csv).

### What was held apart

Every estimate compares forecasts of the **same native state among 210 possibilities**. At each eligible star the exact conditional distribution from its full exterior supplies a second, soft score. None of the decoders sees that exterior. Source β=3.05 is the only physical fit temperature; normalization, logits, weights and checkpoint choices transfer unchanged. β=2.711111111111 is the prespecified transfer, not selected by the results. Other temperatures are descriptive; 3.20 and 3.40 each have only 128 original nonlocal preparation parents. No physical child runs, GPU, RSMI fit or Gate 2 analysis.

Parent selection is uniform among eligible parents, then among their eligible stars (or one first star in sensitivity). Replacing that interior by any assignment of its fixed inventory leaves all candidate eligibility indicators and their species-blind random priority unchanged: their interiors are disjoint. Thus neither selection nor parent weighting favors a particular permutation *within the sector*. Under equilibrium, averaging the full-exterior Gibbs conditional over the unavailable exterior would yield the masked-shell conditional. We check the archive's actual states as well; an energy replay is not an equilibration test.

Gate 2 retains the final 4,096 original TEST members, by the locked hash order, **at each of the ten release temperatures**: 40,960 identities in [roster.npz](roster.npz), `group==3`; no reserve outcomes were selected. Historical experiments used this archive, so this is held out from this experiment, not a previously unexamined dataset. The two 500M cold cohorts have no Gate 2 reserve.

### What is and is not a crystal control

All models saw 150 actual species labels, in the same native frame, with no absolute position, true coordination or hidden gap sites. R4/R5 each have 240 observed nearest-neighbor bonds; the contact mask has 168. Minimum periodic star-to-mask Manhattan distances are 3, 4 and 1 respectively (interaction-term diameter 2). The 933-feature conventional forecaster includes directional subregion composition, 2-site frequencies, supported partial coordination, plaquette motifs, and orientation/phase-resolved packing scores. The rich forecaster also gets the 450 raw per-site ternary indicators. Parameter widths and checkpoint selection curves are in [dev.csv](dev.csv).

[Nishikawa & Hukushima's Fig. 1](https://arxiv.org/html/2003.02872v2/full_pd_1.png) identifies phases by their preferred coordination, not their primitive cells. The [documented source search](mask-audit.json) did not recover authenticated vacancy/A/B bases for Crystal3 and Crystal5. We scanned two *proxy* periodic vacancy packings against the observed sites, all cubic versions/translations and both species assignments, including orientation- and face-resolved matches. They are exact pure-species zero-energy constructions; nothing establishes that they are the two observed crystal phases. Even a win against both cannot satisfy the named-crystal control. Coordinates or reference configurations from the authors could resolve this provenance gap in a new study; they cannot change this one's decision.

### Reproduction

Use the matching nonlocal archive, original [03 banks](../experiment-03-cavity/exact-exteriors-and-replicas.zip), and hash-pinned [cold nonlocal preparation](../cold-frozen-transfer-320-340/bank-receipt.json). The extraction script accepts `--archive`, `--old`, `--cold`, `--scratch` before `build`. The stock library was not modified. NumPy/SciPy for geometry, PyTorch CPU for the ordinary decoders, Matplotlib/pandas for figures. From the fork root:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
work=/home/fritz/rsmi-work/buffered-static-gate1
out=results/buffered-static-gate1
for b in 9 0; do for split in fit dev; do python examples/stock_glass_buffered_banks.py build "$b" "$split"; done; done
# Historical fit command. The exact selected heads are published; no refit is needed to score.
python examples/stock_glass_buffered_fit.py --work "$work" --threads 4
# Only after the model receipt was published:
for b in 9 8 10 11 0 1 2 3 4 5 6 7; do python examples/stock_glass_buffered_banks.py build "$b" assess --frozen-receipt "$out/freeze.json"; done
python examples/stock_glass_buffered_score.py --work "$work" --betas 9 8 10 11 0 1 2 3 4 5 6 7
python examples/stock_glass_buffered_gallery.py --scores "$work/scores"
python examples/stock_glass_buffered_inference.py --scores "$out/parent-scores.zip" --out "$out"
```
