# 02 · Can stock RSMI-NE recover the known two-site coordination law?

**Yes for the microscopic sign; no for calibrated probabilities or a validated information bound.** At β=3.05 all three stock encoders learned which site has fewer occupied neighbors from twelve occupancy bits alone. Their absolute filter-direction cosines against the exact antisymmetric coordination difference were 0.993–0.997. But all three final critics became numerically unreliable, so **0/3 passed the preregistered joint test**. This is a positive calibration of the encoder on a known local mechanism, not a result about distant shells.

[Physical calibration](physical-calibration.png) · [Reverse-event support](rare-support.png) · [Learned operator and oracle gap](operator-recovery.png) · [Actual steps / failure map](fit-status.png) · [Cold signed assignment and lower-site rule](cold-assignment.png)

### Three different questions

**Does the archive exhibit the expected local preference?** Conditional on one A and one B, exchanging labels gives `P(i=A|d)=logistic(−4βd)` (d=mᵢ−mⱼ). All 10/10 tie intervals and 7/7 β=0 |d| intervals include ½. The analytic probability lies inside 52/54 supported pointwise 95% intervals. Departures: β=0.339, |d|=1: observed 0.79065 [0.78702,0.79428], expected 0.79504; β=1.694, |d|=1: observed 0.99920 [0.99892,0.99948], expected 0.99886. These are many descriptive intervals, not a simultaneous equilibrium test.

At β=3.05, |d|=1 had 0 reversals among 16,619 pairs; the law predicts 0.0836 reversals at that support. The conservative one-sided parent-cluster upper bound on the pair-weighted reversal probability is 1.68%. Zero observed reversals does not resolve the predicted small probability. Cold ties: i=A in 210 / 388. Absent strata have no estimate.

**Did the network learn which site has lower coordination?** The sign rule was frozen from TRAIN codes only. Balanced accuracy gives equal weight to `i lower` and `j lower`; its intervals cluster the observed pairs by independent TEST parent. Brier skill is occupancy-only loss minus stock loss. These classifier diagnostics do not require the unstable InfoNCE subtraction and are not substitutes for the joint test.

| Cold seed | Adam steps | Sign balanced accuracy [95%] | Brier skill vs occupancy [95%] | Final stock InfoNCE |
|---:|---:|---:|---:|---|
| 6101 | 10,212 | 0.9922 [0.9917, 0.9927] | 0.2428 [0.2423, 0.2433] | invalid |
| 6102 | 10,212 | 0.9968 [0.9964, 0.9971] | 0.2487 [0.2485, 0.2489] | invalid |
| 6103 | 10,212 | 0.9979 [0.9976, 0.9981] | 0.2480 [0.2478, 0.2483] | invalid |

**Where the models fail.** For cold non-ties with just one neighbor's difference, sign accuracy was 94.86%, 97.85% and 98.57%, respectively; the dominant `|d|=2` stratum had 89,800 pairs. For the 388 exact ties, where the law gives 0.5, mean predicted `P(i=A)` was instead 0.026, 0.933 and 0.951. On the identical cold critic matrices, the largest batchwise stock float32 versus stable float64 discrepancy per seed was 9.77×10⁴, 7.16×10⁸ and 1.61×10⁹ nats, against a frozen tolerance of 0.001. The raw scores are retained for diagnosis, not reported as information evidence.

**Is this long-range amorphous order?** No such inference follows. This is an imposed two-site Hamiltonian channel with only occupancy bits supplied to the encoder. Exact sign classification and predicting a rare unfavorable assignment are different tasks. The stock and analytical InfoNCE use the same contrastive groups, not `accuracy = MI`.

### Coverage and outputs

All 163,840 archived IDs retained (8,192 TRAIN + 8,192 TEST per β). Fifty predetermined site pairs per parent gave 4,096,000 candidates per fold. 1,106,613 TRAIN and 1,106,427 TEST unlike pairs retained; all 81,920 parents per side contributed. Parents were never reweighted by their observed number of unlike pairs. Three paired seeds × ten temperatures; actual 267,989 Adam updates / 133,703,419 training positives. 21/30 reached their full ≥10k step target, 1/30 valid final checkpoints. Five scheduled depths: 850, 1,700, 3,400, 6,800, final. All checkpoint statuses: complete: 73, invalid frozen score: 60, unreached: nonfinite: 17. No replacements or early-checkpoint rescue.

Loss intervals use parent-clustered ratio/contrast errors. Stock positive and null entries reuse a relaxed code draw and the same frozen critic matrix; a fixed within-group derangement breaks the pairing. The row-normalized signed-stratum plots use the float64 decomposition of the same score matrix, not a replacement loss. Raw TFP and the finite-group analytical witness (log conditional assignment probability) are in the stock-vs-oracle table. A missing/flagged result is not zero.

[Every signed d, seed and depth (PDF)](signed-d-atlas.pdf) · [Assignment calibration all temperatures (PDF)](assignment-calibration.pdf) · [Raw/EMA and held learning curves (PDF)](learning.pdf) · [Filter weights, all starts (PDF)](filters.pdf)

[Physical counts, rare-event bounds and odds](physical-law.csv) · [Fold IDs, finite inventory and geometric census](parent-pair-census.json) · [Final gate by seed](final-decisions.csv) · [All checkpoint statuses](status.csv) · [All conditional scores, support and uncertainty](learned-channel.csv) · [True vs shuffled vs exact oracle](stock-vs-oracle.csv) · [Individual contrastive groups](contrastive-groups.csv) · [TRAIN-only sign rules](frozen-rules.csv) · [Filter direction cosines](filter-direction.csv).

[Frozen design](design.json) · [Exact prepared packets and rosters](fixed-pair-banks.zip) · [Measured charge](billing.json). Six raw ZIPs contain the checkpoint weights, TRAIN curves, TEST codes and compact assignment score matrices. To regenerate, unpack the input bank ZIP into `$BANKS` and all six raw ZIPs into `$RUN`:

```sh
python examples/stock_glass_coordination_report.py --inputs "$BANKS" --results "$RUN" --out "$GALLERY"
```

H100 `sb-EhastVIeMrkK39uYhg3Zqt`: $0.81901625 gross, $5.67073697 cumulative, $19.32926303 remains. No experiment 03 launched.

Sources: [Nishikawa–Hukushima](https://doi.org/10.1103/PhysRevLett.125.065501); [Gökmen et al., PRL](https://doi.org/10.1103/PhysRevLett.127.240603), [PRE](https://doi.org/10.1103/PhysRevE.104.064106), [stock tutorial](https://github.com/RSMI-NE/RSMI-NE/blob/main/examples/tutorial.ipynb).
