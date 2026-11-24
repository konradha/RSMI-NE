# Experiment 01 · Does translating the neighborhood improve R4 learning?

**Translations improved optimizer completion, not held-out shell prediction.** At β=3.05, all three eight-origin encoders assigned all 8,192 TEST parents the same hard code `(1,1)` and had zero true-minus-shuffled gain in every batch. The two estimable 8-versus-1 improvements compared zero against a negative fixed-arm score; both intervals include zero. The third fixed arm stopped at 6,515 updates. A degenerate predictor's zero-width gain interval is not an upper bound on physical mutual information. TEST was held out of these fits but was inspected in earlier studies.

[Held-out gains and paired advantage](held-final.png) · [Coverage and actual updates](coverage.png) · [Learning curves, all ten temperatures (PDF)](learning-curves.pdf) · [Hot/cold filters, all matched starts (PDF)](hot-cold-filters.pdf)

### What actually ran

One H100, 90 assigned fits, 1/4/8 nested origins × 3 paired starts × 10 temperatures. 75/90 reached 10,032 Adam updates; 75/90 had valid final assessments. 427/450 scheduled checkpoints passed the frozen numerical gate; 15/30 final 8-versus-1 contrasts were estimable. Status counts: complete=75, nonfinite=15. No reruns or substitutes.

| Distinct origins per TRAIN parent | Valid final fits / 30 | Stock nonfinite exits | Valid fits with one hard code on all TEST parents |
|---:|---:|---:|---:|
| 1 | 15 | 15 | 0 |
| 4 | 30 | 0 | 28 |
| 8 | 30 | 0 | 27 |

Of the 23 checkpoints without valid scores, 20 were never reached following stock numerical stops; three reached checkpoints failed the prespecified frozen-score checks. All 90 fits ran. The β=0 control also had zero absolute eight-origin gain for every start; the full three-seed paired control is unavailable because one fixed arm stopped early.

81,920 distinct TRAIN and 81,920 distinct TEST parents actually covered; parent groups do not multiply with seeds, depths or anchors. Across fits: 862,224 actual optimizer updates, 428,084,400 positive TRAIN rows. A full fit is 76 epochs × 65,536 rows = 4,980,736 positives, all 8,192 distinct TRAIN parents, 131 batches of 500 plus 36 per epoch. TEST is one original anchor on each of 8,192 parents per beta, 16 batches of 500 plus 192. The 1/4/8 arms represent 8,192 / 32,768 / 65,536 distinct parent–origin pairs per temperature, with 65,536 slots in each bank.

### Final held-out effects (nats)

| β | Seed | 8-origin absolute gain [95%] | Paired 8 − 1 gain [95%] |
|---:|---:|---:|---:|
| 0 | 6101 | +0.0000 [+0.0000, +0.0000] | unavailable |
| 0 | 6102 | +0.0000 [+0.0000, +0.0000] | +0.1096 [-0.2557, +0.4750] |
| 0 | 6103 | +0.0000 [+0.0000, +0.0000] | +0.0020 [-0.0977, +0.1017] |
| 3.05 | 6101 | +0.0000 [+0.0000, +0.0000] | unavailable |
| 3.05 | 6102 | +0.0000 [+0.0000, +0.0000] | +0.1674 [-0.1743, +0.5090] |
| 3.05 | 6103 | +0.0000 [+0.0000, +0.0000] | +0.1821 [-0.2233, +0.5875] |

Gain is stock InfoNCE(true) − stock InfoNCE(deranged shell), from the **same** latent draws and score matrix. D subtracts matched arm gains **within each parent batch** before uncertainty estimation. Intervals use 17 disjoint batches, size weighting and t16. An unavailable score is not zero. Numerical exclusions retain raw stock scores, finite-matrix float64 diagnostics, and flags in the batch CSV.

### What varied

On 15 comparable beta/seed cells: median |8−1 final gain| 0.1004 nat; median |8−4| 0.0000; median |K8 final−1,716| 0.0000. Median final K8 range across three seeds: 0.0000 nat (10 temperatures with all three). These are descriptive magnitudes on available cells, not independent replications or tests of variance components. A per-beta breakdown is in [origin/depth](origin-depth-effects.csv) and [seed spread](seed-spread.csv).

Among the 75 full-length finite TRAIN curves, 8 increased by >0.01 nat in the last ten logged epochs versus the preceding ten. Raw and EMA20 curves are TRAIN, not held-out scores. This is not a plateau certificate: no TRAIN-only held-parent fold was reserved and TEST cannot authorize more depth. No extension or experiment 02 launched.

### Data, figures and reproduction

[All 450 depth rows](checkpoint-scores.csv) · [Paired contrasts](paired-differences.csv) · [All fit statuses and positive counts](fit-coverage.csv) · [Batch-level scores and numerical checks](test-batches.csv) · [Design frozen before dispatch](design.json) · [Permanent parent/offset rosters](frozen-rosters-origins.npz) · [First-epoch shuffle replay](shuffle-check.json) · [Charges](billing.json).

Six raw ZIPs, split by seed and beta range, contain the underlying per-fit NPZs (weights at each reached milestone, original training curves, frozen draws, parent-batch receipts). Unzip into the same directory; no dense critic matrices required. The pre-existing runtime cache was retained.

```bash
python examples/stock_glass_origins.py prepare --archive "$ARCHIVE" --rosters "$ROSTERS" --split "$SPLIT" --fixed "$FIXED" --out "$BANKS"
python examples/stock_glass_origins.py launch --inputs "$BANKS" --out "$RUN" --wheels "$WHEELS" --budget 9.66  # paid, fresh output dir
python examples/stock_glass_origins_report.py --results "$RUN" --manifest "$BANKS/prepared.json" --out "$GALLERY"
```

Modal `sb-UBas3Tyfs7COVoWMmnB9fD`; experiment gross $1.58901632, cumulative tracked gross $4.85172072 / $25. Artifacts and tagged bill are gross, not a net invoice.

Model and method: [Nishikawa–Hukushima, PRL 125, 065501](https://doi.org/10.1103/PhysRevLett.125.065501); [Gökmen et al., PRL 127, 240603](https://doi.org/10.1103/PhysRevLett.127.240603); [PRE 104, 064106](https://doi.org/10.1103/PhysRevE.104.064106).
