# RSMI on the field of physical relaxation

**CPU gate: PASS.** At β=3.20 after 500M preparation sweeps and 37,642,814 local sweeps, occupancy-adjusted center–shell correlation exceeded its same-parent displaced control by **0.220 [0.102, 0.334]** (95% whole-parent interval). Twelve of 16 development parents had positive differences; the two prespecified halves averaged +0.117 and +0.322. The unadjusted difference was +0.223. This establishes spatial heterogeneity of *how much* relaxation occurred in this development cohort. It does **not** show that RSMI learns its arrangement beyond conventional shape.

| Temperature, 500M preparation | Saved local lag selected on TRAIN | TRAIN mean normalized Q |
|---|---:|---:|
| β=3.20 | 37,642,814 | 0.79517 |
| β=3.40 | 404,452,948 | 0.80039 |

[Gate figure](gate.png) · [parent-level gate scores](gate-parents.csv) · [gate decision and occupancy adjustment](gate.json) · [matched-Q curves with parent bands](matched-Q.png) · [actual lag choices](matched-lags.json) · [all 83 measured lags](lag-curves-train.csv) · [split and identities](split.csv) · [input checksums and lineage](lineage-train.json) · [map bank hashes](map-banks-train.csv). The two βs did not use the same number of sweeps: a fixed clock would compare different stages of relaxation.

**Question.** At matched mean relaxation, does the *arrangement* of local changes in a 3³ block convey anything about changes at its separated shell beyond activity count, initial occupancy and directional shape? The field describes net change between two saved times of the **same local trajectory**. It does not measure continuous persistence, individual particle motion, static order, or a causal effect.

[Pre-outcome design](https://github.com/konradha/RSMI-NE/blob/9659b814a9634441f91db73d61cebc91ef24fe75/results/history-overlap-field/design.md) · [status](status.json) · [physical archive and Q convention](https://github.com/konradha/lattice-exploration/tree/1cff504a996508c4b1bc8cdb050abfe560fff911/results/l20-paper-intermediate).

**Data.** Each of the 128 preparation parents at each temperature has a 300M- and a 500M-preparation local child. These are paired views of one preparation, not 512 independent parents. Starting vacancies are kept distinct from initially occupied sites that retained or changed species. The 386-site R4 shell is a minimum of 3 periodic Manhattan steps from the 27-site center. Moving the shell by (10,10,10) preserves its entire observed pattern, its parent and time window, but raises its minimum center distance to 15. This torus control is not an infinite-separation ensemble.

**What the CPU gate tests.** Within each source development parent, regress center and shell activity separately on their initial occupied fractions, then compare their true spatial correlation to the displaced one. Only the 16 per-parent differences enter the primary interval; 128,000 overlapping translations supply spatial coverage, not 128,000 standard-error units. A positive gate licenses the more difficult count-and-shape-conditional RSMI test. It does not answer it.

**GPU protocol.** One bounded H100, three paired independent starts (6101–6103), true and displaced-shell stock fits, 32,768 genuine Adam updates per fit minimum. A common, source-DEV-only paired improvement rule can extend all six to 65,536. Conditioning of physical held readouts, entropy matching, admissible numerical precision and no TEST selection are fixed in the design. A missing H100, a numerical failure or an incomplete fit is not a scientific zero. No new MCMC or access to the closed static Gate 2.

**Examples without a learned code.** These [two nonoverlapping blocks](same-count-examples.png) came from the same β=3.20 FIT parent, same starting occupancy (20) and same change count (4). Their changes form four adjacent bonds versus none. They were selected by center geometry alone from 128 eligible translations, not by shell activity. They illustrate why a shape-controlled test is necessary; they are not RSMI discoveries. [Exact coordinates and ternary arrays](same-count-examples.json).

**Offline gate reproduction** from the fork root with the pinned `lattice-exploration` mirror and CPU NumPy/Matplotlib:

```sh
out=results/history-overlap-field
work="$HOME/rsmi-work/history-overlap-field"
OPENBLAS_NUM_THREADS=1 python3 examples/stock_glass_history_maps.py train --scratch "$work" --out "$out"
OPENBLAS_NUM_THREADS=1 python3 examples/stock_glass_history_gate.py --work "$work" --out "$out"
python3 examples/stock_glass_history_shapes.py --bank "$work/b0-a500-q80-fit.npz" --out "$out"
```
