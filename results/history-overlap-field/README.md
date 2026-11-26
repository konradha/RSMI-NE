# RSMI on the field of physical relaxation

**GPU outcome: numerical failure; the RSMI question remains unanswered.** Six workers executed real stock Adam updates concurrently on one **NVIDIA H100 80GB HBM3**. Two displaced-control fits returned **4.0 nats**: seed 6102 at update **718**, seed 6103 at **1,023**. With 48 distinct parents in the contrastive batch, each diagonal log retrieval term is at most ln(48) = **3.871201** nats. These observations are mathematically invalid, not especially good estimates. A first-step receipt verified GPU placement, but is not a successful fit. The first invalid scalar in each failure receipt is known; score matrices and optimizer states at that step were not saved, so the first failing internal operation is **not** identified.

| Paired seed | True: last attested update | Displaced: last attested update |
|---:|---:|---:|
| 6101 | ≥1 | ≥1 |
| 6102 | ≥1,024 | 718 (failed) |
| 6103 | ≥1,024 | 1,023 (failed) |

The four interrupted workers reported progress every 1,024 steps after step 1; their final counts are unknown. **At least 3,791 updates were applied; zero of six fits reached 8,192, let alone the registered minimum of 32,768.** The 65,536 extension criterion was never evaluated. Post-first-step throughput over the four measurable partial intervals ranged from 192 to 259 updates/s per worker; this is not a sustained full-fit benchmark. The two device telemetry samples include setup and do not characterize steady-state utilization. Each step contained 48 different FIT parents. Source TEST and the β=3.40 transfer TEST stayed sealed. No filters, learned code entropies, frozen-critic DEV curves or held conditional gains are reported from nonexistent checkpoints. The positive CPU gate below remains a separate result.

**Termination and cost:** terminated at **11:09:31 UTC**, 185.25 seconds on the local allocation clock (27.16 seconds inside the six-worker wave). **Estimated $0.3111, not an invoice**, including the reserved H100, CPUs and RAM; the $2.22 authorization was not spent. No retry. [Terminal GPU and cost receipt](gpu-receipt.json) · [per-arm updates and measured rates](training-by-seed.csv) · [actual provider billing check](billing-observed.json) · [termination](terminated.json) · [seed 6102 exception](checkpoints/displaced-s6102-failure.json) · [seed 6103 exception](checkpoints/displaced-s6103-failure.json) · [GPU UUID and utilization samples](checkpoints/telemetry.json) · [stock CPU input/critic smoke](gpu-input-smoke.json).

At 11:09:57 UTC the authenticated provider report had posted **$0.19626580** for this app in the still-open hour. That is a partial billing observation, **not added to** the $0.3111 estimate. The exact app/interval query and read-only reconciliation command appear in the billing receipt.

The [conditional conventional readout](readouts/base-receipt.json) did fit all six face targets on source FIT (48 parents × 8,000 centers in 8.34 s), with maximum absolute solver gradient below 2×10⁻⁶. It is a training artifact, not a held shape comparison. Without an admissible learned model there is nothing to compare it against on TEST.

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

**Actual paid allocation:** [sandbox `sb-GQZEBU37Dt4GOBETN0BXJr`](https://modal.com/id/sb-GQZEBU37Dt4GOBETN0BXJr), [app `ap-zhgY5crGUsSFqqDIAX2Da9`](https://modal.com/id/ap-zhgY5crGUsSFqqDIAX2Da9), created 11:06:27 UTC on 2026-11-26. [Provider receipt](provider.json) · [live pricing and hard resource reservation](reservation.json) · [corrected user authorization](paid-release.json) · [admission arithmetic](admission.json). $2.21795123 is **maximum exposure**, not an actual charge. The $165 workspace limit is user-stated, not provider-verified; no provider limits were changed.

**Examples without a learned code.** These [two nonoverlapping blocks](same-count-examples.png) came from the same β=3.20 FIT parent, same starting occupancy (20) and same change count (4). Their changes form four adjacent bonds versus none. They were selected by center geometry alone from 128 eligible translations, not by shell activity. They illustrate why a shape-controlled test is necessary; they are not RSMI discoveries. [Exact coordinates and ternary arrays](same-count-examples.json).

**Offline gate reproduction** from the fork root with the pinned `lattice-exploration` mirror and CPU NumPy/Matplotlib:

```sh
out=results/history-overlap-field
work="$HOME/rsmi-work/history-overlap-field"
OPENBLAS_NUM_THREADS=1 python3 examples/stock_glass_history_maps.py train --scratch "$work" --out "$out"
OPENBLAS_NUM_THREADS=1 python3 examples/stock_glass_history_gate.py --work "$work" --out "$out"
python3 examples/stock_glass_history_shapes.py --bank "$work/b0-a500-q80-fit.npz" --out "$out"
```
