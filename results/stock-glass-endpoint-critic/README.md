# Stock frozen-critic endpoint replay

[TEST: true vs permuted pairs (PNG)](held-out-versus-shuffled.png) · [PDF](held-out-versus-shuffled.pdf) · [TRAIN only (PNG)](endpoint-train-curves.png) · [PDF](endpoint-train-curves.pdf) · [Original ten-temperature 100-epoch gallery](../stock-glass-gpu-v1/gallery/)

| β | Seed | Actual Adam updates | TEST parents scored | True − shuffled, nats (95% interval) | Stock outputs |
|---:|---:|---:|---:|---|---|
| 0 | 6101 | 10,013 | 8,192 | -0.02564 [-0.13107, 0.07979] | [array](b00-s6101.npz) · [filters](filters-b00.png) |
| 3.05 | 6101 | 10,013 | 8,192 | 0.00687 [-0.19933, 0.21306] | [array](b09-s6101.npz) · [filters](filters-b09.png) |

Only a finite 10,013-update checkpoint qualifies. All eight other temperatures lack this frozen-critic readout; seeds 6102 and 6103 never ran. The [immutable ten-worker job](../stock-glass-h100-concurrent-v2/outcome.json) returned no completed checkpoints. Shorter TRAIN traces are not substitutes. The color scale is shared across both endpoint filter figures.

[Plan fixed before TEST](readout-plan.json) · [Batch-level data, actual coverage and provider bill](outcome.json). Within each fixed TEST batch, the same relaxed code draws and stock score matrix serve both masks. Uncertainty is across 17 disjoint parent batches, not shuffle replicates. This contrast is not exact mutual information.
