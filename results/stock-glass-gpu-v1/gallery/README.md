# V1 · 100 epochs, one start per temperature

[Raw and EMA20 training curves (PNG)](training-curves.png) · [PDF](training-curves.pdf) · [Training-only temperature summary (PNG)](temperature-summary.png) · [PDF](temperature-summary.pdf) · [Complete 3D filter atlas (PDF)](all-filter-slices.pdf)

The fork logs one relaxed TRAIN InfoNCE estimate after the first update of each epoch. It is not a held-out score or exact MI. Filters are signed weights in the original learned component order; no seed alignment or sign flips.

| β | 3D filter slices by seed |
|---:|---|
| 0.000 (control) | [81791](filter-b00-s81791.png) |
| 0.339 | [81792](filter-b01-s81792.png) |
| 0.678 | [81793](filter-b02-s81793.png) |
| 1.017 | [81794](filter-b03-s81794.png) |
| 1.356 | [81795](filter-b04-s81795.png) |
| 1.694 | [81796](filter-b05-s81796.png) |
| 2.033 | [81797](filter-b06-s81797.png) |
| 2.372 | [81798](filter-b07-s81798.png) |
| 2.711 | [81799](filter-b08-s81799.png) |
| 3.050 | [81800](filter-b09-s81800.png) |
