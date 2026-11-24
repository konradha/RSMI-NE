# Why the β=2.033 stock TRAIN trajectory reported +8 nats

Only one of the four short fits collapsed: β=2.033333333333, seed 81797. Changing only its seed recovered a nonconstant code; changing only its temperature/roster also recovered one. This is a run-specific instability in this cross, not a β-specific loss of dependence or a universally bad seed. Two seeds cannot estimate a failure rate or establish a physical transition. A positive TRAIN number is not held-out information.

| β | Seed | Mean TRAIN nats, last 20 logged epochs | Final logged hard counts (00, 01, 10, 11) |
|---:|---:|---:|---|
| 2.033333 | 81797 | 0.00000 | 0, 0, 0, 500 |
| 2.033333 | 81798 | 0.46723 | 291, 8, 58, 143 |
| 2.372222 | 81797 | 0.55056 | 23, 204, 232, 41 |
| 2.372222 | 81798 | 0.55808 | 258, 8, 54, 180 |

**Chronology.** The stock loop logs the first 500-parent TRAIN batch of each epoch, after Adam update `1+17*(epoch−1)`. At epoch 32 (528 updates), all 500 thresholded draws were `(1,1)`; this persisted in every later logged batch. At epoch 39 the stock estimate was −4 nats (possible in principle for a lower bound). At epoch 40 (664 updates) it was +8, impossible for a 500-way InfoNCE score: `ln(500)=6.214608`. At that point the two median logits on a fixed 500-parent TRAIN panel were +19.39 and +20.36; the logged relaxed codes had component ranges of only `1.19e−7` and `0`. The first exactly constant relaxed batch was epoch 42; every logged batch from epoch 57 was exactly constant. Thus hard-code collapse precedes the impossible estimate; the latter is not an information spike.

**Numerics.** The fork evaluates TFP 0.25.0 with a diagonal positive mask. It subtracts two separately reduced float32 terms: `mean(diag S) − (mean(row_logsumexp(S)) − log(n))`. Each diagonal entry is bounded by its row logsumexp, so the exact result cannot exceed `log(n)`. Adding a large constant to an entire row cannot change the exact answer, but can destroy float32 accuracy when reducing and subtracting the terms. A fixed CPU diagnostic used 500 rows `S[i,:] = float32(2**26 + r[i])`, `r = np.random.default_rng(81797).normal(0,8,500)`. Every row is constant, so the score is exactly zero. The stock TFP calculation returned **+8.0**; on the identical float32 entries cast to float64, subtracting each row maximum first gave **0.0**. This reproduces a sufficient numerical mechanism without modifying training. The original diagonal did not retain the epoch-40 critic matrix: its absolute scale, first internally inaccurate update, and exact failing reduction operation cannot be recovered from the saved trace. Epoch 40 is the first logged *provable* bound violation.

**Scope and provenance.** The two diagonal runs are the original 100-epoch outputs. Only the two missing cells ran anew, both on one NVIDIA H100, 1,700 measured Adam updates per fit, all 8,192 TRAIN parents per temperature, with the last 192 included. No TEST file was transferred or used in the diagnosis. Gross short-cross cost **$0.26314362**, cumulative tracked gross through this sandbox **$2.97950141** / $25. No additional runs are part of this diagnostic.

[Four TRAIN trajectories and collapse fractions (PNG)](training-cross.png) · [PDF](training-cross.pdf) · [Per-fit evidence and provider receipt](outcome.json) · [New β=2.033 TRAIN output](b06-s81798.npz) · [New β=2.372 TRAIN output](b07-s81797.npz)
