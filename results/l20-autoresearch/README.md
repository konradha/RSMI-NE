# Which information survives compression in the L20 lattice glass?

This experiment reuses the **frozen L20 endpoint archive**. It trains low-capacity discrete representations of local species blocks and asks what they predict about separated environments. No new physical configurations, dynamics or Monte Carlo runs are part of this campaign. The clock is **11:45:25–21:45:25 UTC on 24 November 2026**; it is a maximum elapsed-time allocation, not a claim about time actually spent training. See the [locked protocol](protocol.json) and the [previous short-range result](../l20-operators/README.md).

| Lane | Primary question |
| --- | --- |
| `rg` | Does a learned coarse variable retain a separated signal across scales, beyond scale-matched composition and energy controls? |
| `information` | After matching on all eight ternary block corners, does a compressed variable retain conditional information about its shell? The fraction of measurable strata matters. |
| `learn` | Does a nonlinear discrete channel generalize across at least two buffer layers, against capacity-matched additive and fixed codes? One-layer calibration does not answer this question. |

**Frozen test results.** A four-state variable learned from a 3³ block still predicts its nearest separated shell **after fixing the species at all eight corners**. The conditional bound is **0.04217 [0.04131, 0.04303] nats** at β=3.05 (uniform cubic frame, 4,096 held-out parents, 98.33% parent-block interval). An unchanged critic gives **0.03657 [0.03569, 0.03745]** at β=2.7111. Matching by cube symmetry brings the fraction of cases with a distinct matched donor from 12.8% to 88.3% at the cold endpoint; unmatched cases contribute zero. With fixed laboratory axes the weaker bound is also positive (**0.00464 [0.00441, 0.00488]**). The β=0 and permutation controls do not show positive information. This rules out a corners-only account of this **short-range** association; it does not identify a long-range order parameter. [Frozen scores and controls](information/confirmation.csv) · [plot](information/confirmation.png).

At **two** intervening layers the four-state nonlinear encoder has a cold bound of **0.00311 nats** (multiplicity-adjusted lower bound **0.00227**). It beats the trained additive and fixed controls at β=3.05. The preregistered three-temperature superiority claim **did not pass**: at β=2.3722 its gain over the fixed code was +0.00072 nats but the simultaneous lower limit was −0.00032. A positive association and reproducible superiority over controls are different claims. [All 16 frozen contrasts](learn/confirmation.json) · [frozen weights and parent-block scores](frozen-inference.zip).

**Exact discrete calibration.** Instead of asking a neural critic to find very weak information, take two opposite block corners and two exterior species as the observable. There are only nine possible inputs. We enumerated **every** partition into at most four states on training parents. At β=3.05 the best four-state code across one buffer layer is exactly the two corner-A bits; its validation bound is **0.00902 nats**. Across **two** layers a different partition retains **0.0000974 [0.000089, 0.000106] nats**. The β=0 control is nonpositive; at three layers the cold interval spans zero. Thus the earlier two-layer neural null did **not** mean physical independence. The second-layer signal is about 1/93 as large for this projection; this alone does not establish collective glass order. [Scores](quotient/calibration.csv) · [codes](quotient/codes.json). These are validation diagnostics across 32 whole-parent groups, not the primary frozen tests.

![Information retained by exact four-state quotients of a ternary species pair](quotient/quotient.png)

### Read the checkpoint

The runner creates [status.json](status.json), a [cross-lane scoreboard](leaderboard.csv) and a [progress figure](status.png) at launch, first accepted results, phase changes, failures, half-hour clock boundaries and shutdown. Read the `as_of_utc` timestamp; this is **not** a live process monitor. The scoreboard reports the latest validation observation and the single frozen test observation **per lane**. Its rows are not sorted by effect: unlike metrics, negative pools and conditioning events cannot be ranked on a common numerical scale. A red test marker is placed at the end of the eligible validation prefix, not at a new discovery round. Missing markers are not numerical zeros. The lane directories hold the individual `round-NNNN.json` and (if completed) `confirmation.json` receipts with metrics, controls and sampling counts.

Training and selection use whole-parent training and validation splits. Confirmation reuses neither validation to estimate test uncertainty nor test to select architectures, seeds or anchor sets. Multiple sites, orientations or round numbers do not create independent physical samples. The test fold is held out **for this search**, not a previously unknown physical ensemble: earlier studies used this archive. Read the lane's preregistered gate, matched comparator, null controls, independent-parent uncertainty and cross-family correction before interpreting a test estimate. A variational lower bound is not the true mutual information; a difference of lower bounds is not itself a lower bound on a difference of mutual informations. A nonpositive bound, failed positive control, finite-size ambiguity or explained unmeasurable estimate can be a valid **inconclusive** result, not a task crash. Conversely, a successful process exit alone does not establish a discovery.

### Start or resume from the repository root

Linux, GNU `timeout`, Python 3.11+ and the lane dependencies (NumPy, SciPy, scikit-learn, Matplotlib) are required. Use three non-nested directories: the existing release, durable **private** working storage, and this **public** output tree. Do not run a lane by hand against a managed output directory.

```sh
py="$HOME/.local/share/venvs/rsmi-l20-f898730/bin/python"
data="$HOME/datasets/lattice-glass/l20-16384-v1"
work="$HOME/rsmi-work/l20-autoresearch-1795520725"
out="$PWD/results/l20-autoresearch"
mkdir -p "$work"
"$py" -u examples/glass_autoresearch/runner.py \
  --start-epoch 1795520725 --hours 10 --reserve-minutes 45 \
  --data "$data" --work "$work" --out "$out" \
  --python "$py" --lanes rg information learn --max-workers 3 \
  >>"$work/runner.log" 2>&1
```

Run under a persistent supervisor/session for unattended use. For an isolated rehearsal use **new** work/output paths and a suitable start epoch and duration; do not change a real campaign's clock on restart. The runner launches at most one CPU process tree per lane, sets BLAS/OpenMP thread counts to one, hides GPUs and invokes only the three listed Python analysis entrypoints. It does not dispatch paid jobs, generate configurations or run git. `PUBLISH ...` in the log marks a consistent checkpoint for an operator to review, commit and push; it is not a claim that anything was pushed.

Each completed discovery is a numbered, bounded training/evaluation unit. All reported numbers must be finite or an explained `null` on an inconclusive receipt. At least **two** accepted units are required to open a confirmation ticket. New discovery is forbidden in the final 45 minutes; it may end earlier when the lane reports `search_exhausted: true` or the recent unit durations predict that the next one will not fit. Discovery has a hard cutoff of **21:00:25 UTC** for this protocol. A lane that qualifies locks its receipt prefix and launches `--mode confirm --round <last_completed_discovery>` once; the lane uses its own validation-only selection rule, not the last round's estimate by default. Confirmation jobs get scheduling priority. All jobs are bounded by two hours and their respective absolute cutoff, including a five-second termination grace. If every lane has finished, is quarantined, or cannot meet minimum discovery, the runner exits rather than waiting out the budget.

### Recovery and failure evidence

* Reissue the **same command** with the same private tree, release manifest, public tree, lane set and times. The work journal (`$work/scheduler/campaign.json`) records completed receipt hashes, outstanding launches and spent confirmation tickets; `$work/<lane>/` holds the lane models and caches. Keep both. Already accepted rounds are not relaunched. Concurrent schedulers cannot own the same work or output directory.
* SIGTERM/SIGINT stops new scheduling and publishes a detached checkpoint. Already launched workers retain their per-lane lease, logs and absolute deadline even if the scheduler process dies. A restarted scheduler adopts them rather than starting another copy. After a host loss without a durable exit record, the lane is quarantined, not guessed successful.
* On `FAIL <lane>` inspect `$work/scheduler/<lane>/<attempt>/request.json`, `stdout.log`, `stderr.log`, `exit.json` (when available) and `failure.json`. Requests record the exact arguments, script and release-manifest digests, timing and freeze ticket. Other healthy lanes continue. Missing, malformed, changed or unowned receipts, nonfinite values, nonzero exit and timeout are operational failures, distinct from reported scientific nulls. The final process returns nonzero if a lane remains failed.
* A failed **discovery** may be explicitly rearmed before cutoff with `--retry-discovery <lane>` after diagnosing the cause and restoring consistent private lane state. Preserve invalid output as evidence outside the live receipt path before manual repair. Never erase or replay a valid completed receipt. The scheduler will not overwrite one. A confirmation launch spends its ticket **before** spawning; a failed or ambiguous test evaluation is never retried by the scheduler. Do not replace the private ledger, remove a freeze, or rebuild it from public test scores.
