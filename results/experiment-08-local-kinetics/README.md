# 08 · Does a statically predictable cavity observable also relax slowly?

**Verdict: the stock observables retain real connected memory, but the archive does not establish that they are slower than ordinary packing.** At β=3.05, after 100 attempted internal bonds, raw correlations for the three stock scalars were 0.982–0.988. Subtracting each exterior’s own mean gave **0.332–0.452**. Most of the near-one raw signal is static pinning, not dynamical persistence. The site-linear packing estimate was higher (0.731), but its within-cavity variance came from only **25 effective parents** out of 8,187. The preregistered minimum was 100. The cold learned-minus-packing comparison is **unresolved**, not a win, a tie, or a supported rejection of the +0.05 margin.

### Results at β=3.05

All observables below use the same exteriors and local rates. CI: pointwise 95% over independent parents, conditional on the frozen functions. Initial rate is per attempted internal bond; a smaller rate means slower *initial* decay. ESS is the effective count computed from the contributions to the connected denominator, not a change to the physical cohort.

| Frozen scalar | Static plateau | Initial rate | Connected at 100 [95%] | Connected at 1,000 | Within variance | Parent ESS |
|---|---:|---:|---:|---:|---:|---:|
| Stock 6101 | .978 | .165 | .452 [.379, .525] | .450 | .02229 | 180 |
| Stock 6102 | .976 | .192 | .362 [.297, .426] | .361 | .02419 | 217 |
| Stock 6103 | .973 | .211 | .332 [.270, .394] | .331 | .02695 | 229 |
| Center vacancy | .999 | .201 | .324 [.123, .525] | .318 | .00033 | 13 |
| 03 site-linear packing | .998 | .077 | .731 [.553, .909] | .722 | .00159 | 25 |
| Common minimum-slope scalar | .976 | .00041 | .998 [.994, 1.001] | .997 | 2.262 | 7 |
| Common scalar, restricted TRAIN support | .977 | .00042 | .997 [.994, 1.001] | .997 | 2.314 | 7 |

The last two scalars were normalized to within variance 1 on TRAIN; their TEST within variances were 2.262 and 2.314, respectively. They assign very different weights to rare patterns from the stock or packing functions. Their near-one connected scores are dominated by four parents carrying half of their within variance. An asymptotic interval extending above 1 is shown untrimmed, not interpreted as physical.

**Paired primary differences, stock minus packing:** −.279 [−.463, −.095], −.369 [−.557, −.182], −.399 [−.584, −.214] in seed order. The differences describe this archive. Despite the signs of those intervals, nine parents supplied half the packing denominator; the heaviest 1% supplied 90%. The corresponding stock half-variance counts were 62, 96 and 87. No temperature at which the learned/packing comparison met the registered support criteria showed a lower 95% limit above +.05 for any seed. Comparisons away from β=3.05 are descriptive; an encoder trained at one β is not the same function at another.

**What is slow?** For the stock scalars, 45.3%, 36.2% and 33.2% of connected variance lay in each exterior’s nonstationary band from its gap through gap + .001 per attempt. Changing the bandwidth to .0001 or .01 gave ranges of 45.1–45.3%, 35.9–36.2%, and 33.0–33.2%, respectively. A fast initial decay and a remaining slow component coexist. No individual eigenvector, bulk relaxation time, or asymptotic nonzero *connected* plateau follows from this. The conditional chains are connected and must ultimately lose that memory.

**Why the cold comparison is hard:** 92.8% of cold exteriors put ≥99% of conditional mass on one state. The median effective number of thermally occupied states was 1.00003. The median positive gap was resolved at 1.02×10⁻⁶ per attempt; 917/8,187 gaps fell below the numerical floor. The cold tenth percentile was bounded above by 4.67×10⁻¹², not measured as zero. The median primary slow-band dimension was eight. The strong-support common reference retained only five TRAIN state indicators at this β and likewise failed the TEST parent-support check. Solving more eigenvalues cannot supply missing independent variance contributions.

### Gallery and computation receipt

[Connected vs raw decay and the static plateau](plateau-vs-decay.png) · [initial slopes and fixed-lag memory at all ten β](slope-and-lag.png) · [paired cold comparison and support flags](cold-paired.png) · [gap resolution, thermal support and slow-band power](spectral-support.png).

**Coverage:** all 81,909 eligible TRAIN parents for the common references and all **81,913 eligible TEST parents** for all four lags, initial rates and spectra. No spectral subsampling. The first eight β had 8,192 TEST parents each; the last two had 8,190 and 8,187. Equal energy rows were reused without dropping their parent weights; 72,622 generator evaluations. **No new trajectory, RSMI fit, GPU job, or Modal spend. Final experiment; sequence closed.**

**Checks:** one connected component; 480 undirected edges (120 vacancy–A, 180 vacancy–B, **180 A–B**). Across the cohort, maximum edge detailed-balance error 6.94×10⁻¹⁸; maximum symmetric stationary residual 8.33×10⁻¹⁷. An independently constructed dense generator and direct matrix exponential on the first eligible exterior at *each* β agreed within 2.13×10⁻¹³ in normalized covariance across all four lags. All 48 cubic actions plus an inventory-preserving state relabeling passed; β=0 center-vacancy decay matched `exp(−7t/6)` within 6.2×10⁻¹⁶. Reconstructed stock and packing plateaus agreed with 03 within 5.0×10⁻¹⁴. One cold divide-and-conquer eigensolve failed; QR diagonalization of the **same matrix** recovered it with eigenpair residual 1.67×10⁻¹⁵. No exterior, lag, or observable was removed to make the computation pass.

## What moves, and what the clock means

[Nishikawa & Berthier, PRL 132, 067101](https://doi.org/10.1103/PhysRevLett.132.067101), [paragraph following Eq. (1)](https://arxiv.org/html/2307.08110v2#S0.E1), specify swaps between sites of different occupation **or particle type**, and restrict this rule to nearest neighbors for physical dynamics. **Occupied A↔B exchanges are included**, alongside 0↔A and 0↔B. The confined variant here has six rate-1/6 Poisson clocks (equivalently one rate-one clock choosing a uniform bond), one for each center–arm bond. Equal species or a Metropolis rejection changes nothing. Never choose uniformly from currently legal moves. Nothing crosses the frozen exterior. One time unit is **one attempted internal bond**, not one sweep of an L20 lattice; these curves are not bulk relaxation times or production trajectories.

The 210 distinct labeled star states all contain 2 vacancies, 2 A and 3 B. For an internal unlike pair that swaps x to y, `Q_xy = min(1, exp(−β(E_y−E_x)))/6`; the diagonal is minus the total exit rate. The graph is connected for every finite β: star transpositions generate every permutation of this inventory. A vacancy-free particle exchange is not a change of sector. For β=0, a difference in vacancy indicator between two arms has rate 1/6; center vacancy minus 2/7 has rate 7/6. These are distinct checks of the factor of six.

## What was measured

For every exterior B and frozen scalar f, use its **own** conditional equilibrium mean `m_B = π_B·f`, and `g_B = f−m_B`. The reported connected decay is `C(t) = mean_B ⟨g_B,exp(t Q_B)g_B⟩_π / mean_B Var_π(f)`. Each parent has equal weight in the two sums; near-constant cavities do not get an equal vote in a mean of separately normalized correlations. The initial rate is the ratio of the mean Dirichlet form to the same within variance. There is one term `min(π_x,π_y)(f_y−f_x)²/6` per **undirected** unlike edge. `C(0)=1` only for nonconstant observables.

If one instead centers by a *single overall* mean, the resulting normalized autocorrelation tends to `T = Var_B(m_B)/(Var_B(m_B)+mean_B Var_π(f))`; its value at finite t is `T+(1−T)C(t)`. The second row of the decay plot shows this difference. The plateau measures static boundary selection, not a population that has failed to move.

The three learned functions are exactly the final stock hard-state maps followed by the [03 TRAIN scalar coefficients](../experiment-03-cavity/corrected/), not label matches or newly fitted dynamical readouts. Other functions on the same parents: center vacancy, unchanged 03 TRAIN site-linear packing contrast, and a shared kinetic reference learned on TRAIN alone. The last minimizes `fᵀ mean(D_B) f / fᵀ mean(W_B) f` over the supported state indicators, where `W_B = diag(π_B)−π_B π_Bᵀ`. Drop covariance-null directions including constants. This optimizes the **initial slope**, not a finite-lag correlation, and is not an upper bound on a held score. The registered restricted-support reference is a sensitivity check, never selected instead on TEST. At β=0 the minimum has multiplicity ten: the rate is meaningful, not the direction returned by the solver. At β=2.7111 no state passes the restricted TRAIN support mask; that comparator is undefined, not refitted with a relaxed threshold.

For finite lags, diagonalize the reversible, symmetric negative generator itself, not a near-identity transition matrix. An orthogonal reflection removes the known stationary vector √π before diagonalization; no numerically small *positive* mode is deleted from the connected decay. Unresolved gaps are left-censored, not zero; no inverses of unresolved eigenvalues or inferred dwell times. Slow-subspace power is the fraction of thermal variance projected onto **all nonstationary directions** within a fixed bandwidth above the individual exterior’s gap. This subspace depends on the exterior; it is not the shared TRAIN kinetic contrast. Three widths (.0001, .001 and .01 per attempt) are supplied. The thermal support count is `1/sum π²`; the effective number of contributing parents is a different quantity, `(sum v_B)²/sum v_B²`, `v_B=Var_π(f)`.

## Data and reproduction

[Preregistration](design.json) was [published before evaluating kinetic scores](https://github.com/konradha/RSMI-NE/commit/49d29bde720c986c56729f46fb75a3620df755b2). Source: original [03 exact exterior bank](../experiment-03-cavity/exact-exteriors-and-replicas.zip), one eligible star per original parent in the fixed physical sector. TEST is held from TRAIN reference selection, not an archive untouched by previous questions. Pointwise paired 95% delta-method intervals are conditional on frozen functions; sites and eigenvectors are not independent samples. Scores with under 100 effective parents or within/total variance below 10⁻⁵ are flagged, even if a point estimate is striking. Connected values at or below 10⁻¹⁰ are below the registered reporting resolution.

[Held slopes, variances, concentrations and power](observable-moments.csv) · [connected and raw finite lags](finite-lags.csv) · [paired learned–ordinary and learned–kinetic comparisons](paired-differences.csv) · [gap bounds, band ranks and thermal support](spectrum-support.csv) · [TRAIN variational coverage and rates](train-variational.csv) · [frozen functions, masks, spectra, parent IDs and edge list](frozen-functions.npz) · [numerical audits](matrix-receipt.json) · [cold decision](decision.json). `parents-b00.npz` through [`parents-b09.npz`](parents-b09.npz) contain TEST IDs and original roster indices, exact per-parent means, variances, Dirichlet forms, lag numerators and band powers. The arrays embed column names and validity masks. For a collapsed/undefined function, the mask—not its zero storage column—is authoritative.

To reproduce on CPU, extract the linked 03 bank to `BANK`, then run from the repository root:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 examples/stock_glass_local_kinetics.py --banks BANK --out NEW_DIR --workers 6
python3 examples/stock_glass_kinetics_report.py --out NEW_DIR
python3 examples/stock_glass_kinetics_figures.py --out NEW_DIR
```

Physical and methodological references: [Nishikawa & Hukushima, PRL 125, 065501](https://doi.org/10.1103/PhysRevLett.125.065501); [Nishikawa & Berthier](https://arxiv.org/html/2307.08110v2); Gökmen et al. [PRL 127, 240603](https://doi.org/10.1103/PhysRevLett.127.240603) and [PRE 104, 064106](https://doi.org/10.1103/PhysRevE.104.064106). The confined kinetic comparison is a new diagnostic of static-information-trained functions, not an RSMI prediction or claim made by those papers.
