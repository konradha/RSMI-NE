# HERO V3 decision

| Question | Result |
|---|---|
| Scope | 10 × 16,384 L20 native parents; 81,920 TRAIN and 81,920 TEST, zero identity overlap. Seen temperatures, not an external replication. |
| Optimization | 76 trajectories; 314 checkpoints; 704,512 actual H100 updates in main. Both finalists met the registered paired-doubling plateau at 16,384. The raw and matched laws were evaluated on the same fixed population. |
| Discovery | 61,440 FIT, 10,240 selection, 10,240 rule diagnostics, including β=3.05. All ten temperatures contributed positives. |
| Refit | 180,224 additional H100 updates; five fresh frozen-choice starts (6101–6105) and six fixed-code critics, each trained on all 81,920 permanent TRAIN parents. |
| Final TEST | 81,920 parents projected once. Of 8,192 per β, the 8,192 groups/law actually drew 8,186–8,191 unique parents under uniform, 8,181–8,187 under matched. Eight sites per parent are repeated measurements, not new simulations. |
| Code and finite retrieval | Seed 6101: -7.28274e-05 [-0.000147, 1.3472e-06] nats relative to guessing at eight-way retrieval, averaged across both laws and betas. Hard entropy range 1.55–1.56 bits across temperatures; not the spatial MI. |
| Optimizer variability | Held common hard nats (6101–6105 in order): -7.2827e-05, 2.0285e-05, 1.2147e-05, -5.5805e-05, -0.00010293. Fixed-start SD 5.39476e-05; intervals on comparisons use whole parents, not training seeds. |
| Six paired comparisons | Primary seed minus comparator, nats with simultaneous 95% intervals: site 0.000759414 [0.00059552, 0.0009233]; composition 0.000901929 [0.00068739, 0.0011165]; coordination 0.000928518 [0.00069096, 0.0011661]; crystal 0.0007472 [0.0005261, 0.0009683]; random_affine 0.000716253 [0.0004926, 0.00093991]; random_tanh 0.00067795 [0.00045678, 0.00089912] |
| Planted and shuffled controls | Pre-existing nonphysical affine/uniform fixture: weak 0.0038732, strong 0.16706, shuffled -6.7948e-06 expected nats. All four registered arms passed their synthetic control. Not a power measurement for a coupling injected into the physical archive. |
| First failed mechanism gate | Unstable across starts. Best signed descriptor: 1[V(1,-1,0)=A], fold-1 R² 0.2339 vs required 0.50; conventional site identity leaves 0% residual variance. Anchor-fixed correlations against seed 6101 0.093675, 0.39818; after frozen best cubic alignments 0.16599, 0.65219 (secondary). |
| Held-fold physical increment | For this same descriptor on TRAIN fold 1 only: gain over packing 9.35159e-05, 95% parent CI [-1.76821e-05, 0.000204714] in units of normalized squared error, not nats. Required material gain 0.01. No physical TEST opened for this feature; only the previously frozen retrieval assessment was eligible. |
| Cost | Pilot/main/refit gross USD 0.01119220, 1.32039666, 0.25971645 = $1.59131; H100-equivalent metered seconds 10.00, 1148.00, 225.81. Gross tagged resources, not credit-adjusted invoice. No other GPU used by V3. |
| Inference | The tested V3 forward encoder did not yield a reproducible, nonconventional named operator. No result about all possible spatial channels, phase formation, or transfer to unseen temperatures. Subsequent TRAIN-only schedule diagnostic cannot confirm a discovery. |

[All training seeds and depths](learning.csv) · [per-temperature draw coverage](learning-by-beta.csv) · [held-out code rates and scores](held-by-beta.csv) · [frozen parent-level uncertainty](assessment.json) · [primary learning curves](learning.png) · [final comparison](held-temperature.png)
