"""Case-first lower bounds on I(H; E | all eight ordered ternary corners).

One uniform archived anchor per independent parent per panel. The primary physical
observable includes a uniform cubic coordinate frame. A donor's ordered corner
pattern matches with probability |stabilizer|/48 if it belongs to the case's
orbit and zero otherwise. The stabilizer size is constant over that orbit. Thus
uniform orbit donors, with a uniformly chosen matching orientation, are EXACT
conditional draws (no configuration-frequency weighting or isotropy assumption).
All cases count; K=1 scores zero. Fixed laboratory axes are a separate estimand.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

# This lane coexists with other CPU lanes. Set before numpy/BLAS initialization.
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))
from l20_operators_static import OUT as STATIC, dense, netfile, offsets, prob, readfit, softmax, source, table
from l20_pairs import L, SPLIT, partition

CORNERS = np.array([0, 2, 6, 8, 18, 20, 24, 26]); POW = 3 ** np.arange(8)
NC = 3 ** 8
NAMES = ('nonlinear', 'affine')
METRIC = 'conditional_InfoNCE_nats_per_uniform_positive'
HYP = 'The frozen nonlinear four-state operator retains shell information beyond all eight ordered ternary corners at beta=3.05 in a uniform cubic frame.'
SELECT = 'Maximum lower endpoint of the grouped validation interval for the nonlinear raw conditional bound, earliest round on ties; never differences of MI lower bounds.'
# train panels, validation panels, requested K, hidden units, block size
DESIGNS = [(6, 12, 4, 0, 256), (16, 24, 6, 12, 256), (32, 32, 8, 16, 256),
           (48, 48, 8, 24, 512), (80, 64, 12, 24, 512), (128, 96, 12, 32, 512),
           (192, 128, 16, 32, 512)]
ALPHA = .05 / 3
CI_NOTE = 'Two-sided Bonferroni 98.333333% Student intervals over means of disjoint, equal-sized parent blocks; approximate iid-block inference, not independent sites, panels, seeds or exact finite-sample coverage.'
FIXED = 'bits-body-1: two opposite A-species indicators. H is a function of C; its true conditional MI and its optimal zero-critic bound are exactly zero.'
FREEZE_RULE = 'Only the beta-9 test estimate is primary. Require its lower endpoint >0, no positive beta-0 bound, nonpositive averaged permutation null, positive archived-shell planted signal and fair-bit calibration agreement. beta-8 transfer, affine scores and fixed axes are secondary. Nonpositive bounds do not imply absence.'
EPS = .2


def dump(path, obj):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def save(path, **arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rng(*key):
    return np.random.default_rng(np.random.SeedSequence([430161, *key]))


def geometry():
    v, e = offsets(3, 1)
    def perms(coords):
        lookup = {tuple(k): i for i, k in enumerate(coords)}
        return np.array([[lookup[tuple(k)] for k in coords[:, axes] * signs]
                         for axes in itertools.permutations(range(3))
                         for signs in itertools.product((-1, 1), repeat=3)], dtype=np.int16)
    vp, ep, cp = map(perms, (v, e, v[CORNERS]))
    cs = (np.arange(NC)[:, None] // POW % 3).astype(np.uint8)
    transformed = cs[:, cp] @ POW
    orbit = transformed.min(axis=1)
    eligible = transformed == orbit[:, None]
    choices = np.argsort(~eligible, axis=1, kind='stable').astype(np.uint8)
    return table(v), table(e), vp, ep, orbit, choices, eligible.sum(axis=1)


def encoders():
    meta, enc, _ = readfit(netfile(STATIC/'fits', 3, 1, 'learned', 11))
    af = STATIC.parent/'affine'; receipt = json.loads((af/'freeze.json').read_text())
    path = af/f"seed-{receipt['affine']['seed']}.npz"
    if sha(path) != receipt['affine_weights_sha256']:
        raise ValueError('affine receipt mismatch')
    with np.load(path) as f:
        A, c = f['A'], f['c']
    def project(v):
        return np.stack([prob(v, 3, meta['spec'], enc), softmax(dense(v) @ A + c)])
    return project, {'nonlinear': 'w3-b1-learned-s11', 'static_archive_sha256': sha(STATIC/'fit-receipts.zip'),
                     'affine': path.name, 'affine_sha256': sha(path), 'fixed': FIXED}


def panel_bank(work, meta, fold):
    # Persist anchor indices, not dense ternary patches. Prefixes shared between
    # rounds; uniform iid positions (with replacement), never condition on a hit.
    path = work/f"anchors-{meta['sha256'][:16]}-f{fold}.npy"
    if not path.exists():
        pos = rng(102, meta['index'], fold).integers(L**3, size=(256, SPLIT[fold]), dtype=np.uint16)
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.save(f, pos, allow_pickle=False)
        temp.replace(path)
    return np.load(path, mmap_mode='r', allow_pickle=False)


def frame(v, e, geom, r, cubic):
    _, _, vp, ep, orbit, choices, multiplicity = geom
    c = v[:, CORNERS] @ POW
    if cubic:
        # Each independent parent supplies ONE uniform anchor. A new independent
        # uniform stabilizer representative for each parent, one uniform target
        # frame per orbit. Shared target frame fixes C, not the donors' free axes.
        canonical = choices[c, r.integers(multiplicity[c])]
        unique, inv = np.unique(orbit[c], return_inverse=True)
        target = r.integers(48, size=len(unique))[inv]
        v = np.take_along_axis(np.take_along_axis(v, vp[canonical], axis=1), vp[target], axis=1)
        e = np.take_along_axis(np.take_along_axis(e, ep[canonical], axis=1), ep[target], axis=1)
        c = v[:, CORNERS] @ POW
    return v, e, c


def groups(c, k, r):
    # Every positive asks for donors, even a singleton. The random cyclic order
    # gives a uniform sample without replacement for EACH case conditional on C.
    shuffled = r.permutation(len(c)); order = shuffled[np.argsort(c[shuffled], kind='stable')]
    pos = np.empty_like(order); pos[order] = np.arange(len(c))
    cats, inv, counts = np.unique(c, return_inverse=True, return_counts=True)
    start = (counts.cumsum() - counts)[inv]; n = counts[inv]; rank = pos-start
    j = np.arange(k); ix = order[start[:, None] + (rank[:, None] + j) % n[:, None]]
    mask = j < n[:, None]
    outsider = order[start + (rank+k) % n]
    return ix, mask, outsider, n > k


def samples(state, meta, fold, bank, p, block, k, geom, project, cubic=True, half=None):
    parents = partition(meta['index'])[0][fold]
    first, last = (0, len(parents)) if half is None else (half*len(parents)//2, (half+1)*len(parents)//2)
    vt, et = geom[:2]
    for lo in range(first, last, block):
        ids = parents[lo:lo+block]; a = bank[p, lo:lo+block]
        v = state[ids[:, None], vt[a]]; e = state[ids[:, None], et[a]]
        r = rng(211, meta['index'], fold, p, lo, int(cubic))
        v, e, c = frame(v, e, geom, r, cubic)
        # Explicit condition features; no future response, dynamics, or outer
        # shell labels enter H. Centered species indicators keep the NN small.
        x = np.column_stack((dense(e), dense(v[:, CORNERS])))
        ix, mask, outside, valid = groups(c, k, r)
        yield lo//block, c, x, project(v), e[:, 12] == 1, r.integers(2, size=len(c)), ix, mask, outside, valid


def priors(state, meta, bank, n, project, geom):
    # Leave-half-out nuisance estimate. Exact ordered C; uniform raw anchors,
    # independently oriented; 8 pseudocases shrunk to a global train prior.
    parents = partition(meta['index'])[0][0]; vt, _, vp = geom[:3]
    ret = np.empty((2, 2, NC, 4), np.float32)
    for half, ids in enumerate(np.split(parents, 2)):
        total = np.zeros((2, NC, 4)); count = np.zeros(NC, int)
        sl = slice(half*len(ids), (half+1)*len(ids))
        for p in range(n):
            a = bank[255-p, sl]; v = state[ids[:, None], vt[a]]
            v = np.take_along_axis(v, vp[rng(251, half, p).integers(48, size=len(ids))], axis=1)
            c = v[:, CORNERS] @ POW; q = project(v); count += np.bincount(c, minlength=NC)
            for j in range(2):
                np.add.at(total[j], c, q[j])
        global_p = total.sum(axis=1)/count.sum()
        ret[1-half] = np.log((total + 8*global_p[:, None, :])/(count[None, :, None]+8)).astype('f')
    return ret


def weights(hidden, r):
    n = 316
    return [(r.normal(size=(n, hidden))*.05).astype('f'), np.zeros(hidden, 'f'),
            np.zeros((hidden, 4), 'f'), np.zeros((n, 4), 'f')]


def network(x, w, logp):
    u = np.tanh(x @ w[0] + w[1]); raw = x @ w[3] + u @ w[2]
    norm = logsumexp(raw + logp, axis=1, keepdims=True)
    return raw-norm, u, np.exp(raw+logp-norm)


def ratios(z, ix, mask):
    k = mask.sum(axis=1); g = np.where(mask[:, :, None], z[ix], -np.inf)
    denom = logsumexp(g, axis=1)
    return z - denom + np.log(k)[:, None], np.exp(g-denom[:, None, :]), denom


def train(state, meta, bank, d, geom, project, round_no):
    panels, _, k, h, block = d; ps = priors(state, meta, bank, min(32, panels), project, geom)
    models = {}
    for half in range(2):
        ws = [weights(h, rng(490, round_no, half, j)) for j in range(2)]
        ms = [[np.zeros_like(v) for v in w] for w in ws]; vs = [[v.copy() for v in m] for m in ms]
        step = 0
        for p in range(panels):
            for _, c, x, qs, _, _, ix, mask, _, _ in samples(state, meta, 0, bank, p, block, k, geom, project, half=half):
                step += 1
                rate = .0007*(1 if p < panels*2/3 else .35)*np.sqrt(1-.999**step)/(1-.9**step)
                for j, (q, w) in enumerate(zip(qs, ws)):
                    z, u, posterior = network(x, w, ps[half, j, c]); _, share, _ = ratios(z, ix, mask)
                    g = -q.copy(); np.add.at(g, ix.ravel(), (share*q[:, None, :]).reshape(-1, 4))
                    g -= posterior*g.sum(axis=1, keepdims=True); g /= len(q)
                    du = (g @ w[2].T)*(1-u*u)
                    for a, m, v, grad in zip(w, ms[j], vs[j], (x.T@du, du.sum(0), u.T@g, x.T@g)):
                        m *= .9; m += .1*grad; v *= .999; v += .001*grad**2
                        a -= rate*m/(np.sqrt(v)+1e-8)
        for j, w in enumerate(ws):
            for i, a in enumerate(w): models[f'w{half}_{j}_{i}'] = a
    return dict(models, priors=ps)


def planted(bit, ix, mask):
    # Label association only: B is a tag on an existing shell, H carries its
    # value via a binary symmetric channel. No physical sample is generated.
    q1 = np.where(bit, 1-EPS, EPS); q = np.column_stack((1-q1, q1)); z = np.log(q)
    ratio = ratios(z, ix, mask)[0]; return (q*ratio).sum(axis=1)


def fair_k(k):
    # Exact expectation of oracle InfoNCE for independent fair labels attached
    # to the K archived shells, marginalizing the noisy positive analytically.
    score = 0.
    for m in range(k):
        agree = (1+m)/k; denom = agree*(1-EPS)+(1-agree)*EPS
        score += math.comb(k-1, m)/2**(k-1)*((1-EPS)*np.log((1-EPS)/denom)+EPS*np.log(EPS/(1-denom)))
    return score


def interval(x, alpha=ALPHA):
    x = np.asarray(x); mean = float(x.mean()); error = float(t.ppf(1-alpha/2, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return {'mean': mean, 'lo': mean-error, 'hi': mean+error}


def evaluate(state, meta, fold, bank, panels, d, geom, project, fitted, cubic):
    _, _, k, _, block = d; nb = SPLIT[fold]//block
    ws = [[[fitted[f'w{a}_{j}_{i}'] for i in range(4)] for j in range(2)] for a in range(2)]
    keys = ['nonlinear', 'affine', 'permuted', 'outside_null_times_coverage', 'outside_coverage', 'plant_shell',
            'plant_fair', 'fair_expected', 'coverage', 'mean_k', 'critic_fold_0', 'critic_fold_1', 'nonlinear_minus_affine_score']
    vals = {key: np.zeros(nb) for key in keys}; hist = np.zeros(k+1, np.int64)
    cmass = np.zeros(NC, int); seen = np.zeros(NC, int); analytic = np.array([0]+[fair_k(i) for i in range(1, k+1)])
    for p in range(panels):
        for b, c, x, qs, bit, fair, ix, mask, outside, valid in samples(state, meta, fold, bank, p, block, k, geom, project, cubic):
            n = mask.sum(axis=1); hist += np.bincount(n, minlength=k+1)
            cmass += np.bincount(c, minlength=NC); seen += np.bincount(c[n>1], minlength=NC)
            scores = {}
            for j, name in enumerate(NAMES):
                z = [network(x, ws[a][j], fitted['priors'][a, j, c])[0] for a in range(2)]
                q = qs[j]; ensemble = (z[0]+z[1])*.5
                rr, _, denom = ratios(ensemble, ix, mask); scores[name] = (q*rr).sum(axis=1)
                if j == 0:
                    for a in range(2): scores[f'critic_fold_{a}'] = (q*ratios(z[a], ix, mask)[0]).sum(axis=1)
                    # Average over uniform permutations of the positive index
                    # within the matched set, not a marginal shuffle across C.
                    mean_z = (ensemble[ix]*mask[:, :, None]).sum(axis=1)/n[:, None]
                    scores['permuted'] = (q*(mean_z-denom+np.log(n)[:, None])).sum(axis=1)
                    scores['outside_null_times_coverage'] = valid*(qs[0, outside]*rr).sum(axis=1)
            scores.update(outside_coverage=valid, coverage=n>1, mean_k=n, plant_shell=planted(bit, ix, mask),
                          plant_fair=planted(fair, ix, mask), fair_expected=analytic[n],
                          nonlinear_minus_affine_score=scores['nonlinear']-scores['affine'])
            for key, v in scores.items(): vals[key][b] += v.mean()/panels
    summaries = {key: interval(v) for key, v in vals.items()}
    summaries['fair_mc_minus_exact'] = interval(vals['plant_fair']-vals['fair_expected'], .001)
    summaries['fixed_corner_code'] = summaries['zero_critic'] = {'mean': 0., 'lo': 0., 'hi': 0.}
    counts = dict(parents=SPLIT[fold], independent_blocks=nb, parents_per_block=block, panels=panels,
        positives=int(hist.sum()), positives_with_contrast=int(hist[2:].sum()), k_histogram=hist.tolist(),
        ordered_strata_observed=int((cmass>0).sum()), strata_with_any_donor=int((seen>0).sum()),
        uncovered_physical_mass=float(hist[1]/hist.sum()), mean_negatives=float(hist @ np.arange(-1, k)/hist.sum()),
        expected_unique_positions_per_parent=float(L**3*(-np.expm1(panels*np.log1p(-1/L**3)))))
    return dict(beta=meta['beta'], index=meta['index'], frame='uniform cubic' if cubic else 'fixed laboratory',
                scores=summaries, counts=counts, fair_BSC_true_MI=float(np.log(2)+(1-EPS)*np.log(1-EPS)+EPS*np.log(EPS))), vals


def design(n):
    if not 0 <= n < len(DESIGNS): raise ValueError('round outside prespecified finite design ladder')
    return DESIGNS[n]


def diagnostics(summary):
    s = summary['scores']; cal = s['fair_mc_minus_exact']
    return {'within_stratum_permutation_nonpositive': s['permuted']['mean'] <= 1e-7,
            'outside_parent_null_not_significantly_positive': s['outside_coverage']['mean'] > 0 and s['outside_null_times_coverage']['lo'] <= 1e-7,
            'archive_shell_plant_detected': s['plant_shell']['lo'] > 0,
            'fair_bit_MC_consistent_with_exact_finite_K': cal['lo'] <= 1e-7 and cal['hi'] >= -1e-7,
            'finite_K_expectation_below_known_MI': s['fair_expected']['mean'] <= summary['fair_BSC_true_MI']+1e-7}


def plot_table(out, tag, summaries):
    rows = []
    for r in summaries:
        for key in ('nonlinear', 'affine', 'fixed_corner_code', 'permuted', 'outside_null_times_coverage', 'plant_shell', 'plant_fair', 'fair_expected'):
            rows.append(dict(beta=r['beta'], frame=r['frame'], quantity=key, **r['scores'][key], **{k:r['counts'][k] for k in ('independent_blocks','positives','uncovered_physical_mass','mean_negatives')}))
    with (out/f'{tag}.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    fig, (a,b) = plt.subplots(1,2,figsize=(10,4), layout='constrained'); z=np.arange(len(summaries))
    for j,(key,name) in enumerate([('nonlinear','nonlinear'),('affine','affine'),('permuted','permuted')]):
        v = [r['scores'][key] for r in summaries]; y=np.array([s['mean'] for s in v]); lo=[s['lo'] for s in v]; hi=[s['hi'] for s in v]
        a.errorbar(z+(j-1)*.14,y,yerr=[y-lo,hi-y],label=name,fmt='o',capsize=3)
    a.axhline(0,color='.4',lw=.8); a.set_ylabel('Conditional InfoNCE (nats / case)'); a.legend(fontsize=8)
    b.bar(z,[1-r['counts']['uncovered_physical_mass'] for r in summaries]);b.set_ylim(0,1);b.set_ylabel('Physical case mass with a donor')
    for ax in (a,b): ax.set_xticks(z,[f"β={r['beta']:.2f}\n{r['frame'].split()[0]}" for r in summaries],fontsize=8)
    fig.savefig(out/f'{tag}.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['discover','confirm']); p.add_argument('--round',required=True,type=int)
    for name in ('data','work','out'): p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args(); out=a.out; work=a.work; out.mkdir(parents=True,exist_ok=True); work.mkdir(parents=True,exist_ok=True)
    n=a.round; tag=f'round-{n:04d}' if a.mode=='discover' else 'confirmation'; done=out/f'{tag}.json'
    if done.exists(): return
    d=design(n); geom=geometry(); project, encmeta=encoders(); summaries=[]; provenance={}; blocks={}
    if a.mode=='discover':
        if (out/'freeze.json').exists(): raise ValueError('discovery already frozen')
        state, meta=source(a.data,9); provenance['9']=meta
        model=work/f'critic-{n:04d}.npz'
        if not model.exists(): save(model, **train(state,meta,panel_bank(work,meta,0),d,geom,project,n))
        with np.load(model) as m: fitted=dict(m)
        for cubic in (True,False):
            s,v=evaluate(state,meta,1,panel_bank(work,meta,1),d[1],d,geom,project,fitted,cubic)
            summaries.append(s); blocks.update({f'b9-{int(cubic)}-{k}': x for k,x in v.items()})
        champion=n; gate=diagnostics(summaries[0]); gate['positive_validation_lower_endpoint']=summaries[0]['scores']['nonlinear']['lo']>0
        selection_value=summaries[0]['scores']['nonlinear']['lo']; search_exhausted=n==len(DESIGNS)-1
    else:
        freeze=out/'freeze.json'; rows=[json.loads((out/f'round-{j:04d}.json').read_text()) for j in range(n+1)]
        champion=max(range(n+1), key=lambda j:rows[j]['selection_value']); selection_value=rows[champion]['selection_value']; d=design(champion)
        model=work/f'critic-{champion:04d}.npz'; digest=sha(model)
        if digest!=rows[champion]['critic_sha256']: raise ValueError('model changed since selection')
        lock=dict(through_round=n, selection=SELECT, validation_lower_endpoint=selection_value, selected_round=champion,
                  model_sha256=digest, hypotheses=HYP, gate=FREEZE_RULE, indices=[9,8,0], panels=max(64,d[1]), interval=CI_NOTE)
        if freeze.exists() and json.loads(freeze.read_text()) != lock: raise ValueError('confirmation selection already locked')
        if not freeze.exists(): dump(freeze,lock)
        with np.load(model) as m: fitted=dict(m)
        for idx in lock['indices']:
            state,meta=source(a.data,idx); provenance[str(idx)]=meta
            for cubic in (True,False):
                partial=work/f"locked-{digest[:16]}-{meta['sha256'][:16]}-{int(cubic)}.npz"
                if partial.exists():
                    with np.load(partial) as stored:
                        s=json.loads(str(stored['summary'])); v={k: stored[k] for k in stored if k!='summary'}
                else:
                    s,v=evaluate(state,meta,2,panel_bank(work,meta,2),lock['panels'],d,geom,project,fitted,cubic)
                    save(partial,summary=json.dumps(s,allow_nan=False),**v)
                summaries.append(s); blocks.update({f'b{idx}-{int(cubic)}-{k}': x for k,x in v.items()})
            del state
        gate=diagnostics(summaries[0]); gate['positive_primary_corrected_lower_endpoint']=summaries[0]['scores']['nonlinear']['lo']>0
        gate['beta_zero_not_positive']=summaries[4]['scores']['nonlinear']['lo']<=0; search_exhausted=True
    save(work/f'{tag}-blocks.npz',**blocks); plot_table(out,tag,summaries)
    receipt=dict(mode=a.mode,round=n,status='ok' if all(gate.values()) else 'inconclusive',hypothesis=HYP,metric=METRIC,
        estimate=summaries[0]['scores']['nonlinear']['mean'],interval=summaries[0]['scores']['nonlinear'],selection_basis=SELECT,
        selection_value=selection_value,selected_round=champion,search_exhausted=search_exhausted,gate=gate,gate_rule=FREEZE_RULE,
        counts=summaries[0]['counts'],data=provenance,params=dict(train_panels=d[0],validation_panels=d[1],max_K=d[2],hidden=d[3],parent_block=d[4],
            train_parents=10240, critic_gradient_training_parents_each=5120, nuisance='opposite train half only, 8 pseudoexamples per ordered stratum',
            train_anchor_seed=430161,bridge='average of two critic score arrays; disjoint gradient halves, cross-fitted priors, same frozen encoder (not independent replication)',
            shell='six 5x5 faces, radius 3, w=3, buffer=1; 150 shell labels',corner_indices=CORNERS.tolist()),
        provenance=encmeta,critic_sha256=sha(model),analysis_sha256=sha(Path(__file__)),results=summaries,interval_method=CI_NOTE,
        interpretation='Raw expected log K + f(H,Epos,C) - log sum_j exp f(H,Ej,C) is a lower bound on I(H;E|C). K depends on C and candidate availability, independent of values given C. Every physical case contributes once, including K=1 at zero. Exact expectation over the frozen stochastic encoder. Paired nonlinear minus affine is a critic-score contrast, not a difference or ordering of true informations.',
        matching='Case uniform over parent and anchor; one anchor per parent per panel. Within its orbit uniformly align each independent donor to EXACT ordered C, then select up to K-1 distinct donors in a shuffled cyclic order. For fixed laboratory axes no alignment; do not equate these estimands without isotropy. Shared target orientation per orbit only couples cases within the parent block.',
        calibration='Physical plant: archived shell face-center species A sent through BSC(0.2), oracle critic. Fair calibration independently tags the existing shells with fair bits, same BSC, exact finite-K binomial expectation at actual group sizes. Outside null substitutes H from a disjoint matched parent outside the entire group where available, zero elsewhere; averaged permutation randomizes positive index within the group. Neither null is subtracted.')
    dump(done,receipt)


if __name__=='__main__': main()
