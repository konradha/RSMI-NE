"""One-shot supervised probes for the frozen reverse spatial bottleneck.

control uses no archive inputs. Physical CLI stages require PUBLIC/probe-release-N.json
pinning every read from WORK. Neither a release nor a diagnostic restores the raw arm.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
from time import perf_counter

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit, softmax
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'results/l20-autoresearch/spatial-cycle'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/spatial-cycle'
SEEDS = (260511, 260512)
SIZES = (512, 2048)
STEPS = (256, 512, 1024)
KEYS = ('w1', 'b1', 'w2', 'b2')
EPS = 1e-7
K = 4
RAW_STOP = 'not_run: the registered raw reference failed planted sensitivity before physical fit'
CUTOFF = datetime.fromisoformat('2026-11-24T21:00:25+00:00')
STOP = datetime.fromisoformat('2026-11-24T21:45:25+00:00')


def require(cond, msg):
    if not cond:
        raise ValueError(msg)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def write_npz(path, data):
    with path.open('xb') as f:
        np.savez_compressed(f, **data)


def verified_release(release, stage):
    require(release is not None and release.resolve().parent == PUBLIC, 'Release must be a published probe release')
    lock = json.loads(release.read_text())
    number = lock['number']
    require(type(number) is int and number > 0 and release.name == f'probe-release-{number}.json'
            and lock['stage'] == stage and isinstance(lock['sha256'], dict), 'Wrong numbered release')
    checked = {}

    def artifact(name):
        if name not in checked:
            path = WORK / name
            require(path.resolve().parent == WORK and name in lock['sha256'] and digest(path) == lock['sha256'][name],
                    f'Artifact not in release, or hash mismatch: {name}')
            checked[name] = path
        return checked[name]
    return artifact


def signed(shell):
    require(shell.ndim == 2 and shell.shape[1] == 49 and shell.dtype == np.uint8 and not np.any(shell[:, -1] >> 2),
            'Invalid 386-bit shell')
    x = np.unpackbits(shell, axis=1, count=386, bitorder='little').astype(np.float32)
    x *= 2
    x -= 1
    return x


def grad(x, y, b, params):
    w1, b1, w2, b2, d = params
    hidden = np.tanh(x @ w1 + b1)
    q = softmax(hidden @ w2 + b2, axis=1)
    logits = b[:, None] + d
    nll = np.logaddexp(0, logits) - y[:, None]*logits
    loss = np.sum(q*nll, axis=1, keepdims=True)
    dl = q*(nll-loss)/len(y)
    dh = (dl @ w2.T)*(1-hidden*hidden)
    dd = (q*(expit(logits)-y[:, None])).mean(axis=0)
    return loss.mean(), [x.T @ dh, dh.sum(axis=0), hidden.T @ dl, dl.sum(axis=0), dd]


def gradient_check():
    rng = np.random.default_rng(260572)
    x = rng.normal(size=(13, 7)); y = rng.integers(0, 2, len(x)); b = rng.normal(size=len(x))
    parts = [rng.normal(size=s) for s in ((7, 5), (5,), (5, K), (K,), (K,))]
    _, analytic = grad(x, y, b, parts)
    errors = []
    for p, g in zip(parts, analytic):
        for index in np.ndindex(p.shape):
            orig = p[index]; h = 1e-5
            p[index] = orig+h; plus = grad(x, y, b, parts)[0]
            p[index] = orig-h; minus = grad(x, y, b, parts)[0]
            p[index] = orig
            errors.append(abs((plus-minus)/(2*h)-g[index]))
    return float(max(errors))


def train(shell, y, b, seed):
    from spatial_word import init_encoder
    enc = init_encoder(seed)
    params = [enc[k] for k in KEYS] + [np.linspace(-1, 1, K, dtype=np.float32)]
    rng = np.random.default_rng(seed+7000)
    first = [np.zeros_like(p) for p in params]; second = [np.zeros_like(p) for p in params]
    b = np.asarray(b, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
    for step in range(1, STEPS[-1]+1):
        idx = rng.integers(len(y), size=128)
        _, grads = grad(signed(shell[idx]), y[idx], b[idx], params)
        for p, g, m, v in zip(params, grads, first, second):
            m *= .9; m += .1*g
            v *= .999; v += .001*g*g
            p -= .001/(1-.9**step)*m/(np.sqrt(v)/np.sqrt(1-.999**step)+1e-8)
        if step in STEPS:
            yield step, {k: p.copy() for k, p in zip(KEYS, params)}


def calibrate(z, y, b, k=K):
    # L2 curvature 2 per symbol; tying all four shifts gives curvature 8.
    require(k in (1, K) and np.all((z >= 0) & (z < k)) and len(z) == len(y) == len(b), 'Invalid calibration input')
    alpha = 2*K/k
    d = np.zeros(k)
    for h in range(k):
        idx = z == h
        if not idx.any():
            continue
        bh = b[idx]; yes = y[idx].sum(dtype=np.float64)
        def score(v):
            return expit(bh + v).sum(dtype=np.float64)-yes + alpha*v
        limit = len(bh)/alpha + 1.
        d[h] = brentq(score, -limit, limit, xtol=1e-12)
    return d


def clip(p):
    return np.clip(p, EPS, 1-EPS)


def scores(y, p):
    p = clip(p)
    return np.where(y, np.log(p), np.log1p(-p))


def clustered(d, who, parents):
    require(len(d) == len(who) and len(d) and parents >= 2, 'No paired eligible observations')
    n = np.bincount(who, minlength=parents)
    sums = np.bincount(who, weights=d, minlength=parents)
    point = float(sums.sum()/len(d))
    se = float(np.sqrt(parents/(parents-1)*np.square(sums-point*n).sum()) / len(d))
    w = float(t.ppf(.975, parents-1))*se
    return dict(point=point, se=se, lo=point-w, hi=point+w, mde80=2.80*se,
                projected_mde80_1024=2.80*se*np.sqrt(parents/1024))


def entropies(z, strata, k=K):
    s = np.unique(strata, return_inverse=True)[1]
    joint = np.bincount(k*s+z.astype(np.int64), minlength=(int(s.max())+1)*k).reshape(-1, k)
    marg = joint.sum(axis=0)
    def entropy(counts, totals):
        mask = counts > 0
        return float(-np.sum(counts[mask]*np.log((counts/totals)[mask]))/len(z))
    return dict(K=k, counts=marg.tolist(), H=entropy(marg, len(z)),
                H_given_stratum=entropy(joint, joint.sum(axis=1, keepdims=True)))


def control():
    from spatial_word import apply
    start = perf_counter(); e = gradient_check()
    rng = np.random.default_rng(260571); n = 512*32
    bits = rng.integers(0, 2, (n, 386), dtype=np.uint8)
    shell = np.packbits(bits, axis=1, bitorder='little')
    s = rng.integers(0, 2, n); b = -2+.6*s
    direction = np.where(bits[:, :64].sum(axis=1) >= 32, 1, -1)
    law = b+1.5*direction; y = rng.random(n) < expit(law)
    ft, ca, ho = slice(0, 256*32), slice(256*32, 384*32), slice(384*32, None)
    who = np.repeat(np.arange(128), 32)
    P = lambda v: clustered(v, who, 128)
    bare = scores(y[ho], expit(b[ho]))
    common = calibrate(np.zeros(len(b[ca]), dtype=np.uint8), y[ca], b[ca], 1)[0]
    pooled = scores(y[ho], expit(b[ho]+common))
    oracle = P(scores(y[ho], expit(law[ho]))-bare)
    report = dict(scope='Numeric independent abstract feature/outcome pairs only, no archive.',
                  physical_rows=0, dollars=0, train_parents=256, calibration_parents=128, held_parents=128,
                  rows_per_parent=32, gradient_max_abs=e, oracle_minus_base=oracle,
                  base_absolute=P(bare), common_shift=float(common), calibrated_base_absolute=P(pooled), seeds={})
    passed = e <= 1e-6 and oracle['lo'] > .02
    if e <= 1e-6:
        for seed in SEEDS:
            st = perf_counter(); enc = dict(train(shell[ft], y[ft], b[ft], seed))[STEPS[-1]]
            z = apply(enc, shell[ca]); d = calibrate(z, y[ca], b[ca]); z = apply(enc, shell[ho]); H = entropies(z, s[ho])
            score = scores(y[ho], expit(b[ho]+d[z]))
            v = P(score-bare); u = P(score-pooled)
            ok = min(v['lo'], u['lo']) > .01 and H['H'] >= .35 and np.count_nonzero(H['counts']) >= 2
            report['seeds'][str(seed)] = dict(held_score=P(score), gain_vs_base=v, gain_vs_calibrated_base=u,
                entropy=H, offsets=d.tolist(), seconds=perf_counter()-st, passed=bool(ok))
            passed &= ok
    report.update(passed=bool(passed), elapsed_seconds=perf_counter()-start,
                  probe_design_sha256=digest(PUBLIC/'probe-design.json'), calibration_addendum_sha256=digest(PUBLIC/'probe-calibration-addendum.json'),
                  implementation_sha256=digest(Path(__file__)))
    return report


def read_json(path):
    return json.loads(path.read_bytes())


def bank(artifact, role):
    levels = np.load(artifact('eligible.npy'), allow_pickle=False)
    require(np.array_equal(levels, np.unique(levels)) and len(levels) == 22, 'Eligibility changed')
    with np.load(artifact('rosters.npz'), allow_pickle=False) as roster:
        ref = roster[role+'_parent']
    with np.load(artifact(role+'.npz'), allow_pickle=False) as z:
        parent, s, y, shell = (z[k] for k in ('parent', 'stratum', 'y', 'shell'))
    require(np.array_equal(ref, parent) and len(parent) == len(np.unique(parent)), 'Roster/order mismatch')
    require(s.shape == y.shape == (len(parent), 128) and shell.shape == (*s.shape, 49)
            and np.all((y == 0) | (y == 1)), 'Bad bank shapes/labels')
    keep = np.isin(s, levels)
    who, anchor = np.nonzero(keep)
    return dict(shell=shell[keep], strata=s[keep], y=y[keep], who=who, anchor=anchor, parents=parent,
                all_y=y)


def support(data, n):
    y = data['all_y'][:n]; keep = data['who'] < n; w = data['who'][keep]; yg = data['y'][keep]
    pos = np.bincount(w, weights=yg, minlength=n); neg = np.bincount(w, weights=1.-yg, minlength=n)
    share = float(keep.sum()/y.size); retained = float(yg.sum()/y.sum()) if y.sum() else 0.
    return dict(all_anchors=y.size, eligible_anchors=int(keep.sum()), all_positives=int(y.sum()),
                eligible_positives=int(yg.sum()), eligible_share=share, positive_retention=retained,
                parents_with_positive=int((pos>0).sum()), parents_with_negative=int((neg>0).sum()),
                passed=bool(share >= .25 and retained >= .5 and min((pos>0).sum(), (neg>0).sum()) >= 128))


def count_vacancies(shell):
    counts = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1, dtype=np.uint16)
    return counts[shell].sum(axis=1, dtype=np.uint16)


def space_name(n, seed, step):
    return f'space-k4-n{n}-s{seed}-u{step}.npz'


def frozen_space():
    freeze = read_json(PUBLIC/'space-dev.json')['models']
    names = [space_name(n, seed, step) for n in SIZES for seed in SEEDS for step in STEPS]
    names.extend(f'space-random-s{seed}.npz' for seed in SEEDS)
    return {name: freeze[name]['sha256'] for name in names}


def targeted(artifact):
    data = bank(artifact, 'fit')
    ctl = read_json(PUBLIC/'probe-control.json')['passed']
    counts = count_vacancies(data['shell']); saved = {}; timing = {}; checks = {}
    for n in SIZES:
        checks[str(n)] = support(data, n)
        require(checks[str(n)]['passed'], f'{n} fit prefix lacks support')
        end = np.searchsorted(data['who'], n)
        saved[f'cuts{n}'] = np.quantile(counts[:end], [.25, .50, .75], method='inverted_cdf')
        if not ctl:
            continue
        with np.load(artifact(f'packing-oof-{n}.npz'), allow_pickle=False) as z:
            require(np.array_equal(z['parent'], data['parents'][data['who'][:end]])
                    and np.array_equal(z['anchor'], data['anchor'][:end])
                    and np.array_equal(z['stratum'], data['strata'][:end]), 'OOF opportunity mismatch')
            b = z['packing_oof']
        require(b.shape == (end,) and np.all(np.isfinite(b)), 'Invalid OOF packing offset')
        for seed in SEEDS:
            begin = perf_counter()
            for step, model in train(data['shell'][:end], data['y'][:end], b, seed):
                prefix = f'n{n}s{seed}u{step}'
                timing[prefix] = perf_counter()-begin
                for k in KEYS:
                    saved[prefix+k] = model[k]
    out = WORK/'probe-targeted.npz'
    write_npz(out, saved)
    return dict(outputs={out.name:digest(out)}, support=checks, targeted_admitted=bool(ctl),
                elapsed_by_start_checkpoint=timing,
                scope='Only encoder-fit rows, OOF packing for targeted arm. No hard decoder fit or score.')


def from_bundle(bundle, prefix):
    return {k: bundle[prefix+k] for k in KEYS}


def codes(artifact, pack, counts, targeted_bundle, n, step, admitted, cache):
    from spatial_word import apply
    z = {'occupancy':np.searchsorted(targeted_bundle[f'cuts{n}'], counts, side='left').astype(np.uint8)}
    for seed in SEEDS:
        for arm, name in [('random', f'space-random-s{seed}.npz'), ('native', space_name(n, seed, step))]:
            if name not in cache:
                with np.load(artifact(name), allow_pickle=False) as m:
                    cache[name] = apply(m, pack)
            z[f'{arm}_{seed}'] = cache[name]
        if admitted:
            z[f'targeted_{seed}'] = apply(from_bundle(targeted_bundle, f'n{n}s{seed}u{step}'), pack)
    return z


def offsets(artifact, data, n, raw_available=False):
    import spatial_raw
    model = spatial_raw.load(artifact(f'{"raw" if raw_available else "packing"}-model-{n}.joblib'))
    size = len(data['y']); a = np.empty(size); b = np.empty(size); raw = np.empty(size) if raw_available else None
    for i in range(0, size, 2048):
        sl = slice(i, i+2048)
        shell = data['shell'][sl]; s = data['strata'][sl]
        a[sl], b[sl] = spatial_raw.packing_offsets(model, shell, s)
        if raw_available:
            raw[sl] = spatial_raw.predict(model, shell, s)[:, 2]
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)), 'Nonfinite frozen logits')
    return a, b, raw


def calibration(artifact):
    data = bank(artifact, 'calibration'); nparent = len(data['parents']); y = data['y']
    require(nparent == 384, 'Not the registered exclusive probe-fit cohort')
    sup = support(data, nparent); require(sup['passed'], 'Probe-fit support failed')
    admitted = read_json(PUBLIC/'probe-targeted.json')['targeted_admitted']
    counts = count_vacancies(data['shell']); cache = {}; saved = {}
    with np.load(artifact('probe-targeted.npz'), allow_pickle=False) as targs:
        for n in SIZES:
            a, b, _ = offsets(artifact, data, n)
            no_code = np.zeros(len(y), np.uint8)
            for arm, base in [('stratum_shift', a), ('packing_shift', b)]:
                saved[f'n{n}_{arm}'] = calibrate(no_code, y, base, 1)
            for step in STEPS:
                z = codes(artifact, data['shell'], counts, targs, n, step, admitted, cache)
                for arm, states in z.items():
                    saved[f'n{n}u{step}_{arm}'] = calibrate(states, y, b)
    path = WORK/'probe-calibration.npz'; write_npz(path, saved)
    return dict(outputs={path.name:digest(path)}, support=sup, targeted_admitted=admitted,
                scope='Only 384 disjoint FIT parents calibrated; no dev or exam read; no model selection.',
                raw=RAW_STOP, calibrated_parameters={k:v.tolist() for k,v in saved.items()})


def mismatch(who, strata):
    rng = np.random.default_rng(260573); dest = np.empty(len(who), dtype=np.int64)
    for s in np.unique(strata):
        row = np.flatnonzero(strata == s)
        # Shuffled within-parent order followed by a stable sort on random parent rank.
        rng.shuffle(row); p, inv, sizes = np.unique(who[row], return_inverse=True, return_counts=True)
        shift = int(sizes.max())
        require(2*shift <= len(row), 'No within-stratum donor bijection to another parent')
        order = row[np.argsort(rng.permutation(len(p))[inv], kind='stable')]
        dest[order] = np.roll(order, shift)
    require(np.array_equal(np.sort(dest), np.arange(len(dest))) and np.all(who != who[dest])
            and np.array_equal(strata, strata[dest]), 'Invalid donor map')
    return dest


def predict_probabilities(z, calib, a, b, n, step, raw=None, donor=None):
    """Label-independent FIT-frozen hard probe deployment for already-encoded rows."""
    out = dict(stratum=clip(expit(a)), packing=clip(expit(b)),
               stratum_shift=clip(expit(a+calib[f'n{n}_stratum_shift'][0])),
               packing_shift=clip(expit(b+calib[f'n{n}_packing_shift'][0])))
    if raw is not None:
        out['raw'] = clip(raw)
    for name, state in z.items():
        offset = calib[f'n{n}u{step}_{name}']
        out[name] = clip(expit(b+offset[state if donor is None else state[donor]]))
    for name in ('native', 'targeted', 'random'):
        starts = [name+'_'+str(seed) for seed in SEEDS]
        if all(k in out for k in starts):
            out[name] = (out[starts[0]]+out[starts[1]])*.5
    return out


def rates(z, strata):
    h = {k:entropies(v, strata) for k,v in z.items()}
    for arm in ('native', 'random', 'targeted'):
        names = [f'{arm}_{seed}' for seed in SEEDS]
        if all(k in z for k in names):
            h[arm+'_joint_pair'] = entropies(K*z[names[0]]+z[names[1]], strata, K*K)
    flags = {}
    for seed in SEEDS:
        ref = h[f'native_{seed}']
        for other in ('occupancy', f'targeted_{seed}', f'random_{seed}'):
            if other in h:
                flags[f'native_{seed}_vs_{other}'] = bool(any(abs(ref[k]-h[other][k])>.1 for k in ('H','H_given_stratum')))
    return h, flags


def comparison(p, pm, z, data, raw_available):
    y = data['y']; who = data['who']; strata = data['strata']; np_ = len(data['parents'])
    interval = lambda d: clustered(d, who, np_)
    val = {k:scores(y, v) for k,v in p.items()}
    absolute = {k:interval(v) for k,v in val.items()}; diffs = {}
    for k,v in val.items():
        for ref in ('stratum', 'packing', 'stratum_shift', 'packing_shift', 'raw', 'native', 'targeted'):
            if ref != k and ref in val:
                diffs[f'{k}-{ref}'] = interval(v-val[ref])
    rate, flags = rates(z, strata)
    mis = {k:interval(val[k]-scores(y, pm[k])) for k in z.keys() | {'native','random','targeted'} if k in val}
    scatter = {k:interval(val[k+'_'+str(SEEDS[0])]-val[k+'_'+str(SEEDS[1])])
               for k in ('native', 'random', 'targeted') if k in val}
    return dict(absolute=absolute, paired=differences_or_reason(diffs, raw_available),
                rates=rate, achieved_rate_mismatch_gt_point1=flags, paired_start_scatter=scatter,
                native_minus_other_parent_shell_conditional_on_fixed_matching=mis)


def differences_or_reason(diffs, raw_available):
    if not raw_available:
        diffs['raw_contrasts'] = dict(status='not_run', reason=RAW_STOP)
    return diffs


def hot_null(p, data):
    # Never needed in the current addendum: raw gate closed prior to physical fit.
    from cycle_bank import catalogue
    _, _, ids, p0, _, mu = catalogue()
    s = data['strata']; ix = np.searchsorted(ids, s)
    require(np.array_equal(ids[ix], s), 'Unknown finite-composition stratum')
    res = data['y']-p0[ix]; np_ = len(data['parents'])
    assoc = {k:clustered(res*(logit(clip(v))-logit(clip(p['packing']))), data['who'], np_) for k,v in p.items()}
    a = assoc['native']; y = data['all_y']
    prev = clustered(y.ravel()-mu, np.repeat(np.arange(np_), y.shape[1]), np_)
    return dict(residual_association=assoc, unconditional_prevalence=float(y.mean()), exact_prevalence=mu,
                unconditional_difference=prev, passed=bool(-.001<=a['lo']<=0<=a['hi']<=.001 and abs(prev['point'])<=3*prev['se']))


def evaluate(artifact, role, raw_available=False):
    data = bank(artifact, role); who = data['who']; np_ = len(data['parents']); y = data['y']
    sup = support(data, np_); inds = mismatch(who, data['strata']); counts = count_vacancies(data['shell'])
    admitted = read_json(PUBLIC/'probe-targeted.json')['targeted_admitted']
    role_out = dict(stage=role, support=sup, checkpoints={}, raw_status='available' if raw_available else RAW_STOP,
                    targeted_status='available' if admitted else 'not_run: planted sensitivity gate failed',
                    confirmatory=False, spend_usd=0)
    cache = {}; primary_p = None; gains = {}; predfile = dict(parent=data['parents'][who], anchor=data['anchor'])
    sizes = SIZES if role == 'dev' else (2048,); steps = STEPS if role == 'dev' else (1024,)
    with np.load(artifact('probe-targeted.npz'), allow_pickle=False) as target, np.load(artifact('probe-calibration.npz'), allow_pickle=False) as calib:
        for n in sizes:
            a, b, raw = offsets(artifact, data, n, raw_available)
            for step in steps:
                z = codes(artifact, data['shell'], counts, target, n, step, admitted, cache)
                p = predict_probabilities(z, calib, a, b, n, step, raw)
                pm = predict_probabilities(z, calib, a, b, n, step, raw, inds)
                key = f'n{n}u{step}'; c = comparison(p, pm, z, data, raw_available)
                role_out['checkpoints'][key] = c
                for arm in ('native', 'targeted'):
                    if arm not in p:
                        continue
                    g = scores(y,p[arm])-scores(y,p['packing'])
                    for kind, before in [('update_doubling', (n,step//2,arm)), ('parent_ladder', (512,step,arm))]:
                        if before in gains:
                            c[f'{arm}_{kind}_change_in_gain'] = clustered(g-gains[before], who, np_)
                    gains[n,step,arm] = g
                if n == 2048 and step == 1024:
                    primary_p = p
                    predfile.update(p)
                    predfile.update({k+'_hard':v for k,v in z.items()})
                    predfile['donor_eligible_index'] = inds
                    ctrl = ('stratum','packing','stratum_shift','packing_shift')
                    bounds = {k:c['paired'][f'native-{k}']['lo'] for k in ctrl}
                    h = [c['rates'][f'native_{s}'] for s in SEEDS]
                    role_out['primary_descriptive'] = dict(lower95=bounds, all_lower_gt_margin=all(v>.001 for v in bounds.values()),
                        each_start_all_lower_gt_margin=all(c['paired'][f'native_{seed}-{k}']['lo']>.001 for k in ctrl for seed in SEEDS),
                        noncollapse_both=all(v['H']>=.35 and np.count_nonzero(v['counts'])>=2 for v in h),
                        mde_gate=bool(c['paired']['native-packing']['projected_mde80_1024']<=.001))
    if role == 'hot':
        role_out['finite_composition_null'] = hot_null(primary_p, data)
    path = WORK/f'probe-{role}-predictions.npz'; write_npz(path, predfile)
    role_out['outputs'] = {path.name:digest(path)}
    return role_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('control', 'targeted', 'calibrate', 'dev', 'cold', 'hot', 'transfer'))
    parser.add_argument('--release', type=Path)
    args = parser.parse_args(); stage = args.stage
    require(datetime.now(timezone.utc) < CUTOFF, 'Discovery cutoff passed')
    signal.alarm(max(1, int((STOP-datetime.now(timezone.utc)).total_seconds())))
    require(not (PUBLIC/f'probe-{stage}.json').exists(), 'Immutable public receipt exists')
    start = perf_counter()
    # The actual raw gate failed. No option to authorize an exam via this CLI.
    if stage in ('cold','hot','transfer'):
        raise SystemExit('Exam closed: raw sensitivity gate failed; no exam bank is opened.')
    if stage == 'control':
        require(args.release is None, 'Synthetic only')
        WORK.mkdir(parents=True, exist_ok=True)
        write_json(WORK/'probe-control-started.json', dict(stage=stage, data='synthetic only', authorization='6a7e0935b1b749c682b93389c6dec093d7c75576'))
        report = control()
    else:
        artifact = verified_release(args.release, stage)
        release = read_json(args.release)
        freeze = frozen_space()
        require(read_json(PUBLIC/'space-control.json')['pass'] is True and read_json(PUBLIC/'packing-fit.json')['support_passed'] is True,
                'Spatial plant or packing support gate closed')
        require(read_json(PUBLIC/'raw-stop.json')['status']=='stopped-before-physical-fit', 'Unexpected raw disposition')
        pins = release['sha256']
        for name, sha in freeze.items():
            require(pins.get(name) == sha, 'Missing actual publicly frozen spatial encoder: '+name)
        lineage = read_json(PUBLIC/'lineage.json')
        for name, field in [('rosters.npz','unpublished_private_rosters_sha256'),('eligible.npy','eligible_sha256')]:
            require(pins.get(name) == lineage[field], 'Invalid lineage pin')
        for n in SIZES:
            info = read_json(PUBLIC/'packing-fit.json')['fits'][str(n)]
            key = 'oof' if stage == 'targeted' else 'model'
            require(pins.get(info[key]) == info[key+'_sha256'], 'Packing release differs from fit freeze')
        for prior, file in [('targeted','probe-targeted.npz'), ('calibrate','probe-calibration.npz')]:
            if prior == 'targeted' and stage == 'targeted' or prior == 'calibrate' and stage != 'dev':
                continue
            r = read_json(PUBLIC/f'probe-{prior}.json')
            require(pins.get(file) == r['outputs'][file] and digest(Path(__file__)) == r['implementation_sha256'],
                    'Probe or code differs from previous freeze')
        role = {'targeted':'fit','calibrate':'calibration','dev':'dev'}[stage]
        require(pins.get(role+'.npz') == read_json(PUBLIC/f'bank-{role}.json')['bank_sha256'], 'Role bank differs from issued extraction')
        write_json(WORK/f'probe-{stage}-started.json', dict(stage=stage,release_sha256=digest(args.release),
                   started=datetime.now(timezone.utc).isoformat()))
        report = targeted(artifact) if stage == 'targeted' else calibration(artifact) if stage == 'calibrate' else evaluate(artifact,'dev')
        report.update(release_sha256=digest(args.release), issued_number=release['number'],
                      implementation_sha256=digest(Path(__file__)), actual_label_free_hashes=freeze,
                      calibration_addendum_sha256=digest(PUBLIC/'probe-calibration-addendum.json'))
    report.update(elapsed_seconds=perf_counter()-start)
    write_json(PUBLIC/f'probe-{stage}.json', report)
    print(PUBLIC/f'probe-{stage}.json', flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    main()
