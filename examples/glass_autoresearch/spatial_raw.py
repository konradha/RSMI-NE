"""Unquantized conditional supervised reference for the new spatial-cycle study.

control reads no physical data. fit/dev are separately authorized. No exam CLI.
See results/l20-autoresearch/spatial-cycle/raw-design.json for the fixed contract.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cycle_models as cm  # limits native threads before importing numpy
import joblib
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

PUBLIC = Path(__file__).resolve().parents[2] / 'results/l20-autoresearch/spatial-cycle'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/spatial-cycle'
DESIGN = PUBLIC / 'raw-design.json'
SIZES = (512, 2048)
STEPS = (256, 512)
SEEDS = (260511, 260512)
BATCH = 512
NSITES = 386
MARGIN = .001
CUTOFF = datetime(2026, 11, 24, 21, 0, 25, tzinfo=timezone.utc)


def tangent_indices():
    shell = cm.bm.SHELL
    lookup = {tuple(v): i for i, v in enumerate(shell)}
    tables = []
    for axis in range(3):
        for face in (-4, 4):
            for tangent in range(3):
                if tangent == axis:
                    continue
                for order in (2, 3):
                    delta = np.eye(3, dtype=int)[tangent]
                    starts = shell[(shell[:, axis] == face) & (shell[:, tangent] <= 5-order)]
                    tables.append(np.array([[lookup[tuple(v+i*delta)] for v in starts]
                                            for i in range(order)]))
    return tables


TABLES = tangent_indices()
RIDGE = np.r_[np.full(NSITES, .5), np.full(len(TABLES), .02)]


def motifs(site):
    centered = site-site.mean(axis=1, keepdims=True)
    out = np.empty((len(site), len(TABLES)), 'f')
    for k, table in enumerate(TABLES):
        term = centered[:, table[0]] * centered[:, table[1]]
        if len(table) == 3:
            term *= centered[:, table[2]]
        out[:, k] = term.mean(axis=1)
    return out


def inputs(shell, strata):
    cm.rows(shell, strata)
    site = cm.signed_sites(shell)
    m = np.empty((len(shell), len(TABLES)), 'f')
    conv = np.empty((len(shell), cm.bm.D), 'f')
    for start in range(0, len(shell), 256):
        sl = slice(start, start+256)
        m[sl] = motifs(site[sl])
        conv[sl] = cm.summarize_shells(shell[sl])[:, :cm.bm.D]
    return site, conv, m


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def indices(levels, strata):
    require(np.all(np.isin(strata, levels)), 'Unexpected ineligible stratum')
    return np.searchsorted(levels, strata)


def loss_grad(x, y, base, w, penalty):
    logits = base+x@w
    residual = expit(logits)-y
    return (np.logaddexp(0., (1.-2.*y)*logits).mean()+.5*np.dot(w, penalty*w),
            x.T@residual / len(y)+penalty*w)


def gradient_check():
    rng = np.random.default_rng(260514)
    x = rng.normal(size=(19, len(RIDGE)))
    y = rng.integers(2, size=len(x)); b = rng.normal(size=len(x))
    w = rng.normal(scale=.03, size=x.shape[1]); h = 1e-6
    _, g = loss_grad(x, y, b, w, RIDGE)
    errors = []
    for j in range(len(w)):
        w0 = w[j]; w[j] = w0+h; upper = loss_grad(x, y, b, w, RIDGE)[0]
        w[j] = w0-h; lower = loss_grad(x, y, b, w, RIDGE)[0]
        w[j] = w0; errors.append(abs((upper-lower)/(2*h)-g[j]))
    return float(max(errors))


def train_seed(x, y, base, seed):
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1], 'f'); penalty = RIDGE.astype('f')
    base = base.astype('f'); y = y.astype('f'); snapshots = {}; t = 0
    while t < STEPS[-1]:
        order = rng.permutation(len(y))
        for start in range(0, len(order), BATCH):
            batch = order[start:start+BATCH]
            xb = x[batch]
            _, g = loss_grad(xb, y[batch], base[batch], w, penalty)
            w -= (.20*(1+t/64)**(-.75))*g
            t += 1
            if t in STEPS:
                snapshots[t] = w.copy()
            if t == STEPS[-1]:
                break
    return snapshots


def fit_arrays(features, strata, y, parent, levels):
    site, conv, m = features
    require(np.unique(y).tolist() == [0, 1], 'Both outcomes required')
    ids = indices(levels, strata)
    ordered = np.random.default_rng(260405).permutation(np.unique(parent))
    first = np.isin(parent, ordered[::2])
    start = perf_counter(); oof = np.empty((len(y), 2)); fits = []
    for held in (first, ~first):
        other = ~held
        pack = cm.packing_fit(site[other], conv[other], ids[other], y[other], len(levels))
        oof[held, 0], oof[held, 1] = cm.packing_logits(pack, site[held], conv[held], ids[held])
        fits.append([pack['logistic_iterations'], pack['tree_iterations']])
    pack = cm.packing_fit(site, conv, ids, y, len(levels))
    fits.append([pack['logistic_iterations'], pack['tree_iterations']]); packed = perf_counter()
    scale = cm.scale(m); x = cm.matrix(site, m, scale)
    weights = {seed: train_seed(x, y, oof[:, 1], seed) for seed in SEEDS}
    return (dict(levels=levels, packing=pack, scale=scale, weights=weights), oof,
            dict(eligible_rows=len(y), eligible_parents=len(ordered), baseline_seconds=packed-start,
                 raw_seconds=perf_counter()-packed, baseline_logistic_tree_iterations=fits,
                 seed_updates={str(s): STEPS[-1] for s in SEEDS}, fold_parent_counts=[len(ordered[::2]), len(ordered[1::2])]))


def packing_offsets(model, shell, stratum):
    """Unbounded registered stratum and packing logits in the supplied row order."""
    cm.rows(shell, stratum); ids = indices(model['levels'], stratum)
    site = cm.signed_sites(shell); conv = cm.summarize_shells(shell)[:, :cm.bm.D]
    return cm.packing_logits(model['packing'], site, conv, ids)


def checkpoint_columns(model, features, stratum):
    site, conv, m = features
    a, b = cm.packing_logits(model['packing'], site, conv, indices(model['levels'], stratum))
    x = cm.matrix(site, m, model['scale'])
    cols = dict(stratum=cm.probabilities(a), packing=cm.probabilities(b))
    for t in STEPS:
        pp = [expit(b+x@model['weights'][seed][t]) for seed in SEEDS]
        for seed, p in zip(SEEDS, pp):
            cols[f'raw{t}_{seed}'] = np.clip(p, cm.EPS, 1.-cm.EPS)
        cols[f'raw{t}'] = np.clip((pp[0]+pp[1])*.5, cm.EPS, 1.-cm.EPS)
    return cols


def load(path):
    """Read trusted raw-model-2048.joblib (or its directory)."""
    path = Path(path)
    return joblib.load(path/'raw-model-2048.joblib' if path.is_dir() else path)


def predict(model, shell, stratum):
    """Probability of Y=1, columns stratum / packing / raw final seed ensemble."""
    out = np.empty((len(shell), 3)); cm.rows(shell, stratum)
    for start in range(0, len(shell), 512):
        sl = slice(start, start+512); s = stratum[sl]
        f = inputs(shell[sl], s)
        a, b = cm.packing_logits(model['packing'], f[0], f[1], indices(model['levels'], s))
        x = cm.matrix(f[0], f[2], model['scale'])
        raw = sum(expit(b+x@model['weights'][seed][STEPS[-1]]) for seed in SEEDS)*.5
        out[sl] = np.column_stack((cm.probabilities(a), cm.probabilities(b), np.clip(raw, cm.EPS, 1.-cm.EPS)))
    return out


def interval(v, n):
    st = cm.bm.ratio_interval(v, n)
    require(st['se'] is not None, 'Insufficient parent mass')
    st['mde80'] = 2.8*st['se']; st['projected_mde80_1024'] = st['mde80']*np.sqrt(len(n)/1024)
    return st


def evaluate(model, features, strata, y, who, p):
    cols = checkpoint_columns(model, features, strata)
    names = list(cols); n = np.bincount(who, minlength=p)
    sums = {k: np.bincount(who, weights=cm.scored(y, prob), minlength=p) for k, prob in cols.items()}
    diffs = {'packing-stratum': interval(sums['packing']-sums['stratum'], n)}
    for k in names[2:]:
        for ref in ('stratum', 'packing'):
            diffs[k+'-'+ref] = interval(sums[k]-sums[ref], n)
    for seed in ('', *('_'+str(s) for s in SEEDS)):
        diffs['raw512-raw256'+seed] = interval(sums['raw512'+seed]-sums['raw256'+seed], n)
    gains = [diffs[f'raw512_{s}-packing']['estimate'] for s in SEEDS]
    updates = diffs['raw512-raw256']
    return (dict(absolute={k: interval(v, n) for k, v in sums.items()}, paired=diffs,
                 final_seed_gain_range=float(max(gains)-min(gains)),
                 final_update_point_rising=bool(updates['estimate'] > 0),
                 final_update_95pct_rising=bool(updates['lo'] > 0)), sums, n)


def support(y, s, mask):
    pos = np.sum(y*mask, axis=1); n = mask.sum(1)
    share = float(mask.mean()); retained = float(pos.sum()/max(y.sum(), 1))
    good = [(n-pos > 0).sum(), (pos > 0).sum()]
    return dict(parents=len(y), total_anchors=int(y.size), eligible=int(n.sum()), eligible_positives=int(pos.sum()),
                eligible_fraction=share, retained_positive_fraction=retained, class_parents=list(map(int, good)),
                passed=bool(share >= .25 and retained >= .50 and min(good) >= 128))


def plant():
    rng = np.random.default_rng(260513); p = 384; n_anchor = 32
    density = np.repeat(rng.uniform(.20, .30, p), n_anchor)
    vac = rng.random((len(density), NSITES)) < density[:, None]
    shell = np.packbits(vac, axis=1, bitorder='little'); s = rng.integers(2, size=len(shell))
    # Independent implementation of planted pair anisotropy from native face arrays, not model features.
    face = vac[:, cm.bm.SHELL[:, 0] == -4].reshape(-1, 9, 9).astype(float)
    d = vac.mean(axis=1); z = 2*(face-d[:, None, None])
    anis = ((z[:, :-1, :]*z[:, 1:, :]).mean((1, 2))-(z[:, :, :-1]*z[:, :, 1:]).mean((1, 2)))/(.75*np.sqrt(2/72))
    logits0 = -2.35+.70*s+.45*(d-.25)/.05; logits = logits0+.45*anis
    prob = expit(logits); y = (rng.random(len(s)) < prob).astype('u1')
    return shell, s, y, np.repeat(np.arange(p), n_anchor), logits0, logits


def output_np(path, **arrays):
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)


def control(work):
    path = work/'raw-control.json'; begin = perf_counter()
    require(not path.exists(), 'Control already issued')
    grad = gradient_check(); shell, s, y, parent, logits0, logits = plant()
    cut = int((parent < 256).sum()); begin_features = perf_counter()
    fs = inputs(shell, s); feature_s = perf_counter()-begin_features
    model, _, training = fit_arrays(tuple(f[:cut] for f in fs), s[:cut], y[:cut], parent[:cut], np.arange(2))
    since = perf_counter(); sl = slice(cut, None)
    held, sums, n = evaluate(model, tuple(f[sl] for f in fs), s[sl], y[sl], parent[sl]-256, 128)
    held_s = perf_counter()-since
    # Verify the public issued probability interface has exactly the primary held scoring semantics.
    issued = predict(model, shell[sl], s[sl]); draw = y[sl]
    check = [np.bincount(parent[sl]-256, weights=cm.scored(draw, issued[:, i])) for i in range(3)]
    api_error = float(max(np.abs(observed-sums[k]).max() for observed, k in zip(check, ('stratum','packing','raw512'))))
    a = logits[sl]; b = logits0[sl]; prob = expit(a)
    oracle = float((prob*(a-b)-np.logaddexp(0., a)+np.logaddexp(0., b)).mean())
    keys = ['raw512-packing', 'raw512-stratum']; di = held['paired']
    passed = bool(grad < 1e-6 and api_error < 1e-4 and all(di[k]['lo'] > MARGIN and di[k]['estimate'] > MARGIN+di[k]['projected_mde80_1024'] for k in keys))
    elapsed = perf_counter()-begin
    # Scale only row-dependent work; fixed update count independent of fit n.
    scenarios = {}
    for fraction in (.35, 1.):
        train_rows = sum(SIZES)*128*fraction; test_rows = 512*128*fraction
        seconds = (training['baseline_seconds']*train_rows/cut + training['raw_seconds']*len(SIZES)
                   + feature_s*(SIZES[-1]+512)*128*fraction/len(y) + held_s*len(SIZES)*test_rows/len(draw))
        scenarios[str(fraction)] = dict(estimated_local_seconds=seconds, threefold_planning_allowance=3*seconds)
    report = dict(raw_design_sha256=cm.digest(DESIGN), implementation_sha256=cm.digest(Path(__file__)),
                  synthetic=True, passed=passed, gradient_max_abs_error=grad, issued_score_max_parent_sum_error=api_error,
                  held_event_rate=float(draw.mean()), oracle_expected_held_gain_over_density_stratum=oracle,
                  fit=training, held=held, native_threads=1, planted_fit_parents=256, planted_held_parents=128,
                  seconds=dict(features_all=feature_s, held_scores=held_s, total=elapsed),
                  timing_projection_by_eligible_fraction=scenarios,
                  timing_note='Linear-in-eligible-row planning estimate for two three-fit baselines and fixed updates, not a worst-case bound; L-BFGS iterations/tree cost/cache differ on physical data.',
                  paid_calls=0)
    cm.receipt(path, report)
    print(json.dumps(dict(passed=passed, gradient=grad, api_error=api_error, contrasts={k:di[k] for k in keys}, timings=scenarios, total_seconds=elapsed)), flush=True)
    return passed


def read_bank(work, role, levels):
    if role not in ('fit', 'dev'):
        raise PermissionError('No assessment reader')
    with np.load(work/'rosters.npz', allow_pickle=False) as r:
        expected = r[role+'_parent']
    with np.load(work/(role+'.npz'), allow_pickle=False) as z:
        parent, shell, s, y = (z[k] for k in ('parent', 'shell', 'stratum', 'y'))
    size = {'fit':2048, 'dev':512}[role]
    require(parent.shape == (size,) and np.array_equal(parent, expected) and len(np.unique(parent)) == size,
            'New roster mismatch')
    require(y.shape == (size, 128) and s.shape == y.shape and shell.shape == (*y.shape, 49), 'Unexpected bank shape')
    require(y.dtype == np.uint8 and y.max() <= 1 and np.issubdtype(s.dtype, np.integer), 'Invalid label or stratum')
    keep = np.isin(s, levels); who, anchor = np.where(keep)
    supports = {n: support(y[:n], s[:n], keep[:n]) for n in (SIZES if role == 'fit' else (size,))}
    return shell[keep], s[keep], y[keep], who, anchor, parent, supports


def physical(work, role, design_sha):
    require(datetime.now(timezone.utc) < CUTOFF, 'Discovery closed')
    require(design_sha == cm.digest(DESIGN), 'Missing or incorrect registered raw design')
    ctrl = json.loads((work/'raw-control.json').read_text())
    require(ctrl['passed'] and ctrl['raw_design_sha256'] == design_sha and ctrl['implementation_sha256'] == cm.digest(Path(__file__)), 'No passing locked raw control')
    receipt = work/('raw-'+role+'.json'); require(not receipt.exists(), 'Role already issued')
    fit_receipt = None
    if role == 'dev':
        fit_receipt = json.loads((work/'raw-fit.json').read_text())
        require(fit_receipt['passed'] and fit_receipt['raw_design_sha256'] == design_sha, 'Missing fit receipt')
        for size in SIZES:
            require(cm.digest(work/f'raw-model-{size}.joblib') == fit_receipt['fits'][str(size)]['model_sha256'], 'Model changed')
    start = perf_counter(); levels = np.load(work/'eligible.npy', allow_pickle=False)
    require(levels.ndim == 1 and len(levels) and np.array_equal(levels, np.unique(levels)), 'Invalid frozen levels')
    shell, s, y, who, anchor, parents, supports = read_bank(work, role, levels)
    report = dict(raw_design_sha256=design_sha, implementation_sha256=cm.digest(Path(__file__)),
                  eligible_sha256=cm.digest(work/'eligible.npy'), rosters_sha256=cm.digest(work/'rosters.npz'),
                  bank_sha256=cm.digest(work/(role+'.npz')), support=supports, paid_calls=0, native_threads=1,
                  passed=all(v['passed'] for v in supports.values()))
    if not report['passed']:
        cm.receipt(receipt, report); print('Physical support gate failed', flush=True); return
    if role == 'dev':
        require(all(report[k] == fit_receipt[k] for k in ('eligible_sha256', 'rosters_sha256')), 'Eligibility or roster changed')
    now = perf_counter(); fs = inputs(shell, s); report['features_seconds'] = perf_counter()-now
    if role == 'fit':
        report['fits'] = {}
        for size in SIZES:
            cut = int(np.searchsorted(who, size)); sl = slice(cut)
            model, oof, record = fit_arrays(tuple(f[sl] for f in fs), s[sl], y[sl], parents[who[sl]], levels)
            off = work/f'raw-oof-{size}.npz'; state = work/f'raw-model-{size}.joblib'
            output_np(off, parent=parents[who[sl]], anchor=anchor[sl], stratum=s[sl], stratum_oof=oof[:, 0], packing_oof=oof[:, 1])
            with state.open('xb') as f:
                joblib.dump(model, f, compress=3)
            record.update(model_sha256=cm.digest(state), oof_sha256=cm.digest(off)); report['fits'][size] = record
    else:
        report['ladder'] = {}; evidence = dict(parent=parents, positive=np.bincount(who, weights=y, minlength=len(parents)))
        for size in SIZES:
            before = perf_counter(); model = load(work/f'raw-model-{size}.joblib')
            st, sums, n = evaluate(model, fs, s, y, who, len(parents))
            evidence['n'] = n
            for k, v in sums.items():
                evidence[f'log_score_sum_{size}_{k}'] = v
            st['score_seconds'] = perf_counter()-before; report['ladder'][size] = st
        pair = report['ladder'][2048]['paired']; comparisons = [pair['raw512-'+ref] for ref in ('packing','stratum')]
        report['escalation_gate_passed'] = bool(all(v['estimate'] >= MARGIN+v['projected_mde80_1024'] for v in comparisons) and comparisons[0]['projected_mde80_1024'] <= MARGIN)
        for k in ('stratum','packing','raw512', *(f'raw512_{seed}' for seed in SEEDS)):
            report.setdefault('paired_final_size_change', {})[k] = interval(evidence[f'log_score_sum_2048_{k}']-evidence[f'log_score_sum_512_{k}'], evidence['n'])
        report['incremental_size_change'] = interval((evidence['log_score_sum_2048_raw512']-evidence['log_score_sum_2048_packing'])-(evidence['log_score_sum_512_raw512']-evidence['log_score_sum_512_packing']), evidence['n'])
        path = work/'raw-dev-parents.npz'; output_np(path, **evidence); report['evidence_sha256'] = cm.digest(path)
    report['total_seconds'] = perf_counter()-start
    cm.receipt(receipt, report); print(receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('control','fit','dev')); parser.add_argument('--design-sha256')
    args = parser.parse_args(); WORK.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if args.role == 'control':
            if not control(WORK):
                raise SystemExit(1)
        else:
            physical(WORK, args.role, args.design_sha256)
