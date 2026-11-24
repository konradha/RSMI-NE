"""Conditional predictive value of a frozen two-bit symbol beyond the full cage forecast.

From the repository root:
  python examples/glass_autoresearch/innovation.py cold
  # Publish innovation/cold-lock.json before opening transfer data.
  python examples/glass_autoresearch/innovation.py transfer --lock-sha256 PUBLISHED_SHA256
See results/l20-autoresearch/innovation/design.json. No test mode or new configurations.
"""
import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm, t
from threadpoolctl import threadpool_limits
import joblib
import sklearn

import cage_baseline as cage
import motif
import rate

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/l20-autoresearch'
OUT = RESULTS / 'innovation'
DESIGN = OUT / 'design.json'
BUNDLE = RESULTS / 'mechanism/conventional.joblib'
PRIVATE = Path.home() / 'rsmi-work/l20-autoresearch-1795520725'
ARCHIVE = Path.home() / 'datasets/lattice-glass/l20-16384-v1'
TRIPODS = tuple(f'{prefix}{sp}' for prefix in ('tripod_', 'A_gated_tripod_') for sp in ('vacancy', 'A', 'B'))
ORDINARY = ('two_A_bits', *TRIPODS, 'p_quartiles')
LEARNED = ('posterior', 'hard')
ALL = (*ORDINARY, *LEARNED)
RIDGE = 1.e-5
PLANT = np.array([-.75, -.25, .25, .75])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).hexdigest()


def pinned():
    d = rate.read_json(DESIGN)
    for name, sha in d['fixed_inputs_sha256'].items():
        require(rate.digest(ROOT / name) == sha, 'changed fixed input: ' + name)
    with zipfile.ZipFile(motif.STATIC / 'fit-receipts.zip') as z:
        for name, sha in d['teacher_members_sha256'].items():
            require(hashlib.sha256(z.read(name)).hexdigest() == sha, 'changed teacher: ' + name)
    require(sklearn.__version__ == d['bundle_contract']['sklearn_version'], 'use the pinned scikit-learn version')
    bundle = joblib.load(BUNDLE)  # only after checking the immutable public digest
    report = rate.read_json(RESULTS / 'mechanism/conventional-fits.json')
    for name, value in d['bundle_contract'].items():
        require(bundle[name] == report[name] == value, 'wrong conventional model: ' + name)
    cage.geometry()
    return d, bundle, report


def cold_bank(work, fold, report, bundle):
    path = work / 'mechanism' / fold
    wanted = report['banks'][fold]
    require(rate.digest(path / 'meta.json') == wanted['meta_sha256'], 'cold bank metadata changed')
    for name, sha in wanted['sha256'].items():
        require(rate.digest(path / name) == sha, f'changed cold {fold} bank: {name}')
    meta, features = cage.open_bank(path, (9, fold))
    ids = cage.mmap(path, 'parent_id')
    keys = cage.mmap(path, 'parent_keys')
    f = cage.FOLDS.index(fold)
    require(np.array_equal(ids, cage.partition(9)[0][f]) and len(np.unique(keys, axis=0)) == len(ids),
            'cold parent partition or lineage mismatch')
    cut = meta['fit_parents'] if fold == 'train' else 0
    if fold == 'train':
        reserve = report['reserved_calibration']
        require(cut == 7680 and len(ids) - cut == 2560 and
                sha_array(ids[cut:].astype('<u8')) == reserve['calibration_row_id_sha256'] and
                sha_array(ids[:cut].astype('<u8')) == reserve['fit_row_id_sha256'], 'bad reserved parent suffix')
    # Evaluate only the rows permitted to enter this analysis. The same pinned
    # routine supplies the unrounded full-blend forecast for both cold and transfer.
    pred = cage.probabilities({k: v[cut:] for k, v in features.items()}, bundle['models'])[bundle['full_member']]
    print('scored frozen conventional model:', fold, pred.shape, flush=True)
    info = dict(index=9, beta=meta['beta'], fold=fold, source_sha256=meta['source_sha256'],
                receipt_sha256=meta['receipt_sha256'], bank_meta_sha256=wanted['meta_sha256'],
                bank_array_sha256=wanted['sha256'], origin_seed=meta['origin_seed'],
                original_parent_start=cut, selected_lineage_sha256=sha_array(keys[cut:].astype('<u8')))
    return cohort(features['v'][cut:], pred, cage.mmap(path, 'y')[cut:], ids[cut:], info)


def cohort(v, p, count, ids, info):
    P, R, n = v.shape
    require(n == 27 and p.shape == count.shape == (P, R) and len(np.unique(ids)) == P and
            np.all((p > 0) & (p < 1)) and count.dtype == np.uint8 and count.max() <= 3, 'invalid anchor bank')
    flat = v.reshape(-1, 27)
    pi = motif.teacher(flat, 11).astype(np.float64)
    require(pi.shape == (P * R, 4) and np.all(pi >= 0) and np.allclose(pi.sum(1), 1., rtol=1.e-6),
            'invalid frozen four-symbol channel')
    pi /= pi.sum(1, keepdims=True)  # correct float32 roundoff only
    codes = {'posterior': pi, 'hard': pi.argmax(1).astype('u1'),
             'two_A_bits': (2 * (flat[:, 0] == 1) + (flat[:, 26] == 1)).astype('u1')}
    tripod = flat[:, (9, 3, 1)]
    for sp, name in enumerate(('vacancy', 'A', 'B')):
        c = (tripod == sp).sum(1, dtype='u1')
        codes['tripod_' + name] = c
        codes['A_gated_tripod_' + name] = c * (flat[:, 0] == 1)
    info.update(parents=P, distinct_parent_ids=P, reps=R, anchor_count=P * R, outward_sites=3 * P * R,
                parent_ids_le_u8_sha256=sha_array(ids.astype('<u8')), full_probability_f64_sha256=sha_array(p.astype('<f8', copy=False)))
    return dict(meta=info, ids=ids, shape=(P, R), x=logit(p.ravel()), y=count.ravel() / 3., codes=codes)


def transfer_bank(data, temp, idx, d, bundle):
    require(idx in (8, 0), 'validation transfer indices only')
    source = d['transfer_sources'][str(idx)]
    receipt = data / f'beta-{idx:02}.json'
    require(rate.digest(receipt) == source['receipt_sha256'] and
            rate.read_json(receipt)['endpoint_sha256'] == source['source_sha256'], 'wrong transfer receipt')
    (lookup, states), info = rate.source_rows(data, idx, (1,))
    ids = rate.partition(idx)[0][1]
    selected = lookup[ids]
    require(info['beta'] == source['beta'] and info['sha256'] == source['source_sha256'] and
            info['selected_folds'] == [1] and len(states) == len(ids) == 2048 and
            np.array_equal(np.sort(selected), np.arange(len(ids))), 'invalid released validation rows')
    P, R = len(ids), 192
    seed = [cage.SEED, idx, 1]
    origins = np.random.default_rng(np.random.SeedSequence(seed)).integers(cage.L**3, size=(R, P), dtype=np.uint16).T
    # Match legacy dtype, shape, axis order, origin stream and per-parent feature checks.
    features = {name: np.lib.format.open_memmap(temp / (name + '.npy'), mode='w+', dtype=dtype, shape=(P, R, width))
                for name, width, dtype in [('v', 27, 'u1'), ('m', 27, 'u1'), ('bonds', 54, '<i2')]}
    y = np.lib.format.open_memmap(temp / 'y.npy', mode='w+', dtype='u1', shape=(P, R))
    halo, outward = cage.address(cage.H2), cage.address(cage.TARGET)
    step = max(1, 2048 // R)
    for i in range(0, P, step):
        sl = slice(i, min(P, i + step)); centers = origins[sl]; rows = selected[sl, None, None]
        s = states[rows, halo[centers]].reshape(-1, len(cage.H2))
        for name, values in zip(('v', 'm', 'bonds'), cage.extract(s, R)):
            features[name][sl] = values.reshape(len(centers), R, -1)
        y[sl] = (states[rows, outward[centers]] == 1).sum(-1)
    for arr in (*features.values(), y):
        arr.flush()
    np.save(temp / 'origins.npy', origins, allow_pickle=False)
    np.save(temp / 'parent_id.npy', ids, allow_pickle=False)
    meta = dict(index=idx, beta=info['beta'], fold='validation', source_sha256=info['sha256'],
                receipt_sha256=source['receipt_sha256'], selected_lineage_sha256=info['lineage_sha256']['1'],
                origin_seed=seed, partition_seed=cage.PAIR_SEED, bank_version=cage.BANK_VERSION,
                bank_array_sha256={f.name: rate.digest(f) for f in sorted(temp.glob('*.npy'))},
                parent_order='l20_pairs.partition(index)[0][1]', selected_folds=[1],
                feature_geometry='cage_baseline.H2/V/TARGET, positive internal bonds, equal-label sentinel 32767')
    p = cage.probabilities(features, bundle['models'])[bundle['full_member']]
    return cohort(features['v'], p, y, ids, meta)


def add_quartiles(bank, edges):
    bank['codes']['p_quartiles'] = np.searchsorted(edges, bank['x'], side='right').astype('u1')


def offsets(w):
    return np.r_[w[2:], -sum(w[2:])]


def fit(x, y, h, center):
    xc = x - center; n = len(y)
    def objective(w):
        z = x + w[0] + w[1] * xc
        if h is None:
            r = expit(z) - y
            ll = (np.logaddexp(0., z) - y * z).mean()
            gc = np.empty(0); norm2 = 0.
        else:
            d = offsets(w)
            if h.ndim == 1:
                z += d[h]
                r = expit(z) - y
                ll = (np.logaddexp(0., z) - y * z).mean()
                g = np.bincount(h, weights=r, minlength=4) / n
            else:
                z = z[:, None] + d
                residual = h * (expit(z) - y[:, None]); r = residual.sum(1)
                ll = np.sum(h * (np.logaddexp(0., z) - y[:, None] * z)) / n
                g = residual.mean(0)
            g += RIDGE * d; gc = g[:3] - g[3]; norm2 = d @ d
        return (ll + .5 * RIDGE * (w[1]**2 + norm2), np.r_[r.mean(), r @ xc / n + RIDGE * w[1], gc])
    out = minimize(objective, np.zeros(2 if h is None else 5), method='L-BFGS-B', jac=True,
                   options={'maxiter': 250, 'gtol': 1.e-9, 'ftol': 1.e-15})
    require(out.success and np.isfinite(out.fun) and np.max(abs(out.jac)) < 1.e-6, 'calibration failed: ' + str(out.message))
    return dict(weights=out.x.tolist(), intercept=float(out.x[0]), total_logit_slope=float(1 + out.x[1]),
                category_effects=(offsets(out.x).tolist() if h is not None else None),
                iterations=int(out.nit), objective=float(out.fun), max_abs_gradient=float(np.max(abs(out.jac))))


def score(x, y, h, model, center, shape):
    w = model['weights']; z = x + w[0] + w[1] * (x - center)
    if h is None or h.ndim == 1:
        if h is not None:
            z += np.array(model['category_effects'])[h]
        ll = y * z - np.logaddexp(0., z)
    else:
        z = z[:, None] + np.array(model['category_effects'])
        ll = (h * (y[:, None] * z - np.logaddexp(0., z))).sum(1)
    return ll.reshape(shape).mean(axis=1)


def interval(values):
    n = len(values); mu = float(np.mean(values)); se = float(np.std(values, ddof=1) / np.sqrt(n))
    c, b = t.ppf([.975, .95], n - 1) * se
    return dict(mean=mu, se=se, lo95=mu-c, hi95=mu+c, lower_one95=mu-b, upper_one95=mu+b,
                mde80_one_sided=float(b + norm.ppf(.8) * se), independent_units=n)


def pairing(P, idx):
    require(P % 2 == 0, 'shuffle needs disjoint parent pairs')
    pairs = np.random.default_rng([702813, idx]).permutation(P).reshape(-1, 2)
    donor = np.empty(P, int); donor[pairs] = pairs[:, ::-1]
    return pairs, donor


def evaluate(bank, models, center, y=None):
    x = bank['x']; y = bank['y'] if y is None else y
    shape = bank['shape']; P, R = shape
    pairs, donor = pairing(P, bank['meta']['index'])
    scores = {}; summaries = {}; shuffled = {}; rows = {'parent_id': bank['ids'], 'shuffle_donor_parent_id': bank['ids'][donor]}
    for arm, model in models.items():
        h = bank['codes'].get(arm) if arm != 'offset' else None
        s = score(x, y, h, model, center, shape); scores[arm] = s; rows['score_' + arm] = s
        summary = dict(expected_log_score=interval(s))
        if arm != 'offset':
            hs = h.reshape(P, R, *h.shape[1:])[donor].reshape(h.shape)
            ss = score(x, y, hs, model, center, shape); rows['shuffled_' + arm] = ss
            summary['vs_offset'] = interval(s - scores['offset'])
            mass = np.bincount(h, minlength=4) / len(h) if h.ndim == 1 else h.mean(0)
            summary['symbol_mass'] = mass.tolist()
            shuffled[arm] = dict(shuffled_expected_score=interval(ss[pairs].mean(1)),
                                 shuffled_vs_offset=interval((ss - scores['offset'])[pairs].mean(1)),
                                 matched_vs_shuffled=interval((s - ss)[pairs].mean(1)))
        summaries[arm] = summary
    return scores, dict(cohort=bank['meta'], arms=summaries, shuffled_parent_pairs=shuffled), rows


def choose(scores):
    def best(names):
        return max(names, key=lambda k: scores[k].mean())
    return dict(learned=best(LEARNED), tripod=best(TRIPODS), ordinary=best(ORDINARY), any_code=best(ALL))


def contrasts(scores, sel):
    a = sel['learned']; o = sel['ordinary']
    def diff(i, j):
        return interval(scores[i] - scores[j])
    return dict(any_vs_offset=diff(sel['any_code'], 'offset'), learned_vs_offset=diff(a, 'offset'),
                learned_vs_locked_ordinary=diff(a, o), ordinary_vs_offset=diff(o, 'offset'),
                posterior_vs_locked_ordinary=diff('posterior', o), hard_vs_locked_ordinary=diff('hard', o),
                posterior_vs_hard=diff('posterior', 'hard'),
                learned_vs_each_ordinary={k: diff(a, k) for k in ORDINARY})


def planted(cal, val, center):
    def labels(b, seed):
        true = expit(b['x'] + PLANT[b['codes']['hard']]); return np.random.default_rng(seed).binomial(3, true) / 3.
    y = labels(cal, 445171)
    models = {k: fit(cal['x'], y, cal['codes'].get(k), center) for k in ('offset', 'hard', 'posterior')}
    _, summary, _ = evaluate(val, models, center, labels(val, 445172))
    gain = summary['arms']['hard']['vs_offset']; shuffled = summary['shuffled_parent_pairs']['hard']['matched_vs_shuffled']
    return dict(planted_class_logits=PLANT.tolist(), calibration_seed=445171, validation_seed=445172,
                passed=bool(min(gain['lower_one95'], shuffled['lower_one95']) > 0), models=models,
                validation_arms=summary['arms'], shuffled_parent_pairs=summary['shuffled_parent_pairs'])


def cold(a, d, report, bundle):
    lockpath = OUT / 'cold-lock.json'; parentpath = OUT / 'cold-parent-scores.npz'
    if lockpath.exists() or parentpath.exists():
        raise FileExistsError('refusing to replace cold selection or parent results')
    cal = cold_bank(a.work, 'train', report, bundle); val = cold_bank(a.work, 'validation', report, bundle)
    require(not np.intersect1d(cal['ids'], val['ids']).size, 'shared cold parent IDs')
    edges = np.quantile(cal['x'], [.25, .50, .75]); center = float(cal['x'].mean())
    require(np.all(np.diff(edges) > 0), 'offset quartiles not resolved')
    for bank in (cal, val):
        add_quartiles(bank, edges)
    models = {k: fit(cal['x'], cal['y'], cal['codes'].get(k), center) for k in ('offset', *ALL)}
    sc, summary, rows = evaluate(val, models, center); sel = choose(sc); cc = contrasts(sc, sel)
    delta = d['physical_delta_nats_per_site']; control = planted(cal, val, center)
    gate = dict(any_code=bool(cc['any_vs_offset']['mean'] >= delta),
                learned_advantage=bool(min(cc['learned_vs_offset']['mean'], cc['learned_vs_locked_ordinary']['mean']) >= delta))
    rate.arrays(parentpath, **rows, calibration_parent_ids=cal['ids'])
    lock = dict(version=1, design_sha256=rate.digest(DESIGN), implementation_sha256=rate.digest(Path(__file__)),
                inputs=d['fixed_inputs_sha256'], teacher_members=d['teacher_members_sha256'], transfer_sources=d['transfer_sources'],
                bundle=d['bundle_contract'], calibration=cal['meta'], calibration_center_logit=center,
                quartile_cut_logits=edges.tolist(), models=models, selected=sel, cold_gates=gate,
                delta_nats_per_outward_site=delta, cold_validation=summary, cold_contrasts=cc,
                planted_control=control, cold_parent_scores_sha256=rate.digest(parentpath),
                score='expected log probability in nats per outward site; larger is better',
                claim='One designated warm validation transfer (index 8), hot (index 0) descriptive; not a test-set or CMI estimate.')
    rate.json_out(lockpath, lock)
    print(f'Publish {lockpath} and parent scores before transfer. SHA-256: {rate.digest(lockpath)}', flush=True)
    print('Cold selections:', sel, 'gates:', gate, 'planted-control pass:', control['passed'], flush=True)


def decision(gate, control, estimands, delta):
    if not control:
        return 'calibration control failed; no positive conclusion'
    if not gate:
        return 'cold selection gate closed; descriptive transfer only'
    if min(x['lower_one95'] for x in estimands) > delta:
        return 'supports locked magnitude'
    # The upper bound for the fixed cold-selected comparison need not be adjusted
    # for a worst-looking comparator selected AFTER transfer. Use that fixed pair only.
    if estimands[0]['upper_one95'] < delta:
        return 'excludes locked magnitude on the cold-selected comparison'
    return 'unresolved at the locked magnitude'


def plot(lock, transfer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    names = ['Cold selection', 'Warm transfer', 'Hot diagnostic']
    cc = [lock['cold_contrasts'], *(r['contrasts'] for r in transfer)]
    for ax, key, title in zip(axes, ('any_vs_offset', 'learned_vs_locked_ordinary'),
                              ('Best cold code vs recalibrated p', 'Frozen learned vs best cold ordinary')):
        for i, (r, color) in enumerate(zip(cc, ('#545f71', '#087f83', '#b87540'))):
            u = r[key]; mu = u['mean']
            ax.errorbar(i, mu * 1e6, yerr=1e6 * np.array([[mu-u['lo95']], [u['hi95']-mu]]),
                        color=color, fmt='o', capsize=4)
        ax.axhline(0, color='.5', lw=.8); ax.axhline(1e6*lock['delta_nats_per_outward_site'], color='#b24f39', ls='--', label='Locked magnitude')
        ax.set(xticks=range(3), xticklabels=names, title=title, ylabel='Gain (µnats / outward site)')
        ax.tick_params(axis='x', labelrotation=15); ax.legend(frameon=False, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Four-symbol predictive correction · paired parent 95% intervals')
    fig.savefig(OUT / 'transfer.png', dpi=165); plt.close(fig)


def transfer(a, d, bundle):
    lockpath = OUT / 'cold-lock.json'
    require(rate.digest(lockpath) == a.lock_sha256, 'provide the exact public cold lock SHA-256')
    lock = rate.read_json(lockpath)
    require(lock['design_sha256'] == rate.digest(DESIGN) and lock['implementation_sha256'] == rate.digest(Path(__file__))
            and lock['inputs'] == d['fixed_inputs_sha256'] and lock['transfer_sources'] == d['transfer_sources']
            and lock['cold_parent_scores_sha256'] == rate.digest(OUT / 'cold-parent-scores.npz'), 'changed since public lock')
    if any((OUT / name).exists() for name in ('transfer.json', 'transfer.png', 'transfer-parent-scores.npz')):
        raise FileExistsError('refusing to replace transfer results')
    tempbase = a.work / 'innovation'; tempbase.mkdir(mode=0o700, parents=True, exist_ok=True)
    results = []; allrows = {}; gate = lock['cold_gates']; ctrl = lock['planted_control']['passed']; delta = lock['delta_nats_per_outward_site']
    for idx in (8, 0):
        with TemporaryDirectory(prefix=f'validation-{idx:02}-', dir=tempbase) as tmp:
            bank = transfer_bank(a.data, Path(tmp), idx, d, bundle)
        add_quartiles(bank, lock['quartile_cut_logits'])
        scores, summary, rows = evaluate(bank, lock['models'], lock['calibration_center_logit'])
        cc = contrasts(scores, lock['selected']); summary['contrasts'] = cc
        if idx == 8:
            summary['decisions'] = dict(any_code=decision(gate['any_code'], ctrl, [cc['any_vs_offset']], delta),
                learned_advantage=decision(gate['learned_advantage'], ctrl, [cc['learned_vs_locked_ordinary'],
                    cc['learned_vs_offset'], *cc['learned_vs_each_ordinary'].values()], delta))
        else:
            summary['decisions'] = 'hot specificity diagnostic only'
        results.append(summary); allrows.update({f'b{idx:02}_{k}': v for k, v in rows.items()})
    parentpath = OUT / 'transfer-parent-scores.npz'
    rate.arrays(parentpath, **allrows); plot(lock, results)
    rate.json_out(OUT / 'transfer.json', dict(lock_sha256=a.lock_sha256, selected=lock['selected'], cold_gates=gate,
        delta_nats_per_outward_site=delta, parent_scores_sha256=rate.digest(parentpath), figure_sha256=rate.digest(OUT / 'transfer.png'),
        results=results, interpretation='Validation-only conditional predictive scores for one frozen offset. Not CMI, dynamics, a new test set, or a reversal of the primary negative result.'))
    print('Warm:', results[0]['decisions'], 'See', OUT / 'transfer.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='mode', required=True)
    for mode in ('cold', 'transfer'):
        p = commands.add_parser(mode); p.add_argument('--work', type=Path, default=PRIVATE)
        if mode == 'transfer':
            p.add_argument('--data', type=Path, default=ARCHIVE); p.add_argument('--lock-sha256', required=True)
    a = parser.parse_args()
    # Nested OpenMP inference and BLAS spin waits can overwhelm tiny chunks.
    # This scalar resource bound changes no model, observations or objective.
    with threadpool_limits(limits=1):
        d, bundle, report = pinned()
        if a.mode == 'cold':
            cold(a, d, report, bundle)
        else:
            transfer(a, d, bundle)


if __name__ == '__main__':
    main()
