"""Fit, score, or issue a cold-locked R=4 shell-only vacancy-arrangement predictor."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

# Set before importing numerical libraries. threadpool_limits also covers already-loaded BLAS.
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import joblib
import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import t
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import boundary_templates
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l20_pairs import partition

SCRIPT = Path(__file__).resolve()
DESIGN = SCRIPT.parents[2] / 'results/l20-autoresearch/boundary/design.json'
SHELL = np.array([v for v in itertools.product(range(-4, 5), repeat=3) if max(map(abs, v)) == 4], dtype='i1')
SHELL_SHA = hashlib.sha256(SHELL.tobytes()).hexdigest()
MODEL = 'cold-models.joblib'
LOCK = 'cold-lock.json'
SEED = 210491
BATCH = 256
CAP = 12
MARGIN = .001


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, content):
    with path.open('x') as f:
        json.dump(content, f, indent=2, allow_nan=False)
        f.write('\n')


def provenance():
    return dict(implementation_sha256=sha(SCRIPT), templates_sha256=sha(boundary_templates.__file__),
                partition_sha256=sha(SCRIPT.parents[1] / 'l20_pairs.py'), design_sha256=sha(DESIGN),
                sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__, joblib=joblib.__version__)


def load_bank(path, mode, digest=None, lock=None):
    meta_path = path / 'meta.json'
    meta = json.loads(meta_path.read_bytes())
    index = meta['index']
    require(meta['mode'] == mode and index in (9, 8, 0), 'wrong bank role or index')
    groups = partition(index)[0]
    if mode == 'train':
        require(index == 9, 'fit is cold TRAIN only')
        split, ids = 'native_train', groups[0]
    elif mode == 'issue':
        require(index == 9, 'issue is cold blind only')
        split, ids = 'native_validation_last_1024', groups[1][1024:]
    elif index == 9:
        split, ids = 'native_validation_first_1024', groups[1][:1024]
    else:
        split, ids = 'native_validation', groups[1]
    require(meta['split'] == split and np.array_equal(meta['parent_ids'], ids), 'not the registered parent cohort')
    require(meta['shell_sha256'] == SHELL_SHA, 'shell geometry/order mismatch')
    sig = meta['signature']
    require(isinstance(sig, list) and len(sig) == 3 and all(type(v) is int and 0 <= v <= 5 for v in sig)
            and sorted(sig) == sig, 'invalid coordination signature')
    if lock is not None:
        require(sig == lock['signature'] and meta['lock_sha256'] == digest, 'bank not bound to this signature/lock')
        if index == 9:
            require(meta['source_sha256'] == lock['source_sha256'], 'changed cold release')
    require(sha(path / 'bank.npz') == meta['bank_sha256'], 'bank checksum failed')
    with np.load(path / 'bank.npz', allow_pickle=False) as z:
        keys = ('x', 'parent', 'anchor', 'weight')
        require(mode != 'issue' or 'y' not in z.files, 'issuance requires a label-blind bank')
        data = {k: z[k] for k in (*keys, *(('chain_seed', 'chain_id') if mode == 'issue' else ('y',)))}
    x, parent, anchor, weight = (data[k] for k in keys)
    n = len(parent)
    require(x.shape == (n, 49) and x.dtype == np.uint8 and not np.any(x[:, -1] & 252), 'not little-endian packed shells')
    require(all(data[k].shape == (n,) for k in data if k != 'x') and parent.dtype.kind in 'iu' and anchor.dtype.kind in 'iu', 'invalid bank vectors')
    require(np.isin(parent, ids).all() and (anchor < 8000).all() and (anchor >= 0).all(), 'invalid parent or anchor')
    require(np.isfinite(weight).all() and (weight > 0).all(), 'invalid center weights')
    if mode == 'issue':
        require(np.array_equal(np.sort(parent), np.sort(ids)) and (weight == 1).all(), 'must retain one blind anchor for every parent')
    else:
        require(np.isin(data['y'], [0, 1]).all(), 'invalid arrangement label')
    meta['meta_sha256'] = sha(meta_path)
    return meta, data, ids


# All geometric indices are fixed independently of the labels and the archive.
def geometry():
    lookup = {tuple(v): i for i, v in enumerate(SHELL)}
    radial = np.sort(abs(SHELL), axis=1)
    orbits, radial_idx = np.unique(radial, axis=0, return_inverse=True)
    radial_idx = radial_idx.ravel()
    radial_matrix = radial_idx[:, None] == np.arange(len(orbits))
    parity = ((SHELL % 2) @ np.array([4, 2, 1]))
    parity_matrix = np.eye(8)[parity]
    pairs2 = np.array(list(itertools.combinations(range(8), 2)))
    hamming = np.array([int(a ^ b).bit_count() for a, b in pairs2])
    template_count = parity_matrix.sum(axis=0)[pairs2].sum(axis=1)
    template_fraction = template_count / len(SHELL)
    # Unordered pairs binned by cubic displacement orbit. Include an actual antipodal statistic too.
    i, j = np.triu_indices(len(SHELL), 1)
    disp = np.sort(abs(SHELL[i] - SHELL[j]), axis=1)
    pair_tables = []
    for shape in ((0, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0, 2), (0, 1, 2), (0, 2, 2), (0, 0, 4), (0, 0, 8)):
        mask = (disp == shape).all(axis=1)
        pair_tables.append((i[mask], j[mask]))
    half = np.arange(len(SHELL)//2)
    pair_tables.append((half, len(SHELL)-1-half))
    frequencies, upto = [], []
    for period in (2, 4, 5):
        for v in ((0, 0, 1), (0, 1, 1), (1, 1, 1)):
            # One of each conjugate pair. Distinct vectors that alias at period 2 merely repeat its power.
            star = {tuple(c*s for c, s in zip(p, signs)) for p in itertools.permutations(v)
                    for signs in itertools.product((-1, 1), repeat=3)}
            frequencies.extend(np.asarray(k) / period for k in sorted(star) if k > tuple(-c for c in k))
            upto.append(len(frequencies))
    phase = 2*np.pi*SHELL @ np.asarray(frequencies).T
    fourier = np.concatenate((np.cos(phase), np.sin(phase)), axis=1)
    fourier -= fourier.mean(axis=0)
    # All complete face-tangent 3x3 windows. Integer codes are pooled modulo the eight square symmetries.
    grid = np.arange(9).reshape(3, 3)
    permutations = [np.rot90(g, k).ravel() for g in (grid, grid.T) for k in range(4)]
    bits = (np.arange(512)[:, None] >> np.arange(9)) & 1
    canon = np.minimum.reduce([bits[:, p] @ (1 << np.arange(9)) for p in permutations])
    _, patch_lut = np.unique(canon, return_inverse=True)
    windows = []
    for axis, normal, u, v in itertools.product(range(3), (-4, 4), range(-3, 4), range(-3, 4)):
        window = []
        for du, dv in itertools.product((-1, 0, 1), repeat=2):
            point = [u+du, v+dv]; point.insert(axis, normal)
            window.append(lookup[tuple(point)])
        windows.append(window)
    names = ['density', *['density_' + ''.join(map(str, r)) for r in orbits],
             *['connected_pair_' + k for k in ('001','011','111','002','012','022','004','008','antipodal')],
             *['crystal2_' + k for k in ('edge','face','body')],
             *['crystal20_' + k for k in ('best','mean','rms','gap')],
             *[f'power_{p}_{k}_{s}' for p in (2,4,5) for k in ('001','011','111') for s in ('mean','max')]]
    return (radial_matrix.astype('f'), parity_matrix.astype('f'), pairs2, hamming, template_fraction,
            pair_tables, fourier, upto, np.asarray(windows).T.copy(), patch_lut.ravel(), names)


RADIAL, PARITY, P2, HAM, FRAC, PAIRS, FOURIER, UPT0, WINDOWS, PATCH, NAMES = geometry()
D = len(NAMES)
K = int(PATCH.max()) + 1
RADIAL_SIZES = RADIAL.sum(axis=0)


def features(packed):
    x = np.unpackbits(packed, axis=1, count=len(SHELL), bitorder='little')
    z = x.astype('f'); n = len(z)
    density = z.mean(axis=1, dtype='d')
    cols = [density[:, None], (z @ RADIAL)/RADIAL_SIZES]
    local = []
    for i, j in PAIRS:
        a, b = x[:, i], x[:, j]
        local.append((a*b).mean(axis=1) - density*(a.mean(axis=1) + b.mean(axis=1)) + density**2)
    cols.append(np.column_stack(local))
    overlap = (z @ PARITY)[:, P2].sum(axis=2) / len(SHELL) - density[:, None]*FRAC
    scale = np.sqrt(np.maximum(density*(1-density), 1e-12)[:, None] * FRAC*(1-FRAC))
    cols.append(np.column_stack([(overlap[:, HAM == h] / scale[:, HAM == h]).max(axis=1) for h in (1,2,3)]))
    cols.append(boundary_templates.packing20(z))
    re, im = np.split(z @ FOURIER, 2, axis=1)
    power = np.round((re**2 + im**2) / len(SHELL), 10)
    for a, b in zip([0, *UPT0[:-1]], UPT0):
        cols += [power[:, a:b].mean(axis=1, keepdims=True), power[:, a:b].max(axis=1, keepdims=True)]
    codes = np.zeros((n, WINDOWS.shape[1]), dtype=np.int16)
    for bit, w in enumerate(WINDOWS):
        codes |= x[:, w].astype(np.int16) << bit
    bins = PATCH[codes] + K*np.arange(n)[:, None]
    cols.append(np.bincount(bins.ravel(), minlength=n*K).reshape(n, K) / WINDOWS.shape[1])
    return np.concatenate(cols, axis=1, dtype='f')


def train_rows(data, ids):
    # This second, label-independent cap is ONLY for fitting and tuning. Horvitz-Thompson adjustment
    # composes with the first (bank) cap. Scoring always retains the entire validation bank.
    p = data['parent']; order = np.lexsort((data['anchor'], p))
    rows = []; weights = []; weights0 = data['weight']
    for group in np.split(order, 1 + np.flatnonzero(np.diff(p[order]))):
        if not len(group):
            continue
        pid = int(p[group[0]])
        g = group if len(group) <= CAP else np.random.default_rng([SEED, pid]).choice(group, CAP, replace=False)
        rows.append(g); weights.append(weights0[g] * (len(group)/len(g)))
    require(bool(rows), 'no matched cold training centers')
    rows = np.concatenate(rows)
    tune_ids = np.random.default_rng(SEED).permutation(ids)[:len(ids)//5]
    tune = np.isin(p[rows], tune_ids)
    return rows, np.concatenate(weights).astype('d'), tune, tune_ids


def scale_fit(x, w):
    mu = np.average(x, weights=w, axis=0)
    var = np.average((x-mu.astype('f'))**2, weights=w, axis=0)
    return mu.astype('f'), np.maximum(np.sqrt(var), 1e-6).astype('f')


def standardized(x, fit):
    mu, sd = fit
    return np.clip((x-mu)/sd, -8., 8.)


def mlp_fit(x, y, w, ridge, bias):
    """Eight learned tanh channels, including supervised joint mixing of face-patch frequencies."""
    rng = np.random.default_rng(SEED)
    params = [rng.normal(0, .4/np.sqrt(x.shape[1]), (x.shape[1], 8)).astype('f'), np.zeros(8, 'f'),
              rng.normal(0, .02, 8).astype('f'), np.array(bias, dtype='f')]
    m = [np.zeros_like(v) for v in params]; v2 = [v.copy() for v in m]
    w = (w / w.mean()).astype('f'); step = 0
    for _ in range(16):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 512):
            r = order[start:start+512]; a = x[r]; W, b, c, d = params
            h = np.tanh(a @ W + b)
            delta = (expit(h @ c + d) - y[r]) * (w[r]/len(r))
            dh = (delta[:, None]*c) * (1-h*h)
            grads = [a.T @ dh + ridge*W, dh.sum(axis=0), h.T @ delta + ridge*c, delta.sum()]
            step += 1
            for j, grad in enumerate(grads):
                m[j] *= .9; m[j] += .1*grad
                v2[j] *= .999; v2[j] += .001*grad**2
                params[j] -= .002*(m[j]/(1-.9**step))/(np.sqrt(v2[j]/(1-.999**step)) + 1e-8)
    return params


def predict(model, raw, standardized_x):
    typ = model['kind']
    if typ == 'constant':
        p = np.full(len(raw), model['p'])
    elif typ == 'mlp':
        W, b, c, d = model['params']
        p = expit(np.tanh(standardized_x @ W + b) @ c + d)
    else:
        z = raw if typ == 'tree' else standardized_x
        p = model['estimator'].predict_proba(z[:, model['columns']])[:, 1]
    p = np.clip(np.asarray(p, dtype='d'), 1e-7, 1-1e-7)
    require(np.isfinite(p).all(), 'nonfinite model probability')
    return p


def logscore(y, p):
    return np.where(y == 1, np.log(p), np.log1p(-p))


def fit(a):
    require(a.lock_sha256 is None, 'fit creates the cold lock; no input lock accepted')
    require(not (a.out/LOCK).exists() and not (a.out/MODEL).exists(), 'cold fit already exists')
    m, data, ids = load_bank(a.bank, 'train')
    rows, weights, tune, tune_ids = train_rows(data, ids)
    y = data['y'][rows].astype('f'); f = ~tune
    require(all(len(np.unique(y[mask])) == 2 for mask in (f, tune)), 'both held-in and tuning parents must cover both classes')
    x = np.empty((len(rows), D+K), dtype='f')
    for j in range(0, len(rows), BATCH):
        x[j:j+BATCH] = features(data['x'][rows[j:j+BATCH]])
    scaler = scale_fit(x[f], weights[f]); z = standardized(x, scaler)
    w = weights[f] / weights[f].mean()
    base = float((np.dot(w, y[f])+.5)/(w.sum()+1))
    models = {'intercept': dict(kind='constant', p=base)}
    sets = {'density': [0], 'pair': [i for i,n in enumerate(NAMES) if n.startswith(('density','connected'))],
            'templates': [i for i,n in enumerate(NAMES) if n.startswith('crystal')],
            'fourier': [i for i,n in enumerate(NAMES) if n.startswith('power')], 'packing_linear': list(range(D))}
    for key, cols in sets.items():
        lr = LogisticRegression(C=.1, max_iter=300).fit(z[f][:, cols], y[f], sample_weight=w)
        models[key] = dict(kind='linear', columns=cols, estimator=lr)
    for leaves in (7, 15):
        tree = HistGradientBoostingClassifier(max_leaf_nodes=leaves, min_samples_leaf=64, l2_regularization=5.,
                learning_rate=.05, max_iter=100, max_bins=63, early_stopping=False, random_state=SEED)
        tree.fit(x[f, :D], y[f], sample_weight=w)
        models[f'packing_tree_{leaves}'] = dict(kind='tree', columns=list(range(D)), estimator=tree)
    ordinary = list(models)
    for ridge in (.001, .02):
        models[f'patch_mlp_{ridge}'] = dict(kind='mlp', params=mlp_fit(z[f], y[f], w, ridge, np.log(base/(1-base))))
    tune_x, tune_z = x[tune], z[tune]
    scores = {key: float(np.average(logscore(y[tune], predict(model, tune_x, tune_z)), weights=weights[tune])) for key,model in models.items()}
    best = lambda keys: min(keys, key=lambda k: (-scores[k], k))
    comparator, learned = best(ordinary), best([k for k in models if k not in ordinary])
    kept = {k: models[k] for k in (*ordinary, learned)}
    joblib.dump(dict(models=kept, scaler=scaler), a.out/MODEL, compress=3)
    write_json(a.out/LOCK, dict(signature=m['signature'], index=9, source_sha256=m['source_sha256'],
              train_bank_sha256=m['bank_sha256'], train_meta_sha256=m['meta_sha256'], shell_sha256=SHELL_SHA,
              model_sha256=sha(a.out/MODEL), provenance=provenance(), strongest_baseline=comparator, learned_model=learned,
              valid='Both classes present on both TRAIN partitions; all tuning probabilities finite, clipped to [1e-7,1-1e-7].',
              tune_parent_ids_sha256=hashlib.sha256(tune_ids.astype('<u4').tobytes()).hexdigest(),
              n_fit_parents=len(ids)-len(tune_ids), n_tune_parents=len(tune_ids), n_fit=int(f.sum()), n_tune=int(tune.sum()),
              feature_names=NAMES, patch_orbits=K, tuning_mean_log_score=scores,
              primary='Index 9 first 1024 native validation parents; lower paired two-sided 95% whole-parent log-score gain >= 0.001 nat per eligible center over cold-selected strongest conventional classifier.',
              margin_nat=MARGIN, transfer='Indices 8 and 0 descriptive only, unchanged models; no rescue.',
              issue='Last 1024 native cold validation parents; probabilities conditional on either eligible matched arrangement, not replica recurrence.'))
    print(f'{LOCK} SHA256: {sha(a.out/LOCK)}')


def ratio_interval(numerators, denominators):
    """Ratio of sums, finite-cluster sandwich SE. Zero-mass parents remain in the sample."""
    n = len(denominators); den = float(denominators.sum())
    if den == 0:
        return dict(estimate=None, se=None, lo=None, hi=None)
    est = float(numerators.sum()/den)
    if n < 2 or np.count_nonzero(denominators) < 2:
        return dict(estimate=est, se=None, lo=None, hi=None)
    r = numerators-est*denominators
    se = float(np.sqrt(n/(n-1)*np.dot(r,r))/den); h = float(t.ppf(.975, n-1)*se)
    return dict(estimate=est, se=se, lo=est-h, hi=est+h)


def swapped(y, w, pos, ids, p, p0):
    """Independent disjoint parent pairs. Integrate over donor anchors, not one noisy shuffle draw."""
    def agg(x):
        return np.bincount(pos, weights=w*x, minlength=len(ids))
    den = agg(1); n1 = agg(y); n0 = den-n1
    pair = np.random.default_rng(SEED+1).permutation(len(ids)).reshape(-1, 2)
    donor = np.empty(len(ids), dtype=int); donor[pair.ravel()] = pair[:, ::-1].ravel()
    usable = (den > 0) & (den[donor] > 0)
    def swap_score(prob):
        a = agg(np.log(prob))/np.maximum(den, 1e-30)
        b = agg(np.log1p(-prob))/np.maximum(den, 1e-30)
        return n1*a[donor] + n0*b[donor]
    def compare(v):
        return ratio_interval((usable*v)[pair].sum(axis=1), (usable*den)[pair].sum(axis=1))
    swapped_learned = swap_score(p)
    return dict(disjoint_pairs=len(pair), usable_pairs=int(usable.sum()//2), usable_weight=float(den[usable].sum()),
                actual_minus_swapped_learned=compare(agg(logscore(y,p))-swapped_learned),
                swapped_learned_minus_intercept=compare(swapped_learned-agg(logscore(y,p0))))


def score(a, m, data, ids, lock, predictions):
    y, w = data['y'], data['weight'].astype('d')
    pos_map = np.full(16384, -1); pos_map[ids] = np.arange(len(ids)); pos = pos_map[data['parent']]
    agg = lambda v: np.bincount(pos, weights=w*v, minlength=len(ids)).astype('d')
    den = agg(1); total = den.sum(); n1 = agg(y)
    model_scores = {}; parent_logscore = {}; means = {}
    for key, p in predictions.items():
        s = parent_logscore[key] = agg(logscore(y,p)); means[key] = agg(p)
        model_scores[key] = dict(log_score=ratio_interval(s, den), calibration_bias_predicted_minus_observed=ratio_interval(means[key]-n1, den),
            auc=float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else None)
    l, b = lock['learned_model'], lock['strongest_baseline']
    compare = lambda k, j: ratio_interval(parent_logscore[k]-parent_logscore[j], den)
    primary = compare(l,b); index = m['index']
    npz = a.out / f'parents-{index:02}.npz'
    with npz.open('xb') as f:
        np.savez_compressed(f, parent=ids, eligible_weight=den, tripod_weight=n1,
               **{'log_score_sum_'+k: s for k,s in parent_logscore.items()}, **{'probability_sum_'+k: v for k,v in means.items()})
    result = dict(index=index, signature=m['signature'], lock_sha256=a.lock_sha256, bank_sha256=m['bank_sha256'], bank_meta_sha256=m['meta_sha256'],
        parents_sha256=sha(npz), independent_parents=len(ids), empty_parents=int((den == 0).sum()), eligible_weight=float(total), sampled_centers=len(y),
        observed_tripod=ratio_interval(n1,den), strongest_baseline=b, learned_model=l, models=model_scores,
        learned_minus_strongest=primary, learned_minus_intercept=compare(l,'intercept'), strongest_minus_intercept=compare(b,'intercept'),
        cross_parent_null=swapped(y,w,pos,ids,predictions[l],predictions['intercept']), margin_nat=MARGIN,
        assessment=('descriptive_transfer' if index != 9 else 'inconclusive' if primary['lo'] is None else 'pass' if primary['lo'] >= MARGIN else 'did_not_pass'))
    write_json(a.out/f'score-{index:02}.json', result)
    print(f'index={index}: {result["assessment"]}; learned minus {b}: {primary}')


def issue(a, m, data, lock, predictions):
    predfile = a.out/'issue-09.csv'
    l, b = lock['learned_model'], lock['strongest_baseline']
    with predfile.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('parent','chain_seed','chain_id','anchor','packed_shell_sha256','p_tripod','comparator_p_tripod','predicted_arrangement','comparator_arrangement'))
        def arrangement(v):
            return 'tripod' if v >= .5 else 'opposite_pair_plus_lateral'
        for j,(p,q) in enumerate(zip(predictions[l], predictions[b])):
            identity = [int(data[k][j]) for k in ('parent','chain_seed','chain_id','anchor')]
            writer.writerow((*identity, hashlib.sha256(data['x'][j].tobytes()).hexdigest(), repr(float(p)), repr(float(q)), arrangement(p), arrangement(q)))
    write_json(a.out/'issue-09.json', dict(index=9, signature=m['signature'], lock_sha256=a.lock_sha256, bank_sha256=m['bank_sha256'],
              bank_meta_sha256=m['meta_sha256'], learned_model=l, strongest_baseline=b, rows=len(data['x']), issue_sha256=sha(predfile),
              estimand='P(vacancy tripod | either matched arrangement, shell). One center-blind anchor per parent; not an eligibility or independent replica recurrence probability.'))


def evaluate(a):
    path = a.out/LOCK
    require(a.lock_sha256 and sha(path) == a.lock_sha256, 'published --lock-sha256 required')
    lock = json.loads(path.read_bytes())
    require(lock['provenance'] == provenance() and lock['model_sha256'] == sha(a.out/MODEL), 'locked code, design, environment or model changed')
    mode = 'issue' if a.job == 'issue' else 'validation'
    m, data, ids = load_bank(a.bank, mode, a.lock_sha256, lock)
    if m['index'] != 9:
        cold = a.out/'score-09.json'
        require(cold.exists() and json.loads(cold.read_bytes())['lock_sha256'] == a.lock_sha256, 'assess cold before descriptive transfer')
    dest = 'issue-09' if mode == 'issue' else f'score-{m["index"]:02}'
    require(not (a.out/(dest+'.json')).exists(), 'already scored/issued')
    # Trusted local serialization, checksum-verified against the explicitly pinned lock before loading.
    bundle = joblib.load(a.out/MODEL)
    probs = {k: np.empty(len(data['x'])) for k in bundle['models']}
    for j in range(0, len(data['x']), BATCH):
        sl = slice(j, j+BATCH); x = features(data['x'][sl]); z = standardized(x, bundle['scaler'])
        for k, model in bundle['models'].items():
            probs[k][sl] = predict(model, x, z)
    if mode == 'issue':
        issue(a, m, data, lock, probs)
    else:
        score(a, m, data, ids, lock, probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=('fit','score','issue'))
    parser.add_argument('--bank', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--lock-sha256')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        (fit if args.job == 'fit' else evaluate)(args)
