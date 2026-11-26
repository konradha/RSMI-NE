"""Source-only physical readout fits and admission-locked, whole-parent history scoring.

No TensorFlow training or sampling: the six face targets are simultaneous observations.
Run with OPENBLAS_NUM_THREADS=1; six independent faces run concurrently. See results/history-overlap-field/design.md.
"""
import argparse
import csv
import hashlib
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy


OUT = Path(__file__).resolve().parents[1] / 'results/history-overlap-field'
WORK = Path('/home/fritz/rsmi-work/history-overlap-field')
SEEDS = (6101, 6102, 6103)
ARMS = ('true', 'displaced')
MIN_DEPTHS = (8192, 16384, 32768)
CODES = ('learned', 'count', 'shape')
LN2 = np.log(2.)
RIDGE = .001
FACES = ('z-', 'z+', 'y-', 'y+', 'x-', 'x+')
V = np.array(list(itertools.product(range(-1, 2), repeat=3)))
E = np.array([xyz for xyz in itertools.product(range(-4, 5), repeat=3) if max(map(abs, xyz)) == 4])


def digest(path):
    with path.open('rb') as fp:
        h = hashlib.sha256()
        for chunk in iter(lambda: fp.read(1 << 20), b''):
            h.update(chunk)
        return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def table(path, rows):
    with path.open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader()
        w.writerows(rows)


def shift(x, offset):
    return np.roll(x, -np.asarray(offset), axis=(-3, -2, -1))


def frequency(code):
    return np.bincount(code, minlength=4) / len(code)


def entropy(p):
    return float(-xlogy(p, p).sum() / LN2)


def rate_rule(key, target):
    """FIT-only lexicographic cut values; unseen tuples fall into ordered intervals."""
    keys, count = np.unique(key, return_counts=True)
    if len(keys) < 4:
        raise ValueError('Fewer than four distinct conventional feature tuples')
    cdf = np.r_[0., count.cumsum() / len(key)]
    choices = []
    for perm in set(itertools.permutations(target)):
        lo = 1
        cuts = []
        for j, aim in enumerate(np.cumsum(perm)[:3]):
            hi = len(count) - (3 - j)
            # Clamp to leave at least one observed tuple in each later bin.
            k = min(max(int(np.searchsorted(cdf, aim)), lo), hi)
            k = min((max(lo, k-1), k), key=lambda i: (abs(cdf[i]-aim), i))
            cuts.append(k)
            lo = k + 1
        actual = np.diff(cdf[[0, *cuts, len(count)]])
        choices.append((abs(entropy(actual)-entropy(target)), float(abs(np.sort(actual)-np.sort(target)).sum()), tuple(cuts)))
    return keys[list(min(choices)[2])]


@dataclass
class Bank:
    field: np.ndarray
    ids: np.ndarray
    q: np.ndarray
    lag: int
    v: np.ndarray
    y: np.ndarray
    grid: np.ndarray
    raw: np.ndarray
    count_key: np.ndarray
    shape_key: np.ndarray

    def target(self, displaced):
        return (shift(self.y.reshape(-1, 20, 20, 20, 6).transpose(0, 4, 1, 2, 3), (10, 10, 10))
                .transpose(0, 2, 3, 4, 1).reshape(-1, 6) if displaced else self.y)


def bankpath(a, b, age, level, side):
    return a.work / f'b{b}-a{age}-q{level}-{side}.npz'


def bank(a, b, age, level, side):
    with np.load(bankpath(a, b, age, level, side), allow_pickle=False) as z:
        field, ids, q, lag = z['field'], z['ids'], z['Q'], int(z['lag'].item())
    n = {'fit': 48, 'dev': 16, 'test': 64}[side]
    if (field.shape != (n, 20, 20, 20) or field.dtype != np.uint8 or ids.shape != (n,) or
            len(set(ids)) != n or q.shape != (n,) or field.max() > 2 or not np.isfinite(q).all()):
        raise ValueError('Invalid bank dimensions, identities or ternary states')
    with (a.out/'split.csv').open() as fp:
        roster = sorted((int(r['rank']), int(r['parent_id'])) for r in csv.DictReader(fp)
                        if r['beta']==('3.20','3.40')[b] and r['side']==side)
    if not np.array_equal(ids, np.array([p for _,p in roster], dtype=np.uint64)):
        raise ValueError('Map identities or ordering disagree with immutable preparation split')
    v = np.stack([shift(field, xyz).ravel() for xyz in V], axis=1)
    active = (v == 2).reshape(-1, 3, 3, 3)
    ch = active.sum((1, 2, 3), dtype=np.uint8)
    occ = (v != 0).sum(1, dtype=np.uint8)
    changed = (field == 2).astype(np.uint8)
    bonds, outer, planes, faces = [], [], [], []
    for axis in range(3):
        lines = np.moveaxis(active, 1 + axis, 1)
        bonds.append((lines[:, 1:] & lines[:, :-1]).sum((1, 2, 3), dtype=np.uint8))
        nn = lines.sum((2, 3), dtype=np.uint8)
        outer.extend((nn[:, 0], nn[:, 2]))
        planes.append((nn > 0).sum(1, dtype=np.uint8))
        area = changed
        for t in range(3):
            if axis != t:
                area = sum(np.roll(area, k, axis=t+1) for k in range(-3, 4))
        faces.extend(np.roll(area, -k, axis=axis+1).ravel() for k in (-4, 4))
    count_key = (ch.astype(np.uint64) << 5) | occ
    shape_key = (ch.astype(np.uint64) << 6) | sum(bonds)
    for col, width in [(x, 5) for x in bonds] + [(x, 4) for x in outer] + [(x, 2) for x in planes] + [(occ, 5)]:
        shape_key = (shape_key << width) | col
    bulk = np.repeat(changed.mean((1, 2, 3)), 8000)
    raw = np.column_stack((bulk, *bonds, *outer, *planes)).astype('f4')
    return Bank(field, ids, q, lag, v, np.stack(faces, axis=1), occ.astype('i4')*28 + ch, raw, count_key, shape_key)


def std_design(d, model):
    return (d.raw - model['mu']) / model['sd']


def solve(fun, shape):
    result = minimize(fun, np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=180, gtol=2e-6, ftol=1e-11, maxls=32))
    err = float(abs(result.jac).max())
    if not np.isfinite(result.fun) or err > 3e-5 or not np.isfinite(result.x).all():
        raise RuntimeError(f'Unconverged physical readout: {result.message}, maximum gradient {err}')
    return result.x.reshape(shape), [int(result.nit), err, float(result.fun)]


def identical_features(d):
    """Exact sufficient groups: identical parent bulk, exact counts and all 12 shape terms."""
    order = np.argsort(d.shape_key.reshape(-1, 8000), axis=1, kind='stable')
    order += np.arange(len(d.ids))[:, None]*8000
    order = order.ravel()
    cuts = np.r_[True, d.shape_key[order[1:]] != d.shape_key[order[:-1]]]
    cuts[::8000] = True
    first = np.flatnonzero(cuts)
    inverse = np.empty(len(order), dtype='i4')
    inverse[order] = np.cumsum(cuts, dtype='i4') - 1
    return order[first], inverse, np.diff(np.r_[first, len(order)])


def fit_base(d):
    mu = d.raw.mean(0, dtype=float).astype('f4')
    sd = np.maximum(d.raw.std(0, dtype=float), 1e-8).astype('f4')
    x = (d.raw-mu)/sd
    counts = np.bincount(d.grid, minlength=28**2)
    model = dict(mu=mu, sd=sd, cell_counts=counts, ids=d.ids)
    first, _, multiplicity = identical_features(d)
    xx = x[first].astype(float); grid = d.grid[first]; mass = multiplicity / len(d.v)
    def per_face(face):
        y = d.y[:, face].astype(int)
        tab = np.bincount(d.grid*50+y, minlength=28**2*50).reshape(-1, 50)
        prior = (tab.sum(0)+.5)/(len(y)+25.)
        offset = np.log((tab+prior)/(counts[:, None]+1.))
        yx = np.array([np.bincount(y, weights=col, minlength=50) for col in x.T])/len(y)
        off_observed = (tab*offset).sum()/len(y)
        def loss(v):
            w = v.reshape(x.shape[1], 50)
            val = .5*RIDGE*(v@v) - (yx*w).sum() - off_observed
            grad = RIDGE*w - yx
            for j in range(0, len(first), 4096):
                sl = slice(j, j+4096)
                logits = xx[sl] @ w + offset[grid[sl]]
                peak = logits.max(1)
                logits -= peak[:, None]
                np.exp(logits, out=logits)
                den = logits.sum(1)
                val += mass[sl] @ (peak + np.log(den))
                logits *= (mass[sl]/den)[:, None]
                grad += xx[sl].T @ logits
            return val, grad.ravel()
        w, log = solve(loss, (x.shape[1], 50))
        print('BASE_FACE', FACES[face], 'iterations/gradient/NLL', log, flush=True)
        return offset, w, np.log(prior), log
    with ThreadPoolExecutor(max_workers=6) as pool:
        offsets, slopes, priors, diag = zip(*pool.map(per_face, range(6)))
    model.update(offset=np.stack(offsets), slopes=np.stack(slopes), prior=np.stack(priors), base_solver=np.array(diag),
                 exact_feature_groups=len(first))
    return model


def probabilities(d, model, x, face, sl):
    logits = model['offset'][face, d.grid[sl]] + x[sl] @ model['slopes'][face]
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def symbol_fit(p, y, code, inverse):
    """Fit each four-row main effect to the same categorical offset. No class balancing."""
    weights = np.zeros((4, 50)); logs = []
    for j in range(4):
        have = code == j
        counts = np.bincount(y[have], minlength=50)/len(y)
        if p.ndim == 2:
            group_count = np.bincount(inverse[have], minlength=len(p))
            observed = group_count > 0
            pp = p[observed]
            mass = group_count[observed] / len(y)
        else:
            pp = p
        def loss(w):
            peak = w.max(); e = np.exp(w-peak)
            den = pp @ e
            if pp.ndim == 1:
                normal = (np.log(den)+peak)*counts.sum()
                gr = pp*e/den*counts.sum()
            else:
                normal = mass @ np.log(den) + peak*counts.sum()
                gr = (pp.T @ (mass/den))*e
            return normal-counts@w + .5*RIDGE*(w@w), gr-counts+RIDGE*w
        weights[j], log = solve(loss, (50,)) if have.any() else (np.zeros(50), [0, 0., 0.])
        logs.append(log)
    return weights, logs


def codes(d, ck, head):
    from stock_glass_history_gpu import frozen_logits, hard_labels
    logits = np.concatenate([frozen_logits(ck, d.v[k:k+8192]) for k in range(0, len(d.v), 8192)])
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite deterministic encoder output')
    h = np.asarray(hard_labels(logits), dtype=np.uint8).ravel()
    if len(h) != len(d.v) or h.max() > 3:
        raise ValueError('Bad deterministic four-symbol encoder output')
    if 'count' not in head:
        p = frequency(h)
        head.update(count=rate_rule(d.count_key, p), shape=rate_rule(d.shape_key, p))
    return np.stack((h, np.searchsorted(head['count'], d.count_key, side='right'),
                     np.searchsorted(head['shape'], d.shape_key, side='right'))).astype(np.uint8)


def fit_heads(d, model, batch):
    x = std_design(d, model)
    first, inverse, _ = identical_features(d)
    def per_face(face):
        p = probabilities(d, model, x, face, first)
        for record in batch:
            head, labels = record['head'], record['code']
            for k, h in enumerate(labels):
                for mode, pp in enumerate((np.exp(model['prior'][face]), p)):
                    w, log = symbol_fit(pp, d.y[:, face], h, inverse)
                    head['w'][mode, k, face] = w
                    head['solver'][mode, k, face] = log
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(per_face, range(6)))


def physical_scores(d, model, batch):
    """Output [target=true/shifted,mode=unconditional/conditional,base+3 codes,parent]."""
    x = std_design(d, model)
    targets = (d.y, d.target(True))
    for r in batch:
        r['scores'] = np.zeros((2, 2, 4, len(d.ids)))
    params = {}
    for face, mode in itertools.product(range(6), range(2)):
        weights = np.stack([r['head']['w'][mode,:,face] for r in batch])
        weights -= weights.max(-1, keepdims=True)
        params[face, mode] = weights, np.exp(weights).reshape(-1, 50).T
    def per_parent(parent):
        sl = slice(parent*8000, (parent+1)*8000)
        idx = np.arange(8000)
        for face in range(6):
            ys = [t[sl, face] for t in targets]
            conditional = probabilities(d, model, x, face, sl)
            for mode, p in enumerate((np.exp(model['prior'][face]), conditional)):
                weights, kernel = params[face, mode]
                den = p @ kernel
                base = [np.log(p[y] if not mode else p[idx, y]) for y in ys]
                base_mean = [v.mean() for v in base]
                for j,r in enumerate(batch):
                    s = r['scores'][:, mode, :, parent]
                    s[:, 0] += base_mean
                    for k, h in enumerate(r['code'][:, sl]):
                        w = weights[j, k]
                        four = den[..., (3*j+k)*4:(3*j+k+1)*4]
                        norm = np.log(four[idx,h] if mode else four[h])
                        for t in range(2):
                            s[t, k+1] += (base[t]+w[h, ys[t]]-norm).mean()
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(per_parent, range(len(d.ids))))
    for r in batch:
        r['scores'] /= (6*LN2)
        if not np.isfinite(r['scores']).all():
            raise ValueError('Nonfinite mean-marginal face score')


def checkpoint(a, arm, seed, depth):
    p = a.checkpoints/f'{arm}-s{seed}-u{depth}.npz'
    rec = json.loads(p.with_suffix('.json').read_text())
    if (rec['valid'] is not True or rec['weights_and_adam_finite'] is not True
            or rec['step'] != depth or rec['sha256'] != digest(p)):
        raise ValueError(f'Unregistered or invalid six-arm training checkpoint: {p.name}')
    with np.load(p, allow_pickle=False) as z:
        ck = {k:z[k] for k in z.files}
    if int(ck['adam_iterations'].item()) != depth or not all(np.isfinite(v).all() for v in ck.values() if v.dtype.kind in 'fc'):
        raise ValueError(f'Invalid checkpoint {p.name}')
    return p, ck


def model_path(a):
    return a.out/'readouts'/'conditional-base.npz'


def head_path(a, arm, seed, depth):
    return model_path(a).parent/f'{arm}-s{seed}-u{depth}.npz'


def fields(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def fit_examples(d, h, path):
    """Append frozen symbols to the published pretraining FIT geometry; never reselect."""
    fixed = json.loads(path.read_text())
    occ, changed = fixed['n_initially_occupied'], fixed['n_changed']
    j, = np.flatnonzero(d.ids == int(fixed['parent']))
    pair = [8000*j + r['flat_index'] for r in fixed['rows']]
    if all(d.grid[n] == 28*occ+changed and np.array_equal(d.v[n].reshape(3,3,3), r['ternary'])
           for n,r in zip(pair, fixed['rows'])):
        return [dict(id=str(d.ids[j]), center=list(map(int, np.unravel_index(n%8000, (20,)*3))),
                     symbol=int(h[n]), initial_occupied=occ, changed=changed, axial_bonds=d.raw[n, 1:4].astype(int).tolist(),
                     face_counts=d.raw[n, 4:10].astype(int).tolist(), planes=d.raw[n, 10:13].astype(int).tolist(),
                     field_zyx=d.v[n].reshape(3, 3, 3).tolist()) for n in pair]
    raise ValueError('Published pretraining FIT representatives do not reproduce')


def support(d, model):
    """Equal number of centers per parent. Active-channel covariance is 54x54, not 54 independent tests."""
    sumv = np.zeros(54); cross = np.zeros((54, 54))
    for j in range(0, len(d.v), 8000):
        v = d.v[j:j+8000]
        x = np.stack((v == 1, v == 2), axis=-1).reshape(-1, 54).astype(float)
        sumv += x.sum(0); cross += x.T@x
    mean = sumv/len(d.v)
    cov = cross/len(d.v) - np.outer(mean, mean)
    counts = np.bincount(d.grid, minlength=28**2)
    return dict(ids=d.ids, grid_counts=counts.reshape(28, 28), unseen_grid_fraction=float(counts[model['cell_counts']==0].sum()/len(d.v)),
                channel_means=mean, channel_cov=cov, channel_cov_spectrum=np.linalg.eigvalsh(cov),
                physical_cov=np.cov(d.raw.T, ddof=0), face_targets=np.stack([np.bincount(y, minlength=50) for y in d.y.T]),
                mean_q=d.q.mean(), lag=d.lag, per_parent_q=d.q)


def critic_scores(d, r):
    from stock_glass_history_gpu import dev_draws, frozen_logits, frozen_scores, relaxed_draw
    n = len(d.ids); ii = np.arange(n)
    rows = np.zeros((2, 2, n)); native_delta = 0.
    n_panels = 0
    for origins, u, perm in dev_draws(n, r['seed']):
        origins = np.asarray(origins)
        if origins.shape == (n, 3):
            xyz = origins
            origins = np.ravel_multi_index(xyz.T, (20,)*3)
        else:
            xyz = np.column_stack(np.unravel_index(origins, (20,)*3))
        if not np.array_equal(np.sort(perm), ii) or np.any(perm == ii):
            raise ValueError('Non-deranged critic control')
        h = relaxed_draw(r['ck'], frozen_logits(r['ck'], d.v[ii*8000+origins]), u)
        for t, delta in enumerate((0, 10)):
            ix = (xyz[:, None]+E+delta) % 20
            shell = d.field[ii[:, None], *ix.transpose(2, 0, 1)]
            score = np.asarray(frozen_scores(r['ck'], shell, h), dtype=float)
            if score.shape != (n, n) or not np.isfinite(score).all():
                raise ValueError('Invalid H-row by E-column frozen critic matrix')
            # Subtract row maxima before *both* terms, avoiding catastrophic cancellation.
            score -= score.max(1, keepdims=True)
            stable = (score-logsumexp(score, axis=1, keepdims=True)+np.log(n))/LN2
            f = score.astype('f4')
            fp32 = (f-logsumexp(f, axis=1, keepdims=True)+np.log(np.float32(n)))/np.float32(LN2)
            native_delta = max(native_delta, float(np.max(abs(stable-fp32))))
            rows[t, 0] += stable[ii, ii]; rows[t, 1] += stable[ii, perm]
        n_panels += 1
    if n_panels != 128 or not np.isfinite(rows).all():
        raise ValueError('Incomplete stable held critic evaluation')
    return rows/n_panels, native_delta


def source_paths(a):
    return {side:bankpath(a, 0, 500, 80, side) for side in ('fit', 'dev')}


def training_record(a, depth):
    return a.out/f'readout-dev-u{depth}.json'


def code_counts(labels):
    return np.stack([np.stack([np.bincount(hh, minlength=4) for hh in h.reshape(-1, 8000)]) for h in labels])


def prepare_base(a):
    dest = model_path(a)
    if dest.exists() or (a.out/'readout-freeze.json').exists():
        raise FileExistsError('Conditional source base is already fitted or frozen')
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    train = bank(a, 0, 500, 80, 'fit')
    sha = digest(source_paths(a)['fit'])
    model = fit_base(train)
    np.savez_compressed(dest, **model, fit_sha256=sha)
    np.savez_compressed(dest.parent/'fit-support.npz', **support(train, model))
    dump(dest.parent/'base-receipt.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-start,
         fit_map_sha256=sha, base_sha256=digest(dest), parents=len(train.ids), centers_per_parent=8000,
         exact_feature_groups=model['exact_feature_groups'], face_order=FACES, face_solver=model['base_solver'].tolist(),
         solver_columns=['iterations','max_abs_gradient','penalized_nll_nats'], test_access=False))


def fit(a):
    if training_record(a, a.depth).exists() or (a.out/'readout-freeze.json').exists():
        raise FileExistsError('Readout already registered or frozen')
    checkpoints = {(arm, seed):checkpoint(a, arm, seed, a.depth) for arm, seed in itertools.product(ARMS, SEEDS)}
    paths = source_paths(a); sha = {k:digest(v) for k,v in paths.items()}
    train = bank(a, 0, 500, 80, 'fit')
    basepath = model_path(a)
    basepath.parent.mkdir(parents=True, exist_ok=True)
    if basepath.exists():
        model = fields(basepath)
        if model['fit_sha256'].item() != sha['fit'] or not np.array_equal(model['ids'], train.ids):
            raise ValueError('The frozen base belongs to a different FIT bank')
    else:
        model = fit_base(train)
        np.savez_compressed(basepath, **model, fit_sha256=sha['fit'])
    batch = []; examples = {}
    for arm, seed in itertools.product(ARMS, SEEDS):
        path, ck = checkpoints[arm, seed]
        head = dict(w=np.empty((2, 3, 6, 4, 50)), solver=np.empty((2, 3, 6, 4, 3)))
        labels = codes(train, ck, head)
        head['fit_counts'] = code_counts(labels)
        examples[f'{arm}-{seed}'] = fit_examples(train, labels[0], a.out/'same-count-examples.json')
        batch.append(dict(arm=arm, seed=seed, ck=ck, cksha=digest(path), head=head, code=labels))
    if not (basepath.parent/'fit-support.npz').exists():
        np.savez_compressed(basepath.parent/'fit-support.npz', **support(train, model))
    fit_heads(train, model, batch)
    dev = bank(a, 0, 500, 80, 'dev')
    if not set(train.ids).isdisjoint(dev.ids):
        raise ValueError('Source FIT/DEV parent leakage')
    for r in batch:
        r['code'] = codes(dev, r['ck'], r['head'])
    physical_scores(dev, model, batch)
    receipt = dict(depth=a.depth, source_maps=sha, base=digest(basepath), heads={}, parents_fit=train.ids.astype(str).tolist(),
                   parents_dev=dev.ids.astype(str).tolist(), examples=examples)
    for r in batch:
        head, s = r['head'], r['scores']
        h, precision = critic_scores(dev, r)
        ref = np.stack([r['ck'][f'dev_{arm}_scores'][:, 4:6].mean(0) for arm in ARMS])
        if not np.allclose(h.mean(axis=-1), ref, rtol=2e-9, atol=1e-9):
            raise ValueError(f'Host and checkpoint portable DEV estimates disagree: {h.mean(axis=-1)} vs {ref}')
        # Actual true-shell conditional DEV score; never use TEST to choose comparator.
        selected = int(np.argmax(s[0, 1, 2:].mean(axis=-1)))
        p = head_path(a, r['arm'], r['seed'], a.depth)
        np.savez_compressed(p, **head, checkpoint_sha256=r['cksha'], selected=selected,
                            dev_counts=code_counts(r['code']), dev_scores=s, dev_critic=h,
                            dev_critic_float32_max_abs_bits=precision)
        receipt['heads'][p.stem] = dict(sha256=digest(p), checkpoint_sha256=r['cksha'], selected=CODES[1+selected],
                                        dev_conditional_bits=s[0,1].mean(axis=-1).tolist(), dev_critic_bits=h.mean(axis=-1).tolist(),
                                        fit_code_frequencies=(head['fit_counts'].sum(1)/len(train.v)).tolist())
    dump(training_record(a, a.depth), receipt)


def freeze(a):
    if a.depth not in (32768, 65536):
        raise ValueError('Cannot release a subminimum fit')
    depths = [*MIN_DEPTHS] + ([65536] if a.depth == 65536 else [])
    dest = a.out/'readout-freeze.json'
    if dest.exists():
        raise FileExistsError('Existing readout freeze must not be replaced')
    src = {k:digest(v) for k,v in source_paths(a).items()}
    wave = json.loads((a.checkpoints/'wave.json').read_text())
    decision = json.loads((a.checkpoints/'depth-decision.json').read_text())
    gpu = json.loads((a.out/'gpu-completion.json').read_text())
    if (wave['complete'] is not True or wave['workers']!=6 or decision['selected_depth']!=a.depth
            or gpu['selected_depth']!=a.depth or gpu['actual_updates']!=6*a.depth):
        raise ValueError('GPU wave incomplete or selected depth inconsistent; assessment forbidden')
    base = digest(model_path(a))
    lock = dict(selected_depth=a.depth, depths=depths, base_sha256=base, source_maps=src, heads={}, receipts={},
                frozen_utc=datetime.now(timezone.utc).isoformat(), gpu_completion_sha256=digest(a.out/'gpu-completion.json'),
                depth_decision_sha256=digest(a.checkpoints/'depth-decision.json'),
                readout_script_sha256=digest(Path(__file__)), gpu_script_sha256=digest(Path(__file__).with_name('stock_glass_history_gpu.py')),
                design_sha256=digest(a.out/'design.md'), split_sha256=digest(a.out/'split.csv'))
    for depth in depths:
        path = training_record(a, depth)
        rec = json.loads(path.read_text())
        if rec['base'] != base or src != rec['source_maps'] or rec['depth'] != depth:
            raise ValueError('Source maps/base changed between depths')
        lock['receipts'][str(depth)] = digest(path)
        for arm, seed in itertools.product(ARMS, SEEDS):
            p = head_path(a, arm, seed, depth)
            spec = rec['heads'][p.stem]
            ck, _ = checkpoint(a, arm, seed, depth)
            if digest(ck) != spec['checkpoint_sha256'] or digest(p) != spec['sha256']:
                raise ValueError('Changed source readout or optimizer checkpoint')
            lock['heads'][p.stem] = spec
    dump(dest, lock)


def admit(a):
    if not a.admission:
        raise ValueError('Main investigator must explicitly release held scoring')
    freeze_path = a.out/'readout-freeze.json'
    adm = json.loads(a.admission.read_text())
    if adm.get('assessment_authorized') is not True or adm.get('readout_freeze_sha256') != digest(freeze_path):
        raise ValueError('No assessment admission for this freeze')
    lock = json.loads(freeze_path.read_text())
    if (lock['base_sha256'] != digest(model_path(a)) or lock['design_sha256'] != digest(a.out/'design.md')
            or lock['split_sha256'] != digest(a.out/'split.csv')
            or lock['gpu_completion_sha256'] != digest(a.out/'gpu-completion.json')
            or lock['depth_decision_sha256'] != digest(a.checkpoints/'depth-decision.json')
            or lock['readout_script_sha256'] != digest(Path(__file__))
            or lock['gpu_script_sha256'] != digest(Path(__file__).with_name('stock_glass_history_gpu.py'))):
        raise ValueError('Changed frozen implementation, design or readout base')
    for k, path in source_paths(a).items():
        if digest(path) != lock['source_maps'][k]:
            raise ValueError('Changed source split')
    for depth in lock['depths']:
        if lock['receipts'][str(depth)] != digest(training_record(a, depth)):
            raise ValueError('Changed DEV registration')
        for arm, seed in itertools.product(ARMS, SEEDS):
            p = head_path(a, arm, seed, depth)
            sp = lock['heads'][p.stem]
            ck, _ = checkpoint(a, arm, seed, depth)
            if digest(ck) != sp['checkpoint_sha256'] or digest(p) != sp['sha256']:
                raise ValueError('Changed frozen head or operator')
    return lock


def interval(x, draw):
    v = np.quantile(x[draw].mean(axis=1), [.025, .975])
    return dict(mean=float(x.mean()), lo=float(v[0]), hi=float(v[1]))


def plot(out, contrast, selected, example):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for b, age in itertools.product(range(2), (500, 300)):
        ax = axes[int(age==300), b]
        for j, seed in enumerate(SEEDS):
            rr = sorted((r for r in contrast if r['beta_index']==b and r['age']==age and r['seed']==seed
                         and r['depth']==selected and r['metric']=='true_over_conventional'), key=lambda r:r['level'])
            q = np.array([r['q'] for r in rr]); m = np.array([r['mean'] for r in rr])
            lo = np.array([r['lo'] for r in rr]); hi = np.array([r['hi'] for r in rr])
            ax.plot(q, m, '.-', label=str(seed), color=f'C{j}')
            ax.fill_between(q, lo, hi, alpha=.12, color=f'C{j}')
        ax.axhline(0, lw=.7, c='k')
        ax.set(title=f'β={3.2+.2*b:.2f}, prep {age}M', xlabel='Measured parent-mean Q', ylabel='Increment (bits/block)')
    axes[0,0].legend(title='Seed', fontsize=8)
    fig.suptitle('Conditional learned minus DEV-selected conventional • whole-parent 95% intervals')
    fig.tight_layout(); fig.savefig(out/'matched-q-physical-gain.png', dpi=160); plt.close(fig)
    if example:
        fig, axes = plt.subplots(2, 3, figsize=(6, 4))
        for i, ex in enumerate(example):
            for z, plane in enumerate(ex['field_zyx']):
                axes[i,z].imshow(plane, vmin=0, vmax=2, cmap=matplotlib.colors.ListedColormap(['#ffffff','#aebac5','#be392e']))
                axes[i,z].set(xticks=[], yticks=[], title=f"code {ex['symbol']}, z={z-1}")
        fig.suptitle(f"FIT examples at equal occupied={example[0]['initial_occupied']}, changed={example[0]['changed']}\nwhite vacant · gray same · red changed")
        fig.tight_layout(); fig.savefig(out/'same-count-learned-symbols.png', dpi=160); plt.close(fig)


def assess(a):
    lock = admit(a)  # No TEST filename is opened before all source-model checks.
    selected = lock['selected_depth']
    if (a.out/'readout-decision.json').exists():
        raise FileExistsError('Held decision already exists')
    dest = a.out/'held-parents'; dest.mkdir(exist_ok=True)
    model = fields(model_path(a))
    rec = json.loads(training_record(a, selected).read_text())
    source = set(rec['parents_fit']+rec['parents_dev'])
    boot = [np.random.default_rng(988401+b).integers(64, size=(2048,64)) for b in range(2)]
    all_heads = []
    for depth, arm, seed in itertools.product(lock['depths'], ARMS, SEEDS):
        _, ck = checkpoint(a, arm, seed, depth)
        p = head_path(a, arm, seed, depth)
        all_heads.append(dict(depth=depth, arm=arm, seed=seed, ck=ck, head=fields(p)))
    scores, controls, contrast, rates, support_rows = [], [], [], [], []
    age_gain, map_hashes, decisions = {}, {}, []
    for b in range(2):
        primary_ids = None
        for age, level in itertools.product((500, 300), (80, 75, 85)):
            is_primary = (age, level) == (500, 80)
            p = bankpath(a, b, age, level, 'test')
            d = bank(a, b, age, level, 'test')
            if b == 0 and not source.isdisjoint(d.ids.astype(str)):
                raise ValueError('Source model assessed on non-held parent')
            if primary_ids is None:
                primary_ids = d.ids
            if not np.array_equal(primary_ids, d.ids):
                raise ValueError('Age/Q comparisons must preserve parent identity AND ordering')
            map_hashes[p.name] = digest(p)
            suf = f'b{b}-a{age}-q{level}'
            sup = support(d, model)
            np.savez_compressed(dest/f'support-{suf}.npz', **sup)
            meta = dict(beta_index=b, age=age, level=level, lag=d.lag, q=float(d.q.mean()), parents=len(d.ids), centers_per_parent=8000)
            support_rows.append(dict(**meta, unseen_grid_fraction=sup['unseen_grid_fraction'],
                                     exact_cells=int(np.count_nonzero(sup['grid_counts'])),
                                     active_input_codim=int((sup['channel_cov_spectrum'] < 1e-10).sum())))
            batch = [r for r in all_heads if is_primary or r['depth']==selected]
            for r in batch:
                r['code'] = codes(d, r['ck'], r['head'])
            physical_scores(d, model, batch)
            gains = {}
            for r in batch:
                head = r['head']; s = r['scores']; arm, seed, depth = r['arm'], r['seed'], r['depth']
                comparator = int(head['selected'].item()) + 2
                key = dict(**meta, depth=depth, seed=seed)
                counts = code_counts(r['code'])
                actual = counts.sum(1)/len(d.v); fit = head['fit_counts'].sum(1)/head['fit_counts'].sum(axis=(1,2))[:, None]
                ent = [entropy(x) for x in actual]; fent = [entropy(x) for x in fit]
                supported = bool((actual > 0).all() and (fit > 0).all() and abs(fent[comparator-1]-fent[0]) <= .05)
                for j, name in enumerate(CODES):
                    rates.append(dict(**key, arm=arm, symbol=name, entropy_bits=ent[j], fit_entropy_bits=fent[j],
                                      held_gap_bits=ent[j]-ent[0], fit_gap_bits=fent[j]-fent[0], selected=(j==comparator-1),
                                      **{f'freq{k}':actual[j,k] for k in range(4)}))
                for target, mode, code in itertools.product(range(2), range(2), range(4)):
                    scores.append(dict(**key, arm=arm, target=ARMS[target], mode=('unconditional','conditional')[mode],
                                       code=('base',*CODES)[code], **interval(s[target,mode,code],boot[b])))
                packet = dict(ids=d.ids, scores=s, counts=counts, selected=comparator, per_parent_q=d.q)
                # Frozen-critic checkpoint/controls only on the primary held endpoint; all other views are physical sensitivity.
                if is_primary:
                    h, err = critic_scores(d, r)
                    for t in range(2):
                        rr = dict(**key, arm=arm, target=ARMS[t], max_stable_float32_error_bits=err, panels=128,
                                  interval_scope='whole H parents conditional on fixed cross-parent negative panels')
                        for name, values in (('matched', h[t,0]), ('shuffled', h[t,1]), ('difference', h[t,0]-h[t,1])):
                            rr.update({f'{name}_{stat}_bits':value for stat,value in interval(values,boot[b]).items()})
                        controls.append(rr)
                    supported = bool(supported and np.isfinite(h).all())
                    packet.update(critic=h, float32_max_abs_bits=err)
                np.savez_compressed(dest/f'{suf}-{arm}-s{seed}-u{depth}.npz', **packet)
                gains[arm,seed,depth] = (s[0,1,1]-s[0,1,comparator], s[0,1,1]-s[0,1,0], supported,
                                         s[0,1,1], s[1,1,1]-s[1,1,comparator], s[0,0,1]-s[0,0,comparator])
            for seed, depth in itertools.product(SEEDS, sorted({r['depth'] for r in batch})):
                true, null = gains['true',seed,depth], gains['displaced',seed,depth]
                key = dict(**meta, seed=seed, depth=depth)
                metrics = dict(true_over_conventional=true[0], true_over_base=true[1], displaced_over_conventional=null[0],
                               paired_increment=true[0]-null[0], paired_absolute_score=true[3]-null[3],
                               true_increment_displaced_target=true[4], unconditioned_over_conventional=true[5])
                vals = {}
                for name, v in metrics.items():
                    vals[name] = interval(v, boot[b])
                    contrast.append(dict(**key, metric=name, **vals[name]))
                if depth == selected:
                    age_gain[b,age,level,seed] = true[0]
                if is_primary and depth == selected:
                    cut = 0. if b else .01
                    ok = (true[2] and null[2] and vals['true_over_conventional']['lo']>cut
                          and vals['true_over_base']['lo']>cut and vals['paired_increment']['lo']>0
                          and vals['paired_absolute_score']['lo']>0)
                    decisions.append(dict(**key, gates_passed=bool(ok), support_and_controls=true[2] and null[2],
                                          threshold_bits=cut, primary=vals))
    paired_age = []
    for b, level, seed in itertools.product(range(2), (75,80,85), SEEDS):
        delta = age_gain[b,300,level,seed]-age_gain[b,500,level,seed]
        paired_age.append(dict(beta_index=b, level=level, seed=seed, parents=64, contrast='300M minus 500M gain',
                               **interval(delta,boot[b])))
    table(a.out/'physical-scores.csv', scores); table(a.out/'physical-gains.csv', contrast)
    table(a.out/'symbol-support.csv', rates); table(a.out/'physical-support.csv', support_rows)
    table(a.out/'held-critic.csv', controls); table(a.out/'paired-age.csv', paired_age)
    dump(a.out/'same-count-fit.json', rec['examples'])
    plot(a.out, contrast, selected, rec['examples']['true-6101'])
    passed = len(decisions)==6 and all(r['gates_passed'] for r in decisions)
    dump(a.out/'readout-decision.json', dict(selected_depth=selected, assessment_admission_sha256=digest(a.admission),
         readout_freeze_sha256=digest(a.out/'readout-freeze.json'), test_maps_sha256=map_hashes,
         physical_supports_transfer=passed, decisions=decisions, bootstrap=dict(replicates=2048, unit='parent',
         method='percentile, paired within beta, independent draws across beta'), unit='mean of six marginal log2 scores per block, not joint MI',
         interpretation=('Positive held incremental scores under these controls; not causality or static order.' if passed else
                         'The prespecified held evidence for transferred arrangement information is insufficient.'),
         examples='FIT examples illustrate distinct arrangements only; no held mechanism is inferred.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('base','fit','freeze','assess'))
    p.add_argument('--work', type=Path, default=WORK)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--checkpoints', type=Path)
    p.add_argument('--depth', type=int, choices=(*MIN_DEPTHS,65536))
    p.add_argument('--admission', type=Path)
    a = p.parse_args()
    if a.checkpoints is None:
        a.checkpoints = a.out/'checkpoints'
    if a.action in ('fit','freeze') and a.depth is None:
        p.error('--depth is required for fit and freeze')
    {'base':prepare_base,'fit':fit,'freeze':freeze,'assess':assess}[a.action](a)


if __name__ == '__main__':
    main()
