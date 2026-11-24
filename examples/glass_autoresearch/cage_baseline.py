"""Archived-parent cage bank and frozen, target-supervised conventional competitors.

Y is a *count*: the score is the mean Bernoulli log loss for the three outward
pixels, not likelihood for the joint count and not a three-sample uncertainty estimate.
No configurations, swaps, trajectories, or teacher labels are generated here.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l20_pairs import L, PAIR_SEED, SPLIT, partition

FOLDS = ('train', 'validation', 'test')
V = np.indices((3, 3, 3)).reshape(3, -1).T - 1
AX = np.eye(3, dtype=int)
NN = np.concatenate((AX, -AX))
TARGET = V[0] - 3 * AX
IDEAL = np.array([0, 3, 5], dtype=np.int16)
NO_SWAP = 32767
BANK_VERSION = 1
SEED = 318749
TOL = 0.0001  # nats per outward species pixel, fixed before discovery


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load_json(path):
    return json.loads(Path(path).read_text())


def lock_hash(lock, index):
    if lock is None or not lock.is_file():
        raise ValueError('test requires --lock with the precommitted prediction and artifact hashes')
    d = load_json(lock)
    if (type(d.get('index')) is not int or d['index'] != index or d.get('fold') != 'test'
            or type(d.get('prediction')) not in (int, float) or not np.isfinite(d['prediction'])
            or not isinstance(d.get('metric'), str) or not d['metric'].strip()):
        raise ValueError('incomplete test prediction, metric or cohort in lock')
    for k in ('source_sha256', 'rule_sha256', 'model_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', str(d.get(k, ''))):
            raise ValueError(f'missing or malformed {k}')
    for k in ('rule', 'model'):
        p = lock.parent / d[k + '_path']
        if not p.is_file() or digest(p) != d[k + '_sha256']:
            raise ValueError(f'{k} artifact mismatch')
    return d['source_sha256']


def locator(sites, coords):
    ids = {tuple(c): i for i, c in enumerate(sites)}
    return np.array([ids[tuple(c)] for c in coords.reshape(-1, 3)]).reshape(coords.shape[:-1])


def halo(sites):
    return np.unique(np.concatenate((sites, (sites[:, None] + NN).reshape(-1, 3))), axis=0)


H1 = halo(V)
H2 = halo(H1)
V1 = locator(H1, V)
S1 = locator(H2, H1)
ADJ1 = locator(H1, V[:, None] + NN)
ADJ2 = locator(H2, H1[:, None] + NN)
EDGES = [(i, j, a) for i, v in enumerate(V) for a in range(3) if v[a] < 1
         for j in [int(np.ravel_multi_index(v + AX[a] + 1, (3,) * 3))]]
I, J, AXIS = np.array(EDGES).T
OTH_I = locator(H1, np.array([[V[i] + n for n in NN if np.any(V[i] + n != V[j])] for i, j in zip(I, J)]))
OTH_J = locator(H1, np.array([[V[j] + n for n in NN if np.any(V[j] + n != V[i])] for i, j in zip(I, J)]))


def geometry():
    dist = lambda a, b: np.abs((a[:, None] - b + L//2) % L - L//2).sum(-1).min(1)
    if (len(EDGES) != 54 or not np.array_equal(V[J] - V[I], AX[AXIS])
            or dist(H2, V).max() != 2 or not np.all(dist(TARGET, V) == 3)
            or not np.all(dist(TARGET, H2) == 1)):
        raise ValueError('invalid internal edge/halo/gap geometry')


def address(offsets):
    centers = np.indices((L,) * 3, dtype=np.int16).reshape(3, -1).T
    out = np.zeros((len(centers), len(offsets)), np.uint16)
    for a in range(3):
        out *= L
        out += ((centers[:, a, None] + offsets[:, a]) % L).astype(np.uint16)
    return out


def extract(s, reps):
    # s[anchor,H2] is the ONLY species input; target pixels are not in H2.
    s1 = s[:, S1]; v = s1[:, V1]; o2 = (s != 0).astype(np.int16)
    m1 = sum(o2[:, ADJ2[:, k]] for k in range(6)); m = m1[:, V1]
    r1 = (m1 - IDEAL[s1]) * (s1 != 0); r = r1[:, V1]
    moment = sum(r1[:, ADJ1[:, k]] for k in range(6))
    a, b = v[:, I], v[:, J]; oa, ob = (a != 0).astype(np.int16), (b != 0).astype(np.int16)
    d = ob - oa; ma, mb = m[:, I], m[:, J]; ra, rb = r[:, I], r[:, J]
    ends = ob*(ma-d-IDEAL[b])**2 + oa*(mb+d-IDEAL[a])**2 - ra**2 - rb**2
    costs = ends + 2*d*(moment[:, I]-rb-moment[:, J]+ra) + d*d*(ma-ob+mb-oa)
    # Independent conditional-energy sums on actual measured neighborhoods, all
    # 54 edges of the first anchor of EVERY parent; no new configuration built.
    sub = slice(None, None, reps)
    def site_energy(sp, n):
        return (sp != 0) * (n - IDEAL[sp])**2
    ref = (site_energy(b[sub], ma[sub]-d[sub]) + site_energy(a[sub], mb[sub]+d[sub])
           - site_energy(a[sub], ma[sub]) - site_energy(b[sub], mb[sub]))
    for sites, sign in ((OTH_I, 1), (OTH_J, -1)):
        sp = s1[sub][:, sites]; n = m1[sub][:, sites]
        ref += (site_energy(sp, n+sign*d[sub, :, None]) - site_energy(sp, n)).sum(-1, dtype=np.int16)
    if not np.array_equal(ref, costs[sub]):
        raise ValueError('measured-neighborhood conditional energy disagrees with moment formula')
    costs[a == b] = NO_SWAP
    return v, m, costs


def bank(a):
    test = a.fold == 'test'
    if (a.index, test) not in ((9, False), (6, True)) or a.reps < 1:
        raise ValueError('only index 9 discovery or locked index 6 test; positive reps required')
    wanted = lock_hash(a.lock, a.index) if test else None  # before any archive access
    geometry()
    p = a.data / f'beta-{a.index:02d}.npz'; receipt = load_json(p.with_suffix('.json'))
    if receipt['index'] != a.index or (test and wanted != receipt['endpoint_sha256']):
        raise ValueError('lock/source receipt mismatch')
    sha = digest(p)
    if sha != receipt['endpoint_sha256']:
        raise ValueError('changed raw release')
    fidx = FOLDS.index(a.fold); ids = partition(a.index)[0][fidx]; P, R = len(ids), a.reps
    with np.load(p, allow_pickle=False) as f:
        if str(f['axes']) != 'chain,z,y,x' or float(f['beta']) != float(receipt['beta']):
            raise ValueError('axis or beta mismatch')
        keys = np.column_stack([f[k][ids].astype('<u8') for k in ('chain_seed', 'chain_id')])
        # NPZ decompresses a whole member. Subset BEFORE feature, value, or label inspection.
        raw = f['species']; shape = raw.shape; state = raw[ids]; del raw
    if (shape != (sum(SPLIT), L, L, L) or state.dtype != np.uint8 or state.max() > 2
            or len(np.unique(keys, axis=0)) != P):
        raise ValueError('malformed release or duplicate selected lineage')
    flat = state.reshape(P, -1)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, a.index, fidx]))
    origins = rng.integers(L**3, size=(R, P), dtype=np.uint16).T
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise FileExistsError(f'bank destination must be empty: {a.out}')
    arrays = {}
    for name, n, dtype in [('v', 27, 'u1'), ('m', 27, 'u1'), ('bonds', 54, '<i2'), ('y', None, 'u1')]:
        arrays[name] = np.lib.format.open_memmap(a.out/(name+'.npy'), mode='w+', dtype=dtype,
                                                  shape=(P, R) + (() if n is None else (n,)))
    ht, yt = address(H2), address(TARGET)
    step = max(1, 2048 // R)
    for i in range(0, P, step):
        sl = slice(i, min(i+step, P)); centers = origins[sl]; rows = np.arange(len(centers))[:, None, None]
        x = flat[sl][rows, ht[centers]].reshape(-1, len(H2))
        for name, values in zip(('v', 'm', 'bonds'), extract(x, R)):
            arrays[name][sl] = values.reshape(len(centers), R, -1)
        arrays['y'][sl] = (flat[sl][rows, yt[centers]] == 1).sum(-1)
    for arr in arrays.values():
        arr.flush()
    for name, arr in [('parent_id', ids), ('parent_keys', keys), ('origins', origins)]:
        np.save(a.out/(name+'.npy'), arr, allow_pickle=False)
    cut = 3*P//4 if a.fold == 'train' else None
    dump(a.out/'meta.json', dict(version=BANK_VERSION, index=a.index, fold=a.fold, beta=float(receipt['beta']),
         source=p.name, source_sha256=sha, receipt_sha256=digest(p.with_suffix('.json')), parents=P, reps=R,
         split_sizes=list(SPLIT), partition_seed=PAIR_SEED, origin_seed=[SEED, a.index, fidx], fit_parents=cut,
         parent_key_columns=['chain_seed', 'chain_id'], parent_order='l20_pairs.partition order (not sorted)',
         lineage_sha256=hashlib.sha256(keys.tobytes()).hexdigest(),
         sha256={p.name: digest(p) for p in sorted(a.out.glob('*.npy'))}, lock_sha256=digest(a.lock) if test else None,
         center_axes='periodic z,y,x; flattened x fastest', visible=V.tolist(), target=TARGET.tolist(),
         target_semantics='count species A=1 at three outward sites; divide by three for Bernoulli score',
         bonds='positive z,y,x inside visible, ordered site then axis; equal labels=32767; units H',
         edge_endpoint_axis=[[int(i), int(j), int(a)] for i,j,a in EDGES], equal_label_sentinel=NO_SWAP,
         halo_l1=2, target_l1=3, conditional_energy_checks=P*54))


def mmap(path, name):
    return np.load(Path(path)/(name+'.npy'), mmap_mode='r', allow_pickle=False)


def open_bank(path, expected=None):
    path = Path(path); meta = load_json(path/'meta.json')
    cohort = (meta['index'], meta['fold'])
    if (meta['version'] != BANK_VERSION or cohort not in [(9, 'train'), (9, 'validation'), (6, 'test')]
            or expected is not None and cohort != expected
            or cohort == (6, 'test') and not re.fullmatch('[0-9a-f]{64}', str(meta['lock_sha256']))):
        raise ValueError(f'wrong bank protocol/cohort: {path}')
    if not np.array_equal(meta['visible'], V) or not np.array_equal(meta['target'], TARGET):
        raise ValueError('bank geometry mismatch')
    data = {k: mmap(path, k) for k in ('v', 'm', 'bonds')}
    p, r = meta['parents'], meta['reps']
    if any(data[k].shape != (p, r, n) for k, n in [('v', 27), ('m', 27), ('bonds', 54)]):
        raise ValueError('wrong bank array shape')
    return meta, data


# Only symmetries that fix this corner/pixel average (axis permutations, not
# arbitrary translations or cube reflections). Raw site and bond channels stay
# oriented; pooled channels let the regressors exploit symmetry at finite n.
DIST2 = ((V[:, None] - V)**2).sum(-1)
RAD2 = DIST2[0]
PAIRS = [np.where(np.triu(DIST2 == d, k=1)) for d in np.unique(DIST2) if d]
RADII = [np.where(RAD2 == d)[0] for d in np.unique(RAD2) if d]
INNER = [(i, np.flatnonzero(DIST2[i] == 1)) for i in range(len(V))]
SITE_ORBITS = [np.where((np.sort(V+1, axis=1) == k).all(1))[0]
               for k in np.unique(np.sort(V+1, axis=1), axis=0)]
EDGE_REGIONS = (np.arange(54), np.where(I == 0)[0], np.where(RAD2[I] == 1)[0])
HGB_NAMES = ('species_nonlinear', 'coordination', 'full_class', 'full_threshold')
LADDER = ('marginal', 'composition', 'additive', 'pairs', 'species_nonlinear', 'coordination', 'full')
CAP = 1.e-5


def chunks(data, parents):
    # Uniform parent weight (same R); feature temporaries never span a bank.
    flat = [data[k][:parents].reshape(-1, data[k].shape[-1]) for k in ('v', 'm', 'bonds')]
    for lo in range(0, len(flat[0]), 4096):
        sl = slice(lo, min(lo+4096, len(flat[0])))
        yield sl, flat[0][sl], flat[1][sl], flat[2][sl]


def comp_index(v):
    return (v == 1).sum(1)*28 + (v == 2).sum(1)


def species_features(v, pairs):
    a, b = v == 1, v == 2
    cols = [a, b]
    if pairs:
        # Translationally pooled internal two-point AA, unordered AB and BB
        # by squared separation, plus the corner-rooted pair features (all
        # site labels represented by A/B vs vacancy reference).
        for i, j in PAIRS:
            ai, aj, bi, bj = a[:, i], a[:, j], b[:, i], b[:, j]
            cols += [(ai & aj).mean(1), ((ai & bj) | (bi & aj)).mean(1), (bi & bj).mean(1)]
        for group in RADII:
            for field in (a, b):
                n = field[:, group].mean(1)
                cols += [n * a[:, 0], n * b[:, 0]]
    return np.column_stack(cols).astype(np.float32, copy=False)


def physical_features(v, m, bonds, xpair, p_pair, name):
    # Squared-error residual on the measured fraction, around the same fitted
    # binomial pair baseline. All stages retain the simpler species inputs.
    cols = [xpair, p_pair]
    if name == 'species_nonlinear':
        return np.column_stack(cols).astype('f', copy=False)
    s = v.astype(np.int16); occ = s != 0
    m = m.astype(np.int16); r = (m - IDEAL[s]) * occ
    outer = m.copy()
    for i, nb in INNER:
        outer[:, i] -= occ[:, nb].sum(1, dtype=np.int16)
    cols += [m, r, outer]
    for region in SITE_ORBITS:
        if len(region) > 1:
            cols += [outer[:, region].mean(1), r[:, region].mean(1)]
    for sp in (1, 2):
        for threshold in (-1, 0, 1):
            cols.append(((s == sp) & (r <= threshold)).mean(1))
    if name == 'coordination':
        return np.column_stack(cols).astype('f', copy=False)
    good = bonds != NO_SWAP
    cost = np.where(good, bonds, 0)
    # Equal-label sentinel is missing, never a large numeric barrier.
    if name == 'full_class':
        cols.append(np.where(good, cost, np.float32(np.nan)))
        # Difference of neighbor residual moments, identifiable only where an
        # occupied site faces a vacancy. Algebraic readout, no extra halo.
        a, b = s[:, I], s[:, J]; oa, ob = (a != 0).astype(np.int16), (b != 0).astype(np.int16)
        d = ob-oa; ma, mb = m[:, I], m[:, J]; ra, rb = r[:, I], r[:, J]
        ends = ob*(ma-d-IDEAL[b])**2 + oa*(mb+d-IDEAL[a])**2 - ra**2 - rb**2
        moment_diff = 0.5*d*(cost-ends-d*d*(ma-ob+mb-oa)) + rb-ra
        cols.append(np.where(d != 0, moment_diff, np.nan))
    else:
        # Ordered threshold code plus explicit missing state: distinct fit
        # from the continuous/swap-class readout above.
        cols.append(np.where(good, np.searchsorted([-4, -2, 0, 2, 4, 8], cost), -1))
    for ix in EDGE_REGIONS:
        bb = bonds[:, ix]; ok = good[:, ix]; c = cost[:, ix]
        valid = ok.any(1); n = ok.sum(1)
        cols += [n, np.where(valid, bb.min(1), np.nan), c.sum(1) / np.maximum(n, 1)]
        if name == 'full_class':
            si, sj = s[:, I[ix]], s[:, J[ix]]
            for a, b in ((0, 1), (0, 2), (1, 2)):
                mask = ((si == a) & (sj == b)) | ((si == b) & (sj == a))
                n = mask.sum(1)
                cols += [n, (c*mask).sum(1) / np.maximum(n, 1),
                         ((c <= 0) & mask).sum(1)]
        else:
            for t in (-4, -2, 0, 2, 4, 8):
                cols.append(((c <= t) & ok).sum(1))
    return np.column_stack(cols).astype('f', copy=False)


def matrix(data, parents, convert):
    n = parents * data['v'].shape[1]
    for sl, v, m, b in chunks(data, parents):
        x = convert(sl, v, m, b)
        if sl.start == 0:
            out = np.empty((n, x.shape[1]), dtype='f')
        out[sl] = x
    return out


def ridge_binomial(x, y, offset):
    # Count fractions with a common denominator: exact per-site binomial
    # objective; no stochastic hard-label assignment or sample triplication.
    from scipy.optimize import minimize
    from scipy.special import expit
    ridge = 1.e-5
    def objective(w):
        z = offset + w[0] + x @ w[1:].astype('f')
        resid = expit(z) - y
        loss = np.mean(np.logaddexp(0., z) - y*z) + 0.5*ridge*(w[1:]@w[1:])
        grad = np.r_[resid.mean(), (resid.astype('f') @ x)/len(x) + ridge*w[1:]]
        return loss, grad
    fit = minimize(objective, np.zeros(1+x.shape[1]), jac=True, method='L-BFGS-B',
                   options={'maxiter': 220, 'gtol': 2.e-6, 'ftol': 1.e-10})
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise ValueError('nonfinite binomial fit')
    return {'w': fit.x, 'iterations': int(fit.nit), 'status': str(fit.message),
            'gradient_max': float(np.abs(fit.jac).max()), 'objective': float(fit.fun)}


def logit(p):
    return np.log(p) - np.log1p(-p)


def ridge_prob(x, offset, model):
    from scipy.special import expit
    w = model['w']
    return expit(offset + w[0] + x @ w[1:].astype('f'))


def train_models(data, y, parents):
    from sklearn.ensemble import HistGradientBoostingRegressor
    counts = np.zeros(28**2, dtype=np.int64); sums = np.zeros(28**2)
    for sl, v, _, _ in chunks(data, parents):
        k = comp_index(v); counts += np.bincount(k, minlength=len(counts))
        sums += np.bincount(k, weights=y[sl], minlength=len(sums))
    mu = (y.sum()+.5)/(len(y)+1)
    models = {'marginal': mu, 'composition': (sums + 32*mu)/(counts + 32)}
    off = np.empty_like(y)
    for sl, v, _, _ in chunks(data, parents):
        off[sl] = logit(models['composition'][comp_index(v)])
    # Pair model nests the additive coefficient space, with the identical
    # nonlinear composition table as a logit offset.
    for name in ('additive', 'pairs'):
        x = matrix(data, parents, lambda sl, v, m, b: species_features(v, name == 'pairs'))
        models[name] = ridge_binomial(x, y, off)
        if name == 'pairs':
            p_pair = ridge_prob(x, off, models['pairs'])
        del x
    residual = y - p_pair
    for name in HGB_NAMES:
        x = matrix(data, parents, lambda sl, v, m, b: physical_features(
            v, m, b, species_features(v, True), p_pair[sl], name))
        models[name] = HistGradientBoostingRegressor(loss='squared_error', learning_rate=.075,
            max_leaf_nodes=31 if name == 'full_class' else 15, max_iter=100,
            min_samples_leaf=128, max_bins=63, early_stopping=False, random_state=SEED).fit(x, residual)
        del x
    return models


def probabilities(data, models):
    P, R, _ = data['v'].shape
    names = (*LADDER[:-1], *HGB_NAMES[2:])
    out = {k: np.empty(P*R, dtype=np.float64) for k in names}
    for sl, v, m, b in chunks(data, P):
        comp = models['composition'][comp_index(v)]; off = logit(comp)
        pair = species_features(v, True); additive = np.ascontiguousarray(pair[:, :54])
        p = ridge_prob(pair, off, models['pairs'])
        out['marginal'][sl] = models['marginal']; out['composition'][sl] = comp
        out['additive'][sl] = ridge_prob(additive, off, models['additive']); out['pairs'][sl] = p
        for name in HGB_NAMES:
            x = physical_features(v, m, b, pair, p, name)
            out[name][sl] = np.clip(p + models[name].predict(x), CAP, 1-CAP)
    out['full_blend'] = .5*(out['full_class'] + out['full_threshold'])
    return {k: v.reshape(P, R) for k, v in out.items()}


def predict_bundle(bank_path, bundle_path):
    """Trusted private joblib only. Returns frozen unrounded probabilities; never loads y."""
    import joblib
    from threadpoolctl import threadpool_limits
    p = Path(bundle_path)
    if digest(p) != Path(str(p)+'.sha256').read_text().strip():
        raise ValueError('conventional bundle hash mismatch')
    bundle = joblib.load(p)
    if bundle['feature_version'] != BANK_VERSION:
        raise ValueError('incompatible conventional features')
    _, data = open_bank(bank_path)
    with threadpool_limits(limits=4):
        pred = probabilities(data, bundle['models'])
    pred['full'] = pred[bundle['full_member']]
    pred['selected'] = pred[bundle['simplest']]
    return pred


def parent_loss(pred, y_count):
    f = y_count / 3.
    return -(f*np.log(pred) + (1.-f)*np.log1p(-pred)).mean(axis=1)


def fit(a):
    import joblib
    import sklearn
    from threadpoolctl import threadpool_limits
    tm, train = open_bank(a.train, (9, 'train'))
    vm, val = open_bank(a.validation, (9, 'validation'))
    if tm['source_sha256'] != vm['source_sha256'] or tm['beta'] != vm['beta']:
        raise ValueError('mismatched discovery banks')
    groups = partition(9)[0]
    tids, vids = mmap(a.train, 'parent_id'), mmap(a.validation, 'parent_id')
    if not np.array_equal(tids, groups[0]) or not np.array_equal(vids, groups[1]):
        raise ValueError('expected original canonical training/validation order')
    tk, vk = mmap(a.train, 'parent_keys'), mmap(a.validation, 'parent_keys')
    if len(np.unique(np.concatenate((tk, vk)), axis=0)) != len(tids)+len(vids):
        raise ValueError('shared or duplicate discovery lineage')
    cut = 3*tm['parents']//4
    if tm['fit_parents'] != cut:
        raise ValueError('calibration partition mismatch')
    y = mmap(a.train, 'y')[:cut].reshape(-1) / 3.
    with threadpool_limits(limits=4):
        models = train_models(train, y, cut)
        probs = probabilities(val, models)
    val_y = mmap(a.validation, 'y')
    loss = {k: parent_loss(v, val_y) for k, v in probs.items()}
    means = {k: float(v.mean()) for k, v in loss.items()}
    full = min(('full_class', 'full_threshold', 'full_blend'), key=means.get)
    loss['full'] = loss[full]; means['full'] = means[full]
    strongest = min(LADDER, key=means.get)
    simple = next(k for k in LADDER if means[k] <= means[strongest] + TOL)
    loss['selected'] = loss[simple]; means['selected'] = means[simple]
    a.work.mkdir(parents=True, exist_ok=True); a.out.mkdir(parents=True, exist_ok=True)
    bundle = a.work/'conventional.joblib'
    bundle_info = dict(feature_version=BANK_VERSION, full_member=full, simplest=simple,
                       fit_parents=cut, source_sha256=tm['source_sha256'], sklearn_version=sklearn.__version__)
    joblib.dump(dict(**bundle_info, models=models), bundle, compress=3)
    sha = digest(bundle); sidecar = Path(str(bundle)+'.sha256'); sidecar.write_text(sha+'\n')
    if digest(bundle) != sidecar.read_text().strip():
        raise IOError('serialized bundle did not verify')
    table = a.out/'baseline-validation.csv'
    with table.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parent_id', 'chain_seed', 'chain_id', 'anchors', 'outward_pixels', *loss])
        for i, row in enumerate(vids):
            writer.writerow([int(row), *map(int, vk[i]), vm['reps'], 3*vm['reps'], *(v[i] for v in loss.values())])
    def summarize(k):
        diff = loss[k] - loss['full']
        return dict(loss=means[k], mean_minus_full=float(diff.mean()),
                    se_paired_minus_full=float(diff.std(ddof=1)/np.sqrt(len(diff))))
    dump(a.out/'conventional-fits.json', dict(**bundle_info, model_sha256=sha,
         feature_code_sha256=digest(Path(__file__)), validation_csv_sha256=digest(table),
         banks={k: dict(meta_sha256=digest(path/'meta.json'), **{
             x: meta[x] for x in ('index','beta','fold','parents','reps','source_sha256','lineage_sha256','sha256')})
             for k,path,meta in [('train',a.train,tm),('validation',a.validation,vm)]},
         reserved_calibration=dict(first_training_row=cut, parents=len(tids)-cut,
             fit_row_id_sha256=hashlib.sha256(tids[:cut].astype('<u8').tobytes()).hexdigest(),
             calibration_row_id_sha256=hashlib.sha256(tids[cut:].astype('<u8').tobytes()).hexdigest()),
         unit='nats per outward pixel; average of three marginal Bernoulli losses, mean over origins within parent',
         selection=dict(tolerance_nats=TOL, simplicity_order=LADDER, minimum_validation=strongest,
             rule='first within fixed absolute tolerance of minimum mean parent loss; full is independently the minimum among FULL readouts, never replaced by a simpler model'),
         physical_scope='Fixed conventional readouts only; a residual cannot rule out all physical observables/readouts.',
         fit_description=dict(composition='nonlinear count table shrunk by 32 anchor equivalents toward marginal',
             additive='binomial ridge logit; both species at all 27 sites plus composition offset',
             pairs='additive plus internal separation-averaged AA/AB/BB and target-corner-rooted two-point profiles',
             species_nonlinear='HGB on species/pair features, residual around binomial pairs',
             coordination='HGB adds per-site coordination, deviations, exterior occupancy and pooled thresholds',
             full_class='adds oriented internal costs, recoverable moment contrasts, class-specific summaries',
             full_threshold='separate fit adds threshold-encoded internal edges and cumulative threshold counts',
             full_blend='fixed equal probability average of class and threshold full readouts'),
         hgb={k: models[k].get_params() for k in HGB_NAMES},
         ridge={k: {name:value for name,value in models[k].items() if name!='w'} for k in ('additive','pairs')},
         validation={k:summarize(k) for k in loss}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('bank')
    b.add_argument('--data', type=Path, required=True); b.add_argument('--index', type=int, required=True)
    b.add_argument('--fold', choices=FOLDS, required=True); b.add_argument('--reps', type=int, required=True)
    b.add_argument('--out', type=Path, required=True); b.add_argument('--lock', type=Path)
    f = sub.add_parser('fit')
    for k in ('train','validation','work','out'):
        f.add_argument('--'+k, type=Path, required=True)
    a = parser.parse_args()
    (bank if a.cmd == 'bank' else fit)(a)


if __name__ == '__main__':
    main()
