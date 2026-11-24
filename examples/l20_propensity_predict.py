"""Predict local species-change probabilities on held-out L20 parent configurations.

One file/fold per cohort, one fit per model and lag. No dynamics or encoder
training here. The Bernoulli event refers to the state at t, not a tagged
particle or a first passage (a return to the starting species is a non-event).
"""
import argparse
import csv
import hashlib
import itertools
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from l20_pairs import partition


SEEDS = (11, 29, 47)
CORNERS = tuple(itertools.product(range(2), repeat=3))  # z, y, x; x fastest
IDEAL = np.array([0, 3, 5], dtype=np.int16)
NO_SWAP = np.iinfo(np.int16).max
HGB = dict(loss='squared_error', max_iter=40, max_depth=3, learning_rate=0.1,
           min_samples_leaf=96, max_bins=63, early_stopping=False, random_state=20261125)
CAL_EDGES = np.array([0., .05, .2, .5, .8, 1.])
# Species/m followed by means and SDs of three cube counts, min swap cost,
# then mean posterior q(1),q(2) for three learned and three random encoders.
EXTENSIONS = {'geometry': np.r_[0:8], 'physical': np.r_[0:2, 8], 'barrier': np.r_[0:9],
              'local_rsmi': np.r_[0:2, 9:15], 'local_random': np.r_[0:2, 15:21],
              'rsmi': np.r_[0:15], 'random': np.r_[0:9, 15:21]}


def sha(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def encoder_weights(path, cohort):
    # Never apply the beta-09 encoder to archive-09: unlabeled pretraining
    # there may already have seen some of the archived trajectory parents.
    source = 8 if cohort == 'archive-09' else 9
    weights, random, receipts = [], [], []
    with zipfile.ZipFile(path) as archive:
        for seed in SEEDS:
            member = f'fits/beta-{source:02d}_b1_learned_{seed}.json'
            blob = archive.read(member)
            r = json.loads(blob)
            if (r['index'], r['buffer'], r['method'], r['seed']) != (source, 1, 'learned', seed):
                raise ValueError(f'wrong frozen receipt: {member}')
            w = np.array(r['weights'], dtype=np.float64)
            if w.shape != (8,) or not np.isfinite(w).all() or not np.linalg.norm(w) > 0:
                raise ValueError(f'invalid weights: {member}')
            # The actual iid Gaussian initialization direction, never optimized;
            # match only its norm to the corresponding validation-selected fit.
            v = np.random.default_rng(seed).normal(size=8)
            v *= np.linalg.norm(w) / np.linalg.norm(v)
            weights.append(w)
            random.append(v)
            receipts.append(dict(member=member, sha256=hashlib.sha256(blob).hexdigest(),
                                 beta=r['beta'], source_sha256=r['provenance']['source_sha256'],
                                 encoder_test_sha256=r['provenance']['parent_sha256'][2],
                                 seed=seed, learned=w.tolist(), random=v.tolist()))
    return np.array(weights + random), receipts


def cube_table(weights):
    """All 3**8 patterns; average posterior, NOT weights/logits, over 48 isometries."""
    digits = np.indices((3,) * 8, dtype=np.int16).reshape(8, -1).T
    counts = np.stack([(digits == s).sum(1) for s in range(3)], axis=1)
    logits = np.stack([(digits == s) @ weights.T for s in range(3)], axis=-1)
    prob = np.exp(logits - logits.max(axis=-1, keepdims=True))
    prob /= prob.sum(axis=-1, keepdims=True)
    prob = prob[:, :, 1:].reshape(len(digits), -1)  # vacancy probability is redundant
    orbit = np.empty((48, len(digits)), dtype=np.int16)
    cube = np.arange(8).reshape((2,) * 3)
    powers = 3 ** np.arange(7, -1, -1)
    transforms = itertools.product(itertools.permutations(range(3)), itertools.product((1, -1), repeat=3))
    for k, (perm, signs) in enumerate(transforms):
        order = cube.transpose(perm)[tuple(slice(None, None, sign) for sign in signs)].ravel()
        orbit[k] = digits[:, order] @ powers
    # One actual sum per orbit representative avoids roundoff-based distinctions
    # between patterns differing only by a rigid rotation/reflection.
    representatives, inverse = np.unique(orbit.min(axis=0), return_inverse=True)
    sym = prob[orbit[:, representatives]].mean(axis=0)
    return np.column_stack((counts, counts**2, sym[inverse]))


def neighbor_sum(a):
    return sum(np.roll(a, d, axis) for axis in range(3) for d in (-1, 1))


def coordination_barrier(state):
    """Exact periodic score of lattice-exploration/pilots/l20-b310/paper_barriers.py."""
    occ = (state != 0).astype(np.int16)
    m = neighbor_sum(occ)
    r = (m - IDEAL[state]) * occ
    moment, h = neighbor_sum(r), r**2
    minimum = np.full(state.shape, NO_SWAP, dtype=np.int16)
    for axis in range(3):
        sb, ob, mb, rb, moment_b, hb = (np.roll(a, -1, axis) for a in (state, occ, m, r, moment, h))
        d = ob - occ
        # Endpoints exchange species. The remaining five neighbors of each
        # endpoint gain/lose d occupied neighbors; the two sets are disjoint.
        dh = ob * (m - d - IDEAL[sb])**2 + occ * (mb + d - IDEAL[state])**2 - h - hb
        dh += 2*d*(moment - rb - moment_b + r) + d*d*(m - ob + mb - occ)
        dh[state == sb] = NO_SWAP
        np.minimum(minimum, dh, out=minimum)
        np.minimum(minimum, np.roll(dh, 1, axis), out=minimum)
    return m, minimum


def verify_barrier(state, barrier):
    """Independent literal whole-H differences on one occupied site per parent."""
    def total(s):
        halo = np.pad((s != 0).astype(np.int16), 1, mode='wrap')
        m = (halo[:-2, 1:-1, 1:-1] + halo[2:, 1:-1, 1:-1] + halo[1:-1, :-2, 1:-1]
             + halo[1:-1, 2:, 1:-1] + halo[1:-1, 1:-1, :-2] + halo[1:-1, 1:-1, 2:])
        return int(((m-3)**2 * (s == 1) + (m-5)**2 * (s == 2)).sum())

    probe = np.unravel_index(np.flatnonzero(state)[0], state.shape)
    old, ref, swapped = total(state), NO_SWAP, state.copy()
    for axis, step in itertools.product(range(3), (-1, 1)):
        nb = list(probe)
        nb[axis] = (nb[axis] + step) % state.shape[axis]
        nb = tuple(nb)
        if state[probe] != state[nb]:
            swapped[nb], swapped[probe] = state[probe], state[nb]
            ref = min(ref, total(swapped) - old)
            swapped[probe], swapped[nb] = state[probe], state[nb]
    if ref != barrier[probe]:
        raise ValueError(f'whole-H swap check failed at {probe}: {ref} != {barrier[probe]}')


def features(state, lut):
    m, barrier = coordination_barrier(state)
    verify_barrier(state, barrier)
    code = np.zeros(state.shape, dtype=np.int16)
    axes = (0, 1, 2)
    for corner in CORNERS:
        code *= 3
        code += np.roll(state, tuple(-i for i in corner), axes)
    anchored = lut[code]
    # A given site belongs to the cubes anchored at site - corner. Pool all
    # eight, including the same target site under each corner convention.
    means = sum(np.roll(anchored, corner, axes) for corner in CORNERS) / 8
    fields = means[state != 0]
    fields[:, 3:6] = np.sqrt(np.maximum(0, fields[:, 3:6] - fields[:, :3]**2))
    order = state != 0
    return np.column_stack((state[order], m[order], fields[:, :6], barrier[order], fields[:, 6:])).astype(np.float32)


def correlation(a, b):
    # Checking max/min explicitly handles identical non-dyadic probabilities
    # whose floating-point mean need not equal the repeated value.
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    x, y = a - np.mean(a, dtype=np.float64), b - np.mean(b, dtype=np.float64)
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def group_mean(group, y, weight, prior):
    # One pseudo-site at the parent-weighted coarser group's training estimate.
    size = len(prior)
    return ((np.bincount(group, weight*y, minlength=size) + prior)
            / (np.bincount(group, weight, minlength=size) + 1))


def predict(x, y_train, train_weight, ntrain):
    species = x[:, 0].astype(np.intp) - 1
    joint = 7 * species + x[:, 1].astype(np.intp)
    mu = np.average(y_train, weights=train_weight)
    p_species = group_mean(species[:ntrain], y_train, train_weight, np.full(2, mu))
    p_local = group_mean(joint[:ntrain], y_train, train_weight, np.repeat(p_species, 7))
    base = p_local[joint]
    outputs = {'species': p_species[species[ntrain:]], 'local': base[ntrain:]}
    m = x[:, 1].astype(np.intp)
    h = (m - IDEAL[species+1])**2
    for name, group, prior in [('coordination', m, np.full(7, mu)), ('energy', h, np.full(26, mu)),
                               ('species_energy', 26*species+h, np.repeat(p_species, 26))]:
        outputs[name] = group_mean(group[:ntrain], y_train, train_weight, prior)[group[ntrain:]]
    residual = y_train - base[:ntrain]
    for name, cols in EXTENSIONS.items():
        # Fit increments over the SAME saturated species x coordination baseline;
        # none of the larger regressors must approximate this interaction anew.
        model = HistGradientBoostingRegressor(**HGB)
        model.fit(x[:ntrain, cols], residual, sample_weight=train_weight)
        outputs[name] = np.clip(base[ntrain:] + model.predict(x[ntrain:, cols]), 0, 1)
    return outputs, joint[ntrain:], dict(species=p_species.tolist(), local=p_local.reshape(2, 7).tolist())


def check_input(data):
    cohort, i = str(data['cohort'].item()), int(data['index'].item())
    if i not in range(10) or not (cohort == f'archive-{i:02d}' or (cohort == 'cold-replicated' and i == 9)):
        raise ValueError('invalid cohort/index')
    pids, origin = data['parent_id'], data['origin']
    if pids.dtype != np.uint64 or pids.ndim != 1 or len(np.unique(pids)) != len(pids) or len(pids) < 2:
        raise ValueError('expected distinct uint64 parent IDs')
    if origin.dtype != np.uint8 or origin.shape != (len(pids), 20, 20, 20) or origin.max() > 2:
        raise ValueError('invalid ternary L20 initial states')
    if not np.all(origin.any(axis=(1, 2, 3))):
        raise ValueError('a parent has no occupied sites')
    if cohort == 'cold-replicated' and not np.isin(pids, partition(9)[0][2]).all():
        raise ValueError('cold parent IDs must be beta-09 endpoint rows from the encoder test partition')
    r = int(data['replicas'].item())
    lags, roles = data['lags'], data['roles']
    if (lags.ndim != 1 or lags.dtype != np.int64 or not len(lags) or np.any(lags <= 0)
            or len(np.unique(lags)) != len(lags) or roles.shape != lags.shape or roles.dtype.kind != 'U'
            or not np.isin(roles, ['early', 'alpha', 'early+alpha', 'short', 'primary']).all()):
        raise ValueError('invalid lag/role array')
    changed, half = data['changed'], data['half_changed']
    if (r not in range(1, 256) or changed.dtype != np.uint8 or half.dtype != np.uint8
            or changed.shape != (len(lags), *origin.shape) or half.shape != changed.shape
            or changed.max() > r or half.max() > r//2 or (half > changed).any() or (changed - half).max() > r-r//2):
        raise ValueError('invalid sibling sufficient statistics')


def run(path, fits, out):
    with np.load(path, allow_pickle=False) as f:
        data = {key: f[key] for key in ('cohort', 'index', 'beta', 'replicas', 'parent_id', 'origin',
                                        'lags', 'roles', 'changed', 'half_changed')}
    check_input(data)
    cohort, index, beta, r = (cast(data[k].item()) for k, cast in
                              [('cohort', str), ('index', int), ('beta', float), ('replicas', int)])
    ids, origin = data['parent_id'], data['origin']
    members = np.argsort(ids)
    seed = 20261125 + index
    perm = np.random.default_rng(seed).permutation(len(ids))
    # Fixed sorted-ID order INSIDE each assigned fold too; independent of file ordering.
    train, test = (members[np.sort(g)] for g in np.split(perm, [len(ids)//2]))
    order = np.r_[train, test]
    weights, kernel_receipts = encoder_weights(fits, cohort)
    lut = cube_table(weights)
    sizes = np.count_nonzero(origin, axis=(1, 2, 3))[order]
    starts = np.r_[0, sizes.cumsum()]
    ntrain = starts[len(train)]
    x = np.empty((starts[-1], 21), dtype=np.float32)
    for j, row in enumerate(order):
        x[starts[j]:starts[j+1]] = features(origin[row], lut)
    train_sizes = sizes[:len(train)]
    train_weight = np.repeat(train_sizes.mean() / train_sizes, train_sizes)
    occupied = origin[order] != 0
    metadata = dict(cohort=cohort, index=index, beta=beta, replicas=r,
                    target='Origin occupied; probability of a different species OR vacancy at lag. R=1 is a realized event, not a measured site propensity.',
                    input_file=path.name, input_sha256=sha(path), fits_file=fits.name, fits_sha256=sha(fits),
                    script_sha256=sha(Path(__file__)), python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                    split_seed=seed, train_parent_id=[str(v) for v in ids[train]], test_parent_id=[str(v) for v in ids[test]],
                    split='default_rng(seed).permutation of sorted uint64 IDs, first floor(P/2) train. All sites, lags and children stay with their parent.',
                    encoder_receipts=kernel_receipts, frozen_source='beta-08 for archive-09 only; beta-09 for all other cohorts. Cold parent_id checked against beta-09 encoder-test row indices.',
                    model_definitions=dict(species='Species mean, one pseudo-site at the train mean.',
                                           coordination='Saturated m only.', energy='Saturated on-site h only.', species_energy='Saturated species x on-site h.',
                                           local_rsmi='local plus frozen learned features only, without cube-count or barrier inputs.',
                                           local_random='local plus matched random features only, without cube-count or barrier inputs.',
                                           local='Saturated species x occupied six-neighbor count m, one pseudo-site at the species mean. Local h=(m-l_species)^2, l_1=3,l_2=5, is deterministic here.',
                                           geometry='local plus mean and SD of each label count across eight incident periodic 2x2x2 cubes.',
                                           physical='local plus exact min delta-H over unlike nearest-neighbor swaps. Empty minimum encoded as 32767.',
                                           barrier='geometry plus the exact min delta-H as in physical.',
                                           rsmi='barrier plus q(1),q(2) for each of three frozen learned encoders: average softmax over all 48 cube symmetries, then over all eight incident cubes.',
                                           random='Same as rsmi, but untrained Gaussian initialization directions rescaled to the three learned L2 norms.',
                                           rsmi_shuffle='Permute full rsmi score among occupied sites with identical (species,m), within each test parent, keep labels fixed. PCG64 seed sequence [20261125+index, lag, parent_id].'),
                    regression='Fit squared-error HGB to training ybar minus local; add local back, clip to [0,1]. All parents equal total fit weight, mean weight per site 1.',
                    hgb=HGB, score='Brier = mean over sites of (p-ybar)^2+ybar*(1-ybar); correlation is within-parent Pearson with ybar.',
                    reliability='Within-parent Pearson between first floor(R/2) and remaining child-mean maps on occupied sites; NaN for R=1 or constant map.',
                    calibration='Equal parent weight; for each bin [left,right) (last includes 1): [mass, mean score, observed fraction]; null means empty bin.',
                    calibration_edges=CAL_EDGES.tolist(), per_lag=[])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (lag, role) in enumerate(zip(data['lags'], data['roles'])):
        lag, role = int(lag), str(role)
        y = data['changed'][k, order][occupied].astype(np.float64) / r
        half = data['half_changed'][k, test][occupied[len(train):]].astype(np.float64)
        outputs, group, means = predict(x, y[:ntrain], train_weight, ntrain)
        outputs['rsmi_shuffle'] = outputs['rsmi'].copy()
        cal = {name: np.zeros((len(CAL_EDGES)-1, 3)) for name in outputs}
        for j, row in enumerate(test, start=len(train)):
            a, b = starts[j:j+2] - ntrain
            p_id = int(ids[row])
            sl = slice(a, b)
            observed = y[ntrain+a:ntrain+b]
            split_corr = correlation(half[sl]/(r//2), (observed*r - half[sl])/(r-r//2)) if r >= 2 else np.nan
            rng = np.random.default_rng([seed, lag, p_id])
            for key in np.unique(group[sl]):
                ix = np.flatnonzero(group[sl] == key) + a
                outputs['rsmi_shuffle'][ix] = rng.permutation(outputs['rsmi'][ix])
            common = dict(cohort=cohort, index=index, beta=beta, lag=lag, role=role, replicas=r, parent_id=p_id,
                          mobility=observed.mean(), split_half_correlation=split_corr)
            noise = np.mean(observed * (1 - observed))
            for model, pred in outputs.items():
                p = pred[sl]
                rows.append(dict(**common, model=model, brier=noise + np.mean((p-observed)**2),
                                 correlation=correlation(p, observed), mean_prediction=p.mean()))
                bins = np.searchsorted(CAL_EDGES[1:-1], p, side='right')
                for col, w in enumerate((None, p, observed)):
                    cal[model][:, col] += np.bincount(bins, weights=w, minlength=len(CAL_EDGES)-1)/len(p)
            if cohort == 'cold-replicated' and role == 'primary' and lag == 4096 and row == test[0]:
                example = dict(species=origin[row], observed=data['changed'][k, row].astype(np.float32)/r,
                               parent_id=ids[row], lag=lag, replicas=r)
                for name in ('local', 'barrier', 'rsmi'):
                    volume = np.full(origin[row].shape, np.nan, dtype=np.float32)
                    volume[origin[row] != 0] = outputs[name][sl]
                    example[name] = volume
                np.savez_compressed(out/f'{cohort}-example.npz', **example)
        calibration = {model: [[n/len(test), ps/n if n else None, ys/n if n else None] for n, ps, ys in vals]
                       for model, vals in cal.items()}
        metadata['per_lag'].append(dict(lag=lag, role=role, training_baseline=means, calibration=calibration))
    with (out/f'{cohort}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, rows[0]); writer.writeheader(); writer.writerows(rows)
    (out/f'{cohort}-meta.json').write_text(json.dumps(metadata, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--fits', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.input, args.fits, args.out)
