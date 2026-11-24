"""Cold-only deterministic four-symbol optimization for the common physical face channel."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from threadpoolctl import threadpool_limits


NV, NJ, NY, K = 3**8, 6, 3**4, 4
MAX_SWEEPS = 256
ROWS = np.arange(NV)
DIGITS = ROWS[:, None] // (3 ** np.arange(8)) % 3
AXES = 1 - 2 * np.array(list(itertools.product(range(2), repeat=3)))
TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
# Intercept, eight A occupations, eight B occupations. Vacancy is the reference.
X = np.column_stack((np.ones(NV, dtype=int), DIGITS == 1, DIGITS == 2))
HOT = np.eye(K)


def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def logmass(table):
    return xlogy(table, table).sum() / np.log(2)


class Channel:
    """Normalized joint P(V,W) with W=(J,Y); J independent of V by construction."""

    def __init__(self, counts, smooth):
        t = np.array(counts.transpose(1, 0, 2), dtype=float, order='C')
        if smooth:
            marginal = t.sum(axis=0) + 0.5
            t += marginal / marginal.sum(axis=1, keepdims=True)
        self.joint = t.reshape(NV, NJ * NY)
        self.joint /= self.joint.sum()
        self.p = self.joint.sum(axis=1)
        self.pw = self.joint.sum(axis=0)
        self.lw = logmass(self.pw)
        self.full = float(logmass(self.joint) - logmass(self.p) - self.lw)

    def fit(self, h):
        joint = HOT[h].T @ self.joint
        mass = joint.sum(axis=1)
        q = np.divide(joint, mass[:, None], out=np.tile(self.pw, (K, 1)), where=mass[:, None] > 0)
        return mass, q, float(logmass(joint) - logmass(mass) - self.lw)

    def describe(self, h):
        mass, _, info = self.fit(h)
        entropy = float(-logmass(mass))
        return dict(mi_bits=info, kl_distortion_bits=self.full - info,
                    info_fraction=info / self.full if self.full > 0 else None,
                    rate_H_H_given_J_bits=entropy, H_H_given_YJ_bits=entropy - info,
                    nominal_bits=2, symbol_mass=mass.tolist(),
                    collapsed=np.flatnonzero(mass == 0).tolist())


def load_bank(bank):
    meta = json.loads((bank / 'meta.json').read_text())
    if meta['index'] != 9 or meta['anchors_per_parent'] != 8000 or not meta['train_sha256']:
        raise ValueError('optimizer requires the cold native training bank')
    path = bank / 'train.npy'
    if digest(path) != meta['train_sha256']:
        raise ValueError('training bank checksum mismatch')
    counts = np.load(path, mmap_mode='r', allow_pickle=False)
    if counts.shape != (NJ, NV, NY) or counts.dtype.kind not in 'iu' or counts.min() < 0:
        raise ValueError('expected a nonnegative integer (6,6561,81) train table')
    rows = counts.sum(axis=2, dtype=np.int64)
    if not np.all(rows == rows[0]) or rows[0].sum() != 8000 * meta['train_parents']:
        raise ValueError('inconsistent directional tile counts or training multiplicity')
    return meta, Channel(counts, False), Channel(counts, True)


def load_map(archive, name):
    h = archive[f'{name}_lut']
    if h.shape != (NV,) or h.dtype.kind not in 'iu' or h.min() < 0 or h.max() >= K:
        raise ValueError(f'invalid four-symbol map {name}')
    return h.astype(np.uint8)


def spectral(ch):
    residual = ch.joint / ch.p[:, None] - ch.pw
    residual /= np.sqrt(ch.pw)
    weighted = residual * np.sqrt(ch.p[:, None])
    values, vectors = np.linalg.eigh(weighted.T @ weighted)
    basis = vectors[:, -1:-4:-1].copy()
    basis *= np.where(basis[np.abs(basis).argmax(axis=0), np.arange(3)] < 0, -1, 1)
    return residual @ basis, values[-1:-4:-1].tolist()


def project(target, p):
    """Frequency-weighted least squares; least-norm coefficients work even on sparse support."""
    w = np.sqrt(p[:, None])
    return np.linalg.lstsq(X * w, target * w, rcond=None)[0]


def quantize(coef):
    # Removing a shared affine score leaves the hard classifier unchanged.
    coef = coef - coef.mean(axis=1, keepdims=True)
    scale = np.max(np.abs(coef))
    return np.rint(coef * (4096 / scale if scale > 0 else 1)).astype(np.int64)


def affine_starts(coords, p, original):
    physical = np.zeros((X.shape[1], K), dtype=np.int64)
    physical[1:9] = AXES @ TETRA.T
    return {'tetra_A': physical,
            'rsmi_additive': quantize(project(HOT[original], p)),
            'face_spectral_additive': quantize(project(coords, p) @ TETRA.T)}


def affine_sweep(c, h, coef, subsets, reward):
    """Exact integer coordinate steps on the fixed-decoder expected log score."""
    for k in range(K):
        others = np.delete(np.arange(K), k)
        # Other logits stay fixed throughout the coordinates of class k.
        alt = others[c[:, others].argmax(axis=1)]
        alt_score = c[ROWS, alt]
        delta = reward[:, k] - reward[ROWS, alt]
        for a, ix in enumerate(subsets):
            # At each breakpoint d, the tile first votes for k at integer step d.
            breaks = alt_score[ix] - c[ix, k] + (alt[ix] < k)
            order = np.argsort(breaks, kind='stable')
            b = breaks[order]
            last = np.flatnonzero(np.r_[b[:-1] != b[1:], True])
            prefix = delta[ix[order]].cumsum()[last]
            best = int(prefix.argmax())
            current = delta[ix[h[ix] == k]].sum()
            if max(0.0, prefix[best]) <= current + 1e-14:
                continue
            step = b[last[best]] if prefix[best] > 0 else b[0] - 1
            coef[a, k] += step
            c[ix, k] += step
            h[ix] = np.where(breaks <= step, k, alt[ix])


def iterate(ch, raw, initial, coef=None):
    h = initial.copy()
    if coef is not None:
        coef = coef.copy()
        logits = X @ coef
        subsets = [np.flatnonzero(col) for col in X.T]
    steps = []
    reason = 'sweep_cap'
    mass, q, previous = ch.fit(h)
    for sweep in range(1, MAX_SWEEPS + 1):
        old = h.copy()
        reward = ch.joint @ np.log2(q).T
        if coef is None:
            best = reward.argmax(axis=1)
            move = reward[ROWS, best] - reward[ROWS, h] > 1e-13 * ch.p
            h[move] = best[move]
        else:
            affine_sweep(logits, h, coef, subsets, reward)
        mass, q, info = ch.fit(h)
        changed = int(np.count_nonzero(h != old))
        if info < previous - 2e-12:
            raise ArithmeticError(f'train objective decreased at sweep {sweep}: {previous} -> {info}')
        steps.append(dict(sweep=sweep, changed_tiles=changed, mi_bits=info,
                          kl_distortion_bits=ch.full - info,
                          collapsed=np.flatnonzero(mass == 0).tolist()))
        previous = info
        if not changed:
            reason = 'fixed_assignments' if coef is None else 'no_improving_coordinate'
            break
    if coef is not None and not np.array_equal((X @ coef).argmax(axis=1), h):
        raise ArithmeticError('affine coefficients no longer reproduce train LUT')
    return h, coef, dict(initial=ch.describe(initial), final=ch.describe(h),
                         raw_initial=raw.describe(initial), raw_final=raw.describe(h),
                         termination=reason, trajectory=steps)


def select(runs):
    eligible = [name for name, rec in runs.items() if not rec['raw_final']['collapsed']]
    if not eligible:
        raise RuntimeError('all fits in this family have empirically collapsed states')
    return max(eligible, key=lambda name: runs[name]['final']['mi_bits'])


def alignment(p, target, candidate):
    matrix = np.bincount(K * target.astype(int) + candidate, weights=p, minlength=K*K).reshape(K, K)
    permutation = max(itertools.permutations(range(K)), key=lambda cols: matrix[np.arange(K), cols].sum())
    return dict(confusion_mass=matrix.tolist(), candidate_label_for_target=list(permutation),
                best_relabel_accuracy=float(matrix[np.arange(K), permutation].sum()))


def sectors(features, p, h, entropy):
    keys, inverse = np.unique(features, axis=0, return_inverse=True)
    joint = np.bincount(K * inverse.ravel() + h, weights=p, minlength=len(keys)*K).reshape(-1, K)
    total = joint.sum(axis=1)
    conditional = float(logmass(total) - logmass(joint))
    # Only print the heavy sectors; scores are always over all sectors with positive mass.
    top = np.argsort(-total, kind='stable')[:32]
    return dict(positive_sectors=int(np.count_nonzero(total)), H_H_given_sector_bits=conditional,
                explained_entropy_fraction=1 - conditional / entropy,
                majority_accuracy=float(joint.max(axis=1).sum()),
                top_table=[dict(sector=keys[i].tolist(), symbol_mass=joint[i].tolist())
                           for i in top if total[i] > 0])


def distill(p, h, original, affine):
    a, b = DIGITS == 1, DIGITS == 2
    composition = np.column_stack((a.sum(axis=1), b.sum(axis=1)))
    dip_a, dip_b = a @ AXES, b @ AXES
    physical = np.column_stack((composition, dip_a, dip_b))
    mass = np.bincount(h, weights=p, minlength=K)
    entropy = float(-logmass(mass))
    labels = HOT[h]
    coef = project(labels, p)
    prediction = X @ coef
    additive = alignment(p, h, prediction.argmax(axis=1))
    additive.update(accuracy=float(np.trace(additive['confusion_mass'])),
                    explained_onehot_variance=float(1 - (p @ ((labels-prediction)**2).sum(axis=1)) /
                                                     (1 - mass @ mass)),
                    coefficients=coef.tolist())
    return dict(empirical_symbol_mass=mass.tolist(),
                feature_columns=['n_A', 'n_B', 'A_z', 'A_y', 'A_x', 'B_z', 'B_y', 'B_x'],
                mean_feature_given_symbol=(((labels*p[:, None]).T @ physical) / mass[:, None]).tolist(),
                composition=sectors(composition, p, h, entropy),
                signed_A_dipole=sectors(dip_a, p, h, entropy),
                composition_and_signed_A_dipole=sectors(physical[:, :5], p, h, entropy),
                additive_label_recovery=additive,
                historical_alignment=alignment(p, h, original), affine_alignment=alignment(p, h, affine))


def train(bank, base, out):
    meta, raw, ch = load_bank(bank)
    with np.load(base, allow_pickle=False) as f:
        original, physical = (load_map(f, name) for name in ('rsmi', 'kmeans_A_dipole'))
    coords, eigenvalues = spectral(ch)
    bundle = {}
    records = {'affine': {}, 'discrete': {}}
    winners = {}

    def fit_family(family, starts):
        for name, (initial, coef) in starts.items():
            prefix = f'{family}_{name}'
            h, coef, rec = iterate(ch, raw, initial, coef)
            bundle[f'{prefix}_initial'] = initial
            bundle[f'{prefix}_final'] = h
            if coef is not None:
                bundle[f'{prefix}_coef'] = coef
            records[family][name] = rec
        best = select(records[family])
        winners[family] = best
        bundle[f'{family}_lut'] = bundle[f'{family}_{best}_final']

    fit_family('affine', {key: ((X @ c).argmax(axis=1).astype(np.uint8), c)
                          for key, c in affine_starts(coords, ch.p, original).items()})
    bundle['affine_coef'] = bundle[f"affine_{winners['affine']}_coef"]
    fit_family('discrete', {key: (h, None) for key, h in {
        'rsmi': original, 'kmeans_A_dipole': physical,
        'face_spectral': (coords @ TETRA.T).argmax(axis=1).astype(np.uint8),
        'trained_affine': bundle['affine_lut']}.items()})
    before = ch.describe(original)['mi_bits']
    after = records['discrete']['rsmi']['final']['mi_bits']
    selected = {name: records[name][winner]['final']['mi_bits'] for name, winner in winners.items()}
    report = dict(source={key: meta[key] for key in ('index', 'beta', 'buffer', 'source_sha256',
                                                     'train_sha256', 'train_parents', 'anchors_per_parent')},
                  base_sha256=digest(base), optimizer_sha256=digest(Path(__file__)),
                  observed_tiles=int(np.count_nonzero(raw.p)), unseen_tiles=int(np.count_nonzero(raw.p == 0)),
                  full_tile_mi_bits={'smoothed': ch.full, 'raw': raw.full},
                  spectral_eigenvalues=eigenvalues, runs=records, winners=winners,
                  comparison_smoothed_train_bits=dict(historical=before, historical_refined=after,
                                                      historical_refinement_gain=after-before, **selected,
                                                      discrete_minus_historical=selected['discrete']-before,
                                                      discrete_minus_affine=selected['discrete']-selected['affine']),
                  scope='Finite smoothed cold training objective, local alternating fits; no global optimality or generalization claim.',
                  distillation=distill(raw.p, bundle['discrete_lut'], original, bundle['affine_lut']))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'optimized.npz', **bundle)
    report['optimized_sha256'] = digest(out / 'optimized.npz')
    (out / 'optimization.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'base', 'out'):
        ap.add_argument(f'--{name}', type=Path, required=True)
    args = ap.parse_args()
    with threadpool_limits(limits=1, user_api='blas'):
        train(args.bank, args.base, args.out)


if __name__ == '__main__':
    main()
