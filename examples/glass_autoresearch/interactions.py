"""Held-out conditional interaction hierarchy for the archived four-ternary-site quotient.

Run directly: python examples/glass_autoresearch/interactions.py --data DIR --out DIR
No sampler or test evaluation. The pre-specified secondary analysis is in
results/l20-autoresearch/interactions/design.json. All log scores are in nats.
"""
import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import rel_entr
from scipy.stats import t

from quotient import BUFFERS, marginals, partition, partitions, solutions, tables


INDICES = (9, 0, 5, 7, 8)  # Freeze the cold code first, without validation feedback.
CELLS = (3,) * 4
AXES = tuple(range(4))
TOL = 2e-13
BAND = 1e-5
DELETE_GROUPS = 16
WITHIN = ((0, 1), (2, 3))
CROSS = {f'edge_V{i+1}_Y{j-1}': (i, j) for i in (0, 1) for j in (2, 3)}
SCORES = [(m, 'null') for m in ('single', 'pair', 'triple', 'sat', *CROSS)] + [
    ('sat', 'pair'), ('triple', 'pair'), ('sat', 'triple'),
    ('K4_pair', 'null'), ('K4_sat', 'null'), ('K4_sat', 'K4_pair'), ('sat', 'K4_sat'),
] + [('pair', m) for m in CROSS]
REFITS = [(m, 'null') for m in ('pair', 'triple', 'sat')] + [
    ('sat', 'pair'), ('triple', 'pair'), ('sat', 'triple')]
# Three validation intervals per contrast, one two-sample interval per refit,
# and two conditional Fieller sets. Nominal intervals are reported separately.
FAMILY = len(INDICES) * len(BUFFERS) * (3 * len(SCORES) + len(REFITS) + 2)


def mass(counts):
    p = np.asarray(counts, dtype=np.float64) + 0.5
    return p / p.sum()


def ipf(p, edges):
    """Unique max-entropy extension for compatible, strictly positive margins."""
    target = []
    for edge in edges:
        remove = tuple(i for i in AXES if i not in edge)
        target.append((remove, p.sum(axis=remove, keepdims=True)))
    q = np.full(CELLS, 1 / 81)
    for iteration in range(1, 10001):
        for axes, marginal in target:
            q *= marginal / q.sum(axis=axes, keepdims=True)
        residual = max(float(np.max(abs(q.sum(axis=axes, keepdims=True) - m)))
                       for axes, m in target)
        if residual <= TOL:
            return q.reshape(9, 9), {'sweeps': iteration, 'max_margin_error': residual}
    raise RuntimeError(f'IPF did not converge: max residual {residual:g}, edges={edges}')


def hierarchy(p, extras):
    v, y = p.sum(1), p.sum(0)
    fitted = {'null': np.outer(v, y), 'sat': p}
    margins = {name: tuple(combinations(AXES, k))
               for name, k in [('pair', 2), ('triple', 3)]}
    if extras:
        margins.update(single=tuple(combinations(AXES, 1)))
        margins.update({name: (*WITHIN, edge) for name, edge in CROSS.items()})
    info = {}
    for name, edges in margins.items():
        fitted[name], info[name] = ipf(p.reshape(CELLS), edges)
    return fitted, info


def log_conditionals(fitted, code=None):
    def cond(joint):
        return np.log(joint / joint.sum(1, keepdims=True))

    logq = {name: cond(joint) for name, joint in fitted.items()}
    if code is not None:
        for name in ('pair', 'sat'):
            joined = np.zeros((max(code) + 1, 9))
            np.add.at(joined, code, fitted[name])
            logq['K4_' + name] = cond(joined)[code]
    return logq


def indicator_contrasts():
    first, second = np.indices((3, 3)).reshape(2, -1)
    result = {}
    for name, fun in [('occupied', lambda x: x != 0), ('A', lambda x: x == 1),
                      ('B', lambda x: x == 2),
                      ('A_minus_B', lambda x: (x == 1).astype(float) - (x == 2))]:
        a, b = fun(first).astype(float), fun(second).astype(float)
        result.update({name + '_1': a, name + '_2': b, name + '_sum': a + b,
                       name + '_difference': a - b, name + '_product': a * b})
        if name in ('occupied', 'A'):
            result[name + '_odd_parity'] = (a != b).astype(float)
    result['same_state'] = (first == second).astype(float)
    for name, f in [('sum', first + second), ('difference', first - second)]:
        for op in (np.cos, np.sin):
            result[f'mod3_{name}_{op.__name__}'] = op(2 * np.pi * f / 3)
    return result


CONTRASTS = indicator_contrasts()


def basis(p):
    """Euclidean projectors in square-root-marginal coordinates on active support."""
    active = np.flatnonzero(p > 0)
    weights = p[active]
    root = np.sqrt(weights)
    additive = np.column_stack([active // 3 == s for s in range(3)] +
                               [active % 3 == s for s in range(3)])
    u, s, _ = np.linalg.svd(root[:, None] * (additive - weights @ additive), full_matrices=False)
    u = u[:, s > 1e-13]
    proj = u @ u.T
    parts = {'one': proj, 'connected': np.eye(len(active)) - np.outer(root, root) - proj}
    contrasts = {}
    for name, value in CONTRASTS.items():
        values = value[active]
        z = root * (values - weights @ values)
        norm = np.linalg.norm(z)
        contrasts[name] = z / norm if norm > 1e-13 else None
    return active, root, parts, contrasts


def spectrum(joint):
    p = joint / joint.sum()
    v, y = p.sum(1), p.sum(0)
    iv, rv, pv, cv = basis(v)
    iy, ry, py, cy = basis(y)
    centered = (p - np.outer(v, y))[np.ix_(iv, iy)] / np.outer(rv, ry)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    n = min(len(iv), len(iy)) - 1  # Both marginals have a known constant null direction.
    energy = {f'{a}_to_{b}': float(np.sum((h @ centered @ k) ** 2))
              for a, h in pv.items() for b, k in py.items()}
    total = float(np.sum(centered ** 2))
    if not np.isclose(sum(energy.values()), total, rtol=1e-11, atol=1e-14):
        raise ArithmeticError('weighted transfer blocks do not partition chi-square')
    # Leave null modes unlabelled. Choose a reproducible paired sign for simple modes.
    # In a tied subspace the *span*, not its individual directions, is identified.
    modes, ties = [], []
    for i in range(n):
        if s[i] < 1e-13:
            break
        if u[np.argmax(abs(u[:, i])), i] < 0:
            u[:, i] *= -1
            vt[i] *= -1
        modes.append(dict(sigma=float(s[i]),
                          V_standardized_function=(u[:, i] / rv).tolist(),
                          Y_standardized_function=(vt[i] / ry).tolist(),
                          V_loadings={k: None if z is None else float(u[:, i] @ z) for k, z in cv.items()},
                          Y_loadings={k: None if z is None else float(vt[i] @ z) for k, z in cy.items()}))
    i = 0
    while i < len(modes):
        j = i + 1
        while j < len(modes) and abs(s[j] - s[i]) <= 5e-13:
            j += 1
        ties.append(dict(zero_based_modes=list(range(i, j)),
                         V_projection_squared={k: None if z is None else float(np.sum((u[:, i:j].T @ z)**2)) for k, z in cv.items()},
                         Y_projection_squared={k: None if z is None else float(np.sum((vt[i:j] @ z)**2)) for k, z in cy.items()}))
        i = j
    return dict(singular_values=s[:n].tolist(), chi_square=total, block_energy=energy,
                V_active=iv.tolist(), Y_active=iy.tolist(), modes=modes, subspaces=ties)


def analytic_controls():
    """Population score of deterministic probability tables; never sample from these."""
    i, j, k, l = np.indices(CELLS)
    pv = np.arange(1, 10, dtype=float)
    py = np.array([2, 3, 1, 4, 6, 8, 7, 5, 9], float)
    rho = 0.6
    laws = {'independent_blocks': np.outer(pv / pv.sum(), py / py.sum()),
            'triple': ((1 + rho * (3 * ((i+j-k) % 3 == 0) - 1)) / 81).reshape(9, 9),
            'quad': ((1 + rho * (3 * ((i+j-k-l) % 3 == 0) - 1)) / 81).reshape(9, 9)}
    # With 1/3 of uniform cells in the congruence class, the shifted scores
    # are KL(planted || uniform). All proper planted-subset margins are uniform.
    gain = ((1 + 2*rho)*np.log1p(2*rho) + 2*(1-rho)*np.log1p(-rho)) / 3
    expected = {'independent_blocks': (0., 0., 0.), 'triple': (gain, gain, 0.),
                'quad': (gain, 0., gain)}
    result = {}
    for name, p in laws.items():
        fit, info = hierarchy(p, True)
        q = log_conditionals(fit)
        contrasts = [('sat', 'pair'), ('triple', 'pair'), ('sat', 'triple')]
        observed = [float(np.sum(p * (q[a] - q[b]))) for a, b in contrasts]
        transfer = spectrum(p)
        target = dict.fromkeys(transfer['block_energy'], 0.)
        if name != 'independent_blocks':
            target['connected_to_one' if name == 'triple' else 'connected_to_connected'] = 2*rho*rho
        if not (np.allclose(observed, expected[name], atol=1e-11, rtol=0)
                and all(abs(transfer['block_energy'][k] - x) < 1e-11 for k, x in target.items())
                and abs(np.sum(p * (q['pair'] - q['null']))) < 1e-11):
            raise ArithmeticError(f'{name} analytical calibration failed')
        result[name] = dict(joint=p.tolist(), order=['_minus_'.join(pair) for pair in contrasts],
                            expected=expected[name], observed=observed, ipf=info,
                            expected_transfer_energy=target, transfer=transfer)
    return result


def cluster_scores(groups, differences):
    return np.einsum('gvy,cvy->gc', groups, differences) / groups.sum((1, 2))[:, None]


def intervals(mean, se, df, prefix):
    a, b = t.ppf([.975, 1-.025/FAMILY], df) * se
    return {prefix: float(mean), prefix+'_se': float(se), prefix+'_lo': float(mean-a),
            prefix+'_hi': float(mean+a), prefix+'_sim_lo': float(mean-b), prefix+'_sim_hi': float(mean+b)}


def refit_variance(train, valid):
    # Deletions are on training parents only; never alter the frozen main fits.
    blocks = train.reshape(DELETE_GROUPS, -1, 9, 9).sum(1)
    total = blocks.sum(0)
    target = valid.sum(0) / valid.sum()
    leave = []
    for block in blocks:
        fit, _ = hierarchy(mass(total - block), False)
        q = log_conditionals(fit)
        leave.append([np.sum(target * (q[a] - q[b])) for a, b in REFITS])
    leave = np.asarray(leave)
    variance = (len(leave)-1)/len(leave) * np.sum((leave - leave.mean(0))**2, axis=0)
    return dict(zip(REFITS, variance))


def predictive_fraction(numerator, denominator):
    """Fieller inversion: return disjoint intervals as JSON pairs; null = infinite end."""
    x, y = numerator.mean(), denominator.mean()
    cov = np.cov(numerator, denominator) / len(numerator)
    z2 = t.ppf(1-.025/FAMILY, len(numerator)-1)**2
    # Accepted r satisfy A r^2 + B r + C <= 0.
    a = y*y - z2 * cov[1, 1]
    b = -2*(x*y - z2 * cov[0, 1])
    c = x*x - z2 * cov[0, 0]
    disc = b*b - 4*a*c
    if a == 0:
        if b > 0:
            limits = [[None, -c/b]]
        elif b < 0:
            limits = [[-c/b, None]]
        else:
            limits = [[None, None]] if c <= 0 else []
    elif disc < 0:
        limits = [[None, None]] if a < 0 else []
    else:
        # Stable quadratic roots even when one endpoint is near zero.
        h = -.5 * (b + np.copysign(np.sqrt(disc), b))
        low, high = sorted((h/a, c/h) if h else (0., 0.))
        limits = [[low, high]] if a > 0 else [[None, low], [high, None]]
    return dict(point=float(x/y) if y else None, simultaneous_set=limits,
                conditional_on_training=True)


def panel(index, beta, b, train, val, shuffled, code):
    counts = train.sum(0)
    fitted, info = hierarchy(mass(counts), True)
    q = log_conditionals(fitted, code)
    differences = np.stack([q[a] - q[c] for a, c in SCORES])
    paired = cluster_scores(val, differences)
    null = cluster_scores(shuffled, differences)
    train_var = refit_variance(train, val)
    rows = []
    for n, (hi, lo) in enumerate(SCORES):
        row = dict(index=index, beta=beta, buffer=b, contrast=f'{hi}_minus_{lo}',
                   validation_clusters=len(val), validation_parents_per_cluster=64, simultaneous_family=FAMILY)
        for prefix, values in [('paired', paired[:, n]), ('shuffled', null[:, n]),
                               ('paired_minus_shuffled', paired[:, n] - null[:, n])]:
            row.update(intervals(values.mean(), values.std(ddof=1)/np.sqrt(len(val)), len(val)-1, prefix))
        if (hi, lo) in train_var:
            variance = train_var[hi, lo]
            row['train_delete_group_se'] = float(np.sqrt(variance))
            row.update(intervals(row['paired'], np.sqrt(row['paired_se']**2 + variance),
                                 min(len(val), DELETE_GROUPS)-1, 'two_sample'))
        if (hi, lo) == ('sat', 'pair'):
            row['practically_equivalent'] = all(row[k+'_sim_lo'] > -BAND and row[k+'_sim_hi'] < BAND
                                                for k in ('paired', 'two_sample'))
        rows.append(row)
    shares = {}
    for kind, prefix in [('full', ''), ('cold_K4', 'K4_')]:
        shares[kind] = predictive_fraction(paired[:, SCORES.index((prefix+'pair', 'null'))],
                                           paired[:, SCORES.index((prefix+'sat', 'null'))])
    p = counts / counts.sum()
    record = dict(index=index, beta=beta, buffer=b, cold_K4_code=code.tolist(),
                  training_counts=counts.tolist(), validation_counts=val.sum(0).tolist(),
                  shuffled_validation_counts=shuffled.sum(0).tolist(),
                  empirical_training_MI_nats=float(rel_entr(p, np.outer(p.sum(1), p.sum(0))).sum()),
                  fitted_joints={k: v.tolist() for k, v in fitted.items()}, ipf=info,
                  training_transfer=spectrum(p), predictive_fraction_pair_over_sat=shares)
    return rows, record


def draw(rows, records, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    xs = np.asarray(BUFFERS)
    cold = INDICES[0]
    def curve(ax, index, contrast, label, prefix, **kwargs):
        data = [r for r in rows if r['index'] == index and r['contrast'] == contrast]
        mean, lower, upper = [np.array([r[prefix + key] for r in data]) for key in ('', '_sim_lo', '_sim_hi')]
        ax.errorbar(xs, mean, yerr=np.stack((mean-lower, upper-mean)), marker='o', ms=3,
                    lw=1.2, capsize=2, label=label, **kwargs)
    for ax, contrast, title in zip(axes[0], ('pair_minus_null', 'sat_minus_pair'),
                                  ('Within pair physics: pairwise over block-null', 'Beyond pairs: saturated over pairwise')):
        for index in sorted(INDICES):
            beta = next(r['beta'] for r in records if r['index'] == index)
            curve(ax, index, contrast, f'β={beta:g}', 'two_sample')
        ax.set_title(title)
        ax.legend(fontsize=8, ncols=2)
    ax = axes[1, 0]
    for contrast, label, prefix in [('triple_minus_pair', '3-body over pairwise', 'two_sample'),
                                   ('sat_minus_triple', 'saturated over 3-body', 'two_sample'),
                                   ('K4_sat_minus_K4_pair', 'cold K4: saturated over pairwise', 'paired')]:
        curve(ax, cold, contrast, label, prefix)
    ax.set_title('Cold: higher-order conditional score increments')
    ax.legend(fontsize=8)
    for ax in (axes[0, 1], axes[1, 0]):
        ax.axhspan(-BAND, BAND, color='0.75', alpha=.28, zorder=-1)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_yscale('symlog', linthresh=BAND)
        ax.set_ylabel('paired validation log-score gain (nats)')
        ax.axhline(0., color='0.4', lw=.6)
    ax = axes[1, 1]
    cold_records = [r for r in records if r['index'] == cold]
    for name in cold_records[0]['training_transfer']['block_energy']:
        energy = [r['training_transfer']['block_energy'][name] for r in cold_records]
        ax.plot(xs, energy, marker='o', ms=3, label=name.replace('_to_', ' → '))
    ax.set_yscale('symlog', linthresh=1e-8)
    ax.set_title('Cold: weighted transfer subspaces (descriptive)')
    ax.set_ylabel('squared transfer norm (χ², not MI)')
    ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks(xs)
        ax.set_xlabel('empty layers (buffer b)')
    fig.suptitle('Two-corner channel: frozen conditional predictions vs finite-table structure\n'
                 'Exploratory simultaneous intervals; gray band = ±10⁻⁵ nats', fontsize=14)
    fig.savefig(path, dpi=175)
    plt.close(fig)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def run(data, out):
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'controls.json', analytic_controls())
    ix = tables()
    encodings = {4: partitions()[4]}
    codes, receipts, rows, records = {}, {}, [], []
    for index in INDICES:
        path = data / f'beta-{index:02d}.npz'
        meta = json.loads(path.with_suffix('.json').read_bytes())
        with path.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        if sha != meta['endpoint_sha256']:
            raise ValueError(f'archive changed: {path}')
        receipts[str(index)] = sha
        (tr, va, _), _ = partition(index)
        with np.load(path, allow_pickle=False) as archive:
            if str(archive['axes']) != 'chain,z,y,x' or float(archive['beta']) != float(meta['beta']):
                raise ValueError(f'archive metadata mismatch: {path}')
            state, beta = archive['species'], float(archive['beta'])
        train, _ = marginals(state, tr, ix, index, False)
        if index == INDICES[0]:
            for j, b in enumerate(BUFFERS):
                codes[b] = solutions(train[:, j].sum(0), encodings)[0]['K=4']
        val, shuffled = marginals(state, va, ix, index, True)
        del state
        for j, b in enumerate(BUFFERS):
            r, record = panel(index, beta, b, train[:, j], val[:, j], shuffled[:, j], codes[b])
            rows.extend(r)
            records.append(record)
        print(f'β index {index}: complete (10240 train, 2048 validation; test not scored)', flush=True)
    with (out / 'interactions.csv').open('w', newline='') as f:
        columns = dict.fromkeys(k for row in rows for k in row)
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    dump(out / 'tables.json', dict(source_sha256=receipts, frame_seed=652827,
                                  pair_index='3*first+second; rows V, columns Y',
                                  simultaneous_family=FAMILY, practical_band=BAND, panels=records))
    draw(rows, records, out / 'interactions.png')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', required=True, type=Path)
    cli.add_argument('--out', required=True, type=Path)
    args = cli.parse_args()
    run(args.data, args.out)
