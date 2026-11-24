"""Archive-only canonical prediction from the occupied mask. See mask-oracle/design.json."""
import argparse
import fcntl
import hashlib
import json
import os
from itertools import product
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

import mask_law
from vacancy_dipole import selected_member, sha

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))
from l20_pairs import L, PAIR_SEED, SPLIT, partition

DESIGN = EXAMPLES.parent / 'results/l20-autoresearch/mask-oracle/design.json'
INDICES = (3, 9, 5, 0)
N_A, N_OCC, VOLUME = 2400, 6000, L**3
K = np.arange(7)
EB, D = (K - 5)**2, 4*K - 16
DIRECTIONS = tuple(product(range(3), (-1, 1)))
OUTPUTS = ('parent-observables.npz', 'mask-oracle.png', 'results.json')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def validation(data, index, pin):
    require(index in INDICES, 'only the four locked validation cohorts may be read')
    path = data / f'beta-{index:02d}.npz'
    receipt_path = path.with_suffix('.json')
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt['index'] == index and float(receipt['beta']) == pin['beta'] and
            sha(path) == receipt['endpoint_sha256'] == pin['endpoint_sha256'], 'wrong release')
    ids = partition(index)[0][1]
    require(L == 20 and SPLIT == (10240, 2048, 4096) and PAIR_SEED == 769315 and
            ids.shape == (SPLIT[1],) and np.unique(ids).size == SPLIT[1] and
            ids.min() >= 0 and ids.max() < sum(SPLIT), 'wrong validation rows')
    with ZipFile(path) as archive:
        for key, expect in [('axes', 'chain,z,y,x'), ('beta', receipt['beta'])]:
            with archive.open(key + '.npy') as stream:
                require(np.lib.format.read_array(stream, allow_pickle=False).item() == expect,
                        f'{path}: wrong {key}')
        lineage = np.column_stack([selected_member(archive, key, ids, ())
                                   for key in ('chain_seed', 'chain_id')]).astype('<u8')
        require(np.unique(lineage, axis=0).shape[0] == len(ids) and
                array_hash(lineage) == pin['validation_lineage_sha256'], 'wrong validation lineage')
        species = selected_member(archive, 'species', ids, (L, L, L))
    require(species.dtype == np.uint8 and species.max() <= 2, 'wrong species encoding')
    return ids, lineage, species, dict(**pin, receipt_sha256=sha(receipt_path),
                                       row_sha256=array_hash(ids.astype('<u8')), fold='validation')


def geometry(occ):
    """No labeled input is accepted; keep sign conventions explicit at the center."""
    require(occ.dtype == np.bool_ and occ.shape == (L, L, L), 'expected one Boolean mask')
    near = {(a, s): np.roll(occ, -s, axis=a) for a, s in DIRECTIONS}
    m = sum(v.astype(np.int8) for v in near.values())
    return near, m


def target_indices():
    # Independent coordinate check: roll(arr,+3*s)[c] must be arr[c-3*s*e_a].
    flat = np.arange(VOLUME).reshape((L,)*3)
    coords = np.indices(flat.shape)
    targets = {}
    for a, s in DIRECTIONS:
        j = coords.copy()
        j[a] = (j[a] - 3*s) % L
        ids = np.ravel_multi_index(tuple(j), flat.shape)
        require(np.all(ids != flat) and np.array_equal(ids, np.roll(flat, 3*s, axis=a)),
                'outward target or nonself dyad failed')
        targets[a, s] = ids
    return targets


def predict(occ, beta, targets):
    """All predictors take geometry, fixed composition and beta, never central/target labels."""
    near, m = geometry(occ)
    counts = np.bincount(m[occ], minlength=7)
    require(counts.sum() == N_OCC, 'unexpected occupied composition')
    p, q, mu, var = mask_law.moments(counts, beta, N_A, center=3)
    remaining = counts.copy()
    remaining[3] -= 1
    require(abs(counts @ p - N_A) < 2e-7 and
            (counts[3] == 0 or abs(remaining @ q - (N_A-1)) < 2e-7),
            'canonical one-site or distinct-dyad conservation failed')
    require(np.isfinite(var) and var >= 0 and abs(mu - counts @ (EB + p*D)) < 2e-6,
            'invalid conditional energy moments')
    center = occ & (m == 3)
    total = 0
    target_k = np.zeros(7, dtype=np.int64)
    for a, s in DIRECTIONS:
        b, d = (v for v in range(3) if v != a)
        allowed = center & ~near[a, s]
        w = (allowed * (near[b, 1].astype(np.int8) + near[b, -1]) *
             (near[d, 1].astype(np.int8) + near[d, -1]))
        frames = sum((allowed & near[b, i] & near[d, j]).astype(np.int8)
                     for i, j in product((-1, 1), repeat=2))
        require(np.array_equal(frames, w), 'octant weights disagree at a center')
        ids = targets[a, s]
        total += int(w.sum())
        occupied_target = occ.ravel()[ids] & (w > 0)
        target_k += np.bincount(m.ravel()[ids[occupied_target]], weights=w[occupied_target],
                                minlength=7).astype(np.int64)
    require(target_k.sum() <= total and not np.any(target_k[remaining <= 0]), 'impossible target')
    den, num = p[3] * total, p[3] * (target_k @ q)
    if beta == 0:
        # Sampling N_A from N_OCC *without replacement*, conditional on this actual mask.
        f, g = N_A / N_OCC, (N_A-1) / (N_OCC-1)
        v = N_A*(1-f) * (counts @ (D - (counts @ D)/N_OCC)**2) / (N_OCC-1)
        require(np.allclose(p[counts > 0], f, atol=1e-11, rtol=0) and
                (not counts[3] or np.allclose(q[remaining > 0], g, atol=1e-11, rtol=0)) and
                np.allclose([mu, var, den, num], [counts @ (EB+f*D), v, f*total,
                            f*g*target_k.sum()], atol=1e-7, rtol=1e-11), 'beta=0 identity failed')
    return dict(counts=counts, p=p, conditional=q, geometric_den=total, target_k=target_k,
                energy_pred=mu/VOLUME, energy_cond_var=var/VOLUME**2, cage_pred_den=den,
                cage_pred_num=num)


def measure(species):
    """Separate scoring path: only now select the observed A centers and target labels."""
    occ, aa = species != 0, species == 1
    require(aa.sum() == N_A, 'unexpected archived A count')
    near, m = geometry(occ)
    obs_e = (((m[aa]-3)**2).sum() + ((m[species == 2]-5)**2).sum()) / VOLUME
    num = den = 0
    for a, s in DIRECTIONS:
        b, d = (v for v in range(3) if v != a)
        central_a = aa & (m == 3) & ~near[a, s]
        target_a = np.roll(aa, 3*s, axis=a)
        for i, j in product((-1, 1), repeat=2):
            cage = central_a & near[b, i] & near[d, j]
            den += np.count_nonzero(cage)
            num += np.count_nonzero(cage & target_a)
    return dict(energy_obs=obs_e, cage_obs_den=den, cage_obs_num=num)


def observations(species, beta, targets):
    # Entire-cohort prediction pass. No species-stratum totals or scored statistics are fed back.
    pred = [predict(s != 0, beta, targets) for s in species]
    obs = [measure(s) for s in species]
    return {k: np.asarray([r[k] for r in rows]) for rows in (pred, obs) for k in rows[0]}


def summarize(data):
    e, u, v = (data[k] for k in ('energy_obs', 'energy_pred', 'energy_cond_var'))
    # Center before forming squared sums: small between-parent variances are of interest.
    residual = e - u
    de, du, dr = e-e.mean(), u-u.mean(), residual-residual.mean()
    n = len(e)
    vals = np.column_stack((e, u, residual, v, residual**2, de, du, dr, de**2, du**2, dr**2,
                           *[data[k] for k in ('cage_obs_num','cage_obs_den','cage_pred_num','cage_pred_den')]))

    def estimands(sums, size):
        e, u, r, v, mse, de, du, dr, de2, du2, dr2, on, od, pn, pd = (sums/size).T
        ve, vu, vr = de2-de**2, du2-du**2, dr2-dr**2
        require(np.all(od > 0) and np.all(pd > 0) and np.all(ve > 0) and np.all(v > 0),
                'undefined pooled cage or variance statistic')
        return dict(energy_obs=e, energy_pred=u, energy_residual=r, observed_variance=ve,
                    mask_mean_variance=vu, residual_variance=vr, conditional_variance=v,
                    residual_mse=mse, scatter_canonical=np.sqrt(v), residual_rmse=np.sqrt(mse),
                    mse_over_conditional_variance=mse/v, mse_minus_conditional_variance=mse-v,
                    explained_centered=1-vr/ve, predictive_r2=1-mse/ve,
                    canonical_mask_fraction=vu/(vu+v), cage_obs=on/od, cage_pred=pn/pd,
                    cage_residual=on/od-pn/pd)

    sums = vals.sum(axis=0)
    point, deleted = estimands(sums, n), estimands(sums - vals, n-1)
    critical = t.ppf(.975, n-1)
    stats = {}
    for key, estimate in point.items():
        loo = deleted[key]
        se = np.sqrt((n-1)*np.mean((loo - loo.mean())**2))
        stats[key] = dict(mean=float(estimate), se=float(se), lo95=float(estimate-critical*se),
                          hi95=float(estimate+critical*se))
    return stats


def decision(interval, margin):
    lo, hi = interval['lo95'], interval['hi95']
    if -margin < lo and hi < margin:
        return 'equivalent within locked margin'
    if hi < -margin or lo > margin:
        return 'discrepancy beyond locked margin'
    return 'inconclusive at locked margin'


def plot(path, cohorts, plan):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), layout='constrained')
    rows = sorted(cohorts, key=lambda r: r['beta'])
    x = np.arange(len(rows))

    def series(ax, key, color, offset=0, scale=1, label=None):
        y, lo, hi = scale*np.array([[r['metrics'][key][s] for s in ('mean', 'lo95', 'hi95')]
                                    for r in rows]).T
        ax.errorbar(x+offset, y, yerr=[y-lo, hi-y], color=color, fmt='o', capsize=3, label=label)

    series(axes[0], 'energy_residual', '#206e70')
    series(axes[1], 'explained_centered', '#206e70', -.10, label='Measured: 1 − Var(resid.)/Var(E)')
    series(axes[1], 'canonical_mask_fraction', '#a85023', .10, label='Canonical: between-mask share')
    axes[1].legend(fontsize=7, frameon=False)
    series(axes[2], 'cage_residual', '#206e70', scale=100)
    for ax, margin in [(axes[0], plan['energy_margin_per_site']),
                       (axes[2], 100*plan['cage_margin_probability'])]:
        xp = next(i for i, r in enumerate(rows) if r['index'] == plan['primary_index'])
        ax.fill_between([xp-.35, xp+.35], -margin, margin, color='#206e70', alpha=.15)
    for ax, title, ylabel in zip(axes, ['Energy bias', 'Energy variance fraction', 'Oriented cage: outer A'],
                                 ['Measured − predicted (energy/site)', 'Fraction', 'Measured − predicted (pp)']):
        ax.axhline(0, lw=.7, c='.55', zorder=0)
        ax.set(title=title, ylabel=ylabel, xticks=x,
               xticklabels=[f"{r['beta']:.3g}" + ('*' if r['index']==3 else '') for r in rows],
               xlabel='Inverse temperature β (* locked)')
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(data, out, plan, lock):
    targets = target_indices()
    cohorts, parents = [], {}
    for index in INDICES:
        pin = plan['release'][str(index)]
        ids, lineage, species, source = validation(data, index, pin)
        col = observations(species, pin['beta'], targets)
        del species
        metrics = summarize(col)
        decisions = {endpoint: (decision(metrics[key], plan[margin]) if index == plan['primary_index']
                                else 'descriptive') for endpoint, key, margin in
                     [('primary_energy', 'energy_residual', 'energy_margin_per_site'),
                      ('secondary_cage', 'cage_residual', 'cage_margin_probability')]}
        totals = {key: float(col[key].sum()) for key in
                  ('cage_pred_den', 'cage_pred_num', 'cage_obs_den', 'cage_obs_num')}
        cohorts.append(dict(index=index, beta=pin['beta'], source=source, parents=len(ids),
                            pooled_cage=totals, metrics=metrics, decisions=decisions))
        parents.update({f'{name}_{index:02}': array for name, array in
                        dict(row=ids, lineage=lineage, **col).items()})
        print(f'Completed validation index {index}: {len(ids)} parents', flush=True)
    np.savez_compressed(out/OUTPUTS[0], **parents)
    plot(out/OUTPUTS[1], cohorts, plan)
    result = dict(design_sha256=lock, cohorts=cohorts, parent_unit='distinct native validation lineage',
                  energy_unit='Hamiltonian energy per lattice site; variances in (energy/site)^2',
                  code_sha256={f.name: sha(f) for f in [Path(__file__), Path(mask_law.__file__),
                              Path(__file__).with_name('vacancy_dipole.py'), EXAMPLES/'l20_pairs.py']},
                  files_sha256={f: sha(out/f) for f in OUTPUTS[:-1]},
                  checks='All source-row, independent octant/target checks, one-site and conditional composition checks and beta-zero controls passed.',
                  limits=plan['limits'])
    (out/OUTPUTS[2]).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def run(data, out, lock):
    blob = DESIGN.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == lock, 'provide the pre-scoring design SHA-256')
    plan = json.loads(blob)
    require(tuple(plan['indices']) == INDICES and plan['primary_index'] == 3, 'wrong design')
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in OUTPUTS:
            if os.path.lexists(out/name):
                raise FileExistsError(f'{out/name}: refusing to score over existing outputs')
        with TemporaryDirectory(dir=out) as work:
            work = Path(work)
            evaluate(data, work, plan, lock)
            for name in OUTPUTS:
                os.link(work/name, out/name)  # no-clobber even if another writer ignores flock
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lock-sha256', required=True, help='SHA-256 of the pre-scoring design.json')
    args = parser.parse_args()
    run(args.data, args.out, args.lock_sha256)
