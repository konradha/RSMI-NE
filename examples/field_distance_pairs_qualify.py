"""Run the locked synthetic qualification, never open physical lattice data."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

import field_distance_pairs as pair
from stock_glass_cavity_exact import C

OUT = Path(__file__).resolve().parents[1] / 'results/frozen-field-distance'
DIST = tuple(range(3, 10))
PRIMARY = [DIST.index(5), DIST.index(6)]
SEED = 711526
PARENTS = 128
MARGIN = .01
FACTOR = 72


def exact(grid, fields, distances):
    """Slow independent coordinate oracle, no Gram matrices or histogram correction."""
    points = np.argwhere(grid >= 0)
    counts = np.zeros((len(distances), 3), dtype=int)
    result = np.full((len(fields), len(distances), 3, 2), np.nan)
    for j, d in enumerate(distances):
        for axis in range(3):
            target = points.copy()
            target[:, axis] = (target[:, axis] + d) % 20
            joint = grid[tuple(target.T)] >= 0
            n = counts[j, axis] = joint.sum()
            if n >= 2:
                left = fields[:, grid[tuple(points[joint].T)]].copy()
                right = fields[:, grid[tuple(target[joint].T)]].copy()
                left -= left.mean(axis=1, keepdims=True)
                right -= right.mean(axis=1, keepdims=True)
                cross = np.einsum('mpck,mpck->mc', left, right) / (n-1)
                result[:, j, axis, 0] = cross[:, axis]
                result[:, j, axis, 1] = cross[:, [c for c in range(3) if c != axis]].mean(axis=1)
    return counts, result.mean(axis=2)


def structural_checks():
    # Translation invariance on a torus gives the same support from every center;
    # place the origin across all three wrap boundaries explicitly.
    gap = {}
    origin = (C + np.array([19, 19, 19])) % 20
    for d in DIST:
        for a in range(3):
            shifted = origin.copy()
            shifted[:, a] = (shifted[:, a] + d) % 20
            delta = abs(origin[:, None] - shifted)
            g = int(np.minimum(delta, 20-delta).sum(axis=2).min())
            assert g == d-2 and g >= 1
        gap[d] = g

    rng = np.random.default_rng([SEED, 471])
    grid = rng.integers(0, 210, size=pair.GRID, dtype=np.int16)
    grid[rng.random(pair.GRID) < .77] = -1
    fields = rng.normal(size=(3, 210, 3, 4))
    fields[1] = fields[0] @ np.linalg.qr(rng.normal(size=(4, 4)))[0]
    fields[2] = rng.normal(size=(3, 4))  # Constant on states but not on axes/copies.
    bank = pair.prepare(fields)
    counts, cov, shuffled, meta = pair.score(grid.ravel(), bank, DIST, [SEED, 472])
    mask = grid >= 0
    nullgrid = grid.copy()
    nullgrid[mask] = np.random.default_rng([SEED, 472]).permutation(grid[mask])
    np.testing.assert_array_equal(np.sort(nullgrid[mask]), np.sort(grid[mask]))
    errs = []
    for states, output in ((grid, cov), (nullgrid, shuffled)):
        refcounts, ref = exact(states, fields, DIST)
        np.testing.assert_array_equal(counts, refcounts)
        np.testing.assert_allclose(output, ref, atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(output[0], output[1], atol=3e-12, rtol=3e-12)
        np.testing.assert_array_equal(output[2], 0.)
        errs.append(float(np.max(abs(ref-output))))
    assert meta['n_eligible'] == int(mask.sum())

    grid = np.full(pair.GRID, -1, dtype=np.int16)
    grid[[0, 1]] = 0; grid[[5, 6]] = 1; grid[13] = 2
    fields = np.zeros((1, 210, 3, 1)); fields[0, :3, 0, 0] = [2., -1., 3.]
    n, measured, _, _ = pair.score(grid, fields, (5,), 43)
    refcounts, ref = exact(grid, fields, (5,))
    np.testing.assert_array_equal(n, refcounts)
    np.testing.assert_allclose(measured, ref, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(measured[0, 0, 0], 0., atol=3e-12)
    mean = fields[0, grid[grid >= 0], 0, 0].mean()
    naive = float((2.-mean)*(-1.-mean)*n[0, 0]/(n[0, 0]-1)/3)
    assert abs(naive) > .1
    bank = pair.prepare(fields)
    grid.fill(-1)
    empty, a, b, _ = pair.score(grid, bank, (5,), 43)
    assert np.all(empty == 0) and np.isnan(a).all() and np.isnan(b).all()
    grid[0, :7, :7] = 0; grid[5, 0, 0] = 1
    one, a, b, _ = pair.score(grid, bank, (5,), 43)
    assert one[0, 0] == 1 and (one[0, 1:] >= 2).all() and np.isnan(a).all() and np.isnan(b).all()
    return dict(minimum_star_site_distances=gap, oracle_max_absolute_error=dict(true=errs[0], shuffled=errs[1]),
                supported_counts_match=True, orthogonal_copy_rotation_invariant=True, constant_field_zero=True,
                permutation_histogram_preserved=True, selection_fixture=dict(counts=n.tolist(),
                axis0_endpoint_means=[2., -1.], naive_global_mean_L=naive,
                exact_joint_centered_L=float(measured[0, 0, 0])), unsupported_one_axis_counts=one.tolist(),
                empty_and_one_pair_return_nan=True)


def fixture(parent):
    noise = np.random.default_rng([SEED, parent, 0]).normal(size=(3, *pair.GRID))
    planted = noise.copy()
    for c in range(3):
        for a in range(3):
            weight = .040 if c == a else -.036
            for d in (5, 6):
                planted[c] += weight*np.roll(noise[c], -d, axis=a)
    eligible = np.random.default_rng([SEED, parent, 1]).random(pair.GRID) < .16
    for field in (planted, noise):
        labels = sum(((field[c] > 0).astype(np.int16) << c) for c in range(3))
        labels[~eligible] = -1
        yield labels


def interval(parent_values):
    n = len(parent_values)
    sd = parent_values.std(axis=0, ddof=1)
    mean = parent_values.mean(axis=0)
    half = t.ppf(1 - .05/(2*FACTOR), n-1) * sd/np.sqrt(n)
    return {name: dict(mean=float(m), sd_across_parents=float(s), lo=float(m-h), hi=float(m+h),
                       halfwidth=float(h), excludes_zero=bool(abs(m)>h),
                       beyond_practical_margin=bool(abs(m)-h>MARGIN),
                       inside_equivalence_margin=bool(abs(m)+h<MARGIN))
            for name, m, s, h in zip(('L', 'T'), mean, sd, half)}


def run():
    audit = structural_checks()
    # Only the low three bits are used by this synthetic label law.
    fields = (2*((np.arange(210)[:, None] >> np.arange(3)) & 1)-1).astype(float)[None, :, :, None]
    bank = pair.prepare(fields)
    data = np.empty((2, PARENTS, 2, len(DIST), 2))
    counts = np.empty((PARENTS, len(DIST), 3), dtype=int)
    eligible = np.empty(PARENTS, dtype=int)
    for parent in range(PARENTS):
        for law, labels in enumerate(fixture(parent)):
            n, cov, shuffled, meta = pair.score(labels, bank, DIST, [SEED, parent, 2])
            if law == 0:
                counts[parent] = n; eligible[parent] = meta['n_eligible']
            else:
                np.testing.assert_array_equal(n, counts[parent]); assert meta['n_eligible'] == eligible[parent]
            data[law, parent] = cov[0], shuffled[0]
    # Do not drop a failed distance or substitute a partner from another parent.
    assert np.isfinite(data[:, :, :, PRIMARY, :]).all()
    pooled = data[:, :, :, PRIMARY, :].mean(axis=3)
    result = {}
    for law, name in enumerate(('plant', 'iid')):
        raw, null = pooled[law, :, 0], pooled[law, :, 1]
        result[name] = dict(raw=interval(raw), shuffled=interval(null), paired=interval(raw-null))
    theory = 2/np.pi*np.arcsin(np.array([.040, -.036])/(1 + 2*.040**2 + 4*.036**2))
    close = all(abs(result['plant']['raw'][p]['mean'] - val) < MARGIN for p, val in zip(('L', 'T'), theory))
    sign = all(result['plant'][key]['L']['lo'] > 0 > result['plant'][key]['T']['hi'] for key in ('raw', 'paired'))
    spurious = any(result[law][key][p]['beyond_practical_margin'] for law, key in
                   (('plant', 'shuffled'), ('iid', 'raw'), ('iid', 'shuffled'), ('iid', 'paired')) for p in ('L', 'T'))
    primary_n = counts[:, PRIMARY, :]
    report = dict(qualification_passed=bool(close and sign and not spurious),
                  checks=dict(plant_within_margin_of_analytic_target=bool(close), plant_signed_detection=bool(sign),
                              null_spurious_practical_promotion=bool(spurious)),
                  sha256={f.name: sha256(f.read_bytes()).hexdigest() for f in
                          (OUT/'pair-fixture-design.json', Path(pair.__file__), Path(__file__))},
                  estimator_audit=audit, analytic_population=dict(L=float(theory[0]), T=float(theory[1]),
                  iid=0., variance=1., distances=[5, 6]), independent_parents=PARENTS, root_seed=SEED,
                  inference=dict(parent_unit='one synthetic cube', alpha=.05, simultaneous_factor=FACTOR,
                  critical=float(t.ppf(1-.05/(2*FACTOR), PARENTS-1)), practical_effect=MARGIN),
                  support=dict(min_parent_eligible=int(eligible.min()), max_parent_eligible=int(eligible.max()),
                  minimum_per_parent_primary_axis=int(primary_n.min()), pooled_primary_counts=primary_n.sum(0).tolist()),
                  results=result)
    path = OUT/'pair-qualification.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(path=path.name, passed=report['qualification_passed'], checks=report['checks'],
                          plant=result['plant'], iid=result['iid'])), flush=True)
    if not report['qualification_passed']:
        raise SystemExit('Locked synthetic qualification failed; do not open held outcomes.')


if __name__ == '__main__':
    run()
