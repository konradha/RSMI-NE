"""Lock the star-state fields. Inputs: 03 final filters, 06 actions, source TRAIN one-point mass only.

Run from the repository with OPENBLAS_NUM_THREADS=1 python3 examples/field_distance_maps.py.
Native component order is the array order used by 03: (z,y,x), minus/plus within axis.
Only the eligible 210-state table has rows. No spatial, held, or transfer data are read.
"""
import argparse
import csv
import hashlib
import io
import json
from itertools import combinations_with_replacement, permutations
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from stock_glass_cavity_exact import C, X
from stock_glass_cubic import SEEDS, geometry

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'results/experiment-03-cavity'
CUBIC = ROOT / 'results/experiment-06-cubic-families'
OUT = ROOT / 'results/frozen-field-distance'
DEPTH = 10218
N = len(X)
TOL = 2e-12


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_sha(a):
    return sha(np.ascontiguousarray(a).tobytes())


def file_sha(p):
    with p.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def same_bytes(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def put_json(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def put_csv(p, rows):
    with p.open('w', newline='') as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_frozen():
    """Byte replay before reading the TRAIN mass. Never load exterior rows or held marginals."""
    with ZipFile(REF / 'exact-exteriors-and-replicas.zip') as src:
        geom_bytes = src.read('state-geometry.npz')
    with np.load(io.BytesIO(geom_bytes)) as geom:
        same_bytes(X, geom['states'])
        same_bytes(C, geom['star'])
    np.testing.assert_array_equal(C, [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0],
                                      [0, 1, 0], [0, 0, -1], [0, 0, 1]])
    mats, sites, action, classes = geometry()
    actions_file = CUBIC / 'group-actions.npz'
    with np.load(actions_file) as frozen:
        for key, a in [('matrices', mats), ('site_permutations', sites),
                       ('state_permutations', action), ('proper_classes', classes)]:
            same_bytes(a, frozen[key])
    group = {g.tobytes(): j for j, g in enumerate(mats)}
    assert len(group) == 48
    for j, g in enumerate(mats):
        np.testing.assert_array_equal(X[action[j]][:, sites[j]], X)
        np.testing.assert_array_equal(C[sites[j]], C @ g.T)
        for k, h in enumerate(mats):
            np.testing.assert_array_equal(action[j, action[k]], action[group[(g @ h).tobytes()]])
    inverse = np.argsort(action, axis=1)
    for j, g in enumerate(mats):
        np.testing.assert_array_equal(inverse[j], action[group[g.T.tobytes()]])

    source = REF / 'corrected'
    filters = source / 'stock-checkpoints.zip'
    recorded = source / 'code-maps.npz'
    inputs = {'03_geometry_npz_member': sha(geom_bytes), '03_checkpoints_zip': file_sha(filters),
              '03_code_maps_npz': file_sha(recorded), '06_actions_npz': file_sha(actions_file)}
    assert inputs['03_code_maps_npz'] == json.loads((CUBIC / 'design.json').read_bytes())['pins']['corrected/code-maps.npz']
    assert inputs['03_geometry_npz_member'] == '576606e81a088cb1e05ceb9aa258c7863caea8ea9795ffda8d813bfd133e9cc6'
    assert inputs['03_checkpoints_zip'] == 'abe6192fd7ac37f35518a5e20e1204639927402b8a5bd0f0a233f5456ce269fc'
    assert inputs['06_actions_npz'] == '40ea9e5a24143d4efddc9789923ad04672073a1312ca2640008593e8f24936fd'
    labels = {}
    replay = {}
    flat = np.stack([X == 1, X == 2], axis=-1).reshape(N, 14).astype('f4')
    with ZipFile(filters) as archive, np.load(recorded) as maps:
        for s in SEEDS:
            stem = f'b09-s{s}'
            key = f'{stem}_d{DEPTH}'
            member = archive.read(stem + '.npz')
            with np.load(io.BytesIO(member)) as fit:
                w = fit[f'w{DEPTH}']
                saved = fit[f'logits{DEPTH}']
            weights = w[tuple((C + 1).T)].reshape(14, 2)
            # 03 stock H100 TF32 round-to-nearest-even operands, float32 accumulation.
            bits = weights.view('u4')
            tf32 = ((bits + 4095 + ((bits >> 13) & 1)) & np.uint32(0xffffe000)).view('f4')
            logits = flat @ tf32
            same_bytes(logits, saved)
            same_bytes(logits.astype('f8'), maps[key + '_logits'])
            y = (2 * (logits[:, 0] >= 0) + (logits[:, 1] >= 0)).astype('u1')
            same_bytes(np.eye(4)[y], maps[key + '_hard_Q'])
            labels[s] = y
            replay[s] = dict(checkpoint_member_sha256=sha(member), filter_sha256=array_sha(w),
                             logits_f32_sha256=array_sha(logits), uint8_labels_sha256=array_sha(y),
                             categories=np.bincount(y, minlength=4).tolist(),
                             min_abs_logit=float(np.min(np.abs(logits))), max_logit_error=0.)
    lock = dict(source_beta=3.05, depth=DEPTH, source_inputs_sha256=inputs,
                states_sha256=array_sha(X), axis_order=['z', 'y', 'x'],
                sites=C.tolist(), state_count=N, categories='2*(logit0>=0)+(logit1>=0)',
                labels_equivalent_to_03_hard_Q_bytewise=True, weights_to_logits_byte_exact=True,
                products_checked=48**2, inverses_checked=48, group_actions_replayed_bytewise=True,
                replay=replay)
    return labels, mats, action, inverse, lock


def invariant_mass(raw, action):
    assert raw.shape == (N,) and np.all(np.isfinite(raw)) and np.all(raw >= 0) and raw.sum() > 0
    raw = raw.astype('d') / raw.sum()
    orbit = action.min(axis=0)
    _, orbit = np.unique(orbit, return_inverse=True)
    m = (np.bincount(orbit, weights=raw) / np.bincount(orbit))[orbit]
    m /= m.sum()
    np.testing.assert_allclose(m, raw[action].mean(axis=0), rtol=0, atol=1e-15)
    assert (m > 0).all() and np.all(m[action] == m)
    return raw, m


def vector_field(y, mats, inverse):
    """Active: F_iak(hs) = h_ij F_jak(s), not its transpose. Keep every (a,k)."""
    # Exact integer numerator; the orbit sum of each R_ik is zero. Thus subtracting
    # the invariant TRAIN category means before projecting is exactly redundant.
    indicator = np.eye(4, dtype='i2')[y[inverse]]
    f = np.einsum('gik,gsa->siak', mats, indicator, dtype='i2').astype('d') * (3 / 48)
    return f.reshape(N, 3, 12)


def equivariance_error(v, mats, action):
    return max(float(np.max(np.abs(v[ix] - g @ v))) for g, ix in zip(mats, action))


def diagnostics(y, v, m):
    """06 label-span T1u projector trace / centered rank (not unwhitened raw power)."""
    marginal = np.bincount(y, weights=m, minlength=4)
    keep = marginal > 0
    h = -float(marginal[keep] @ np.log(marginal[keep]))
    # The Cartesian trace of the exact matrix units is the T1u character projector.
    projected = np.trace(v.reshape(N, 3, 4, 3), axis1=1, axis2=3)[:, keep]
    # Centered one-hot columns divided by sqrt(p_a) form a tight frame on their
    # rank (#nonempty-1) span. Projecting kills the removed constants.
    power = float((m @ (projected**2) / marginal[keep]).sum() / (keep.sum() - 1))
    assert 0 <= power <= 1 + TOL
    return h, power


def packing_features():
    p, n = X[:, 2::2], X[:, 1::2]
    d = np.stack([(p == t).astype('d') - (n == t) for t in range(3)], axis=-1)
    basic_names = [f'D_{t}' for t in range(3)]
    terms, names = [d], basic_names.copy()
    for t in range(3):
        terms.append(d * (X[:, 0] == t)[:, None, None])
        names.extend(f'{key}*center_{t}' for key in basic_names)
    for a, b in combinations_with_replacement(range(3), 2):
        pos, neg = (p == a) & (n == b), (p == b) & (n == a)
        count = (pos | neg).sum(axis=1)
        terms.append(d * count[:, None, None])
        names.extend(f'{key}*n_opposite_{a}{b}' for key in basic_names)
        if a != b:
            terms.append((pos.astype('d') - neg)[:, :, None])
            names.append(f'axis_ordered_{a}{b}-{b}{a}')
    return d, np.concatenate(terms, axis=2), names


def basis(x, m):
    """Q spans tied equivariant vector copies, with <u,v>=sum_s m_s u.v / 3."""
    weighted = (x * np.sqrt(m[:, None, None] / 3)).reshape(3 * N, -1)
    _, sig, vt = np.linalg.svd(weighted, full_matrices=False)
    use = sig > 1e-12 * sig[0]
    q = x @ (vt[use].T / sig[use])
    np.testing.assert_allclose(np.einsum('s,sic,sid->cd', m / 3, q, q), np.eye(use.sum()), atol=TOL, rtol=0)
    return q, sig


def residual(v, q, m):
    coef = np.einsum('s,sij,sik->jk', m / 3, q, v)
    r = v - q @ coef
    np.testing.assert_allclose(np.einsum('s,sij,sik->jk', m / 3, q, r), 0, atol=TOL, rtol=0)
    return r


def variance(v, m):
    return float(np.einsum('s,sij,sij->', m / 3, v, v))


def main(out, mass_path):
    labels, mats, act, inv, lock = load_frozen()
    raw, m = invariant_mass(np.load(mass_path), act)
    d, enriched, feature_names = packing_features()
    q0, sig0 = basis(d, m)
    q1, sig1 = basis(enriched, m)
    assert q0.shape[2] == 2 and equivariance_error(enriched, mats, act) == 0
    assert np.abs(residual(q0, q1, m)).max() < TOL
    assert np.max(np.abs(mats.sum(axis=0))) == 0

    vectors, catalog, match, seed_info, covs = [], [], [], {}, {}
    field_detail = dict(states=X, sites=C, mass_train_raw=raw, mass_train_orbit=m,
                        label_seeds=np.array(SEEDS), learned_labels=np.stack([labels[s] for s in SEEDS]),
                        packing_columns=np.array(feature_names), basic_singular_values=sig0,
                        enriched_singular_values=sig1)

    def store(name, v, total_var):
        raw_var = variance(v, m)
        f = v / np.sqrt(total_var)
        err = equivariance_error(f, mats, act)
        mean = np.einsum('s,sic->ic', m, f)
        centered = f - mean
        # Check every component and cross-copy covariance, not just aggregate trace.
        cov = np.einsum('s,sic,sjd->ijcd', m, centered, centered, optimize=True)
        copy_cov = np.trace(cov, axis1=0, axis2=1) / 3
        iso = float(np.abs(cov - np.eye(3)[:, :, None, None] * copy_cov).max())
        assert max(iso, err, float(np.abs(mean).max())) < TOL
        frac = raw_var / total_var
        np.testing.assert_allclose(copy_cov.trace(), frac, rtol=0, atol=TOL)
        covs['copy_cov_' + name] = copy_cov
        catalog.append(dict(name=name, copies=v.shape[2], raw_per_component_variance=raw_var,
                            total_raw_per_component_variance=total_var, normalization_rms=np.sqrt(total_var),
                            normalized_variance=frac, rms_equivariance_max_abs=err,
                            rms_train_mean_max_abs=float(np.abs(mean).max()),
                            normalized_full_cov_isotropy_max_abs=iso))
        vectors.append(f)

    for seed, y in labels.items():
        v = vector_field(y, mats, inv)
        assert equivariance_error(v, mats, act) == 0
        # Under any of the 24 category renamings the copy axis merely permutes.
        for pi in permutations(range(4)):
            rec = vector_field(np.asarray(pi)[y], mats, inv).reshape(N, 3, 4, 3)
            assert np.array_equal(rec[:, :, pi, :], v.reshape(N, 3, 4, 3))
        h, power = diagnostics(y, v, m)
        r0, r1 = residual(v, q0, m), residual(v, q1, m)
        total_var = variance(v, m)
        assert total_var > 0 and variance(r1, m) <= variance(r0, m) + TOL <= total_var + 2*TOL
        for typ, arr in [('total', v), ('basic_residual', r0), ('enriched_residual', r1)]:
            store(f'{typ}_s{seed}', arr, total_var)
        rows = []
        for draw in range(1, 65):
            perm = np.random.default_rng([609006, seed, draw]).permutation(N)
            yy = y[perm]
            np.testing.assert_array_equal(np.bincount(yy, minlength=4), np.bincount(y, minlength=4))
            vv = vector_field(yy, mats, inv)
            assert equivariance_error(vv, mats, act) == 0
            hh, pp = diagnostics(yy, vv, m)
            dist2 = float(((hh - h) / np.log(4))**2 + (pp - power)**2)
            rows.append(dict(seed=seed, draw=draw, uint8_labels_sha256=array_sha(yy), power_T1u=pp,
                             entropy_nats=hh, learned_power_T1u=power, learned_entropy_nats=h,
                             power_difference=pp-power, entropy_difference_nats=hh-h,
                             entropy_difference_unit=(hh-h)/np.log(4), squared_distance=dist2))
        nearest = sorted(rows, key=lambda r: (r['squared_distance'], r['draw']))
        chosen = [row['draw'] for row in nearest[:8]]
        for rank, row in enumerate(nearest, 1):
            row.update(rank=rank, selected=int(rank <= 8))
        match.extend(rows)
        for draw in chosen:
            yy = y[np.random.default_rng([609006, seed, draw]).permutation(N)]
            vv = vector_field(yy, mats, inv)
            rr = residual(vv, q1, m)
            total = variance(vv, m)
            store(f'random_total_s{seed}_d{draw:02}', vv, total)
            store(f'random_enriched_residual_s{seed}_d{draw:02}', rr, total)
        d_entropy = float(np.mean([row['entropy_difference_unit'] for row in nearest[:8]]))
        d_power = float(np.mean([row['power_difference'] for row in nearest[:8]]))
        seed_info[seed] = dict(entropy_nats=h, power_T1u=power, selected_draws=chosen,
                               selected_max_squared_distance=nearest[7]['squared_distance'],
                               selected_mean_entropy_unit_minus_true=d_entropy,
                               selected_mean_T1u_power_minus_true=d_power,
                               matching_pass_both_at_0p10=max(abs(d_entropy), abs(d_power)) <= .10,
                               total_raw_variance=total_var,
                               basic_residual_raw_variance=variance(r0, m),
                               enriched_residual_raw_variance=variance(r1, m),
                               basic_residual_fraction=variance(r0, m)/total_var,
                               enriched_residual_fraction=variance(r1, m)/total_var)
    store('packing_basic', q0, variance(q0, m))
    store('packing_enriched', q1, variance(q1, m))
    assert len(catalog) == 59
    names = np.array([row['name'] for row in catalog])
    fields = np.zeros((len(names), N, 3, max(f.shape[-1] for f in vectors)), dtype='d')
    for i, f in enumerate(vectors):
        fields[i, :, :, :f.shape[-1]] = f
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'fields.npz', fields=fields, names=names)
    np.savez_compressed(out/'field-bank.npz', **field_detail, **covs)
    put_json(out/'map-lock.json', lock)
    put_csv(out/'field-bank.csv', catalog)
    put_csv(out/'random-matching.csv', match)
    info = dict(normalization='per-component total TRAIN RMS (trace over all copies); no re-scaling residuals',
                centering='Orbit-averaged equal-parent source TRAIN state mass. Polar-vector mean exactly zero by inversion. Covariance audit subtracts floating-point TRAIN mean.',
                representation='F_iak(s)=(3/48) sum_g R_ik(g) 1[label(g^-1 s)=a]. Flatten copies (category a, input axis k); all names and axes retained, no alignment/selection.',
                rename_permutations_per_learned_map=24, all_partition_integer_fields_exactly_equivariant=True,
                covariance='All 48 actions and every state checked. Every cross-copy TRAIN Cartesian covariance equals I times copy_cov in field-bank.npz, within field-bank.csv errors.',
                projections='TRAIN weighted LS shared across the 3 vector components, separately for each (a,k). Basic D_t=1[plus=t]-1[minus=t]. Enriched includes these, D_t*center_species_j, D_t*N_ab for each unordered opposite-pair count across all axes, and signed same-axis ordered contrasts ab-ba for a<b. No products of modifiers, exterior, or nonlinear terms.',
                comparators='Label-free packing_basic/enriched: equal trace of TRAIN-orthonormal vector columns. Seed-specific fitted vector = total - corresponding residual on same scale; autocovariances are not additive.',
                feature_order=feature_names, basic_rank=q0.shape[-1], enriched_rank=q1.shape[-1],
                random_matching='PCG64([609006,seed,draw]), draws 1..64 (06). Keep nearest 8 to true map by squared Euclidean distance in H_nats/log(4) and 06 centered label-span T1u projector trace / rank on orbit-averaged TRAIN mass; ties by lower draw. No spatial statistics used.',
                source_train_mass_sha256=file_sha(mass_path), orbit_mass_sha256=array_sha(m),
                raw_vs_orbit_max_mass_change=float(np.abs(raw-m).max()), min_orbit_state_mass=float(m.min()),
                max_normalized_equivariance_error=max(r['rms_equivariance_max_abs'] for r in catalog),
                max_normalized_full_cov_isotropy_error=max(r['normalized_full_cov_isotropy_max_abs'] for r in catalog),
                seeds=seed_info, field_shape=list(fields.shape), artifacts_sha256={name: file_sha(out/name) for name in
                    ('fields.npz','field-bank.npz','field-bank.csv','map-lock.json','random-matching.csv')})
    put_json(out/'mapping-receipt.json', info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--mass', type=Path, default=OUT/'train-mass.npy')
    args = parser.parse_args()
    main(args.out, args.mass)
