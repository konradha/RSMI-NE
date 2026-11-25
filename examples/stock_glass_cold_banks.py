"""Prepare the locked cold-transfer banks. No trajectories, fits or target readouts.

``prepare(prep, banks, out)`` takes the nonlocal campaign's prep directory and
original extracted 03 banks. In each target file, the first row of a run of
``parent`` is its primary star; ``ids``, ``counts``, ``energy`` and ``shift``
cover the full roster, including parents with no matching star.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

import stock_glass_cavity_exact as exact
from stock_glass_modal import SEEDS, sha

RESULTS = Path(__file__).resolve().parents[1] / 'results'
DESIGN = RESULTS / 'cold-frozen-transfer-320-340/design.json'
REF = RESULTS / 'experiment-03-cavity'
REF05 = RESULTS / 'experiment-05-inventory-transfer'
HEADS = RESULTS / 'experiment-07-boundary-forecast/fit-b09.npz'
BETAS = {10: '3.20', 11: '3.40'}
AGES = (300, 500)
SEED = 202611251283
DEPTH = 10218


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _check(path, expected):
    digest = sha(path)
    _require(digest == expected, f'SHA256 mismatch: {path}')
    return digest


def _digest(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def _source(banks, pins):
    """Establish disjointness against all 20 source rosters, not just scored rows."""
    old_pins = json.loads((REF05 / 'design.json').read_bytes())['pins']
    manifest_hash = _check(banks / 'prepared.json', old_pins['parent-census.json'])
    meta = json.loads((banks / 'prepared.json').read_bytes())
    hashes = meta['inputs']
    _require(hashes['b09-test.npz'] == pins['source_test'], 'wrong original held bank')
    keys = set()
    roster_receipts = {}
    for b in range(10):
        record = meta['coverage'][b]
        _require(record['index'] == b, 'unexpected source beta order')
        for side in ('train', 'test'):
            name = f'b{b:02}-{side}.npz'
            _check(banks / name, hashes[name])
            with np.load(banks / name) as bank:
                ids = bank['all_ids']
                identity = _digest(ids)
                _require(ids.dtype == np.dtype('<u8') and ids.shape == (8192, 2)
                         and identity == record[side]['identity'], f'invalid source roster: {name}')
                here = set(map(tuple, ids.tolist()))
                _require(len(here) == len(ids) and keys.isdisjoint(here), 'source roster collision')
                keys.update(here)
                if b == 9 and side == 'test':
                    sel = bank['selected']
                    np.testing.assert_array_equal(sel, np.flatnonzero(bank['eligible_counts']))
                    _require(len(sel) == 8187 and record['beta'] == '3.05', 'incorrect source TEST')
                roster_receipts[name] = dict(roster_sha256=identity, bank_sha256=hashes[name])
    geom = 'state-geometry.npz'
    _check(banks / geom, hashes[geom])
    with np.load(banks / geom) as f:
        for k, v in [('states', exact.X), ('star', exact.C), ('touch', exact.U),
                     ('inner_edges', exact.A), ('cavity_to_outer', exact.J), ('ell', exact.L)]:
            np.testing.assert_array_equal(v, f[k])
        grid = f['anchors']
    np.testing.assert_array_equal(grid, np.indices((6, 6, 6)).reshape(3, -1).T * 3)
    sites = ((grid[:, None] + exact.C) % 20).reshape(-1, 3)
    _require(len(np.unique(sites, axis=0)) == 216 * 7, 'candidate interiors overlap')
    return keys, grid, dict(prepared_sha256=manifest_hash, geometry_sha256=hashes[geom],
                            rosters=roster_receipts, parents=len(keys), source_test_eligible=8187,
                            source_test_roster=8192)


def _frozen(out, pins):
    """Replay only the three specified final checkpoints, using 05 TF32-RN."""
    files = dict(stock_checkpoints=REF/'corrected/stock-checkpoints.zip',
                 code_maps=REF/'corrected/code-maps.npz', linear=REF/'frozen-modes.npz',
                 transfer05=REF05/'frozen-transfer.npz', heads07=HEADS)
    for k, path in files.items():
        _check(path, pins[k])
    first = REF / 'corrected/first-updates.json'
    first_hash = _check(first, json.loads((REF05/'design.json').read_bytes())['pins']['corrected/first-updates.json'])
    checksums = json.loads(first.read_bytes())['checkpoints']
    features = np.stack((exact.X == 1, exact.X == 2), axis=-1).reshape(210, 14).astype('d')
    xyz = tuple((exact.C + 1).T)
    labels, coeffs, nulls, audit = [], [], [], []
    with (zipfile.ZipFile(files['stock_checkpoints']) as pack, np.load(files['code_maps']) as maps,
          np.load(files['linear']) as modes, np.load(files['transfer05']) as transfer):
        linear = modes['b09_linear']
        _require(linear.shape == (210,) and np.isfinite(linear).all(), 'invalid site-linear comparator')
        for seed in SEEDS:
            stem = f'b09-s{seed}'
            member = stem + '.npz'
            key = stem + f'_d{DEPTH}'
            data = pack.read(member)
            _require(hashlib.sha256(data).hexdigest() == checksums[member], f'invalid {member}')
            with np.load(io.BytesIO(data)) as checkpoint:
                r = json.loads(checkpoint['receipt'].item())
                _require(r['seed'] == seed and r['index'] == 9 and r['beta'] == '3.05'
                         and r['fit_status'] == 'complete' and r['steps'] == r['final'] == DEPTH
                         and r['checkpoints'][str(DEPTH)]['map_valid'], f'incomplete {member}')
                weights = checkpoint[f'w{DEPTH}'][xyz].reshape(14, 2)
                native = checkpoint[f'logits{DEPTH}']
            np.testing.assert_array_equal(native, maps[key + '_logits'])
            _require(weights.dtype == np.float32 and np.isfinite(weights).all()
                     and native.shape == (210, 2) and np.isfinite(native).all(), 'invalid frozen logits')
            bits = weights.view('u4')
            qw = ((bits + 4095 + ((bits >> 13) & 1)) & np.uint32(0xffffe000)).view('f4')
            z = features @ qw
            np.testing.assert_allclose(z, native, rtol=2e-5, atol=3e-5)
            np.testing.assert_array_equal(z >= 0, native >= 0)
            # Same exact-addition certificate for borderline signs as in 05.
            near = abs(z) <= 16 * np.finfo('f4').eps * (features @ abs(qw))
            for i, h in zip(*np.nonzero(near)):
                terms = [float(w).as_integer_ratio() for w in qw[features[i].astype(bool), h]]
                den = max(d for _, d in terms)
                bound = sum(abs(n) * (den // d) for n, d in terms)
                _require(bound.bit_length() <= 23, 'unresolved TF32 sign')
            lab = (2 * (z[:, 0] >= 0) + (z[:, 1] >= 0)).astype('u1')
            q = maps[key + '_hard_Q']
            a = maps[key + '_hard_a']
            np.testing.assert_array_equal(np.eye(4)[lab], q)
            np.testing.assert_array_equal(lab, transfer[stem + '_223'])
            np.testing.assert_array_equal(a, transfer[stem + '_a'])
            mass = modes['b09_mean'] @ q
            np.testing.assert_allclose(mass, transfer[stem + '_source_mass'], rtol=1e-14, atol=1e-15)
            _require(a.shape == (4,) and np.isfinite(a).all() and (mass > 1e-14).all()
                     and transfer[stem + '_identified'].all(), 'unidentified 03 TRAIN contrast')
            rng = np.random.Generator(np.random.PCG64([509005, 2, 2, 3, seed]))
            null = np.stack([lab[rng.permutation(len(lab))] for _ in range(64)])
            counts = np.bincount(lab, minlength=4)
            _require(all(np.array_equal(np.bincount(v, minlength=4), counts) for v in null),
                     'null membership counts changed')
            labels.append(lab)
            coeffs.append(a)
            nulls.append(null)
            audit.append(dict(seed=seed, update=DEPTH, kernel_sha256=checksums[member],
                              states=210, signs=420, sign_mismatches=0, membership_mismatches=0,
                              near_zero_signs=int(near.sum()), max_logit_error=float(abs(z-native).max()),
                              counts=counts.tolist(), labels_sha256=_digest(lab), coeffs_sha256=_digest(a),
                              null_labels_sha256=_digest(null)))
    np.savez_compressed(out/'frozen.npz', labels=np.stack(labels), coeffs=np.stack(coeffs),
                        linear=linear, null_labels=np.stack(nulls))
    return dict(first_updates_sha256=first_hash, seeds=audit, linear_sha256=_digest(linear))


def _snapshot(file, age, ph):
    """Check every packed site in native C order: chain,z,y,x; +x starts at bit 2."""
    with np.load(file) as z:
        _require(str(z['plan_sha256']) == ph and int(z['sweep']) == age * 10**6,
                 f'wrong lineage: {file.name}')
        cid, beta, raw, energy = (z[k] for k in ('ids', 'betas', 'states', 'energies'))
        _require(cid.ndim == 1 and cid.dtype == np.uint64 and beta.shape == cid.shape
                 and energy.shape == cid.shape and energy.dtype == np.int32
                 and raw.dtype == np.uint8 and raw.shape == (len(cid), 8000), 'invalid native schema')
        np.testing.assert_array_equal(z['parent_ids'], cid)
        np.testing.assert_array_equal(z['parent_age'], np.zeros_like(cid))
        np.testing.assert_array_equal(z['parent_hash'], np.full(cid.shape, ''))
        ac = z['accepted']
        _require(ac.shape == cid.shape and ac.dtype == np.uint64 and (ac <= age * 10**6 * 8000).all(),
                 'invalid accepted counts')
    packed = raw.reshape(-1, 20, 20, 20)
    lat = packed & 3
    for sp, expected in enumerate((2000, 2400, 3600)):
        _require(np.all((lat == sp).sum(axis=(1, 2, 3)) == expected), 'invalid composition')
    occ = (lat != 0).astype('u1')
    coord = np.zeros_like(occ)
    for d, axis in enumerate((3, 2, 1)):
        for k, step in enumerate((-1, 1)):
            neighbor = np.roll(occ, step, axis=axis)
            np.testing.assert_array_equal((packed >> (2 + 2*d + k)) & 1, neighbor)
            coord += neighbor
    h = (occ * (coord - exact.L[lat])**2).sum(axis=(1, 2, 3), dtype=np.int32)
    np.testing.assert_array_equal(h, energy)
    # Canonical per-beta roster is ordered by chain ID. Aliases, if present,
    # can never increase the independent-parent count or disagree on payload.
    unique, first, inverse = np.unique(cid, return_index=True, return_inverse=True)
    np.testing.assert_array_equal(unique, np.arange(256, dtype=np.uint64))
    if len(first) != len(cid):
        for v in (beta, raw, energy, ac):
            np.testing.assert_array_equal(v, v[first][inverse])
    np.testing.assert_array_equal(beta[first], np.tile(list(BETAS.values()), 128))
    return unique, lat[first], energy[first], len(cid) - len(first)


def _schedule(ids, b, grid):
    shift = np.empty((len(ids), 3), 'u1')
    order = np.empty((len(ids), 216), 'u1')
    for i, (seed, cid) in enumerate(ids):
        rng = np.random.Generator(np.random.PCG64([309031, b, int(seed), int(cid)]))
        shift[i] = rng.integers(20, size=3)
        order[i] = rng.permutation(216)
    return shift, (grid[order] + shift[:, None, :]) % 20, _digest(order)


def _stars(lat, ids, shift, candidates, energy, b, age):
    pos = (candidates[:, :, None, :] + exact.C) % 20
    states = lat[np.arange(len(ids))[:, None, None], *np.moveaxis(pos, -1, 0)]
    ok = ((states == 0).sum(2) == 2) & ((states == 1).sum(2) == 2)
    counts = ok.sum(1).astype('u1')
    # nonzero is C-ordered: all stars of a parent in its original random priority order.
    parent, priority = np.nonzero(ok)
    _require(len(parent) > 0, 'no eligible target')
    anchor = candidates[parent, priority]
    state = states[parent, priority]
    obs = exact.LOOK[state @ exact.CODE]
    _require((obs < 210).all(), 'bad state lookup')
    np.testing.assert_array_equal(state, exact.X[obs])

    def gather(off):
        shape = (-1,) + (1,) * (off.ndim-1)
        xyz = (anchor.reshape(shape + (3,)) + off) % 20
        return lat[parent.reshape(shape), *np.moveaxis(xyz, -1, 0)]

    occ = (state != 0).astype('i2')
    su = gather(exact.U)
    fc = (gather(exact.C[:, None] + exact.DIR) != 0).sum(2) - occ @ exact.A.T
    fu = (gather(exact.U[:, None] + exact.DIR) != 0).sum(2) - occ @ exact.J.T
    _require(((fc >= 0) & (fc <= 6 - exact.A.sum(1))).all() and
             ((fu >= 0) & (fu <= 6 - exact.J.sum(1))).all(), 'invalid exterior counts')
    en = exact.energies(fc, su, fu)
    row = int(parent[0])
    check = exact.audit(lat[row], anchor[0], fc[0], su[0], fu[0], en[0], float(BETAS[b]))
    np.testing.assert_array_equal(np.bincount(parent, minlength=len(ids)), counts)
    arrays = dict(ids=ids, energy=energy, shift=shift, counts=counts, parent=parent.astype('u1'),
                  anchor=anchor.astype('u1'), observed=obs, relative_E=en, su=su, fu=fu.astype('u1'))
    primary = np.full((len(ids), 3), -1, dtype='i2')
    eligible, first = np.unique(parent, return_index=True)
    primary[eligible] = anchor[first]
    missing = counts == 0
    coverage = dict(index=b, beta=BETAS[b], age_M=age, roster=len(ids), eligible=len(eligible),
                    missed=int(missing.sum()), excluded_ids=ids[missing].tolist(),
                    candidates=int(ok.size), eligible_stars=len(parent),
                    eligible_count_histogram=np.bincount(counts).tolist(),
                    shift_axis_histograms=[np.bincount(v, minlength=20).tolist() for v in shift.T],
                    primary_wraps=int(np.any((primary[eligible] == 0) | (primary[eligible] == 19), axis=1).sum()),
                    roster_sha256=_digest(ids), energy_sha256=_digest(energy), shifts_sha256=_digest(shift),
                    selected_anchors_sha256=_digest(primary),
                    full_H_audit=dict(parent=row, id=ids[row].tolist(), **check))
    return arrays, primary, coverage


def prepare(prep: Path, banks: Path, out: Path) -> dict:
    """Write frozen.npz and target-{10,11}-{300,500}.npz; return JSON-ready evidence."""
    pins = json.loads(DESIGN.read_bytes())['pins']
    plan_hash = _check(prep/'plan.json', pins['plan'])
    plan = json.loads((prep/'plan.json').read_bytes())
    _require(plan['seed'] == SEED and plan['move'] == 'nonlocal' and plan['L'] == 20 and
             plan['chains'] == 128 and plan['width'] == 256 and plan['betas'] == list(BETAS.values())
             and plan['sweeps'] == [500_000_000]*2, 'unexpected preparation')
    batch = prep/'batch-00000000'
    head_path = batch/'head.json'
    _require(set(prep.glob('batch-*/head.json')) == {head_path}, 'unexpected preparation heads')
    _check(head_path, pins['head'])
    head = json.loads(head_path.read_bytes())
    _require(head['batch'] == 0 and head['plan_sha256'] == plan_hash, 'wrong batch head')
    files = {}
    for age in AGES:
        name = f'{age * 10**6:020d}-{pins[str(age)+"M"]}.npz'
        _require([n for n in head['snapshots'] if n.startswith(name[:21])] == [name], 'wrong age entry')
        _check(batch/name, pins[f'{age}M'])
        files[age] = batch/name

    out.mkdir(parents=True, exist_ok=True)
    source_keys, grid, source = _source(banks, pins)
    frozen = _frozen(out, pins)
    hashes = {'frozen.npz': sha(out/'frozen.npz')}
    coverage, pairs, native = [], [], []
    primary, schedules, rosters = {}, {}, {}
    for age, path in files.items():
        cid, lat, energy, duplicates = _snapshot(path, age, plan_hash)
        native.append(dict(age_M=age, checkpoint=str(path.relative_to(prep)), sha256=pins[f'{age}M'],
                           distinct_parents=len(cid), duplicate_rows=duplicates, site_masks_checked=6*8000*len(cid),
                           energy_mismatches=0, composition_per_parent=[2000, 2400, 3600]))
        for b in BETAS:
            rows = np.arange(b - 10, 256, 2)
            ids = np.column_stack((np.full(len(rows), SEED, dtype='<u8'), cid[rows]))
            here = set(map(tuple, ids.tolist()))
            _require(len(here) == 128 and source_keys.isdisjoint(here), 'identity collision')
            if age == AGES[0]:
                _require(all(here.isdisjoint(map(tuple, other.tolist())) for other in rosters.values()),
                         'identity reused across temperatures')
                rosters[b] = ids
                schedules[b] = _schedule(ids, b, grid)
            else:
                np.testing.assert_array_equal(ids, rosters[b])
            shift, candidates, order_hash = schedules[b]
            arrays, selection, record = _stars(lat[rows], ids, shift, candidates, energy[rows], b, age)
            name = f'target-{b}-{age}.npz'
            np.savez_compressed(out/name, **arrays)
            hashes[name] = sha(out/name)
            coverage.append(dict(bank=name, candidate_priority_sha256=order_hash, **record))
            if age == AGES[0]:
                primary[b] = selection
            else:
                old = primary[b]
                ok = (old[:, 0] >= 0) & (selection[:, 0] >= 0)
                same = ok & (old == selection).all(1)
                pairs.append(dict(index=b, beta=BETAS[b], roster=len(ids), complete=int(ok.sum()),
                                  unchanged=int(same.sum()), changed=int((ok & ~same).sum()),
                                  incomplete_ids=ids[~ok].tolist(), unchanged_ids=ids[same].tolist()))
    return dict(design_sha256=sha(DESIGN), pins=pins, artifacts=hashes, source=source, frozen=frozen,
                native=native, coverage=coverage, age_pairs=pairs, source_target_identity_overlap=0,
                target_parents=256, native_axes='chain,z,y,x', star_order='center,-z,+z,-y,+y,-x,+x')
