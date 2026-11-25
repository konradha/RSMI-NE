"""Gate 1: freeze parent identities first; materialize only the released cohorts.

Use ``roster`` before any physical outcomes, then ``build b side`` only after its
release. Paths are overridable for an installation with the same pinned inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/buffered-static-gate1'
WORK = Path('/home/fritz/rsmi-work')
OLD = WORK / 'stock-cavity-03-banks'
SCRATCH = WORK / 'buffered-static-gate1'
COLD = Path('/home/fritz/code/lattice-exploration/runs/l20-b320-b340-128x2-500m-aging-v1/prep')
COLD_DESIGN = ROOT / 'results/cold-frozen-transfer-320-340/design.json'
COLD_SEED = 202611251283
GROUP = ('fit', 'dev', 'assess', 'reserve')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pinned(path, expected):
    require(sha(path) == expected, f'SHA256 mismatch: {path}')
    return path


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view('u1')).hexdigest()


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def manifest(old):
    pins = json.loads((ROOT / 'results/experiment-05-inventory-transfer/design.json').read_bytes())['pins']
    return json.loads(pinned(old / 'prepared.json', pins['parent-census.json']).read_bytes())


def cold_source(prep):
    pins = json.loads(COLD_DESIGN.read_bytes())['pins']
    plan = json.loads(pinned(prep / 'plan.json', pins['plan']).read_bytes())
    require(plan['seed'] == COLD_SEED and plan['move'] == 'nonlocal' and plan['L'] == 20,
            'wrong cold campaign')
    batch = prep / 'batch-00000000'
    head = json.loads(pinned(batch / 'head.json', pins['head']).read_bytes())
    name = f'{500_000_000:020d}-{pins["500M"]}.npz'
    require(name in head['snapshots'] and head['plan_sha256'] == pins['plan'], 'wrong cold endpoint')
    return pinned(batch / name, pins['500M'])


def rank_rows(b, ids):
    return sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f'buffer1:{b}:{int(ids[i, 0])}:{int(ids[i, 1])}'.encode('ascii')).digest())


def make_roster(args):
    meta = manifest(args.old)
    ids, native, original, groups, indices, records = [], [], [], [], [], []
    source_pins = {}

    def add(b, group, keys, rows, old_rows):
        ids.append(keys); native.append(rows); original.append(old_rows)
        groups.append(np.full(len(keys), GROUP.index(group), 'u1'))
        indices.append(np.full(len(keys), b, 'u1'))
        records.append(dict(beta_index=b, group=group, parents=len(keys), ids_sha256=digest(keys),
                            native_rows_sha256=digest(rows), used_in_gate1=(group == 'assess' or
                            (b in (0, 9) and group in ('fit', 'dev')))))

    for b in range(10):
        index_space = []
        for side in ('train', 'test'):
            name = f'b{b:02}-{side}.npz'
            source_pins[name] = meta['inputs'][name]
            with np.load(pinned(args.old / name, source_pins[name])) as bank:
                keys = bank['all_ids']
                rows = bank['native_rows'].astype('<u2')
            require(keys.shape == (8192, 2) and keys.dtype == '<u8' and
                    digest(keys) == meta['coverage'][b][side]['identity'], 'invalid old IDs')
            index_space.append(rows)
            order = np.asarray(rank_rows(b, keys), dtype='<u2')
            n = 6144 if side == 'train' else 4096
            labels = ('fit', 'dev') if side == 'train' else ('assess', 'reserve')
            for group, ix in zip(labels, (order[:n], order[n:])):
                add(b, group, keys[ix], rows[ix], ix)
        np.testing.assert_array_equal(np.sort(np.concatenate(index_space)), np.arange(16384))
    path = cold_source(args.cold)
    source_pins[path.name] = sha(path)
    with np.load(path) as z:
        cid, labels = z['ids'], z['betas']
        np.testing.assert_array_equal(cid, z['parent_ids'])
        np.testing.assert_array_equal(cid, np.arange(256))
        np.testing.assert_array_equal(labels, np.tile(['3.20', '3.40'], 128))
    for b, label in ((10, '3.20'), (11, '3.40')):
        rows = np.flatnonzero(labels == label).astype('<u2')
        keys = np.column_stack((np.full(len(rows), COLD_SEED, '<u8'), cid[rows]))
        ix = rank_rows(b, keys)
        add(b, 'assess', keys[ix], rows[ix], np.full(len(rows), 65535, '<u2'))
    arrays = dict(ids=np.concatenate(ids), native_rows=np.concatenate(native),
                  original_side_rows=np.concatenate(original), group=np.concatenate(groups),
                  beta_index=np.concatenate(indices))
    require(len(np.unique(arrays['ids'], axis=0)) == len(arrays['ids']) == 164096, 'reused identities')
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / 'roster.npz'
    np.savez_compressed(dest, **arrays)
    receipt = dict(artifact=dest.name, sha256=sha(dest), source_manifest_sha256=sha(args.old/'prepared.json'),
                   source_pins=source_pins, group_codes=dict(enumerate(GROUP)), groups=records,
                   sort='ascending binary SHA256 of ASCII buffer1:{decimal beta_index}:{decimal seed}:{decimal chain_id}, within original side',
                   native_rows='zero-based release species array or native 500M cold checkpoint',
                   original_side_rows='zero-based row of 03 TRAIN or TEST, 65535 for 500M',
                   beta_values=[q['beta'] for q in meta['coverage']] + ['3.20', '3.40'],
                   roster_scope='No observations, exteriors, teacher energies or eligibility used.')
    json_write(args.out / 'roster.json', receipt)
    print(dest, receipt['sha256'], flush=True)


def staged(archive, key, directory):
    """Stage a compressed member as uninterpreted bytes. Decode ONLY authorized rows."""
    dest = directory / f'{key}.npy'
    with archive.open(dest.name) as source, dest.open('wb') as target:
        shutil.copyfileobj(source, target, length=1 << 20)
    return np.load(dest, mmap_mode='r', allow_pickle=False)


def flat(xyz):
    return xyz @ np.array([400, 20, 1], dtype='i2')


def geometry(old, meta):
    import stock_glass_cavity_exact as ex
    from stock_glass_buffered_features import mask
    f = pinned(old/'state-geometry.npz', meta['inputs']['state-geometry.npz'])
    with np.load(f) as data:
        for key, expected in (('states', ex.X), ('star', ex.C), ('touch', ex.U), ('ell', ex.L),
                              ('inner_edges', ex.A), ('cavity_to_outer', ex.J)):
            np.testing.assert_array_equal(data[key], expected)
        grid = data['anchors'].astype('i2')
    np.testing.assert_array_equal(grid, np.indices((6, 6, 6)).reshape(3, -1).T * 3)
    require(len(np.unique(flat((grid[:, None] + ex.C) % 20))) == 216*7, 'grid interiors overlap')
    xyz = np.indices((20, 20, 20), dtype='i2').reshape(3, -1).T
    off = dict(star=ex.C, touch=ex.U, **{f'r{r}': mask(r) for r in (2, 4, 5)})
    tables = {k: flat((xyz[:, None] + v.astype('i2')) % 20) for k, v in off.items()}
    checks = {}
    for r in (2, 4, 5):
        k = f'r{r}'
        require(tables[k].shape == (8000, 150) and len(np.unique(tables[k][0])) == 150,
                'incorrect number of measured shell sites')
        # Check the translation table on both sides of every periodic seam.
        for center in ((0, 0, 0), (0, 19, 9), (10, 10, 10), (19, 19, 19)):
            row = int(flat(np.array(center)))
            delta = np.abs(xyz[tables[k][row]][:, None] - xyz[tables['star'][row]])
            d = np.minimum(delta, 20-delta).sum(2)
            require(d.min() == r-1, f'incorrect periodic distance {k} at {center}')
            require(len(np.unique(tables[k][row])) == 150, 'duplicate shell sites')
        checks[k] = dict(sites=150, minimum_periodic_star_L1=r-1, offset_sha256=digest(off[k]))
    return grid, tables, checks


def first_replay(args, b, orig, arrays, shifts, counts, temp, meta):
    side = 'test' if args.side == 'assess' else 'train'
    name = f'b{b:02}-{side}.npz'
    parent, first = np.unique(arrays['parent'], return_index=True)
    with zipfile.ZipFile(pinned(args.old/name, meta['inputs'][name])) as z:
        np.testing.assert_array_equal(staged(z, 'eligible_counts', temp)[orig], counts)
        np.testing.assert_array_equal(staged(z, 'all_shifts', temp)[orig], shifts)
        selected = staged(z, 'selected', temp)
        # 03 packed one first star per eligible parent. TEST includes a sealed
        # reserve, even in the old eligible-ID index. In the sorted sparse
        # index, selected[k] must lie in [k,k+n_missed]. Read a packed row
        # only when *every* ID in that interval belongs to the allowed cohort.
        if side == 'test':
            allowed = np.zeros(8192, bool)
            allowed[orig] = True
            prefix = np.r_[0, np.cumsum(~allowed)]
            k = np.arange(len(selected))
            m = len(allowed)-len(selected)
            k = k[prefix[k+m+1] == prefix[k]]
            side_rows = selected[k]
        else:
            # All of original b09/b00 TRAIN is released; do not access the
            # held-from-fitting DEV anchor/X/E for the FIT comparison.
            k = np.searchsorted(selected, orig[parent])
            side_rows = selected[k]
            np.testing.assert_array_equal(side_rows, orig[parent])
        inverse = np.full(8192, -1, dtype='i4')
        inverse[orig[parent]] = first
        ix = inverse[side_rows]
        require((ix >= 0).all(), '03 replay row not authorized')
        for key, old_key in (('anchor', 'anchor'), ('observed', 'observed'), ('energy', 'relative_E')):
            np.testing.assert_array_equal(arrays[key][ix], staged(z, old_key, temp)[k])
    return dict(shifts_and_eligible_counts_matched=len(orig), first_anchor_native_X_and_teacher_matched=len(ix),
                blind_packed_row_rule='only rows provably outside reserve' if side == 'test' else 'released TRAIN')


def cold_replay(b, native_rows, arrays, counts):
    parent, first = np.unique(arrays['parent'], return_index=True)
    ref = ROOT / f'results/cold-frozen-transfer-320-340/target-{b}-500.npz'
    receipt = json.loads((ref.parent/'bank-receipt.json').read_bytes())
    with np.load(pinned(ref, receipt['artifacts'][ref.name])) as z:
        old_rows = native_rows // 2
        np.testing.assert_array_equal(z['ids'][old_rows], arrays['ids'])
        np.testing.assert_array_equal(z['counts'][old_rows], counts)
        k = np.searchsorted(z['parent'], old_rows[parent])
        for key, old_key in (('anchor', 'anchor'), ('observed', 'observed'), ('energy', 'relative_E')):
            np.testing.assert_array_equal(arrays[key][first], z[old_key][k])
    return dict(eligible_counts_matched=len(counts), first_anchor_native_X_and_teacher_matched=len(first))


def build(args):
    import stock_glass_cavity_exact as ex
    from stock_glass_modal import INDEX
    b, side = args.beta_index, args.side
    design = args.out/'design.md'
    require(design.is_file(), 'design must be published before reading a cohort')
    if side == 'assess':
        require(args.frozen_receipt is not None and args.frozen_receipt.is_file(), 'frozen receipt required')
    else:
        require(b in (0, 9) and side in ('fit', 'dev'), 'target TRAIN/DEV is sealed')
    args.scratch.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((args.out/'roster.json').read_bytes())
    with np.load(pinned(args.out/'roster.npz', receipt['sha256'])) as z:
        ix = (z['group'] == GROUP.index(side)) & (z['beta_index'] == b)
        ids, rows, orig = (z[k][ix] for k in ('ids', 'native_rows', 'original_side_rows'))
    require(len(ids) > 0, 'empty cohort')
    beta = float(receipt['beta_values'][b])
    meta = manifest(args.old)
    grid, table, mask_audit = geometry(args.old, meta)
    if b < 10:
        release = json.loads(pinned(args.archive/'release-index.json', INDEX).read_bytes())
        endpoint = release['endpoints'][b]
        require(endpoint['release_index'] == b and float(endpoint['beta']) == beta and
                release['axes'] == ['chain', 'z', 'y', 'x'], 'wrong beta or coordinate order')
        native_hash = endpoint['npz_sha256']
        path = pinned(args.archive/endpoint['npz'], native_hash)
        state_key, energy_key = 'species', 'energy'
    else:
        path = cold_source(args.cold)
        native_hash = receipt['source_pins'][path.name]
        state_key, energy_key = 'states', 'energies'
    blocks = {k: [] for k in ('parent', 'anchor', 'observed', 'energy', 'r2', 'r4', 'r5')}
    shifts = np.empty((len(ids), 3), 'u1')
    counts = np.zeros(len(ids), dtype='u1')
    native_h = np.empty(len(ids), 'i4')
    direct = None
    with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
        temp = Path(tmp)
        with zipfile.ZipFile(path) as archive, np.load(path) as header:
            if b < 10:
                require(str(header['axes']) == 'chain,z,y,x' and float(header['beta']) == beta, 'native metadata')
                np.testing.assert_array_equal(ids[:, 0], header['chain_seed'][rows])
                np.testing.assert_array_equal(ids[:, 1], header['chain_id'][rows])
            else:
                require(int(header['sweep']) == 500_000_000 and str(header['plan_sha256']) ==
                        json.loads(COLD_DESIGN.read_bytes())['pins']['plan'], 'cold time or lineage')
                np.testing.assert_array_equal(header['ids'][rows], ids[:, 1])
                np.testing.assert_array_equal(header['parent_ids'][rows], ids[:, 1])
                require((header['betas'][rows].astype('d') == beta).all(), 'cold beta mismatch')
            species = staged(archive, state_key, temp)
            native_h[:] = staged(archive, energy_key, temp)[rows]
        for start in range(0, len(ids), 64):
            sl = slice(start, min(start+64, len(ids)))
            raw = species[rows[sl]].reshape(-1, 20, 20, 20)
            require(raw.dtype == np.uint8, 'native species must be uint8')
            lat = raw if b < 10 else raw & 3
            for s, n in enumerate((2000, 2400, 3600)):
                require(np.all((lat == s).sum(axis=(1, 2, 3)) == n), 'invalid native composition')
            occ = lat != 0
            coord = np.zeros_like(lat)
            for d, axis in enumerate((3, 2, 1)):
                for k, step in enumerate((-1, 1)):
                    neighbor = np.roll(occ, step, axis=axis)
                    if b >= 10:
                        np.testing.assert_array_equal((raw >> (2 + 2*d + k)) & 1, neighbor)
                    coord += neighbor
            calc_h = (occ * (coord - ex.L[lat])**2).sum(axis=(1, 2, 3), dtype='i4')
            np.testing.assert_array_equal(calc_h, native_h[sl])
            lattice = lat.reshape(len(lat), -1)
            coords = coord.reshape(len(lat), -1)
            anchors = np.empty((len(lat), 216, 3), 'u1')
            for j, (seed, cid) in enumerate(ids[sl]):
                rng = np.random.Generator(np.random.PCG64([309031, b, int(seed), int(cid)]))
                shifts[start+j] = rng.integers(20, size=3)
                anchors[j] = (grid[rng.permutation(216)] + shifts[start+j]) % 20
            idx = flat(anchors)
            state = lattice[np.arange(len(lat))[:, None, None], table['star'][idx]]
            ok = ((state == 0).sum(2) == 2) & ((state == 1).sum(2) == 2)
            counts[sl] = ok.sum(1)
            j, priority = np.nonzero(ok)
            if not len(j):
                continue
            center = idx[j, priority]
            anchor = anchors[j, priority]
            state = state[j, priority]
            obs = ex.LOOK[state @ ex.CODE]
            np.testing.assert_array_equal(ex.X[obs], state)
            loc = j[:, None]
            oc = (state != 0).astype('i2')
            su = lattice[loc, table['touch'][center]]
            fc = coords[loc, table['star'][center]] - oc @ ex.A.T
            fu = coords[loc, table['touch'][center]] - oc @ ex.J.T
            require((fc >= 0).all() and (fc <= 6-ex.A.sum(1)).all() and
                    (fu >= 0).all() and (fu <= 6-ex.J.sum(1)).all(), 'invalid boundary counts')
            energy = ex.energies(fc, su, fu)
            if direct is None:
                wrap = np.flatnonzero(((anchor == 0) | (anchor == 19)).any(1))
                i = int(wrap[0]) if len(wrap) else 0
                # 03 audit adds signed coordinates; preserve signed arithmetic.
                result = ex.audit(lat[j[i]], anchor[i].astype('i2'), fc[i], su[i], fu[i], energy[i], beta)
                direct = dict(id=ids[start+j[i]].tolist(), **result)
            for key, val in (('parent', (j+start).astype('<u2')), ('anchor', anchor),
                             ('observed', obs), ('energy', energy)):
                blocks[key].append(val)
            for key in ('r2', 'r4', 'r5'):
                blocks[key].append(lattice[loc, table[key][center]])
        arrays = {k: np.concatenate(v) for k, v in blocks.items()}
        arrays['ids'] = ids
        np.testing.assert_array_equal(np.bincount(arrays['parent'], minlength=len(ids)), counts)
        replay = (first_replay(args, b, orig, arrays, shifts, counts, temp, meta) if b < 10 else
                  cold_replay(b, rows, arrays, counts))
        dest = args.scratch/f'b{b:02}-{side}.npz'
        pending = temp / dest.name
        np.savez(pending, **arrays)
        pending.replace(dest)
    audit = dict(beta_index=b, beta=beta, side=side, bank=dest.name, sha256=sha(dest),
                 design_sha256=sha(design), roster_sha256=receipt['sha256'], native_sha256=native_hash,
                 frozen_receipt_sha256=sha(args.frozen_receipt) if side == 'assess' else None,
                 roster_parents=len(ids), eligible_parents=int(np.count_nonzero(counts)),
                 missing_ids=ids[counts == 0].tolist(), candidate_stars=len(ids)*216,
                 eligible_stars=int(counts.sum()), eligible_count_hist=np.bincount(counts).tolist(),
                 observed_hist=np.bincount(arrays['observed'], minlength=210).tolist(),
                 parent_rows_sorted=True, parent_id_sha256=digest(ids), mask_geometry=mask_audit,
                 native_H_sha256=digest(native_h), checked_full_H=len(ids), full_H_errors=0,
                 composition=[2000, 2400, 3600], checked_native_sites=len(ids)*8000,
                 cold_neighbor_bits_checked=6*len(ids)*8000 if b >= 10 else 0,
                 species_decode='native uint8 (0,V;1,A;2,B)' if b < 10 else 'native low two bits (&3)',
                 axes='chain,z,y,x', center_order='center,-z,+z,-y,+y,-x,+x', full_210_state_audit=direct,
                 disjoint_candidate_sites=216*7, original_replay=replay)
    json_write(args.out/f'bank-audit-b{b:02}-{side}.json', audit)
    print(dest, audit['eligible_parents'], audit['eligible_stars'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, default=OLD)
    p.add_argument('--cold', type=Path, default=COLD)
    p.add_argument('--out', type=Path, default=RESULTS)
    p.add_argument('--scratch', type=Path, default=SCRATCH)
    p.add_argument('--archive', type=Path, default=Path('/home/fritz/datasets/lattice-glass/l20-16384-v1'))
    c = p.add_subparsers(dest='command', required=True)
    c.add_parser('roster')
    sub = c.add_parser('build')
    sub.add_argument('beta_index', type=int, choices=range(12))
    sub.add_argument('side', choices=('fit', 'dev', 'assess'))
    sub.add_argument('--frozen-receipt', type=Path)
    args = p.parse_args()
    if args.command == 'roster':
        make_roster(args)
    else:
        build(args)


if __name__ == '__main__':
    main()
