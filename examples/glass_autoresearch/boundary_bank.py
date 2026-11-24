"""Build private, occupancy-only L20 boundary banks from the fixed endpoint release.

Use a separate --out directory for each mode and index. The lock digest supplied
for validation/issue must be that of the previously published cold model lock.
"""
import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from rate import arrays, digest, json_out, partition, source_rows, table


L = 20
VOLUME = L**3
RADIUS = 4
SHELL = np.array([p for p in product(range(-4, 5), repeat=3) if max(map(abs, p)) == RADIUS], dtype=np.int8)
NN = np.array([np.eye(3, dtype=np.int8)[axis] * sign for axis in range(3) for sign in (-1, 1)])
CAP = 128
SAMPLE_SEED = 2187481
ISSUE_SEED = 5430143
PINS = {
    9: (3.05, 'f1ec9cb1dc837d58e9be4914dfc0b3cfb5760ece2f4e2ce995a0bc3e03261dca'),
    8: (2.711111111111, 'd6d8ad2476edae426a48ad94c1a324aa992d7b3a61d505795f324009f61ba87d'),
    0: (0., '0d2d07fa20a9fbf480775a248e1352aacdfdc3d7c3cb685fec22c52d709cafbf'),
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def pcg(seed, index, row):
    return np.random.Generator(np.random.PCG64([seed, index, int(row)]))


def shell_table():
    # Every rotated/reflected displacement has a unique bit, not an extra observation.
    if SHELL.shape != (386, 3):
        raise ValueError('invalid shell registration')
    lut = table(SHELL)
    for a in (0, 19, 7999):
        xyz = (np.asarray(np.unravel_index(a, (L,)*3)) + SHELL) % L
        if not np.array_equal(lut[a], np.ravel_multi_index(xyz.T, (L,)*3)):
            raise ValueError('periodic shell registration mismatch')
    return lut


def check_lock(args):
    if args.mode == 'train':
        if args.index != 9 or args.lock is not None or args.lock_sha256 is not None:
            raise ValueError('selection uses cold native TRAIN only, without a model lock')
        return None
    if args.mode == 'issue' and args.index != 9:
        raise ValueError('prospective issuance uses only the cold validation reserve')
    if args.lock is None or not args.lock_sha256:
        raise ValueError('supply the published cold model lock and its prior SHA256')
    payload = args.lock.read_bytes()
    if sha_bytes(payload) != args.lock_sha256:
        raise ValueError('cold model lock digest mismatch')
    sig = json.loads(payload)['signature']
    if (not isinstance(sig, list) or len(sig) != 3 or
            any(type(v) is not int or not 0 <= v <= 5 for v in sig) or sig != sorted(sig)):
        raise ValueError('lock must specify a sorted three-neighbor coordination signature')
    return sig


def native_ids(index, mode):
    train, validation, _ = partition(index)[0]
    if len(train) != 10240 or len(validation) != 2048:
        raise ValueError('unexpected native partition')
    if mode == 'train':
        return train, 'native_train'
    if index != 9:
        return validation, 'native_validation'
    if mode == 'issue':
        return validation[-1024:], 'native_validation_last_1024'
    return validation[:1024], 'native_validation_first_1024'


def check_receipt(path, index):
    beta, sha = PINS[index]
    receipt = json.loads(path.with_suffix('.json').read_bytes())
    if receipt['endpoint_sha256'] != sha or float(receipt['beta']) != beta:
        raise ValueError('source receipt differs from fixed release registration')


def cold_subset(path, ids, shell_indices=None):
    """Restricted analogue of rate.source_rows: do not materialize the other half-fold.

    For issuance, read only contiguous runs of the 386 chosen bytes, restoring
    their shell order. ZIP decompression may stream over intervening bytes, but
    no interior or other parent state is delivered to the issue extractor.
    """
    beta, sha = PINS[9]
    if digest(path) != sha:
        raise ValueError('changed cold archive')
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['axes']) != 'chain,z,y,x' or float(archive['beta']) != beta:
            raise ValueError('cold archive metadata mismatch')
        seeds, chains = archive['chain_seed'][ids], archive['chain_id'][ids]
    keys = np.column_stack((seeds, chains)).astype('<u8')
    if len(np.unique(keys, axis=0)) != len(ids):
        raise ValueError('duplicate selected lineage')
    width = VOLUME if shell_indices is None else len(SHELL)
    result = np.empty((len(ids), width), np.uint8)
    with zipfile.ZipFile(path) as z, z.open('species.npy') as f:
        readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        shape, fortran, dtype = readers[np.lib.format.read_magic(f)](f)
        if shape != (16384, L, L, L) or fortran or dtype != np.uint8:
            raise ValueError('invalid physical row format')
        base = f.tell()
        for i in np.argsort(ids):
            start = base + int(ids[i]) * VOLUME
            if shell_indices is None:
                f.seek(start)
                if f.readinto(memoryview(result[i])) != VOLUME:
                    raise ValueError('truncated archive row')
            else:
                order = np.argsort(shell_indices[i])
                sites = shell_indices[i, order].astype(np.int32)
                for run in np.split(np.arange(width), np.flatnonzero(np.diff(sites) != 1) + 1):
                    f.seek(start + int(sites[run[0]]))
                    vals = np.frombuffer(f.read(len(run)), dtype=np.uint8)
                    if len(vals) != len(run):
                        raise ValueError('truncated boundary')
                    result[i, order[run]] = vals
    if result.max() > 2:
        raise ValueError('invalid archive alphabet')
    return result, seeds, chains, sha_bytes(keys.tobytes())


def candidates(states, nn):
    """Return (batch parent position, anchor, sorted-signature code, orbit) at all 3:3 centers."""
    occ = (states != 0).reshape(-1, L, L, L)
    k = np.zeros(occ.shape, dtype=np.uint8)
    for axis in (1, 2, 3):
        for sign in (-1, 1):
            k += np.roll(occ, sign, axis)
    occ, k = occ.reshape(-1, VOLUME), k.reshape(-1, VOLUME)
    p, a = np.nonzero(~occ & (k == 3))
    arms = nn[a]
    occupied = occ[p[:, None], arms]
    # The three vacancies either occupy one site per opposite pair (8 tripods)
    # or include a complete pair and one lateral site (12 T's). This test and
    # the unordered occupied coordination tuple commute with every cubic isometry.
    y = np.all(occupied[:, ::2] != occupied[:, 1::2], axis=1).astype(np.uint8)
    coords = k[p[:, None], arms][occupied].reshape(-1, 3).astype(np.int16)
    coords.sort(axis=1)
    return p, a, 49 * coords[:, 0] + 7 * coords[:, 1] + coords[:, 2], y


def batches(states, loc, nn):
    for start in range(0, len(loc), 32):
        block = states[loc[start:start+32]]
        yield start, block, candidates(block, nn)


def select_signature(states, loc, nn):
    counts = np.zeros((7**3, 2), dtype=np.int64)
    for _, _, (_, _, code, y) in batches(states, loc, nn):
        counts += np.bincount(2 * code + y, minlength=counts.size).reshape(counts.shape)
    # Exact comparison of ab/(a+b); ascending base-7 code gives the lexicographic tiebreak.
    best, num, den = None, 0, 1
    for code, vals in enumerate(counts):
        t, tripod = map(int, vals)
        if t * tripod * den > num * (t + tripod):
            best, num, den = code, t * tripod, t + tripod
    if best is None:
        raise ValueError('no signature supports both arrangements on cold train')
    return [best // 49, best // 7 % 7, best % 7], counts


def matched_bank(states, loc, ids, index, sig, nn, shell):
    selected = sig[0] * 49 + sig[1] * 7 + sig[2]
    counts = np.zeros((len(ids), 2), dtype=np.int32)
    xs, ys, parents, anchors, weights = [], [], [], [], []
    for start, block, (p, a, code, y) in batches(states, loc, nn):
        mask = code == selected
        p, a, y = p[mask], a[mask], y[mask]
        n = len(block)
        freq = np.bincount(2*p+y, minlength=2*n).reshape(n, 2)
        counts[start:start+n] = freq
        sizes = freq.sum(axis=1)
        ends = np.cumsum(sizes)
        keep = []
        for i, (size, end) in enumerate(zip(sizes, ends)):
            if size > CAP:
                take = np.sort(pcg(SAMPLE_SEED, index, ids[start+i]).choice(size, CAP, replace=False))
            else:
                take = np.arange(size)
            keep.append(end - size + take)
        take = np.concatenate(keep)
        pp, aa = p[take], a[take]
        xs.append(np.packbits(block[pp[:, None], shell[aa]] == 0, axis=1, bitorder='little'))
        ys.append(y[take]); parents.append(ids[start+pp].astype(np.int32)); anchors.append(aa.astype(np.int16))
        weights.append(sizes[pp] / np.minimum(sizes[pp], CAP))
    return dict(zip(('x', 'y', 'parent', 'anchor', 'weight'), map(np.concatenate, (xs, ys, parents, anchors, weights)))), counts


def run(args):
    sig = check_lock(args)  # before opening even a receipt from a transfer cohort
    ids, split = native_ids(args.index, args.mode)
    out = args.out
    bank, meta = out/'bank.npz', out/'meta.json'
    if bank.exists() or meta.exists():
        raise FileExistsError('refusing to replace an existing bank')
    archive = args.data/f'beta-{args.index:02d}.npz'
    check_receipt(archive, args.index)
    shell = shell_table()
    details = {}
    if args.mode == 'issue':
        # Anchors depend only on fixed stream registration and native row, before
        # any microscopic byte is read. No center test, retries or eligibility mask.
        anchor = np.array([pcg(ISSUE_SEED, args.index, r).integers(VOLUME) for r in ids], dtype=np.int16)
        raw, seeds, chains, lineage = cold_subset(archive, ids, shell[anchor])
        data = dict(x=np.packbits(raw == 0, axis=1, bitorder='little'), parent=ids.astype(np.int32),
                    anchor=anchor, weight=np.ones(len(ids)), chain_seed=seeds, chain_id=chains)
        details.update(issued_parent_count=len(ids), eligibility_inspected=False, issue_seed=ISSUE_SEED,
                       anchor_rule='PCG64([issue_seed,index,native_row]).integers(8000), independent of the lattice')
    else:
        nn = table(NN)
        if args.index == 9 and args.mode == 'validation':
            states, _, _, lineage = cold_subset(archive, ids)
            loc = np.arange(len(ids))
        else:
            fold = int(args.mode == 'validation')
            (lookup, states), src = source_rows(args.data, args.index, (fold,))
            if (src['beta'], src['sha256']) != PINS[args.index]:
                raise ValueError('source differs from pinned archive')
            loc = lookup[ids]
            if np.any(loc < 0):
                raise ValueError('missing native row')
            lineage = src['lineage_sha256'][str(fold)]
        if args.mode == 'train':
            sig, census = select_signature(states, loc, nn)
            details['train_signature_census_T_tripod'] = {
                f'{s//49},{s//7%7},{s%7}': v.tolist() for s, v in enumerate(census) if v.any()}
        data, counts = matched_bank(states, loc, ids, args.index, sig, nn, shell)
        total = counts.sum(axis=0)
        if args.mode == 'train' and not np.array_equal(total, census[sig[0]*49 + sig[1]*7 + sig[2]]):
            raise ValueError('census and extraction disagree')
        details.update(eligible_counts_T_tripod=total.tolist(), eligible_per_parent_T_tripod=counts.tolist(),
                       nonempty_parents=int(np.count_nonzero(counts.sum(axis=1))),
                       nonempty_parents_T_tripod=(counts > 0).sum(axis=0).tolist(),
                       anchors_censused_per_parent=VOLUME, cap=CAP, sample_seed=SAMPLE_SEED,
                       sampling='Within each parent, all eligible anchors up to cap, else PCG64([sample_seed,index,native_row]) uniform without replacement; weight = eligible/retained.')
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    arrays(bank, **data)
    beta, source_sha = PINS[args.index]
    json_out(meta, dict(schema='l20.boundary-bank.v1', mode=args.mode, split=split, index=args.index,
                        beta=beta, signature=sig, parent_ids=ids.tolist(), n=len(data['x']),
                        bank_sha256=digest(bank), source_sha256=source_sha, source_file=archive.name,
                        source_receipt_sha256=digest(archive.with_suffix('.json')), lineage_sha256=lineage,
                        implementation_sha256=digest(Path(__file__)), lock_sha256=args.lock_sha256,
                        shell_sha256=sha_bytes(SHELL.tobytes()), shell_sites=len(SHELL),
                        shell_order='int8 (dz,dy,dx) lexicographic, max(abs)=4, periodic L=20; offsets C bytes hashed',
                        x_encoding='vacant=1, occupied=0, little-endian packbits, 49 bytes (6 high zero padding bits)',
                        anchor_encoding='native flat z*400+y*20+x; identifiers, eligibility and anchors are not predictors',
                        target='3:3 occupied/vacant axial neighbors of a vacant center; match sorted k of the three occupied neighbors. tripod=1 (one vacant arm per axis); T=0.',
                        **details))
    print('BOUNDARY_BANK', args.mode, args.index, len(ids), len(data['x']), bank, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'validation', 'issue'))
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--index', required=True, type=int, choices=sorted(PINS))
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--lock-sha256')
    os.umask(0o077)  # Private banks, including the temporary files used for atomic publication.
    run(parser.parse_args())
