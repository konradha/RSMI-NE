"""#19–20: reversible Dirichlet forms and two-bond gate changes, never trajectories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import monotonic
from zipfile import ZipFile

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from fifteen_dynamics_closure import IDEAL, edge_cost, nb_sum, sha, store
from tile_codec import TILE, POW
from tranche2_gates_stats import fit_locks, report

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/l20-autoresearch/tranche2'
CODE = ROOT / 'results/l20-autoresearch/codec/codec.npz'
N, BETA, B = 8000, 3.05, 128
K = 2 * np.pi / 20
SLICES = {'fit': (0, 4096), 'development': (4096, 4608), 'audit': (4608, 5120)}
COLUMNS = ['eligible', 'accepted_e', 'open_count', 'close_count', 'open_flux', 'close_flux',
           'open_magnitude_flux', 'close_magnitude_flux', 'far_eligible', 'far_crossings',
           'changed_acceptance_count', 'far_changed_acceptance_count']


def section(file, name, start, stop):
    """Read only the requested contiguous bank rows, including for audit-blind calibration."""
    with file.open(name + '.npy') as src:
        version = np.lib.format.read_magic(src)
        shape, fortran, dt = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}[version](src)
        if shape[0] != 5120 or fortran or not 0 <= start < stop <= shape[0]:
            raise ValueError((name, shape, start, stop))
        row = int(np.prod(shape[1:], dtype=np.int64)) * dt.itemsize
        src.seek(src.tell() + start * row)
        return np.frombuffer(src.read((stop - start) * row), dt).reshape(stop - start, *shape[1:])


def geometry(lut):
    i = np.arange(N).reshape(20, 20, 20)
    r = np.indices(i.shape).reshape(3, -1)
    plus = np.column_stack([np.roll(i, -1, d).ravel() for d in range(3)])
    neighbors = np.column_stack([np.roll(i, sign, d).ravel() for d in range(3) for sign in (-1, 1)])
    a = np.repeat(i.ravel(), 3); b = plus.ravel()
    tile = np.ravel_multi_index(r // 2, (10,) * 3)
    power = POW[np.ravel_multi_index(r % 2, (2,) * 3)].astype(np.int16)
    tile_site = i.reshape(10, 2, 10, 2, 10, 2).transpose(0, 2, 4, 1, 3, 5).reshape(1000, 8)
    t = np.cos(K * (r[2, tile_site[:, 0]] + 0.5)); w = np.cos(K * r[2])
    words = np.arange(3**8)[:, None] // POW % 3
    stag = (words == 0).astype(np.int16) @ (1 - 2*TILE)
    feat = np.column_stack([(stag*stag).sum(1), lut[:, None] == np.arange(3)]).astype(np.int16)
    same = tile[a] == tile[b]
    # Establish owner-intersection sets before computing any effects, including PBC.
    def owners(u, v):
        return {u, v, *neighbors[u], *neighbors[v]}
    sy = plus[0, 1]; sx = plus[0, 2]; syx = plus[sy, 2]; fy = i[0, 5, 0]
    assert owners(0, sx) & owners(sy, syx) == {0, sx, sy, syx}
    assert not owners(0, sx) & owners(fy, plus[fy, 2])
    return dict(a=a, b=b, plus=plus, neighbors=neighbors, tile=tile, power=power, tile_site=tile_site,
                t=t, w=w, feat=feat, same=same, lut=lut, far=np.roll(i, -5, axis=1).ravel())


def energy(s, m):
    return (s != 0) * (m - IDEAL[s])**2


def energy_jump(x, o, m, w, g):
    """Fourier-weighted sum of the two endpoint changes and ten other owner changes."""
    a, b = g['a'], g['b']; sa, sb = x[a], x[b]; oa, ob = o[a], o[b]
    d = ob-oa; residual = (m-IDEAL[x])*o
    moment = nb_sum((w*residual).reshape(20, 20, 20)).ravel()
    count = nb_sum((w*o).reshape(20, 20, 20)).ravel()
    endpoint = w[a]*(energy(sb, m[a]-d)-energy(sa, m[a])) + w[b]*(energy(sa, m[b]+d)-energy(sb, m[b]))
    outside = 2*d*(moment[a]-w[b]*residual[b]-moment[b]+w[a]*residual[a])
    outside += d*d*(count[a]-w[b]*ob+count[b]-w[a]*oa)
    return endpoint+outside


def dirichlet(x, o, m, de, prob, words, g, check):
    a, b = g['a'], g['b']; w, t, table = g['w'], g['t'], g['feat']; tile, power = g['tile'], g['power']
    A, B_ = (x == 1).astype(np.int16), (x == 2).astype(np.int16)
    fields = np.r_[w@A, w@B_, w@energy(x, m), t@table[words]]
    delta = np.empty((3*N, 7)); dw = w[a]-w[b]
    delta[:, 0] = dw*(A[b]-A[a]); delta[:, 1] = dw*(B_[b]-B_[a]); delta[:, 2] = energy_jump(x, o, m, w, g)
    ta, tb = tile[a], tile[b]; wa, wb = words[ta], words[tb]; same = g['same']; v = x[b]-x[a]
    nexta = wa + v*(power[a] - same*power[b]); nextb = wb - v*power[b]*(~same)
    delta[:, 3:] = ((table[nexta]-table[wa])*t[ta, None] + (table[nextb]-table[wb])*t[tb, None])
    err = np.max(np.abs(delta[:, :2].sum(0) + 2*(1-np.cos(K))*fields[:2]))
    if err > 1e-10 or (check and (delta[x[a] == x[b]] != 0).any()):
        raise ValueError('unit-acceptance density drift or equal-species null control')
    if check and not np.array_equal(energy_jump(x, o, m, np.ones_like(w), g), de):
        raise ValueError('uniform-weight owner sum and tested native delta differ')
    D = (delta.T*prob) @ delta/(2*3*N)
    return fields, D, err


def acc(d):
    return np.exp(-BETA*np.maximum(d, 0))


def shared_owner(s, se, sf, m, m_e, m_f, on_e):
    """Only a scalar label and coordination substitutions in a single owner term."""
    sef = se if on_e else sf
    return energy(sef, m+m_e+m_f)-energy(se, m+m_e)-energy(sf, m+m_f)+energy(s, m)


def mixed(x, o, m, g):
    a = np.arange(N); b = g['plus'][:, 2]; c = g['plus'][:, 1]; d = g['plus'][c, 2]
    u, v = o[b]-o[a], o[d]-o[c]
    result = shared_owner(x[a], x[b], x[a], m[a], -u, v, True)
    result += shared_owner(x[b], x[a], x[b], m[b], u, -v, True)
    result += shared_owner(x[c], x[c], x[d], m[c], u, -v, False)
    result += shared_owner(x[d], x[d], x[c], m[d], -u, v, False)
    return result


def gates(x, o, m, de, words, g, check):
    b = g['plus'][:, 2]; c = g['plus'][:, 1]; d = b[c]; cf = g['far']; dfar = b[cf]
    e = de[:, 2]; f = e[c]; mix = mixed(x, o, m, g); aft = f+mix
    ae = acc(e); old, new = acc(f), acc(aft); change = new-old
    unlike = x != x[b]; ok = unlike & (x[c] != x[d]); far = unlike & (x[cf] != x[dfar])
    opening = ok & (old < .5) & (new >= .5); closing = ok & (old >= .5) & (new < .5)
    # Beyond overlapping owners the empty mixed sum is identically zero. A distinct
    # scalar-barrier oracle below checks both near and far without building a new state.
    error = np.max(np.abs(ae-np.exp(-BETA*e)*acc(-e)))
    if error > 1e-12 or np.any(mix[~ok]):
        raise ValueError('Metropolis balance or a null-edge mixed difference failed')
    if check:
        verify_barriers(x, o, m, g, c, f, aft)
        verify_barriers(x, o, m, g, cf, e[cf], e[cf])
        if np.any(acc(np.zeros_like(f)) != 1):
            raise ValueError('zero barrier acceptance control')
    so, sc = ae[opening].sum(), ae[closing].sum()
    row = [ok.sum(), ae[ok].sum(), opening.sum(), closing.sum(), so, sc, ae[opening]@change[opening],
           -ae[closing]@change[closing], far.sum(), 0, np.count_nonzero(change[ok]), 0]
    # Positive-f exact strata only; four labels occupy adjacent columns for standardization.
    use = ok & (old < .5); sube = np.maximum(e[use], 0); subf = f[use]
    if np.any(sube >= B) or np.any(subf >= B):
        raise ValueError('integer barrier exceeds storage bounds')
    label = g['lut'][words[g['tile'][c[use]]]]
    key = 4*(B*sube.astype(np.int32)+subf)+label
    keys, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv); flux = np.bincount(inv, weights=(ae*opening)[use])
    return row, keys, counts, flux, error


def verify_barriers(x, o, m, g, c, initial, expected):
    """Independent scalar f-jump sum over 12 owners, with e's source substitutions.

    Each array is indexed by edge anchor (a different hypothetical exchange at each
    r), never by a common modified physical lattice. Repeat for the reverse e too.
    """
    a = np.arange(N); b = g['plus'][:, 2]; d = g['plus'][c, 2]; nb = g['neighbors']
    sites = [c, d] + [nb[c, k] for k in (0, 1, 2, 3, 5)] + [nb[d, k] for k in (0, 1, 2, 3, 4)]
    # nb column 4 is +x, column 5 -x. Both unlike and equal edges included.
    for after in (False, True):
        total = np.zeros(N, dtype=np.int16); u = (o[b]-o[a])*int(after); v = o[d]-o[c]
        for j, owner in enumerate(sites):
            na = (nb[owner] == a[:, None]).any(1); nb_ = (nb[owner] == b[:, None]).any(1)
            coord = m[owner] + u*(na.astype(np.int16)-nb_)
            typ = np.where((owner == a) & after, x[b], np.where((owner == b) & after, x[a], x[owner]))
            target = x[d] if j == 0 else x[c] if j == 1 else typ
            dm = -v if j == 0 else v if j == 1 or j < 7 else -v
            total += energy(target, coord+dm)-energy(typ, coord)
        if not np.array_equal(total, expected if after else initial):
            raise ValueError('twelve-owner substituted neighbor barrier mismatch')


def run(bank, stage, out, release):
    start, stop = SLICES[stage]
    if stage == 'audit' and not release:
        raise ValueError('explicit public audit release required')
    out.mkdir(parents=True, exist_ok=True); path = bank/'b09.npz'
    meta = json.loads(path.with_suffix('.json').read_bytes())
    if (meta['index'], meta['beta'], meta['fit_stop'], meta['dev_stop'], meta['audit_stop']) != (9, BETA, 4096, 4608, 5120):
        raise ValueError('not the registered cold TRAIN bank')
    if sha(path) != meta['bank_sha256']:
        raise ValueError('bank digest mismatch')
    with np.load(CODE, allow_pickle=False) as f:
        g = geometry(f['lut'])
    with ZipFile(path) as z:
        packet = {k: section(z, k, start, stop) for k in ('parent', 'chain_id', 'chain_seed')}
        states, energies = (section(z, k, start, stop) for k in ('species', 'energy'))
    if states.shape != (stop-start, 20, 20, 20) or states.dtype != np.uint8:
        raise ValueError('incorrect bank dimensions or species dtype')
    this = set(zip(packet['chain_id'].tolist(), packet['chain_seed'].tolist()))
    if len(this) != stop-start:
        raise ValueError('duplicate lineage identity')
    for prev in list(SLICES)[:list(SLICES).index(stage)]:
        with np.load(out/f'19-{prev}-parents.npz') as f:
            if np.intersect1d(f['parent'], packet['parent']).size or this.intersection(zip(f['chain_id'].tolist(), f['chain_seed'].tolist())):
                raise ValueError('fit, development, audit identity overlap')
    n = stop-start; fields = np.empty((n, 7)); dmat = np.empty((n, 7, 7)); gate = np.empty((n, len(COLUMNS)))
    key, counts, flux, ptr = [], [], [], [0]; errs = np.zeros(2); then = monotonic(); checks = 0
    for j, (state, en) in enumerate(zip(states, energies)):
        x = state.ravel().astype(np.int16); o = (x != 0).astype(np.int16); m = nb_sum(o.reshape(state.shape)).ravel()
        if energy(x, m).sum() != en or np.count_nonzero(x == 1) != 2400 or np.count_nonzero(o) != 6000:
            raise ValueError('native energy/inventory mismatch')
        de = edge_cost(state); words = x[g['tile_site']]@POW.astype(np.int16)
        check = j % 256 == 0 or j == n-1; checks += int(check)
        fields[j], dmat[j], i1 = dirichlet(x, o, m, de.ravel(), acc(de.ravel()), words, g, check)
        gate[j], k, nn, oo, i2 = gates(x, o, m, de, words, g, check)
        key.append(k); counts.append(nn); flux.append(oo); ptr.append(ptr[-1]+len(k)); errs = np.maximum(errs, [i1, i2])
    compute = monotonic()-then
    np.savez_compressed(out/f'19-{stage}-parents.npz', **packet, F=fields, D=dmat)
    data = dict(indptr=np.asarray(ptr), keys=np.concatenate(key), counts=np.concatenate(counts), flux=np.concatenate(flux),
                ncols=4*B*B, columns=np.asarray(COLUMNS), gate=gate)
    np.savez_compressed(out/f'20-{stage}-parents.npz', **packet, **data)
    if stage == 'fit':
        fit_locks(out, fields, dmat, data, B)
    timing = dict(analytic_seconds=compute, seconds_per_parent=compute/n, owner_crosscheck_parents=checks,
                  max_absolute_hot_density_drift_error=errs[0], max_local_detailed_balance_error=errs[1],
                  beta_zero_crossing_count=0, far_shared_Hamiltonian_owners=0, near_shared_Hamiltonian_owners=4)
    lineage = dict(bank=meta, interval=[start, stop], parent_row_sha256=hashlib.sha256(packet['parent'].astype('<i8').tobytes()).hexdigest(),
                   prior_scope='fixed historical LUT as supplied observable; independent fit/development/audit within reused TRAIN only',
                   codec_sha256=sha(CODE), design_sha256=sha(out/'19-20-design.json'), audit_release=release,
                   duplicate_parent_or_chain_keys=0, prior_fold_overlaps=0, physical_states_generated=0, physical_attempts_executed=0)
    report(out, stage, fields, dmat, data, lineage, timing)
    print(stage, n, timing, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('bank', type=Path)
    p.add_argument('stage', choices=SLICES); p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--audit-release', help='Public lock commit authorized by supervisor; audit only.')
    args = p.parse_args(); run(args.bank, args.stage, args.out, args.audit_release)
