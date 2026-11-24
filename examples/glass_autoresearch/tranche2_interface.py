"""Item 23: independent interior exchange from crossing owner terms, not hybrid boxes."""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2] / 'results/l20-autoresearch'
OUT = ROOT / 'tranche2'
DESIGN = OUT / '23-design.json'
MAPS = ROOT / 'faces/optimized.npz'
ARMS = ('inventory', 'optimized', 'affine')
GAINS = ('inventory_minus_optimized', 'inventory_minus_affine', 'affine_minus_optimized')
TILE = np.array(list(product(range(2), repeat=3)))
POW = 3**np.arange(8)
ANCHORS = np.indices((20,)*3).reshape(3, -1).T
DIR = np.vstack((np.eye(3, dtype=int), -np.eye(3, dtype=int)))
IDEAL = np.array([0, 3, 5], dtype=np.int16)
K = 4*81
MARGIN = 0.05
WORDS = np.arange(3**8)[:, None] // POW % 3
INVENTORY = 9*(WORDS == 1).sum(1) + (WORDS == 2).sum(1)


def index(offset):
    return np.ravel_multi_index(((ANCHORS + offset[..., None, :]) % 20).transpose(-1, *range(offset.ndim)), (20,)*3)


CENTER = index(TILE)
OUTSIDE = index(TILE[:, None, :] + (2*TILE[:, None, :]-1)*np.eye(3, dtype=int))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def mappings():
    if sha(MAPS) != 'ec838a3eb8e3b033ff980d38b91176404f81417969284feebff407ca76dc96c1':
        raise ValueError('published encoder checksum mismatch')
    with np.load(MAPS, allow_pickle=False) as z:
        labels = np.stack((np.zeros(len(WORDS), int), z['discrete_lut'], z['affine_lut']))
    if np.any((labels < 0) | (labels >= 4)):
        raise ValueError('invalid two-bit label')
    return 4*INVENTORY + labels


def rows(base, idx, start, stop):
    """Contiguous registered TRAIN rows only; no materialization of held-back boxes."""
    path = base / f'b{idx:02}.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    if idx not in (0, 9) or meta['split'] != 'native TRAIN order only':
        raise ValueError('not a registered archive')
    def member(z, key):
        with z.open(key + '.npy') as f:
            if np.lib.format.read_magic(f) != (1, 0):
                raise ValueError('unexpected header')
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            if fortran or shape[0] != meta['rows'] or not 0 <= start < stop <= shape[0]:
                raise ValueError('unexpected bank layout')
            step = int(np.prod(shape[1:]))*dtype.itemsize
            f.seek(f.tell()+start*step)
            return np.frombuffer(f.read((stop-start)*step), dtype=dtype).reshape(stop-start, *shape[1:])
    with ZipFile(path) as z:
        d = {k:member(z, k) for k in ('species', 'parent', 'chain_seed', 'chain_id', 'energy')}
    if np.unique(np.stack([d['chain_seed'], d['chain_id']], axis=1), axis=0).shape[0] != stop-start:
        raise ValueError('lineage duplication')
    return d, meta


def word_index(s):
    return POW @ s.ravel()[CENTER]


def support(s, tables):
    # No energy, work, or exterior examined during calibration.
    possible = np.stack([np.bincount(a, minlength=K) != 0 for a in tables])
    minimum = np.full((3, K), np.iinfo(np.int64).max)
    for half in np.split(s, len(s)//16):
        words = np.zeros(len(WORDS), np.int64)
        for x in half:
            words += np.bincount(word_index(x), minlength=len(WORDS))
        for i, a in enumerate(tables):
            minimum[i] = np.minimum(minimum[i], np.bincount(a, weights=words, minlength=K).astype(np.int64))
    keep = np.any(possible, axis=0).reshape(-1, 4).any(1) & np.all((~possible | (minimum >= 4)).reshape(3, -1, 4), axis=(0, 2))
    return {'inventory_key':'9*nA+nB', 'inventories':np.flatnonzero(keep).tolist(),
            'possible_cells':possible.astype(int).tolist(), 'min_count_each_cell_over_16_halves':minimum.tolist()}


def features(s, energy):
    o = (s != 0).astype(np.int16)
    n = sum(np.roll(o, shift, axis) for axis in range(3) for shift in (-1, 1))
    if (o*(n-IDEAL[s])**2).sum() != energy:
        raise ValueError('owner Hamiltonian differs from native box energy')
    wi = word_index(s)
    own = o.ravel()[CENTER]
    eo = o.ravel()[OUTSIDE].sum(1, dtype=np.int16)
    mu = (n-IDEAL[s]).ravel()
    # The outside owner's neighbor count excludes the original corner i.
    halo = (o.ravel()*(2*mu+1))[OUTSIDE].sum(1, dtype=np.int16) - 2*own*eo
    u = np.concatenate((own, 2*own*(mu[CENTER]-eo)))
    x = np.concatenate((eo**2+halo, eo))
    return wi, u, x


def half_moments(s, energies, tables):
    # Accumulate sufficient statistics by word ONCE, then collapse to each arm's cells.
    # Float64 bincount sums are exact integers here; integer cross-products below.
    stats = np.zeros((34, len(WORDS)), dtype=np.int64)
    for box, energy in zip(s, energies):
        w, u, x = features(box, energy)
        stats[0] += np.bincount(w, minlength=len(WORDS))
        for j, r in enumerate((*u, *x, (u*x).sum(0))):
            stats[j+1] += np.bincount(w, weights=r, minlength=len(WORDS)).astype(np.int64)
    return np.array([[np.bincount(a, weights=r, minlength=K) for r in stats] for a in tables], dtype=np.int64)


def cross(m1, m2, cells, inventories):
    n1, n2 = m1[:, 0], m2[:, 0]
    if np.any(n1.sum(1) != 16*8000) or np.any(n2.sum(1) != 16*8000):
        raise ArithmeticError('anchor count')
    pairs = n1*n2
    bad = np.any((cells & (pairs == 0)).reshape(3, 81, 4), axis=(0, 2))
    ok = np.isin(np.arange(81), inventories) & ~bad
    keep = np.repeat(ok, 4)
    # U1.X2 + U2.X1 - U1.X1 - U2.X2 over every ordered cross-half opportunity.
    num = ((m1[:, 1:17]*m2[:, 17:33] + m2[:, 1:17]*m1[:, 17:33]).sum(1)
           -m1[:, 33]*n2 -m2[:, 33]*n1)
    mean = np.divide(num, pairs, out=np.zeros(num.shape, float), where=pairs != 0)
    freq = (n1+n2)*keep
    denom = freq.sum(1)
    if np.any(denom != denom[0]) or denom[0] == 0:
        raise ArithmeticError('no common inventory support')
    return ((freq*mean).sum(1)/denom, float(denom[0]/(32*8000)), (pairs*keep).sum(1),
            ok, np.stack([n1, n2]), num, pairs)


def local_parts(box, anchor):
    """Independent, literal list of 32 owner neighborhoods; no exchanged state built."""
    inside = {tuple(v) for v in TILE}
    halo = sorted({tuple(v+d) for v in TILE for d in DIR} - inside)
    def fetch(v):
        return box[tuple(((anchor+v) % 20).T)]
    u = fetch(TILE)
    inn = np.array([np.count_nonzero(fetch(np.array([v+d for d in DIR if tuple(v+d) in inside]))) for v in TILE])
    outer = np.array([np.count_nonzero(fetch(np.array([v+d for d in DIR if tuple(v+d) not in inside]))) for v in TILE])
    halo = np.array(halo)
    ext = fetch(halo)
    rem = np.array([np.count_nonzero(fetch(np.array([v+d for d in DIR if tuple(v+d) not in inside]))) for v in halo])
    adj = np.abs(halo[:, None]-TILE).sum(2) == 1
    if len(halo) != 24 or not np.all(adj.sum(1) == 1):
        raise ArithmeticError('wrong boundary geometry')
    return (u, inn), (outer, ext, rem, adj.argmax(1))


def owner_energy(a, b):
    u, inn = a
    outer, ext, rem, near = b
    return int((((inn+outer-IDEAL[u])**2)*(u != 0)).sum()
               + (((rem+(u[near] != 0)-IDEAL[ext])**2)*(ext != 0)).sum())


def control(s, energies, block):
    w1, u1, x1 = features(s[0], energies[0]); w2, u2, x2 = features(s[16], energies[16])
    rng = np.random.default_rng([23, block]); total = 0; nonzero = 0
    for i in rng.choice(8000, 32, replace=False):
        candidates = np.flatnonzero(INVENTORY[w2] == INVENTORY[w1[i]])
        if not len(candidates):
            continue
        j = rng.choice(candidates)
        a, A = local_parts(s[0], ANCHORS[i]); b, B = local_parts(s[16], ANCHORS[j])
        cross_work = int((u1[:, i]-u2[:, j]) @ (x2[:, j]-x1[:, i]))
        original = owner_energy(a, A) + owner_energy(b, B)
        exchanged = owner_energy(a, B) + owner_energy(b, A)
        if exchanged-original != cross_work or original-exchanged != -cross_work:
            raise ArithmeticError('crossing or re-exchange identity')
        total += 1; nonzero += int(cross_work != 0)
    return [total, nonzero]


def toy_identity():
    # Nonseparable Hamiltonian with pure-interior and pure-exterior offsets.
    h = np.array([[1., 0., 2.], [3., -1., 4.], [2., 3., -1.]])
    out = []
    for beta in (0., 1.3):
        for sub in ([0, 1, 2], [0, 2]):
            e = h[sub]; p = np.exp(-beta*e); p /= p.sum()
            r = p.sum(0)[None, :]*p.sum(1)[:, None]
            w = sum(p[i,j]*p[k,l]*(e[i,l]+e[k,j]-e[i,j]-e[k,l]) for i,j in np.ndindex(p.shape) for k,l in np.ndindex(p.shape))
            kl = float((p*np.log(p/r)).sum()+(r*np.log(r/p)).sum())
            if abs(beta*w - 2*kl) > 1e-11 or (not beta and abs(w) > 1e-12):
                raise ArithmeticError('conditional symmetrized KL identity')
            out.append(dict(beta=beta, interior_rows=sub, W=float(w), jeffreys=kl))
    return out


def intv(y, alpha):
    se = y.std(0, ddof=1)/np.sqrt(len(y)); mean = y.mean(0)
    delta = t.ppf(1-alpha/2, len(y)-1)*se
    return np.stack((mean, mean-delta, mean+delta, se), axis=-1).tolist()


def evaluate(base, idx, start, stop, fit, stage, tables):
    d, meta = rows(base, idx, start, stop)
    measured = []; controls = []
    for i in range(0, len(d['parent']), 32):
        s, energies = d['species'][i:i+32], d['energy'][i:i+32]
        m1 = half_moments(s[:16], energies[:16], tables); m2 = half_moments(s[16:], energies[16:], tables)
        measured.append(cross(m1, m2, np.asarray(fit['possible_cells'], dtype=bool), fit['inventories']))
        controls.append(control(s, energies, i//32))
    work, mass, denoms, inv, count, numerator, pairs = map(np.asarray, zip(*measured))
    gains = work[:, [0, 0, 2]] - work[:, [1, 2, 1]]
    margincenter = gains - MARGIN*work[:, :1]
    evidence = OUT / f'23-{stage}-b{idx:02}-blocks.npz'
    if evidence.exists():
        raise FileExistsError(evidence)
    np.savez_compressed(evidence, parent=d['parent'].reshape(-1,32), chain_seed=d['chain_seed'].reshape(-1,32),
        chain_id=d['chain_id'].reshape(-1,32), work=work, mass=mass, common_inventories=inv,
        retained_pair_counts=denoms, one_parent_counts_by_half=count, work_numerators=numerator,
        all_pair_counts=pairs, controls=controls)
    ci = np.array(intv(margincenter, .05/3)); gain = np.array(intv(gains, .05/3))
    adequate = bool(np.all(mass >= .9))
    def verdict(interval):
        if not adequate: return 'inadequate support'
        if interval[1] > 0: return 'material reduction supported'
        if interval[2] < 0: return 'material reduction excluded'
        return 'unresolved'
    audit_mde = (t.ppf(1-.05/6, 15)+.842)*margincenter.std(0, ddof=1)/4
    return dict(phase=stage, idx=idx, beta=meta['beta'], staged_bank_sha256=meta['bank_sha256'],
        native_source_sha256=meta['source_sha256'], native_train_positions=[start,stop], block_size=32,
        blocks=len(work), anchors_per_parent=8000, arms=ARMS, evidence_file=evidence.name, evidence_sha256=sha(evidence),
        interval_columns=['mean','lower','upper','block_standard_error'],
        independent_owner_checks_and_nonzero_W=np.sum(controls,0).tolist(), toy_KL_identity=toy_identity(),
        native_energy_checked_parents=len(d['parent']), min_mean_max_common_mass=[float(mass.min()),float(mass.mean()),float(mass.max())],
        additional_missing_inventories_per_block=[np.setdiff1d(fit['inventories'], np.flatnonzero(row)).tolist() for row in inv],
        matched_cross_pairs_per_arm=denoms.sum(0).tolist(), work_CI95=intv(work,.05),
        work_CI_three_arm_simultaneous=intv(work,.05/3), gain_order=GAINS,
        gain_CI_simultaneous=gain.tolist(), gain_fraction_of_inventory_work=(gains.mean(0)/work[:,0].mean()).tolist() if idx else None,
        cold_material_margin=MARGIN*float(work[:,0].mean()) if idx else None,
        gain_minus_margin_CI_simultaneous=ci.tolist() if idx else None,
        significant_signed_direction=[('positive' if q[1]>0 else 'negative' if q[2]<0 else 'unresolved') for q in gain] if idx else None,
        disposition=[verdict(q) for q in ci] if idx else None,
        projected_80pct_detectable_increment_beyond_margin_for_512_parents=audit_mde.tolist() if idx and stage=='dev' else None,
        projected_80pct_detectable_fraction_of_W0=(audit_mde/work[:,0].mean()).tolist() if idx and stage=='dev' else None,
        population_jeffreys_by_arm_at_cold_estimate=(meta['beta']*.5*work.mean(0)).tolist() if idx else None,
        uncertainty='Disjoint blocks of 32 parents; paired t approximation, not independent tiles or pair opportunities.')


def plot(report, hot, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), layout='constrained')
    def intervals(ax, data, labels, title):
        y = np.arange(len(data)); m,lo,hi,_ = np.array(data).T
        ax.errorbar(m, y, xerr=[m-lo,hi-m], fmt='o', color='#175777', capsize=3)
        ax.set_yticks(y, labels); ax.set_ylim(len(data)-.5, -.5); ax.set_title(title,fontsize=10)
        ax.axvline(0, color='.55', ls=':'); ax.set_xlabel('Energy per exchange')
    intervals(axes[0], report['work_CI95'], ARMS, 'Cold residual work (95% CI)')
    intervals(axes[1], report['gain_minus_margin_CI_simultaneous'], ['Inv − opt','Inv − affine','Affine − opt'], 'Cold gain minus 5% margin¹')
    intervals(axes[2], hot['work_CI_three_arm_simultaneous'], ARMS, 'β = 0 development null¹')
    fig.suptitle('Unrelaxed inventory-matched interface exchange • ¹ simultaneous intervals', fontsize=11)
    fig.savefig(out, dpi=170); plt.close(fig)


def run(a):
    tables = mappings(); fitpath = OUT/'23-support.json'; devpath = OUT/'23-dev.json'
    meta = dict(design_sha256=sha(DESIGN), script_sha256=sha(Path(__file__)), maps_sha256=sha(MAPS))
    if a.phase == 'dev':
        if fitpath.exists() or devpath.exists(): raise FileExistsError('development already calibrated')
        fit = {str(idx):support(rows(a.bank, idx, 0, 256)[0]['species'], tables) for idx in (0, 9)}
        save_json(fitpath, dict(**meta, fit=fit))
        hot = evaluate(a.bank, 0, 256, 512, fit['0'], 'dev', tables)
        cold = evaluate(a.bank, 9, 4096, 4608, fit['9'], 'dev', tables)
        save_json(devpath, dict(**meta, support_sha256=sha(fitpath), hot=hot, cold=cold))
        print(json.dumps({'hot_work':hot['work_CI_three_arm_simultaneous'],'cold_work':cold['work_CI95'],
              'cold_gain':cold['gain_minus_margin_CI_simultaneous'],'support':[hot['min_mean_max_common_mass'],cold['min_mean_max_common_mass']],
              'audit_projected_MDE':cold['projected_80pct_detectable_fraction_of_W0'],'dev_sha256':sha(devpath)}, indent=2), flush=True)
    else:
        if not a.released_dev_sha256 or sha(devpath) != a.released_dev_sha256:
            raise ValueError('audit requires released fixed development receipt checksum')
        dev = json.loads(devpath.read_text()); fit = json.loads(fitpath.read_text())
        if any(dev[k] != v or fit[k] != v for k,v in meta.items()) or dev['support_sha256'] != sha(fitpath):
            raise ValueError('changed registration, code, or frequency calibration')
        out = OUT/'23-audit.json'
        if out.exists(): raise FileExistsError('single audit already scored')
        cold = evaluate(a.bank, 9, 4608, 5120, fit['fit']['9'], 'audit', tables)
        save_json(out, dict(**meta, issued_dev_sha256=sha(devpath), support_sha256=sha(fitpath), cold=cold))
        if 'material reduction supported' in cold['disposition']:
            plot(cold, dev['hot'], OUT/'23-interface.png')
        print(json.dumps({'audit':cold['disposition'],'work':cold['work_CI95'],'gain':cold['gain_minus_margin_CI_simultaneous']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['dev', 'audit']); parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--released-dev-sha256'); run(parser.parse_args())
