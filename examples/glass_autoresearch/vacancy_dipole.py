"""Explain the selected inward vacancy tripod using actual L20 validation shells only."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from zipfile import ZipFile

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))
from l20_pairs import L, PAIR_SEED, SPLIT, partition

DEFAULT_OUT = EXAMPLES.parent / 'results/l20-autoresearch/vacancy-dipole'
DESIGN = DEFAULT_OUT / 'design.json'
INDICES = (0, 7, 9)
SIGNS = tuple(itertools.product((-1, 1), repeat=3))
AXES = 'xyz'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def selected_member(z, name, ids, row_shape):
    """Forward seeks skip non-selected rows inside the deflate stream, no full ndarray."""
    with z.open(name + '.npy') as f:
        version = np.lib.format.read_magic(f)
        header = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}[version]
        shape, fortran, dtype = header(f)
        if fortran or shape != (sum(SPLIT), *row_shape) or dtype.kind not in 'ui':
            raise ValueError(f'unexpected {name} header: {shape}, {dtype}')
        result = np.empty((len(ids), *row_shape), dtype=dtype)
        row_bytes = math.prod(row_shape) * dtype.itemsize
        start = f.tell()
        for pos in np.argsort(ids):
            address = start + int(ids[pos]) * row_bytes
            if f.tell() != address:
                f.seek(address)
            block = f.read(row_bytes)
            if len(block) != row_bytes:
                raise ValueError(f'truncated {name}')
            result[pos:pos+1] = np.frombuffer(block, dtype=dtype).reshape(1, *row_shape)
    return result


def validation(data, index, pin):
    if index not in INDICES:
        raise ValueError('non-validation temperature forbidden')
    path = data / f'beta-{index:02d}.npz'
    receipt_path = path.with_suffix('.json')
    if sha(receipt_path) != pin['receipt_sha256'] or sha(path) != pin['endpoint_sha256']:
        raise ValueError(f'{path}: release hash mismatch')
    receipt = json.loads(receipt_path.read_text())
    if (receipt['index'] != index or float(receipt['beta']) != pin['beta'] or
            receipt['endpoint_sha256'] != pin['endpoint_sha256']):
        raise ValueError('receipt identity mismatch')
    ids = partition(index)[0][1]
    if len(ids) != SPLIT[1] or len(np.unique(ids)) != len(ids):
        raise ValueError('invalid native validation split')
    with ZipFile(path) as z:
        for name, expected in [('axes', 'chain,z,y,x'), ('beta', pin['beta'])]:
            with z.open(name + '.npy') as f:
                item = np.lib.format.read_array(f, allow_pickle=False).item()
            if (float(item) if name == 'beta' else item) != expected:
                raise ValueError(f'unexpected archive {name}: {item}')
        keys = np.column_stack([selected_member(z, name, ids, ()) for name in ('chain_seed', 'chain_id')]).astype('<u8')
        state = selected_member(z, 'species', ids, (L, L, L))
    if len(np.unique(keys, axis=0)) != len(keys) or state.dtype != np.uint8 or state.max() > 2:
        raise ValueError('malformed selected species/lineage')
    return ids, keys, state


def count_parents(state):
    """One reversible six-bit shell per site; independent direct checks in all frames."""
    hist = np.empty((len(state), 3, 64), dtype=np.uint16)
    checked = 0
    for first in range(0, len(state), 16):
        s = state[first:first+16]
        v = (s == 0).astype(np.int8)
        plus = [np.roll(v, -1, a) for a in (-1, -2, -3)]
        minus = [np.roll(v, 1, a) for a in (-1, -2, -3)]
        k = sum(1-u for u in plus) + sum(1-u for u in minus)
        d = [u-w for u,w in zip(plus, minus)]
        b = sum(u*w for u,w in zip(plus, minus))
        if np.any(sum(di*di for di in d) != 6-k-2*b):
            raise ArithmeticError('opposite-pair identity failed')
        for signs in SIGNS:
            direct = sum((plus if sign > 0 else minus)[a] for a, sign in enumerate(signs))
            if np.any(2*direct != 6-k + sum(sign*di for sign, di in zip(signs, d))):
                raise ArithmeticError(f'nonzero decomposition residual for {signs}')
            checked += direct.size
        mask = np.zeros_like(s)
        for a,(u,w) in enumerate(zip(plus, minus)):
            mask |= u.astype('u1') << (2*a)
            mask |= w.astype('u1') << (2*a+1)
        code = 64*s + mask
        offsets = (192*np.arange(len(s))).reshape(-1,1,1,1)
        hist[first:first+len(s)] = np.bincount((offsets+code).ravel(), minlength=192*len(s)).reshape(-1,3,64)
    if np.any(hist.sum(axis=(1,2)) != L**3):
        raise ArithmeticError('parent volume mismatch')
    return hist, checked


def choose(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def catalog():
    """Evaluate observables per actual mask, S6 orbit and signed-axis O_h orbit."""
    bits = (np.arange(64)[:,None] >> np.arange(6)) & 1
    plus, minus = bits[:,::2], bits[:,1::2]
    n = bits.sum(1)
    d = plus-minus
    q = (d*d).sum(1)
    b = (plus*minus).sum(1)
    if not np.array_equal(n, 2*b+q):
        raise ArithmeticError('invalid catalog')
    # Enumerate the cube frames using neighbor occupancies, not the predicted law.
    tripods = np.stack([np.where(np.array(s) > 0, plus, minus).sum(1) for s in SIGNS], axis=1)
    sign_matrix = np.asarray(SIGNS)
    high_basis = np.column_stack([sign_matrix[:,i]*sign_matrix[:,j] for i,j in itertools.combinations(range(3),2)] + [sign_matrix.prod(1)])
    if (np.any(tripods @ high_basis) or np.any(tripods.sum(1) != 4*n) or
            not np.array_equal(tripods @ sign_matrix, 4*d)):
        raise ArithmeticError('incorrect Walsh decomposition')
    mass = np.stack([(tripods == j).mean(1) for j in range(4)], axis=1)
    pair_fit = np.array([[choose(int(singles), j-int(double)) / 2**int(singles) for j in range(4)]
                         for singles,double in zip(q,b)])
    centered = tripods-n[:,None]/2
    if (not np.array_equal(mass, pair_fit) or not np.array_equal((centered**2).mean(1), q/4) or
            not np.array_equal((centered**4).mean(1), (3*q*q-2*q)/16)):
        raise ArithmeticError('conditional octant law failed')
    opp = b/3
    perp = (n*(n-1)/2-b)/12
    measures = {'opp_double_axes': b, 'single_axes_q': q, 'opposite_VV': opp,
                'perpendicular_VV': perp, 'opposite_minus_perpendicular_VV': opp-perp,
                'opposite_cov_at_k': opp-(n/6)**2, 'perpendicular_cov_at_k': perp-(n/6)**2,
                'centered_T_second': (centered**2).mean(1), 'centered_T_fourth': (centered**4).mean(1)}
    measures.update({f'P_T{j}': mass[:,j] for j in range(4)})
    measures.update({f'P_b{j}': b == j for j in range(4)})
    for i,axis in enumerate(AXES):
        measures[f'd{axis}'] = d[:,i]
        measures[f'd{axis}2'] = d[:,i]**2  # d_i^4=d_i^2 for {-1,0,1}
        for j in range(i):
            prod = d[:,j]*d[:,i]
            measures[f'd{AXES[j]}_d{axis}'] = prod
            measures[f'd{AXES[j]}2_d{axis}2'] = prod**2
    names = list(measures)
    obs = np.column_stack(list(measures.values()))
    perm, cubic = np.empty_like(obs), np.empty_like(obs)
    for count in range(7):
        group = n == count
        perm[group] = obs[group].mean(0)
        pred = [choose(3,j)*choose(3,count-j)/choose(6,count) for j in range(4)]
        if (not np.allclose(mass[group].mean(0), pred, atol=1e-15, rtol=0) or
                not np.isclose(b[group].mean(), count*(count-1)/10) or
                not np.isclose((d[group,0]**2).mean(), count*(6-count)/15)):
            raise ArithmeticError('enumeration does not recover hypergeometric identities')
        for doubles in np.unique(b[group]):
            orbit = group & (b == doubles)
            cubic[orbit] = obs[orbit].mean(0)
    return names, n, obs, perm, cubic


def jackknife(sums, counts):
    """Paired ratio jackknife, numerator columns may already be parent differences."""
    denom = counts.sum()
    total = sums.sum(0)
    mean = total/denom
    if np.count_nonzero(counts) < 8:
        return mean, [None]*len(mean)
    deleted = (total[None,:]-sums)/(denom-counts[:,None])
    se = deleted.std(0, ddof=0) * np.sqrt(len(counts)-1)
    return mean, se


def table(index, beta, hist, tables):
    names, n, obs, perm, cubic = tables
    values = np.column_stack((obs, perm, cubic, obs-perm, obs-cubic))
    strata = [('all', 'all', hist.sum(1), np.ones(64, dtype=bool))]
    strata.extend((c, k, hist[:,c,:], n == 6-k) for c,k in itertools.product(range(3),range(7)))
    rows = []
    for species,k,h,mask in strata:
        selected = h[:,mask].astype(float)
        counts = selected.sum(1)
        total = int(counts.sum())
        parents = int(np.count_nonzero(counts))
        common = dict(index=index, beta=beta, species=species, k=k, sites=total, parents_with_sites=parents,
                      site_weight=total/(len(hist)*L**3))
        groups = ('observed','perm','cubic','obs_minus_perm','obs_minus_cubic')
        if total:
            means, ses = jackknife(selected @ values[mask], counts)
            half_multiplier = student.ppf(.975,len(hist)-1)
        for i,name in enumerate(names):
            r = dict(common, observable=name)
            for j,label in enumerate(groups):
                col = j*len(names)+i
                mu = float(means[col]) if total else None
                se = float(ses[col]) if parents >= 8 else None
                r[label] = mu
                r[label+'_se'] = se
                r[label+'_lo95'] = None if se is None else float(mu-half_multiplier*se)
                r[label+'_hi95'] = None if se is None else float(mu+half_multiplier*se)
            rows.append(r)
    return rows


def figure(rows, path):
    fig, axes = plt.subplots(1,3, figsize=(12.5,4.1), layout='constrained')
    colors = ['#6c6c6c','#cd7c2c','#047b83']
    groups = {i:[r for r in rows if r['index']==i] for i in INDICES}
    for i,col in zip(INDICES,colors):
        rr = groups[i]
        beta = rr[0]['beta']
        for spec, ax in zip((1,2), axes[:2]):
            sub = [r for r in rr if r['species']==spec and r['observable']=='opp_double_axes' and r['parents_with_sites']>=8]
            x = np.array([r['k'] for r in sub], float)+.06*(INDICES.index(i)-1)
            y = np.array([r['obs_minus_perm'] for r in sub]); lo = np.array([r['obs_minus_perm_lo95'] for r in sub])
            ax.errorbar(x,y,yerr=y-lo,fmt='o-',lw=1.1,ms=3,capsize=2,color=col,label=f'β = {beta:.3g}')
        sub = [next(r for r in rr if r['species']=='all' and r['observable']==f'P_T{j}') for j in range(4)]
        y = np.array([r['obs_minus_perm'] for r in sub])*100
        lo = np.array([r['obs_minus_perm_lo95'] for r in sub])*100
        axes[2].errorbar(np.arange(4)+.09*(INDICES.index(i)-1),y,yerr=y-lo,fmt='o-',lw=1.1,ms=4,capsize=2,color=col)
    for ax,species in zip(axes[:2],('A','B')):
        ax.set(title=f'{species} center: antipodal vacancy pairs',xlabel='Occupied neighbors k',xticks=range(7))
    axes[0].set_ylabel('Extra double-vacancy axes per center')
    axes[0].legend(frameon=False,fontsize=9)
    axes[2].set(title='All centers: random octant',xlabel='Tripod vacancy count T',xticks=range(4),ylabel='Excess probability (percentage points)')
    for ax in axes:
        ax.axhline(0,c='.6',lw=.8,zorder=0)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Departure from uniform placement at measured species and coordination\nWhole-parent, paired pointwise 95% intervals',fontsize=13)
    fig.savefig(path,dpi=160); plt.close(fig)


def run(data, out):
    out.mkdir(parents=True,exist_ok=True)
    design = json.loads(DESIGN.read_text())
    tables = catalog()
    arrays, all_rows, sources, pooled = {}, [], {}, {}
    for index in INDICES:
        pin = design['release'][str(index)]
        ids, keys, state = validation(data, index, pin)
        hist, ntripods = count_parents(state)
        del state
        arrays.update({f'row_{index}':ids, f'lineage_{index}':keys, f'counts_{index}':hist})
        rows = table(index,pin['beta'],hist,tables)
        all_rows.extend(rows)
        sources[str(index)] = dict(**pin, parents=len(ids), fold='native validation', partition_seed=PAIR_SEED,
                                  validation_row_sha256=hashlib.sha256(ids.astype('<u8').tobytes()).hexdigest(),
                                  validation_lineage_sha256=hashlib.sha256(keys.tobytes()).hexdigest(),
                                  sites=int(hist.sum()), occupied_masks=int((hist.sum((0,1))>0).sum()),
                                  octant_site_checks=ntripods, max_abs_twice_tripod_residual=0,
                                  max_abs_single_axis_sum_rule_residual=0,
                                  max_abs_Walsh_degree_2_or_3_coefficient=0,
                                  max_abs_cubic_octant_probability_residual=0)
        pooled[str(index)] = {r['observable']:{col:r[col] for col in ('observed','perm','obs_minus_perm','obs_minus_perm_lo95','obs_minus_perm_hi95')}
                              for r in rows if r['species']=='all' and r['observable'] in ('opp_double_axes','single_axes_q','P_T0','P_T1','P_T2','P_T3','opposite_cov_at_k','perpendicular_cov_at_k')}
    hist_path = out/'parent-masks.npz'
    np.savez_compressed(hist_path,**arrays)
    csv_path = out/'responses.csv'
    with csv_path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    fig_path = out/'vacancy-dipole.png'
    figure(all_rows,fig_path)
    summary = dict(design_sha256=sha(DESIGN), code_sha256=sha(Path(__file__)), sources=sources,
                   provenance='Release/receipt authenticated; only native validation rows selected. Direct checks at every site/frame; Walsh and conditional octant-law checks exhaust the entire 64-mask support.',
                   uncertainty=design['uncertainty'], pooled=pooled,
                   interpretation='Conditioning on observed n and b exactly reproduces the octant-averaged count distribution on each shell. This is combinatorics, not independent predictive evidence, and does not establish a symmetry representation for the teacher, a new order, dynamics, or cross-site explanatory sufficiency.',
                   files_sha256={p.name:sha(p) for p in (hist_path,csv_path,fig_path)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f'{len(all_rows)} calibration rows; {sum(x["octant_site_checks"] for x in sources.values())} exact site/octant checks; {out}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='l20-16384-v1 endpoint release')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.data,args.out)
