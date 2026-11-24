"""Archive-only #01 three-shell tie forecast and #02 complete planar homometry.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1, see --help. No simulation.
Native z,y,x; every edge and planar window included without content-based sampling.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, gammaln
from scipy.stats import t
from vacancy_free_energy import log_partition, verify as verify_partition


BETA = {9: 3.05, 8: 2.711111111111, 7: 2.372222222222, 0: 0.0}
N_A = 2400
NEAR = tuple(p for p in product((-1, 0, 1), repeat=3) if sum(i*i for i in p) == 1)
SHELLS = tuple(tuple(p for p in product((-1, 0, 1), repeat=3)
                     if sum(i*i for i in p) == r) for r in (1, 2, 3))
K = np.arange(7)
CAL_EDGES = np.array([.5, .55, .65, .75, .85, .95, 1.])


def sha(path):
    h = sha256()
    with path.open('rb') as src:
        for b in iter(lambda: src.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def ratio(scores, weights):
    scores, weights = np.asarray(scores), np.asarray(weights)
    n = len(weights)
    if not weights.sum():
        return dict(mean=None, se=None, ci95=None, parents=0)
    mean = scores.sum()/weights.sum()
    parents = int(np.count_nonzero(weights))
    if parents < 2:
        return dict(mean=float(mean), se=None, ci95=None, parents=parents)
    resid = (scores - mean*weights)/weights.sum()
    se = np.sqrt(n/(n-1)*np.sum(resid*resid))
    d = t.ppf(.975, n-1)*se
    return dict(mean=float(mean), se=float(se), ci95=[float(mean-d), float(mean+d)], parents=parents)


def plus(a, b):
    return tuple(u+v for u, v in zip(a, b))


def geometry(L):
    grid = np.arange(L**3).reshape((L,)*3)
    axes = tuple(range(3))
    rolls = {}

    def lookup(p):
        if p not in rolls:
            rolls[p] = np.roll(grid, tuple(-u for u in p), axes).ravel().astype('i2')
        return rolls[p]

    edges = []
    for axis in axes:
        left = (0, 0, 0)
        right = tuple(int(j == axis) for j in axes)
        ends = {left, right}
        shells = []
        for offsets in SHELLS:
            a, b = ({plus(e, d) for d in offsets} - ends for e in (left, right))
            # Equal contributions of shared exterior sites cancel from the comparison.
            shells.append(tuple(np.stack([lookup(p) for p in sorted(s)], axis=1)
                                for s in (a-b, b-a)))
        sides = np.array([[p for p in sorted({plus(e,d) for d in NEAR}-ends)]
                          for e in (left,right)])
        sides = [[tuple(p) for p in row] for row in sides]
        assert len(set(sides[0]+sides[1]) | ends) == 12
        ext = np.stack([[lookup(p) for p in row] for row in sides], axis=-1).transpose(1,2,0)
        # (site,side,5 neighbors,5 exterior neighbors of each neighbor)
        base = np.array([[[lookup(z) for z in sorted({plus(p,d) for d in NEAR}-ends)]
                          for p in row] for row in sides]).transpose(3,0,1,2)
        affected = np.sort(np.column_stack([lookup(left), lookup(right), ext.reshape(L**3,10)]), axis=1)
        edges.append((lookup(right), shells, ext, base, affected))
    return edges


def placement_hists(occ, prefix, ids, edge):
    """Both candidate histograms from unaffected sums and exterior bits only.

    Prefix differences sum disjoint flat runs; their retained degrees have closed
    neighborhoods disjoint from BOTH ends. No candidate bit in local lookups.
    """
    _, _, ext, base, affected = edge
    a = affected[ids]
    far = prefix[a[:,0]] + prefix[-1] - prefix[a[:,-1]+1]
    far += (prefix[a[:,1:]]-prefix[a[:,:-1]+1]).sum(axis=1, dtype=far.dtype)
    occupied = occ[ext[ids]]
    degree = occ[base[ids]].sum(axis=-1)
    # For occupied neighbors, the histogram of their degrees with BOTH candidates removed.
    spectra = np.stack([((degree == k) & occupied).sum(axis=-1) for k in range(7)], axis=-1)
    assert np.array_equal(occupied[:,0].sum(1), occupied[:,1].sum(1))
    common = far + spectra.sum(1) + (occupied[:,0].sum(1)[:,None] == K)
    promote = -spectra.copy()
    promote[:,:,1:] += spectra[:,:,:-1]
    h = common[:,None,:] + promote
    assert np.all(h >= 0) and np.all(h.sum(axis=-1) == 6000)
    assert np.array_equal(h[:,0]@K, h[:,1]@K)
    return h, spectra


METRICS = ('eligible','unlike','informative','changed_hist','gain','favored_residual',
           'favored_observed','favored_forecast','expected_gain','shell1','shell12','direct_check')


def one_edges(species, edges, beta, control=False):
    # Discard colors before constructing eligibility or likelihood; target revealed only in score.
    occ = (species != 0).ravel()
    cube = occ.reshape(species.shape)
    deg = sum(np.roll(cube, sign, axis=ax).astype('i1') for ax in range(3) for sign in (-1,1)).ravel()
    full = ((deg[:,None] == K) & occ[:,None]).astype('i2')
    prefix = np.zeros((len(occ)+1,7), 'i2')
    np.cumsum(full, axis=0, dtype='i2', out=prefix[1:])
    hrows, patterns, ys = [], [], []
    rec = Counter()
    cal = np.zeros((len(CAL_EDGES)-1,3))
    for edge in edges:
        right, shells, ext, base, affected = edge
        keep = occ != occ[right]
        rec['unlike'] += int(keep.sum())
        for r, (a,b) in enumerate(shells):
            keep &= occ[a].sum(axis=1) == occ[b].sum(axis=1)
            if r < 2:
                rec[('shell1','shell12')[r]] += int(keep.sum())
        ids = np.flatnonzero(keep)
        if not len(ids):
            continue
        h, spectra = placement_hists(occ, prefix, ids, edge)
        if control:
            # The directly observed census is a check only; independently verify the
            # untouched exterior and both fillings' local histograms on one edge/axis.
            j = int(ids[0]); e = np.array([j,right[j]])
            keepfar = np.ones(len(occ),bool); keepfar[affected[j]] = False
            directfar = full[keepfar].sum(0)
            original = occ[e]
            for side in (0,1):
                bins = directfar.copy()
                for s in (0,1):
                    for k, site in enumerate(ext[j,s]):
                        if occ[site]:
                            bins[int(occ[base[j,s,k]].sum()) + int(s==side)] += 1
                bins[int(occ[ext[j,side]].sum())] += 1
                np.testing.assert_array_equal(bins,h[0,side])
            np.testing.assert_array_equal(prefix[-1],h[0,int(np.flatnonzero(original)[0])])
            rec['direct_check'] += 1
        ys.append(occ[ids]); hrows.append(h); patterns.append(spectra)
    if not hrows:
        return np.array([rec[key] for key in METRICS]), cal, Counter()
    h = np.concatenate(hrows); y = np.concatenate(ys); pat = np.concatenate(patterns)
    z = log_partition(h.reshape(-1,7),beta,N_A).reshape(-1,2)
    odds = z[:,0]-z[:,1]
    if beta == 0:
        assert np.all(odds == 0)
    conf = expit(np.abs(odds)); favored_obs = np.where(odds==0,.5,(odds>0)==y)
    signed_odds = np.where(y,odds,-odds)
    gain = np.log(2.) - np.logaddexp(0., -signed_odds)
    informative = conf >= .55
    rec.update(eligible=len(y), informative=int(informative.sum()),
               changed_hist=int(np.any(h[:,0] != h[:,1],axis=1).sum()),
               gain=gain.sum(), favored_residual=(favored_obs-conf).sum(),
               favored_observed=favored_obs.sum(), favored_forecast=conf.sum(),
               expected_gain=(np.log(2.)-np.logaddexp(0.,-np.abs(odds))
                              -expit(-np.abs(odds))*np.abs(odds)).sum())
    ibin = np.searchsorted(CAL_EDGES[1:-1],conf,side='right')
    for j, w in enumerate((None,conf,favored_obs)):
        cal[:,j] = np.bincount(ibin,weights=w,minlength=len(cal))
    # Count each unordered degree-spectrum pair with the same orientation rule;
    # record model's favored spectrum and actual agreement. No score-selected patterns.
    # Digits in base 6 encode counts <=5. Lexicographic comparison is arbitrary but fixed.
    code = pat[informative] @ (6**K)
    lo = code.min(axis=1); hi = code.max(axis=1)
    lowfav = (code[:,0]==lo) == (odds[informative]>0)
    lowocc = (code[:,0]==lo) == y[informative]
    pcount = Counter(zip(lo.tolist(),hi.tolist(),lowfav.tolist(),lowocc.tolist()))
    return np.array([rec[key] for key in METRICS]), cal, pcount


def edge_receipt(bank, index, out, label):
    data = np.load(bank,allow_pickle=False); ids = data['parent']; boxes = data['species']
    edges = geometry(20); metrics, cals = [], []; patterns = Counter()
    if len(np.unique(data['chain_id']))!=len(ids) or len(np.unique(ids))!=len(ids):
        raise ValueError('expected one row per distinct parent and chain')
    for i, box in enumerate(boxes):
        row, cal, pat = one_edges(box, edges, BETA[index], i==0)
        metrics.append(row); cals.append(cal); patterns.update(pat)
    metrics = np.stack(metrics); cals = np.stack(cals); m = dict(zip(METRICS, metrics.T))
    nf = out/f'01-{label}-parents.npz'
    if nf.exists():
        raise FileExistsError(nf)
    np.savez_compressed(nf, parent=ids, chain_id=data['chain_id'], chain_seed=data['chain_seed'],
                        fields=np.array(METRICS), metrics=metrics, calibration_count_pred_obs=cals)
    paircount = Counter()
    for (a,b,_,_), n in patterns.items():
        paircount[a,b] += n
    pattern = None
    if paircount:
        a,b = min(paircount,key=lambda k:(-paircount[k], k)); denom = paircount[a,b]
        def decode(c):
            return [(c//6**k)%6 for k in range(7)]
        pattern = dict(spectra_k0_to_6=[decode(a),decode(b)],edges=denom,
                       first_favored=sum(n for (x,z,f,o),n in patterns.items() if (x,z)==(a,b) and f),
                       first_occupied=sum(n for (x,z,f,o),n in patterns.items() if (x,z)==(a,b) and o),
                       interpretation='Each occupied neighbor counted in that spectrum is promoted from k to k+1 by choosing that end. Other end stays unpromoted; occupied endpoint degree tied.')
    n = len(ids); total = np.full(n,24000); w = m['eligible']; effect = ratio(m['gain'],w)
    calib = ratio(m['favored_residual'],w)
    informative_parents = int((m['informative']>0).sum())
    gate = (effect['ci95'] is not None and effect['ci95'][0] >= .01 and calib['ci95'][0]>=-.03
            and calib['ci95'][1]<=.03 and informative_parents>=128)
    r = dict(item=1,index=index,beta=BETA[index],cohort=label,parents=n,edges_per_parent=24000,
             opportunities=int(total.sum()), eligible=int(w.sum()), eligible_parents=int((w>0).sum()),
             unlike_and_cumulative_equal_shells=[int(m[k].sum()) for k in ('unlike','shell1','shell12','eligible')],
             eligible_frequency=ratio(w,total), informative_parents=informative_parents,
             informative_count=int(m['informative'].sum()), informative_fraction=ratio(m['informative'],w),
             changed_hist_fraction=ratio(m['changed_hist'],w),gain_nat_per_eligible=effect,
             expected_gain_under_forecast=ratio(m['expected_gain'],w), favored_calibration_residual=calib,
             favored_forecast=ratio(m['favored_forecast'],w), favored_observed=ratio(m['favored_observed'],w),
             reliability=[dict(pmax_range=[float(CAL_EDGES[i]),float(CAL_EDGES[i+1])],count=int(c[:,0].sum()),
                               forecast=ratio(c[:,1],c[:,0]), observation=ratio(c[:,2],c[:,0]))
                          for i,c in enumerate(cals.transpose(1,0,2))],
             modal_informative_unordered_pattern=pattern, gate_except_separate_beta0_check=bool(gate),
             beta_zero_all_weights_tie_assertion=(index==0),
             whole_parent_approx_80pct_power_mde_at_5pct=(2.80*effect['se'] if effect['se'] is not None else None),
             checks=dict(explicit_exterior_and_both_histograms=int(m['direct_check'].sum()),
                         canonical_projection=verify_partition()),
             method='See 01-02-development-design.json. Positive-axis enumeration = both random signs by exact conjugacy of the two fillings. Unaffected degree sums exclude both end neighborhoods; local histogram terms read exterior bits only. Fixed global N_A=2400. Finite released parents need not be at equilibrium.',
             source_sha256=sha(bank), per_parent_sha256=sha(nf),code_sha256=sha(Path(__file__)))
    dump(out/f'01-{label}.json',r)
    print(label,'eligible',int(w.sum()),'gain',effect,'calibration',calib,'informative parents',informative_parents,flush=True)
    return r


def homometry():
    """All 4x4 binary masks, little-endian word at row*4+col.

    Every possible oriented displacement and its negative is kept. Register
    an oriented word to the least *full table* across its D4 orbit; only then
    order words lexically. Minimizing just the word and comparing its vector
    table would incorrectly miss differently oriented homometric shapes.
    """
    words = np.arange(65536, dtype='u4')
    bits = ((words[:,None] >> np.arange(16,dtype='u4')) & 1).astype('u1')
    pop = bits.sum(axis=1,dtype='u1')
    ij = np.indices((4,4)).reshape(2,16)
    low = np.stack([np.where(bits, coord, 4).min(axis=1) for coord in ij],axis=1)
    high = np.stack([np.where(bits, coord, -1).max(axis=1) for coord in ij],axis=1)
    low[0] = 0
    dims = high-low+1
    tight = words >> (4*low[:,0] + low[:,1]).astype('u4')
    transforms = []
    tile = np.arange(16).reshape(4,4)
    for refl in (tile,tile[:,::-1]):
        for rot in range(4):
            digit = np.rot90(refl,rot).ravel()
            transforms.append(digit)
    oriented = np.stack([bits[:,m] @ (1<<np.arange(16,dtype='u4')) for m in transforms],axis=1)
    orbit = tight[oriented]
    shape_id = orbit.min(axis=1)
    disp = np.zeros((65536,49),'u1')
    for r,(di,dj) in enumerate(product(range(-3,4), repeat=2)):
        legal = np.all((ij.T+[di,dj] >= 0) & (ij.T+[di,dj] < 4),axis=1)
        shift = 4*di+dj
        aligned = words >> shift if shift >= 0 else (words << -shift)
        mask = sum(1<<int(i) for i in np.flatnonzero(legal))
        disp[:,r] = pop[words & aligned & mask]
    assert np.array_equal(disp,disp[:,::-1])
    assert np.array_equal(disp[:,24],pop) and np.array_equal(disp.sum(1),pop.astype(int)**2)
    # A full common signed table and a shared bounding box are required.
    groups = defaultdict(list); reg = {}
    for a in np.unique(shape_id):
        def key(w):
            return (*dims[w],disp[w].tobytes(),int(w))
        rep = min(orbit[a],key=key)
        groups[key(rep)[:-1]].append(int(a)); reg[int(a)]=int(rep)
    pairs = sorted(pair for g in groups.values() for pair in combinations(sorted(g),2))
    # Verify each witness by an independent ordered-differences Counter, not
    # only by the packed shifts used to build the finite table.
    def difference_counter(w):
        coords = ij[:,bits[w].astype(bool)].T
        return Counter(tuple(int(z) for z in a-b) for a in coords for b in coords)
    for a,b in pairs:
        u,v = reg[a],reg[b]
        assert not np.any(orbit[u] == v)
        assert difference_counter(u) == difference_counter(v)
        assert np.array_equal(dims[u],dims[v])
    return dict(shape=shape_id, reg=reg, pairs=pairs, dims=dims, low=low, bits=bits,
                disp=disp, orbit=orbit, oriented=oriented, transforms=transforms,
                orbit_shapes=len(reg), signature_classes=len(groups))


def plane_word(box,axis):
    vac = np.moveaxis(box==0,axis,0).astype('u2')
    w = np.zeros(vac.shape,'u2')
    for a,b in product(range(4),repeat=2):
        w |= np.roll(vac,(-a,-b),axis=(1,2)) << (4*a+b)
    return w


def hom_support(boxes, tab):
    shape = tab['shape']
    ids = sorted({w for pair in tab['pairs'] for w in pair})
    compact = np.full(65536,len(ids),'i4'); compact[ids] = np.arange(len(ids))
    lookup = compact[shape]
    counts = np.zeros((len(boxes),len(ids)),'i4')
    for p,box in enumerate(boxes):
        for axis in range(3):
            counts[p] += np.bincount(lookup[plane_word(box,axis)].ravel(),minlength=len(ids)+1)[:-1]
    return ids, counts


def external(boxes, tab, pair, luts):
    """One target. Register lookup coordinates using the D4 transform that
    gives the common diffraction frame; read only actual observed sites."""
    n=len(boxes); hits=np.zeros((n,2),'i4'); bonds=np.zeros((n,2),'i4')
    labels=np.zeros((n,2,2,4),'i4')
    record = {}
    for j,a in enumerate(pair):
        for word in np.flatnonzero(tab['shape']==a):
            u = tab['reg'][a]
            tr = int(np.flatnonzero(tab['orbit'][word] == u)[0])
            tword = tab['oriented'][word,tr]
            y,x = tab['low'][tword]
            h,w = tab['dims'][u]
            # Each entry maps the registered tight rectangle to a native in-plane offset.
            native = tab['transforms'][tr].reshape(4,4)[y:y+h,x:x+w]
            record[int(word)] = (j,native//4,native%4)
    for p,box in enumerate(boxes):
        for axis in range(3):
            s=np.moveaxis(box,axis,0); words=plane_word(box,axis)
            for word in record:
                z,y,x=np.nonzero(words==word)
                if not len(z):
                    continue
                j,dy,dx=record[word];hits[p,j]+=len(z)
                yy=(y[:,None,None]+dy)%20; xx=(x[:,None,None]+dx)%20
                for sign in (-1,1):
                    v = s[(z[:,None,None]+3*sign)%20,yy,xx]==0
                    bonds[p,j] += (v[:,1:,:]&v[:,:-1,:]).sum() + (v[:,:,1:]&v[:,:,:-1]).sum()
                    # Present each actual 2x2x2 tile in a common frame
                    # (normal,row,col); first layer is at the observed tight-box corner.
                    tile = np.stack([s[(z[:,None,None]+d*sign)%20,yy[:,:2,:2],xx[:,:2,:2]]
                                     for d in (0,1)],axis=1)
                    code = tile.reshape(-1,8) @ (3**np.arange(8))
                    for k,lut in enumerate(luts):
                        labels[p,j,k] += np.bincount(lut[code].astype(int),minlength=4)
    h,w = tab['dims'][tab['reg'][pair[0]]]
    return hits,bonds,2*(h*(w-1)+(h-1)*w)*hits,labels


def hom_receipt(bank, zero, out, lut_path):
    from math import comb
    from scipy.stats import beta as bdist

    tab = homometry()
    data = np.load(bank,allow_pickle=False); n = len(data['parent']); boxes = data['species']
    hot = np.load(zero,allow_pickle=False)
    ids, c = hom_support(boxes,tab)
    _, c0 = hom_support(hot['species'],tab)
    columns = {w:i for i,w in enumerate(ids)}
    cases = []
    for a,b in tab['pairs']:
        cs = c[:,[columns[a],columns[b]]]
        parents = (cs>0).sum(0)
        cases.append((int(parents.min()),int(cs.sum(0).min()),int(cs.sum()),a,b))
    best=min(cases,key=lambda z:(-z[0],-z[1],-z[2],z[3],z[4])) if cases else None
    chosen=best[3:] if best else None
    supported=bool(best and best[0]>=128 and best[1]>=1024)
    row=[]
    for a in ids:
        w=tab['reg'][a]; j=columns[a]; cb=c[:,j]; contributors=int((cb>0).sum())
        multiplicity=int((tab['shape']==a).sum()); vac=int(tab['bits'][w].sum())
        p0=multiplicity*comb(2000,vac)*comb(6000,16-vac)/comb(8000,16)/comb(16,vac)
        row.append(dict(shape=a,registered_word=w,box=tab['dims'][w].tolist(),
                        bitmap=[''.join(map(str,r)) for r in tab['bits'][w].reshape(4,4)],
                        occurrences=int(cb.sum()), parents=contributors,
                        upper95_parent_occurrence=(float(bdist.ppf(.95,contributors+1,n-contributors)) if contributors<n else 1.),
                        cold_opportunity_frequency=ratio(cb,np.full(n,24000)),orbit_padded_words=multiplicity,
                        beta0_analytic_opportunity_frequency=p0,beta0_observed_occurrences=int(c0[:,j].sum()),
                        beta0_expected_occurrences=p0*24000*len(c0)))
    r=dict(item=2,cohort='b09 native train, first 256; b00 train descriptive support control only',
           opportunities=n*24000,parents=n, windows_per_parent=24000,finite_masks=65536,
           d4_translation_classes=tab['orbit_shapes'],registered_full_displacement_classes=tab['signature_classes'],
           homometric_pairs=tab['pairs'],shape_details=row,
           selected_by_support=chosen,minimum_parent_support=128,minimum_window_support=1024,
           useful_external_floor=.01,any_homometric_occurrences=int(c.sum()),
           any_homometric_parents=int((c.sum(1)>0).sum()), opportunity_mass=float(c.sum()/(24000*n)),
           source_sha256=sha(bank),beta0_source_sha256=sha(zero),code_sha256=sha(Path(__file__)),
           finite_null_checks='All 65536 full signed vector arrays sum to n_vac^2, symmetric under vector reversal, zero displacement n_vac. Each candidate verified with independent ordered coordinate subtraction, exact box equality and disjoint tight D4 orbits.',
           references=['https://arxiv.org/abs/0808.0094','https://arxiv.org/abs/2605.29945'],
           distinction='Ambiguity of complete internal patch diffraction, not a whole-system pairwise-law test, finite-wavevector score or reconstruction claim.')
    evidence=dict(parent=data['parent'],chain_id=data['chain_id'],chain_seed=data['chain_seed'],
                  shape_ids=np.array(ids,dtype='u2'),occurrences=c,beta0_parent=hot['parent'],
                  beta0_chain_id=hot['chain_id'],beta0_chain_seed=hot['chain_seed'],beta0_occurrences=c0)
    if chosen:
        a,b=chosen
        r['registered_common_signed_vector_displacements']=tab['disp'][tab['reg'][a]].reshape(7,7).tolist()
        r['selected_counts']=[int(c[:,columns[s]].sum()) for s in chosen]
        r['selected_parents']=[int((c[:,columns[s]]>0).sum()) for s in chosen]
    if supported:
        maps = np.load(lut_path,allow_pickle=False)
        hits,scores,denom,labels=external(boxes,tab,chosen,[maps['affine_lut'],maps['discrete_lut']])
        np.testing.assert_array_equal(hits,c[:,[columns[a],columns[b]]])
        res=[ratio(scores[:,k],denom[:,k]) for k in range(2)]
        influence=(scores-denom*np.array([x['mean'] for x in res]))/denom.sum(0)
        mean = res[0]['mean']-res[1]['mean']
        se = float(np.sqrt((n/(n-1))*np.sum(np.diff(influence,axis=1)**2)))
        d = float(t.ppf(.975,n-1))*se
        r.update(measurement='Actual vacancy-pair fraction on opposite registered faces +/-3; one pooled statistic.',
                 face_means=res, first_minus_second=dict(mean=mean,ci95=[mean-d,mean+d],se=se,mde_80pct=2.80*se),
                 code_compatibility=dict(labels=labels.sum(0).tolist(),axes=['shape','affine/discrete','symbol'],
                     description='Actual 2x2x2 registered corner, normal both signs. Distinct whole-patch masks can yield shared symbol values; this is neither identification by a single map nor a measurement of a dense code field.'),
                 lut_sha256=sha(lut_path),decision='Supported; this TRAIN physical contrast is exploratory. No independent evaluation in this receipt.')
        evidence.update(face_vacancy_pairs=scores,face_bonds=denom,local_readouts=labels)
    else:
        r.update(decision='No pair passes the joint independent-parent and occurrence thresholds on cold train. Physical target and LUTs NOT read; no validation. No relaxed displacement match.',
                 code_compatibility='Neither affine nor discrete 2x2x2 tile sees an entire planar 4x4 mask. No supported physical comparison; dense-map separability not tested.')
    nf=out/'02-parents.npz'
    if nf.exists():
        raise FileExistsError(nf)
    np.savez_compressed(nf,**evidence);r['per_parent_sha256']=sha(nf)
    dump(out/'02-development.json',r)
    print('02',chosen,r.get('selected_counts'),r.get('selected_parents'),r['decision'],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['development','validate01'])
    p.add_argument('--bank',type=Path,required=True,help='Private shared bank directory; validation takes the issued .npz file')
    p.add_argument('--out',type=Path,default=Path('results/l20-autoresearch/fifteen'))
    p.add_argument('--lock',type=Path,help='Published item 01 validation lock, required for validation')
    p.add_argument('--lock-sha256',help='Publicly committed digest of validation lock')
    args=p.parse_args();out=args.out
    if args.mode=='development':
        for index in (9,0):
            edge_receipt(args.bank/f'b{index:02d}.npz',index,out,f'development-b{index:02d}')
        hom_receipt(args.bank/'b09.npz',args.bank/'b00.npz',out,out.parent/'faces'/'optimized.npz')
    else:
        if not args.lock or not args.lock_sha256 or sha(args.lock)!=args.lock_sha256:
            raise ValueError('no verified public lock')
        lock=json.loads(args.lock.read_text())
        if lock['item']!=1 or lock['index']!=7 or lock['n_native_validation']!=512 or lock['code_sha256']!=sha(Path(__file__)):
            raise ValueError('lock must pin item, index, size and implementation')
        # Issuance claim before opening outcome bytes; no fallback/rerun or different cohort.
        dump(out/'01-issue.json',dict(index=7,parents=512,public_lock_sha256=args.lock_sha256))
        with np.load(args.bank,allow_pickle=False) as meta:
            if len(meta['parent'])!=512 or abs(float(meta['beta'])-BETA[7])>1e-10:
                raise ValueError('issued bank metadata does not match lock')
        r=edge_receipt(args.bank,7,out,'validation-b07')
        print('PRIMARY',r['gate_except_separate_beta0_check'],flush=True)


if __name__=='__main__':
    main()
