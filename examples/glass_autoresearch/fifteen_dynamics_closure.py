"""Item 13: analytic finite CPU local-attempt kernel; no swap or trajectory is executed."""
import argparse
import hashlib
import itertools
import json
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from scipy.stats import t

from tile_codec import TILE, POW
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l20_pairs import partition

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'results/l20-autoresearch'
N = 8000
UNIT = 2**64
IDEAL = np.array([0, 3, 5], np.int16)
BETA = Decimal('3.05')
SOURCE = 'https://github.com/konradha/lattice-exploration/blob/2975fa6a6fbeb99e6b3f2156b489963f56bfebbe/native/'


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def store(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def geometry():
    """One representative of each touching physical unordered edge, canonical +axis."""
    edge = set()
    for v in TILE:
        for axis in range(3):
            for sign in (-1, 1):
                a = v.copy()
                if sign == -1:
                    a[axis] -= 1
                edge.add((tuple(a), axis))
    edge = sorted(edge)
    a = np.array([v for v, _ in edge]); axis = np.array([d for _, d in edge]); b = a + np.eye(3, dtype=int)[axis]
    power = {tuple(v): int(w) for v, w in zip(TILE, POW)}
    wa, wb = [np.array([power.get(tuple(v), 0) for v in off]) for off in (a, b)]
    assert len(a) == 36 and np.sum((wa != 0) & (wb != 0)) == 12
    anchor = np.moveaxis(np.indices((10, 10, 10)), 0, -1).reshape(-1, 1, 3)*2
    def idx(x):
        return np.ravel_multi_index(((anchor+x) % 20).transpose(2, 0, 1), (20,)*3)
    return idx(TILE), idx(a), idx(b), wa-wb, axis


def nb_sum(v):
    return sum(np.roll(v, s, d) for d in range(3) for s in (-1, 1))


def edge_cost(state):
    """Full endpoint and ten other owner terms; check independent native K formula."""
    x = state.astype(np.int16, copy=False)
    o = (x != 0).astype(np.int16); m = nb_sum(o); r = (m-IDEAL[x])*o; moment = nb_sum(r)
    k = moment + m*(m+1)//2 - o*(IDEAL[x]+1)*m
    costs = []
    for axis in range(3):
        y, p, n, s, u, kb = (np.roll(v, -1, axis) for v in (x, o, m, r, moment, k))
        d = p-o
        # Neighbors of a other than b gain d; neighbors of b other than a lose d.
        # There are no shared other neighbors on the simple cubic L20 lattice.
        h = p*(m-d-IDEAL[y])**2 + o*(n+d-IDEAL[x])**2 - r*r-s*s
        h += 2*d*(moment-s-u+r) + d*d*(m-p+n-o)
        ab = 2*(IDEAL[x]-IDEAL[y])*(m-n)
        ac = 2*(kb-k-IDEAL[x]*n-m-n+2*IDEAL[x])
        bc = 2*(k-kb-IDEAL[y]*m-m-n+2*IDEAL[y])
        oracle = np.where(o*p, ab, np.where(o, ac, bc))
        if not np.array_equal(h[x != y], oracle[x != y]):
            raise ValueError('endpoint/halo dE disagrees with native K expression')
        costs.append(h.ravel())
    return np.stack(costs, axis=1)


def thresholds(beta):
    with localcontext() as ctx:
        ctx.prec = 100
        return [UNIT if beta == 0 or d == 0 else int(Decimal(UNIT)*(-beta*d).exp()) for d in range(128)]


def flux(src, dst, weight):
    # Own-symbol entry is zero in the off-diagonal vector, not diagonal persistence.
    return np.stack([((src[:, None] != j)*(dst == j)*weight).sum(1)/3 for j in range(4)], axis=1)


def witness(src, dst, h, word, site, thresh):
    for w in np.unique(word):
        ix = np.flatnonzero(word == w)
        if len(ix) < 2:
            continue
        for j in range(4):
            if j == src[ix[0]]:
                continue
            values = [sum(thresh[max(int(de), 0)] for de in h[i, dst[i] == j]) for i in ix]
            lo, hi = min(values), max(values)
            if (hi-lo)/(3*UNIT) < .001:
                continue
            def data(val):
                i = ix[values.index(val)]; de, n = np.unique(h[i, dst[i] == j], return_counts=True)
                return dict(anchor_zyx=list(map(int, np.unravel_index(int(site[i, 0]), (20,)*3))),
                            dE_multiplicities={str(int(d)): int(c) for d, c in zip(de, n)},
                            attempt_numerator=str(val), per_attempt=val/(3*N*UNIT), flux=val/(3*UNIT))
            return dict(ternary_word=int(w), source=int(src[ix[0]]), destination=j,
                        attempt_denominator=str(3*N*UNIT), low=data(lo), high=data(hi))
    return None


def interval(x, alpha=.05):
    x = np.asarray(x); mean = np.mean(x, axis=0); err = np.std(x, axis=0, ddof=1)/np.sqrt(len(x))*t.ppf(1-alpha/2, len(x)-1)
    return np.stack([mean, mean-err, mean+err]).tolist()


def summarize(s, rates, weights, free):
    out = []
    for i in range(4):
        select = s == i; wt = weights[select]; v = rates[select]
        mean = np.average(v, axis=0, weights=wt); var = np.average((v-mean)**2, axis=0, weights=wt)
        sub = []
        for z in [False, True]:
            keep = free[select] == z
            sub.append(dict(any_free=z, examples=int(keep.sum()), parents=int(np.any(select & (free==z),axis=1).sum()),
                            mean=np.average(v[keep], axis=0, weights=wt[keep]).tolist()))
        row = mean/N; row[i] = 1.-row.sum()
        assert np.isclose(row.sum(), 1) and np.min(row) >= 0
        out.append(dict(source=i, tiles=int(select.sum()), parents=int(select.any(1).sum()),
                        offdiag_mean_flux=mean.tolist(), full_one_attempt_probability=row.tolist(),
                        within_symbol_variance_each_destination=var.tolist(), trace=float(var.sum()), barrier_subgroups=sub))
    return out


def reg_errors(source, rates, comp, level):
    # Last 128 parents are *training-partition* internal diagnostics, not evaluation outcomes.
    k = len(source)//2; pred = None; errors = []
    for key in (source, source*81 + comp, (source*81 + comp)*5 + level):
        a = key[:k].ravel(); b = key[k:]; size = int(key.max())+1
        n = np.bincount(a, minlength=size)
        sums = np.stack([np.bincount(a, weights=rates[:k, :, j].ravel(), minlength=size) for j in range(4)], axis=1)
        lookup = sums/np.maximum(n, 1)[:, None]
        this = lookup[b]; pred = this if pred is None else np.where((n[b] != 0)[..., None], this, pred)
        errors.append(((rates[k:]-pred)**2).sum(-1).mean(1))
    return np.stack(errors, axis=-1)


def run(bank, out):
    out.mkdir(parents=True, exist_ok=True); path = bank/'b09.npz'; meta = json.loads(path.with_suffix('.json').read_bytes())
    assert meta['bank_sha256'] == sha(path) and meta['index'] == 9
    lut_files = [RESULT/'codec/codec.npz', RESULT/'faces/optimized.npz']; names = ['rsmi', 'affine']
    maps = [np.load(path)[key] for path, key in zip(lut_files, ['lut', 'affine_lut'])]
    site, a, b, dw, axes = geometry(); thresh = thresholds(BETA); probs = np.array([t/UNIT for t in thresh]); witnesses = {}
    with np.load(path) as f:
        parent = f['parent']; ids = np.stack([f['chain_seed'], f['chain_id']], axis=1)
        assert len(parent) == len(np.unique(ids,axis=0)) == 256 and np.array_equal(parent, partition(9)[0][0][:256]); states = f['species']
    sources = np.empty((2,256,1000),np.uint8); rates = np.empty((2,256,1000,4)); comps = np.empty((256,1000),np.int16); levels = np.empty_like(comps)
    powers = POW.astype(int); nontrivial = 0; no_accept = 0
    for p, state in enumerate(states):
        x = state.ravel().astype(np.int16); d = x[site]; word = d@powers; sub = word[:,None] + (x[b]-x[a])*dw
        de = edge_cost(state)[a, axes]; unlike = x[a] != x[b]; nontrivial += int(unlike.sum()); weight = probs[de.clip(min=0)]; no_accept += int(((weight==0)&unlike).sum())
        minimum = np.min(np.where(unlike, de, 1000), axis=1)
        levels[p] = np.select([minimum<=0,minimum<=2,minimum<=4,minimum<1000], [0,1,2,3], default=4)
        comps[p] = 9*(d==1).sum(1)+(d==2).sum(1)
        for m, lut in enumerate(maps):
            s, dest = lut[word], lut[sub]; f = flux(s, dest, weight); rates[m,p] = f; sources[m,p] = s
            assert np.max(f.sum(1)/N) <= 36/(3*N) and np.min(f) >= 0
            if names[m] not in witnesses:
                w = witness(s, dest, de, word, site, thresh)
                if w: witnesses[names[m]] = dict(release_row=int(parent[p]), **w)
            if p == 0:
                # Exact beta-zero unordered counts vs separately counted oriented incidences.
                internal = (np.abs(dw)[None,:] != 0)  # No reliance on physical acceptances here.
                z = flux(s,dest,np.ones_like(weight))
                assert np.array_equal(np.rint(z*3).astype(int), np.stack([((dest==j)&(s[:,None]!=j)&unlike&internal).sum(1) for j in range(4)],axis=-1))
        if p == 0:
            assert not flux(np.zeros(len(d)), np.zeros_like(sub), weight).any()
    # Common reweighting on *identical* microstate realizations; label alignment never changes the LUT.
    joint = np.bincount((4*sources[0]+sources[1]).ravel(),minlength=16).reshape(4,4)
    perm = max(itertools.permutations(range(4)), key=lambda p: sum(joint[i,j] for i,j in enumerate(p)))
    aligned = np.argsort(perm)[sources[1]]; c = joint[:,perm]; sym = np.where((c>0)&(c.T>0), (c+c.T)/2, 0.)
    adjust = np.divide(sym,c,out=np.zeros((4,4)),where=c>0); wgt = adjust[sources[0],aligned]
    def entropy(s,w):
        mass = np.bincount(s.ravel(),weights=w.ravel(),minlength=4)/w.sum(); return float(-np.sum(mass*np.log2(mass)))
    counts = {}; tr = []; packet = dict(parent=parent, chain_seed=ids[:,0], chain_id=ids[:,1]); free = levels==0
    for m,name in enumerate(names):
        s, v = sources[m],rates[m]; q = {}
        for label,w in [('native',np.ones_like(wgt)),('common_entropy',wgt)]:
            z = summarize(s,v,w,free); means = np.array([e['offdiag_mean_flux'] for e in z]); sq = ((v-means[s])**2).sum(-1)
            # Parent additive contributions to pooled variance, scaled by the average parent weight.
            within = (w*sq).sum(1)/(w.sum()/len(w))
            q[label] = dict(entropy_bits=entropy(s,w), symbols=z, trace_parent_interval=interval(within)); packet[name+'_'+label+'_within'] = within
            packet[name+'_'+label+'_symbol_mass'] = np.stack([(w*(s==j)).sum(1) for j in range(4)],axis=1)
            packet[name+'_'+label+'_weighted_rate_sum'] = np.stack([((w*(s==j))[...,None]*v).sum(1) for j in range(4)],axis=1)
            packet[name+'_'+label+'_weighted_rate2_sum'] = np.stack([((w*(s==j))[...,None]*(v*v)).sum(1) for j in range(4)],axis=1)
            if label == 'common_entropy': tr.append(within)
        err = reg_errors(s.astype(np.int32),v,comps,levels); packet[name+'_internal_mse'] = err
        q['internal_prediction'] = dict(fit_native_train_parents=128, other_native_train_parents=128, model_order=['H','H+A/B counts','H+A/B counts+min touching barrier bin'], trace_MSE_interval=interval(err), barrier_beyond_composition_gain=interval(err[:,1]-err[:,2]), all_beyond_H_gain=interval(err[:,0]-err[:,2])); counts[name] = q
    assert abs(counts['affine']['common_entropy']['entropy_bits']-counts['rsmi']['common_entropy']['entropy_bits']) < 1e-13
    diff = tr[0]-tr[1]; rows = out/'13-parent-evidence.npz'; np.savez_compressed(rows, **packet)
    store(out/'13-kinetic-closure.json',dict(question='Does a single frozen registered tile symbol close the one-local-attempt physical kernel?',item=13,design='13-design.json',contrast_to_history='Per-destination exact instantaneous flux, not physical mobility prediction, mean acceptance, or face log score.',source=meta,parents=256,tiles=256000,edges_per_tile=dict(internal=12,crossing=24),touching_edge_exposures=256000*36,unlike_edge_exposures=nontrivial,finite_threshold_zero_among_unlike=no_accept,physical_states_generated=0,physical_transitions_executed=0,source_kernel=[SOURCE+'common/proposal.hpp',SOURCE+'common/efma.hpp',SOURCE+'cpu/chain.hpp'],lut_sha256=dict(zip(names,map(sha,lut_files))),
        exact_acceptance='Numerator T(d)=2^64 if d<=0, else floor(2^64 exp(-3.05*d)); denominator per unordered edge 3*8000*2^64. Convert to double only after threshold construction; witness numerators summed as integers.',units='Reported F is off-diagonal one-attempt probability times 8000 (instantaneous sweep-scaled flux, not finite-sweep probability). Variance units F^2; own-symbol entry set to zero for variance and MSE.',
        contrast=counts, common_entropy_design=dict(rsmi_aligned_to_affine=list(perm),joint_counts=c.tolist(),cell_weights=adjust.tolist(),retained_tiles=int((wgt>0).sum()),effective_tiles=float(wgt.sum()**2/np.square(wgt).sum()),same_weighted_population=True), rsmi_minus_affine_matched_trace=dict(parent95=interval(diff),parent_fifteen_bonferroni=interval(diff,.05/15),MDE80_approx=(t.ppf(1-.05/30,len(diff)-1)+.841621)*float(diff.std(ddof=1))/np.sqrt(len(diff))),
        witness_registration='Translate each indicated even-grid anchor to the same origin. Periodic translation preserves canonical weight, physical kernel and tile registration, so the two observed environments certify different rates for one fixed tile. This is a coordinate argument; no translated box was created.',
        witnesses=witnesses, parent_evidence_sha256=sha(rows),controls=dict(cached_CPU_vs_twelve_owner_energy='agreed on all unlike edges in each whole box',beta_zero='first parent, both LUTs: accepted off-diagonal counts match unlike destination counts',constant_code='exactly zero off-diagonal flux',normalization='all row means sum to 1; every tile escape <=36/(3*8000)'),decision='Exact one-tile strong lumpability refuted for both codes by equal-word integer witnesses.' if len(witnesses)==2 else 'See available witnesses; absence cannot certify lumpability.',interpretation='Exploratory parent intervals conditional on these trained codes, common empirical weights/means; not whole-field closure or a proof of non-Markov equilibrium time series.',citations=['https://doi.org/10.1016/S0167-2789(02)00446-3','https://doi.org/10.1073/pnas.2427246122']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('bank',type=Path); p.add_argument('--out',type=Path,default=RESULT/'fifteen'); args=p.parse_args();run(args.bank,args.out)
