"""Forward HERO: FIT-only alignment, one microscopic distillate, no assessment reader.

Coordinates throughout are (z,y,x); positive bonds point in increasing coordinate.
Public rules and controls are JSON dictionaries, not pickles or executable strings.
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


XYZ = np.array(list(product(range(-1, 2), repeat=3)), dtype=int)
AXIS = ('z', 'y', 'x')
SPECIES = ('vac', 'A', 'B')
ORIGIN = 13
NN = np.flatnonzero(np.abs(XYZ).sum(axis=1) == 1)
INDEX = {tuple(r): i for i, r in enumerate(XYZ)}
STEP = np.eye(3, dtype=int)
EDGES = tuple(np.array([(i, INDEX[tuple(r + u)]) for i, r in enumerate(XYZ)
                        if tuple(r + u) in INDEX]).T for u in STEP)
LINES = tuple(np.array([(i, INDEX[tuple(r + u)], INDEX[tuple(r + 2*u)])
                        for i, r in enumerate(XYZ) if tuple(r + 2*u) in INDEX]).T for u in STEP)
SQUARES = tuple(np.array([(i, INDEX[tuple(r+u)], INDEX[tuple(r+v)], INDEX[tuple(r+u+v)])
                          for i, r in enumerate(XYZ) if tuple(r+u+v) in INDEX]).T
                for a in range(3) for u, v in [STEP[np.arange(3) != a]])
CHUNK = 2048
N_STRATA = 640


def rotations():
    """Proper signed-coordinate permutations (no translations, mirrors or site alignment)."""
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            m = STEP[list(axes)] * np.asarray(signs)[:, None]
            if round(np.linalg.det(m)) == 1:
                yield m


@lru_cache(maxsize=2)
def _template_masks(species):
    normal, period = ((1, 1, 1), 3) if species == 1 else ((1, 2, 3), 7)
    phases = [(XYZ @ (rot @ normal)) % period == phase for rot in rotations() for phase in range(period)]
    return np.ascontiguousarray(np.unique(phases, axis=0).T, dtype=np.float32)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str
    args: tuple
    cost: int
    conventional: bool
    symmetry: str
    formula: str


def _dictionary():
    f = []
    def add(name, kind, args, cost, conv, sym, formula):
        f.append(Feature(name, kind, args, cost, conv, sym, formula))
    scalar = 'invariant under all proper cubic rotations'
    axis = 'fixed unoriented axis; permuted by proper rotations'
    signed = 'fixed oriented component; proper rotations can permute and negate it'
    for s, species in enumerate(SPECIES):
        for i, r in enumerate(XYZ):
            coord = ','.join(map(str, r))
            add(f'site:{coord}:{species}', 'site', (i,s), 1, True,
                scalar if i == ORIGIN else 'fixed offset; permuted by proper rotations', f'1[V({coord})={species}]')
        add('count:'+species, 'count', (s,), 1, True, scalar, f'number of {species} sites in 27-site V')
        add('center_nn:'+species, 'nn', (s,), 2, True, scalar, f'number of {species} among the six nearest neighbors of the origin')
    add('internal_oo', 'oo', (), 2, True, scalar, 'occupied-occupied edges among the 54 internal nearest-neighbor edges')
    add('internal_AB', 'ab', (), 2, True, scalar, 'unlike occupied A/B edges among the 54 internal nearest-neighbor edges')
    add('center_energy', 'energy', (), 2, True, scalar, '1[origin occupied]*(number of its occupied nearest neighbors - preferred degree 3(A) or 5(B))^2')
    for a, ax in enumerate(AXIS):
        add('oo:'+ax, 'oo_axis', (a,), 2, True, axis, f'occupied-occupied internal edges parallel to {ax}')
        add('AB_signed:'+ax, 'signed', (a,), 3, True, signed, f'#(A->B) - #(B->A) among 18 edges oriented along +{ax}')
        add('AvB_signed:'+ax, 'line', (a,), 3, False, signed, f'#(A,vac,B) - #(B,vac,A) in 9 straight length-three lines along +{ax}')
        add('square_diagonal:'+ax, 'square', (a,), 3, False, axis, f'number of 12 open elementary squares normal to {ax} with one vacancy diagonal and complementary occupied diagonal')
    for s, normal, p in ((1,(1,1,1),3), (2,(1,2,3),7)):
        add('crystal:'+SPECIES[s], 'crystal', (s,), 4, True, scalar,
            f'max fraction of 27 sites matching pure {SPECIES[s]} with vacancy where r·g{normal}=phase mod {p}, over all proper cubic g and phases')
    return {entry.name: entry for entry in f}


FEATURES = _dictionary()
CONVENTIONAL = tuple(f for f in FEATURES.values() if f.conventional) + tuple(
    Feature(f'pair:{AXIS[a]}:{s}{d}', 'pair', (a,s,d), 0, True, '', '')
    for a,s,d in product(range(3), repeat=3)) + tuple(
    Feature(f'center_degree:{s}:{k}', 'degree', (s,k), 0, True, '', '')
    for s,k in product(range(3), range(7)))


class _Word:
    def __init__(self, v):
        self.v = v
        self.cache = {}

    @cached_property
    def center(self):
        return self.v[:, ORIGIN]

    @cached_property
    def nn_occ(self):
        return np.count_nonzero(self.v[:, NN], axis=1)

    def get(self, kind, *args):
        key = (kind, *args)
        if key in self.cache:
            return self.cache[key]
        v = self.v
        if kind == 'site':
            i,s = args; x = v[:,i] == s
        elif kind == 'count':
            x = np.count_nonzero(v == args[0], axis=1)
        elif kind == 'nn':
            x = np.count_nonzero(v[:,NN] == args[0], axis=1)
        elif kind == 'oo':
            x = sum(self.get('oo_axis',a) for a in range(3))
        elif kind == 'ab':
            x = sum(self.get('pair',a,s,d) for a in range(3) for s,d in ((1,2),(2,1)))
        elif kind == 'oo_axis':
            i,j = EDGES[args[0]]; x = np.count_nonzero((v[:,i] != 0) & (v[:,j] != 0), axis=1)
        elif kind == 'pair':
            a,s,d = args; i,j = EDGES[a]; x = np.count_nonzero((v[:,i] == s) & (v[:,j] == d), axis=1)
        elif kind == 'signed':
            a, = args; x = self.get('pair',a,1,2) - self.get('pair',a,2,1)
        elif kind == 'energy':
            x = (self.center != 0) * (self.nn_occ - np.where(self.center == 1, 3, 5))**2
        elif kind == 'degree':
            s,k = args; x = (self.center == s) & (self.nn_occ == k)
        elif kind == 'line':
            i,j,k = LINES[args[0]]; void = v[:,j] == 0
            x = (np.count_nonzero((v[:,i] == 1) & void & (v[:,k] == 2), axis=1)
                 - np.count_nonzero((v[:,i] == 2) & void & (v[:,k] == 1), axis=1))
        elif kind == 'square':
            i,j,k,l = SQUARES[args[0]]; a,b,c,d = (v[:,ix] == 0 for ix in (i,j,k,l))
            x = np.count_nonzero((a & d & ~b & ~c) | (b & c & ~a & ~d), axis=1)
        elif kind == 'crystal':
            s, = args; occ = (v == s).astype('f4'); dif = (v == 0).astype('f4') - occ
            x = (occ.sum(1) + (dif @ _template_masks(s)).max(1)) / 27.
        else:
            raise KeyError(kind)
        self.cache[key] = x
        return x

    def feature(self, f):
        return self.get(f.kind, *f.args)

    def matrix(self, features):
        a = np.empty((len(self.v),len(features)), dtype='f8')
        for i,f in enumerate(features):
            a[:,i] = self.feature(f)
        return a

    def stratum(self):
        return ((self.get('count',0)//3)*8 + self.get('count',1)//4)*8 + self.get('oo')//8


def _input(v):
    v = np.asarray(v)
    if v.ndim < 1 or v.shape[-1] != 27 or v.dtype.kind not in 'ui' or (v.size and (v.min() < 0 or v.max() > 2)):
        raise ValueError('V must be integer species 0,1,2 with final dimension 27')
    return v.reshape(-1,27), v.shape[:-1]


def _mapped(v, fn):
    v, shape = _input(v)
    result = np.empty(len(v), dtype='f8')
    for j in range(0,len(v),CHUNK):
        result[j:j+CHUNK] = fn(_Word(v[j:j+CHUNK]))
    return result.reshape(shape)


def feature_value(v, name):
    f = FEATURES[name]
    return _mapped(v, lambda w: w.feature(f))


@lru_cache(maxsize=2)
def _weights(control):
    rng = np.random.default_rng(71991)
    if control == 'random_affine':
        return rng.standard_normal(54) / np.sqrt(54.)
    return rng.standard_normal((54,16)) / np.sqrt(54.), rng.standard_normal(16) / 4.


def _control_value(w, control):
    if control == 'site':
        return w.center
    if control == 'composition':
        return w.get('count',0)
    if control == 'coordination':
        return w.get('oo')
    if control == 'crystal':
        return np.maximum(w.get('crystal',1), w.get('crystal',2))
    x = np.concatenate((w.v == 1, w.v == 0), axis=1).astype('f8')
    if control == 'random_affine':
        return x @ _weights(control)
    if control == 'random_tanh':
        a,b = _weights(control)
        return np.tanh(x @ a) @ b
    raise KeyError(control)


def fit_control(v, control, mass=None):
    """Calibrate hard thresholds on FIT only; equality goes in the lower bin."""
    x = _mapped(v, lambda w: _control_value(w, control)).ravel()
    if not len(x):
        raise ValueError('empty FIT')
    if mass is None:
        mass = np.full(4,.25)
    else:
        mass = np.sort(np.asarray(mass, dtype='f8'))[::-1]
        if mass.shape != (4,) or np.any(mass < 0) or not np.isclose(mass.sum(),1):
            raise ValueError('mass must be the FIT probability vector for four hard symbols')
        mass = mass / mass.sum()
    cuts = [] if control == 'site' else np.quantile(x, np.clip(mass.cumsum()[:-1],0,1), method='inverted_cdf').tolist()
    h = x.astype(int) if control == 'site' else np.searchsorted(cuts,x,side='left')
    return dict(control=control, cuts=cuts, seed=71991, target_mass=None if control == 'site' else mass.tolist(),
                fit_mass=(np.bincount(h, minlength=4)/len(h)).tolist())


def hard_control(v, spec):
    control = spec['control']
    x = _mapped(v, lambda w: _control_value(w, control))
    return x.astype(np.int64) if control == 'site' else np.searchsorted(spec['cuts'], x, side='left')


def _prob(l):
    if l.ndim != 3 or l.shape[-1] != 4 or not np.isfinite(l).all():
        raise ValueError('expected finite [parent,anchor,4] logits')
    e = np.exp(l.astype('f8') - l.max(axis=-1,keepdims=True))
    return e / e.sum(axis=-1,keepdims=True)


def align_fit(logits, seeds):
    """Freeze permutation and contrast on D. No spatial realignment is possible here."""
    if not logits or sorted(set(seeds)) != list(seeds) or len(seeds) != len(logits):
        raise ValueError('distinct numerically sorted seeds and one tensor per seed required')
    p0 = _prob(logits[0]).reshape(-1,4)
    h0 = logits[0].argmax(axis=-1).ravel()
    freq = np.bincount(h0,minlength=4)
    mass = p0.mean(0)
    ref = sorted(range(4), key=lambda i: (-freq[i],-mass[i],i))
    href = np.argsort(ref)[h0]
    pref = p0[:,ref]
    out = []
    for k,l in enumerate(logits):
        if l.shape != logits[0].shape:
            raise ValueError('unpaired FIT anchors across seeds')
        p = _prob(l).reshape(-1,4)
        h = l.argmax(axis=-1).ravel()
        conf = np.bincount(4*href+h, minlength=16).reshape(4,4)
        soft = pref.T @ p
        def match(perm):
            return (-int(conf[np.arange(4),perm].sum()), -float(soft[np.arange(4),perm].sum()), perm)
        perm = ref if k == 0 else min(permutations(range(4)), key=match)
        raw = l[...,perm[0]].astype('f8') - l[...,perm[1]]
        out.append(dict(seed=int(seeds[k]), aligned_to_raw=list(perm), mean=float(raw.mean()), sd=float(raw.std()),
                        fit_modal_count_tie=bool(len(set(freq.tolist())) < 4) if k == 0 else None))
    return out


def contrast(logits, maps):
    zs,hs,report = [],[],[]
    for l,m in zip(logits,maps,strict=True):
        perm = m['aligned_to_raw']
        raw = l[...,perm[0]].astype('f8') - l[...,perm[1]]
        h = np.argsort(perm)[l.argmax(axis=-1)]
        p = _prob(l)[...,perm]
        hard = np.bincount(h.ravel(),minlength=4)/h.size
        soft = p.mean(axis=(0,1))
        def entropy(prob):
            return -(prob*np.log2(np.maximum(prob,1e-300))).sum(axis=-1)
        hs.append(h); zs.append((raw-m['mean'])/m['sd'] if m['sd'] > 0 else np.zeros_like(raw))
        report.append(dict(seed=m['seed'], raw_mean=float(raw.mean()), raw_sd=float(raw.std()),
                           hard_mass=hard.tolist(), soft_mass=soft.tolist(), hard_entropy_bits=float(entropy(hard)),
                           mean_draw_entropy_bits=float(entropy(p).mean()), draw_marginal_entropy_bits=float(entropy(soft)),
                           draw_information_bits=float(entropy(soft)-entropy(p).mean()),
                           argmax_tie_fraction=float(np.mean((l==l.max(axis=-1,keepdims=True)).sum(axis=-1)>1))))
    return np.mean(zs,axis=0),zs,hs,report


def _pair_row(x,y,h,k):
    # Per-parent sufficient statistics; equal numbers of anchors for every parent.
    return np.column_stack((x.mean(1), y.mean(1), (x*x).mean(1), (y*y).mean(1), (x*y).mean(1),
                            (h==k).mean(1), *((h==i).mean(1) for i in range(4)),
                            *((k==i).mean(1) for i in range(4))))


def _pair_measures(m):
    vx = np.maximum(0.,m[...,2]-m[...,0]**2)
    vy = np.maximum(0.,m[...,3]-m[...,1]**2)
    den = np.sqrt(vx*vy)
    cov = m[...,4]-m[...,0]*m[...,1]
    chance = (m[...,6:10]*m[...,10:14]).sum(axis=-1)
    corr = np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-14)
    kap = np.divide(m[...,5]-chance,1.-chance,out=np.zeros_like(chance),where=chance < 1.-1e-14)
    return np.stack((corr,kap),axis=-1)


def _repro(zs,hs,seeds):
    pairs = list(combinations(range(len(zs)),2))
    n = len(zs[0])
    if n < 3:
        raise ValueError('parent jackknife requires at least three parents')
    crit = float(student_t.ppf(1.-.05/(2*len(pairs)),n-1)) if pairs else None
    report = []
    for i,j in pairs:
        rows = _pair_row(zs[i],zs[j],hs[i],hs[j]); mean = rows.mean(0)
        estimate = _pair_measures(mean)
        loo = _pair_measures((rows.sum(0)-rows)/(n-1))
        se = np.sqrt((n-1)*np.mean((loo-loo.mean(0))**2,axis=0))
        lower = estimate - crit*se
        null = _pair_measures(_pair_row(zs[i],np.roll(zs[j],1,axis=0),hs[i],np.roll(hs[j],1,axis=0)).mean(0))
        report.append(dict(seeds=[seeds[i],seeds[j]], pearson=float(estimate[0]), kappa=float(estimate[1]),
                           hard_agreement=float(mean[5]), pearson_lower=float(lower[0]), kappa_lower=float(lower[1]),
                           parent_shift_pearson=float(null[0]), parent_shift_kappa=float(null[1])))
    passed = len(seeds) >= 3 and all(r['pearson'] >= .60 and r['kappa'] >= .30 and r['pearson_lower'] > .40
                                      and r['kappa_lower'] > .10 for r in report)
    return report,passed


def _moments(v,y):
    """Single-feature normal equations, never reading any exterior."""
    v,shape = _input(v)
    if shape != y.shape or len(v) == 0:
        raise ValueError('target/word shape mismatch')
    y = y.ravel().astype('f8')
    features = tuple(FEATURES.values())
    s = np.zeros((3,len(features)))
    for j in range(0,len(v),CHUNK):
        x = _Word(v[j:j+CHUNK]).matrix(features)
        s += np.stack((x.sum(0), np.einsum('ij,ij->j',x,x), y[j:j+CHUNK] @ x))
    ex,ex2,exy = s/len(v)
    return float(y.mean()),float(y.var()),ex,np.maximum(0.,ex2-ex*ex),exy-ex*y.mean()


def _rank(vd,yd,vr,yr):
    ym,yv,xm,xv,cxy = _moments(vd,yd)
    ym2,yv2,xm2,xv2,cxy2 = _moments(vr,yr)
    b = np.divide(cxy,xv,out=np.zeros_like(xv),where=xv>1e-14)
    a = ym-b*xm
    mse = np.maximum(0.,yv-2*b*cxy+b*b*xv)
    mse2 = np.maximum(0.,yv2-2*b*cxy2+b*b*xv2+(ym2-a-b*xm2)**2)
    rows = []
    for j,f in enumerate(FEATURES.values()):
        rows.append(dict(name=f.name,cost=f.cost,conventional=f.conventional,intercept=float(a[j]),slope=float(b[j]),
                         fit_variance=float(xv[j]),fit_r2=float(1.-mse[j]/yv) if yv>1e-14 and xv[j]>1e-14 else None,
                         rule_r2=float(1.-mse2[j]/yv2) if yv2>1e-14 and xv[j]>1e-14 else None))
    fit = sorted((r for r in rows if r['fit_r2'] is not None), key=lambda r:(-r['fit_r2'],r['cost'],r['name']))
    rule = [r for r in rows if r['rule_r2'] is not None]
    if not rule:
        return rows,fit[0] if fit else None,None
    best = max(r['rule_r2'] for r in rule)
    ordered = sorted((r for r in rule if r['rule_r2'] >= best-.02),key=lambda r:(r['cost'],-r['rule_r2'],r['name']))
    return rows,fit[0] if fit else None,ordered[0]


def apply_rule(v,spec):
    x = feature_value(v,spec['feature'])
    return spec['sign']*(x-spec['fit_mean'])/spec['fit_sd']


def _raw_f(w,spec):
    return spec['sign']*(w.feature(FEATURES[spec['feature']])-spec['fit_mean'])/spec['fit_sd']


def _projection(w,spec):
    m = spec['conventional_model']
    if m['name'] == 'exact_dictionary_identity':
        return _raw_f(w,spec)
    x = np.full(len(w.v),m['intercept'],dtype='f8')
    if m['coef'] is not None:
        x += w.matrix(CONVENTIONAL) @ m['coef']
    if m['stratum_offset'] is not None:
        x += np.asarray(m['stratum_offset'])[w.stratum()]
    return x


def conventional_rule(v,spec):
    return _mapped(v,lambda w: _projection(w,spec))


def residual_rule(v,spec):
    if spec['conventional_model']['name'] == 'exact_dictionary_identity':
        _,shape = _input(v)
        return np.zeros(shape,dtype='f8')
    return _mapped(v,lambda w: _raw_f(w,spec)-_projection(w,spec))


def _sufficient(v,spec,parity):
    d = len(CONVENTIONAL)+2  # intercept, ordinary features, f
    gram = np.zeros((d,d)); groups = np.zeros((N_STRATA,d))
    # Parent parity, not anchor parity. Do not materialize a second copy of the FIT bank.
    nparent = max(1,CHUNK//v.shape[1])
    for j in range(parity,len(v),2*nparent):
        w = _Word(v[j:j+2*nparent:2].reshape(-1,27))
        z = np.empty((len(w.v),d)); z[:,0] = 1.; z[:,-1] = _raw_f(w,spec)
        z[:,1:-1] = w.matrix(CONVENTIONAL)
        gram += z.T @ z
        np.add.at(groups,w.stratum(),z)
    return gram,groups


def _models(gram,groups):
    a = gram/gram[0,0]
    xm = a[0,1:-1]; ym = a[0,-1]
    cov = a[1:-1,1:-1]-np.outer(xm,xm)
    e,u = np.linalg.eigh(cov)
    keep = e > e[-1]*1e-10
    b = u[:,keep] @ ((u[:,keep].T @ (a[1:-1,-1]-xm*ym))/e[keep])
    out = {}
    for name in ('intercept','stratum','ols','ols_plus_stratum'):
        linear = name.startswith('ols')
        strat = 'stratum' in name
        coef = b if linear else np.zeros_like(b)
        intercept = float(ym - xm@coef)
        offsets = np.divide(groups[:,-1]-groups[:,0]*intercept-groups[:,1:-1]@coef, groups[:,0],
                            out=np.zeros(len(groups)),where=groups[:,0]>0) if strat else None
        out[name] = dict(name=name,intercept=intercept,coef=coef.tolist() if linear else None,
                         stratum_offset=offsets.tolist() if strat else None)
    return out


def _mse(gram,groups,m):
    coef = np.zeros(len(gram)-2) if m['coef'] is None else m['coef']
    u = np.r_[m['intercept'],coef,-1.]
    sse = u @ gram @ u
    if m['stratum_offset'] is not None:
        off = np.asarray(m['stratum_offset'])
        sse += 2.*off @ (groups@u) + off**2 @ groups[:,0]
    return max(0.,float(sse/gram[0,0]))


def _fit_projection(v,spec):
    folds = [_sufficient(v,spec,i) for i in range(2)]
    if any(g[0,0] == 0 for g,_ in folds):
        raise ValueError('both parent parity folds must be populated')
    gram = folds[0][0]+folds[1][0]; groups = folds[0][1]+folds[1][1]
    errors = Counter()
    for i,(g,s) in enumerate(folds):
        models = _models(g,s); other = folds[1-i]
        for name,m in models.items():
            errors[name] += _mse(*other,m) * other[0][0,0] / gram[0,0]
    models = _models(gram,groups)
    best = next(name for name in models if errors[name] <= min(errors.values())+1e-12)
    if FEATURES[spec['feature']].conventional:
        m = dict(name='exact_dictionary_identity'); errors[m['name']] = 0.
    else:
        m = models[best]
    return m,dict(errors),_mse(gram,groups,models['stratum']),int(np.count_nonzero(groups[:,0]))


def _within(y,strata):
    y = y.ravel(); s = strata.ravel().astype(np.int64)
    mass = np.bincount(s,minlength=N_STRATA)
    sums = np.bincount(s,weights=y,minlength=N_STRATA)
    avg = np.divide(sums,mass,out=np.zeros_like(sums),where=mass>0)
    return dict(within_s_variance=float(np.mean((y-avg[s])**2)), total_variance=float(y.var()),
                occupied_strata=int(np.count_nonzero(mass)),singleton_strata=int(np.count_nonzero(mass==1)))


def distill(vfit,vrule,ld,lr,seeds,source):
    start = time.monotonic()
    if ld[0].shape[0] > min(len(vfit),1024) or np.shape(vrule)[:-1] != lr[0].shape[:-1]:
        raise ValueError('D/RULE shape mismatch')
    vd = vfit[:len(ld[0])]
    maps = align_fit(ld,seeds)
    yd,zd,hd,pd = contrast(ld,maps)
    yr,zr,hr,pr = contrast(lr,maps)
    pairs,stable = _repro(zr,hr,seeds)
    stable &= all(r['raw_sd'] >= .25 and min(r['hard_mass']) >= .02 for r in pd+pr)
    rows,bfit,sel = _rank(vd,yd,vrule,yr)
    sd = _mapped(vd,lambda w:w.stratum()); sr = _mapped(vrule,lambda w:w.stratum())
    report = dict(source=source, fit_parents=len(vfit), d_parents=len(vd), rule_parents=len(vrule),
                  d_anchors=vd.shape[1], rule_anchors=vrule.shape[1], alignment=maps, d_seed_statistics=pd,
                  rule_seed_statistics=pr, rule_reproducibility=pairs, cross_seed_passed=bool(stable),
                  d_teacher_variance=_within(yd,sd), rule_teacher_variance=_within(yr,sr),
                  dictionary_scores=rows,best_fit_only=bfit,selection=sel,
                  status_note='RULE is development and shared with symbolic selection; never an assessment result.')
    spec = dict(schema='hero-rule-1',source=source,feature=None,symbolic_passed=False,rediscovery=False,
                residual_fraction=None,status='no_nondegenerate_teacher_or_candidate')
    if sel is not None:
        f = FEATURES[sel['name']]
        x = feature_value(vfit,f.name)
        mu = float(x.mean()); sig = float(x.std())
        sign = int(np.sign(sel['slope']))
        spec.update(feature=f.name,formula=f.formula,symmetry=f.symmetry,fit_mean=mu,fit_sd=sig,sign=sign,
                    units='signed FIT population SD of microscopic feature; positive points toward reference sector 1 vs 2',
                    domain='periodic L20 center extraction, 27 open-cube sites in lexicographic (z,y,x) order, 0 vacant/1 A/2 B',
                    teacher='mean of per-seed D-standardized aligned natural log-odds; continuous; not a two-bit symbol',
                    teacher_fit_affine=dict(intercept=sel['intercept'],slope=sel['slope']))
        if sig*sig >= 1e-8 and sign:
            model,errors,within,ns = _fit_projection(vfit,spec)
            spec['conventional_model'] = model
            spec['conventional_feature_order'] = [g.name for g in CONVENTIONAL] if model.get('coef') is not None else []
            fv = sign*(x-mu)/sig; rv = residual_rule(vfit,spec)
            fr = apply_rule(vrule,spec); rr = residual_rule(vrule,spec)
            var = float(fv.var()); fraction = float(rv.var()/var)
            spec.update(symbolic_passed=bool(stable and sel['rule_r2'] >= .50),
                        rediscovery=f.conventional, residual_fraction=fraction,fit_variance=var,
                        fit_residual_variance=float(rv.var()),fit_within_s_variance=within)
            if not stable:
                status = 'unstable_teacher_explanatory_only'
            elif sel['rule_r2'] < .50:
                status = 'insufficient_symbolic_fidelity_explanatory_only'
            elif f.conventional:
                status = 'conventional_rediscovery_no_exam'
            elif fraction < .10:
                status = 'insufficient_distinct_residual_no_exam'
            else:
                status = 'symbolic_nominee_only_pending_boundary_gate'
            spec['status'] = status
            report.update(fit_projection_oof_mse=errors,fit_projection_chosen=model['name'],fit_occupied_strata=ns,
                          fit_residual_mean=float(rv.mean()),rule_projection_mse=float(np.mean(rr**2)),
                          rule_residual_variance=float(rr.var()),rule_f_variance=_within(fr,sr))
        else:
            spec['status'] = 'vanishing_feature_or_zero_slope_no_exam'
    report['elapsed_seconds'] = time.monotonic()-start
    return spec,report


def _file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def _write_json(path,obj):
    with Path(path).open('x') as f:
        f.write(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def finite_controls():
    """Test finite arithmetic and selection, not a glass ensemble, exterior, or cavity."""
    start = time.monotonic()
    out = {}
    def require(name,ok,details):
        out[name] = dict(passed=bool(ok),details=details)
    full = np.ones((1,27),dtype=np.uint8); void = full*0
    require('open_geometry', len(NN)==6 and sum(e.shape[1] for e in EDGES)==54
            and all(e.shape[1]==12 for e in SQUARES) and all(e.shape[1]==9 for e in LINES)
            and feature_value(full,'internal_oo').item()==54 and feature_value(void,'internal_oo').item()==0
            and feature_value(full,'center_energy').item()==9 and feature_value(full*2,'center_energy').item()==1,
            '54 occupied edges on all-A word, zero on all-vacant word; origin A/B energies 9/1')
    v = np.zeros((1,27),dtype='u1')
    ends = (INDEX[(0,0,-1)],INDEX[(0,0,1)]); v[0,ends] = [1,2]
    rr = np.array([INDEX[tuple(r*np.array([1,-1,-1]))] for r in XYZ]); vw = v[:,rr]
    require('signed_open_line', feature_value(v,'AvB_signed:x').item()==1 and feature_value(vw,'AvB_signed:x').item()==-1
            and feature_value(v,'AvB_signed:z').item()==0,'isolated A-vac-B in +x: +1, under proper 180 degree z rotation: -1')
    v[0,ORIGIN] = 2
    require('oriented_bond',feature_value(v,'AB_signed:x').item()==1 and feature_value(v[:,rr],'AB_signed:x').item()==-1,
            'isolated A->B->B line')
    for s,normal,p in [(1,(1,1,1),3),(2,(1,2,3),7)]:
        cr = np.where((XYZ@normal)%p==1,0,s)[None,:].astype('u1')
        vals = [feature_value(cr[:,[INDEX[tuple(r@rot)] for r in XYZ]],'crystal:'+SPECIES[s]).item() for rot in rotations()]
        require('crystal_'+SPECIES[s],len(vals)==24 and min(vals)==1.,dict(rotations=len(vals),min_literal_agreement=min(vals)))

    # Finite random test *words*, not physical configurations or shell samples.
    rng = np.random.default_rng(129913)
    vd = rng.integers(0,3,size=(160,24,27),dtype='u1')
    vr = rng.integers(0,3,size=(96,24,27),dtype='u1')
    seeds = [7,13,19]
    def plant(w, center=9.125):
        x = (w==0).sum(axis=-1)-center
        tag = np.arange(x.size).reshape(x.shape)%11
        base = np.stack((x,-x,np.where(tag==0,40.,-40.),np.where(tag==1,40.,-40.)),axis=-1)
        return [base,base[...,[3,1,0,2]]+tag[...,None],base[...,[2,0,3,1]]-2.*tag[...,None]]
    tied = plant(vr,9.)
    tmaps = align_fit(plant(vd,9.),seeds)
    _,_,th,_ = contrast(tied,tmaps)
    ties = (tied[0]==tied[0].max(axis=-1,keepdims=True)).sum(axis=-1)>1
    mismatch = (th[0]!=th[1]) | (th[0]!=th[2])
    require('first_raw_argmax_ties',not np.any(mismatch & ~ties) and np.any(mismatch & ties),
            dict(tied=int(ties.sum()),disagree_non_tied=int(np.sum(mismatch & ~ties)),disagree_tied=int(np.sum(mismatch & ties))))
    ld = plant(vd); lr = plant(vr)
    spec,record = distill(vd,vr,ld,lr,seeds,dict(kind='finite_arithmetic_control'))
    row = record['selection']; reps = record['rule_reproducibility']
    require('permutation_and_gauge', all(r['pearson']>.99 and r['kappa']>.99 for r in reps),
            dict(reproducibility=reps,alignment=record['alignment']))
    require('planted_count',spec['symbolic_passed'] and spec['feature']=='count:vac' and row['rule_r2']>.99
            and record['best_fit_only']['fit_r2']>.99,dict(rule=spec['feature'],fit_r2=record['best_fit_only']['fit_r2'],
            rule_r2=row['rule_r2'],teacher_sd=[r['raw_sd'] for r in record['d_seed_statistics']]))
    require('exact_conventional_residual',spec['rediscovery'] and spec['fit_residual_variance']==0 and (residual_rule(vr,spec)==0).all()
            and np.array_equal(conventional_rule(vr,spec),apply_rule(vr,spec)),
            dict(fit_within_s_variance=spec['fit_within_s_variance'],fit_residual_variance=spec['fit_residual_variance'],
                 projection=spec['conventional_model']['name'],status=spec['status']))
    controls = {}
    for ctr in ('site','composition','coordination','crystal','random_affine','random_tanh'):
        m = fit_control(vd,ctr); h = hard_control(vd,m)
        controls[ctr] = dict(mass=(np.bincount(h.ravel(),minlength=4)/h.size).tolist(),
                             agrees=m['fit_mass']==(np.bincount(h.ravel(),minlength=4)/h.size).tolist())
    match = fit_control(vd,'random_affine',mass=[.1,.2,.3,.4])
    require('frozen_baseline_codes', all(row['agrees'] for row in controls.values()) and
            np.max(np.abs(np.asarray(match['fit_mass'])-match['target_mass'])) <= 1/np.prod(vd.shape[:-1])+1e-15,
            dict(modes=controls,matched=match))
    # Exercise the nonconventional model branch with an explicit signed microscopic line.
    l = feature_value(vd,'AvB_signed:x')
    alt = dict(feature='AvB_signed:x',sign=1,fit_mean=float(l.mean()),fit_sd=float(l.std()))
    proj,errors,within,_ = _fit_projection(vd,alt); alt['conventional_model'] = proj
    res = residual_rule(vr,alt)
    require('nonconventional_projection', np.max(np.abs(res-(apply_rule(vr,alt)-conventional_rule(vr,alt))))<1e-8
            and np.var(res)>.10,dict(model=proj['name'],fit_oof_mse=errors,fit_within_s_variance=within,
                                    independent_word_residual_variance=float(res.var())))
    # Null is an independently parent-shifted third seed; it cannot be rescued by changing labels on RULE.
    maps = align_fit(ld,seeds)
    _,zs,hs,_ = contrast([lr[0],lr[1],np.roll(lr[2],1,axis=0)],maps)
    pairs,passed = _repro(zs,hs,seeds)
    require('unstable_null_rejected',not passed,pairs)
    return dict(kind='nonphysical finite words; no archive, E, dynamics, or cavity',passed=all(r['passed'] for r in out.values()),
                preregistration_commit='ac5550ef2a47847f06fb9b0aa895e1c20f1ea0ef',implementation_sha256=_file_sha(__file__),
                checks=out,elapsed_seconds=time.monotonic()-start)


def _native(a):
    from hero_train import center_logits
    if a.fit.name != 'fit.npz' or a.rule.name != 'rule.npz' or a.rule.parent.resolve() != a.fit.parent.resolve():
        raise ValueError('this stage accepts only the two sibling discovery roles fit.npz and rule.npz')
    if a.ladder not in (256,1024,4096) or not a.release.is_file():
        raise ValueError('nominated registered ladder and explicit release receipt required')
    a.out.mkdir(parents=True,exist_ok=True)
    if any((a.out/name).exists() for name in ('rule-development.json','rule.json')):
        raise FileExistsError('cannot overwrite a spent distillation')
    models = [(int(m.split(':',1)[0]),Path(m.split(':',1)[1])) for m in a.model]
    seeds = [s for s,_ in models]
    if sorted(set(seeds))!=seeds:
        raise ValueError('pass distinct paired seed checkpoints in numerically increasing order')
    source = dict(ladder=a.ladder,release_sha256=_file_sha(a.release),implementation_sha256=_file_sha(__file__),
                  checkpoints=[dict(seed=s,sha256=_file_sha(p),file=p.name) for s,p in models])
    with np.load(a.fit,allow_pickle=False) as fit,np.load(a.rule,allow_pickle=False) as rule:
        parents = [np.asarray(b['parent']) for b in (fit,rule)]
        if any(p.ndim!=1 or len(np.unique(p))!=len(p) for p in parents) or len(np.intersect1d(*parents)):
            raise ValueError('overlapping/nonunique FIT/RULE parents')
        vfit,vr = fit['V'],rule['V']
        if any(v.shape!=(len(p),64,27) or v.dtype!=np.uint8 for v,p in zip((vfit,vr),parents)) or len(vfit)<a.ladder:
            raise ValueError('geometry/registered FIT ladder not available')
        ld,lr = [],[]
        for _,path in models:
            with np.load(path,allow_pickle=False) as model:
                ld.append(center_logits(model,vfit[:min(a.ladder,1024)]))
                lr.append(center_logits(model,vr))
        source['parent_order_sha256'] = [hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest() for p in parents]
        spec,report = distill(vfit,vr,ld,lr,seeds,source)
    _write_json(a.out/'rule-development.json',report)
    _write_json(a.out/'rule.json',spec)
    print(json.dumps({k:spec[k] for k in ('feature','symbolic_passed','rediscovery','residual_fraction','status')}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest='mode',required=True)
    check = modes.add_parser('controls'); check.add_argument('--out',type=Path,required=True)
    run = modes.add_parser('distill')
    for key in ('fit','rule','release','out'):
        run.add_argument('--'+key,type=Path,required=True)
    run.add_argument('--ladder',type=int,required=True); run.add_argument('--model',action='append',required=True)
    a = p.parse_args()
    if a.mode == 'distill':
        _native(a)
    else:
        receipt = finite_controls()
        _write_json(a.out,receipt)
        print(json.dumps(dict(passed=receipt['passed'],failed=[k for k,r in receipt['checks'].items() if not r['passed']],
                              elapsed_seconds=receipt['elapsed_seconds'])))
        if not receipt['passed']:
            raise SystemExit(1)
