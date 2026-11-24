"""Secondary, archive-only finite-channel bottleneck experiment; see rate/design.json."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import xlogy
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l20_operators_static import table
from l20_pairs import partition

MASKS = ('corners', 'axial')
SIZE = {'corners': 3**8, 'axial': 3**7}
GEOMETRIES = [(m, b) for b in (1, 2, 3) for m in MASKS]
EXPOSURES = (64, 256, 1024, 4096, 8000)
LAST = 32
VAL_ANCHORS = 128
LOG2 = math.log(2.)
TINY = 1.e-300
FEE = .004  # reporting/selection only: nats per achieved bit
VERSION = 1
DIGITS = np.arange(81)[:, None] // (3**np.arange(4)) % 3
_, COMPOSITION = np.unique(np.stack([(DIGITS==s).sum(1) for s in (1,2)], axis=1), axis=0, return_inverse=True)
COMPOSITION = COMPOSITION.ravel()
COARSEN = np.eye(15)[COMPOSITION]


class Pause(Exception):
    """The next indivisible operation would approach the caller's deadline."""


def budget(end, reserve=4.):
    if time.time() + reserve >= end:
        raise Pause


def read_json(path):
    return json.loads(path.read_bytes())


def json_out(path, obj):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def arrays(path, **data):
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **data)
    os.replace(tmp, path)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_rows(data, index, folds):
    """Materialize only allowed physical rows of the release's monolithic NPY member."""
    if set(folds) - {0, 1}: raise ValueError('test access forbidden')
    path = data / f'beta-{index:02}.npz'; receipt = read_json(path.with_suffix('.json'))
    sha = digest(path)
    if sha != receipt['endpoint_sha256']: raise ValueError(f'changed release: {path}')
    groups, _ = partition(index)
    ids = np.sort(np.concatenate([groups[f] for f in folds]))
    with np.load(path, allow_pickle=False) as f:
        beta = float(f['beta'])
        if str(f['axes']) != 'chain,z,y,x' or beta != float(receipt['beta']): raise ValueError('release metadata mismatch')
        keys = np.column_stack((f['chain_seed'],f['chain_id']))
        if len(np.unique(keys[ids],axis=0)) != len(ids): raise ValueError('duplicate selected lineage')
    data_rows = np.empty((len(ids),8000),np.uint8)
    with zipfile.ZipFile(path) as z, z.open('species.npy') as s:
        ver = np.lib.format.read_magic(s)
        headers = {(1,0):np.lib.format.read_array_header_1_0, (2,0):np.lib.format.read_array_header_2_0}
        shape, fortran, dtype = headers[ver](s)
        if shape != (16384,20,20,20) or fortran or dtype != np.uint8: raise ValueError('unexpected physical row format')
        start = s.tell(); pos = 0
        for stop in np.r_[np.where(np.diff(ids)!=1)[0]+1,len(ids)]:
            s.seek(start+int(ids[pos])*8000)
            target = memoryview(data_rows[pos:stop]).cast('B')
            if s.readinto(target)!=len(target): raise ValueError('truncated archive row')
            pos = stop
    if data_rows.max()>2: raise ValueError('invalid species')
    lookup = np.full(16384,-1,np.int32); lookup[ids]=np.arange(len(ids),dtype=np.int32)
    return (lookup,data_rows), dict(index=index,beta=beta,sha256=sha,selected_folds=list(folds),
            lineage_sha256={str(f):hashlib.sha256(keys[groups[f]].astype('<u8').tobytes()).hexdigest() for f in folds})


def ent(p, axis=-1):
    return -xlogy(p, p).sum(axis=axis)


def ci(x):
    x = np.asarray(x, float)
    center = x.mean()
    d = t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(center), float(center-d), float(center+d)]


def anchor_design(index):
    # Random cyclic translate of a shared random traversal, plus parent-specific stride.
    # Each prefix is marginally uniform and all 8000 centers occur once per parent.
    rng = np.random.default_rng([803112, index])
    traversal = rng.permutation(8000).astype(np.int32)
    coprime = np.array([i for i in range(1, 8000, 2) if i % 5], np.int32)
    return traversal, rng.integers(8000, size=16384, dtype=np.int32), rng.choice(coprime, 16384)


def tables(mask, b):
    vis = (list(itertools.product((-1, 1), repeat=3)) if mask == 'corners' else
           [(0, 0, 0)] + [tuple(sign * (a == i) for a in range(3)) for i in range(3) for sign in (-1, 1)])
    faces = []
    for axis, sign in itertools.product(range(3), (-1, 1)):
        square = []
        for uv in itertools.product((-1, 1), repeat=2):
            p = list(uv); p.insert(axis, sign*(2+b)); square.append(p)
        faces.append(table(np.asarray(square)))
    return table(np.asarray(vis)), faces


def extraction(state, rows, centers, lut):
    # Flattened, little-endian ternary code. Avoid a sites x lattice-volume temporary.
    out = np.zeros(centers.shape, np.uint16)
    for k in range(lut.shape[1]-1, -1, -1):
        out *= 3
        out += state[rows[:, None], lut[centers, k]]
    return out


def observations(state, rows, design, lut, start, stop):
    traversal, phases, strides = design
    a = traversal[(phases[rows, None] + strides[rows, None]*np.arange(start, stop, dtype=np.int32)) % 8000]
    lookup, data_rows = state; selected = lookup[rows]
    if selected.min()<0: raise ValueError('attempt to use an unopened parent')
    v = extraction(data_rows, selected, a, lut[0])
    ys = [extraction(data_rows, selected, a, face).astype(np.uint8) for face in lut[1]]
    return v, ys


def count_channel(state, meta, rows, design, lut, mask, b, exposure, work, end):
    # Disjoint training-parent subsets tune the Dirichlet strength. Both enter final fit.
    path = work/f'counts-{mask}-{b}.npz'
    shape = (2, SIZE[mask], 6, 81)
    n = 0
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if str(f['sha']) != meta['sha256'] or int(f['version']) != VERSION:
                raise ValueError('changed release or counts format')
            n = int(f['n']); c = f['c']
    else:
        c = np.zeros(shape, np.uint32)
    if n > exposure:
        raise ValueError('requested an earlier training exposure after advancing its counts')
    while n < exposure:
        budget(end, 7.)
        stop = min(exposure, n + 32)
        for s, group in enumerate((rows[:-1024], rows[-1024:])):
            v, ys = observations(state, group, design, lut, n, stop)
            v = v.ravel().astype(np.int32) * 81
            for j, y in enumerate(ys):
                increment = np.bincount(v + y.ravel(), minlength=shape[1]*81)
                c[s, :, j, :] += increment.reshape(-1, 81).astype(np.uint32)
        n = stop
        arrays(path, c=c, n=n, sha=meta['sha256'], version=VERSION)
    return c


def channel(c):
    # Grid validation is conditional log likelihood of Y given V,J on OTHER train parents.
    train, hold = c
    nv = train[:, 0].sum(-1, dtype=float)
    marginal = train.sum(axis=0, dtype=float) + .5
    marginal /= marginal.sum(-1, keepdims=True)
    alphas = (.5, 2., 8., 32., 128., 512., 2048., 8192., 32768.)
    scores = []
    for alpha in alphas:
        pred = train + alpha*marginal
        pred /= (nv+alpha)[:, None, None]
        np.log(pred, out=pred)
        scores.append(float(np.einsum('vjy,vjy->', hold, pred, optimize=False) / hold.sum(dtype=float)))
    alpha = alphas[int(np.argmax(scores))]
    p = c.sum(0, dtype=np.float64); marginal = p.sum(0) + .5
    marginal /= marginal.sum(-1, keepdims=True)
    nv = p[:, 0].sum(-1); p += alpha * marginal; p /= (nv+alpha)[:, None, None]
    return nv / nv.sum(), p.reshape(len(nv), -1), dict(alpha=alpha, grid=alphas, validation_logscore=scores,
            represented_v=int(np.count_nonzero(nv)), total_v=len(nv), training_parent_holdout=1024)


def onset(w, p):
    # Gram power iteration on the smaller (j,y) side; leading singular value squared.
    # This is the Hessian instability of the uninformative split, not a global transition.
    j = p.shape[1] // 81
    m = w @ p
    a = (p-m) * (np.sqrt(w/j)[:, None] / np.sqrt(np.maximum(m, TINY)))
    rng = np.random.default_rng(4403); z = rng.normal(size=(p.shape[1], 3))
    z = np.linalg.qr(z)[0]
    for _ in range(40):
        z = np.linalg.qr(a.T @ (a @ z))[0]
    az = a @ z
    vals, rot = np.linalg.eigh(az.T @ az)
    z = z @ rot[:, ::-1]; az = az @ rot[:, ::-1]; vals = vals[::-1]
    err = np.linalg.norm(a.T @ az[:, 0] - vals[0]*z[:, 0])
    # Conditional split directions in visible coordinates (zero-probability rows are inert).
    dirs = az / np.sqrt(np.maximum(w, 1.e-20))[:, None]
    dirs /= np.sqrt(np.maximum(vals, 1.e-20))
    return max(float(vals[0]), 1.e-12), np.clip(dirs, -8, 8), dict(eigenvalues=vals.tolist(), leading_residual=float(err))


def softmax(z):
    z -= z.max(axis=1, keepdims=True)
    np.exp(z, out=z); z /= z.sum(axis=1, keepdims=True)
    return z


def moments(w, p, q, logm, faces=6):
    joint = w[:, None] * q; prior = joint.sum(0)
    dec = (joint.T @ p) / np.maximum(prior, TINY)[:, None]
    # Even extinct classes receive a normalized, harmless decoder.
    dec[prior <= TINY] = np.exp(logm)
    logd = np.log(np.maximum(dec, TINY))
    gain = p @ (logd-logm).T / faces
    relevance = float(np.einsum('vh,vh->', joint, gain))
    rate = float(ent(prior) - w @ ent(q))
    return prior, dec, gain, relevance, rate


def fit(w, p, dirs, lam, scale, k, seed, warm, cp, end, deep):
    rng = np.random.default_rng(seed); logm = np.log(np.maximum(w @ p, TINY))
    phases = [max(lam, .86*scale), math.sqrt(max(lam, .86*scale)*lam), lam] if warm is None else [lam]
    phases = list(dict.fromkeys(phases))
    first = it = 0; traces = []; history = []
    if cp.exists():
        with np.load(cp, allow_pickle=False) as f:
            q = f['q']; first = int(f['phase']); it = int(f['it']); traces = json.loads(str(f['traces'])); history = f['history'].tolist()
    elif warm is None:
        q = softmax(.75*(dirs @ rng.normal(size=(dirs.shape[1], k))) + .25*rng.normal(size=(len(w), k)))
    else:
        q = np.array(warm, dtype=float, copy=True)
    def save(ph, t0):
        arrays(cp, q=q, phase=ph, it=t0, traces=json.dumps(traces), history=np.asarray(history))
    for ph in range(first, len(phases)):
        beta_cost = phases[ph]; final = ph == len(phases)-1
        limit = (240 if deep else 85) if final else (55 if deep else 24)
        # Break symmetry again when cooling: identical daughter classes cannot split in exact BA.
        if ph and not it and not history:
            noise = np.random.default_rng([seed, ph]); amp = .08 if not final else .035
            q = softmax(np.log(np.maximum(q, TINY)) + amp*(dirs @ noise.normal(size=(dirs.shape[1], k))))
        while True:
            if time.time()+5 >= end:
                save(ph, it); raise Pause
            prior, dec, gain, relevance, rate = moments(w, p, q, logm)
            obj = relevance - beta_cost*rate; history.append(obj)
            new = softmax(gain / beta_cost + np.log(np.maximum(prior, TINY)))
            tv = np.abs(q-new).sum(1) * .5; residual = float(w @ tv)
            if it >= limit or (it >= 3 and residual <= 2.e-6):
                diffs = np.diff(history)
                traces.append(dict(lambda_nats=beta_cost, iterations=it, objective_initial=history[0], objective_final=obj,
                        min_step=float(diffs.min()) if len(diffs) else 0., monotone=bool(not len(diffs) or diffs.min() >= -5.e-12),
                        weighted_BA_TV=residual, max_BA_TV=float(tv[w>0].max()), converged=residual <= 2.e-6))
                history = []; it = 0; break
            q = new; it += 1
            if it % 50 == 0: save(ph, it)
    # Unseen visible rows receive the training prior (no fitted conditional information).
    q[w == 0] = prior
    return q, dec, prior, dict(traces=traces, rate_bits=rate/LOG2, relevance_nats=relevance,
            objective_nats=relevance-lam*rate, entropy_bits=float(ent(prior)/LOG2), occupancy=prior.tolist())


def valbank(state, rows, design, lut, n):
    # 32 equally sized, disjoint groups of full independent parents; never regroup by site.
    v, faces = observations(state, rows, design, lut, 0, n)
    return v.reshape(32, 64, n), [y.reshape(32, 64, n) for y in faces]


def partner_map(index):
    rng = np.random.default_rng([289331, index]); out = np.empty((32, 64), int)
    for g in range(32):
        order = rng.permutation(64); out[g, order] = np.roll(order, -1)
    return out


def evaluate(v, ys, q, dec, prior, partner, keep=False):
    g = len(v); k = q.shape[1]; jn = len(ys); dec = dec.reshape(k, jn, -1)
    ny = dec.shape[-1]; logd = np.log(np.maximum(dec, TINY))
    # Counts use only V, not labels; the stochastic encoder draw is integrated exactly.
    vf = v.reshape(g, -1)
    frequencies = np.stack([np.bincount(x, minlength=len(q)) for x in vf]).astype(float)
    frequencies /= frequencies.sum(1, keepdims=True)
    ph = frequencies @ q; hcond = frequencies @ ent(q)
    blockrate = (ent(ph) - hcond) / LOG2
    upper = (-ph @ np.log(np.maximum(prior, TINY)) - hcond) / LOG2
    global_rate = (ent(ph.mean(0)) - hcond.mean()) / LOG2
    ce = np.zeros(g); nullce = np.zeros(g); hblock = np.zeros(g); hloo = np.zeros(g); hpool = 0.; marginal_ll = np.zeros(g)
    comp_ll = np.zeros(g); comp_null = np.zeros(g); comp_hloo = np.zeros(g); comp_hpool = 0.
    idx = np.arange(g)[:, None]
    for j, y in enumerate(ys):
        predicted = q @ logd[:, j, :]
        ce += predicted[v, y].mean(axis=(1,2)) / jn
        nullce += predicted[v, y[idx, partner]].mean(axis=(1,2)) / jn
        counts = np.stack([np.bincount(one.ravel(), minlength=ny) for one in y]).astype(float)
        counts /= counts.sum(1, keepdims=True)
        pooled = counts.mean(0); hpool += float(ent(pooled)) / jn
        hblock += ent(counts) / jn
        hloo += ent(np.maximum((g*pooled - counts)/(g-1), 0.)) / jn
        marginal_ll += counts @ np.log(np.maximum(prior @ dec[:, j, :], TINY)) / jn
        if ny == 81:
            # C=(number A, number B) is a deterministic target map. This uses the SAME
            # decoder, not a refitted baseline or a subtraction of independently fit bounds.
            coarse_dec = dec[:, j, :] @ COARSEN
            expected = q @ np.log(np.maximum(coarse_dec, TINY))
            c = COMPOSITION[y]
            comp_ll += expected[v,c].mean(axis=(1,2)) / jn
            comp_null += expected[v,c[idx,partner]].mean(axis=(1,2)) / jn
            cc = counts @ COARSEN; m = cc.mean(0)
            comp_hpool += float(ent(m)) / jn
            comp_hloo += ent(np.maximum((g*m-cc)/(g-1),0.)) / jn
    # Jackknife pseudo-observations only supply the SE; do NOT bias-correct the entropy point.
    def pooled_ci(per_group, total, leave_one):
        x = g*total - (g-1)*leave_one + per_group
        centered = x - x.mean() + total + per_group.mean()
        return ci(centered)
    rate_loo = (ent(np.maximum((g*ph.mean(0)-ph)/(g-1),0.)) - (g*hcond.mean()-hcond)/(g-1)) / LOG2
    rate_pseudo = g*global_rate-(g-1)*rate_loo
    result = dict(bound_nats=pooled_ci(ce, hpool, hloo), null_nats=pooled_ci(nullce, hpool, hloo),
            block_concavity_nats=ci(hblock+ce), paired_minus_null_nats=ci(ce-nullce),
            logscore_gain_over_train_marginal_nats=ci(ce-marginal_ll), decoder_logscore=ci(ce),
            empirical_rate_bits=ci(rate_pseudo-rate_pseudo.mean()+global_rate), reference_rate_upper_bits=ci(upper),
            block_rate_bits=ci(blockrate), entropy_H_bits=float(ent(ph.mean(0))/LOG2),
            mean_conditional_entropy_H_bits=float(hcond.mean()/LOG2), effective_classes=float(np.exp(ent(ph.mean(0)))),
            pooled_target_entropy_nats=hpool, parents=2048, parent_blocks=g, anchors_per_parent=v.shape[-1])
    if ny == 81:
        result['face_mechanism'] = dict(composition_nats=pooled_ci(comp_ll, comp_hpool, comp_hloo),
                arrangement_given_composition_nats=pooled_ci(ce-comp_ll,hpool-comp_hpool,hloo-comp_hloo),
                null_composition_nats=pooled_ci(comp_null,comp_hpool,comp_hloo),
                null_arrangement_given_composition_nats=pooled_ci(nullce-comp_null,hpool-comp_hpool,hloo-comp_hloo))
    if keep:
        result['blocks'] = dict(logscore=ce.tolist(), shuffled_logscore=nullce.tolist(), entropy=hblock.tolist(),
                                rate_bits=blockrate.tolist(), reference_upper_bits=upper.tolist())
    return result


def calibration(v, partner):
    # Use an actual archived visible species as a known binary source, and independently
    # corrupt its label at each row. This is label calibration, not a physical generator.
    bit = (v % 3 == 1).astype(np.uint8); prob = float(bit.mean()); prior = np.array([1-prob, prob]); q = np.eye(2)
    rng = np.random.default_rng(331156); rows = []
    for flip in (0., .15, .5):
        d = np.array([[1-flip, flip], [flip, 1-flip]])
        ys = [(bit ^ (rng.random(bit.shape) < flip)).astype(np.uint8) for _ in range(6)]
        row = evaluate(bit, ys, q, np.repeat(d[:, None, :],6,axis=1), prior, partner)
        independent = evaluate(bit, ys, np.ones((2,1)), np.tile(prior@d,(1,6,1)), np.ones(1), partner)
        # Direct deterministic-channel/null limits plus BA on the exact corrupted law.
        exact = float(ent(prior @ d) - ent(d[0])); logm = np.log(np.maximum(prior@d,TINY))
        qfit = np.array([[.85,.15],[.15,.85]])
        for _ in range(100):
            pi, _, gain, rel, ri = moments(prior, d, qfit, logm, faces=1)
            qfit = softmax(gain/.08 + np.log(np.maximum(pi, TINY)))
        _, _, _, rel, ri = moments(prior, d, qfit, logm, faces=1)
        rows.append(dict(flip=flip, source_A_fraction=prob, known_identity_MI_nats=exact,
                known_identity_rate_bits=float(ent(prior)/LOG2), identity_bound=row['bound_nats'],
                measured_identity_rate=row['empirical_rate_bits'], constant_rate=independent['empirical_rate_bits'],
                constant_bound=independent['bound_nats'], BA_lambda=.08, BA_rate_bits=ri/LOG2, BA_relevance_nats=rel))
    return rows


def utility(row):
    val = row['validation']; return val['bound_nats'][1] - FEE * max(val['empirical_rate_bits'][0], 0.)


def frontier(rows):
    best = -np.inf; out = []
    for r in sorted(rows, key=lambda x: (x['validation']['empirical_rate_bits'][0], -x['validation']['bound_nats'][0])):
        value = r['validation']['bound_nats'][0]
        if value > best:
            best = value; out.append(r)
    return out


def plan(r, previous):
    if r < 6:
        mask,b = GEOMETRIES[r]
        return dict(mask=mask, b=b, exposure=16, reason='common pilot: crossed mask, buffer, alphabet and onset-scaled geometric costs',
                    trials=[dict(k=k, ratio=x, seed=41, start=None) for k in (2,4,8) for x in (.80,.28,.098)])
    b = 1+(r-6)%3; level = (r-6)//3; exposure = EXPOSURES[min(level,len(EXPOSURES)-1)]
    candidates = [row for doc in previous if doc['plan']['b']==b for row in doc['models']]
    depth = max(x['exposure'] for x in candidates); last = [x for x in candidates if x['exposure']==depth]
    # Keep range strata even if scores are unresolved; don't select a range for being easy.
    front = frontier(last); champion = max(front, key=utility); mask = champion['mask']
    near = [x for x in last if x['mask']==mask]; front = frontier(near)
    # Local sensitivity: if statistically compatible, spend some resolution on the largest
    # score slope around the utility knee; never demand significance for further exposure.
    competing = [x for x in front if x['validation']['bound_nats'][2] - FEE*x['validation']['empirical_rate_bits'][0] >= utility(champion)]
    if level % 3 == 1 and len(competing)>1:
        def slope(x):
            others = [z for z in front if abs(z['validation']['empirical_rate_bits'][0]-x['validation']['empirical_rate_bits'][0])>.015]
            if not others: return 0.
            z = min(others,key=lambda a: abs(a['validation']['empirical_rate_bits'][0]-x['validation']['empirical_rate_bits'][0]))
            return abs(z['validation']['bound_nats'][0]-x['validation']['bound_nats'][0])/abs(z['validation']['empirical_rate_bits'][0]-x['validation']['empirical_rate_bits'][0])
        champion = max(competing, key=lambda x:(slope(x),utility(x)))
    center = champion['ratio']; k = champion['k']; factor = max(1.09, 1.85**(1/(1+.25*level)))
    trials = [dict(k=k,ratio=center,seed=41,start=champion['id']), dict(k=k,ratio=center,seed=193+level,start=None),
              dict(k=k,ratio=max(.0003,center/factor),seed=241+level,start=None),
              dict(k=k,ratio=min(1.12,center*factor),seed=241+level,start=None)]
    trials += [dict(k=kk,ratio=center,seed=241+level,start=None) for kk in (2,4,8) if kk!=k]
    return dict(mask=mask,b=b,exposure=exposure,previous=champion['id'],previous_exposure=depth,
                eligible_ids=[x['id'] for x in competing],reason='within-range Pareto utility knee or interval-compatible sensitive slope; new anchors until all 8000, matched continuation vs independent annealed seed, flanking cost and alphabet controls',trials=trials)


def plot(docs, path, transfer=None):
    fig, axs = plt.subplots(1, 3, figsize=(12,4), sharey=True, layout='constrained')
    for b, ax in enumerate(axs,1):
        for mask, color, mark in [('corners','#155a75','o'),('axial','#bd642d','^')]:
            rows = [r for d in docs for r in d['models'] if r['b']==b and r['mask']==mask]
            if not rows: continue
            for r in rows:
                v = r['validation']; x = v['empirical_rate_bits'][0]
                ax.scatter(x,v['bound_nats'][0],color=color,marker=mark,s=9,alpha=.24)
            fr = frontier(rows); x = [r['validation']['empirical_rate_bits'][0] for r in fr]
            vals = np.array([r['validation']['bound_nats'] for r in fr]); nu = [r['validation']['null_nats'][0] for r in fr]
            ax.plot(x,vals[:,0],marker=mark,color=color,ms=3,label=mask+' val'); ax.fill_between(x,vals[:,1],vals[:,2],color=color,alpha=.13)
            ax.plot(x,nu,':',color=color,label=mask+' paired null')
        if transfer and transfer['b']==b:
            for row in transfer['temperatures']:
                s=row['validation']; ax.scatter(s['empirical_rate_bits'][0],s['bound_nats'][0],marker='x',s=45,label=f'frozen β={row["beta"]:g}')
        ax.axhline(0,color='gray',lw=.6); ax.set_title(f'buffer {b}, face radius {b+2}'); ax.set_xlabel('achieved rate (bits)')
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    axs[0].set_ylabel('predictive lower score (nats / face)'); fig.suptitle('Archive multiview bottleneck · validation scan, nominal parent intervals')
    tmp = path.with_suffix('.tmp.png'); fig.savefig(tmp,dpi=140); plt.close(fig); os.replace(tmp,path)


def discover(args):
    out,work,r,end = args.out,args.work,args.round,args.end_epoch
    done = out/f'round-{r:04}.json'; partial = out/f'round-{r:04}.partial.json'
    if done.exists(): return
    if (out/'freeze.json').exists(): raise ValueError('cold selection already frozen')
    if not 0 <= r <= LAST: raise ValueError(f'finite campaign ends at round {LAST}')
    docs = [read_json(out/f'round-{i:04}.json') for i in range(r)]
    if partial.exists(): doc = read_json(partial); stage=doc['plan']
    else:
        stage=plan(r,docs); doc=dict(version=VERSION,round=r,mode='descriptive validation discovery',plan=stage,models=[],stop=False,last_round=LAST)
        json_out(partial,doc)
    mask,b,n = stage['mask'],stage['b'],stage['exposure']; budget(end,7.)
    state,meta=source_rows(args.data,9,(0,1)); groups,_=partition(9); design=anchor_design(9); lut=tables(mask,b)
    if 'source' in doc and doc['source']['sha256']!=meta['sha256']: raise ValueError('changed cold snapshot')
    doc['source']=meta
    c=count_channel(state,meta,groups[0],design,lut,mask,b,n,work,end); budget(end)
    w,p,doc['shrinkage']=channel(c); del c
    critical,dirs,doc['onset']=onset(w,p); doc['onset']['lambda_local']=critical
    v,ys=valbank(state,groups[1],design,lut,VAL_ANCHORS); del state
    partner=partner_map(9)
    if not r and 'controls' not in doc: doc['controls']=calibration(v,partner)
    json_out(partial,doc)
    for i,spec in enumerate(stage['trials']):
        if i<len(doc['models']): continue
        budget(end); key=f'r{r:04}-m{i:02}'; cp=work/f'{key}-iter.npz'; model=work/f'{key}.npz'; warm=None
        if spec['start']:
            with np.load(work/(spec['start']+'.npz'),allow_pickle=False) as f: warm=f['q']
        lam=critical*spec['ratio']; q,dec,prior,train=fit(w,p,dirs,lam,critical,spec['k'],spec['seed'],warm,cp,end,r>=6)
        val=evaluate(v,ys,q,dec,prior,partner,keep=True)
        arrays(model,q=q,dec=dec,prior=prior)
        row=dict(id=key,mask=mask,b=b,exposure=n,**spec,lambda_nats=lam,train=train,validation=val,model_sha256=digest(model))
        doc['models'].append(row); json_out(partial,doc); cp.unlink(missing_ok=True)
    if r>=6:
        a,z=doc['models'][:2]; doc['continuation_minus_restarted']=dict(objective_nats=a['train']['objective_nats']-z['train']['objective_nats'],
             paired_validation_logscore_nats=ci(np.subtract(a['validation']['blocks']['logscore'],z['validation']['blocks']['logscore'])))
    doc['stop']=r==LAST; doc['next_round']=None if r==LAST else r+1
    plot(docs+[doc],out/'frontier.png'); json_out(done,doc); partial.unlink(missing_ok=True)


def confirm(args):
    out,work=args.out,args.work; path=out/'freeze.json'; result=out/'confirmation.json'
    if result.exists(): return
    if path.exists(): lock=read_json(path)
    else:
        files=sorted(p for p in out.glob('round-*.json') if not p.name.endswith('.partial.json'))
        docs=[read_json(p) for p in files]; rows=[r for d in docs for r in d['models']]
        if not rows: raise ValueError('complete at least one cold discovery round before freezing')
        winner=max(frontier(rows),key=utility)
        lock=dict(version=VERSION,id=winner['id'],model_sha256=winner['model_sha256'],mask=winner['mask'],b=winner['b'],
                  source_sha256=docs[0]['source']['sha256'],last_completed=docs[-1]['round'],criterion='Pareto-eligible highest nominal lower score minus .004 nats per achieved bit; retrospective exploratory selection, not a gate')
        json_out(path,lock)
    model=out/'frozen-encoder-decoder.npz'
    if not model.exists():
        src=work/(lock['id']+'.npz')
        if digest(src)!=lock['model_sha256']: raise ValueError('selected private weights changed')
        tmp=model.with_suffix('.npz.tmp'); tmp.write_bytes(src.read_bytes()); os.replace(tmp,model)
    if digest(model)!=lock['model_sha256']: raise ValueError('frozen weights changed')
    with np.load(model,allow_pickle=False) as f: q,dec,prior=f['q'],f['dec'],f['prior']
    partial=out/'confirmation.partial.json'; doc=read_json(partial) if partial.exists() else dict(**lock,temperatures=[],mode='unchanged descriptive transfer, validation parents only')
    for idx in (9,6,0):
        if any(x['index']==idx for x in doc['temperatures']): continue
        budget(args.end_epoch,8.); state,meta=source_rows(args.data,idx,(1,)); groups,_=partition(idx)
        if idx==9 and meta['sha256']!=lock['source_sha256']: raise ValueError('cold release changed')
        v,ys=valbank(state,groups[1],anchor_design(idx),tables(lock['mask'],lock['b']),256); del state
        val=evaluate(v,ys,q,dec,prior,partner_map(idx),keep=True)
        doc['temperatures'].append(dict(index=idx,beta=meta['beta'],source_sha256=meta['sha256'],validation=val)); json_out(partial,doc)
    docs=[read_json(out/f'round-{i:04}.json') for i in range(lock['last_completed']+1)]
    plot(docs,out/'transfer.png',doc); doc['stop']=True; json_out(result,doc); partial.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','work','out'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--round',type=int,required=True); parser.add_argument('--end-epoch',type=float,required=True)
    parser.add_argument('--mode',choices=('discover','confirm'),default='discover'); a=parser.parse_args()
    a.out.mkdir(exist_ok=True,parents=True); a.work.mkdir(exist_ok=True,parents=True)
    def alarm(signum, frame):
        raise Pause
    signal.signal(signal.SIGALRM, alarm)
    try:
        budget(a.end_epoch)
        signal.setitimer(signal.ITIMER_REAL, max(.001,a.end_epoch-time.time()-1.))
        (discover if a.mode=='discover' else confirm)(a)
    except Pause: print('Deadline guard: durable prefix; resume the same stage.',flush=True)
    finally: signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__': main()
