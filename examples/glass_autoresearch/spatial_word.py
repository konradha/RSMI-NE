"""Label-free reverse spatial bottleneck. See space-design.json; no outcome access."""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'

import numpy as np
from scipy.stats import t

SEEDS = (260511, 260512)
ENC = ('w1', 'b1', 'w2', 'b2')
METRICS = ('hard_native', 'hard_broken', 'hard_difference', 'soft_native', 'soft_broken', 'soft_difference')
BITS = 2 * ((np.arange(4096)[:, None] >> np.arange(12)) & 1).astype('f4') - 1
CUTOFF = datetime.fromisoformat('2026-11-24T21:00:25+00:00')


def init_encoder(seed):
    rng = np.random.default_rng(seed)
    def weight(a, b):
        r = np.sqrt(6 / (a + b))
        return rng.uniform(-r, r, (a, b)).astype('f4')
    return dict(w1=weight(386, 32), b1=np.zeros(32, 'f4'), w2=weight(32, 4), b2=np.zeros(4, 'f4'))


def model(seed):
    return dict(**init_encoder(seed), word_table=np.zeros((4096, 4), 'f4'), bit_weight=np.zeros((12, 4), 'f4'))


def signed(packed):
    if packed.shape[-1] != 49 or packed.dtype != np.uint8:
        raise ValueError('expected uint8 packed 386-bit shell')
    return 2 * np.unpackbits(packed, axis=-1, count=386, bitorder='little').astype('f4') - 1


def softmax(a):
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def forward(m, x):
    h = np.tanh(x @ m['w1'] + m['b1'])
    return h, softmax(h @ m['w2'] + m['b2'])


def apply(m, packed):
    """Frozen hard encoder. Chunk large maps; keep original leading shape."""
    flat = packed.reshape(-1, 49)
    out = np.empty(len(flat), 'u1')
    for i in range(0, len(flat), 2048):
        h = np.tanh(signed(flat[i:i+2048]) @ m['w1'] + m['b1'])
        out[i:i+2048] = (h @ m['w2'] + m['b2']).argmax(axis=-1)
    return out.reshape(packed.shape[:-1])


class Opportunities:
    """Compact parent->stratum indices. Each donor is uniform over other parents."""
    def __init__(self, stratum, eligible, minimum):
        p, a = stratum.shape
        self.a = a
        self.eligible = np.asarray(eligible).reshape(-1)
        if len(np.unique(self.eligible)) != len(self.eligible):
            raise ValueError('duplicate eligible stratum')
        self.sid = np.full(p*a, -1, np.int32)
        starts = np.zeros((len(self.eligible), p), np.int32)
        self.count = np.zeros_like(starts)
        self.ranks = np.zeros_like(starts)
        self.parents = np.zeros_like(starts)
        self.nparent = np.zeros(len(self.eligible), np.int32)
        chunks = []
        available = []
        offset = 0
        for s, key in enumerate(self.eligible):
            ix = np.flatnonzero(stratum.reshape(-1) == key)
            self.sid[ix] = s
            counts = np.bincount(ix // a, minlength=p)
            self.count[s] = counts
            starts[s] = offset + np.cumsum(counts) - counts
            live = np.flatnonzero(counts)
            self.nparent[s] = len(live)
            self.parents[s, :len(live)] = live
            self.ranks[s, live] = np.arange(len(live))
            chunks.append(ix)
            offset += len(ix)
            if len(live) >= minimum:
                available.append(ix)
        self.starts = starts
        self.pool = np.concatenate(chunks)
        self.allowed = np.concatenate(available) if available else np.empty(0, np.int64)
        self.summary = {'eligible_anchors': int(offset), 'all_anchors': p*a, 'supported_anchors': len(self.allowed),
                        'stratum_keys': self.eligible.tolist(), 'parents_per_stratum': self.nparent.tolist(),
                        'opportunities_per_stratum': self.count.sum(axis=1).tolist()}

    def donors(self, focal, k, rng):
        # Sequential uniform sampling without replacement by skipping sorted exclusions.
        s = self.sid[focal]
        rank = self.ranks[s, focal // self.a]
        n = self.nparent[s]
        excluded = np.empty((len(focal), k+1), np.int32)
        excluded[:, 0] = rank
        cols = []
        for j in range(1, k+1):
            r = rng.integers(n-j, dtype=np.int32)
            for v in np.sort(excluded[:, :j], axis=1).T:
                r += r >= v
            excluded[:, j] = r
            p = self.parents[s, r]
            cols.append(self.pool[self.starts[s,p] + rng.integers(self.count[s,p], dtype=np.int32)])
        return np.column_stack(cols)


class Trainer:
    def __init__(self, m, seed):
        self.m = m
        self.rng = np.random.default_rng([seed, 331])
        self.mom = {k: np.zeros_like(v) for k, v in m.items()}
        self.vel = {k: np.zeros_like(v) for k, v in m.items()}
        self.step = 0
        self.donor_counts = None
        self.focal_counts = None

    def update(self, flat_shell, flat_word, bank):
        ids = self.rng.choice(bank.allowed, size=128)
        donors = bank.donors(ids, 7, self.rng)
        if self.donor_counts is None:
            self.donor_counts = np.zeros(len(flat_word) // bank.a, np.int64)
            self.focal_counts = np.zeros_like(self.donor_counts)
        np.add.at(self.donor_counts, donors.ravel() // bank.a, 1)
        np.add.at(self.focal_counts, ids // bank.a, 1)
        v = flat_word[np.column_stack([ids, donors])]
        m = self.m
        x = signed(flat_shell[ids]); h, q = forward(m, x)
        c = m['word_table'][v] + BITS[v] @ m['bit_weight']
        logits = (c * q[:,None,:]).sum(axis=-1)
        delta = softmax(logits); delta[:,0] -= 1; delta /= len(ids)
        dq = (delta[:,:,None] * c).sum(axis=1)
        dz = q * (dq - (dq*q).sum(axis=1, keepdims=True))
        dh = (dz @ m['w2'].T) * (1 - h*h)
        dc = (delta[:,:,None] * q[:,None,:]).reshape(-1, 4)
        v = v.ravel()
        tab = np.column_stack([np.bincount(v, weights=dc[:,j], minlength=4096) for j in range(4)]).astype('f4')
        grads = dict(w1=x.T @ dh, b1=dh.sum(axis=0), w2=h.T @ dz, b2=dz.sum(axis=0),
                     word_table=tab, bit_weight=BITS[v].T @ dc)
        self.step += 1
        for k, g in grads.items():
            self.mom[k] *= .9; self.mom[k] += .1*g
            self.vel[k] *= .999; self.vel[k] += .001*g*g
            m[k] -= .001 * (self.mom[k] / (1-.9**self.step)) / (np.sqrt(self.vel[k] / (1-.999**self.step)) + 1e-8)


def reuse(counts):
    n = counts.sum()
    return {'draws': int(n), 'distinct_parents': int(np.count_nonzero(counts)), 'max_draws_per_parent': int(counts.max(initial=0)),
            'kish_distinct_parent_equivalent_not_a_sample_size': float(n*n / np.dot(counts, counts)) if n else 0.}


def score_block(model, shell, word, stratum, eligible, seed):
    """Stateless, within-block-only retrieval; caller must supply DISJOINT groups of distinct parents.

    Native and broken log probabilities returned as gains over uniform log(1/8).
    `sums` contains whole-block totals in METRICS order; divide by `count` across groups.
    A different independent parent supplies the broken candidate and each distractor.
    No physical labels, read/write access or model fitting in this function.
    """
    bank = Opportunities(stratum, eligible, 9)
    ids = bank.allowed
    if not len(ids):
        return dict(sums=[0.]*6, count=0, support=bank.summary, donor_reuse=reuse(np.zeros(len(word), np.int64)))
    d = bank.donors(ids, 8, np.random.default_rng(seed))
    words = word.reshape(-1)[np.column_stack([ids, d])]
    c = model['word_table'] + BITS @ model['bit_weight']
    _, q = forward(model, signed(shell.reshape(-1,49)[ids]))
    cw = c[words]
    soft = np.einsum('nck,nk->nc', cw, q)
    hard = cw[np.arange(len(q))[:,None], np.arange(9), q.argmax(axis=1)[:,None]]
    sums = []
    for score in [hard, soft]:
        # Native in col 0, shared distractors 1:8, broken in 8.
        decoys = np.logaddexp.reduce(score[:,1:8].astype('f8'), axis=1)
        n = np.log(8.) - np.logaddexp(0, decoys - score[:,0]); b = np.log(8.) - np.logaddexp(0, decoys - score[:,8])
        sums.extend([n.sum(), b.sum(), (n-b).sum()])
    return dict(sums=sums, count=len(ids), support=bank.summary,
                donor_reuse=reuse(np.bincount(d.ravel() // bank.a, minlength=len(word))))


def interval(totals, counts):
    g = len(counts); n = counts.sum()
    if not n or g < 2:
        return {'estimate': None, 'se': None, 'ci95': [None,None], 'mde80': None}
    mu = totals.sum() / n
    se = np.linalg.norm(totals - counts*mu) * np.sqrt(g/(g-1)) / n
    crit = t.ppf(.975, g-1)
    return {'estimate': float(mu), 'se': float(se), 'ci95': [float(mu-crit*se), float(mu+crit*se)], 'mde80': float((crit+.842)*se)}


def grouped_scores(m, data, eligible):
    p = len(data['word'])
    if p % 32:
        raise ValueError('role must partition into complete disjoint groups of 32 parents')
    groups = np.random.default_rng(88001).permutation(p).reshape(-1, 32)
    blocks = [score_block(m, data['shell'][ix], data['word'][ix], data['stratum'][ix], eligible, [88001,g])
              for g, ix in enumerate(groups)]
    return blocks


def arrays(blocks):
    return np.array([b['sums'] for b in blocks]), np.array([b['count'] for b in blocks])


def summarize(totals, counts):
    return dict(zip(METRICS, [interval(s, counts) for s in totals.T]))


def rates(codes, st, eligible):
    counts = np.array([np.bincount(codes[st == s], minlength=4) for s in eligible]); n = counts.sum()
    if not n:
        return dict(K=4, marginal=None, conditional=None)
    def ent(x):
        z = x[x>0] / x.sum(); return float(-z @ np.log(z))
    return dict(K=4, marginal=ent(counts.sum(axis=0)), conditional=sum(row.sum()*ent(row) for row in counts if row.sum())/n,
                counts=counts.sum(axis=0).tolist(), counts_by_stratum=counts.tolist())


def hash_model(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_free(seed, broken=False):
    rng = np.random.default_rng(seed)
    p,a = 512,32
    latent = rng.integers(2, size=(p,a,2), dtype='u1')
    vlatent = rng.integers(2, size=(p,a,2), dtype='u1') if broken else latent
    st = rng.integers(8, size=(p,a))
    x = rng.integers(2, size=(p,a,386), dtype='u1')
    v = rng.integers(2, size=(p,a,12), dtype='u1')
    for j in range(2):
        x[:,:,j*32:(j+1)*32] = latent[:,:,j,None] ^ (rng.random((p,a,32)) < .1)
        v[:,:,j*4:(j+1)*4] = vlatent[:,:,j,None] ^ (rng.random((p,a,4)) < .1)
    return dict(shell=np.packbits(x, axis=-1, bitorder='little'), word=(v @ (1 << np.arange(12))).astype('u2'), stratum=st)


def dump(path, data):
    if path.exists():
        raise FileExistsError(path)
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def control(public):
    train, dev, broken = archive_free(79700), archive_free(79701), archive_free(79702, True)
    eligible = np.arange(8); bank = Opportunities(train['stratum'], eligible, 8)
    m = model(SEEDS[0]); opt = Trainer(m, SEEDS[0]); start = time.perf_counter()
    for _ in range(1024):
        opt.update(train['shell'].reshape(-1,49), train['word'].ravel(), bank)
    seconds = time.perf_counter() - start
    report = dict(status='completed synthetic only', native_access=False, train_seed=79700, eval_seeds=[79701,79702],
                  encoder_seed=SEEDS[0], fit_seconds=seconds, fit_support=bank.summary,
                  focal_reuse=reuse(opt.focal_counts), donor_reuse=reuse(opt.donor_counts))
    for key, data in [('true',dev), ('independent',broken)]:
        start = time.perf_counter(); blocks = grouped_scores(m, data, eligible); tot,n = arrays(blocks)
        report[key] = dict(metrics=summarize(tot,n), seconds=time.perf_counter()-start, groups=len(n), supported=int(n.sum()),
                           eligible=int(sum(b['support']['eligible_anchors'] for b in blocks)), rate=rates(apply(m,data['shell']),data['stratum'],eligible))
    pos = report['true']['metrics']['hard_difference']; neg = report['independent']['metrics']['hard_difference']
    report['pass'] = bool(pos['ci95'][0] > .05 and abs(neg['estimate']) <= max(.02,2.5*neg['se']))
    dump(public/'space-control.json', report)
    print(json.dumps({k: report[k] for k in ['pass','fit_seconds']}, sort_keys=True))
    return report['pass']


def load_role(work, role, n):
    # No arbitrary role command-line forwarding: only fit/dev even in evaluation helpers.
    if role not in ('fit', 'dev'):
        raise ValueError(role)
    with np.load(work/f'unlabeled-{role}.npz', allow_pickle=False) as f:
        if set(f.files) != {'parent','word','stratum','shell'}:
            raise ValueError('input firewall rejected keys')
        d = dict(f)
    if (d['parent'].shape[0] != n or d['word'].shape != (n,128) or d['stratum'].shape != (n,128)
            or d['shell'].shape != (n,128,49) or d['shell'].dtype != np.uint8 or not np.issubdtype(d['word'].dtype,np.integer)
            or d['word'].min() < 0 or d['word'].max() >= 4096 or len(np.unique(d['parent'],axis=0)) != n):
        raise ValueError('invalid lineage, word range or input layout')
    return d


def run(work, public):
    receipt = json.loads((public/'space-control.json').read_text())
    if not receipt['pass'] or datetime.now(timezone.utc) >= CUTOFF:
        raise RuntimeError('control/time gate closed')
    fit = load_role(work,'fit',2048); dev = load_role(work,'dev',512); eligible = np.load(work/'eligible.npy', allow_pickle=False)
    if len(np.unique(np.concatenate([fit['parent'],dev['parent']]), axis=0)) != 2560:
        raise ValueError('cross-role parent overlap')
    curves = {}; report = dict(status='label-free development only', models={}, ladder={}, input_keys=['parent','word','stratum','shell'])
    for seed in SEEDS:
        m = init_encoder(seed); path = work/f'space-random-s{seed}.npz'; save(path, **m)
        hard = apply(m,dev['shell']); save(work/f'space-random-s{seed}-dev-hard.npz',hard=hard)
        report['models'][path.name] = dict(sha256=hash_model(path), dev_rate=rates(hard,dev['stratum'],eligible))
    for n in (512,2048):
        bank = Opportunities(fit['stratum'][:n], eligible, 8)
        report['ladder'][str(n)] = dict(fit_support=bank.summary, checkpoints={})
        if not len(bank.allowed):
            raise ValueError('no supported eligible training anchors')
        for seed in SEEDS:
            m = model(seed); opt = Trainer(m,seed); elapsed = 0.
            for u in (256,512,1024):
                if datetime.now(timezone.utc) >= CUTOFF:
                    raise RuntimeError('discovery cutoff')
                begin = time.perf_counter()
                for _ in range(opt.step, u):
                    opt.update(fit['shell'][:n].reshape(-1,49),fit['word'][:n].ravel(),bank)
                elapsed += time.perf_counter()-begin
                path = work/f'space-k4-n{n}-s{seed}-u{u}.npz'; save(path, **m)
                begin = time.perf_counter(); blocks = grouped_scores(m, dev, eligible); totals, counts = arrays(blocks)
                if 'dev_group_support' not in report:
                    report['dev_group_support'] = [dict(count=b['count'], support=b['support'], donor_reuse=b['donor_reuse']) for b in blocks]
                hard = apply(m,dev['shell']); save(work/f'{path.stem}-dev-hard.npz',hard=hard)
                report['models'][path.name] = dict(sha256=hash_model(path), total_fit_seconds=elapsed,
                    dev_seconds=time.perf_counter()-begin, metrics=summarize(totals,counts),dev_rate=rates(hard,dev['stratum'],eligible),
                    fit_focal_reuse=reuse(opt.focal_counts),fit_donor_reuse=reuse(opt.donor_counts))
                curves[n,seed,u] = totals
                if n == 2048 and u == 1024:
                    save(work/f'{path.stem}-optimizer.npz', **{'mom_'+k:v for k,v in opt.mom.items()},
                         **{'vel_'+k:v for k,v in opt.vel.items()}, rng=json.dumps(opt.rng.bit_generator.state),
                         focal=opt.focal_counts, donors=opt.donor_counts)
        for u in (256,512,1024):
            mean = np.mean([curves[n,seed,u] for seed in SEEDS], axis=0)
            row = dict(ensemble_mean_of_paired_seed_scores=summarize(mean,counts))
            if u>256:
                prev = np.mean([curves[n,seed,u//2] for seed in SEEDS], axis=0)
                row['paired_update_gain'] = summarize(mean-prev, counts)
            if n>512:
                small = np.mean([curves[512,seed,u] for seed in SEEDS], axis=0)
                row['paired_parent_ladder_gain'] = summarize(mean-small, counts)
            report['ladder'][str(n)]['checkpoints'][str(u)] = row
    gain = report['ladder']['2048']['checkpoints']['1024']['paired_update_gain']['hard_difference']['ci95'][0]
    report['extension_dev_trigger'] = bool(gain is not None and gain>0); report['extension_authorized'] = False
    report['interpretation'] = 'Retrieval gain in nats relative to uniform eight-way choice; difference = paired native minus other-parent broken. Not MI, not physical outcome log score. Effective independent units for dev uncertainty are 16 disjoint parent groups, not anchors, donor counts, or starts.'
    dump(public/'space-dev.json',report)


def extend(work, public):
    """One diagnostic continuation; never replace the lock's primary checkpoints."""
    prior = json.loads((public/'space-dev.json').read_text())
    names = [f'space-k4-n2048-s{s}-u1024.npz' for s in SEEDS]
    reserve = 30 + 3*sum(prior['models'][name]['total_fit_seconds'] + 2*prior['models'][name]['dev_seconds'] for name in names)
    if not prior['extension_dev_trigger'] or (CUTOFF-datetime.now(timezone.utc)).total_seconds() < reserve:
        raise RuntimeError('registered extension trigger/time gate not satisfied')
    fit = load_role(work,'fit',2048); dev = load_role(work,'dev',512)
    if len(np.unique(np.concatenate([fit['parent'],dev['parent']]), axis=0)) != 2560:
        raise ValueError('cross-role parent overlap')
    eligible = np.load(work/'eligible.npy', allow_pickle=False)
    bank = Opportunities(fit['stratum'], eligible, 8)
    flat = fit['shell'].reshape(-1,49); words = fit['word'].ravel()
    before, after = [], []
    report = dict(status='one authorized diagnostic doubling only; primary unchanged', models={})
    for name, seed in zip(names, SEEDS):
        path = work/name
        if hash_model(path) != prior['models'][name]['sha256']:
            raise ValueError('primary checkpoint digest mismatch')
        with np.load(path, allow_pickle=False) as f:
            m = dict(f)
        opt = Trainer(m,seed)
        with np.load(work/f'{path.stem}-optimizer.npz', allow_pickle=False) as f:
            opt.mom = {k:f['mom_'+k] for k in m}; opt.vel = {k:f['vel_'+k] for k in m}
            opt.rng.bit_generator.state = json.loads(str(f['rng'].item()))
            opt.focal_counts = f['focal']; opt.donor_counts = f['donors']
        opt.step = 1024
        old, counts = arrays(grouped_scores(m,dev,eligible)); before.append(old)
        start = time.perf_counter()
        while opt.step < 2048:
            if opt.step % 32 == 0 and datetime.now(timezone.utc) >= CUTOFF:
                raise RuntimeError('discovery cutoff')
            opt.update(flat, words, bank)
        extra = time.perf_counter() - start
        target = work/f'space-k4-n2048-s{seed}-u2048.npz'; save(target, **m)
        hard = apply(m,dev['shell']); save(work/f'{target.stem}-dev-hard.npz',hard=hard)
        new, counts = arrays(grouped_scores(m,dev,eligible)); after.append(new)
        report['models'][target.name] = dict(sha256=hash_model(target), extra_fit_seconds=extra,
            total_fit_seconds=extra+prior['models'][name]['total_fit_seconds'], metrics=summarize(new,counts),
            dev_rate=rates(hard,dev['stratum'],eligible),fit_focal_reuse=reuse(opt.focal_counts),fit_donor_reuse=reuse(opt.donor_counts))
    mean = np.mean(after,axis=0)
    report['ensemble_mean_of_paired_seed_scores'] = summarize(mean,counts)
    report['paired_update_gain'] = summarize(mean-np.mean(before,axis=0),counts)
    lower = report['paired_update_gain']['hard_difference']['ci95'][0]
    report['still_rising_at_compute_cap'] = bool(lower is not None and lower>0)
    dump(public/'space-extension.json',report)


def save(path, **kw):
    with path.open('xb') as f:
        np.savez(f,**kw)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action',choices=['control','released-fit-dev','released-extension']); ap.add_argument('--public',required=True,type=Path); ap.add_argument('--work',type=Path)
    args = ap.parse_args()
    if args.action == 'control':
        raise SystemExit(0 if control(args.public) else 1)
    if args.work is None:
        ap.error('native actions require --work')
    (extend if args.action == 'released-extension' else run)(args.work,args.public)
