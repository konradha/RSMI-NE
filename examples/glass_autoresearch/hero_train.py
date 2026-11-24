"""Forward HERO: exact categorical expected-hard InfoNCE; fit/tune only."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student

K, C, G, WIDTH = 4, 8, 8, 16
SEEDS = (6101, 6102, 6103, 6104, 6105)
ARMS = ('affine', 'tanh')
LAWS = ('uniform', 'matched')
BASELINES = ('site', 'composition', 'coordination', 'crystal', 'random_affine', 'random_tanh')
HERE = Path(__file__).resolve().parents[2]
PUBLIC = HERE/'results/l20-autoresearch/hero-central'
WORK = Path.home()/'rsmi-work/l20-autoresearch-1795520725/hero-central'
LOG_C = math.log(C)
CELLS = np.arange(27).reshape(3, 3, 3)
EDGE_L = np.concatenate([np.take(CELLS, [0, 1], axis=a).ravel() for a in range(3)])
EDGE_R = np.concatenate([np.take(CELLS, [1, 2], axis=a).ravel() for a in range(3)])


def dump_new(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write('\n')


def strata(v):
    v = np.asarray(v)
    flat = v.reshape(-1, 27)
    s = np.empty(len(flat), dtype=np.uint16)
    for lo in range(0, len(flat), 4096):
        a = flat[lo:lo+4096]
        occ = a != 0
        e = (occ[:, EDGE_L] & occ[:, EDGE_R]).sum(axis=1)
        s[lo:lo+len(a)] = (((27-occ.sum(axis=1))//3)*8+(a == 1).sum(axis=1)//4)*8+e//8
    return s.reshape(v.shape[:-1])


def indicators(a):
    """Losslessly represent ternary state, species B implicit, interleaved by site."""
    a = np.asarray(a)
    out = np.empty((*a.shape[:-1], a.shape[-1]*2), np.float32)
    out[..., ::2] = (a == 0)
    out[..., 1::2] = (a == 1)
    out[..., ::2] -= .25
    out[..., 1::2] -= .30
    return out


def init_model(arch, seed):
    if arch not in ARMS:
        raise ValueError(arch)
    m = {}
    def branch(rng, name, din, dout):
        m[name+'W'] = (rng.normal(size=(din, dout))*np.sqrt(2/(din+dout))).astype('f')
        m[name+'b'] = np.zeros(dout, np.float32)
    rv, re = (np.random.default_rng(k) for k in np.random.SeedSequence(seed).spawn(2))
    if arch == 'tanh':
        branch(rv, 'v1', 54, WIDTH)
    branch(rv, 'v', WIDTH if arch == 'tanh' else 54, K)
    branch(re, 'e1', 772, WIDTH)
    branch(re, 'e', WIDTH, K)
    return m


def _branch(m, input_, key):
    if key+'1W' in m:
        input_ = np.tanh(input_ @ m[key+'1W'] + m[key+'1b'])
    return input_ @ m[key+'W'] + m[key+'b'], input_


def _apply(m, a, key, width):
    a = np.asarray(a)
    if a.shape[-1] != width:
        raise ValueError(f'{key} expects exactly {width} sites')
    shape = a.shape[:-1]
    a = a.reshape(-1, width)
    # Snapshot compressed checkpoint arrays once rather than once per chunk.
    p = {k: m[k] for k in m if k.startswith(key) and k.endswith(('W','b'))}
    out = np.empty((len(a), K), np.float32)
    for lo in range(0, len(a), 1024):
        out[lo:lo+1024] = _branch(p, indicators(a[lo:lo+1024]), key)[0]
    return out.reshape(*shape, K)


def center_logits(model, V):
    return _apply(model, V, 'v', 27)


def shell_logits(model, E):
    return _apply(model, E, 'e', 386)


def softmax(x, axis=-1):
    y = np.exp(x-x.max(axis=axis, keepdims=True))
    return y/y.sum(axis=axis, keepdims=True)


def logsoftmax(x, axis):
    y = x-x.max(axis=axis, keepdims=True)
    return y-np.log(np.exp(y).sum(axis=axis, keepdims=True))


def hbits(p):
    p = np.asarray(p, dtype='d')
    return -(p*np.log2(np.maximum(p, 1e-300))).sum(axis=-1)


class DrawLaw:
    """Empirical stratum -> uniform supported parents -> uniform eligible anchor."""
    def __init__(self, v):
        self.p, self.q = v.shape[:2]
        s = strata(v).ravel()
        parent = np.repeat(np.arange(self.p), self.q)
        # Stable ordering keeps parents and then translations contiguous within S.
        self.rows = np.argsort(s, kind='stable')
        ss, pp = s[self.rows], parent[self.rows]
        self.starts = np.flatnonzero(np.r_[True, (pp[1:] != pp[:-1]) | (ss[1:] != ss[:-1])])
        self.counts = np.diff(np.r_[self.starts, len(ss)])
        srun = ss[self.starts]
        unique, first, nparents = np.unique(srun, return_index=True, return_counts=True)
        mass = np.bincount(s, minlength=int(unique.max())+1)[unique]
        keep = nparents >= C
        if not np.any(keep):
            raise ValueError('No matched stratum has eight distinct parents')
        self.first, self.nparents = first[keep], nparents[keep]
        self.law = mass[keep]/mass[keep].sum()
        self.support = dict(parents=self.p, anchors_per_parent=self.q,
                            retained_anchor_fraction=float(mass[keep].sum()/len(ss)),
                            strata=[dict(s=int(a), anchors=int(b), parents=int(c), used=bool(d))
                                    for a,b,c,d in zip(unique,mass,nparents,keep)])

    def sample(self, rng, n, law):
        if law == 'uniform':
            size = np.full(n, self.p, dtype=np.int64)
        elif law == 'matched':
            bins = rng.choice(len(self.first), n, p=self.law)
            size = self.nparents[bins]
        else:
            raise ValueError(law)
        # Floyd's uniform C-subset algorithm, constant cost even at exactly C parents.
        pick = np.empty((n,C), dtype=np.int64)
        for i in range(C):
            top = size-C+i
            t = rng.integers(top+1)
            if i:
                t = np.where((pick[:,:i] == t[:,None]).any(axis=1), top, t)
            pick[:,i] = t
        # All C positions contribute as positive; no privileged position in estimator.
        if law == 'uniform':
            return pick*self.q+rng.integers(self.q, size=pick.shape)
        run = self.first[bins,None]+pick
        return self.rows[self.starts[run]+rng.integers(self.counts[run])]


class Bank:
    def __init__(self, v, e, fingerprint=None):
        if e.shape != (*v.shape[:2], 386) or v.shape[-1] != 27:
            raise ValueError('Bad geometry')
        self.draws = DrawLaw(v)
        self.v, self.e = v.reshape(-1,27), e.reshape(-1,386)
        self.fingerprint = fingerprint

    @classmethod
    def native(cls, role, n, release):
        if role not in ('fit','tune'):
            raise ValueError(role)
        # Do not even stat native inputs until an independently supplied release is checked.
        with open(release) as f:
            auth = json.load(f)
        if role not in auth['training_bank_roles']:
            raise PermissionError(f'{role} not in training_bank_roles')
        path = WORK/(role+'.npz')
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1<<20), b''):
                digest.update(b)
        if digest.hexdigest() != auth['sha256'][role]:
            raise ValueError(f'{role} digest mismatch')
        with np.load(path, allow_pickle=False) as a:
            parent = a['parent'][:n]
            if len(parent) != n or len(np.unique(parent)) != n:
                raise ValueError('Unexpected parent support')
            v, e = a['V'][:n], a['E'][:n]
            if v.shape != (n,64,27):
                raise ValueError('Expected exactly 64 anchors per parent')
        return cls(v, e, digest.hexdigest())


def _backward(m, grad, key, x, hidden, dz):
    grad[key+'W'] = hidden.T @ dz
    grad[key+'b'] = dz.sum(axis=0)
    if key+'1W' in m:
        dh = (dz @ m[key+'W'].T)*(1-hidden*hidden)
        grad[key+'1W'] = x.T @ dh
        grad[key+'1b'] = dh.sum(axis=0)


def objective_grad(m, v, e, frozen=None):
    """Exactly sum hard-h log candidate probabilities, NOT log of a q mixture."""
    if frozen is None:
        x = indicators(v)
        zv, hv = _branch(m, x, 'v')
        q = softmax(zv).reshape(-1,C,K)
    else:
        q = np.eye(K, dtype='f')[frozen].reshape(-1,C,K)
    y = indicators(e)
    ze, he = _branch(m, y, 'e')
    lp = logsoftmax(ze.reshape(-1,C,K), axis=1)
    eq = (q*lp).sum(axis=2, keepdims=True)
    b = len(e)
    grad = {}
    de = (np.exp(lp)*q.sum(axis=1,keepdims=True)-q)/b
    _backward(m, grad, 'e', y, he, de.reshape(-1,K))
    if frozen is None:
        dv = q*(eq-lp)/b
        _backward(m, grad, 'v', x, hv, dv.reshape(-1,K))
    return -float(eq.mean()), grad


class Learner:
    def __init__(self, arch, seed, objective, bank, frozen=None):
        self.m = init_model(arch, seed)
        if frozen is not None:
            self.m = {k:v for k,v in self.m.items() if k.startswith('e')}
        self.adam1 = {k:np.zeros_like(v) for k,v in self.m.items()}
        self.adam2 = {k:np.zeros_like(v) for k,v in self.m.items()}
        self.bank, self.frozen = bank, frozen
        self.rng = np.random.default_rng([seed,917])
        self.law, self.it = objective, 0

    def advance(self, end):
        normsq, last_loss, nlog = np.zeros(2), 0., 0
        start, cpu = time.monotonic(), time.process_time()
        while self.it < end:
            n = min(128,end-self.it)
            idx = self.bank.draws.sample(self.rng, n*G, self.law).reshape(n,G*C)
            for row in idx:
                self.it += 1
                lr = .0008/math.sqrt(1+self.it/1024)
                last_loss, grad = objective_grad(self.m, self.bank.v[row], self.bank.e[row],
                                                  None if self.frozen is None else self.frozen[row])
                scale = lr*math.sqrt(1-.999**self.it)/(1-.9**self.it)
                eps = 1e-8*math.sqrt(1-.999**self.it)
                logging = self.it % 32 == 0
                for k,g in grad.items():
                    a, b = self.adam1[k], self.adam2[k]
                    a *= .9
                    a += .1*g
                    b *= .999
                    b += .001*(g*g)
                    self.m[k] -= scale*a/(np.sqrt(b)+eps)
                    if logging:
                        normsq[int(k[0]=='e')] += float(np.vdot(g,g))
                nlog += int(logging)
        return dict(last_minibatch_loss=last_loss, gradient_rms_norm_encoder_critic=np.sqrt(normsq/max(nlog,1)).tolist(),
                    train_sec=time.monotonic()-start, train_cpu_sec=time.process_time()-cpu)


class EvalSet:
    def __init__(self, bank, seed):
        self.bank = bank
        self.idx = {law:bank.draws.sample(np.random.default_rng([seed,i]),4096,law) for i,law in enumerate(LAWS)}
        rows, inv = np.unique(np.asarray(list(self.idx.values())), return_inverse=True)
        self.rows = rows
        self.locs = inv.reshape(2,4096,C)
        self.parent_block = {law:(idx//bank.draws.q)%16 for law,idx in self.idx.items()}

    def score(self, m, frozen=None):
        b = self.bank
        q_all = softmax(center_logits(m,b.v)) if frozen is None else np.eye(K,dtype='f')[frozen]
        hard_all = q_all.argmax(axis=-1)
        rates = q_all.mean(axis=0, dtype=np.float64)
        hardp = np.bincount(hard_all,minlength=K)/len(q_all)
        hm, hc = float(hbits(rates)), float(hbits(q_all).mean())
        ze = shell_logits(m,b.e[self.rows])
        report = dict(stochastic_frequency=rates.tolist(), hard_frequency=hardp.tolist(), H_bits=hm,
                      H_given_V_bits=hc,I_V_H_bits=hm-hc,hard_H_bits=float(hbits(hardp)))
        retained = []
        for index,law in enumerate(LAWS):
            rows = self.idx[law]
            lp = logsoftmax(ze[self.locs[index]].astype('d'),1)+LOG_C
            hard = np.take_along_axis(lp,hard_all[rows,None],axis=-1)[...,0].mean(axis=1)
            soft = (lp*q_all[rows]).sum(axis=-1).mean(axis=1)
            report[law] = dict(hard_nats=float(hard.mean()), expected_nats=float(soft.mean()))
            block = self.parent_block[law]
            retained.append([float(hard[(block!=j).all(axis=1)].mean()) for j in range(16)])
        report['common_hard_nats'] = sum(report[l]['hard_nats'] for l in LAWS)/2
        report['delete_parent_block_common_hard_nats'] = np.mean(retained,axis=0).tolist()
        return report


def key(arch, law, parents, seed, end):
    return f'{arch}-{law}-p{parents}-s{seed}-u{end}'


def _checkpoint(path, learner):
    with path.open('xb') as f:
        np.savez(f, **learner.m, **{'adam1_'+k:v for k,v in learner.adam1.items()},
                 **{'adam2_'+k:v for k,v in learner.adam2.items()},
                 iteration=learner.it, rng_state=json.dumps(learner.rng.bit_generator.state))


def native_run(release, arch, law, parents, seed, end, resume=0):
    if parents not in (256,1024,4096) or seed not in SEEDS or end not in [2**i for i in range(9,17)]:
        raise ValueError('Not a registered rung')
    fit, tune = Bank.native('fit',parents,release), Bank.native('tune',512,release)
    fe, te = EvalSet(fit,1701), EvalSet(tune,1901)
    root = WORK/'checkpoints'
    root.mkdir(exist_ok=True,parents=True)
    spec = None
    if arch in BASELINES:
        from hero_rule import fit_control, hard_control
        spec = fit_control(fit.v,arch)
        ff, ft = hard_control(fit.v,spec).ravel(), hard_control(tune.v,spec).ravel()
    else:
        ff = ft = None
    z = Learner(arch if spec is None else 'affine', seed, law, fit, ff)
    if resume:
        path = root/(key(arch,law,parents,seed,resume)+'.npz')
        with np.load(path,allow_pickle=False) as a:
            for k in z.m:
                z.m[k],z.adam1[k],z.adam2[k] = a[k],a['adam1_'+k],a['adam2_'+k]
            z.rng.bit_generator.state = json.loads(str(a['rng_state']))
            z.it = int(a['iteration'])
        if z.it != resume:
            raise ValueError('Checkpoint iteration mismatch')
    for power in range(9,int(math.log2(end))+1):
        u = 2**power
        if u <= z.it:
            continue
        stamp = key(arch,law,parents,seed,u)
        wall, cpu = time.monotonic(), time.process_time()
        metrics = z.advance(u)
        fs, ts = fe.score(z.m,ff), te.score(z.m,ft)
        path = root/(stamp+'.npz')
        _checkpoint(path,z)
        dump_new(PUBLIC/'learning'/(stamp+'.json'),dict(arm=arch,law=law,seed=seed,parents=parents,iteration=u,
                 checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frozen_spec=spec,
                 fit=fs,tune=ts, fit_support=fit.draws.support if u==512 else None,
                 tune_support=tune.draws.support if u==512 else None, fit_bank_sha256=fit.fingerprint,
                 tune_bank_sha256=tune.fingerprint,stage_wall_sec=time.monotonic()-wall,
                 stage_cpu_sec=time.process_time()-cpu,**metrics))
        print(stamp,ts['common_hard_nats'],flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release',type=Path, required=True)
    ap.add_argument('--arch',choices=ARMS+BASELINES, required=True)
    ap.add_argument('--law',choices=LAWS, required=True)
    ap.add_argument('--parents',type=int,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--end',type=int,required=True)
    ap.add_argument('--resume',type=int,default=0)
    native_run(**vars(ap.parse_args()))
