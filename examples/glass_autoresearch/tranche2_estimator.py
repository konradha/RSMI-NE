"""Matched soft-encoder objectives on archived native TRAIN words; never read beyond an authorized phase."""
import hashlib
import itertools
import json
import os
import zipfile
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.special import xlogy
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/l20-autoresearch/tranche2'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/tranche2'
BANK = WORK / 'b09.npz'
V, J, Y, H = 6561, 6, 81, 4
SEEDS = (2101, 2102, 2103)
RUNGS = (256, 1024, 4096)
CHECK = (0, 640, 1280, 2560, 5120)
ARMS = ('direct', 'contrastive', 'distilled')
DIG = np.arange(V)[:, None] // 3**np.arange(8) % 3
X = np.column_stack((DIG == 1, DIG == 2)).astype('f4')
TILE = np.array(list(itertools.product(range(2), repeat=3)))
ORIGINS = np.array(list(itertools.product(range(0, 20, 2), repeat=3)))
FACES = [np.insert(np.array(list(itertools.product(range(2), repeat=2))), axis, n, axis=1) for axis in range(3) for n in (-2, 3)]


def jout(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def extract(a, b):
    # Stream archived species member through authorized end only; skip directly over fit/dev when auditing.
    with zipfile.ZipFile(BANK) as z, z.open('species.npy') as f:
        ver = np.lib.format.read_magic(f)
        meta = np.lib.format._read_array_header(f, ver)
        assert meta == ((5120, 20, 20, 20), False, np.dtype('u1')), meta
        f.seek(f.tell() + a * 8000)
        ix = [np.ravel_multi_index(((ORIGINS[:, None, :] + off) % 20).transpose(2, 0, 1), (20,)*3) for off in [TILE, *FACES]]
        v = np.empty((b-a, 1000), 'u2'); y = np.empty((b-a, 1000, J), 'u1')
        for p in range(0, b-a, 32):
            n = min(b-a-p, 32)
            s = np.frombuffer(f.read(n*8000), dtype='u1').reshape(n, 8000)
            v[p:p+n] = s[:, ix[0]] @ (3**np.arange(8)).astype('u2')
            for j in range(J):
                y[p:p+n, :, j] = s[:, ix[j+1]] @ np.array([1, 3, 9, 27], 'u1')
    return v, y


def counts(v, y):
    vi = v.astype('i4') * Y
    return np.stack([np.bincount((vi+y[:, :, j]).ravel(), minlength=V*Y).reshape(V, Y) for j in range(J)], axis=1)


class Channel:
    def __init__(self, c):
        self.c = c
        self.n = int(c[:, 0].sum()); self.p = (c[:, 0].sum(1)/self.n).astype('f4')
        assert np.all(c.sum(2) == c[:, :1].sum(2))
        self.pw = c.sum(0).reshape(-1)/(self.n*J)
        self.mat = np.ascontiguousarray(c.reshape(V, J*Y)/(self.n*J), dtype='f4')

    def soft(self, q, grad=False):
        joint = (q.T @ self.mat).astype('f8')
        independent = joint.sum(1)[:, None] * self.pw
        lr = np.log(np.divide(joint, independent, out=np.ones_like(joint), where=(joint > 0) & (independent > 0)))
        score = float((joint*lr).sum())
        return (score, self.mat @ lr.T.astype('f4')) if grad else score

    def hard(self, lut):
        return np.array([self.c[lut == h].sum(0) for h in range(H)])


def decoder(c):
    m = c.sum(0) + 0.5; m /= m.sum(1, keepdims=True)
    pred = c + 81*m; pred /= pred.sum(2, keepdims=True)
    return np.log(pred/m)


def hard_stats(c):
    p = c/c.sum(); ph = p.sum((1, 2)); pj = p.sum(0)
    return dict(hard_plugin_nats=float(xlogy(p, p).sum() - xlogy(ph, ph).sum() - xlogy(pj, pj).sum()), hard_entropy_nats=float(-xlogy(ph, ph).sum()), hard_mass=ph.tolist())


def parent_scores(lut, lr, v, y):
    res = np.zeros(len(v))
    for start in range(0, len(v), 64):
        sl = slice(start, start+64); h = lut[v[sl]]
        for j in range(J):
            res[sl] += lr[h, j, y[sl, :, j]].mean(1)/J
    return res


def teacher(ch):
    sides = 1 - 2*TILE
    tetra = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    h = ((DIG == 1) @ sides @ tetra.T).argmax(1)
    prior = ch.c.sum(0) + 0.5; prior /= prior.sum(1, keepdims=True)
    joint = (ch.c + prior).reshape(V, J*Y)
    path = []
    for step in range(256):
        cc = np.stack([joint[h == k].sum(0) for k in range(H)])
        m = cc.sum(1); freq = np.divide(cc, m[:, None], out=np.tile(prior.ravel()/J, (H,1)), where=m[:, None]>0)
        reward = joint @ np.log(freq).T; alt = reward.argmax(1)
        old = reward[np.arange(V), h]; best = reward[np.arange(V), alt]
        move = best > old + 1e-12*joint.sum(1); path.append(int(move.sum()))
        if not move.any(): break
        h[move] = alt[move]
    return h.astype('u1'), path


def init(seed):
    r = np.random.default_rng(seed)
    return [r.normal(0, 1/4, (16, 32)).astype('f4'), np.zeros(32, 'f4'), r.normal(0, 1/np.sqrt(32), (32, H)).astype('f4'), np.zeros(H, 'f4')]


def forward(p, x):
    z = np.tanh(x @ p[0] + p[1]); logits = z @ p[2] + p[3]
    q = np.exp(logits-logits.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    return z, q


def back(p, x, z, delta):
    dz = (delta @ p[2].T) * (1-z*z)
    return [x.T @ dz, dz.sum(0), z.T @ delta, delta.sum(0)]


def softback(q, g):
    return q*(g-(q*g).sum(1, keepdims=True))


class Adam:
    def __init__(self, pars, lr):
        self.p = pars; self.lr = lr; self.m = [np.zeros_like(p) for p in pars]; self.v = [a.copy() for a in self.m]; self.n = 0

    def update(self, grad):
        self.n += 1
        a = self.lr*(1-0.999**self.n)**0.5/(1-0.9**self.n)
        for p, m, v, g in zip(self.p, self.m, self.v, grad):
            m *= 0.9; m += 0.1*g; v *= 0.999; v += 0.001*g*g
            p += a*m/(np.sqrt(v)+1e-8*(1-0.999**self.n)**0.5)


def nce_sample(rng, vf, yf, batch=1024):
    pos = rng.integers(len(vf), size=batch); j = rng.integers(J, size=batch)
    yy = np.empty((batch, 16), 'u1'); yy[:, 0] = yf[pos, j]
    yy[:, 1:] = yf[rng.integers(len(vf), size=(batch, 15)), j[:, None]]
    return vf[pos], j, yy


def nce(p, critic, samp):
    v, j, yy = samp; x = X[v]; z, q = forward(p, x); b = len(x)
    idx = j[:, None]*Y + yy; scores = critic[idx]
    centered = scores - scores.max(1, keepdims=True)
    weights = np.exp(centered); den = weights.sum(1, keepdims=True); weights /= den
    pos = centered[:, 0] - np.log(den[:, 0]) + np.log(16.)
    grad = back(p, x, z, softback(q, pos/b))
    weights[:, 0] -= 1; weights *= -(q[:, None, :]/b)
    ids = idx.ravel(); cg = np.stack([np.bincount(ids, weights[:, :, h].ravel(), minlength=J*Y) for h in range(H)], axis=1).astype('f4')
    return float((q*pos).sum()/b), grad, cg


def full_obj(p, ch, arm, target):
    z, q = forward(p, X)
    if arm == 'direct':
        obj, g = ch.soft(q, True); delta = softback(q, g)
    else:
        obj = float(ch.p @ np.log(np.maximum(q[np.arange(V), target], 1e-38)))
        delta = -q.copy(); delta[np.arange(V), target] += 1; delta *= ch.p[:, None]
    return obj, back(p, X, z, delta)


def ci(a):
    a = np.asarray(a); m = a.mean(); h = t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
    return [float(m-h), float(m+h)]


def gradvec(gs):
    return np.concatenate([g.ravel() for g in gs])


def noise(p, critic, vf, yf, seed):
    r = np.random.default_rng(seed)
    g = np.array([gradvec(nce(p, critic, nce_sample(r, vf, yf))[1]) for _ in range(12)], dtype='f8')
    return dict(n=12, encoder_batch_gradient_mean_norm=float(np.linalg.norm(g.mean(0))), encoder_batch_gradient_rms_noise=float(np.sqrt(g.var(0, ddof=1).sum())))


def measure(p, target, fit, dev, dv, dy):
    _, q = forward(p, X); h = np.argmax(q, axis=1).astype('u1'); rec = {}
    fc = fit.hard(h); lr = decoder(fc)
    for name, ch, c in [('fit', fit, fc), ('dev', dev, dev.hard(h))]:
        s = hard_stats(c); s.update(soft_exact_MI_nats=ch.soft(q), hard_fit_decoder_nats=float((lr*c).sum()/c.sum()), teacher_agreement=float(ch.p[h == target].sum(dtype='f8'))); rec[name] = s
    par = parent_scores(h, lr, dv, dy)
    assert abs(par.mean()-rec['dev']['hard_fit_decoder_nats']) < 1e-10
    rec['dev']['parent_predictive_ci95'] = ci(par)
    rec['fit_minus_dev_optimism_nats'] = rec['fit']['hard_fit_decoder_nats']-par.mean()
    return rec, h, lr, par, decoder(dev.hard(h))


def controls():
    # Synthetic COUNTS, no physical configurations. Balanced 4-state V&Y with P(correct)=.94; six orientation permutations.
    v = np.array([0,1,3,4]); c = np.zeros((V,J,Y), 'i4')
    for j in range(J):
        for k in range(H):
            c[v[k],j,:4] = 2; c[v[k],j,(k+j)%4] = 94
    ch = Channel(c); gt = np.zeros(V, int); gt[v] = np.arange(H); one = np.eye(H,dtype='f4')[gt]
    exact = float(np.log(4)+.94*np.log(.94)+.06*np.log(.02)); measured = ch.soft(one)
    assert abs(exact-measured) < 1e-6
    p = init(91); rng = np.random.default_rng(16); critic = rng.normal(0,.3,(J*Y,H)).astype('f4'); sf = np.repeat(v, 100)
    sy = np.stack([np.concatenate([np.repeat(np.arange(Y), c[k,j]) for k in v]) for j in range(J)],axis=1)
    samp = nce_sample(rng, sf, sy); checks = {}
    for arm in ARMS:
        def obj():
            if arm == 'contrastive':
                s,g,cg = nce(p,critic,samp); return s, g+[cg]
            return full_obj(p,ch,arm,gt)
        val, gr = obj(); pars = p+[critic] if arm == 'contrastive' else p
        # Direction of gradient scaled to unit norm is sensitive to sign, encoder chain and critic derivative.
        gn = np.linalg.norm(gradvec(gr)); d = [g/gn for g in gr]; eps = .003
        saved = [a.copy() for a in pars]
        for a,old,dr in zip(pars,saved,d): a[:] = old + eps*dr
        plus = obj()[0]
        for a,old,dr in zip(pars,saved,d): a[:] = old - eps*dr
        minus = obj()[0]
        for a,old in zip(pars,saved): a[:] = old
        num = (plus-minus)/(2*eps); checks[arm] = dict(analytic=float(gn), finite_difference=float(num), abs_error=float(abs(gn-num)))
        assert abs(gn-num)<2e-4, checks
    result = {}
    for arm in ARMS:
        pp = init(91); cc = np.zeros_like(critic); ap = Adam(pp,.005); ac = Adam([cc],.01); r = np.random.default_rng(121)
        for step in range(640):
            if arm == 'contrastive':
                _,g,cg = nce(pp,cc,nce_sample(r,sf,sy)); ac.update([cg])
            else:
                _,g = full_obj(pp,ch,arm,gt)
            ap.update(g)
        h = forward(pp,X)[1].argmax(1); recovered = hard_stats(ch.hard(h))['hard_plugin_nats']
        result[arm] = dict(hard_MI_nats=recovered, recovered_all_four=bool(len(np.unique(h[v]))==4))
        # A missed local optimum is a scientific result; do not silently change initialization.
    return dict(analytic_MI=exact, direct_onehot_MI=measured, planted_description='4 equally likely tile words 0,1,3,4; 94% correct directional face letter, 2% each other, four letters rotated by J. Analytical maximum ln4+.94ln.94+.06ln.02.', gradient_checks=checks, learned_640_updates=result)
