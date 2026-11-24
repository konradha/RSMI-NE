"""Registered four-state conditional predictor for the open-patch cycle.

cycle_models.py preflight --work DIR                 # synthetic data only
cycle_models.py fit --work DIR [--public DIR]        # one authorized fit
cycle_models.py dev --work DIR                      # frozen checkpoint diagnostics
load(path) and predict(model, packed_shell, stratum) return the three probabilities
in stratum, packing, hard learned order. All three apply the same eligibility mask
upstream; attempting prediction for an ineligible stratum is an error.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import joblib
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit, softmax
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import boundary_models as bm

PUBLIC = Path(__file__).resolve().parents[2] / 'results/l20-autoresearch/cycle-boundary'
MODEL = 'cycle-model.joblib'
CHECKPOINTS = (8, 16, 32)
BATCH = 512
EPS = 1e-7


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def receipt(path, report):
    with path.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')


def rows(shell, stratum):
    shell = np.asarray(shell)
    if shell.dtype != np.uint8 or shell.ndim != 2 or shell.shape[1] != 49:
        raise ValueError('Expected [N,49] uint8 little-endian shell bytes')
    if np.shape(stratum) != (len(shell),) or not np.issubdtype(np.asarray(stratum).dtype, np.integer):
        raise ValueError('Expected integer stratum[N]')
    if np.any(shell[:, -1] & 252):
        raise ValueError('Nonzero bits outside registered shell')


def summarize_shells(shell):
    x = np.empty((len(shell), bm.D + bm.K), 'f')
    for start in range(0, len(x), 256):
        sl = slice(start, start+256)
        x[sl] = bm.features(shell[sl])
    return x


def signed_sites(shell):
    x = np.unpackbits(shell, axis=1, bitorder='little', count=len(bm.SHELL)).astype('f')
    x *= 2
    x -= 1
    return x


def scale(x):
    return (x.mean(0, dtype='d').astype('f'), np.maximum(x.std(0, dtype='d'), 1e-6).astype('f'))


def matrix(site, features, scaler, dtype='f'):
    x = np.empty((len(site), site.shape[1]+features.shape[1]), dtype)
    d = site.shape[1]
    x[:, :d] = site
    np.subtract(features, scaler[0], out=x[:, d:])
    x[:, d:] /= scaler[1]
    return x


def nuisance(ids, y, size):
    count = np.bincount(ids, minlength=size)
    yes = np.bincount(ids, weights=y, minlength=size)
    return np.log((yes+.5)/(count-yes+.5))


def packing_fit(site, conv, ids, y, size):
    alpha = nuisance(ids, y, size)
    a = alpha[ids]
    scaler = scale(conv)
    x = matrix(site, conv, scaler, 'd')

    def objective(w):
        logits = a + (x @ w[:-1]) + w[-1]
        residual = expit(logits)-y
        loss = np.logaddexp(0., (1.-2.*y)*logits).mean() + .0005*np.dot(w[:-1], w[:-1])
        grad = np.r_[x.T @ (residual/len(y)) + .001*w[:-1], residual.mean()]
        return loss, grad

    result = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
    if not result.success:
        raise ValueError(f'Packing logistic failed: {result.message}')
    p = expit(a)
    tree = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=.05,
                  l2_regularization=1., min_samples_leaf=20, early_stopping=False, random_state=260405)
    tree.fit(conv, y, sample_weight=np.where(y, 1.-p, p))
    return dict(alpha=alpha, scaler=scaler, w=result.x[:-1].copy(), b=float(result.x[-1]),
                tree=tree, logistic_iterations=int(result.nit), tree_iterations=int(tree.n_iter_))


def packing_logits(model, site, conv, ids):
    a = model['alpha'][ids]
    w = model['w']
    linear = site @ w[:site.shape[1]] + ((conv-model['scaler'][0])/model['scaler'][1]) @ w[site.shape[1]:]
    return a, a + .5*(linear + model['b'] + model['tree'].decision_function(conv))


def initialize(d, rng, dtype='f'):
    def weight(n, k):
        bound = np.sqrt(6/(n+k))
        return rng.uniform(-bound, bound, (n,k)).astype(dtype)
    return [weight(d,32), np.zeros(32,dtype), weight(32,4), np.zeros(4,dtype),
            np.asarray([-.75,-.25,.25,.75],dtype=dtype)]


def variational(x, y, base, params):
    w, b, v, c, delta = params
    h = np.tanh(x @ w + b)
    q = softmax(h @ v + c, axis=1)
    z = base[:,None] + delta
    ce = np.logaddexp(0., (1.-2.*y[:,None])*z)
    conditional = (q*ce).sum(axis=1, keepdims=True)
    ds = q*(ce-conditional)/len(y)
    dh = (ds @ v.T)*(1.-h*h)
    dd = (q*(expit(z)-y[:,None])).mean(axis=0)
    return conditional.mean(), [x.T @ dh, dh.sum(0), h.T @ ds, ds.sum(0), dd]


def state(x, params):
    w, b, v, c = params[:4]
    return (np.tanh(x @ w + b) @ v + c).argmax(axis=1)


def decode(code, base, y):
    offsets = np.zeros(4)
    for k in range(4):
        keep = code == k
        if keep.any():
            a, b = base[keep], y[keep]
            offsets[k] = brentq(lambda d: (expit(a+d)-b).sum()+d, -len(b)-1., len(b)+1.)
    return offsets


def probabilities(logits):
    return np.clip(expit(logits), EPS, 1.-EPS)


def scored(y, p):
    return np.where(y, np.log(p), np.log1p(-p))


def diagnostic(y, strata, z, a, base, offsets):
    score = scored(y[:,None], probabilities(np.column_stack((a, base, base+offsets[z]))))
    means = score.mean(axis=0)
    c = np.bincount(z, minlength=4)
    mass = c/len(y)
    yes = np.bincount(z, weights=y, minlength=4).astype(int)
    tab = np.zeros((len(np.unique(strata)),4,2), dtype=np.int64)
    levels, group = np.unique(strata, return_inverse=True)
    np.add.at(tab, (group,z,y.astype(int)), 1)
    return dict(mean_logscore_nats=dict(zip(('stratum','packing','learned'),means.tolist())),
                gain_vs_packing=float(means[2]-means[1]), gain_vs_stratum=float(means[2]-means[0]),
                hard_state_count=c.tolist(), hard_positive_count=yes.tolist(),
                hard_entropy_nats=float(-(mass[mass>0]*np.log(mass[mass>0])).sum()),
                noncollapse=bool((mass>=.01).sum()>=2), eligible=len(y),
                stratum_state_negative_positive={str(int(s)):v.tolist() for s,v in zip(levels,tab)})


def train_encoder(x, y, base, a, strata):
    rng = np.random.default_rng(260406)
    params = initialize(x.shape[1], rng)
    m = [np.zeros_like(v) for v in params]
    vv = [np.zeros_like(v) for v in params]
    step = 0
    snapshots, report = {}, {}
    y32, base32 = y.astype('f'), base.astype('f')
    for epoch in range(1,33):
        order = rng.permutation(len(y))
        for start in range(0,len(y),BATCH):
            r = order[start:start+BATCH]
            _, grads = variational(x[r], y32[r], base32[r], params)
            step += 1
            for j,g in enumerate(grads):
                m[j] *= .9
                m[j] += .1*g
                vv[j] *= .999
                vv[j] += .001*g*g
                params[j] -= .001*(m[j]/(1.-.9**step))/(np.sqrt(vv[j]/(1.-.999**step))+1e-8)
        if epoch in CHECKPOINTS:
            z = state(x, params)
            offsets = decode(z, base, y)
            snapshots[epoch] = dict(params=[v.copy() for v in params], offsets=offsets)
            d = diagnostic(y, strata, z, a, base, offsets)
            soft = 0.
            for start in range(0,len(y),BATCH):
                sl = slice(start,start+BATCH)
                soft += float(variational(x[sl],y32[sl],base32[sl],params)[0])*len(y[sl])
            report[str(epoch)] = dict(**d, steps=step, soft_expected_train_logscore=-soft/len(y),
                                      soft_decoder=params[-1].tolist(), hard_decoder=offsets.tolist())
    return snapshots,report


def fit_arrays(shell, strata, y, parent, levels):
    rows(shell,strata)
    if not len(y) or np.unique(y).tolist()!=[0,1] or np.any(~np.isin(strata,levels)):
        raise ValueError('Invalid eligible fit bank')
    ids = np.searchsorted(levels,strata)
    feats = summarize_shells(shell)
    site = signed_sites(shell)
    conv = np.ascontiguousarray(feats[:,:bm.D])
    parents = np.random.default_rng(260405).permutation(np.unique(parent))
    first = np.isin(parent,parents[::2])
    a, base = np.empty(len(y)), np.empty(len(y))
    fit_steps = []
    for held in (first,~first):
        other = ~held
        pack = packing_fit(site[other],conv[other],ids[other],y[other],len(levels))
        a[held],base[held] = packing_logits(pack,site[held],conv[held],ids[held])
        fit_steps.append([pack['logistic_iterations'],pack['tree_iterations']])
    pack = packing_fit(site,conv,ids,y,len(levels))
    fit_steps.append([pack['logistic_iterations'],pack['tree_iterations']])
    scale_orbits = scale(feats[:,bm.D:])
    x = matrix(site,feats[:,bm.D:],scale_orbits)
    del site, conv, feats
    snapshots,checkpoints = train_encoder(x,y,base,a,strata)
    model = dict(levels=np.asarray(levels), packing=pack, orbit_scaler=scale_orbits, checkpoints=snapshots)
    return model, dict(checkpoints=checkpoints, baseline_logistic_tree_steps=fit_steps,
                       fit_score_scope='OOF parent-isolated packing offsets, same fit rows used for encoder and hard decoder (not validation)',
                       eligible_parents=len(parents), fold_parent_counts=[len(parents[::2]),len(parents[1::2])])


def load(path):
    """Load frozen joblib from a trusted model file or its directory."""
    path = Path(path)
    return joblib.load(path/MODEL if path.is_dir() else path)


def evaluated_blocks(model, shell, strata, epochs):
    rows(shell,strata)
    if not np.isin(strata,model['levels']).all():
        raise ValueError('Attempted inference outside pilot eligibility')
    for start in range(0,len(shell),512):
        sl = slice(start,start+512)
        site = signed_sites(shell[sl]); f = summarize_shells(shell[sl])
        a, b = packing_logits(model['packing'],site,f[:,:bm.D],np.searchsorted(model['levels'],strata[sl]))
        x = matrix(site,f[:,bm.D:],model['orbit_scaler'])
        z = [state(x,model['checkpoints'][e]['params']) for e in epochs]
        yield sl,a,b,z


def predict(model, shell, stratum):
    """P(Y=1) by stratum-only, packing ensemble, hard four-symbol predictor (epoch 32)."""
    out = np.empty((len(shell),3))
    offsets = model['checkpoints'][32]['offsets']
    for sl, a, b, (z,) in evaluated_blocks(model,shell,stratum,(32,)):
        out[sl] = probabilities(np.column_stack((a,b,b+offsets[z])))
    return out


def development(model, shell, strata, y, parent):
    a, b, zs = np.empty(len(y)), np.empty(len(y)), np.empty((3,len(y)),'i1')
    for sl, aa, bb, zz in evaluated_blocks(model,shell,strata,CHECKPOINTS):
        a[sl], b[sl], zs[:,sl] = aa,bb,zz
    d = {str(e):diagnostic(y,strata,z,a,b,model['checkpoints'][e]['offsets']) for e,z in zip(CHECKPOINTS,zs)}
    gains = [item['gain_vs_packing'] for item in d.values()]
    return dict(checkpoints=d, eligible_parents=len(np.unique(parent)),
                gain_still_rising_at_32=bool(gains[2]>gains[1]),
                gains_monotonically_rising=bool(gains[2]>gains[1]>gains[0]), selected_epoch=32)


def bank(work,role,levels):
    if role not in ('fit','dev'):
        raise ValueError('Unauthorized role')
    with np.load(work/f'{role}.npz',allow_pickle=False) as d:
        p, y, s, shell = (d[k] for k in ('parent','y','stratum','shell'))
    with np.load(work/'rosters.npz',allow_pickle=False) as r:
        if not np.array_equal(p,r[f'{role}_parent']):
            raise ValueError('Role parent order mismatch')
    if p.ndim!=1 or len(np.unique(p))!=len(p) or y.shape!=(len(p),128) or s.shape!=y.shape or shell.shape!=(*y.shape,49):
        raise ValueError('Role array shape/parent mismatch')
    if y.dtype!=np.uint8 or y.max()>1 or not np.issubdtype(s.dtype,np.integer):
        raise ValueError('Bad role labels or strata')
    parent,anchor = np.where(np.isin(s,levels))
    if len(parent)==0:
        raise ValueError('No eligible observations')
    return shell[parent,anchor],s[parent,anchor],y[parent,anchor],p[parent]


def gradient_check():
    rng = np.random.default_rng(260408)
    params = initialize(5,rng,'d')
    x, y, base = rng.normal(size=(9,5)), rng.integers(2,size=9), rng.normal(size=9)
    _, g = variational(x,y,base,params)
    error = 0.
    for v, analytic in zip(params,g):
        for idx in np.ndindex(v.shape):
            init = v[idx]; v[idx] = init+1e-5
            plus = variational(x,y,base,params)[0]
            v[idx] = init-1e-5
            minus = variational(x,y,base,params)[0]
            v[idx] = init
            error = max(error,abs((plus-minus)/2e-5-analytic[idx]))
    return float(error)


def preflight(work):
    out = work/'cycle-control.json'
    if out.exists():
        raise FileExistsError(out)
    err = gradient_check()
    rng = np.random.default_rng(260407)
    n = 96*128
    bits = rng.random((n,len(bm.SHELL)))<.25
    signs = 2*rng.integers(2,size=(n,2))-1
    z,yy,x = bm.SHELL.T
    for j,strip in enumerate((x<0,x>=0)):
        ix = np.flatnonzero((abs(z)==4)&(abs(yy)<=2)&strip)
        bits[:,ix] = rng.random((n,len(ix))) < np.where(np.sign(z[ix])*signs[:,j,None]>0,.75,.05)
    strata = rng.integers(2,size=n)
    y = (rng.random(n)<expit(.6*(2*strata-1)+2.5*signs.prod(1))).astype('u1')
    parent = np.arange(n)//128
    shell = np.packbits(bits,axis=1,bitorder='little')
    fit = parent<64
    arrays = (shell,strata,y,parent)
    m,train = fit_arrays(*(v[fit] for v in arrays),np.arange(2))
    dev = development(m,*(v[~fit] for v in arrays))
    last = dev['checkpoints']['32']
    passed = err<1e-6 and last['gain_vs_packing']>.01 and last['hard_entropy_nats']>.20 and last['noncollapse']
    receipt(out,dict(passed=bool(passed),finite_difference_max_abs_error=err, fit=train,held=dev,
                     scope='Synthetic positive control, NOT sensitivity bound on physical signal',code_sha256=digest(__file__)))
    print(f'control pass={passed} grad_error={err:.3g} held_gain={last["gain_vs_packing"]:.6g} entropy={last["hard_entropy_nats"]:.5g}',flush=True)


def main(args):
    work = args.work
    work.mkdir(parents=True,exist_ok=True)
    if args.job=='preflight':
        return preflight(work)
    levels = np.load(work/'eligible.npy',allow_pickle=False)
    if levels.ndim!=1 or not np.array_equal(levels,np.unique(levels)) or not len(levels):
        raise ValueError('Expected sorted unique eligible strata')
    model_path = work/MODEL
    report_path = work/f'cycle-{args.job}.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    if args.job=='fit':
        if model_path.exists() or (args.public/MODEL).exists():
            raise FileExistsError('Registered model already exists; no new starts')
        with (work/'cycle-control.json').open() as f:
            if not json.load(f)['passed']:
                raise ValueError('Synthetic control failed; no physical fit')
        m,report = fit_arrays(*bank(work,'fit',levels),levels)
        with model_path.open('xb') as f:
            joblib.dump(m,f,compress=3)
        with model_path.open('rb') as src,(args.public/MODEL).open('xb') as dest:
            shutil.copyfileobj(src,dest)
    else:
        with (work/'cycle-fit.json').open() as f:
            if json.load(f)['model_sha256']!=digest(model_path):
                raise ValueError('Model changed after fit')
        report = development(load(model_path),*bank(work,'dev',levels))
    receipt(report_path,dict(**report,model_sha256=digest(model_path),eligible_sha256=digest(work/'eligible.npy'),
                        roster_sha256=digest(work/'rosters.npz'), bank_sha256=digest(work/f'{args.job}.npz'),
                        code_sha256=digest(__file__), boundary_models_sha256=digest(bm.__file__),
                        templates_sha256=digest(bm.boundary_templates.__file__),shell_sha256=bm.SHELL_SHA,
                        design_sha256=digest(args.public/'model-design.json'),sklearn=sklearn.__version__,numpy=np.__version__))
    print(report_path,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job',choices=('preflight','fit','dev'))
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--public',type=Path,default=PUBLIC)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
