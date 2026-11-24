"""Fit four-state nonlinear static RSMI and matched controls on independent L20 parents.

Run from repo root with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1:
  python examples/l20_operators_static.py --data ~/datasets/lattice-glass/l20-16384-v1
  python examples/l20_operators_static_analysis.py --data ~/datasets/lattice-glass/l20-16384-v1
The first command fits/selects using train/validation only. The second freezes every checkpoint.
"""
import argparse
import hashlib
import io
import itertools
import json
import multiprocessing as mp
import zipfile
from pathlib import Path

import numpy as np
from l20_pairs import L, PAIR_SEED, SPLIT, partition

OUT = Path(__file__).resolve().parents[1] / 'results/l20-operators/static'
EPOCHS, BATCH, HIDDEN, VREPS, TREPS = 36, 128, 24, 6, 16
SEEDS = (11, 29, 47)
EYE = np.eye(4, dtype=np.float32)
# Vacancy is the reference channel; separate trainable A/B weights plus bias span
# arbitrary ternary one-site functions. Fixed affine scaling, independent of beta.
CODE = np.array([[-.5, -.5], [1.5, -.5], [-.5, 1.5]], dtype=np.float32)


def dump(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def source(data, index):
    p = data / f'beta-{index:02d}.npz'
    receipt = json.loads(p.with_suffix('.json').read_bytes())
    with p.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if sha != receipt['endpoint_sha256']:
        raise ValueError(f'Changed endpoint {p}')
    with np.load(p, allow_pickle=False) as f:
        labels = f['species']; keys = np.column_stack((f['chain_seed'], f['chain_id'])); beta = float(f['beta'])
        if str(f['axes']) != 'chain,z,y,x' or beta != float(receipt['beta']):
            raise ValueError('Axis or temperature mismatch')
    if (labels.dtype != np.uint8 or labels.shape != (sum(SPLIT), L, L, L) or labels.max() > 2
            or len(np.unique(keys, axis=0)) != len(labels)):
        raise ValueError('Invalid ternary endpoint array or duplicate lineage')
    groups, _ = partition(index)
    return labels.reshape(-1, L**3), dict(index=index, beta=beta, sha256=sha, split=SPLIT, patch_seed=PAIR_SEED,
        group_lineage_sha256=[hashlib.sha256(keys[g].astype('<u8').tobytes()).hexdigest() for g in groups])


def offsets(w, b):
    r = w//2
    v = np.indices((w,)*3).reshape(3, -1).T-r
    # A complete centered 5x5 patch on each face; closed under all 48 cube isometries.
    e = []
    for axis, sign, (a, c) in itertools.product(range(3), (-1, 1), itertools.product(range(-2, 3), repeat=2)):
        u = [a, c]; u.insert(axis, sign*(r+b+1)); e.append(u)
    return v, np.array(e)


def table(off):
    centers = np.indices((L,)*3, dtype=np.int16).reshape(3, -1).T
    ids = np.zeros((L**3, len(off)), np.uint16)
    for d in range(3):
        ids *= L; ids += ((centers[:, d, None] + off[None, :, d].astype(np.int16)) % L).astype(np.uint16)
    return ids


def banks(state, index, w, b, fold, n):
    groups, rngs = partition(index); g = groups[fold]; rng = rngs[fold]
    vt, et = map(table, offsets(w, b))
    v = np.empty((n, len(g), vt.shape[1]), np.uint8); e = np.empty((n, len(g), et.shape[1]), np.uint8)
    for i in range(n):
        rows = rng.permutation(g) if fold == 0 else g
        a = rng.integers(L**3, size=len(rows))
        v[i] = state[rows[:, None], vt[a]]; e[i] = state[rows[:, None], et[a]]
    return v, e


def dense(v):
    return CODE[v].reshape(len(v), -1)


def init(rng, n, enc=False, w=None, warm=False):
    W = (rng.normal(size=(n, HIDDEN))/np.sqrt(n)).astype('f')
    U = (rng.normal(size=(HIDDEN, 4))*(.4 if enc else .03)/np.sqrt(HIDDEN)).astype('f')
    A = np.zeros((n, 4), 'f'); c = np.zeros(4, 'f')
    if warm:
        # Weak corner-site identity bias breaks the constant-channel optimum;
        # all input, hidden, output and shortcut coefficients stay trainable.
        xyz = rng.integers(2, size=3)*(w-1); j = np.ravel_multi_index(xyz, (w,)*3)
        A[2*j:2*j+2, :3] = [[-.7, .7, 0], [-.7, 0, .7]]
    return [W, np.zeros(HIDDEN, 'f'), U, A, c]


def forward(x, weights):
    W, b, U, A, c = weights
    h = np.tanh(x@W + b)
    return h, h@U + x@A + c


def backward(x, h, weights, dz):
    dh = (dz@weights[2].T)*(1-h*h)
    return [x.T@dh, dh.sum(0), h.T@dz, x.T@dz, dz.sum(0)]


def softmax(z):
    z = np.exp(z-z.max(axis=-1, keepdims=True)); return z/z.sum(axis=-1, keepdims=True)


def physical(v, w):
    """Counts of internal vacancy/particle nearest-neighbor interfaces; no halo."""
    occ = (v != 0).reshape((-1, w, w, w))
    return sum((np.diff(occ, axis=d)).sum(axis=(1, 2, 3)) for d in (1, 2, 3))


def fixed_spec(v, w):
    # Thresholds are fit to a single train epoch only; frozen for every beta.
    count = [(v == s).sum(1) for s in (0, 1)]
    parity = np.where(np.indices((w,)*3).sum(0).ravel()%2, -1, 1)
    par = (v != 0)@parity
    specs = {f'site-{z}{y}{x}': dict(kind='site', xyz=[z,y,x],
            j=int(np.ravel_multi_index(np.array([z,y,x])+w//2, (w,)*3)))
        for z,y,x in itertools.combinations_with_replacement(range(w//2+1), 3)}
    # Two resolved binary indicators use all four channels: compare near
    # and separated boundary sites, not merely a three-state corner label.
    for name, xyz in [('near',(0,0,1)),('edge',(0,0,w-1)),('face',(0,w-1,w-1)),('body',(w-1,)*3)]:
        for s in range(3):
            specs[f'bits-{name}-{s}'] = dict(kind='bits', species=s,
                    j=int(np.ravel_multi_index(xyz,(w,)*3)))
    specs.update(composition=dict(kind='composition', thresholds=[float(np.median(c)) for c in count]),
        interfaces=dict(kind='interfaces', thresholds=np.quantile(physical(v,w), [.25,.5,.75]).tolist()),
        parity=dict(kind='parity', thresholds=np.quantile(par, [.25,.5,.75]).tolist()))
    return specs


def codes(v, w, spec):
    k = spec['kind']
    if k == 'site': return v[:, spec['j']]
    if k == 'bits': return 2*(v[:,0]==spec['species'])+(v[:,spec['j']]==spec['species'])
    t = spec['thresholds']
    if k == 'composition': return 2*((v==0).sum(1)>t[0])+((v==1).sum(1)>t[1])
    s = physical(v,w) if k == 'interfaces' else (v!=0)@np.where(np.indices((w,)*3).sum(0).ravel()%2,-1,1)
    return np.searchsorted(t, s).astype(np.uint8)


def prob(v, w, spec, enc):
    if spec['kind'] in ('learned','random'):
        return softmax(forward(dense(v), enc)[1])
    return EYE[codes(v, w, spec)]


def evaluate(vbank, ebank, w, spec, enc, critic):
    """Per disjoint parent batch, mean over repeated anchors *within* batch."""
    result = np.zeros((vbank.shape[1]//BATCH, 2))
    for v, e in zip(vbank, ebank):
        p = prob(v, w, spec, enc).reshape(-1, BATCH, 4)
        z = forward(dense(e), critic)[1].reshape(-1, BATCH, 4).astype(float)
        z -= z.max(1, keepdims=True); r = z-np.log(np.exp(z).mean(1, keepdims=True))
        other = (r.sum(1, keepdims=True)-r)/(BATCH-1)
        for j, a in enumerate((r, other)):
            result[:,j] += (p*a).sum(2).mean(1)
    return result / len(vbank)


def netfile(out, w,b,m,s): return out/f'w{w}-b{b}-{m}-s{s}'


def readfit(path):
    meta_path=path.with_suffix('.json'); arrays=path.with_suffix('.npz')
    if meta_path.exists():
        meta=json.loads(meta_path.read_bytes())
    else:
        with zipfile.ZipFile(path.parent.parent/'fit-receipts.zip') as archive:
            meta=json.loads(archive.read(f'fits/{meta_path.name}'))
            arrays=io.BytesIO(archive.read(f'fits/{arrays.name}'))
    with np.load(arrays,allow_pickle=False) as f:
        return meta, [f[f'enc{i}'] for i in range(5) if f'enc{i}' in f], [f[f'crit{i}'] for i in range(5)]

def pack_receipts(out):
    with zipfile.ZipFile(out/'fit-receipts.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((out/'fits').glob('*')):
            if p.suffix in ('.json','.npz'): archive.write(p,p.relative_to(out))


def fit(w,b,method,seed,spec,train,val,out):
    learn = method=='learned'; rng=np.random.default_rng(seed)
    enc=init(rng,2*w**3,True,w,learn) if method in ('learned','random') else []
    critic=init(np.random.default_rng(9601),2*train[1].shape[-1])
    weights=(enc if learn else []) + critic
    mom=[np.zeros_like(a) for a in weights]; vel=[np.zeros_like(a) for a in weights]
    best=-np.inf; history=[]; n=0
    # Include epoch zero for weak/no-signal comparisons (no test-based clipping).
    for epoch in range(EPOCHS+1):
        if epoch:
            raw=train[0][epoch-1]; xe=dense(train[1][epoch-1]); total=0.
            xv=dense(raw) if learn else None; p0=None if learn else prob(raw,w,spec,enc)
            for j in range(0,len(raw),BATCH):
                ix=slice(j,j+BATCH); x=xe[ix]; he,z=forward(x,critic)
                if learn:
                    hv,logits=forward(xv[ix],enc); p=softmax(logits)
                else: p=p0[ix]
                z-=z.max(0); ez=np.exp(z); r=z-np.log(ez.mean(0)); ei=ez/ez.sum(0)
                contribution=(p*r).sum(1,keepdims=True); total+=contribution.sum()
                dz=(ei*p.sum(0)-p)/BATCH
                grad=(backward(xv[ix],hv,enc,-p*(r-contribution)/BATCH) if learn else [])+backward(x,he,critic,dz)
                n+=1; rate=(.002 if epoch<=24 else .00065)*np.sqrt(1-.999**n)/(1-.9**n)
                for a,g,m,s in zip(weights,grad,mom,vel):
                    m*=.9; m+=.1*g; s*=.999; s+=.001*g*g
                    a-=rate*m/(np.sqrt(s)+1e-8)
        if epoch==0 or epoch%3==0:
            score=evaluate(*val,w,spec,enc,critic)[:,0].mean()
            history.append(dict(epoch=epoch,validation=float(score),training=float(total/len(raw)) if epoch else None))
            if score>best:
                best=float(score); selected=epoch; ckpt=[a.copy() for a in enc+critic]
    path=netfile(out,w,b,method,seed)
    meta=dict(width=w,buffer=b,method=method,seed=seed,spec=spec,validation=best,epoch=selected,history=history)
    dump(path.with_suffix('.json'),meta)
    np.savez_compressed(path.with_suffix('.npz'),**{f'enc{i}':a for i,a in enumerate(ckpt[:len(enc)])},
                        **{f'crit{i}':a for i,a in enumerate(ckpt[len(enc):])})
    print(f'w{w} b{b} {method} {seed} epoch {selected} val={best:.5g}',flush=True)


def geometry(job):
    data,out,w,b=job
    state,_=source(data,9)
    train=banks(state,9,w,b,0,EPOCHS); val=banks(state,9,w,b,1,VREPS); del state
    specs=fixed_spec(train[0][0],w)
    jobs=[('learned',s,dict(kind='learned')) for s in SEEDS]+[(m,11,s) for m,s in specs.items()]+[('random',11,dict(kind='random'))]
    for m,s,spec in jobs: fit(w,b,m,s,spec,train,val,out)


def args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=OUT); return p.parse_args()


if __name__=='__main__':
    a=args(); fits=a.out/'fits'; fits.mkdir(exist_ok=True,parents=True)
    jobs=[(a.data,fits,w,b) for b,w in itertools.product((1,2),(3,5))]
    _,meta=source(a.data,9); del _
    dump(a.out/'design.json',dict(source=meta,epochs=EPOCHS,train_parents_per_epoch=SPLIT[0],
        pairs_seen_per_fit=EPOCHS*SPLIT[0],validation_anchors=VREPS,test_anchors=TREPS,batch=BATCH,
        optimizer='Adam 0.002 epochs 1-24, then 0.00065; validation every 3 epochs incl 0',
        encoder='separate species indicators + 24 tanh units + linear skip, categorical softmax; weak random-corner start',
        critic='150 shell labels with 2 separate species indicators, 24 tanh units + linear skip to four scores',
        shell='fixed centered 5x5 face patch on each of six faces at Chebyshev radius width//2+buffer+1',
        train='same-beta in-batch negatives, exact expected hard-state InfoNCE; new periodic anchor and parent permutation per epoch',
        selection='highest validation across 12 learned encoders; best deterministic control by validation within geometry (random separately)',
        inference='no test in fitting; 16 anchors for each of 4096 test parents; paired t intervals on 32 disjoint batches of 128; all off-diagonal shuffle pairs',
        gate='Assess raw lower bound vs zero, learned minus strongest fixed, then generalization at frozen beta=2.7111 and 0; pointwise intervals, no power-law inference'))
    with mp.get_context('fork').Pool(3) as pool: pool.map(geometry,jobs)
    pack_receipts(a.out)
