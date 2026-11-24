"""Second, neural directional bottleneck. Ordered six-face propensity contrasts, not equilibrium MI.

With K distinct parents and one free bond per parent, s[h,j] is a nonlinear critic of the
future six-face environment j. Train both networks on the EXACT expectation over four symbols:
  mean_i sum_h q[h|V_i] (s[h,i] - logsumexp_j s[h,j] + log K).
No relaxed symbol, inner-product surrogate, or supervised face classifier is used in this
objective. A separate decoder maps argmax symbols back to proper binary-event probabilities.

Usage from repo root (OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):
  python examples/l20_operators_neural_direction.py fit --input /path/to/cold-replicated.npz --motifs /path/to/development-motifs.npz
  python examples/l20_operators_neural_direction.py score --input /path/to/confirm-09.npz --motifs /path/to/confirm-09-motifs.npz --out results/l20-operators/directional-neural/confirm-09.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp, softmax
from scipy.stats import t as student
from threadpoolctl import threadpool_limits

from l20_operators_directional import csv_write, dump, offsets, clipped_logit, metrics, sha
from l20_operators_directional_geometry import CORE, FACES

TREE = Path('results/l20-operators/directional')
OUT = Path('results/l20-operators/directional-neural')
SEEDS = (11, 29, 47)
K = 16
EPOCHS, STEPS, EVALS = 48, 48, 128
LR, DECAY = .0015, .0005
VARIANTS = ('neural', 'neural-random', 'learned')  # learned = primary frozen tree
ALL = ('direction', 'learned', 'neural-random', 'neural', 'barrier', 'barrier+learned', 'barrier+neural-random', 'barrier+neural')


def load(args, meta, fitting):
    with np.load(args.input, allow_pickle=False) as f:
        ids, state, beta, r, lags = (f[k] for k in ('parent_id','origin','beta','replicas','lags'))
        beta, r = float(beta), int(r)
        check_counts = None if fitting else f['changed']
    if fitting:
        if args.input.name != 'cold-replicated.npz' or sha(args.input)!=meta['input_sha256']:
            raise ValueError('only the checksum-frozen archived bank is accepted for fitting')
        if set(ids)!=set(meta['fit_parent_id']+meta['validation_parent_id']): raise ValueError('incorrect parent split')
    elif np.isclose(beta, 3.05) and set(ids.tolist()) & set(meta['fit_parent_id']+meta['validation_parent_id']):
        raise ValueError('confirmation overlaps archived parents')
    with np.load(args.motifs, allow_pickle=False) as f:
        b={k:f[k] for k in ('parent_id','anchor','axis','sign','h','observed','predicted','n_occupied','lags','fit_lags','models','target_index')}
    if not set(b['parent_id']).issubset(ids) or len(set(ids))!=len(ids): raise ValueError('motif-parent mismatch')
    if fitting and list(b['lags'])!=[4096,16384]: raise ValueError('unexpected archived motif lags')
    if not fitting and not np.array_equal(b['lags'], lags): raise ValueError('confirmation lags differ')
    lookup={int(p):i for i,p in enumerate(ids)}; rows=np.array([lookup[int(p)] for p in b['parent_id']])
    basis=np.eye(3,dtype=int)[(b['axis'][:,None]+np.arange(3))%3]; basis[:,0]*=b['sign'][:,None]
    xyz=(b['anchor'][:,None]+CORE@basis)%20; b['raw']=state[rows[:,None],xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]]
    valid=b['target_index']>=0
    if not np.array_equal(valid.sum(2),b['n_occupied']): raise ValueError('target counts differ')
    for d, face in enumerate(FACES):
        xyz=(b['anchor'][:,None]+face@basis)%20
        idx=np.ravel_multi_index(tuple(xyz.transpose(2,0,1)), (20,20,20))
        occupied=state[rows[:,None],xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]]!=0
        if not np.array_equal(np.where(occupied,idx,-1),b['target_index'][:,d,:len(face)]):
            raise ValueError('target geometry differs from origin')
    if check_counts is not None:
        # Bind response bank to the specified confirmation trajectories, not merely the same IDs.
        for j,c in enumerate(check_counts):
            a=c.reshape(len(ids),-1)[rows[:,None,None],np.maximum(0,b['target_index'])]
            y=(a*valid).sum(2)/np.maximum(b['n_occupied']*float(r),1.)
            mask=b['n_occupied']>0
            if not np.allclose(y[mask],b['observed'][j][mask],atol=1e-7,rtol=0): raise ValueError('incorrect response bank')
    b.update(ids=ids,beta=beta,replicas=r)
    return b


def groups(pid, ids):
    return [ix for p in ids if len(ix:=np.flatnonzero(pid==p))]


def parent_weights(b, ids):
    w=np.zeros(len(b['parent_id']))
    for ix in groups(b['parent_id'],ids): w[ix]=1/len(ix)
    return w/w.sum()


def centered(y, n):
    mask=n>0; y=np.where(mask,y,0.)
    return np.where(mask,y-(y*mask).sum(1,keepdims=True)/mask.sum(1,keepdims=True),0.)


def prepare(b, norm=None, fit_ids=None):
    hot=np.eye(3)[b['raw']].reshape(len(b['raw']),-1)
    z=np.stack([centered(y,b['n_occupied']) for y in b['observed']]); mask=b['n_occupied']>0
    if norm is None:
        w=parent_weights(b,fit_ids); mu=w@hot
        # A common scalar across six directions; preserve ordering and signs. Equal parent/bond/direction.
        s=np.sqrt(w @ ((z[0]**2).sum(1)/mask.sum(1))); norm={'mu':mu.tolist(), 'scale':float(max(s,1e-8))}
    x=2*(hot-norm['mu']); e=np.arcsinh(z/norm['scale'])
    # Empty faces contribute zero and carry an explicit mask (all six nonempty in development).
    ee=np.concatenate((e,np.broadcast_to((~mask).astype(float), e.shape)),axis=2)
    return x,ee,norm


def draws(gr, rng, repeats, k=K):
    k=min(k,len(gr)); rows=np.empty((repeats,k),dtype=int)
    for i in range(repeats):
        sel=rng.choice(len(gr),k,replace=False)
        rows[i]=[g[rng.integers(len(g))] for j in sel for g in (gr[j],)]
    return rows


def initialize(seed):
    rng=np.random.default_rng(seed)
    def matrix(a,b): return rng.normal(0, np.sqrt(2/(a+b)), size=(a,b))
    return dict(A=matrix(108,32),a=np.zeros(32),B=matrix(32,4),b=np.zeros(4),
                C=matrix(12,32),c=np.zeros(32),D=matrix(32,4),d=np.zeros(4))


def forward(x,e,h,m,variant):
    u=np.tanh(x@m['A']+m['a']); q=softmax(u@m['B']+m['b'],axis=-1) if variant!='learned' else np.eye(4)[h]
    v=np.tanh(e@m['C']+m['c']); s=v@m['D']+m['d']
    return u,q,v,s


def contribution(q,s):
    # Shapes (...,K,H); denominator is over environments, not codes.
    return q*(s-logsumexp(s,axis=-2,keepdims=True)+np.log(s.shape[-2]))


def step(x,e,h,m,mom,var,t):
    u,q,v,s=forward(x,e,h,m,var); k=len(x)
    gain=s-logsumexp(s,axis=0,keepdims=True)+np.log(k)
    ds=(q-softmax(s,axis=0)*q.sum(0))/k
    dv=(ds@m['D'].T)*(1-v*v)
    grad=dict(D=v.T@ds,d=ds.sum(0),C=e.T@dv,c=dv.sum(0))
    if var=='neural':
        dl=q*(gain-(q*gain).sum(1,keepdims=True))/k
        du=(dl@m['B'].T)*(1-u*u)
        grad.update(B=u.T@dl,b=dl.sum(0),A=x.T@du,a=du.sum(0))
    for key,g in grad.items():
        if key.isupper(): g=g-DECAY*m[key]
        v1,v2=mom[key]
        v1*=.9; v1+=.1*g; v2*=.999; v2+=.001*g*g
        m[key]+=LR*(v1/(1-.9**t))/(np.sqrt(v2/(1-.999**t))+1e-8)


def evaluate(m,x,e,h,bank,var):
    _,q,_,s=forward(x[bank],e[bank],h[bank],m,var)
    hard=q.argmax(-1)
    return (contribution(q,s).sum(-1), contribution(np.eye(4)[hard],s).sum(-1))


def train(seed,var,x,e,h,train_gr,train_eval,val_eval):
    params=initialize(seed); mom={k:(np.zeros_like(a),np.zeros_like(a)) for k,a in params.items()}
    rng=np.random.default_rng(3000+seed); records=[]; best=None; best_epoch=0; best_score=-np.inf
    # Fixed budget, validation-only stopping point, including the untouched initialization.
    for epoch in range(EPOCHS+1):
        if epoch:
            for i,ids in enumerate(draws(train_gr,rng,STEPS)):
                step(x[ids],e[ids],h[ids],params,mom,var,(epoch-1)*STEPS+i+1)
        tr, th=evaluate(params,x,e,h,train_eval,var); va,vh=evaluate(params,x,e,h,val_eval,var)
        records.append(dict(seed=seed, variant=var, epoch=epoch, fit_nce=tr.mean(),fit_hard_nce=th.mean(),
                            validation_nce=va.mean(),validation_hard_nce=vh.mean()))
        if va.mean()>best_score:
            best_score=float(va.mean());best_epoch=epoch;best={k:v.copy() for k,v in params.items()}
    return best,records,dict(seed=seed,epoch=best_epoch,validation_nce=best_score)


def hardcode(x,m):
    return (np.tanh(x@m['A']+m['a'])@m['B']+m['b']).argmax(1)


def physical(b,j,tree_channel):
    p=b['predicted'][j,b['models'].tolist().index('barrier')]
    # Remove the primary tree pipeline's common direction-only calibration before
    # fitting the matched four-state residual offsets. No recalibration at confirmation.
    delta=np.array(tree_channel[str(b['fit_lags'][j])]['barrier'])[0]
    return expit(clipped_logit(np.nan_to_num(p,nan=.5))-delta)


def fit_decoders(b,hs,ids,tree_channel):
    w=parent_weights(b,ids); n=b['n_occupied']; valid=n>0; w=w[:,None]*valid/valid.sum(1,keepdims=True)
    w*=np.count_nonzero(w)/w.sum(); ix=w.sum(1)>0
    decoder={}
    for j,lag in enumerate(b['fit_lags']):
        y=np.nan_to_num(b['observed'][j]); ray=np.broadcast_to(tree_channel[str(lag)]['direction'],y.shape)
        base=physical(b,j,tree_channel); table={}
        for name,h in hs.items():
            for key,ref in [(name,ray),('barrier+'+name,base)]: table[key]=offsets(h[ix],y[ix],w[ix],ref[ix]).tolist()
        decoder[str(lag)]=table
    return decoder


def predict(b,x,chosen,decoder,tree_channel):
    hs={name:hardcode(x,chosen[name]) for name in ('neural','neural-random')}; pp=[]
    for j,lag in enumerate(b['fit_lags']):
        found={name:b['predicted'][j,b['models'].tolist().index(name)] for name in ('direction','barrier','learned','barrier+learned')}
        phys=physical(b,j,tree_channel); ray=np.array(tree_channel[str(lag)]['direction'])
        for name,h in hs.items():
            for key,base in ((name,ray),('barrier+'+name,phys)):
                found[key]=expit(clipped_logit(base)+np.array(decoder[str(lag)][key])[h])
        pp.append(np.stack([found[key] for key in ALL]))
    return np.stack(pp), hs


def scores(b,pp,split):
    rows=[]
    for pid in b['ids']:
        ix=np.flatnonzero(b['parent_id']==pid); n=b['n_occupied'][ix]; v=n>0
        w=v/(len(v)*v.sum(1,keepdims=True)) if len(ix) else None
        for j,lag in enumerate(b['lags']):
            y=np.nan_to_num(b['observed'][j,ix]);
            for k,name in enumerate(ALL):
                row=dict(parent_id=int(pid),beta=b['beta'],lag=int(lag),fit_lag=int(b['fit_lags'][j]),replicas=b['replicas'],split=split[int(pid)],
                         model=name,n_bonds=len(ix),n_faces=int(v.sum()),n_sites=int(n.sum()))
                row.update(metrics(y,np.nan_to_num(pp[j,k,ix]),w) if len(ix) else dict.fromkeys(('observed','predicted','logloss','brier','direction_mse'),float('nan')))
                rows.append(row)
    return rows


def nce_rows(b,x,e,chosen,split):
    rows=[]
    for key in sorted(set(split.values())):
        ids=[p for p in b['ids'] if split[int(p)]==key]; gr=groups(b['parent_id'],ids)
        rng=np.random.default_rng(828131); order=rng.permutation(len(gr)); blocks=np.array_split(order,int(np.ceil(len(gr)/K)))
        for block_no,block in enumerate(blocks):
            bank=draws([gr[i] for i in block],rng,EVALS)
            for j,lag in enumerate(b['lags']):
                common=dict(split=key,beta=b['beta'],lag=int(lag),critic_fit_lag=4096,parent_block=block_no,k=len(block),repeats=EVALS,
                            parent_ids=';'.join(str(b['parent_id'][gr[i][0]]) for i in block))
                rows.append(dict(common,model='zero',nce_nats=0.,hard_nce_nats=0.))
                for var in VARIANTS:
                    q,hard=evaluate(chosen[var],x,e[j],b['h'],bank,var)
                    rows.append(dict(common,model=var,nce_nats=q.mean(),hard_nce_nats=hard.mean()))
    return rows


def run_fit(a):
    out=a.model; out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists(): raise FileExistsError('refusing to overwrite neural freeze')
    tree_meta=json.loads((TREE/'freeze.json').read_text()); channel=json.loads((TREE/'channel.json').read_text())
    b=load(a,tree_meta,True); train_ids=tree_meta['fit_parent_id']; val_ids=tree_meta['validation_parent_id']; x,e,norm=prepare(b,fit_ids=train_ids)
    tr=groups(b['parent_id'],train_ids); va=groups(b['parent_id'],val_ids)
    train_eval=draws(tr,np.random.default_rng(15582),EVALS); val_eval=draws(va,np.random.default_rng(32789),EVALS)
    selected={}; checkpoints=[]; history=[]; weights={}; files=[]
    for var in VARIANTS:
        byseed=[]
        for seed in SEEDS:
            best,trace,stats=train(seed,var,x,e[0],b['h'],tr,train_eval,val_eval)
            fname=f'{var}-{seed}.npz'; np.savez_compressed(out/fname,**best); files.append(fname)
            weights[var,seed]=best; byseed.append(stats); history.extend(trace); checkpoints.append(dict(variant=var,**stats))
        selected[var]=max(byseed,key=lambda r:r['validation_nce'])
    chosen={var:weights[var,stats['seed']] for var,stats in selected.items()}
    hs={var:hardcode(x,chosen[var]) for var in ('neural','neural-random')}
    decoder=fit_decoders(b,hs,train_ids,channel); dump(out/'decoder.json',decoder)
    pp,_=predict(b,x,chosen,decoder,channel); split={int(p):('fit' if p in train_ids else 'validation') for p in b['ids']}
    csv_write(out/'epochs.csv',history); csv_write(out/'checkpoints.csv',checkpoints); csv_write(out/'development.csv',scores(b,pp,split)); csv_write(out/'development-nce.csv',nce_rows(b,x,e,chosen,split))
    files+=['epochs.csv','checkpoints.csv','development.csv','development-nce.csv','decoder.json']
    here=Path(__file__)
    dump(out/'freeze.json',dict(schema=1, secondary_to='directional/freeze.json', normalization=norm, selection=selected,
          fit_parent_id=train_ids,validation_parent_id=val_ids,seeds=SEEDS, K=K,epochs=EPOCHS,batches_per_epoch=STEPS,validation_draws=EVALS,
          learning_rate=LR, weight_l2=DECAY, objective='Exact expected categorical InfoNCE; K distinct parents, independent motifs, primary lag only. Negative log denominator sums over all K environments for each of four symbols.',
          architecture='q: centered 36x3 core one-hot, tanh(32), 4-way softmax. s: asinh(mean-centered directional propensity/fit RMS), six empty-face flags, tanh(32), four output logits.',
          controls='same initialization neural-random (no encoder updates, independently critic-trained); frozen primary hard tree plus trained nonlinear critic; constant zero critic. Each nonconstant method chooses its own seed/epoch by the same expected NCE on archive validation; argmax decoder is fitted only on old training parents.',
          outcome='six face conditional species-change rates from initially occupied sites and all available children; empirical noisy replicated map, not latent infinite-replica dynamics. No averages over orientation; no claim of static order or causality.',
          uncertainty='proper scores by parent; raw InfoNCE by nonoverlapping groups of up to 16 parents, averaging 128 independent motif resamples conditional on the group, not 128 new parent replications. The single archived validation block is used for selection, not independent final evidence.',
          selected_hard_counts={v:{s:np.bincount(h[np.isin(b['parent_id'],ii)],minlength=4).tolist() for s,ii in [('fit',train_ids),('validation',val_ids)]} for v,h in hs.items()},
          input_sha256=sha(a.input),motifs_sha256=sha(a.motifs),tree_freeze_sha256=sha(TREE/'freeze.json'),
          source_sha256=sha(here),files={k:sha(out/k) for k in files}))
    print(f'Frozen {out}/freeze.json',flush=True)


def run_score(a):
    out=a.model; meta=json.loads((out/'freeze.json').read_text())
    if sha(Path(__file__))!=meta['source_sha256'] or sha(TREE/'freeze.json')!=meta['tree_freeze_sha256']: raise ValueError('scoring source or tree receipt changed')
    tree_meta=json.loads((TREE/'freeze.json').read_text()); channel=json.loads((TREE/'channel.json').read_text())
    if sha(TREE/'channel.json')!=tree_meta['files']['channel.json']: raise ValueError('changed primary decoder')
    for fname in ['decoder.json']+[f"{v}-{s['seed']}.npz" for v,s in meta['selection'].items()]:
        if sha(out/fname)!=meta['files'][fname]: raise ValueError(f'changed frozen {fname}')
    b=load(a,meta,False); x,e,_=prepare(b,meta['normalization']); chosen={}
    for v,s in meta['selection'].items():
        with np.load(out/f"{v}-{s['seed']}.npz",allow_pickle=False) as f: chosen[v]=dict(f)
    pp,hs=predict(b,x,chosen,json.loads((out/'decoder.json').read_text()),channel); split={int(p):'confirmation' for p in b['ids']}
    csv_write(a.out,scores(b,pp,split)); raw=a.out.with_name(a.out.stem+'-nce.csv'); csv_write(raw,nce_rows(b,x,e,chosen,split))
    if a.predictions:
        np.savez_compressed(a.predictions,lags=b['lags'],models=ALL,parent_id=b['parent_id'],axis=b['axis'],anchor=b['anchor'],sign=b['sign'],
                            h=hs['neural'],h_random=hs['neural-random'],predicted=pp.astype(np.float32))
    dump(a.out.with_suffix('.json'),dict(freeze_sha256=sha(out/'freeze.json'),input_sha256=sha(a.input),motifs_sha256=sha(a.motifs),
                                         scores_sha256=sha(a.out),nce_sha256=sha(raw)))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter); sub=p.add_subparsers(dest='cmd',required=True)
    for key in ('fit','score'):
        a=sub.add_parser(key); a.add_argument('--input',type=Path,required=True); a.add_argument('--motifs',type=Path,required=True)
        a.add_argument('--model',type=Path,default=OUT)
        if key=='score': a.add_argument('--out',type=Path,required=True); a.add_argument('--predictions',type=Path)
    args=p.parse_args()
    with threadpool_limits(limits=1): (run_fit if args.cmd=='fit' else run_score)(args)

if __name__=='__main__': main()
