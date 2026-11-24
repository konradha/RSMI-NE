"""Search for a four-state local real-space RG map on released L20 endpoints only.

Run with --mode discover|confirm --round N --data RELEASE --work PRIVATE --out PUBLIC.
Every reported bound is for a *hard* four-state code, not a continuous relaxation.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
from itertools import permutations, product
import json
import os
from pathlib import Path
import sys

for _lib in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_lib] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l20_pairs import L, partition
from l20_operators_static import backward, forward, init, softmax, source, table

B = 64
LAST = 95
SEED = 571361
COLD, WARM, HOT = 9, 7, 0
EYES = {k: np.eye(k, dtype='f') for k in (3, 4)}
BITS = np.indices((2,)*3).reshape(3, -1).T
HAM = np.abs(BITS[:, None] - BITS).sum(2)
PAIR = [np.where(np.triu(HAM == d, k=1)) for d in (1, 2, 3)]
SECTORS = ('vector', 'staggered', 'scalar')
RULES = ('species-A', 'species-B', 'vacancy', 'composition', 'checkerboard', 'surface', 'energy')
# In each successive block of four forced arms, cover both depths and samplings.
ARMS = [(depth, layout, sector, gap) for gap in (1, 2) for sector in SECTORS
        for depth, layout in ((2, 'tile'), (1, 'slide'), (2, 'slide'), (1, 'tile'))]
PROTOCOL = json.loads((Path(__file__).resolve().parents[2] / 'results/l20-autoresearch/rg/protocol.json').read_text())
HYP, METRIC = PROTOCOL['hypothesis'], PROTOCOL['metric']
PRIMARY = ('learned', 'gain', 'paired_minus_shuffled')


def put_json(path, x):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def records(out, last):
    return [json.loads((out/f'round-{r:04d}.json').read_text()) for r in range(last+1)]


def champion(log):
    return max(log, key=lambda rec: rec['selection_score'])


def width(depth, layout):
    return 2**depth


def pitch(depth, layout):
    return width(depth, layout) if layout == 'tile' else 1


@lru_cache(maxsize=40)
def geometry(depth, layout, gap):
    w = width(depth, layout); q = 1 if depth == 1 else 2
    v = np.indices((w,)*3).reshape(3, -1).T
    child = np.ravel_multi_index((2*BITS[:, None]+BITS).transpose(2,0,1), (w,)*3) if depth == 2 else None
    # The near and far faces face each other across the V block, with an explicit
    # empty raw-lattice buffer. Tangential patch starts cover both ends of the face.
    e = []
    cell = np.indices((q,)*3).reshape(3, -1).T
    for axis, side, u, h in product(range(3), (-1,1), (0,w-q), (0,w-q)):
        off = [u,h]; off.insert(axis, -gap-q if side == -1 else w+gap)
        e.extend(cell+off)
    e = np.asarray(e); d = abs(((e[:,None]-v + L//2) % L) - L//2).max(axis=2)
    if d.min() <= gap or np.ptp(np.vstack((v,e)), axis=0).max() >= L:
        raise ValueError('Aliased or touching raw support')
    if layout == 'tile' and (L%w or (child is not None and len(np.unique(child)) != w**3)):
        raise ValueError('Nonpartitioning tile geometry')
    return table(v), table(e), child


def samples(state, rows, arm, rng):
    depth, layout, _, gap = arm; vt, et, _ = geometry(depth, layout, gap)
    s = pitch(depth,layout); xyz = s*rng.integers(L//s, size=(3,len(rows)))
    a = np.ravel_multi_index(tuple(xyz), (L,)*3)
    v = state[rows[:,None], vt[a]]; e = state[rows[:,None], et[a]].reshape(len(rows),24,-1)
    env = np.stack([(e == j).mean(axis=2, dtype='f') for j in (1,2)],axis=-1).reshape(len(rows),48)
    return v, 2*env-0.6


def batches(state, train, arm, n, epochs, stream):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, *stream]))
    for _ in range(epochs):
        rows = rng.choice(train, size=n, replace=False)
        yield samples(state, rows, arm, rng)


def features(raw, k, sector):
    x = EYES[k][raw]
    if sector == 'vector':
        # Retain all signed components: averaging orientations would kill these modes.
        return np.einsum('nsc,si->nic', x, (2*BITS-1).astype('f'), optimize=True).reshape(len(x),-1)/2
    if sector == 'staggered':
        return (x * (1-2*(BITS.sum(1)%2)).astype('f')[None,:,None]).sum(1)/2
    return np.concatenate([x.sum(1)/4, *[(x[:,a]*x[:,b]).sum(1)/np.sqrt(np.float32(len(a))) for a,b in PAIR]], axis=1)


def code(enc, x):
    return forward(x,enc)[1].argmax(1).astype('B')


def inputs(v, arm, first):
    depth, layout, sector, gap = arm
    if depth == 1: return features(v,3,sector)
    ci = geometry(depth,layout,gap)[2]
    low = code(first,features(v[:,ci].reshape(-1,8),3,sector)).reshape(-1,8)
    return features(low,4,sector)


def netcode(v, arm, model):
    first, second = model
    return code(second if arm[0]==2 else first, inputs(v,arm,first))


class Adam:
    def __init__(self, params):
        self.p = params; self.m = [np.zeros_like(x) for x in params]; self.v = [np.zeros_like(x) for x in params]; self.i = 0

    def step(self, grads):
        self.i += 1; rate = .003*np.sqrt(1-.999**self.i)/(1-.9**self.i)
        for p,m,v,g in zip(self.p,self.m,self.v,grads):
            m *= .9; m += .1*g; v *= .999; v += .001*g*g; p -= rate*m/(np.sqrt(v)+1e-8)


def nce_logits(scores):
    z = scores-scores.max(axis=-2,keepdims=True)
    return z-np.log(np.exp(z).mean(axis=-2,keepdims=True))


def fit_encoder(state, train, arm, first, n, epochs, seed):
    dim = inputs(np.zeros((1,width(*arm[:2])**3),'B'),arm,first).shape[1]
    rng = np.random.default_rng(seed); enc = init(rng, dim, enc=True); crit = init(rng,48)
    # Start off the uniform categorical saddle without privileging an orientation.
    enc[3][:] = rng.normal(0,.4/np.sqrt(dim),enc[3].shape)
    opt = Adam(enc+crit)
    for v,e in batches(state,train,arm,n,epochs,[1,seed,arm[0]]):
        x = inputs(v,arm,first)
        for j in range(0,n,B):
            sl = slice(j,j+B); h, a = forward(x[sl],enc); p = softmax(a); hc,z = forward(e[sl],crit)
            r = nce_logits(z); da = p*((p*r).sum(1,keepdims=True)-r)/B
            dz = (softmax(z.T).T*p.sum(0)-p)/B
            opt.step(backward(x[sl],h,enc,da)+backward(e[sl],hc,crit,dz))
    return enc


@lru_cache(maxsize=3)
def cube(w):
    g = np.indices((w,)*3).reshape(3,-1).T
    links = np.abs(g[:,None]-g).sum(2)==1
    exposed = 6-links.sum(1)
    return (1-2*(g.sum(1)%2)).astype('f'), links.astype('f'), exposed.astype('f')


def quantities(v, rho):
    w = round(v.shape[1]**(1/3)); parity,links,exposed = cube(w)
    occ = (v!=0).astype('f'); count = (v==1).sum(1); vacancy = (v==0).sum(1)
    mc = occ@links + exposed*rho
    energy = (occ*((mc-np.array([0,3,5],dtype='f')[v])**2+exposed*rho*(1-rho))).sum(1)
    return {'composition': np.column_stack((count,vacancy)), 'checkerboard':occ@parity,
            'surface':occ@exposed, 'energy':energy}


def calibrate(v):
    rho = float((v!=0).mean()); q = quantities(v,rho)
    return dict(rho=rho, **{k: (np.median(x,axis=0) if k=='composition' else np.quantile(x,[.25,.5,.75])).tolist() for k,x in q.items()})


def fixed(v, cal):
    out = {k: 2*(v[:,0]==s)+(v[:,-1]==s) for k,s in zip(RULES[:3], (1,2,0))}
    for k,val in quantities(v,cal['rho']).items():
        if k == 'composition':
            b = val>cal[k]; out[k] = 2*b[:,0]+b[:,1]
        else: out[k] = np.searchsorted(cal[k],val)
    return out


def all_codes(v, arm, models, cal):
    by = fixed(v,cal)
    for i,m in enumerate(models): by[f'learned{i}'] = netcode(v,arm,m)
    return by


def fit_critics(state, train, arm, models, cal, n, epochs, r):
    names = [*RULES, 'learned0','learned1']
    critics = {k:init(np.random.default_rng(SEED+43),48) for k in names}
    opts = {k:Adam(c) for k,c in critics.items()}
    for v,e in batches(state,train,arm,n,epochs,[2,r]):
        codes = all_codes(v,arm,models,cal)
        for k,c in critics.items():
            p = EYES[4][codes[k]]
            for j in range(0,n,B):
                sl = slice(j,j+B); hc,z = forward(e[sl],c)
                dz = (softmax(z.T).T*p[sl].sum(0)-p[sl])/B
                opts[k].step(backward(e[sl],hc,c,dz))
    return critics


def measure(state, rows, arm, models, cal, critics, idx, fold, reps):
    rng = np.random.default_rng(np.random.SeedSequence([SEED,31,idx,fold])); ng = len(rows)//B
    groups = {k:np.zeros((ng,2)) for k in critics}; hist = {k:np.zeros(4,int) for k in critics}
    for _ in range(reps):
        v,e = samples(state,rows,arm,rng); codes = all_codes(v,arm,models,cal)
        for k,c in critics.items():
            h = codes[k]; hist[k] += np.bincount(h,minlength=4)
            r = nce_logits(forward(e,c)[1].astype(float).reshape(ng,B,4)); shuffled = (r.sum(1,keepdims=True)-r)/(B-1)
            for j,s in enumerate((r,shuffled)):
                groups[k][:,j] += np.take_along_axis(s,h.reshape(ng,B,1),axis=2).mean((1,2))/reps
    return groups, {k:v.tolist() for k,v in hist.items()}


def align(v, arm, models, transform=None):
    w = width(*arm[:2]); a = netcode(v,arm,models[0]); mesh = np.arange(w**3).reshape((w,)*3)
    def agree(order,sign,perm=None):
        ix = mesh.transpose(order)[tuple(slice(None,None,s) for s in sign)].ravel()
        b = netcode(v[:,ix],arm,models[1]); conf = np.bincount(4*b+a,minlength=16).reshape(4,4)
        if perm is None: perm = linear_sum_assignment(-conf)[1].tolist()
        score = float(np.mean(a==np.asarray(perm)[b])); chance = float(np.dot(np.bincount(a,minlength=4),np.bincount(np.asarray(perm)[b],minlength=4)) / len(a)**2)
        return dict(agreement=score, marginal_chance=chance, axes=list(order), signs=list(sign), label_permutation=perm)
    if transform is not None: return agree(transform['axes'],transform['signs'],transform['label_permutation'])
    same = agree((0,1,2),(1,1,1)); best = same
    for order,sign in product(permutations(range(3)),product((1,-1),repeat=3)):
        trial = agree(order,sign)
        if trial['agreement']>best['agreement']: best=trial
    return {'fixed_frame':same,'cube_and_labels':best}


def archive(path, models, critics):
    arrays = {f'm{i}s{j}p{k}':x for i,model in enumerate(models) for j,stage in enumerate(model) for k,x in enumerate(stage)}
    arrays.update({f'c{name}p{k}':x for name,critic in critics.items() for k,x in enumerate(critic)})
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(path, rec):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    with np.load(path,allow_pickle=False) as f:
        def weights(prefix): return [f[prefix+str(k)] for k in range(5) if prefix+str(k) in f]
        return [(weights(f'm{i}s0p'),weights(f'm{i}s1p')) for i in range(2)], {k:weights(f'c{k}p') for k in (*RULES,'learned0','learned1')}


def interval(x, endpoints=18):
    n=len(x); est=float(np.mean(x)); se=float(np.std(x,ddof=1)/np.sqrt(n))
    def ci(alpha):
        d=float(t.ppf(1-alpha/2,n-1))*se; return [est-d,est+d]
    return dict(mean=est,se=se,pointwise95=ci(.05), simultaneous95=ci(.05/endpoints), independent_parent_groups=n)


def discover_plot(out, recs):
    fig,ax=plt.subplots(figsize=(7,3.5)); x=[d['round'] for d in recs]; a=np.array([d['val_learned'] for d in recs]); b=np.array([d['val_competitor'] for d in recs]); ax.plot(x,a,'o-',ms=3,label='Learned hard code'); ax.plot(x,b,'s-',ms=3,label='Best analytic code'); ax.axhline(0,c='.5',lw=.7); ax.set(xlabel='Discovery round (arm / training batch varies)',ylabel='Cold validation InfoNCE [nats]'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'validation.png',dpi=140); plt.close(fig)


def discover(args):
    r=args.round; path=args.out/f'round-{r:04d}.json'
    if path.exists(): return
    if (args.work/'frozen.json').exists(): raise ValueError('Discovery closed by confirmation lock')
    if not 0<=r<=LAST: raise ValueError('Discovery round outside protocol')
    previous=records(args.out,r-1); exploit=(r%4==3)
    arm=tuple(champion(previous)['arm']) if exploit else ARMS[(r-(r//4))%len(ARMS)]
    depth, layout, sector, gap=arm
    n,epochs=(2048,10) if r==0 else ((6144,24) if r<32 else (10240,36 if r<64 else 64))
    state,meta=source(args.data,COLD); train,val,_=partition(COLD)[0]
    cal=calibrate(next(batches(state,train,arm,n,1,[3,r]))[0]); models=[]; seeds=[SEED+104729*r+i*1543 for i in range(2)]
    for seed in seeds:
        first=fit_encoder(state,train,(1,layout,sector,gap),[],n,epochs,seed)
        second=fit_encoder(state,train,arm,first,n,epochs,seed+1987) if depth==2 else []
        models.append((first,second))
    critics=fit_critics(state,train,arm,models,cal,n,epochs,r)
    scores,counts=measure(state,val,arm,models,cal,critics,COLD,1,4)
    means={k:float(s[:,0].mean()) for k,s in scores.items()}; base=max(RULES,key=means.get)
    def ranking(k): return min(means[k],means[k]-means[base],float((scores[k][:,0]-scores[k][:,1]).mean()))
    key=max(('learned0','learned1'),key=ranking); v,_=samples(state,val,arm,np.random.default_rng(SEED+51))
    ck=archive(args.work/f'round-{r:04d}.npz',models,critics)
    rec=dict(mode='discover',round=r,status='ok' if ranking(key)>0 else 'inconclusive', hypothesis=HYP,metric=METRIC.replace('held-out','validation'),estimate=means[key]-means[base],arm=list(arm),
        selection_basis=PROTOCOL['discovery'],selection_score=ranking(key),schedule='exploit' if exploit else 'forced',selected=key,competitor=base,val_learned=means[key],val_competitor=means[base],
        val_bounds=means,val_shuffled={k:float(s[:,1].mean()) for k,s in scores.items()},seed_alignment=align(v,arm,models),calibration=cal,checkpoint_sha256=ck,
        data=meta,params=dict(batch=B,training_parents_per_epoch=n,epochs_per_stage=epochs,seeds=seeds,arm_axes=['depth','layout','sector','empty_raw_layers'],width=width(depth,layout),field_pitch=pitch(depth,layout),signed_channels=True),
        counts=dict(train_pool=len(train),validation_parents=len(val),validation_anchors_per_parent=4,validation_hard_state_counts=counts),provenance=PROTOCOL,search_exhausted=r==LAST)
    discover_plot(args.out,[*previous,rec]); put_json(path,rec)


def spatial(state, rows, arm, model):
    # Compute full-grid maps for whole parents, never use adjacent overlapping
    # patches as independent measurements. FFT circular products are per parent.
    depth,layout,_,gap=arm; w=width(depth,layout); s=pitch(depth,layout); side=L//s
    vt=geometry(depth,layout,gap)[0]; starts=np.indices((side,)*3).reshape(3,-1)*s
    ix=vt[np.ravel_multi_index(tuple(starts),(L,)*3)]; counts=[]; powers=[]; crosses=[]
    lags=[d for d in range(1,side//2+1) if min(d*s,L-d*s)-(w-1)>gap]
    freqs=np.rint(np.fft.fftfreq(side)*side).astype(int); k2=sum(x*x for x in np.meshgrid(freqs,freqs,freqs,indexing='ij')); ks=np.unique(k2)
    for parent in rows:
        v=state[parent,ix]
        h=np.concatenate([netcode(v[j:j+1024],arm,model) for j in range(0,len(v),1024)])
        f=EYES[4][h].reshape(side,side,side,4); counts.append(f.mean((0,1,2)))
        fft=np.fft.fftn(f,axes=(0,1,2)); power=(abs(fft)**2).sum(3)/side**3
        powers.append([power[k2==k].mean() for k in ks if k>0]); crosses.append([[f.reshape(-1,4).T@np.roll(f,-d,axis=axis).reshape(-1,4)/side**3 for axis in range(3)] for d in lags])
    p=np.mean(counts,axis=0).astype(float); mu=np.outer(p,p); cov=np.diag(p)-mu; eig,U=np.linalg.eigh(cov); use=eig>1e-7; whitening=U[:,use]/np.sqrt(eig[use]); means=np.mean(crosses,axis=0)
    corr=[]
    for j,d in enumerate(lags):
        # Leave-one-parent-out jackknife includes uncertainty of the subtraction.
        cross=np.asarray(crosses)[:,j]; tr=cross.trace(axis1=2,axis2=3); m=len(rows); loo=(m*p-np.asarray(counts))/(m-1); jk=(m*tr.mean(0)-tr)/(m-1)-(loo**2).sum(1,keepdims=True); se=np.sqrt((m-1)/m*((jk-jk.mean(0))**2).sum(0))
        for a in range(3):
            conn=means[j,a]-mu; modes=np.linalg.eigvalsh(whitening.T@((conn+conn.T)/2)@whitening).tolist()
            corr.append(dict(raw_distance=d*s,axis='zyx'[a],trace=float(np.trace(conn)),parent_jackknife_se=float(se[a]),whitened_eigenvalues=modes))
    power=np.asarray(powers); return dict(depth=depth,layout=layout,raw_width=w,grid_pitch=s,grid_side=side,parents=len(rows),probabilities=p.tolist(),connected=corr,
        nonzero_spectrum=[dict(k2_integer=int(k),multiplicity=int((k2==k).sum()),trace_power=float(power[:,j].mean()),parent_se=float(power[:,j].std(ddof=1)/np.sqrt(len(rows)))) for j,k in enumerate(ks[ks>0])],
        spectral_normalization='Trace of hard-indicator structure factor at nonzero k; FFT(k)^2 divided by number of coarse sites. k=2*pi*n/L in raw site units; averaged within squared integer radii.')


def confirm_plot(out, data):
    fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained'); betas=[v['data']['beta'] for v in data]
    for k,col in [('learned','#15578a'),('control','#bb6433'),('paired_minus_shuffled','#3b8766')]:
        vals=[v['contrasts'][k] for v in data]; y=np.array([q['mean'] for q in vals]); lo,hi=np.array([q['pointwise95'] for q in vals]).T; axs[0].errorbar(betas,y,yerr=[y-lo,hi-y],marker='o',label=k,color=col)
    axs[0].axhline(0,c='.5',lw=.7); axs[0].set(xlabel='Inverse temperature β',ylabel='Nats',title='Frozen hard code · pointwise 95%'); axs[0].legend(fontsize=7,frameon=False)
    for v in data:
        field=v['spatial'][-1]; pts=[p for p in field['connected'] if p['axis']=='z']; spec=field['nonzero_spectrum']; label=f"β={v['data']['beta']:.3g}"
        axs[1].errorbar([p['raw_distance'] for p in pts],[p['trace'] for p in pts],yerr=[1.96*p['parent_jackknife_se'] for p in pts],label=label,marker='o'); axs[2].plot([np.sqrt(p['k2_integer']) for p in spec],[p['trace_power'] for p in spec],'.-',label=label)
    axs[1].axhline(0,c='.5',lw=.7); axs[1].set(xlabel='Raw center separation',ylabel='Connected state trace',title='Buffered, disjoint · z axis'); axs[2].set(xlabel='|k| L / 2π',ylabel='Structure factor trace',title='All nonzero Fourier shells'); axs[2].legend(fontsize=8,frameon=False); fig.savefig(out/'confirmation.png',dpi=150); plt.close(fig)


def confirm(args):
    path=args.out/'confirmation.json'; lock=args.work/'frozen.json'
    if path.exists(): return
    if lock.exists():
        frozen=json.loads(lock.read_text())
        if frozen['through_round']!=args.round: raise ValueError('Cannot change a frozen confirmation prefix')
    else:
        best=champion(records(args.out,args.round)); frozen=dict(through_round=args.round,best=best); put_json(lock,frozen)
    best=frozen['best']; arm=tuple(best['arm']); chosen=best['selected']; control=best['competitor']; j=int(chosen[-1]); r=best['round']; cal=best['calibration']; models,critics=restore(args.work/f'round-{r:04d}.npz',best)
    data=[]
    for idx in (COLD,WARM,HOT):
        state,meta=source(args.data,idx); test=partition(idx)[0][2]
        if idx==COLD and meta['sha256']!=best['data']['sha256']: raise ValueError('Cold archive differs from discovery')
        scores,counts=measure(state,test,arm,models,cal,critics,idx,2,12)
        p,sh=scores[chosen].T; baseline=scores[control][:,0]; cont={k:interval(x) for k,x in dict(learned=p,gain=p-baseline,paired_minus_shuffled=p-sh,shuffled=sh,control=baseline).items()}
        v,_=samples(state,test,arm,np.random.default_rng(SEED+109+idx)); alignment={k:align(v,arm,models,tf) for k,tf in best['seed_alignment'].items()}
        other='tile' if arm[1]=='slide' else 'slide'
        fields=[spatial(state,test[:48],(d,layout,*arm[2:]),models[j])
                for d in range(1,arm[0]+1) for layout in (other,arm[1])]
        data.append(dict(data=meta,parents=len(test),anchors_per_parent=12,hard_state_counts=counts,contrasts=cont,other_frozen_bounds={k:interval(x[:,0]) for k,x in scores.items()},alignment=alignment,spatial=fields)); del state
    primary=all(temp['contrasts'][k]['simultaneous95'][0]>0 for temp in data[:2] for k in PRIMARY)
    rec=dict(mode='confirm',round=args.round,status='ok' if primary else 'inconclusive',hypothesis=HYP,metric=METRIC,estimate=data[0]['contrasts']['gain']['mean'],selection_basis=PROTOCOL['discovery'],selected_round=r,selected=chosen,competitor=control,arm=list(arm),checkpoint_sha256=best['checkpoint_sha256'],primary_pass=primary,primary_contrasts=PRIMARY,
        results=data,params=best['params'],data=[v['data'] for v in data],counts=dict(test_parents_per_beta=[v['parents'] for v in data],spatial_parents_per_beta=48,negative_group_size=B),provenance=PROTOCOL)
    csvpath=args.out/'confirmation.csv'; tmp=csvpath.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['beta','contrast','estimate_nats','point_lo','point_hi','primary_family_lo','primary_family_hi'])
        for temp in data:
            for k,val in temp['contrasts'].items(): wr.writerow([temp['data']['beta'],k,val['mean'],*val['pointwise95'],*val['simultaneous95']])
    tmp.replace(csvpath); confirm_plot(args.out,data); put_json(path,rec)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--mode',choices=('discover','confirm'),required=True); p.add_argument('--round',type=int,required=True)
    for key in ('data','work','out'): p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args(); args.data=args.data.expanduser(); args.out.mkdir(parents=True,exist_ok=True); args.work.mkdir(parents=True,exist_ok=True)
    (discover if args.mode=='discover' else confirm)(args)


if __name__=='__main__': main()
