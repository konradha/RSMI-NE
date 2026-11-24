"""#25. Actual shared hard LUTs, deterministic block-coordinate credit assignment.

Fit: --rung {256,1024,4096}; --extend repeats the deterministic largest rung to 8.
Audit: --audit-release PUBLICATION --checkpoint N, after explicit release only.
The original archive is not read here. Each mode opens only its prescribed bank slice.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from tranche2_hierarchy_kernels import Kernels

WORK = Path.home()/'rsmi-work/l20-autoresearch-1795520725/tranche2'
OUT = Path(__file__).resolve().parents[2]/'results/l20-autoresearch/tranche2'
SEEDS = (251,252,253)
WD = (np.arange(6561)[:,None]//3**np.arange(7,-1,-1)%3).astype(np.uint8)
TD = (np.arange(65536)[:,None]//4**np.arange(8)%4).astype(np.uint8)


def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def select(z, name, lo, hi):
    with z.open(name+'.npy') as f:
        version = np.lib.format.read_magic(f)
        shape, order, dtype = { (1,0): np.lib.format.read_array_header_1_0, (2,0): np.lib.format.read_array_header_2_0 }[version](f)
        assert not order and shape[0]==5120 and 0<=lo<hi<=5120
        stride = int(np.prod(shape[1:]))*dtype.itemsize
        f.seek(f.tell()+lo*stride)
        return np.frombuffer(f.read((hi-lo)*stride), dtype=dtype).reshape(hi-lo,*shape[1:])


def load(k, lo, hi):
    t=perf_counter()
    with ZipFile(WORK/'b09.npz') as z:
        ids = {key: select(z,key,lo,hi) for key in ('parent','chain_id','chain_seed')}
        s = select(z,'species',lo,hi)
    assert s.dtype==np.uint8 and s.shape[1:]==(20,20,20) and s.max()<=2
    assert len(np.unique(ids['parent']))==hi-lo
    assert len(np.unique(np.column_stack([ids['chain_seed'],ids['chain_id']]),axis=0))==hi-lo
    w,y,yc = k.extract(s)
    # Spot-check all 64 sites, including a wrapping origin, without a second physical state.
    for p,i,j,l in ((0,0,0,0),(hi-lo-1,9,9,9)):
        q=1000*p+100*i+10*j+l
        at=lambda a,b,c: int(s[p,(2*i+a)%20,(2*j+b)%20,(2*l+c)%20])
        def tern(d):return sum(v*3**h for h,v in enumerate(reversed(d)))
        expected=[tern([at(2*x+h,2*b+t,2*c+u) for h,t,u in np.ndindex(2,2,2)]) for x,b,c in np.ndindex(2,2,2)]
        assert np.array_equal(w[q],expected) and y[q]==tern([at(6,b+1,c+1) for b,c in np.ndindex(2,2)]) and yc[q]==tern([at(3,b,c) for b,c in np.ndindex(2,2)])
    return w,y,yc,ids,perf_counter()-t


def marginal(y):
    a=np.bincount(y,minlength=81)+.5
    return a/a.sum()


def decoder(counts):
    prob=counts.astype(np.float64)+.5
    return prob/prob.sum(1,keepdims=True)


def grouped(data, mapping, size):
    return np.stack([data[mapping==i].sum(0) for i in range(size)])


def backoff(c, digits, categories, prior):
    """Shrink each sparse conditional word row to its summed single-site effects."""
    counts=c.sum(1,dtype=np.float64)
    base=(c.sum(0,dtype=np.float64)+.5)/(counts.sum()+40.5)
    pred=np.broadcast_to(base,(len(c),81)).copy()
    for d in digits.T:
        a=grouped(c,d,categories).astype(np.float64)+40.5*base
        pred+=a[d]/a.sum(1)[d,None]-base
    np.maximum(pred,.05*base,out=pred); pred/=pred.sum(1,keepdims=True)
    return ((c+prior*pred)/(counts[:,None]+prior)).astype(np.float32),counts


def initial(post, counts, base, seed):
    """Four approximately equally spaced signed contrasts in the two strongest response directions."""
    q=(post-base)/np.sqrt(base)
    cov=(q.T*(counts/counts.sum()))@q
    _,vec=np.linalg.eigh(cov)
    phi=np.random.default_rng(seed).uniform(0,np.pi/2)
    angles=phi+np.pi/2*np.arange(4)
    dirs=(np.stack([np.cos(angles),np.sin(angles)],1)@vec[:,-2:].T)/np.sqrt(base)
    sd=np.sqrt((dirs**2*base).sum(1,keepdims=True))
    probs=base*np.exp(.04*np.clip(dirs/sd,-10,10))
    return probs/probs.sum(1,keepdims=True)


def assign(post,d):
    return (post@np.log(d).T.astype(np.float32)).argmax(1).astype(np.uint8)


def intervals(v):
    return {'mean':float(np.mean(v)), 'se':float(np.std(v,ddof=1)/np.sqrt(len(v))),
            'interval95':(v.mean()+np.array([-1,1])*1.96*v.std(ddof=1)/np.sqrt(len(v))).tolist(),
            'mde80':float(2.8*v.std(ddof=1)/np.sqrt(len(v)))}


def measure(k,h,y,hd,yd,base):
    tab=k.hist('tally',(4,81),h,y)
    dec=decoder(tab);value=np.log(dec/base)
    out=np.zeros(len(hd)//1000)
    k.run('parentmean',len(out),hd,yd,value,out)
    pc=tab.sum(1)/tab.sum(); dc=np.bincount(hd,minlength=4)/len(hd)
    stat={'fit_score':float((tab*value).sum()/len(h)), 'dev_score':intervals(out),
          'fit_final_freq':pc.tolist(),'dev_final_freq':dc.tolist(),
          'dev_entropy_bits':float(-(dc[dc>0]*np.log2(dc[dc>0])).sum())}
    return stat,out,dec


def greedy(k,w,y,seed):
    word=np.ascontiguousarray(w[:,0]);tab=k.hist('tophist',(6561,81),word,y)
    post,n=backoff(tab,WD,3,1000)
    base=marginal(y);rule=assign(post,initial(post,n,base,seed));curve=[]
    for _ in range(48):
        c=grouped(tab,rule,4);d=decoder(c);new=assign(post,d)
        curve.append({'hard_fit':float((c*np.log(d/base)).sum()/n.sum()),'changed_words':int(np.sum(new!=rule))})
        if np.array_equal(new,rule):break
        rule=new
    return rule,curve


def fit_top(k,w,y,child):
    tup=k.codes(w,child);c=k.hist('tophist',(65536,81),tup,y)
    post,n=backoff(c,TD,4,3000)
    return tup,post,n


def proposal(k,w,y,child,top,tup,value):
    # dE / d child_onehot[w,k] at each occurrence of the deterministic shared categoricals,
    # keeping the other 7 children fixed. Simultaneous word replacements need not be additive.
    g=np.zeros((6561,4));k.run('adjoint',len(w),w,tup,y,top,value,g)
    n=8*np.bincount(w[:,0],minlength=6561).astype(float)
    # Every even child occurs once in each position on the periodic even grid.
    g-=g.mean(1,keepdims=True)
    mean=g.sum(0)/n.sum();pooled=np.broadcast_to(mean,g.shape).copy()
    for d in WD.T:
        for v in range(3):
            m=d==v;pooled[m]+=g[m].sum(0)/max(n[m].sum(),1)-mean
    grad=(g+4000*pooled)/(n+4000)[:,None]
    pref=grad.argmax(1).astype(np.uint8);idx=np.arange(len(n));gain=grad[idx,pref]-grad[idx,child]
    order=np.argsort(-gain);order=order[(gain[order]>0)&(n[order]>0)]
    cum=np.cumsum(n[order]); current=float(value[top[tup],y].mean());best=current; chosen=child
    for fraction in (.12,.04,.012):
        ids=order[:min(len(order),1+np.searchsorted(cum,fraction*n.sum()))]
        cand=child.copy();cand[ids]=pref[ids]
        t=k.codes(w,cand);score=float(value[top[t],y].mean())
        if score>best+1.e-12:best=score;chosen=cand
    changed=(chosen!=child)
    diagnostics={'finite_child_grad_rms':float(np.sqrt((n[:,None]*grad**2).sum()/(4*n.sum()))),
                 'unshrunk_grad_rms':float(np.sqrt((g**2/np.maximum(1,n[:,None])).sum()/(4*n.sum()))),
                 'updated_child_words':int(changed.sum()),'updated_fit_child_mass':float(n[changed].sum()/n.sum()),
                 'verified_hard_gain_before_top_refit':best-current}
    return chosen,diagnostics


def rawpost(k,w,y):
    hist=k.hist('rawhist',(8,6561,81),w,y)
    out=np.empty(hist.shape,np.float32);ns=[]
    for j,c in enumerate(hist):out[j],n=backoff(c,WD,3,2500);ns.append(n)
    return out,np.array(ns)


def rawrule(post,base,dec):
    # A single four-logit classifier of 64 raw sites (8 word-by-position feature tables).
    # Subtract the seven duplicated intercepts; only the final output is quantized.
    return np.ascontiguousarray((post@np.log(dec).T.astype(np.float32))-.875*(base@np.log(dec).T).astype(np.float32))


def fit(rung,steps):
    k=Kernels(WORK);start=perf_counter()
    w,y,yc,fitids,t0=load(k,0,rung);wd,yd,_,dids,t1=load(k,4096,4608)
    keys=lambda g:set(zip(g['chain_seed'].tolist(),g['chain_id'].tolist()))
    assert not keys(fitids)&keys(dids) and not np.intersect1d(fitids['parent'],dids['parent']).size
    np.savez_compressed(OUT/f'25-split-{rung}.npz',**{'fit_'+a:b for a,b in fitids.items()},**{'dev_'+a:b for a,b in dids.items()})
    base=marginal(y);t=perf_counter();raw,nraw=rawpost(k,w,y);tproj=perf_counter()-t
    out={'rung':rung,'anchors_per_parent':1000,'batch':'all %d faces each sweep'%len(y),'seeds':{},'raw_suffstat_seconds':tproj,
         'feature_extract_seconds':[t0,t1],'source':'b09 native TRAIN, no pooled maps', 'common_unique_parent_keys':0}
    vectors={}
    for seed in SEEDS:
        t=perf_counter();child0,greedycurve=greedy(k,w,yc,seed);greedytime=perf_counter()-t
        t=perf_counter();tuple0,post0,n0=fit_top(k,w,y,child0)
        top0=assign(post0,initial(post0,n0,base,seed));inittime=perf_counter()-t
        sd={'greedy_child_training':greedycurve,'greedy_child_seconds':greedytime,'shared_top_init_seconds':inittime,'arms':{}}
        out['seeds'][str(seed)]=sd
        for arm in ('raw','frozen','joint'):
            t=perf_counter(); curve=[]
            if arm=='raw':
                d=initial(raw.reshape(-1,81),nraw.ravel(),base,seed);rule=rawrule(raw,base,d)
            else:
                child=child0.copy();top=top0.copy();tup=tuple0;post=post0
            for s in range(steps+1):
                tc=perf_counter();extra={}
                if arm=='raw':
                    if s:rule=rawrule(raw,base,dec)
                    h=k.raw(w,rule);hd=k.raw(wd,rule)
                else:
                    if s:
                        if arm=='joint':
                            new,extra=proposal(k,w,y,child,top,tup,np.ascontiguousarray(np.log(dec/base)))
                            if not np.array_equal(new,child):
                                child=new;tup,post,_=fit_top(k,w,y,child)
                        prop=assign(post,dec)
                        # Accept using actual (not smoothed surrogate) hard fit log score, holding decoder fixed.
                        val=np.log(dec/base)
                        if val[prop[tup],y].mean()>=val[top[tup],y].mean():top=prop
                    h=top[tup];hd=top[k.codes(wd,child)]
                    extra['child_frequencies']= (np.bincount(child[w[:,0]],minlength=4)/len(w)).tolist()
                    extra['child_fit_mass_changed_since_greedy']=float(np.mean(child[w[:,0]]!=child0[w[:,0]]))
                stat,vec,dec=measure(k,h,y,hd,yd,base)
                stat.update(extra,sweep=s,elapsed_s=perf_counter()-t,step_seconds=perf_counter()-tc)
                curve.append(stat)
                if s in (0,1,2,4,8):vectors[f'{arm}_{seed}_{s}']=vec
                if s==steps and rung==4096:
                    target=OUT/f'25-model-{arm}-{seed}-{steps}.npz'
                    args={'rule':rule} if arm=='raw' else {'child':child,'top':top}
                    np.savez_compressed(target,dec=dec,base=base,**args)
            sd['arms'][arm]=curve
            print(rung,seed,arm,steps,curve[-1]['dev_score'],f'{perf_counter()-t:.1f}s',flush=True)
        assert np.array_equal(vectors[f'joint_{seed}_0'],vectors[f'frozen_{seed}_0'])
        dump(f'25-curves-{rung}-{steps}.json',out)
    out['elapsed_seconds']=perf_counter()-start
    for s in (1,2,4,8):
        if s<=steps:
            diff=np.mean([vectors[f'joint_{q}_{s}']-vectors[f'frozen_{q}_{s}'] for q in SEEDS],axis=0)
            out.setdefault('paired_seed_mean',{})[s]=intervals(diff)
    np.savez_compressed(OUT/f'25-dev-parents-{rung}-{steps}.npz',**vectors)
    dump(f'25-curves-{rung}-{steps}.json',out)


def audit(release,step):
    lock=json.loads((OUT/'25-lock.json').read_text())
    assert step==lock['checkpoint'] and release.strip()
    if (OUT/'25-audit.json').exists():raise FileExistsError('audit receipt already issued')
    from tranche2_hierarchy_raw import audit_rawlinear
    k=Kernels(WORK);w,y,_,ids,t=load(k,4608,5120);v={}
    prior=np.load(OUT/'25-split-4096.npz');keys=lambda s,c:set(zip(s.tolist(),c.tolist()))
    for fold in ('fit','dev'):
        assert not np.intersect1d(prior[fold+'_parent'],ids['parent']).size
        assert not keys(prior[fold+'_chain_seed'],prior[fold+'_chain_id'])&keys(ids['chain_seed'],ids['chain_id'])
    summary={'release':release,'checkpoint':step,'seconds_extract':t,'parents':512,'seed_scores':{}}
    for arm in ('raw','frozen','joint'):
        for seed in SEEDS:
            path=OUT/f'25-model-{arm}-{seed}-{step}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==lock['sha256'][path.name]
            m=np.load(path);h=k.raw(w,m['rule']) if arm=='raw' else m['top'][k.codes(w,m['child'])]
            vec=np.zeros(512);k.run('parentmean',512,h,y,np.ascontiguousarray(np.log(m['dec']/m['base'])),vec)
            v[f'{arm}_{seed}']=vec;summary['seed_scores'][f'{arm}_{seed}']=intervals(vec)
        v[arm]=np.mean([v[f'{arm}_{s}'] for s in SEEDS],axis=0);summary[arm]=intervals(v[arm])
    v['joint_minus_frozen']=v['joint']-v['frozen'];summary['joint_minus_frozen']=intervals(v['joint_minus_frozen'])
    summary['paired_seeds']={str(s):intervals(v[f'joint_{s}']-v[f'frozen_{s}']) for s in SEEDS}
    audit_rawlinear(k,w,y,ids,release,lock)
    np.savez_compressed(OUT/'25-audit-parents.npz',**ids,**v);dump('25-audit.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rung',type=int,choices=(256,1024,4096));p.add_argument('--extend',action='store_true');p.add_argument('--audit-release');p.add_argument('--checkpoint',type=int,choices=(4,8));a=p.parse_args()
    if a.audit_release:audit(a.audit_release,a.checkpoint)
    elif a.rung:fit(a.rung,8 if a.extend else 4)
    else:p.error('supply a rung or released audit')
