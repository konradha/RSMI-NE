"""Descriptive decomposition of the previously selected inward-vacancy tripod.

One exterior A site is registered to an axis. Preserve the entire inner-tripod
law conditional on center species and coordination; fit only the A conditional.
A conventional pair model has logit a+b*v_parallel+c*(v_trans1+v_trans2).
A saturated model also sees the full joint state of those three vacancy bits,
with transverse exchange symmetry imposed in both. No test rows, trajectories,
new configurations, or changes to the frozen mechanism prediction.
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import t

from rate import arrays, digest, json_out, partition, source_rows

INDICES = (9, 7, 0)
ORIENT = tuple(product((-1, 1), repeat=3))
P, Q = np.indices((2, 3)).reshape(2, -1)
X = np.array([np.ones(6), P, Q]).T
FEATURE = 6 * np.arange(21).reshape(3, 7)
MODES = ('coordination', 'isotropic', 'registered_pairs', 'unrestricted')


def hist(states):
    """(parent, species*7+m, 3*parallel+transverse, [N, n_A])."""
    out = np.empty((len(states), 21, 6, 2), dtype=np.int32)
    for start in range(0, len(states), 12):
        s = states[start:start+12].reshape(-1, 20, 20, 20)
        b = len(s); off = (126*np.arange(b)).reshape(-1, 1, 1, 1)
        vacancy = {(ax,sg): np.roll(s, -sg, ax+1) == 0 for ax in range(3) for sg in (-1, 1)}
        outer = {(ax,sg): np.roll(s, 3*sg, ax+1) == 1 for ax in range(3) for sg in (-1, 1)}
        m = 6-sum(v.astype('i1') for v in vacancy.values()); base = off + FEATURE[s,m]
        counts = np.zeros((b*126, 2), dtype='i8')
        for signs in ORIENT:
            v = [vacancy[axis, sg].astype('i1') for axis,sg in enumerate(signs)]; total = v[0]+v[1]+v[2]
            for ax,sg in enumerate(signs):
                code = (base + total + 2*v[ax]).ravel()
                counts[:,0] += np.bincount(code, minlength=b*126)
                counts[:,1] += np.bincount(code[outer[ax,sg].ravel()], minlength=b*126)
        h = counts.reshape(b, 21, 6, 2)
        if not (np.all(h[:,:,:,0].sum((1,2)) == 8000*24) and np.all(h[:,:,:,1].sum((1,2)) == 2400*24)):
            raise ValueError('axis enumeration altered the exact margins')
        # Separate direct-lattice check of the joint code, axis, wrap and target.
        if not start:
            ref = np.zeros((21,6,2),'i8'); sites=np.indices((20,)*3).reshape(3,-1).T
            for signs in ORIENT:
                step=np.diag(signs); inside=(sites[:,None,:]+step)%20; outside=(sites[:,None,:]-3*step)%20
                def sample(k):return s[0][tuple(k.transpose(2,0,1))]
                v=(sample(inside)==0).astype('i1'); y=sample(outside)==1
                for ax in range(3):
                    zone=7*s[0].ravel()+m[0].ravel(); k=2*v[:,ax]+v.sum(1)
                    np.add.at(ref[:,:,0],(zone,k),1);np.add.at(ref[:,:,1],(zone,k),y[:,ax])
            if not np.array_equal(ref,h[0]): raise ValueError('direct lattice counting disagrees')
        out[start:start+b] = h
    return out


def fit(counts):
    # Fixed smoothing: Jeffreys half-count in *physically possible* cells;
    # impossible cells never enter any likelihood or empirical joint law.
    m = np.arange(21)[:,None]%7
    possible = ((P+Q <= 6-m) & (3-P-Q <= m))
    n = counts[...,0]+possible; a = counts[...,1]+.5*possible
    sat = np.divide(a,n,out=np.full_like(a,.3),where=n>0)
    if np.any(counts[...,0][~possible]):raise ValueError('geometrically impossible inner configuration')
    out = {'unrestricted':sat}
    for name,design in [('coordination',X[:,:1]),('isotropic',np.stack([X[:,0],P+Q],axis=1)),('registered_pairs',X)]:
        pars=np.zeros((21,design.shape[1]),dtype='d');pars[:,0]=np.log(.3/.7)
        for _ in range(40):
            p=expit(pars@design.T);grad=(a-n*p)@design
            curv=np.einsum('st,ti,tj->sij',n*p*(1-p),design,design)
            step=(np.linalg.pinv(curv,hermitian=True,rcond=1e-12)@grad[...,None])[...,0]
            pars += step/np.maximum(1.,np.max(abs(step),axis=1,keepdims=True)/2.)
            if np.max(np.abs(grad)/np.maximum(1,n.sum(1))[:,None])<1e-11:break
        else:raise ValueError('binomial score equations did not converge')
        out[name]=expit(pars@design.T)
    return out


def losses(counts,models):
    n,a=counts[...,0],counts[...,1];out=[]
    for k in MODES:
        q=np.clip(models[k],1e-15,1-1e-15);out.append(-(a*np.log(q)+(n-a)*np.log1p(-q)).sum((-1,-2))/n.sum((-1,-2)))
    return np.stack(out,axis=-1)


def compare(train,valid):
    tot=train.sum(0,dtype='i8'); md=fit(tot); ll=losses(valid,md); vp=ll.mean(0);
    groups=np.array_split(np.arange(len(train)),32)
    if len(set(map(len,groups)))!=1:raise ValueError('unequal training packets')
    lh=np.array([losses(valid.sum(0,dtype='i8'),fit(tot-train[group].sum(0,dtype='i8'))) for group in groups]);g=len(groups)
    result={}
    for i,j in ((0,1),(1,2),(0,2),(2,3),(0,3)):
        v=ll[:,i]-ll[:,j]; d=lh[:,i]-lh[:,j]; v1=v.var(ddof=1)/len(v);v2=(g-1)*np.mean((d-d.mean())**2);se=np.sqrt(v1+v2)
        df=(v1+v2)**2/(v1**2/(len(v)-1)+v2**2/(g-1)) if se else len(v)-1
        z=v.mean();c=t.ppf(1-.05/(2*15),df);h=t.ppf(.975,df)
        result[f'{MODES[j]}_over_{MODES[i]}']=dict(nats=float(z),se=float(se),lo95=float(z-h*se),hi95=float(z+h*se),lo_family=float(z-c*se),hi_family=float(z+c*se),train_se=float(np.sqrt(v2)),validation_se=float(np.sqrt(v1)))
    return dict(loss_nats=dict(zip(MODES,map(float,vp))),contrasts=result)


def plot(path,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9,3.5),layout='constrained');beta=[r['source']['beta'] for r in rows]
    for a,key,title in ((ax,'registered_pairs_over_coordination','Outer A: contribution of vacancy pairs'),(bx,'unrestricted_over_registered_pairs','What a nonlinear tripod adds')):
        d=[r['contrasts'][key] for r in rows];y=np.array([v['nats'] for v in d])*1e6;lo=np.array([v['lo_family'] for v in d])*1e6;hi=np.array([v['hi_family'] for v in d])*1e6
        a.errorbar(beta,y,yerr=np.stack([y-lo,hi-y]),fmt='o',color='#086f75',capsize=4);a.axhline(0,color='.55',lw=.8);a.set(xlabel='Inverse temperature β',ylabel='Held-out gain (µnats / outer site)',title=title);a.set_xticks(sorted(beta),[f'{v:.2f}' for v in sorted(beta)]);a.yaxis.set_major_locator(MaxNLocator(5));a.spines[['top','right']].set_visible(False)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);results=[]
    for index in INDICES:
        (lookup,states),meta=source_rows(args.data,index,(0,1));ids=partition(index)[0];bank=[]
        for f in (0,1):
            # Index only a batch at a time so archive storage is not doubled.
            pieces=[hist(states[lookup[batch]]) for batch in np.array_split(ids[f],max(1,len(ids[f])//512))];bank.append(np.concatenate(pieces))
        filename=f'parents-{index:02d}.npz';arrays(args.out/filename,train=bank[0],validation=bank[1],train_rows=ids[0],validation_rows=ids[1]);response=compare(*bank);results.append(dict(source=meta,counts_sha256=digest(args.out/filename),**response));print(meta['beta'],response['contrasts']['unrestricted_over_registered_pairs'],flush=True)
    json_out(args.out/'results.json',dict(scope='Descriptive on earlier validation cohorts; per-temperature trained competitors. Not a rescue/replication of the frozen mechanism test.',geometry='All 8000 center sites, eight octants and three registered outer targets per L20 parent. Counts are trials and number of outer A sites. Conditioning stratum: central species and total coordination. Six cells = 3*parallel_vacancy_bit + transverse_vacancy_count.',model='All models have arbitrary inner-tripod distribution. Species/coordination-specific logistic conditional: intercept only, common slope on tripod count, distinct parallel/transverse slopes, or six unrestricted logits. Equivalently the registered pair model allows any many-body relation within the tripod and no outer couplings beyond individual vacancies in each stratum.',statistics='32 delete-group train jackknife variance plus separate variance of paired validation parent means; Welch t intervals, family=15 contrasts. No sites, rotations or targets treated as independent parents. Finite projection only.',temperatures=results));plot(args.out/'tripod-pairs.png',results)


if __name__=='__main__':main()
