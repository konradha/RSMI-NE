"""Bound the extra information in a farther tile once six adjacent coarse labels are known.
This single-site second-shell test is not a proof of a 3D Markov property.
"""
import argparse
from itertools import product
from pathlib import Path
import json
import numpy as np
from scipy.special import xlogy
from scipy.stats import t
from rate import partition, source_rows, digest, json_out
from tile_codec import tile_index, pack

NAMES=('rsmi','species-A','kmeans_A_dipole','composition'); DIR=tuple(product(range(3),(-1,1)));PHASE=tuple(product((0,1),repeat=3)); GROUPS=32
KEYS=('entropy','near_lower','near_upper','far_lower','far_upper','far_empirical','frozen_far_log_gain')
def histogram(h):
    n=np.zeros_like(h,dtype='i4'); h=h.astype('i4',copy=False)
    for j,(a,s) in enumerate(DIR):n+=np.roll(h,-s,axis=a+1)<<(2*j)
    base=16*n+h
    return np.stack([np.bincount((base+4*np.roll(h,-2*s,axis=a+1)).ravel(),minlength=65536) for a,s in DIR]).reshape(6,4096,4,4)


def conditional(c):
    n=c.sum(-1,keepdims=True);return -np.sum(xlogy(c,np.divide(c,n,out=np.zeros_like(c,dtype=float),where=n>0)))/c.sum()


def fit(c):
    near=c[0].sum(1);m=(near.sum(0)+.5)/(near.sum()+2.);n=(near+16*m)/(near.sum(-1,keepdims=True)+16);f=(c+128*n[None,:,None,:])/(c.sum(-1,keepdims=True)+128)
    return -np.log(m),-np.log(n),-np.log(f)


def metrics(c,md):
    near=c[0].sum(1);m=near.sum(0);den=m.sum();hm=conditional(m);hn=conditional(near);hf=conditional(c);ce0=(m@md[0])/den;cen=(near*md[1]).sum()/den;cef=(c*md[2]).sum()/c.sum()
    result=np.array([hm,hm-cen,ce0-hn,hn-cef,cen-hf,hn-hf,cen-cef]);
    if result[3]>result[5]+1e-12 or result[4]<result[5]-1e-12:raise ArithmeticError('empirical finite-alphabet sandwich violated')
    return result


def report(groups,trained):
    md=fit(trained);total=groups.sum(0,dtype='i8');m=metrics(total,md);deleted=np.stack([metrics(total-g,md) for g in groups]);se=np.sqrt((GROUPS-1)*np.mean((deleted-deleted.mean(0))**2,axis=0));k=t.ppf(.975,GROUPS-1);r={n:dict(estimate=float(v),se=float(e),lo95=float(v-k*e),hi95=float(v+k*e)) for n,v,e in zip(KEYS,m,se)}
    return r


def controls(train,val):
    # Deterministic relabelings of the archived counts: F is constant, or exactly H.
    # Both hold H,N fixed; their empirical conditional information is known exactly.
    ref=conditional(val[0].sum(axis=1));res={}
    for copy in (False,True):
        def plant(c):
            v=c[0].sum(1);u=np.zeros_like(c)
            for g in range(4):u[:,:,g if copy else 0,g]=v[:,g]
            return u
        pt,pv=plant(train),plant(val);s=metrics(pv,fit(pt));exact=ref if copy else 0.
        if abs(s[5]-exact)>1e-10 or not s[3]<=exact+1e-10 or not s[4]>=exact-1e-10:raise ValueError('known conditional channel failed')
        res['copy' if copy else 'constant']=dict(exact_empirical=float(exact),lower=float(s[3]),upper=float(s[4]))
    return res


def stage(a,index):
    (lookup,state),meta=source_rows(a.data,index,(0,1));ids=partition(index)[0];design=json.loads((a.out/'design.json').read_bytes());source=a.codec/'comparators.npz'
    if digest(source)!=design['maps_sha256']:raise ValueError('wrong coarse maps')
    train={k:np.zeros((6,4096,4,4),'i8') for k in NAMES};groups={k:np.zeros((GROUPS,6,4096,4,4),'u4') for k in NAMES};offsets=tuple(np.array(v) for v in PHASE)
    with np.load(source,allow_pickle=False) as f:maps={k:f[k+'_lut'] for k in NAMES}
    for fold in (0,1):
        sel=ids[fold];gs=len(sel)//GROUPS
        for start in range(0,len(sel),64):
            s=state[lookup[sel[start:start+64]]].reshape(-1,20,20,20);g=start//gs
            if start//gs!=(min(len(sel),start+64)-1)//gs:raise ValueError('chunk crosses a parent-group boundary')
            tiles=np.concatenate([tile_index(np.roll(s,tuple(-ph),axis=(1,2,3)) if ph.any() else s) for ph in offsets])
            for k,tab in maps.items():
                h=tab[tiles];counts=histogram(h)
                if not np.all(counts.sum((1,2,3))==len(s)*8000):raise ValueError('wrong number of tile sites')
                if not np.array_equal(counts.sum(axis=2),np.broadcast_to(counts[0].sum(axis=1),(6,4096,4))):raise ValueError('near marginal changed with far direction')
                if not fold:train[k]+=counts
                else:groups[k][g]+=counts.astype('u4')
                if k=='rsmi' and fold==1 and start==0:
                    archived=np.load(a.codec/f'packed-{index:02d}.npy',mmap_mode='r')[:len(s)]
                    if not np.array_equal(pack(h[:len(s)]),archived):raise ValueError('coarse field differs from frozen archive')
    results={k:report(groups[k],train[k]) for k in NAMES};cal=controls(train['rsmi'],groups['rsmi'].sum(0,dtype='i8'));r=results['rsmi'];decision=('passes both registered locality conditions' if r['near_lower']['lo95']>.02 and r['far_upper']['hi95']<.01 else 'does not establish both registered locality conditions')
    json_out(a.out/f'beta-{index:02d}.json',dict(source=meta,design_sha256=digest(a.out/'design.json'),script_sha256=digest(Path(__file__)),controls=cal,models=results,decision=(decision if index==8 else 'descriptive'),units='nats per central coarse symbol, uniform registered direction; intervals conditional on fixed probability fits',parents=len(ids[1]),parent_groups=GROUPS));print('LOCALITY',index,decision,{k:r[k] for k in ('near_lower','far_lower','far_upper')},flush=True)


def plot(out):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    rs=[json.loads((out/f'beta-{i:02d}.json').read_bytes()) for i in (9,8,0)];fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout='constrained');labels=['RSMI','Corner A','Dipole','Composition']
    for ax,j in zip(axs,rs):
        for ix,n in enumerate(NAMES):
            r=j['models'][n];lo=r['far_lower']['lo95'];hi=r['far_upper']['hi95'];ax.plot((ix,ix),(lo,hi),lw=3,c='#107d84');ax.scatter(ix,r['far_empirical']['estimate'],s=15,c='black')
        ax.axhline(.01,c='#a85438',ls='--',lw=.8);ax.axhline(0,c='.65',lw=.7);ax.set(title=f"β = {j['source']['beta']:.2f}",xticks=np.arange(4),xticklabels=labels);ax.tick_params(axis='x',rotation=22);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Extra information in one outer tile (nats)');fig.savefig(out/'locality.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--codec',type=Path,required=True);args=p.parse_args();
    for idx in (9,8,0):
        path=args.out/f'beta-{idx:02d}.json'
        if path.exists():
            r=json.loads(path.read_bytes());assert r['script_sha256']==digest(Path(__file__)) and r['design_sha256']==digest(args.out/'design.json');continue
        stage(args,idx)
    plot(args.out)
