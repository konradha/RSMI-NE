"""Post-small-pilot diagnostic: lower-variance direct raw input reference (prespecified in 25-raw-diagnostic-design)."""
import argparse
import hashlib
import json
from time import perf_counter
from tranche2_hierarchy import WORK, OUT, SEEDS, WD, Kernels, load, marginal, measure, intervals, dump
import numpy as np

SITES=[tuple(2*a+b for a,b in zip(u,v)) for u in np.ndindex(2,2,2) for v in np.ndindex(2,2,2)]
FACE=list(np.ndindex(2,2));FD=(np.arange(81)[:,None]//3**np.arange(3,-1,-1)%3).astype(np.uint8)

def indicators(words):
    return np.stack([words==0,words==1],axis=-1).reshape(len(words),-1).astype(np.float32)


def sufficient(w,y):
    xx=np.zeros((128,128));xy=np.zeros((128,8));xmean=np.zeros(128);ymean=np.zeros(8)
    for lo in range(0,len(w),16384):
        sl=slice(lo,lo+16384);a=indicators(WD[w[sl]]);b=indicators(FD[y[sl]])
        xx+=a.T@a;xy+=a.T@b;xmean+=a.sum(0);ymean+=b.sum(0)
    symxx=np.zeros_like(xx);symxy=np.zeros_like(xy);xm=np.zeros_like(xmean);ym=np.zeros_like(ymean)
    for swap,ry,rz in np.ndindex(2,2,2):
        def yz(y,z,edge):
            if swap:y,z=z,y
            return (edge-y if ry else y,edge-z if rz else z)
        px=[2*SITES.index((x,*yz(y,z,3)))+s for x,y,z in SITES for s in (0,1)]
        py=[2*FACE.index(yz(y,z,1))+s for y,z in FACE for s in (0,1)]
        symxx+=xx[np.ix_(px,px)];symxy+=xy[np.ix_(px,py)];xm+=xmean[px];ym+=ymean[py]
    xm/=8*len(w);ym/=8*len(w);cov=symxx/(8*len(w))-np.outer(xm,xm);cross=symxy/(8*len(w))-np.outer(xm,ym)
    cov.flat[::129]+=.01*cov.diagonal().mean()
    coef=np.linalg.solve(cov,cross)/np.sqrt(ym*(1-ym))
    # Common intercept drops from Lloyd distances and from the prediction PC centering.
    tile=indicators(WD);return np.stack([tile@coef[16*i:16*(i+1)] for i in range(8)]).astype(np.float32)


def predict(w,coef):
    p=np.zeros((len(w),8),np.float32)
    for i in range(8):p+=coef[i,w[:,i]]
    return p


def classify(p,c):
    return (p@c.T-.5*(c*c).sum(1)).argmax(1).astype(np.uint8)


def means(p,h):return np.stack([p[h==i].mean(0,dtype=np.float64) for i in range(4)]).astype(np.float32)


def fit():
    k=Kernels(WORK);wd,yd,*_=load(k,4096,4608);result={};vecs={}
    for rung in (256,1024,4096):
        start=perf_counter();w,y,*_=load(k,0,rung);base=marginal(y);coef=sufficient(w,y);p=predict(w,coef);pd=predict(wd,coef)
        # Float64 moments computed in chunks: no 4M×8 double precision copy.
        cov=sum(p[t:t+32768].astype(float).T@p[t:t+32768] for t in range(0,len(p),32768))/len(p)
        mu=p.mean(0,dtype=float);cov-=np.outer(mu,mu);_,v=np.linalg.eigh(cov);pc=(p@v[:,-2:].astype(np.float32)-mu@v[:,-2:]).astype(np.float32)
        r={'fit_seconds':perf_counter()-start,'seeds':{}};result[rung]=r
        for seed in SEEDS:
            startseed=perf_counter();angle=np.random.default_rng(seed).uniform(0,np.pi/2);cs,sn=np.cos(angle),np.sin(angle)
            h=((cs*pc[:,0]+sn*pc[:,1]>0).astype(np.uint8)+2*(cs*pc[:,1]-sn*pc[:,0]>0).astype(np.uint8));c=means(p,h);curve=[]
            for step in range(1,9):
                h=classify(p,c);hd=classify(pd,c);stat,vv,dec=measure(k,h,y,hd,yd,base);stat.update(step=step,seconds=perf_counter()-startseed);curve.append(stat)
                if step in (1,2,4,8):vecs[f'r{rung}_{seed}_{step}']=vv
                if step==8 and rung==4096:np.savez_compressed(OUT/f'25-model-rawlinear-{seed}-8.npz',coef=coef,centroids=c,base=base,dec=dec)
                if step<8:c=means(p,h)
            r['seeds'][seed]=curve
        r['total_seconds']=perf_counter()-start
        print(rung,r['total_seconds'],intervals(np.mean([vecs[f'r{rung}_{s}_8'] for s in SEEDS],axis=0)),flush=True)
    np.savez_compressed(OUT/'25-rawlinear-dev-parents.npz',**vecs);dump('25-rawlinear-development.json',result)


def audit_rawlinear(k,w,y,ids,release,lock):
    vectors={}
    for s in SEEDS:
        path=OUT/f'25-model-rawlinear-{s}-8.npz';assert lock['diagnostic']['sha256'][path.name]==hashlib.sha256(path.read_bytes()).hexdigest()
        m=np.load(path);h=classify(predict(w,m['coef']),m['centroids']);v=np.empty(512);k.run('parentmean',512,h,y,np.ascontiguousarray(np.log(m['dec']/m['base'])),v);vectors[str(s)]=v
    vectors['mean']=np.mean(list(vectors.values()),axis=0);np.savez_compressed(OUT/'25-rawlinear-audit-parents.npz',parent=ids['parent'],**vectors)
    dump('25-rawlinear-audit.json',{'release':release,'scores':{name:intervals(val) for name,val in vectors.items()}})


if __name__=='__main__':
    fit()
