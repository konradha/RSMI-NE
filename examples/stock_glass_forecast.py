"""Experiment 07. Supervised CPU heads on frozen 03 targets, not RSMI training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path
import struct

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve
from scipy.special import log_softmax

from stock_glass_cavity_exact import C, U, J, prob
from stock_glass_modal import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/experiment-07-boundary-forecast'
REF = ROOT/'results/experiment-03-cavity'
SEEDS = (6101,6102,6103)
VIEWS = ('packing','directional','rotated')
LAM = np.array([0.1,0.01,0.001,0.0001])  # strongest first for ties and warm starts
CHANNELS = ('A','B','coord','coord2','A_coord','B_coord')
ORB = (abs(U).max(1)==2).astype(int)
OPP = [(i,int(np.nonzero((U==-z).all(1))[0][0])) for i,z in enumerate(U) if tuple(z)>tuple(-z)]
assert len(OPP)==9 and not {tuple(z) for z in U}.intersection(map(tuple,C))

def rotation_indices():
    mats=[]
    for p in permutations(range(3)):
        for s in product((-1,1),repeat=3):
            m=np.eye(3,dtype=int)[list(p)]*np.array(s)[:,None]
            if round(np.linalg.det(m))==1:mats.append(m)
    mats.sort(key=lambda m:(not np.array_equal(m,np.eye(3)),tuple(m.flatten())))
    # Active R: the transformed field at coordinate u is the original at R^{-1}u.
    rot=np.array([[np.flatnonzero((U==u).all(1))[0] for u in U@m] for m in mats])
    assert len(np.unique(rot,axis=0))==24 and np.all(ORB[rot]==ORB)
    return rot
ROT=rotation_indices()

def assignments(ids,b,domain,n):
    return np.fromiter((int.from_bytes(sha256(domain+struct.pack('<IQQ',b,int(a),int(c))).digest()[:8],'little')%n for a,c in ids),np.int16,len(ids))

def exterior(su,fu,rot):
    # fu already excludes every neighbor in C; even vacant exterior sites can have occupied neighbors.
    assert su.shape==fu.shape and su.shape[1]==18 and np.all(su<=2) and np.all(fu<=6-J.sum(1))
    u=fu.astype(float); a=su==1; b=su==2; z=np.stack((a,b,u,u*u,u*a,u*b),axis=2)
    means=np.stack([z[:,ORB==o].mean(1) for o in range(2)],axis=1); baseline=means.reshape(len(z),-1)
    residual=z-means[:,ORB]; zr=residual[np.arange(len(z))[:,None],ROT[rot]]
    return (baseline,np.column_stack((baseline,residual.reshape(len(z),-1))),np.column_stack((baseline,zr.reshape(len(z),-1)))), z

def load(b,side,banks):
    design=json.loads((OUT/'design.json').read_bytes()); manifest=banks/'prepared.json'
    assert sha(manifest)==design['bank_manifest']; meta=json.loads(manifest.read_bytes()); name=f'b{b:02}-{side}.npz'; assert sha(banks/name)==meta['inputs'][name]
    with np.load(banks/name) as data:
        j=data['selected']; ids=data['all_ids'][j]; rot=assignments(ids,b,b'07-rotate',24); features,sites=exterior(data['su'],data['fu'],rot)
        beta=float(meta['coverage'][b]['beta']); p=prob(data['relative_E'],beta)  # label generation only
    assert len(ids)==len(np.unique(ids,axis=0))==meta['coverage'][b][side]['selected_parents']; file=REF/'corrected/code-maps.npz';assert sha(file)==design['code_maps']
    with np.load(file) as frozen:
        maps=np.stack([frozen[f'b{b:02}-s{k}_d{10218 if b==9 else 10257}_hard_Q'] for k in SEEDS]);np.testing.assert_array_equal(maps,np.eye(4)[maps.argmax(2)])
    q=np.stack([p@m for m in maps]);q/=q.sum(2,keepdims=True);return beta,ids,j,rot,features,sites,q,maps.any(1)

def fit_soft(x,q,lam,start):
    # x has intercept in the last column. Absent categories handled before entry.
    def fun(w):
        w=w.reshape(x.shape[1],-1);lp=log_softmax(x@w,axis=1);grad=x.T@(np.exp(lp)-q)/len(x);grad[:-1]+=lam*w[:-1]
        return -(q*lp).sum()/len(x)+0.5*lam*np.sum(w[:-1]**2),grad.ravel()
    r=minimize(fun,start.ravel(),jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':3e-7,'maxiter':500,'maxls':40});assert r.success,r.message
    return r.x.reshape(start.shape),[int(r.nit),float(abs(r.jac).max())]

def standard(x,y):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1.;return np.column_stack(((x-mu)/sd,np.ones(len(x)))),np.column_stack(((y-mu)/sd,np.ones(len(y)))),mu,sd

def raw(w,mu,sd):
    w=w.copy();w[:-1]/=sd.reshape((-1,)+(1,)*(w.ndim-1));w[-1]-=mu@w[:-1];return w

def ridge_terms(x,s):
    return x.T@x/len(x), x.T@s/len(x)

def ridge(g,h,l):
    v=g.copy();v.flat[:v.size-1:v.shape[1]+1]+=l;return solve(v,h,assume_a='pos')

def select(values):return np.flatnonzero(values<=values.min()+1e-12)[0]

def classify(x,w,mask):
    lp=log_softmax(x@w[:-1]+w[-1],axis=1);out=np.zeros((len(x),4));out[:,mask]=np.exp(lp);return out,lp

def dual_head(x,q,mask,fold):
    qa=q[:,mask];s=(q*q).sum(1);cv=np.zeros((3,len(LAM),2));opt=[]
    for k in range(3):
        tr=fold!=k;va=~tr;u,v,_,_=standard(x[tr],x[va]);g,h=ridge_terms(u,s[tr]);w=np.zeros((u.shape[1],mask.sum()));w[-1]=np.log(qa[tr].mean(0));
        for a,l in enumerate(LAM):
            if mask.sum()>1:w,status=fit_soft(u,qa[tr],l,w);opt.append([k,a,*status])
            lp=log_softmax(v@w,axis=1);st=np.clip(v@ridge(g,h,l),.25,1.);cv[k,a]=[-(qa[va]*lp).sum(),np.sum((st-s[va])**2)]
    i,j=map(select,(cv.sum(0)/len(x)).T);u,_,mu,sd=standard(x,x[:0]);g,h=ridge_terms(u,s);w=np.zeros((u.shape[1],mask.sum()));w[-1]=np.log(qa.mean(0));
    if mask.sum()>1:w,status=fit_soft(u,qa,LAM[i],w);opt.append([-1,i,*status])
    w=raw(w,mu,sd);v=raw(ridge(g,h,LAM[j]),mu,sd);return w,v,cv,np.array(opt).reshape(-1,4),np.array([LAM[i],LAM[j]])

def sliced(z,q):
    # Differences oriented by index as stored in OPP; tags below give final sign.
    d=np.stack([z[:,i]-z[:,j] for i,j in OPP],axis=1).reshape(len(z),-1);d-=d.mean(0);sd=np.sqrt((d*d).sum(0));out=[]
    for t in q.T:
        y=t-t.mean();den=sd*np.linalg.norm(y); cor=np.divide(y@d,den,out=np.zeros_like(sd),where=den>1e-12);h=int(np.argmax(abs(cor)));out.append((h,np.sign(cor[h]),cor[h]))
    return np.array(out)

def fit_one(args):
    b,banks,out=args;beta,ids,rows,rot,feat,z,qs,masks=load(b,'train',banks);fold=assignments(ids,b,b'07-fold',3);counts=np.bincount(fold,minlength=3);assert counts.min()>2000
    saved=dict(beta=beta,train_ids=ids,rows=rows,rotations=rot,fold=fold,masks=masks,design_hash=sha(OUT/'design.json'),fit_source=sha(Path(__file__)));full=[]
    for seed,q,mask in zip(SEEDS,qs,masks):
        k=f's{seed}_';s=(q*q).sum(1);saved.update({k+'prior':q.mean(0),k+'mean_s':s.mean(),k+'slices':sliced(z,q)})
        for view,x in zip(VIEWS,feat):
            pref=k+view+'_';w,v,cv,opt,lam=dual_head(x,q,mask,fold);pr,_=classify(x,w,mask);pred=np.clip(x@v[:-1]+v[-1],.25,1.);vals=[w,v,cv,opt,lam,np.quantile(pr.max(1),[.1,.5,.9]),np.quantile(pred,.1)]
            saved.update({pref+k:a for k,a in zip(('W','v','cv_sums','convergence','lambda','quantiles','low_strength_cut'),vals)})
    path=out/f'fit-b{b:02}.npz';np.savez_compressed(path,**saved);print('FIT_LOCKED',b,len(ids),sha(path),flush=True);return path.name,sha(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--banks',type=Path,required=True);p.add_argument('--workers',default=4,type=int);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();assert not (a.out/'fit-lock.json').exists();a.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:locks=dict(pool.map(fit_one,[(b,a.banks,a.out) for b in range(10)]))
    (a.out/'fit-lock.json').write_text(json.dumps(locks,indent=2)+'\n')
if __name__=='__main__':main()
