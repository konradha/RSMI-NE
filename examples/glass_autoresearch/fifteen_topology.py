"""Development-only: parity of marked defects, cubical H1 and typed k-core winding."""
import argparse,json,math
from itertools import combinations,product
from pathlib import Path
from collections import deque
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import binom,beta as beta_dist,t
from rate import digest,json_out

AX=np.eye(3,dtype=int);DIR=np.r_[AX,-AX];L=20;N=8000
CUBE=tuple(product(range(2),repeat=3));ROD=tuple((0,0,x) for x in range(8));PATCH=np.array(list(product(range(2),range(2),range(3))));PAT=np.arange(4096);BITS=((PAT[:,None]>>np.arange(12))&1).astype('i1');MOM=np.column_stack([BITS,*[BITS[:,i]*BITS[:,j] for i,j in combinations(range(12),2)]]).astype(float)

def at(a,r):return np.roll(a,tuple(-np.asarray(r)),axis=(-3,-2,-1))
def interval(x):
    v=np.asarray(x,float);n=len(v);m=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(n));h=float(t.ppf(.975,n-1)*se);return dict(parents=n,estimate=m,lo=m-h,hi=m+h,se=se,mde80_approx=2.80*se)
def hot_diff(a,b):
    x=interval(a);y=interval(b);se=np.hypot(x['se'],y['se']);m=x['estimate']-y['estimate'];return dict(estimate=m,se=se,lo=m-1.97*se,hi=m+1.97*se,mde80_approx=2.80*se)
def uppercounts(k,n):return [float(beta_dist.ppf(.025,k,n-k+1)) if k else 0.,float(beta_dist.ppf(.975,k+1,n-k)) if k<n else 1.]

def parity(s):
    occ=s!=0;m=sum(at(occ,r).astype('i1') for r in DIR);mark=occ&(m%2==0);K=int(mark.sum());M=int(occ.sum());null=[]
    for n in range(9):
        q=1.;term=1.
        for i in range(1,n+1):term*=(-2)*(K-i+1)/(M-i+1);q+=math.comb(n,i)*term
        null.append(q)
    ans=[]
    for shape in (CUBE,ROD):
        total=sum(at(occ,r).astype('i1') for r in shape);hits=sum(at(mark,r).astype('i1') for r in shape);cross=np.zeros(s.shape,'i2');inside=set(shape)
        for r in shape:
            for d in DIR:
                q=tuple(np.asarray(r)+d)
                if q not in inside:cross+=at(occ,r)&at(occ,q)
        if np.any((total+cross-hits)%2):raise ArithmeticError('defect parity / cut bonds identity violated')
        obs=(1-2*(hits%2)).mean();base=np.asarray(null)[total].mean();ans+=[float(obs),float(base),float(obs-base)]
    return ans+[K]

def topology_table():
    coords={tuple(v):i for i,v in enumerate(PATCH)};edges=[];faces=[]
    for v in PATCH:
        for d in AX:
            if tuple(v+d) in coords:edges.append((coords[tuple(v)],coords[tuple(v+d)]))
    ed={frozenset(e):i for i,e in enumerate(edges)}
    for v in PATCH:
        for a,b in combinations(AX,2):
            sq=[tuple(v+d) for d in (0,a,a+b,b)]
            if not all(q in coords for q in sq):continue
            ids=[coords[q] for q in sq];boundary=sum(1<<ed[frozenset((ids[k],ids[(k+1)%4]))] for k in range(4));faces.append((sum(1<<i for i in ids),boundary))
    betti=[]
    for mask in range(4096):
        parent=list(range(12));e=0;cc=mask.bit_count()
        def root(i):
            while i!=parent[i]:i=parent[i]
            return i
        for i,j in edges:
            if (mask>>i)&(mask>>j)&1:
                e+=1;a,b=root(i),root(j)
                if a!=b:parent[a]=b;cc-=1
        piv={}
        for m,v in faces:
            if mask&m!=m:continue
            while v and v.bit_length() in piv:v^=piv[v.bit_length()]
            if v:piv[v.bit_length()]=v
        betti.append(e-mask.bit_count()+cc-len(piv))
    b=np.array(betti);filled=faces[0][0];assert b.min()==b[0]==b[-1]==b[filled]==0 and b.max()>0
    return b

def pairfit(counts):
    target=counts@MOM/counts.sum();a=MOM
    def fun(w):
        lw=a@w;n=logsumexp(lw);return n-target@w,(np.exp(lw-n)@a)-target
    fit=minimize(fun,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1800,'ftol':1e-15,'gtol':2e-8,'maxcor':35});z=a@fit.x;prob=np.exp(z-logsumexp(z));err=np.max(abs(prob@a-target));
    if err>2e-6 or abs(prob.sum()-1)>1e-12:raise ArithmeticError(f'inadequate moment fit {err}')
    return prob,err

def canonical12():
    p=[]
    for k in BITS.sum(1):
        k=int(k);p.append(math.prod(6000-j for j in range(k))*math.prod(2000-j for j in range(12-k))/math.prod(8000-j for j in range(12)))
    if abs(sum(p)-1)>1e-12:raise ArithmeticError('fixed composition normalization')
    return np.array(p)

GRID=np.indices((L,)*3).reshape(3,-1).T;NN=np.array([np.ravel_multi_index(((GRID+d)%L).T,(L,)*3) for d in DIR]).T

def winding(core):
    live=core.ravel();seen=np.zeros(N,bool);lifts=np.zeros((N,3),'i2');w=np.zeros(3,bool)
    for src in np.flatnonzero(live):
        if seen[src]:continue
        seen[src]=True;q=deque([src]);
        while q:
            u=q.popleft()
            for j,v in enumerate(NN[u]):
                if not live[v]:continue
                z=lifts[u]+DIR[j]
                if seen[v]:w|=(z!=lifts[v]);continue
                seen[v]=True;lifts[v]=z;q.append(v)
        if w.all():break
    return w

def pruning(s):
    occ=s!=0;deg=sum(at(occ,d).astype('i1') for d in DIR);thr=np.where(s==1,3,5);core=occ;first=None;iters=0
    while True:
        new=core&(deg>=thr)
        if first is None:first=new.mean()
        if np.array_equal(new,core):break
        iters+=1;core=new;deg=sum(at(core,d).astype('i1') for d in DIR)
    w=winding(core);initial=sum(at(occ,d).astype('i1') for d in DIR);kind=7*(s.astype('i2')-1)+initial;cnt=np.bincount(kind[occ],minlength=14);mix=np.zeros((14,14),'i8')
    for d in DIR:
        act=occ&at(occ,d);a=kind[act];b=at(kind,d)[act];mix+=np.bincount(14*a+b,minlength=196).reshape(14,14)
    return np.r_[first,core.mean(),w.any(),iters],cnt,mix

def tree(cnt,mix):
    d=np.tile(np.arange(7),2);k=np.repeat((3,5),7);P=np.divide(mix,mix.sum(1)[:,None],out=np.zeros_like(mix,dtype=float),where=mix.sum(1)[:,None]>0);x=np.ones(14)
    for i in range(10000):
        a=binom.sf(k-2,np.maximum(d-1,0),P@x);a[d==0]=0
        if np.max(abs(x-a))<1e-12:break
        x=a
    else:raise ArithmeticError('tree messages did not converge')
    return float((cnt@(binom.sf(k-1,d,P@x)))/cnt.sum()*.75)

def run(a):
    o=a.out;o.mkdir(parents=True,exist_ok=True);betti=topology_table();canonical=canonical12();groups={};rows={};r3={};r4={};r6={}
    for idx in (9,0,8):
        with np.load(a.bank/f'b{idx:02}.npz') as z:states=z['species'];ids=z['parent']
        pv=[];h=[];pr=[];mix=np.zeros((14,14),'i8');cnt=np.zeros(14,'i8')
        for s in states:
            pv.append(parity(s));word=sum((at(s,v)!=0).astype('i2')*(1<<j) for j,v in enumerate(PATCH));h.append(np.bincount(word.ravel(),minlength=4096));row,nn,mm=pruning(s);pr.append(row);mix+=mm;cnt+=nn
        par=np.stack(pv);h=np.stack(h);pr=np.stack(pr);key=f'b{idx:02}';contrast=par[:,2]-par[:,5];groups[idx]=contrast;r3[key]=dict(corrected_cube_minus_rod=interval(contrast),cube=interval(par[:,2]),rod=interval(par[:,5]),marks_per_box=interval(par[:,-1]),identities_checked=2*N*len(ids));
        holes=h@(betti>0)/N;p,err=pairfit(h[:128].sum(0));q=float(p@(betti>0));splitfit=[float(pairfit(block.sum(0))[0]@(betti>0)) for block in np.array_split(h[:128],2)];r4[key]=dict(train_parents=128,calibration_parents=128,exact_pair_fitted_probability=q,gradient_max=float(err),separate_64parent_refits=splitfit,observed_calibration=interval(holes[128:]),paired_residual_conditional_on_fit=interval(holes[128:]-q),canonical_beta0_reference=float(canonical@(betti>0)),parents_with_hole=int(np.sum(holes[128:]>0)),catalogue_masks_with_hole=int((betti>0).sum()));
        n=len(pr);kw=int(pr[:,2].sum());r6[key]=dict(winding=float(kw/n),winding_boxes=kw,parents=n,exact95=uppercounts(kw,n),first_occupied_volume_fraction=interval(pr[:,0]),final_occupied_volume_fraction=interval(pr[:,1]),iterations=interval(pr[:,3]),tree_final_volume_fraction=tree(cnt,mix));rows[key+'_parent']=ids;rows[key+'_parity']=par;rows[key+'_topology']=holes;rows[key+'_core']=pr;rows[key+'_histo_train']=h[:128].sum(0);rows[key+'_type_degree_counts']=cnt;rows[key+'_edge_mixing']=mix
    r3['cold_minus_beta0']=hot_diff(groups[9],groups[0]);r3['margin']=.01;r4['margin']=.001;r6['winding_margin']=.05;np.savez_compressed(o/'topology-parents.npz',betti=betti,**rows)
    for k,v in [(3,r3),(4,r4),(6,r6)]:json_out(o/f'{k:02}-development.json',dict(results=v,code=digest(Path(__file__)),parent_evidence=digest(o/'topology-parents.npz'),scope='Native training parents only; calibration, not family-wide discovery.'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
