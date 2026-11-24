"""Canonical species-exchange check for the RSMI-identified oriented vacancy tripod.
Uses a closed-form conditional probability; never samples, shuffles or changes a lattice.
"""
import argparse
import json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.stats import t
from rate import digest, json_out, partition, source_rows

NAMES=('observed_A','exact_conditional_A','residual_A','total_enrichment','occupancy_contribution','coordination_contribution','residual_enrichment')


def moments(s,check=False):
    occ=(s!=0);aa=(s==1);near={(a,i):np.roll(occ,-i,a+1) for a,i in product(range(3),(-1,1))};m=sum(v.astype(np.int8) for v in near.values());center=aa&(m==3);n=len(s);code=m+(7*np.arange(n)).reshape(-1,1,1,1);cnt=np.bincount(code[occ],minlength=n*7).reshape(n,7);ns=np.bincount(code[aa],minlength=n*7).reshape(n,7)
    if not np.all(ns.sum(1)==2400) or not np.all(cnt.sum(1)==6000):raise ValueError('wrong composition')
    # Condition on the observed center being A, without conditioning on the target label.
    # In the same coordination stratum only, remove that one particle from the urn.
    cn=cnt.copy();sn=ns.copy(); cn[:,3]-=1;sn[:,3]-=1
    q=np.divide(sn,cn,out=np.zeros_like(sn,dtype=float),where=cn>0);qp=q[np.arange(n)[:,None,None,None],m]*occ
    sums=np.zeros((n,2,4)); axes=tuple(range(1,4))
    for a,i in near:
        b,c=(j for j in range(3) if j!=a);w=center*(~near[a,i]);w=w*(near[b,1].astype('i1')+near[b,-1])*(near[c,1].astype('i1')+near[c,-1])
        if check:
            direct=sum((center & ~near[a,i] & near[b,j] & near[c,k]).astype('i1') for j,k in product((-1,1),repeat=2))
            if not np.array_equal(w,direct):raise ValueError('octant multiplicity mismatch')
        yn=np.roll(aa,3*i,a+1);ot=np.roll(occ,3*i,a+1);pred=np.roll(qp,3*i,a+1)
        for k,ww in enumerate((w,4*center)):
            sums[:,k,0]+=ww.sum(axes);sums[:,k,1]+=(ww*yn).sum(axes);sums[:,k,2]+=(ww*pred).sum(axes);sums[:,k,3]+=(ww*ot).sum(axes)
    if check:
        # Exact sum of A-target conditional probabilities over ALL nonself occupied
        # positions for a central A,m=3 must be N_A-1 at each valid parent.
        valid=ns[:,3]>=1
        if not np.allclose((q*cn).sum(1)[valid],2399,rtol=0,atol=1e-9):raise ValueError('canonical finite-urn sum rule failed')
    return sums


def estimand(s):
    n,a,p,o=np.moveaxis(s,-1,0);actual=a/n;pred=p/n;occ=o/n;given=p/o
    z=np.stack([actual[...,0],pred[...,0],(actual-pred)[...,0],actual[...,0]-actual[...,1],(occ[...,0]-occ[...,1])*given[...,1],occ[...,0]*(given[...,0]-given[...,1]),(actual-pred)[...,0]-(actual-pred)[...,1]],axis=-1)
    if not np.allclose(z[...,3],z[...,4:].sum(-1),rtol=0,atol=1.e-12):raise ValueError('conditional enrichment does not close')
    return z


def run(a):
    plan=json.loads((a.out/'design.json').read_bytes()); rr=[];columns={}
    for i in (5,9,7,0):
        (look,states),meta=source_rows(a.data,i,(1,));ids=partition(i)[0][1];
        if i==5 and meta['sha256']!=plan['primary_sha256']:raise ValueError('changed primary archive')
        vals=np.concatenate([moments(states[look[ids[p:p+32]]].reshape(-1,20,20,20),not p) for p in range(0,len(ids),32)]);total=vals.sum(0);point=estimand(total);loo=estimand(total-vals);se=np.sqrt((len(ids)-1)*np.mean((loo-loo.mean(0))**2,axis=0));half=t.ppf(.975,len(ids)-1)*se;d={k:dict(mean=float(m),lo95=float(m-h),hi95=float(m+h),se=float(s)) for k,m,h,s in zip(NAMES,point,half,se)};r=d['residual_A'];delta=plan['margin'];within=r['lo95']>-delta and r['hi95']<delta;known=None
        if i in (9,7,0):
            other=json.loads((a.out.parent/'tripod-cell/cell-results.json').read_bytes());prior=next(v for v in other['temperatures'] if v['index']==i);known=prior['actual_A'];
            if abs(point[0]-known)>1e-12:raise ValueError('orientation, A identity or primary selection mismatch')
        rr.append(dict(source=meta,parents=len(ids),cell_events=int(total[0,0]),parents_with_cell=int((vals[:,0,0]>0).sum()),observed_reproduction=known,metrics=d,decision=('equivalent within fixed margin' if within else 'difference exceeds margin' if r['hi95'] < -delta or r['lo95']>delta else 'inconclusive at margin') if i==5 else 'descriptive'))
        columns[f'rows_{i:02}']=ids;columns[f'counts_{i:02}']=vals; print('decoration',i,r,rr[-1]['decision'],flush=True)
    np.savez_compressed(a.out/'parent-counts.npz',**columns);json_out(a.out/'results.json',dict(plan_sha256=digest(a.out/'design.json'),code_sha256=digest(Path(__file__)),parents_sha256=digest(a.out/'parent-counts.npz'),cohorts=rr));plot(a.out,rr)


def plot(out,rows):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.5,3.7),layout='constrained');colors=['#157878','#b06c22','#854a79'];labels=['Vacancy probability','Species mix by coordination','Species residual'];rs=sorted(rows,key=lambda r:r['source']['beta']);x=np.arange(len(rs));names=[f"{r['source']['beta']:.2f}" for r in rs]
    def series(ax,key,color,label,dx=0):
        v=np.array([[r['metrics'][key][s] for s in ('mean','lo95','hi95')] for r in rs])*100;y,l,h=v.T;ax.errorbar(x+dx,y,yerr=[y-l,h-y],fmt='o',label=label,c=color,capsize=3)
    series(ax,'residual_A','#157878','Measured − exact conditional');ax.axhspan(-.05,.05,alpha=.1,color='#157878');ax.set_title('Outer A in the selected cage');ax.set_ylabel('Excess probability (pp)')
    for j,k in enumerate(NAMES[4:]):series(bx,k,colors[j],labels[j],(j-1)*.08)
    bx.set_title('Contrast against all A centers with m=3');bx.set_ylabel('Contribution (pp)');bx.legend(fontsize=8,frameon=False)
    for a in (ax,bx):a.axhline(0,lw=.7,color='.55');a.set(xticks=x,xticklabels=names,xlabel='Inverse temperature β');a.spines[['top','right']].set_visible(False)
    fig.savefig(out/'decoration.png',dpi=165);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    if (a.out/'results.json').exists():raise FileExistsError('already scored; refusing to replace')
    run(a)
