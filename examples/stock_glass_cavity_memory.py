"""Exact boundary memory. The critic and sampled replica pairs do not enter these calculations."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import t
from stock_glass_cavity_exact import X,O,L,M,C,U,A,J,prob
import stock_glass_modal as stock


def moments(p):
    mean=p.mean(0);center=p-p[0];center-=center.mean(0)  # identically zero for a constant row
    return mean,center.T@center/len(p),np.diag(mean)-np.outer(mean,mean)


def solve(c,s,basis):
    cb=basis.T@c@basis;sb=basis.T@s@basis;vals,v=eigh((sb+sb.T)/2);keep=vals>max(1e-14,1e-12*vals[-1]);
    if not keep.any():return np.array([]),np.empty((len(basis),0))
    w=v[:,keep]/np.sqrt(vals[keep]);a=w.T@cb@w;ev,u=eigh((a+a.T)/2);ev=ev[::-1];assert ev.min()>-2e-8 and ev.max()<1+2e-8
    return np.clip(ev,0,1),basis@w@u[:,::-1]


def orient(v,mean):
    v=v-mean@v;sgn=np.sign(v[np.argmax(np.abs(v),axis=0),np.arange(v.shape[1])]);return v*np.where(sgn==0,1,sgn)


def evaluate(p,f):
    # Shape: boundaries x categories. p may be p_B Q for an intrinsically noisy code.
    if f is None:return np.nan,np.full(len(p),np.nan)
    f=np.asarray(f,dtype='d');f=f-p.mean(0)@f;u=p@f;v=p@(f*f);u0=u[0];du=u-u0;mu=du.mean()+u0
    num=(du-du.mean())**2;den=v-2*mu*u+mu*mu;dv=den.mean()
    if not np.isfinite(dv) or dv<1e-12:return np.nan,np.full(len(p),np.nan)
    r=num.mean()/dv;assert -.0000001<=r<=1.0000001
    return float(r),(num-r*den)/dv


def interval(q):
    a,inf=q
    if not np.isfinite(a):return (np.nan,)*3
    half=t.ppf(.975,len(inf)-1)*np.std(inf,ddof=1)/np.sqrt(len(inf));return a,a-half,a+half


def compare(q,base,ratio=False):
    a,i=q;b,j=base
    if not np.isfinite([a,b]).all() or (ratio and interval(base)[1]<=1e-4):return (np.nan,)*3
    return interval((a/b,(i-(a/b)*j)/b) if ratio else (a-b,i-j))


def add(row,name,vals):row[name],row[name+'_lo'],row[name+'_hi']=vals


def loadside(inputs,b,side,beta):
    with np.load(inputs/f'b{b:02}-{side}.npz',allow_pickle=False) as z:return prob(z['relative_E'],beta),z['observed'],z['selected']


def candidates():
    site=[('vac_'+str(i),(X[:,i]==0).astype('d')) for i in range(7)]
    cen=[('center_'+x,(X[:,0]==i).astype('d')) for i,x in enumerate(('vac','A','B'))]
    plus=X[:,[2,4,6]];minus=X[:,[1,3,5]];both=(plus!=0)&(minus!=0)
    coord=[('center_m',M[:,0].astype('d')),('center_penalty',(O[:,0]*(M[:,0]-L[X[:,0]])**2).astype('d')),('like_opposite',(both&(plus==minus)).sum(1).astype('d')),('occupied_opposite',both.sum(1).astype('d'))]
    return dict(vacancy=site,central_species=cen,coordination=coord,best_simple=site+cen+coord)


def benchmark(inputs,out):
    meta=json.loads((inputs/'prepared.json').read_bytes());out.mkdir(parents=True,exist_ok=True);spec=[];bench=[];mass=[];conv=[];maps={};audit=[]
    linear=np.concatenate([X==1,X==2],axis=1).astype('d');eye=np.eye(210)
    for b in range(10):
        beta=float(meta['coverage'][b]['beta']);tr,otr,jtr=loadside(inputs,b,'train',beta);te,ote,jte=loadside(inputs,b,'test',beta)
        for side,p in [('train',tr),('test',te)]:assert np.allclose(p.sum(1),1,rtol=0,atol=1e-14) and np.isfinite(p).all()
        m,c,s=moments(tr); mt=te.mean(0);ess=tr.sum(0)**2/np.maximum((tr*tr).sum(0),1e-300);use=m>=1e-10;ev,V=solve(c,s,eye[:,use]);V=orient(V,m);assert len(ev)>1
        oe=V[:,0];ref=evaluate(te,oe);linear_ev,lv=solve(c,s,linear);lf=orient(lv[:,:1],m)[:,0];base=evaluate(te,lf);records={'oracle':('full_indicator',oe),'linear':('all_site_A_B_span',lf)}
        for name,items in candidates().items():
            score=[f@c@f/(f@s@f) if f@s@f>1e-12 else -np.inf for _,f in items];label,f=items[np.argmax(score)];records[name]=(label,f)
        strong=(m>=1e-3)&(ess>=100)
        if strong.any():
            _,v2=solve(c,s,eye[:,strong]);common=orient(v2[:,:1],m)[:,0]
        else:common=None  # Frozen sensitivity mask is empty; do not relax its threshold.
        records['common_mass_oracle']=('p>=.001,ESS>=100',common);maps[f'b{b:02}_train_modes']=V;maps[f'b{b:02}_eigenvalues']=ev;maps[f'b{b:02}_mean']=m
        for k in range(len(ev)):
            row=dict(index=b,beta=beta,mode=k+1,train_value=ev[k],next_gap=ev[k]-ev[k+1] if k+1<len(ev) else np.nan);add(row,'held',interval(evaluate(te,V[:,k])) if k<12 else (np.nan,)*3);spec.append(row)
        vari=m*oe**2;vari/=vari.sum();idx=np.argsort(-oe**2);n50=np.searchsorted(vari[idx].cumsum(),.5)+1;support=float(m[idx[:n50]].sum());obs_tr=np.bincount(otr,minlength=210);obs_te=np.bincount(ote,minlength=210)
        for x in range(210):mass.append(dict(index=b,beta=beta,state=x,pattern=''.join('.AB'[i] for i in X[x]),train_mass=m[x],test_mass=mt[x],train_observed=obs_tr[x],test_observed=obs_te[x],boundary_ess=ess[x],in_unrestricted=use[x],in_common=strong[x],mode_value=oe[x],variance_share=vari[x]))
        for name,(desc,f) in records.items():
            if f is not None:maps[f'b{b:02}_{name}']=f
            r=evaluate(te,f);rr=dict(index=b,beta=beta,observable=name,descriptor=desc,parents=len(te),train_T=evaluate(tr,f)[0]);add(rr,'T',interval(r));add(rr,'oracle_minus',compare(ref,r));add(rr,'minus_linear',compare(r,base));add(rr,'fraction',compare(r,ref,True));bench.append(rr)
        for half,ph in enumerate((tr[::2],tr[1::2])):
            mh,ch,sh=moments(ph);eh,vh=solve(ch,sh,eye[:,mh>=1e-10]);held=evaluate(te,orient(vh[:,:1],mh)[:,0]);conv.append(dict(index=b,beta=beta,half=half,train_parents=len(ph),train_top=eh[0],held_top=held[0],full_train_top=ev[0]))
        if b==0:assert np.max(ev)==0 and ref[0]==0 and np.max(np.abs(tr-1/210))<1e-15
        q=interval(ref);audit.append(dict(index=b,beta=beta,train=len(tr),test=len(te),train_supported=int(use.sum()),common_supported=int(strong.sum()),train_entropy_states=float(np.exp(-sum(m[m>0]*np.log(m[m>0])))),test_mean_effective_states=float(np.exp(-np.sum(np.where(te>0,te*np.log(np.maximum(te,1e-300)),0),axis=1)).mean()),near_leading_count=int(np.sum(ev[0]-ev<=.01)),top_gap=float(ev[0]-ev[1]),half_variance_mass=support,half_variance_states=int(n50),variance_effective_states=float(1/(vari@vari)),train_top=float(ev[0]),held_top=q[0],held_lo=q[1],held_hi=q[2],linear_held=base[0]));print('SPECTRUM',b,ev[0],q[0],base[0],flush=True)
    for rows,name in [(bench,'benchmark'),(spec,'spectrum'),(mass,'state-support'),(conv,'train-halves')]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False,float_format='%.13g')
    np.savez_compressed(out/'frozen-modes.npz',**maps);stock.write_json(out/'exact-receipt.json',dict(design=meta['design'],inputs=stock.sha(inputs/'prepared.json'),eigenbasis=stock.sha(out/'frozen-modes.npz'),by_beta=audit,energy_audits=meta['audit'],unique=meta['unique']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();benchmark(a.inputs,a.out)
