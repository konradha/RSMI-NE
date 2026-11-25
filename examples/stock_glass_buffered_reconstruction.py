"""Post-lock identifiability diagnostic, not a candidate predictor. No native states, teacher energies, TEST or outcomes are loaded."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import stock_glass_buffered_fit as f


def shell(side):
    assert side in ('fit','dev')
    with np.load(f.WORK/f'b09-{side}.npz') as z:
        return z['r4'],z['parent'].astype(int)

def execute(out):
    train,p=shell('fit');dev,dp=shell('dev');wt=f.parent_weights(p);w=f.parent_weights(dp);wt/=wt.sum();w/=w.sum();s=f.bins(train);sd=f.bins(dev);n=np.bincount(s,wt,minlength=1600);mt=np.full(1600,np.average(f.plant_t(train),weights=wt));sel=n>0;mt[sel]=np.bincount(s,wt*f.plant_t(train),minlength=1600)[sel]/n[sel]
    def blocks(s,w):
        for i in range(0,len(s),640):
            sh=s[i:i+640];x,detail=f.features(sh,4);t=f.plant_t(sh);y=np.column_stack((detail,t,t-mt[f.bins(sh)]));yield i,torch.from_numpy(x.astype('d')),torch.from_numpy(y),torch.from_numpy(w[i:i+640])
    xx=xy=mx=my=None
    for i,x,y,w0 in blocks(train,wt):
        if xx is None:
            xx=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float64);xy=torch.zeros(x.shape[1],y.shape[1],dtype=torch.float64);mx=torch.zeros(x.shape[1],dtype=torch.float64);my=torch.zeros(y.shape[1],dtype=torch.float64)
        z=x*w0[:,None];xx.addmm_(x.T,z);xy.addmm_(z.T,y);mx+=z.sum(0);my+=w0@y
    xx-=torch.outer(mx,mx);sdx=xx.diag().clamp_min(0).sqrt().clamp_min(.05);G=xx/(sdx[:,None]*sdx);B=(xy-torch.outer(mx,my))/sdx[:,None];eig=torch.linalg.eigvalsh(G);lam=1e-6;coef=torch.linalg.solve(G+lam*torch.eye(len(mx),dtype=torch.float64),B)
    unique,first=np.unique(dp,return_index=True);ss=np.empty((len(dp),len(my)));den=np.empty_like(ss)
    for i,x,y,_ in blocks(dev,w):
        yp=((x-mx)/sdx)@coef+my;ss[i:i+len(x)]=(yp-y).numpy()**2;den[i:i+len(x)]=(y-my).numpy()**2
    nn=np.diff(np.r_[first,len(dp)]);ss=np.add.reduceat(ss,first)/nn[:,None];den=np.add.reduceat(den,first)/nn[:,None];obs=1-ss.sum(0)/den.sum(0);rows=[dict(mask_site=i,species=k,r2=obs[3*i+k]) for i in range(150) for k in range(3)];f.dump_csv(out/'descriptor-sites.csv',rows)
    maps=dict(all_indicators=np.arange(450),vacancy=np.arange(0,450,3),A=np.arange(1,450,3),B=np.arange(2,450,3),plant_raw=[450],plant_minus_fit_bin_mean=[451]);ns=len(nn);rng=np.random.default_rng(36195);summ={}
    for k,col in maps.items():
        err=ss[:,col].sum(1);d=den[:,col].sum(1);ww=rng.multinomial(ns,np.full(ns,1/ns),size=2048);draw=1-(ww@err)/(ww@d);summ[k]=dict(r2=float(1-err.sum()/d.sum()),lo=np.quantile(draw,.025),hi=np.quantile(draw,.975))
    receipt=dict(scope='secondary source FIT/DEV only; no X or q; not a promotion route',b=9,r=4,fit_parents=len(np.unique(p)),fit_stars=len(p),dev_parents=ns,dev_stars=len(dp),columns=len(mx),rank_at_relative_eigenvalue_cutoff_1e_10=int((eig>1e-10*eig[-1]).sum()),ridge=lam,measure='1 - parent-weighted development squared reconstruction error / error around FIT mean; unbounded linear outputs, not probabilities; percentile CI resamples DEVELOPMENT parents conditional on one fit.',summary=summ,site_indicator_r2_quantiles=np.quantile(obs[:450],[0,.1,.5,.9,1]).tolist());(out/'descriptor-reconstruction.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,default=f.WORK);parser.add_argument('--out',type=Path,default=f.OUT);a=parser.parse_args();f.WORK=a.work;torch.set_num_threads(2);execute(a.out)
