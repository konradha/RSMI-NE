"""Buffered Gate 1 only. Fit on authorized fit/dev banks; assessment is a separate hash-guarded command."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from stock_glass_buffered_features import features, mask
from stock_glass_cavity_exact import X, prob

OUT=Path(__file__).resolve().parents[1]/'results/buffered-static-gate1'
WORK=Path('/home/fritz/rsmi-work/buffered-static-gate1')
BETAS=[float(f'{i*3.05/9:.12f}') for i in range(10)]+[3.2,3.4]
STATE=torch.tensor(np.eye(3,dtype='f4')[X][:,:,1:].reshape(210,14).T.copy())
ORDER=('prior','composition','base','rich'); EPOCHS=(0,1,2,4,8,16); FITSEED=40301; FLOOR=1e-5
SCENARIOS=[('physical',9,r) for r in (4,2,5)]+[(s,9,4) for s in ('observed_fit','shuffle','plant','nullplant')]+[('beta0',0,4)]

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def dump_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def bins(shell):return ((shell==0).sum(1)//4)*40+(shell==1).sum(1)//4

def plant_t(shell):
    return (shell[:,16]==0).astype(float)-(shell[:,8]==0)  # first face (-2..2)²: (+1,-1) and (-1,+1)

def parent_weights(parent):
    return 1./np.bincount(parent)[parent]

def ave(a,w):return np.einsum('i,ij->j',w,a,dtype='d',optimize=False)/w.sum()

def init_plant(s,q,w):
    ix=bins(s);size=1600;t=plant_t(s);tot=np.bincount(ix,w,minlength=size);active=tot>0;mean=ave(q,w);mean=(1-1e-9)*mean+1e-9/210
    nuisance=np.tile(mean,(size,1));avg=np.full(size,np.average(t,weights=w));avg[active]=np.bincount(ix,t*w,minlength=size)[active]/tot[active]
    mass=np.zeros((size,210));np.add.at(mass,ix,w[:,None]*q);nuisance[active]=.998*mass[active]/tot[active,None]+.002*mean
    p=nuisance[ix];h=(X[:,0]==0).astype(float);m=(nuisance@h)[ix];g=t-avg[ix];lo=0.;hi=.99/2
    def kl(a):
        # Two binary group multipliers suffice; signed gradient in composition bins is centered.
        x=a*g*(1-m);y=-a*g*m
        return np.average((m*(1+x)*np.log1p(x)+(1-m)*(1+y)*np.log1p(y))/np.log(2),weights=w)
    if kl(hi)>=.01:
        for _ in range(42):
            mid=(lo+hi)/2
            if kl(mid)<.01:lo=mid
            else:hi=mid
    return dict(nuisance=nuisance,mean_t=avg,a=np.array(hi),achieved=np.array(kl(hi)))

def make_plant(s,ids,parent,plant,enabled,b):
    ix=bins(s);p=plant['nuisance'][ix];g=plant_t(s)-plant['mean_t'][ix];h=(X[:,0]==0).astype(float);m=np.sum(p[:,h.astype(bool)],axis=1)
    q=p*(1+(plant['a'] if enabled else 0)*g[:,None]*(h-m[:,None]));assert np.min(q)>0 and abs(q.sum(1)-1).max()<1e-9
    u=np.empty(len(parent));ui,first,counts=np.unique(parent,return_index=True,return_counts=True)
    for i,j,k in zip(ui,first,counts):
        rng=np.random.default_rng([40312,b,*map(int,ids[i])]);shared=rng.random();u[j:j+k]=np.where(rng.random(k)<.25,shared,rng.random(k))
    obs=(u[:,None]>q.cumsum(1)).sum(1);assert obs.max()<210
    return q.astype('f4'),obs,p.astype('f4')

def data(b,side,r,label,plant=None,frac=1.):
    with np.load(WORK/f'b{b:02}-{side}.npz') as d:
        ids=d['ids'];par=d['parent'].astype(int);keep=par<int(len(ids)*frac);par=par[keep];s=d[f'r{r}'][keep];obs=d['observed'][keep].astype(int);q=prob(d['energy'][keep],BETAS[b]).astype('f4')
    n0=len(np.unique(par));p0=None;stats={}
    if label=='shuffle':
        parents,ix=np.unique(par,return_index=True);s,q,obs,par=s[ix],q[ix],obs[ix],parents
        strata=bins(s);dest=[];donor=[];rng=np.random.default_rng([40319,b,{'fit':1,'dev':2,'assess':3}[side]])
        for k in np.unique(strata):
            ii=rng.permutation(np.flatnonzero(strata==k))
            if len(ii)>1:dest.extend(ii);donor.extend(np.roll(ii,1))
        dest,donor=np.array(dest),np.array(donor);order=np.argsort(dest);dest,donor=dest[order],donor[order];assert (par[dest]!=par[donor]).all() and (strata[dest]==strata[donor]).all()
        stats=dict(eligible=n0,retained=len(dest),bins=len(np.unique(strata[dest])),raw_vacancy_abs_diff=float(abs((s[dest]==0).sum(1)-(s[donor]==0).sum(1)).mean()),raw_A_abs_diff=float(abs((s[dest]==1).sum(1)-(s[donor]==1).sum(1)).mean()))
        s,q,obs,par=s[donor],q[dest],obs[dest],par[dest]
    if label in ('plant','nullplant'):q,obs,p0=make_plant(s,ids,par,plant,label=='plant',b)
    unique,first,par=np.unique(par,return_index=True,return_inverse=True);w=parent_weights(par);assert len(unique)==len(first)
    return dict(s=s,q=q,obs=obs,ids=ids[unique],par=par,first=first,w=w,p0=p0,stats=stats)

def raw(s,r):
    out=None;nb=None
    for i in range(0,len(s),768):
        base,detail=features(s[i:i+768],r)
        if out is None:
            nb=base.shape[1];out=np.empty((len(s),nb+detail.shape[1]),dtype='f4')
        out[i:i+len(base),:nb]=base;out[i:i+len(base),nb:]=detail
    assert np.isfinite(out).all();return torch.from_numpy(out),nb

def normalize(tr,w):
    w=torch.from_numpy(w.astype('f4'));w/=w.sum();mu=w@tr;ss=torch.zeros_like(mu)
    for i in range(0,len(w),1024):ss+=w[i:i+1024]@((tr[i:i+1024]-mu)**2)
    return mu,ss.sqrt().clamp(min=.05)

def prior(d,label):
    y=ave(d['q'],d['w']) if label!='observed_fit' else np.bincount(d['obs'],d['w'],minlength=210)/d['w'].sum();return torch.tensor(np.log((1-FLOOR)*y+FLOOR/210),dtype=torch.float32)

class Decoder(nn.Module):
    def __init__(self,nin,nuse,nonlinear,offset):
        super().__init__();self.nonlin=nonlinear;self.width=nuse
        # The full matrix is initialized once per paired fit; multiplying/padding zeros in restricted arms is unnecessary.
        self.input=nn.Linear(nin,48);self.out=nn.Linear(48,224);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias);self.register_buffer('offset',offset.clone())
    def forward(self,x):
        h=nn.functional.linear(x[:,:self.width],self.input.weight[:,:self.width],self.input.bias);v=self.out(h.tanh() if self.nonlin else h);return self.offset+v[:,:210]+v[:,210:]@STATE

def ln_p(logits):return ((1-FLOOR)*torch.softmax(logits,dim=1)+FLOOR/210).log()

def target(d,observed):return torch.from_numpy(np.eye(210,dtype='f4')[d['obs']] if observed else d['q'])

def held_loss(model,x,y,w):
    with torch.no_grad():
        v=sum((-(y[i:i+1024]*ln_p(model(x[i:i+1024]))).sum(1)*w[i:i+1024]).sum().item() for i in range(0,len(y),1024));return v/(w.sum().item()*np.log(2))

def fit_pair(tr,dv,r,label,b,frac,rows):
    xt,nb=raw(tr['s'],r);xd,nb2=raw(dv['s'],r);assert nb==nb2;mu,scale=normalize(xt,tr['w']);xt.sub_(mu).div_(scale);xd.sub_(mu).div_(scale)
    off=prior(tr,label);wt=torch.from_numpy((tr['w']/tr['w'].mean()).astype('f4'));wd=torch.from_numpy(dv['w'].astype('f4'));yt=target(tr,label=='observed_fit');yd=target(dv,label=='observed_fit');n=len(xt)
    saved=dict(mu=mu,scale=scale,nb=nb,width=xt.shape[1],prior=off,models={});info=dict(label=label,b=b,r=r,fraction=frac,parents=len(tr['ids']),dev_parents=len(dv['ids']),anchors=n,dev_anchors=len(xd))
    for name,width in [('composition',3),('base',nb),('rich',xt.shape[1])]:
        best=np.inf
        for nonlin in (False,True):
            torch.manual_seed(FITSEED);model=Decoder(xt.shape[1],width,nonlin,off);opt=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=.01);gen=torch.Generator().manual_seed(FITSEED);step=0
            for ep in range(17):
                if ep:
                    for ii in torch.randperm(n,generator=gen).split(1024):
                        logits=model(xt[ii]);loss=(-(yt[ii]*logits.log_softmax(dim=1)).sum(1)*wt[ii]).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step();step+=1
                if ep in EPOCHS:
                    score=held_loss(model,xd,yd,wd);rows.append(dict(**info,model=name,architecture='tanh' if nonlin else 'affine',epoch=ep,updates=step,fit_logloss_bits=held_loss(model,xt,yt,wt),dev_logloss_bits=score,active_parameters=width*48+48+48*224+224))
                    if score<best or (score==best and ep<saved['models'][name]['epoch']):
                        best=score;saved['models'][name]=dict(width=width,nonlin=nonlin,epoch=ep,loss=best,state={k:v.detach().clone() for k,v in model.state_dict().items()})
        print('FIT',label,b,r,frac,name,saved['models'][name]['epoch'],round(best,6),flush=True)
    return saved

def freeze(bankdir):
    bankdir.mkdir(parents=True,exist_ok=True);rows=[];shuffles=[];idx={};fit_hashes={}
    for b in (0,9):
        for side in ('fit','dev'):
            p=WORK/f'b{b:02}-{side}.npz';fit_hashes[p.name]=digest(p)
    dry=data(9,'fit',4,'physical');plant=init_plant(dry['s'],dry['q'],dry['w']);del dry;np.savez_compressed(bankdir/'plant.npz',**plant)
    for label,b,r in SCENARIOS:
        for frac in ((1.,.25,.5) if label=='physical' and r==4 else (1.,)):
            tr=data(b,'fit',r,label,plant,frac);dv=data(b,'dev',r,label,plant);saved=fit_pair(tr,dv,r,label,b,frac,rows)
            if tr['stats']:shuffles += [dict(side=side,**d['stats']) for side,d in [('fit',tr),('dev',dv)]]
            if frac==1.:
                dest=bankdir/f'{label}-b{b:02}-r{r}.pt';torch.save(saved,dest);idx[dest.name]=digest(dest)
            dump_csv(OUT/'dev.csv',rows)
    (OUT/'freeze.json').write_text(json.dumps(dict(design=digest(OUT/'design.md'),roster=digest(OUT/'roster.npz'),fit_banks=fit_hashes,heads=idx,plant_sha=digest(bankdir/'plant.npz'),fit_plant_oracle_bits=float(plant['achieved']),fit_plant_a=float(plant['a']),shuffle=shuffles,torch=torch.__version__,cpu_threads=torch.get_num_threads()),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=4);p.add_argument('--work',type=Path,default=WORK);args=p.parse_args();WORK=args.work;torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);freeze(OUT/'heads')
