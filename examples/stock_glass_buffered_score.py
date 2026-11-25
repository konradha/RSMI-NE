"""One blind scoring pass. No model updates. Produce paired scores at the independent-parent level."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import stock_glass_buffered_fit as f


def group(v,d):
    # Bank order is the species-blind priority within each parent.
    first=d['first'];return np.stack([np.add.reduceat(v.astype('d'),first,axis=0)/np.diff(np.r_[first,len(v)]).reshape((-1,)+(1,)*(v.ndim-1)),v[first]],axis=1)


def predictors(saved):
    out=[]
    for key in f.ORDER[1:]:
        spec=saved['models'][key];m=f.Decoder(saved['width'],spec['width'],spec['nonlin'],saved['prior']);m.load_state_dict(spec['state']);out.append(m.eval())
    return out


def invariant(saves):
    rng=np.random.default_rng(911);checks=[]
    for label,_,r,saved in saves:
        pos=f.mask(r);centers=np.array([[0,19,2],[19,0,19],[10,10,10]]);ix=tuple(np.moveaxis((centers[:,None,:]+pos)%20,-1,0));lat=rng.integers(3,size=(3,20,20,20));j=np.arange(3)[:,None];s=lat[j,*ix];alt=(lat+rng.integers(1,3,size=lat.shape))%3;alt[j,*ix]=s;sa=alt[j,*ix];assert np.count_nonzero(lat!=alt)==3*(8000-150)
        x,n=f.raw(s,r);xa,na=f.raw(sa,r);assert n==na and torch.equal(x,xa);x=(x-saved['mu'])/saved['scale'];xa=(xa-saved['mu'])/saved['scale'];m=predictors(saved)
        with torch.no_grad():
            for h in m:assert torch.equal(f.ln_p(h(x)),f.ln_p(h(xa)))
        checks.append(dict(label=label,r=r,hidden_changes=3*(8000-150),parents=3,predictors=['composition','base','rich'],max_absolute_prediction_change=0))
    (f.OUT/'predictor-invariance.json').write_text(json.dumps(dict(success=True,checks=checks),indent=2)+'\n')


def score(label,b,r,d,saved,dest):
    q=d['q'];obs=d['obs'];m=predictors(saved);native=[];soft=[];lp=[];stats=[];cal=[];labelobs=np.eye(210,dtype='f4')[obs];allmass={};discr=[];confs=[]
    with torch.no_grad():
        for i in range(0,len(q),1024):
            sl=slice(i,i+1024);s=d['s'][sl];x,nb=f.raw(s,r);assert nb==saved['nb'];x.sub_(saved['mu']).div_(saved['scale']);lp=[f.ln_p(saved['prior'].expand(len(s),-1))]+[f.ln_p(h(x)) for h in m];a=np.stack([t.numpy() for t in lp],axis=1)/np.log(2);native.append(np.take_along_axis(a,obs[sl,None,None],axis=2).squeeze(2));soft.append(np.einsum('ijk,ik->ij',a,q[sl]));pm=np.exp2(a[:,2:,:]);conf=pm.max(2);pred=pm.argmax(2);confs.append(np.stack([conf,pred==obs[sl,None],q[np.arange(i,i+len(s))[:,None],pred]],axis=2));
            for k,v in enumerate(pm.transpose(1,0,2),2):allmass[k]=allmass.get(k,0)+np.einsum('i,ij->j',d['w'][sl],v,dtype='d')
    soft=group(np.concatenate(soft),d);native=group(np.concatenate(native),d);extra={};n=len(d['ids']);w=d['w'];actual=group(labelobs,d)[:,0];teacher=group(q,d)[:,0];diff=actual-teacher
    sem=diff.std(axis=0,ddof=1)/np.sqrt(n);counts=labelobs.sum(0);oc=np.zeros(210,dtype=int)
    for start,end in zip(d['first'],np.r_[d['first'][1:],len(obs)]):oc[np.unique(obs[start:end])]+=1
    for j in range(210):
        cal.append(dict(state=j,obs_anchors=int(counts[j]),obs_parents=int(oc[j]),native_mass=actual[:,j].mean(),exact_mass=teacher[:,j].mean(),base_mass=allmass[2][j]/n,rich_mass=allmass[3][j]/n,obs_minus_exact=diff[:,j].mean(),se_paired_parents=sem[j]))
    conf=np.concatenate(confs);edges=np.array([0,.01,.025,.05,.10,.20,.40,.60,.80,1.00001]);topcal=[]
    for j,h in enumerate(('base','rich')):
        x=conf[:,j];ix=np.digitize(x[:,0],edges)-1
        for ib in range(len(edges)-1):
            sel=ix==ib
            if sel.any():topcal.append(dict(model=h,bin_low=edges[ib],bin_high=min(1.,edges[ib+1]),parents=len(np.unique(d['par'][sel])),anchors=int(sel.sum()),parent_weight=w[sel].sum()/n,pred_confidence=np.average(x[sel,0],weights=w[sel]),obs_accuracy=np.average(x[sel,1],weights=w[sel]),exact_agreement=np.average(x[sel,2],weights=w[sel])))
    if d['p0'] is not None:
        oq=np.log2(q.astype(float));o0=np.log2(d['p0'].astype(float));diff=oq-o0;ix=np.arange(len(obs));oracle=group(np.stack([(q*diff).sum(1),diff[ix,obs]],1),d);scoreq=group(np.stack([(q*oq).sum(1),oq[ix,obs]],1),d);extra.update(oracle=oracle,oracle_base=scoreq-np.stack([soft[:,:,2],native[:,:,2]],2))
    key=dict(label=label,b=b,beta=f.BETAS[b],r=r);name=f'{label}-b{b:02}-r{r}';np.savez_compressed(dest/f'{name}.npz',**key,ids=d['ids'],native=native,soft=soft,anchors=np.diff(np.r_[d['first'],len(q)]),**extra);print('SCORE',name,n,len(q),flush=True)
    return [dict(**key,**v) for v in cal],[dict(**key,**v) for v in topcal],dict(**key,n=n,stars=len(q),native_supported_states=int(np.sum(counts>0)),teacher_min=float(teacher.mean(0).min()),total_variation_obs_exact=float(np.abs(actual.mean(0)-teacher.mean(0)).sum()/2),**d['stats'])


def load_head(path,sha):
    assert f.digest(path)==sha
    # DEV loss metadata is a NumPy float64. Keep the restricted weights loader.
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('f8'))]):
        return torch.load(path,map_location='cpu',weights_only=True)

def run(betas):
    lock=json.loads((f.OUT/'freeze.json').read_bytes());assert lock['design']==f.digest(f.OUT/'design.md') and lock['roster']==f.digest(f.OUT/'roster.npz');path=f.OUT/'heads';dest=f.WORK/'scores';dest.mkdir(exist_ok=True,parents=True)
    assert lock['plant_sha']==f.digest(path/'plant.npz');plant=dict(np.load(path/'plant.npz'));saves=[]
    for label,train_b,r in f.SCENARIOS:
        name=f'{label}-b{train_b:02}-r{r}.pt';saves.append((label,train_b,r,load_head(path/name,lock['heads'][name])))
    invariant(saves)
    for b in betas:
        audit=json.loads((f.OUT/f'bank-audit-b{b:02}-assess.json').read_bytes())
        assert audit['frozen_receipt_sha256']==f.digest(f.OUT/'freeze.json') and audit['roster_sha256']==lock['roster']
        assert audit['sha256']==f.digest(f.WORK/f'b{b:02}-assess.npz') and audit['side']=='assess' and audit['beta_index']==b
        cats=[];tops=[];coverage=[]
        for label,t,r,saved in saves:
            if label!='physical' and b!=t:continue
            d=f.data(b,'assess',r,label,plant);z,tops0,c=score(label,b,r,d,saved,dest);cats.extend(z);tops.extend(tops0);coverage.append(c)
        f.dump_csv(dest/f'calibration-b{b:02}.csv',cats);f.dump_csv(dest/f'confidence-b{b:02}.csv',tops); keys=set().union(*(v.keys() for v in coverage));f.dump_csv(dest/f'coverage-b{b:02}.csv',[{k:c.get(k,'') for k in sorted(keys)} for c in coverage])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--betas',type=int,nargs='+',required=True);p.add_argument('--work',type=Path,default=f.WORK);p.add_argument('--threads',type=int,default=4);a=p.parse_args();f.WORK=a.work;torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);run(a.betas)
