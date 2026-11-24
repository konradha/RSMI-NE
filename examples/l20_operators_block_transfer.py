"""Does the validation-selected equilibrium code predict *physical* block mobility?

Train nuisance/offset calibrations only on the old 64 parents. Score the frozen
calibration on new beta-09 parents and transfer unchanged to beta-08. Hard H:
no continuous logits/soft probabilities or multiple orientations at evaluation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingRegressor
from l20_operators_static import readfit,prob,init,table
from l20_propensity_predict import coordination_barrier

SITES=512
EPS=1e-6
MODELS=('kinetics','kinetics+learned','kinetics+init','kinetics+corners')


def dataset(path,model):
    meta,enc,_=readfit(model);w=int(meta['width']);seed=int(meta['seed']);enc0=init(np.random.default_rng(seed),2*w**3,True,w,True)
    radius=w//2; off=np.indices((w,)*3).reshape(3,-1).T-radius;ixall=table(off)
    with np.load(path,allow_pickle=False) as f:
        ids=f['parent_id'];origin=f['origin'];lags=f['lags'];changed=f['changed'];beta=float(f['beta']);R=int(f['replicas'])
    X=[];codes=[];Y=[]
    for i,(key,s) in enumerate(zip(ids,origin)):
        pick=np.random.default_rng([725109,int(key)]).choice(8000,SITES,replace=False);ix=ixall[pick];v=s.ravel()[ix];occ=v!=0;n=occ.sum(1);assert n.min()>0
        m,b=coordination_barrier(s);b=b.ravel()[ix];m=m.ravel()[ix]
        # Physical competitor sees the full local swap costs, not just labels visible to H.
        feats=[(v==1).mean(1),(v==2).mean(1),(b<=0).mean(1),(b<=2).mean(1),np.minimum(b.min(1),30),np.mean(np.exp(-beta*np.maximum(b,0).astype(float)),axis=1)]
        for label in (1,2):
            mask=(v==label);feats.extend([(mask&(m==k)).mean(1) for k in range(7)]);feats.extend([(mask&(b<=k)).mean(1) for k in (0,1,2,4,8)])
        # The strongest validation-selected fixed spatial code (bits-body-1).
        corners=2*(v[:,0]==1)+(v[:,-1]==1)
        X.append(np.stack(feats,1));codes.append(np.stack([prob(v,w,meta['spec'],e).argmax(1) for e in (enc,enc0)]+[corners],1))
        Y.append((changed[:,i].reshape(len(lags),-1)[:,ix]*occ).sum(-1)/(R*n))
    return beta,lags,ids,np.array(X,dtype=np.float32),np.array(codes,np.int8),np.array(Y)


def reg():return HistGradientBoostingRegressor(max_iter=80,max_depth=3,min_samples_leaf=128,learning_rate=.07,early_stopping=False,random_state=9023)


def offsets(p,y,h):
    # One Bernoulli intercept per predicted-risk stratum and code, shrink to no correction.
    out=np.zeros((4,4));p=np.clip(p,EPS,1-EPS);bins=np.quantile(p,[.25,.5,.75]);b=np.searchsorted(bins,p);lp=logit(p)
    for i in range(4):
        for k in range(4):
            sel=(b==i)&(h==k);l=lp[sel];target=y[sel].sum();d=0.
            for _ in range(16):
                q=expit(l+d);step=(q.sum()-target+2*d)/(np.sum(q*(1-q))+2);d-=np.clip(step,-2,2)
                if abs(step)<1e-8:break
            out[i,k]=d
    return bins,out


def fit(path,model,dest,receipt):
    beta,lags,ids,x,c,y=dataset(path,model);assert beta==3.05 and len(ids)==64
    groups=np.array_split(np.random.default_rng(17316).permutation(len(ids)),4);fits={}
    for lag in (4096,16384):
        j=np.flatnonzero(lags==lag).item();target=y[:,j];oof=np.empty(target.shape)
        for fold in groups:
            train=np.setdiff1d(np.arange(len(ids)),fold);f=reg().fit(x[train].reshape(-1,x.shape[-1]),target[train].ravel());oof[fold]=f.predict(x[fold].reshape(-1,x.shape[-1])).reshape(len(fold),-1)
        calibrations=[offsets(oof.ravel(),target.ravel(),v) for v in [np.zeros(target.size,int),*(c[:,:,k].ravel() for k in range(3))]]
        fits[lag]=(reg().fit(x.reshape(-1,x.shape[-1]),target.ravel()),calibrations)
    joblib.dump(fits,dest,compress=3);receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(dict(development_parent_ids=ids.tolist(),selection=str(model.name),patches_per_parent=SITES,source_frozen_weights_sha256=hashlib.sha256(model.with_suffix('.npz').read_bytes()).hexdigest(),local_calibration_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),models=MODELS,calibration='Fixed depth-3 histogram boosting for physical baseline, then equal-architecture logit-offset calibration of 4 risk quartiles x 4 hard symbols, using parent-grouped four-fold out-of-fold baseline predictions, ridge strength 2. Baseline uses only one symbol.',target='Uniform random anchors without replacement per parent, mean same-species loss among initially occupied block sites, equal block and parent weight. No transfer refits.'),indent=2)+'\n')


def score(path,model,cal,out):
    beta,lags,ids,x,c,y=dataset(path,model);fits=joblib.load(cal);res=[]
    for j,lag in enumerate(lags):
        train_lag={116:4096,462:16384,4096:4096,16384:16384}.get(int(lag));
        if train_lag not in fits:continue
        regressor,calibrations=fits[train_lag];p=np.clip(regressor.predict(x.reshape(-1,x.shape[-1])),EPS,1-EPS);target=y[:,j]
        for name,(bins,d),h in zip(MODELS,calibrations,[np.zeros(p.shape,int),*(c[:,:,k].ravel() for k in range(3))]):
            v=expit(logit(p)+d[np.searchsorted(bins,p),h]).reshape(target.shape);sq=(v-target)**2+target*(1-target)
            for i,pid in enumerate(ids):res.append(dict(beta=beta,lag=int(lag),fit_lag=train_lag,parent_id=int(pid),model=name,predicted=v[i].mean(),observed=target[i].mean(),brier=sq[i].mean(),logloss=-np.mean(target[i]*np.log(v[i])+(1-target[i])*np.log1p(-v[i]))))
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=res[0]);w.writeheader();w.writerows(res)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=['fit','score']);a.add_argument('--input',type=Path,required=True);a.add_argument('--operator',type=Path,required=True);a.add_argument('--calibration',type=Path,required=True);a.add_argument('--out',type=Path,required=True);v=a.parse_args()
    if v.mode=='fit':fit(v.input,v.operator,v.calibration,v.out)
    else:score(v.input,v.operator,v.calibration,v.out)
