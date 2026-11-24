"""Freeze/test one spatial prediction of the archived glass motif. No simulation.

Estimate an overlap-weighted high-minus-low contrast, comparing measured A
occupancy to the frozen conventional forecast in the *same* physical strata.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit
from scipy.stats import t

NSTRATA=9*7*5
DELTA=.002


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_bytes())


def save(p,x):
    tpath=p.with_suffix('.tmp');tpath.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tpath,p)


def array(bank,name):return np.load(bank/(name+'.npy'),mmap_mode='r',allow_pickle=False)


def data(bank,rule,model,calib):
    from cage_baseline import predict_bundle
    from motif import observable
    meta=read(bank/'meta.json');first=meta['fit_parents'] if calib else 0
    select=slice(first,None);v=array(bank,'v')[select];m=array(bank,'m')[select];b=array(bank,'bonds')[select];y=array(bank,'y')[select].astype(float)/3
    # Evaluate the formula and fitted forecast before looking at response contrasts.
    f=np.stack([observable(row,rule) for row in v]); assert f.shape==y.shape and np.all(np.isfinite(f))
    p=np.array(predict_bundle(bank,model)['full'][select],dtype=float);assert p.shape==y.shape and np.all((p>0)&(p<1))
    bc=b[:,:,:3].min(2); k=np.select((bc==32767,bc<=0,bc<=2,bc<=6),(4,0,1,2),default=3)
    c=((3*v[:,:,0].astype(int)+v[:,:,26])*7+m[:,:,0])*5+k
    return meta,c,f,p,y,(v[:,:,0]==1).astype(float)-(v[:,:,26]==1),array(bank,'parent_id')[select]


def cutoff(f):
    v,n=np.unique(f,return_counts=True)
    if len(v)<2:raise ValueError('motif constant on calibration parents; no physical contrast identifiable')
    return float(v[1:][np.argmin(np.abs(np.cumsum(n)[:-1]/n.sum()-.5))])


def aggregate(s,g,y,p):
    # Shape parent x stratum x arm x (count, observed, expected). Never bootstrap sites.
    out=np.zeros((len(s),NSTRATA,2,3)); mid=2*s+g
    for i,key in enumerate(mid):
        for j,w in enumerate((None,y[i],p[i])):out[i,:,:,j]=np.bincount(key,weights=w,minlength=2*NSTRATA).reshape(NSTRATA,2)
    return out


def overlap(a):
    n=a.sum(0)[:,:,0];enough=np.all((n>=32)&((a[:,:,:,0]>0).sum(0)>=8),axis=1)
    h=np.zeros(NSTRATA);h[enough]=2*n[enough,0]*n[enough,1]/n[enough].sum(1)
    if h.sum()==0:raise ValueError('no matched support; no meaningful transfer contrast')
    return h/h.sum()


def jack_stat(a,w,sign):
    use=w>0; a=a[:,use];w=w[use];total=a.sum(0); n=total[:,:,0]
    # No dropped strata at transfer: missing arm or single contributing parent invalidates the test.
    present=(a[:,:,:,0]>0).sum(0); valid=np.all(present>=2,1)
    support={'fraction_frozen_weight_with_both_arms':float(w[valid].sum()),'anchor_mass_in_frozen_strata':None,'strata':int(len(w)),'minimum_parents_per_arm':int(present.min())}
    if not valid.all():return {'status':'unidentified: no jackknife support in a frozen stratum','support':support}
    def contrast(x):
        av=x[...,1:]/x[...,0,None]; d=np.einsum('...tk,t->...k',av[...,1,:]-av[...,0,:],w);return np.stack((d[...,0],d[...,1],sign*(d[...,0]-d[...,1])),axis=-1)
    mean=contrast(total); P=len(a); estimates=[]
    for i in range(0,P,64):estimates.append(contrast(total-a[i:i+64]))
    leave=np.concatenate(estimates); se=np.sqrt((P-1)/P*((leave-leave.mean(0))**2).sum(0)); half=t.ppf(.975,P-1)*se
    return {'status':'ok','parents':P,'support':support,'contrast':{k:{'value':float(v),'se':float(s),'lo':float(v-h),'hi':float(v+h)} for k,v,s,h in zip(('observed','conventional','signed_excess'),mean,se,half)},'detectable_absolute_increment_at_80pct':float(2.802*se[-1])}


def extra(p,y,f):
    # Matched bias and one centered feature, calibrated only on the training suffix.
    f=f.ravel(); y=y.ravel(); lp=logit(p.ravel());center=float(f.mean());scale=float(f.std()) or 1.; x=(f-center)/scale; b=np.zeros(2)
    for i in range(30):
        q=expit(lp+b[0]+b[1]*x);ww=q*(1-q);g=q-y; grad=np.array([g.mean(),(g*x).mean()+1e-5*b[1]])
        h=np.array([[ww.mean(),(ww*x).mean()],[(ww*x).mean(),(ww*x*x).mean()+1e-5]])
        d=np.linalg.solve(h,grad);b-=d
        if np.linalg.norm(d)<1e-9:break
    return dict(center=center,scale=scale,offset=b[0],slope=b[1])


def adjusted(p,f,m):return logit(p)+m['offset']+m['slope']*(f-m['center'])/m['scale']


def evaluate(x,lock):
    meta,s,f,p,y,pol,ids=x; w=np.array(lock['strata_weights']); hi=f>=lock['threshold']; a=aggregate(s,hi,y,p);result=jack_stat(a,w,lock['direction'])
    result['support']['anchor_mass_in_frozen_strata']=float((w[s]>0).mean());result.update(beta=meta['beta'],index=meta['index'],fold=meta['fold'],parent_ids_sha256=hashlib.sha256(ids.astype('<u8').tobytes()).hexdigest(),anchor_rows=int(y.size),fraction_high=float(hi.mean()))
    # Secondary paired predictive gain, NOT another test of the primary prediction.
    pred=adjusted(p,f,lock['motif_adjustment']);base=adjusted(p,pol,lock['species_adjustment']);gain=((np.logaddexp(0,base)-y*base)-(np.logaddexp(0,pred)-y*pred)).mean(1)
    err=t.ppf(.975,len(gain)-1)*gain.std(ddof=1)/np.sqrt(len(gain));result['secondary_logscore_gain_over_matched_species_adjustment_nats']=[float(gain.mean()),float(gain.mean()-err),float(gain.mean()+err)]
    if result['status']=='ok':
        eff=result['contrast']['signed_excess'];result['decision']=('supports transferable increment' if eff['lo']>DELTA else 'excludes predicted signed increment' if eff['hi']<DELTA else 'inconclusive at sealed threshold');result['equivalent_within_two_per_thousand']=bool(eff['lo']>-DELTA and eff['hi']<DELTA)
    return result


def freeze(a):
    if a.lock.exists():raise FileExistsError('refusing to replace a sealed hypothesis')
    if read(a.train/'meta.json')['fold']!='train' or read(a.validation/'meta.json')['fold']!='validation':raise ValueError('discovery partitions only')
    rule=read(a.rule);x=data(a.train,rule,a.model,True); meta,s,f,p,y,pol,_=x;thr=cutoff(f);aa=aggregate(s,f>=thr,y,p);ww=overlap(aa);cal=jack_stat(aa,ww,1);v=cal['contrast']['signed_excess']['value'];sign=1 if v>=0 else -1
    source=read(a.data/'beta-06.json')['endpoint_sha256']  # sidecar only; no test features or outcomes
    lock=dict(index=6,fold='test',beta='2.033333333333',prediction=DELTA,metric='signed excess A occupancy probability high-minus-low over full calibrated-on-cold conventional baseline at distance 3, fixed overlap weights',
              direction=sign,threshold=thr,calibration_signed_before_orientation=v,strata_weights=ww.tolist(),
              model_path=os.path.relpath(a.model.resolve(),a.lock.parent.resolve()),model_sha256=sha(a.model),rule_path=os.path.relpath(a.rule.resolve(),a.lock.parent.resolve()),rule_sha256=sha(a.rule),source_sha256=source,
              motif_adjustment=extra(p,y,f),species_adjustment=extra(p,y,pol),extract_seed=318749,test_anchors_per_parent=256,
              baseline_readout='raw frozen full model; separate common-dimensional logit adjustments only for secondary log score',estimand_sha256=sha(a.lock.parent/'estimand.json'),implementation_sha256=sha(Path(__file__)))
    cv=evaluate(data(a.validation,rule,a.model,False),lock);lock['cold_validation']=cv
    lock['prediction_plain']=f"Fixed motif {'high minus low' if sign==1 else 'low minus high'} has at least 0.2 percentage points more outer A occupancy than explained by conventional predictors, within cold-defined strata, at beta=2.033333333333. A contrary or unresolved observation counts against this mechanism."
    save(a.lock,lock);print(lock['prediction_plain'],cv.get('contrast'),flush=True)


def score(a):
    if (a.out/'transfer.json').exists():raise FileExistsError('this transfer was already scored')
    lock=read(a.lock);rp=a.lock.parent/lock['rule_path'];mp=a.lock.parent/lock['model_path'];meta=read(a.bank/'meta.json')
    if not (lock['implementation_sha256']==sha(Path(__file__)) and sha(rp)==lock['rule_sha256'] and sha(mp)==lock['model_sha256'] and lock['estimand_sha256']==sha(a.lock.parent/'estimand.json') and meta['fold']==lock['fold'] and meta['index']==lock['index'] and meta['source_sha256']==lock['source_sha256'] and meta['reps']==lock['test_anchors_per_parent']): raise ValueError('physical test differs from lock')
    r=evaluate(data(a.bank,read(rp),mp,False),lock);r['lock_sha256']=sha(a.lock);save(a.out/'transfer.json',r);plot(lock,r,a.out/'transfer.png');print(json.dumps(r),flush=True)


def plot(lock,r,path):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9,3.8),layout='constrained');labels=['Cold validation','Frozen transfer'];co=['#176c79','#be7435'];ax.axhline(0,color='.6',lw=.6);bx.axhline(0,color='.6',lw=.6);bx.axhline(.2,color='#b14a49',ls='--',label='Locked prediction: +0.2 pp')
    for i,v in enumerate([lock['cold_validation'],r]):
        if v['status']!='ok':continue
        for j,n in enumerate(('observed','conventional','signed_excess')):
            m=v['contrast'][n]; aa=ax if j<2 else bx; xpos=i+(-.1 if j==0 else .1 if j==1 else 0); aa.errorbar(xpos,100*m['value'],yerr=100*np.array([[m['value']-m['lo']],[m['hi']-m['value']]]),fmt=['o','s','D'][j],color=(co[j] if j<2 else '#655085'),capsize=4,label=n.capitalize() if i==0 and j<2 else None)
    for a,title in [(ax,'Outer A occupancy · high minus low'),(bx,'Excess in the predicted direction')]:a.set(xticks=[0,1],xticklabels=labels,title=title,ylabel='Percentage points',xlim=(-.4,1.4));a.legend(frameon=False,fontsize=8);a.spines[['top','right']].set_visible(False)
    fig.savefig(path,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    for mod,keys in [('freeze',('train','validation','rule','model','data','lock')),('score',('bank','lock','out'))]:
        c=s.add_parser(mod)
        for k in keys:c.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();(freeze if a.mode=='freeze' else score)(a)
