"""Repeated whole-parent refits of the *cold conventional baseline*, never the transfer test.

A stability distribution, not a second hypothesis test. Frozen rule, strata, thresholds,
competitor list and model hyperparameters. Every replicate refits every stage and scores
both a fixed validation fold and a fresh bootstrap of its parents. No simulator or GPU.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from cage_baseline import (LADDER, TOL, digest, load_json, mmap, open_bank,
                           parent_loss, predict_bundle, probabilities, train_models)
from motif import observable

NAMES = ('pairs', 'coordination', 'full_class', 'full_threshold', 'full_blend')
KEY = 191803


def write_json(path,x):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def weight(vec, strata, arm, parents):
    """Exact stratum-standardized high-low weights, not a site bootstrap."""
    z=2*strata+arm; multiplicity=np.broadcast_to(parents[:,None],z.shape)
    cnt=np.bincount(z.ravel(),weights=multiplicity.ravel(),minlength=2*len(vec)).reshape(-1,2)
    if np.any(cnt[vec>0]<1): raise ValueError('missing frozen support: do not drop strata')
    coef=np.divide(vec[:,None]*np.array([-1.,1.]),cnt,out=np.zeros_like(cnt),where=cnt>0)
    return coef[strata,arm]*multiplicity


def summarize(out,work,about,reason):
    rec=[load_json(f) for f in sorted(work.glob('rep-*.json'))];n=len(rec)
    if not n:return
    if [v['rep'] for v in rec]!=list(range(n)) or any(v['plan']!=about for v in rec):raise ValueError('mixed bootstrap output')
    rows=[dict(rep=r['rep'],method=k,**v) for r in rec for k,v in r['methods'].items()];tmp=out/'replicates.csv.tmp'
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    os.replace(tmp,out/'replicates.csv');stats={}
    for k in NAMES:
        series={}
        for field in ('training_only','joint','log_loss'):
            a=[r['methods'][k][field] for r in rec];qt=np.quantile(a,[.025,.5,.975]);series[field]=dict(mean=float(np.mean(a)),sd=float(np.std(a,ddof=1)) if n>=2 else None,percentiles=qt.tolist() if n>=40 else None)
        stats[k]=series
    write_json(out/'progress.json',dict(completed=n,reason=reason,source=about,methods=stats,
        best_full_counts={k:sum(r['winner']==k for r in rec) for k in NAMES[2:]},simplest_counts={k:sum(r['simple']==k for r in rec) for k in LADDER},
        units='signed residual outer-A probability; joint resamples fitting and validation parents; training_only fixes all validation outcomes and strata. Percentiles describe the bootstrap distribution conditional on the selected motif and are not new test intervals.',
        elapsed_fit_seconds=sum(r['time_s'] for r in rec)))
    if n<8:return
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9,3.4),layout='constrained');labs=('Pair','Coord.','Costs: class','Costs: thresh.','Costs: blend');ax.axhline(0,c='.55',lw=.7)
    for j,k in enumerate(NAMES):
        for off,field,col,lab in ((-.12,'training_only','#147877','Refit parents only'),(.12,'joint','#ae6633','Refit + validation parents')):
            a=np.array([r['methods'][k][field] for r in rec])*100;lo,med,hi=np.quantile(a,[.1,.5,.9]);ax.errorbar(j+off,med,yerr=[[med-lo],[hi-med]],fmt='o',c=col,capsize=3,label=lab if j==0 else None)
    ax.set(xticks=range(len(NAMES)),xticklabels=labs,xlabel='Conventional readout',ylabel='Observed minus predicted (pp)',title='Middle 80% of parent resamples');ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=7)
    for field,col,lab in [('joint','#ae6633','Refit + validation'),('training_only','#147877','Refit only')]:
        a=np.array([r['methods']['full_blend'][field] for r in rec])*100;ids=np.arange(2,n+1);var=(np.cumsum(a*a)[1:]-np.cumsum(a)[1:]**2/ids)/(ids-1);bx.plot(ids,np.sqrt(np.maximum(0,var)),c=col,label=lab)
    bx.set(xlabel='Complete parent refits',ylabel='Bootstrap SD (pp)',title='Cold conventional blend');bx.legend(fontsize=7)
    for a in (ax,bx):a.spines[['top','right']].set_visible(False)
    tmp=out/'stability.tmp.png';fig.savefig(tmp,dpi=155);plt.close(fig);os.replace(tmp,out/'stability.png')


def run(a):
    a.work.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
    with (a.work/'writer.lock').open('a') as lease,threadpool_limits(limits=a.threads):
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        h=load_json(a.lock);rule=load_json(a.lock.parent/h['rule_path']);fit=load_json(a.lock.parent/'conventional-fits.json');bundle=a.lock.parent/h['model_path']
        if digest(a.lock.parent/h['rule_path'])!=h['rule_sha256'] or digest(bundle)!=h['model_sha256'] or fit['full_member']!='full_blend':raise ValueError('unexpected frozen discovery artifacts')
        tm,train=open_bank(a.train,(9,'train'));vm,val=open_bank(a.validation,(9,'validation'))
        for p,m,k in [(a.train,tm,'train'),(a.validation,vm,'validation')]:
            if digest(p/'meta.json')!=fit['banks'][k]['meta_sha256'] or any(digest(p/name)!=sha for name,sha in m['sha256'].items()):raise ValueError('changed source bank')
        cut=tm['fit_parents'];P=vm['parents'];v=val['v'];b=val['bonds'][:,:,:3].min(2);cat=np.select((b==32767,b<=0,b<=2,b<=6),(4,0,1,2),default=3)
        strata=((3*v[:,:,0].astype(int)+v[:,:,-1])*7+val['m'][:,:,0])*5+cat;arm=(observable(v,rule).reshape(v.shape[:2])>=h['threshold']).astype(int);vec=np.array(h['strata_weights']);yf=mmap(a.train,'y')[:cut];yc=mmap(a.validation,'y');y=yc/3.;direction=h['direction']
        w=weight(vec,strata,arm,np.ones(P));p0=predict_bundle(a.validation,bundle)['full_blend'];r0=direction*np.sum(w*(y-p0));target=h['cold_validation']['contrast']['signed_excess']['value'];
        if abs(r0-target)>1e-10:raise ValueError(f'cold weighted contrast differs: {r0} rather than {target}')
        plan=dict(lock=digest(a.lock),implementation=digest(Path(__file__)),baseline=digest(Path(__file__).with_name('cage_baseline.py')),seed=KEY,max_reps=a.max_reps,end_epoch=a.end_epoch,original_contrast=float(r0),fit_parents=cut,validation_parents=P)
        pp=a.out/'design.json'
        if pp.exists() and load_json(pp)!=plan:raise ValueError('resume design mismatch')
        write_json(pp,plan);about=digest(pp);recfiles=sorted(a.work.glob('rep-*.json'));elapsed=[load_json(f)['time_s'] for f in recfiles];n=len(elapsed);last=time.time();stop='finite bootstrap completed'
        while n<a.max_reps:
            reserve=max(240.,1.8*max(elapsed[-8:],default=120.))
            if time.time()+reserve>a.end_epoch:stop='absolute campaign deadline reserve';break
            start=time.time();rng=np.random.default_rng(np.random.SeedSequence([KEY,n]));ids=rng.integers(0,cut,size=cut);mult=np.bincount(rng.integers(0,P,size=P),minlength=P);w2=weight(vec,strata,arm,mult)
            sampled={k:data[ids] for k,data in train.items()};models=train_models(sampled,yf[ids].reshape(-1)/3.,cut);pr=probabilities(val,models);del sampled,models
            losses={k:float(np.average(parent_loss(v,yc),weights=mult)) for k,v in pr.items()};winner=min(NAMES[2:],key=losses.get);losses['full']=losses[winner];simple=next(k for k in LADDER if losses[k]<=min(losses[j] for j in LADDER)+TOL)
            res={k:dict(training_only=float(direction*np.sum(w*(y-pr[k]))),joint=float(direction*np.sum(w2*(y-pr[k]))),log_loss=losses[k]) for k in NAMES};dt=time.time()-start;elapsed.append(dt);write_json(a.work/f'rep-{n:04d}.json',dict(rep=n,plan=about,time_s=dt,methods=res,winner=winner,simple=simple));n+=1
            if n in (1,8,32,64,128,256,512) or time.time()-last>=1800:
                summarize(a.out,a.work,about,'in progress');print(f'REFIT_CHECKPOINT {n} at {time.strftime("%H:%M UTC",time.gmtime())}',flush=True);last=time.time()
        summarize(a.out,a.work,about,stop);print('REFIT_DONE',n,stop,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','validation','work','out','lock'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--end-epoch',required=True,type=int);p.add_argument('--max-reps',type=int,default=1024);p.add_argument('--threads',type=int,default=3);run(p.parse_args())
