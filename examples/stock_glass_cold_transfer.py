"""Frozen 3.05 transfer. Resample parents, never windows, null maps, or optimization seeds."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import xlogy
from stock_glass_forecast import SEEDS,REF,VIEWS,exterior,assignments,classify,sha
from stock_glass_cavity_exact import prob,X

RESULT=REF.parent; OUT=RESULT/'cold-frozen-transfer-320-340'; S05=RESULT/'experiment-05-inventory-transfer'; S07=RESULT/'experiment-07-boundary-forecast'; L2=np.log(2.)

def entropy(q):return -xlogy(q,q).sum(-1)/L2

def deficit(q):
    # Avoid 1 minus a number indistinguishable from 1 in strongly pinned stars.
    return 2*sum(q[...,a]*q[...,b] for a in range(4) for b in range(a))

def scatter(rows,ix,n):
    shape=rows.shape[1:];s=np.zeros((n,*shape));np.add.at(s,ix,rows)
    return s/np.maximum(1,np.bincount(ix,minlength=n)).reshape((-1,)+(1,)*len(shape))

def features(data,model,heads,k,beta,b,view):
    ids,ix=data['ids'],data['parent'];take=np.arange(len(ix)) if view=='all' else np.unique(ix,return_index=True)[1];ix=ix[take];p=prob(data['relative_E'][take],beta);obs=data['observed'][take];n=len(ids);maps=np.vstack((model['labels'][k],model['null_labels'][k]));lab=maps[0];q=(p@np.eye(4)[maps.T].reshape(210,-1)).reshape(len(p),65,4);q/=q.sum(-1,keepdims=True);df=deficit(q);h=entropy(q);dnull=df[:,1:].mean(1);anc=np.bincount(ix,minlength=n);observed=np.eye(4)[lab[obs]];q0=q[:,0,:];ss=1-df[:,0];base='s%d_'%SEEDS[k];mask=heads['masks'][k];rot=assignments(ids,b,b'07-rotate',24);feat,_=exterior(data['su'][take],data['fu'][take],rot[ix]);point={}
    def add(key,arr):point[key]=scatter(np.asarray(arr,float),ix,n)
    add('state_same',(p*p).sum(1));add('state_Hcond',entropy(p));add('state_effective',1/(p*p).sum(1));add('Hcond',h[:,0]);add('same',ss);add('ambiguous',ss<.8)
    for j in range(4):add(f'emp_h{j}',observed[:,j]);add(f'model_h{j}',q0[:,j]);add(f'gap_h{j}',observed[:,j]-q0[:,j])
    for j in range(3):
        pred=p@(X[:,0]==j);raw=(X[obs,0]==j).astype(float);add(f'emp_center{j}',raw);add(f'model_center{j}',pred);add(f'gap_center{j}',raw-pred)
    prior=heads[base+'prior'];add('prior_bits',-(q0[:,mask]*np.log2(prior[mask])).sum(1));add('prior_strength_mse',(ss-float(heads[base+'mean_s']))**2)
    for vn,fx in zip(VIEWS,feat):
        prefix=base+vn+'_';pr,lp=classify(fx,heads[prefix+'W'],mask);add(vn+'_bits',-(q0[:,mask]*lp).sum(1)/L2);top=pr.argmax(1);add(vn+'_mode_match',top==q0.argmax(1));add(vn+'_expected_mass',q0[np.arange(len(q)),top]);v=heads[prefix+'v'];st=np.clip(fx@v[:-1]+v[-1],.25,1.);add(vn+'_strength_mse',(st-ss)**2)
    for nm,f in [('stock',model['coeffs'][k][lab]),('linear',model['linear']),('vac',(X[:,0]==0).astype(float))]:
        mu=p@f;add(nm+'_mu',mu);add(nm+'_f2',p@(f*f));add(nm+'_wv',(p*(f[None,:]-mu[:,None])**2).sum(1))
    D=scatter(df,ix,n);point['df2']=D[:,0]**2;point['ndf2']=D[:,1:].mean(1)**2
    keys=list(point);pack=np.column_stack([point[t] for t in keys]);features=np.column_stack([pack,scatter(q,ix,n).reshape(n,-1),D,scatter(h[:,1:],ix,n)]);return features,keys,anc

def calc(weights,f,keys,count):
    w=weights*(count>0);n=w.sum(1);a=(w@f)/n[:,None];m={k:a[:,i] for i,k in enumerate(keys)};i=len(keys);q=a[:,i:i+260].reshape(-1,65,4);d=a[:,i+260:i+325];null_h_cond=a[:,i+325:];den=deficit(q);A=1-d/den;H=entropy(q);m.update(learned_A=A[:,0],null_A=A[:,1:].mean(1),advantage=A[:,0]-A[:,1:].mean(1),H=H[:,0],MI=H[:,0]-m['Hcond'],null_H=H[:,1:].mean(1),null_Hcond=null_h_cond.mean(1),null_MI=(H[:,1:]-null_h_cond).mean(1),null_same=1-d[:,1:].mean(1),indep=1-den[:,0],null_indep=1-den[:,1:].mean(1),denom=den[:,0],min_null_denom=den[:,1:].min(1),deficit_ess=n*d[:,0]**2/np.maximum(m['df2'],1e-300),null_deficit_ess=n*d[:,1:].mean(1)**2/np.maximum(m['ndf2'],1e-300),forecast_gain=m['packing_bits']-m['directional_bits'],rotated_gain=m['rotated_bits']-m['directional_bits'],mode_match=m['directional_mode_match'])
    for nm,tk in [('stock','T'),('linear','linear_T'),('vac','vac_T')]:
        mu=m[nm+'_mu'];tot=m[nm+'_f2']-mu**2;m[tk]=1-m[nm+'_wv']/np.where(tot>1e-12,tot,np.nan)
        if nm=='stock':m['within_var']=m[nm+'_wv'];m['total_var']=tot;m['between_var']=tot-m['within_var'];
    m['T_minus_linear']=m['T']-m['linear_T'];m['T_minus_vac']=m['T']-m['vac_T']
    allow=[k for k in m if k not in ('df2','ndf2') and not k.endswith(('_f2','_mu','_wv'))];return {k:m[k] for k in allow},A,H

def evaluate(weights,f,keys,counts):
    chunks=[];ra=[];rh=[]
    for i in range(0,len(weights),128):
        m,a,h=calc(weights[i:i+128],f,keys,counts);chunks.append(m);ra.append(a);rh.append(h)
    return {k:np.concatenate([m[k] for m in chunks]) for k in chunks[0]},np.concatenate(ra),np.concatenate(rh)

def bootstrap(b,n,B):return np.random.default_rng([120334,b,2]).multinomial(n,np.full(n,1/n),size=B).astype(float)

def ci(arr):return np.quantile(arr,[.025,.975],axis=0)

def row(ctx,key,x):
    l,h=ci(x[1:]);return dict(**ctx,metric=key,value=x[0],lo=l,hi=h)

def source_bank(banks,spec):
    path=banks/'b09-test.npz';assert sha(path)==spec['pins']['source_test'];
    with np.load(path) as z:return dict(ids=z['all_ids'][z['selected']],parent=np.arange(len(z['selected'])),**{k:z[k] for k in ['relative_E','observed','su','fu']})

def age_energy(a,c,w,beta):
    # Take integer differences before scaling; a zero endpoint of the bootstrap CI stays exactly zero.
    vals=np.column_stack((a,c,c.astype('i8')-a))
    draws=(w@vals)/(8000*w.sum(1)[:,None])
    return [row(dict(beta=beta,n=len(a)),key,z) for key,z in zip(('300M','500M','paired_500_minus_300'),draws.T)]

def run(banks,out):
    spec=json.loads((OUT/'design.json').read_text());targets={};allm={};es=[];deltas=[];nulls=[];support=[];age_rows=[];matches=[];fin=[];dec=[];checks=[];mask0={};weights={};summary={};atoms={};cold07=pd.read_csv(S07/'held-scores.csv');r03=pd.read_csv(REF/'corrected/frozen-readout.csv');r05=pd.read_csv(S05/'categorical.csv');r05n=pd.read_csv(S05/'random-partitions.csv');r05n=r05n[(r05n.sector==223)&(r05n['index']==9)];heads_path=S07/'fit-b09.npz';assert sha(heads_path)==spec['pins']['heads07'];src=source_bank(banks,spec)
    receipt=json.loads((out/'bank-receipt.json').read_bytes())
    assert receipt['design_sha256']==sha(OUT/'design.json') and all(sha(out/name)==value for name,value in receipt['artifacts'].items())
    for b in (10,11):
        for age in ('300','500'):
            with np.load(out/f'target-{b}-{age}.npz') as z:targets[b,age]={k:z[k] for k in z.files}
        a,c=targets[b,'300'],targets[b,'500'];np.testing.assert_array_equal(a['ids'],c['ids']);weights[b]=bootstrap(b,len(c['ids']),4096);mask0[b]=(a['counts']>0)&(c['counts']>0)
        age_rows+=age_energy(a['energy'],c['energy'],np.vstack([np.ones(len(a['ids'])),weights[b]]),3.2 if b==10 else 3.4)
    bwsrc=bootstrap(9,len(src['ids']),2048);rng=np.random.default_rng([120334,9,128]);matchW=np.zeros((1024,len(src['ids'])),dtype='u1');
    for r in matchW:r[rng.choice(len(r),128,replace=False)]=1
    with np.load(out/'frozen.npz') as zz,np.load(heads_path) as hh:
        model={k:zz[k] for k in zz.files};heads={k:hh[k] for k in hh.files};
        for b,beta,age,view in [(9,3.05,'source','primary')]+[(b,beta,age,view) for b,beta in [(10,3.20),(11,3.40)] for age in ('300','500') for view in ('primary','all')]:
            data=src if b==9 else targets[b,age];w=np.vstack([np.ones(len(data['ids'])),bwsrc if b==9 else weights[b]]);
            for k,seed in enumerate(SEEDS):
                f,keys,count=features(data,model,heads,k,beta,b,view);m,ra,rh=evaluate(w,f,keys,count);ctx=dict(beta=beta,age=age,view=view,seed=seed,n=int(sum(count>0)),stars=int(count.sum()));allm[b,age,view,seed]=m;es.extend(row(ctx,key,x) for key,x in m.items());d=mask0.get(b);valid=count>0;ns=valid.sum();j0=keys.index('stock_mu')
                meanf=f[valid,j0].mean();tot=f[valid,keys.index('stock_f2')]-2*meanf*f[valid,j0]+meanf**2;tes=tot.sum()**2/(tot@tot);rarity=min(m['deficit_ess'][0],m['null_deficit_ess'][0])<10 or min(m['denom'][0],m['min_null_denom'][0])<.05;summary[b,age,view,seed]=dict(n=ns,total_ess=tes,rare=rarity);support.append(dict(**ctx,total_ess=tes,rare_deficit=rarity,deficit_ess=m['deficit_ess'][0],null_deficit_ess=m['null_deficit_ess'][0],median_parent_deficit=np.median(f[valid,len(keys)+260]),max_parent_deficit=f[valid,len(keys)+260].max(),eligible_min=count[valid].min(),eligible_max=count.max()));
                for nm in ('stock','linear','vac'):
                    v=f[valid,keys.index(nm+'_wv')];support[-1][nm+'_within_ess']=v.sum()**2/(v@v) if (v@v)>0 else 0.
                    en=f[valid,keys.index(nm+'_mu')].mean();tv=f[valid,keys.index(nm+'_f2')]-2*en*f[valid,keys.index(nm+'_mu')]+en**2;support[-1][nm+'_total_ess']=tv.sum()**2/(tv@tv) if (tv@tv)>0 else 0.
                for j in range(65):
                    lo,hi=ci(ra[1:,j]);nulls.append(dict(**ctx,partition=j,normalized=ra[0,j],lo=lo,hi=hi,H_bits=rh[0,j]))
                if b==9:
                    o=r03[(r03['index']==9)&(r03.seed==seed)&(r03.channel=='hard')&r03.final].iloc[0];c=r05[(r05['index']==9)&(r05.sector==223)&(r05.seed==seed)].iloc[0];prev=cold07[(cold07['index']==9)&(cold07.seed==seed)].set_index('view');pairs={'T':o['T'],'learned_A':c.normalized,'null_A':c.random_mean_normalized,'advantage':c.true_minus_random_normalized,'null_H':c.random_entropy_mean/L2,'H':c.entropy/L2,**{v+'_bits':prev.loc[v,'bits'] for v in ('prior',*VIEWS)}};orig=r05n[r05n.seed==seed].sort_values('permutation');assert len(orig)==64;np.testing.assert_allclose(ra[0,1:],orig.normalized,atol=1e-10,rtol=0)
                    for key,old in pairs.items():np.testing.assert_allclose(m[key][0],old,atol=1e-9,rtol=0);checks.append(dict(seed=seed,metric=key,replayed=m[key][0],published=old,difference=m[key][0]-old))
                    mm,_,_=evaluate(matchW,f,keys,count)
                    for key in ('T','MI','H','Hcond','advantage','state_same','linear_T','vac_T','forecast_gain'):
                        x=mm[key];lo,hi=ci(x);matches.append(dict(seed=seed,metric=key,n=128,subsets=len(x),full=m[key][0],mean=x.mean(),median=np.median(x),p025=lo,p975=hi))
                else:
                    # Exact delete-one-parent sensitivity of the nonlinear entropy functional. Conditional entropy is linear.
                    qm=f[valid,len(keys):len(keys)+4];bar=qm.mean(0);leave=(qm.sum(0)-qm)/(ns-1);leave=np.clip(leave,0,1);hjack=ns*entropy(bar)-(ns-1)*entropy(leave).mean();fin.append(dict(**ctx,I_plugin=m['MI'][0],I_jack=hjack-m['Hcond'][0],estimated_bias=m['H'][0]-hjack));
                    if not np.all(valid==d):mp,_,_=evaluate(w*d,f,keys,count)
                    else:mp=m
                    atoms[b,age,view,seed]=mp
            print('TRANSFER_SCORED',b,age,view,flush=True)
    for seed in SEEDS:
        for view in ('primary','all'):
            for name,p1,p0,rare in [('temperature500',allm[11,'500',view,seed],allm[10,'500',view,seed],any(summary[b,'500',view,seed]['rare'] for b in (10,11))),*((f'age{code}',atoms[b,'500',view,seed],atoms[b,'300',view,seed],any(summary[b,age,view,seed]['rare'] for age in ('300','500'))) for b,code in [(10,320),(11,340)])]:
                for key in ('advantage','MI','T','linear_T','vac_T','state_same','learned_A','null_A','H','Hcond','forecast_gain','same','total_var','within_var'):
                    deltas.append(row(dict(seed=seed,name=name,view=view,rare=bool(rare)),key,p1[key]-p0[key]))
            for b,beta in [(10,3.20),(11,3.40)]:
                m=allm[b,'500',view,seed];s=summary[b,'500',view,seed];lo,hi=ci(m['advantage'][1:]);dec.append(dict(seed=seed,view=view,endpoint=str(beta),retained=bool(np.quantile(m['T'][1:],.025)>.90 and s['total_ess']>=10 and m['total_var'][0]>1e-10),specificity=decision(lo,hi,s['rare']),n=s['n']));
            r=next(r for r in deltas if r['name']=='temperature500' and r['view']==view and r['seed']==seed and r['metric']=='advantage');dec.append(dict(seed=seed,view=view,endpoint='3.40 minus 3.20',specificity=decision(r['lo'],r['hi'],r['rare'])))
    for fn,rows in [('estimates',es),('contrasts',deltas),('partitions',nulls),('support',support),('age-energy',age_rows),('source-n128',matches),('entropy-sensitivity',fin),('source-replay',checks),('decisions',dec)]:pd.DataFrame(rows).to_csv(out/(fn+'.csv'),index=False,float_format='%.15g')

def decision(lo,hi,rare):return 'unresolved: rare support' if rare else ('above +.01' if lo>.01 else 'below +.01' if hi<.01 else 'unresolved: interval')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-banks',type=Path,required=True);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--prep',type=Path,help='Rebuild derived banks from the two pinned nonlocal snapshots');a=p.parse_args()
    if a.prep:
        from stock_glass_cold_banks import prepare
        receipt=prepare(a.prep,a.source_banks,a.out);(a.out/'bank-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    run(a.source_banks,a.out)
