"""05: exact on each target's own parents; only analytic ordinary comparators use target TRAIN."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import xlogy
from stock_glass_inventory_banks import OUT,REF,DESIGN,SECTORS,SOURCE,SEEDS,Sector,pinned,stock,prob
from stock_glass_cavity_memory import moments,solve,orient,interval,compare,add
from stock_glass_boundary_couplings import scalar,mean_if,diff


def entropy(m):
    m=m/m.sum();return float(-np.sum(xlogy(m,m)))


def measure_scalar(p,f,b):
    if f is None:
        blank=(np.nan,np.full(len(p),np.nan));return {k:blank for k in ('T','between','total')}
    m,_=scalar(p,f)
    if m['total'][0]<1e-12:m['T']=(np.nan,np.full(len(p),np.nan))
    if b==0 and np.isfinite(m['T'][0]):assert m['T'][0]<1e-24;m['T']=(0.,np.zeros(len(p)));m['between']=(0.,np.zeros(len(p)))
    return m


def labels_metric(y,b,counts):
    m=y.mean(0);m/=m.sum();center=y-y[0];center-=center.mean(0);n=len(y);distinct=np.count_nonzero(counts)
    if distinct==1:center[:]=0
    if b==0:assert np.max(abs(center))<1e-13;center[:]=0
    num=mean_if((center*center).sum(1));d=2*sum(m[i]*m[j] for i in range(4) for j in range(i));den=(d,-2*(center@m));ratio=(num[0]/d,(num[1]-num[0]/d*den[1])/d) if d>0 else (np.nan,np.full(n,np.nan))
    same=float((y*y).sum(1).mean());ind=float(m@m);assert abs(same-ind-num[0])<2e-12 and (not np.isfinite(ratio[0]) or -.0000001<=ratio[0]<=1.0000001)
    return {'numerator':num,'denominator':den,'normalized':ratio},m,entropy(m),same,ind


def controls(p,labels,seed,s,b,ctx,individual):
    rng=np.random.default_rng([509005,*map(int,s),seed]);lab=np.stack([labels]+[labels[rng.permutation(len(labels))] for _ in range(64)]);counts=np.bincount(labels,minlength=4);assert all(np.array_equal(np.bincount(v,minlength=4),counts) for v in lab)
    # Same fixed assignments for every exterior. Matrix multiplication batches null partitions, not parents.
    assignment=np.eye(4)[lab.T].reshape(len(labels),65*4);Y=(p@assignment).reshape(len(p),65,4);fl=[];H=[];tot={};distrib={k:[] for k in ('numerator','normalized')}
    for k in range(65):
        metrics,m,h,ss,ind=labels_metric(Y[:,k,:],b,counts);row={**ctx,'permutation':k,'learned':k==0,'entropy':h,'effective_labels':np.exp(h),'same_label_shared':ss,'same_label_independent':ind,**{f'P{i}':x for i,x in enumerate(m)},**{f'states_label{i}':int(x) for i,x in enumerate(counts)}}
        for key,val in metrics.items():add(row,key,interval(val))
        if k==0:true=metrics;truth=row
        else:
            H.append(h);fl.append(m)
            for key in distrib:
                val=metrics[key];distrib[key].append(val[0]);tot[key]=tot.get(key,np.zeros(len(p)))+val[1];add(row,'true_minus_'+key,interval(diff(true[key],val)))
            individual.append(row)
    out={**truth,'random_entropy_mean':np.mean(H),'random_entropy_p025':np.quantile(H,.025),'random_entropy_p975':np.quantile(H,.975),'entropy_difference':truth['entropy']-np.mean(H)}
    for i,x in enumerate(np.mean(fl,axis=0)):out[f'random_P{i}_mean']=x
    for key,data in distrib.items():
        av=float(np.mean(data));base=(av,tot[key]/64);add(out,'random_mean_'+key,interval(base));add(out,'true_minus_random_'+key,interval(diff(true[key],base)));out['random_p025_'+key],out['random_p975_'+key]=np.quantile(data,[.025,.975]);out['random_fraction_ge_true_'+key]=float(np.mean(np.array(data)>=true[key][0])) if np.isfinite(av) else np.nan
    return out


def baseline(p,x,s,b,fixed):
    m,c,S=moments(p);use=m>=1e-10;ess=p.sum(0)**2/np.maximum((p*p).sum(0),1e-300);common=(ess>=100)&(m>=1e-3);eye=np.eye(len(m));line=np.concatenate([x==1,x==2],axis=1).astype('d');ev,v=solve(c,S,eye[:,use]);_,l=solve(c,S,line);oracle=orient(v[:,:1],m)[:,0];linear=orient(l[:,:1],m)[:,0]
    if s==SOURCE:oracle=fixed[f'b{b:02}_oracle'];linear=fixed[f'b{b:02}_linear']  # reuse published estimands even within near-degenerate eigenspaces
    cm=None
    if common.any():
        _,u=solve(c,S,eye[:,common]);cm=orient(u[:,:1],m)[:,0] if u.shape[1] else None
    var=m*(oracle-m@oracle)**2;ix=np.argsort(-(oracle-m@oracle)**2);end=np.searchsorted(var[ix].cumsum(),.5*var.sum())+1
    return {'oracle':oracle,'linear':linear,'vacancy':(x[:,0]==0).astype('d'),'common_oracle':cm},dict(train_supported=int(use.sum()),common_supported=int(common.sum()),train_effective_states=np.exp(entropy(m)),train_top=ev[0],train_gap=ev[0]-ev[1] if len(ev)>1 else np.nan,half_variance_mass=m[ix[:end]].sum()),m,ess


def reader(b,s,side,g,a,receipt,meta):
    name=f'b{b:02}-{side}.npz' if s==SOURCE else f'b{b:02}-{s}-{side}.npz';root=a.source_banks if s==SOURCE else a.banks;h=meta['inputs'][name] if s==SOURCE else receipt['target_banks'][name];assert stock.sha(root/name)==h
    with np.load(root/name) as z:
        E=z['relative_E'] if s==SOURCE else g.energies(z['fc'],z['su'],z['fu']);p=prob(E,float(meta['coverage'][b]['beta']));sel=z['selected'];obs=z['observed'];assert np.array_equal(np.flatnonzero(z['eligible_counts']),sel)
    assert np.isfinite(p).all() and np.max(abs(p.sum(1)-1))<1e-14
    if b==0:assert np.max(abs(p-1/len(g.x)))<1e-14
    return p,obs


def run(a):
    meta=pinned(a.reference);receipt=json.loads((a.out/'bank-receipt.json').read_bytes());assert receipt['design']==stock.sha(DESIGN) and receipt['transferred']==stock.sha(a.out/'frozen-transfer.npz') and receipt['source_manifest']==stock.sha(a.source_banks/'prepared.json');weights=np.load(a.out/'frozen-transfer.npz');fixed=np.load(a.reference/'frozen-modes.npz');prior=pd.read_csv(a.reference/'corrected/frozen-readout.csv');prior=prior[prior.final&(prior.channel=='hard')];bench=pd.read_csv(a.reference/'benchmark.csv');rows=[];nulls=[];cr=[];ordinary=[];support=[];states=[];checks=[];basis={};geoms={s:Sector(s) for s in SECTORS}
    for s,g in geoms.items():
        for b in range(10):
            tr,otr=reader(b,s,'train',g,a,receipt,meta);te,ote=reader(b,s,'test',g,a,receipt,meta);ctx=dict(sector=s,index=b,beta=float(meta['coverage'][b]['beta']),train_parents=len(tr),test_parents=len(te));tag=f'{s}_b{b:02}';ops,stats,m,ess=baseline(tr,g.x,s,b,fixed);mt=te.mean(0);collision=mean_if((te*te).sum(1));sup={**ctx,**stats,'states':len(m),'test_effective_states':np.exp(entropy(mt)),'test_observed_distinct':len(np.unique(ote)),'train_observed_distinct':len(np.unique(otr))};add(sup,'state_coincidence',interval(collision));support.append(sup)
            for i,(p,q,e) in enumerate(zip(m,mt,ess)):states.append(dict(**ctx,state=i,pattern=''.join('.AB'[j] for j in g.x[i]),train_mass=p,test_mass=q,train_weight_ess=e,train_observed=int(np.sum(otr==i)),test_observed=int(np.sum(ote==i))))
            ref={k:measure_scalar(te,v,b) for k,v in ops.items()};oracle=ref['oracle']['T'];line=ref['linear']['T'];vac=ref['vacancy']['T'];
            for k,fn in ops.items():
                row=dict(**ctx,observable=k);m0=ref[k]
                for key,val in m0.items():add(row,key,interval(val))
                add(row,'oracle_minus',compare(oracle,m0['T']));add(row,'fraction',compare(m0['T'],oracle,True));ordinary.append(row)
                if fn is not None:basis[tag+'_'+k]=fn
                if s==SOURCE and k in ('oracle','linear','common_oracle'):
                    rr=bench[(bench['index']==b)&(bench.observable==('common_mass_oracle' if k=='common_oracle' else k))].iloc[0];v0=np.array(interval(m0['T']));v1=rr[['T','T_lo','T_hi']].to_numpy(float);np.testing.assert_allclose(v0,v1,atol=1e-9,rtol=0,equal_nan=True);checks.append(dict(index=b,observable=k,match=True,published=rr['T'],replayed=v0[0]))
            for seed in SEEDS:
                stem=f'b{b:02}-s{seed}';lab=weights[stem+'_'+s];a0=weights[stem+'_a'];available=weights[stem+'_identified'];sm=weights[stem+'_source_mass'];has=np.bincount(lab,minlength=4)>0;missing=has&~available;mass=np.bincount(lab,weights=mt,minlength=4);one=int(has.sum())==1;status='constant_target' if one else 'unidentified_source_coefficients' if missing.any() else 'identified';fn=np.zeros(len(lab)) if one else None if missing.any() else a0[lab];metric=measure_scalar(te,fn,b);T=metric['T'];status='tiny_target_variance' if status=='identified' and not np.isfinite(T[0]) else status;con=dict(**ctx,seed=seed)
                learned=dict(**con,scalar_status=status,missing_labels=','.join(map(str,np.flatnonzero(missing))),target_missing_mass=mass[missing].sum(),source_min_mass_active=sm[has].min(),target_label_count=int(has.sum()))
                for key,val in metric.items():add(learned,key,interval(val))
                add(learned,'fraction',compare(T,oracle,True));add(learned,'minus_linear',compare(T,line));add(learned,'minus_vacancy',compare(T,vac));rows.append(learned);cc=controls(te,lab,seed,s,b,con,nulls);cr.append(cc)
                if s==SOURCE:
                    rr=prior[(prior['index']==b)&(prior.seed==seed)].iloc[0];a1=[learned[k] for k in ('T','T_lo','T_hi','between','total','fraction','fraction_lo','fraction_hi','minus_linear','minus_linear_lo','minus_linear_hi')];a2=rr[['T','T_lo','T_hi','between_test','total_test','fraction','fraction_lo','fraction_hi','minus_linear','minus_linear_lo','minus_linear_hi']].to_numpy(float);np.testing.assert_allclose(a1,a2,rtol=0,atol=1e-9,equal_nan=True);checks.append(dict(index=b,observable=stem,match=True,published=rr['T'],replayed=T[0]))
        print('SCORED_SECTOR',s,flush=True)
    for name,records in [('frozen-scores',rows),('ordinary',ordinary),('categorical',cr),('random-partitions',nulls),('support',support),('states',states),('source-equality',checks)]:pd.DataFrame(records).to_csv(a.out/(name+'.csv'),index=False,float_format='%.13g')
    np.savez_compressed(a.out/'target-comparators.npz',**basis)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,default=REF);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--banks',type=Path,required=True);p.add_argument('--source-banks',type=Path,required=True);run(p.parse_args())
