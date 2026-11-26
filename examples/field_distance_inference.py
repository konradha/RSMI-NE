"""Independent-parent estimates of the locked disjoint-star signed correlations."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from field_distance_inputs import COHORTS,DISTS,OUT,save_json,sha,need

SEEDS=(6101,6102,6103)
DIRS=('L','T')
PRIMARY=(DISTS.index(5),DISTS.index(6))
CONTRASTS=('R','R_minus_shuffle','R_minus_shuffle_minus_random_corrected')


def stats(x,level=1):
    x=x[np.isfinite(x)]; n=len(x)
    if n<2:return dict(n=n,mean=np.nan,se=np.nan,low=np.nan,high=np.nan,halfwidth=np.nan)
    m=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n));h=float(t.ppf(1-.05/(2*level),n-1)*se)
    return dict(n=n,mean=m,se=se,low=m-h,high=m+h,halfwidth=h)


def report(scratch):
    sc=json.loads((OUT/'score-receipt.json').read_text());need(sc['complete'],'incomplete scoring');
    bank=pd.read_csv(OUT/'field-bank.csv').set_index('name'); maps=json.loads((OUT/'mapping-receipt.json').read_text());
    with np.load(OUT/'fields.npz') as z:names=list(z['names'].astype(str))
    need(set(s for s in names)==set(bank.index),'bank names mismatch');m=len(names);dist_rows=[];primary=[];support=[];comparison=[]
    for tag,(_,_,n) in COHORTS.items():
        arr=np.empty((n,2,m,len(DISTS),2));pairs=np.zeros((n,len(DISTS),3),np.int32);seen=np.zeros(n,bool);eligible=np.zeros(n,np.int32)
        with np.load(scratch/f'{tag}-parents.npz') as z:ids=z['ids'];expelig=z['eligible']
        for r in sc['rows'][tag]:
            lo=r['lo'];end=lo+r['parents'];i=slice(lo,end);path=scratch/f'{tag}-{lo:05d}.npz'
            need(sha(path)==r['sha256'] and not np.any(seen[i]),'chunk digest or overlap');seen[i]=True
            with np.load(path) as z:np.testing.assert_array_equal(z['ids'],ids[i]);arr[i]=z['corr'];eligible[i]=z['eligible'];pairs[i]=z['counts']
        need(seen.all(),'row coverage');np.testing.assert_array_equal(eligible,expelig)
        ok=(pairs[:,PRIMARY,:]>=2).all(axis=(1,2));nkeep=int(ok.sum());min_pairs=int(pairs[ok][:,PRIMARY,:].sum(axis=0).min());supp=nkeep>=96 and nkeep>=.8*n and min_pairs>=300
        for k,d in enumerate(DISTS):
            d_ok=(pairs[:,k,:]>=2).all(1)
            for a,axis in enumerate('zyx'):
                cc=pairs[:,k,a];support.append(dict(cohort=tag,d=d,axis=axis,roster=n,eligible_stars=eligible.sum(),supported_all_axes=int(d_ok.sum()),n_primary=nkeep,all_pairs=int(cc.sum()),pairs_on_primary_parents=int(cc[ok].sum()),min_pairs=int(cc.min()),median_pairs=np.median(cc),max_pairs=int(cc.max()),parents_with_0_or_1=int((cc<2).sum()),primary_support_pass=bool(supp)))
        pooled=arr[:,:,:,PRIMARY,:].mean(axis=3);need(np.isfinite(pooled[ok]).all() and np.isnan(pooled[~ok]).all(),'primary support disagreement')
        np.savez_compressed(OUT/f'{tag}-parent-primary.npz',native_ids=ids,eligible_centers=eligible,pair_counts=pairs,distances=DISTS,names=np.asarray(names),pooled=pooled,channels=np.array(['observed','within_parent_shuffle']),polarizations=DIRS)
        # All state permutations remain separately identifiable in the rows and parent arrays.
        for mi,name in enumerate(names):
            for source,curves in [('observed',arr[:,0,mi]),('shuffle',arr[:,1,mi]),('corrected',arr[:,0,mi]-arr[:,1,mi])]:
                curves=np.concatenate([curves,curves[:,PRIMARY].mean(1)[:,None,:]],1)
                for di,d in enumerate((*DISTS,'pool')):
                    for pol,direction in enumerate(DIRS):
                        st=stats(curves[:,di,pol]);vr=bank.at[name,'normalized_variance'];dist_rows.append(dict(cohort=tag,model=name,d=d,channel=source,polarization=direction,**st,train_variance_fraction=vr,mean_in_own_train_variance=st['mean']/vr if vr>0 else np.nan,low_in_own_train_variance=st['low']/vr if vr>0 else np.nan,high_in_own_train_variance=st['high']/vr if vr>0 else np.nan))
        for seed in SEEDS:
            sel=lambda mode:names.index(f'{mode}_s{seed}');rnd=lambda mode:[k for k,name in enumerate(names) if name.startswith(f'random_{mode}_s{seed}_d')];nr=rnd('enriched_residual');nt=rnd('total');need(len(nr)==len(nt)==8,'random selection size')
            real=pooled[:,:,sel('enriched_residual')]; rand=pooled[:,:,nr].mean(axis=2);rand_total=pooled[:,:,nt].mean(axis=2);corrected=real[:,0]-real[:,1]; excess=corrected-(rand[:,0]-rand[:,1]);total=pooled[:,:,sel('total')];basic=pooled[:,:,sel('basic_residual')];pk=pooled[:,:,names.index('packing_enriched')]
            rec=maps['seeds'][str(seed)];vpass=rec['enriched_residual_fraction']>=.01 and rec['total_raw_variance']>1e-10
            for di,direction in enumerate(DIRS):
                for name,v in zip(CONTRASTS,[real[:,0],corrected,excess]):
                    st=stats(v[:,di],72 if tag!='source-train' else 1);primary.append(dict(cohort=tag,seed=seed,polarization=direction,contrast=name,**st,roster=n,primary_parents=nkeep,min_eligible_pairs=min_pairs,support_pass=supp,variance_pass=vpass,matching_pass=rec['matching_pass_both_at_0p10'],precision_pass=st['halfwidth']<=.01,equivalent_total_units=st['low']>-.01 and st['high']<.01,variance_fraction=rec['enriched_residual_fraction'],raw_variance=rec['enriched_residual_raw_variance'],total_raw_variance=rec['total_raw_variance']))
                for name,v in [('total',total[:,0]),('enriched_packing',pk[:,0]),('total_minus_enriched_packing',total[:,0]-pk[:,0]),('basic_residual',basic[:,0]),('enriched_residual',real[:,0]),('matched_random_total',rand_total[:,0]),('learned_minus_random_total',total[:,0]-rand_total[:,0]),('matched_random_enriched_residual',rand[:,0]),('learned_minus_random_enriched_residual',real[:,0]-rand[:,0]),('matched_random_corrected_enriched_residual',rand[:,0]-rand[:,1])]:comparison.append(dict(cohort=tag,seed=seed,polarization=direction,comparison=name,**stats(v[:,di])))
    curves=pd.DataFrame(dist_rows);curves.to_csv(OUT/'distance-curves.csv.gz',index=False);pp=pd.DataFrame(primary);pp.to_csv(OUT/'primary.csv',index=False);pd.DataFrame(support).to_csv(OUT/'pair-support.csv',index=False);pd.DataFrame(comparison).to_csv(OUT/'matched-comparisons.csv',index=False)
    held=pp[pp.cohort!='source-train'];cold=held[held.cohort!='beta-zero'];zero=held[held.cohort=='beta-zero'];checks={}
    for pol in DIRS:
        sel=cold[cold.polarization==pol];zz=zero[zero.polarization==pol];common=bool(sel.support_pass.all() and sel.precision_pass.all() and sel.variance_pass.all() and sel.matching_pass.all() and zz.support_pass.all() and zz.precision_pass.all() and zz.equivalent_total_units.all())
        checks[pol]=dict(positive=common and bool((sel.low>.01).all()),negative=common and bool((sel.high<-.01).all()),all_physical_positive_margins=bool((sel.low>.01).all()),all_physical_negative_margins=bool((sel.high<-.01).all()),beta_zero_intervals_equivalent=bool(zz.equivalent_total_units.all()),all_72_intervals_defined=bool(len(held)==72 and np.isfinite(held['mean']).all()))
    r=cold[cold.contrast=='R'];decision=dict(reproducible_novel_direction=any(v['positive'] or v['negative'] for v in checks.values()),direction_decisions=checks,matched_nulls_in_caliper_all_seeds=bool(held.matching_pass.all()),residual_variance_pass_all_seeds=bool(held.variance_pass.all()),raw_conditional_residual_intervals_inside_margin_all_physical=bool(r.equivalent_total_units.all()),max_abs_simultaneous_raw_residual_endpoint=float(np.abs(r[['low','high']].to_numpy()).max()),practical_margin=.01,simultaneous_family_size=72,primary_support_all=bool(held.support_pass.all()),primary_precision_all=bool(held.precision_pass.all()),input_parents=sum(v[2] for v in COHORTS.values()),scope='Only this signed d=5/6 two-point, frozen 223-sector vector field; not arbitrary nonlinear dependencies, all encoders or all temperatures.',gpu_seconds=0,new_mcmc=0)
    save_json(OUT/'decision.json',decision);print(json.dumps(decision),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('scratch',type=Path);report(ap.parse_args().scratch)
