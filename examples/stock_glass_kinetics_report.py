"""Pool exact parent moments; parent influences preserve pairing of observables and baselines."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student
from stock_glass_local_kinetics import OUT,REF,NAMES,LAGS,WIDTH,SEEDS


def ratio(n,d):
    if np.mean(d)<=0:return np.nan,np.full(len(d),np.nan)
    r=n.mean()/d.mean();return r,(n-r*d)/d.mean()

def write(row,key,r):
    value,inf=r;hw=student.ppf(.975,len(inf)-1)*inf.std(ddof=1)/np.sqrt(len(inf));row.update({key:value,key+'_lo':value-hw,key+'_hi':value+hw})

def sub(a,b):return a[0]-b[0],a[1]-b[1]

def support(u,v,valid):
    n=len(u);between=(u-u.mean())**2;tot=v.mean()+between.mean();ess=v.sum()**2/(v@v) if v.sum()>0 else 0.;ordered=np.sort(v)[::-1];share=np.cumsum(ordered)/v.sum() if ess else np.zeros(n);frac=v.mean()/tot if tot>0 else np.nan
    return dict(valid=bool(valid),adequate=bool(valid and frac>=1e-5 and ess>=100),within_mean=v.mean(),within_total_fraction=frac,total_variance=tot,within_ESS=ess,top_one_percent_share=share[int(np.ceil(n*.01))-1] if ess else np.nan,parents_for_half=np.searchsorted(share,.5)+1 if ess else 0,peak_parent_share=share[0] if ess else np.nan),between

def report(out):
    rec=json.loads((out/'matrix-receipt.json').read_bytes());summary=[];spectra=[];thermal=[];paired=[];detailed=[];bench=pd.read_csv(REF/'benchmark.csv');old=pd.read_csv(REF/'corrected/frozen-readout.csv');max_plateau_error=0.;tr=pd.read_csv(out/'train-variational.csv');
    for audit in rec['audits']:
        b=audit['index'];beta=audit['beta'];
        with np.load(out/f'parents-b{b:02}.npz') as data:
            Z=data['moments'];assert list(data['names'])==list(NAMES);valid=data['valid'];sp=pd.DataFrame(data['spectral'],columns=data['spectral_fields']);ids=data['ids'];n=len(ids);assert len(np.unique(ids,axis=0))==n==audit['test']
        gap=sp.gap.to_numpy();resolved=np.isfinite(gap);floor=sp.resolution.to_numpy();low=np.where(resolved,gap,0);high=np.where(resolved,gap,floor);gg=dict(index=b,beta=beta,n=n,resolved=resolved.sum(),unresolved=(~resolved).sum(),resolution_min=floor.min(),resolution_max=floor.max(),true_zero_modes=1,max_unresolved_nonstationary=int(sp.unresolved_count.max()),mean_thermal_states=sp.thermal_states.mean(),median_thermal_states=sp.thermal_states.median(),p90_thermal_states=sp.thermal_states.quantile(.9),fraction_single_state_99=(sp.top_probability>=.99).mean(),roundoff_negative_magnitude_max=sp.min_eigenvalue_below_floor.max());
        for z in (.1,.5,.9):gg[f'gap_p{round(100*z)}_lo']=np.quantile(low,z);gg[f'gap_p{round(100*z)}_hi']=np.quantile(high,z)
        for k in ['0001','001','01']:gg['median_band_rank_'+k]=sp['band_rank_'+k].median()
        for k in ['balance_error','stationary_error','norm_error','slope_error','row_error']:gg['max_'+k]=sp[k].max()
        spectra.append(gg);scores={};ok={}
        for k,name in enumerate(NAMES):
            u,v,df,*rest=Z[:,:,k].T;sup,between=support(u,v,valid[k]);base=dict(index=b,beta=beta,name=name,n=n,ntrain=int(tr[tr['index']==b].ntrain.iloc[0]),**sup);w=v+between;r={'slope':ratio(df,v),'plateau':ratio(between,w)}
            base['dirichlet_mean']=df.mean()
            for label,contribution in [('dirichlet',df),('cov100',rest[2])]:
                weight=contribution.sum();base[label+'_ESS']=weight**2/(contribution@contribution) if weight>0 else np.nan
            for j,lag in enumerate(LAGS):r[f'C{lag:.0f}']=ratio(rest[j],v);r[f'raw{lag:.0f}']=ratio(rest[j]+between,w)
            for j,width in enumerate(WIDTH):r[f'power{width:g}']=ratio(rest[4+j],v)
            for key,val in r.items():write(base,key,val)
            if valid[k]:
                assert 0<=base['plateau']<=1+1e-12 and -1e-12<=base['C1000']<=base['C100']+1e-10<=base['C10']+2e-10<=base['C1']+3e-10<=1+4e-10
                for lag in LAGS:assert abs(base[f'raw{lag:.0f}']-(base['plateau']+(1-base['plateau'])*base[f'C{lag:.0f}']))<2e-12
                if name in ('6101','6102','6103','site_linear'):
                    ref=bench[(bench['index']==b)&(bench.observable=='linear')] if name=='site_linear' else old[(old['index']==b)&(old.channel=='hard')&(old.seed==int(name))&old.final];err=abs(base['plateau']-ref['T'].iloc[0]);max_plateau_error=max(err,max_plateau_error);assert err<1e-9
            scores[name]=r;ok[name]=sup['adequate'];summary.append(base)
            for j,lag in enumerate(LAGS):
                key=f'C{lag:.0f}';d=dict(index=b,beta=beta,name=name,n=n,lag=int(lag),adequate=ok[name],valid=valid[k],below_reporting_resolution=bool(valid[k] and r[key][0]<=1e-10));write(d,'connected',r[key]);write(d,'raw',r[f'raw{lag:.0f}']);write(d,'plateau',r['plateau']);detailed.append(d)
        for name in NAMES[:3]:
            for ref in NAMES[3:]:
                for key in ['slope','C1','C10','C100','C1000','power0.001']:
                    d=dict(index=b,beta=beta,name=name,reference=ref,metric=key,n=n,adequate=ok[name] and ok[ref]);write(d,'learned_minus_reference',sub(scores[name][key],scores[ref][key]));paired.append(d)
    for name,rows in [('observable-moments',summary),('finite-lags',detailed),('paired-differences',paired),('spectrum-support',spectra)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False,float_format='%.13g')
    target=[r for r in paired if r['index']==9 and r['reference']=='site_linear' and r['metric']=='C100'];passed=all(r['adequate'] and r['learned_minus_reference_lo']>.05 for r in target);ruled=any(r['adequate'] and r['learned_minus_reference_hi']<=.05 for r in target);state='passes' if passed else 'ruled out' if ruled else 'unresolved';decision=dict(cold_margin=.05,lag=100,unit='attempted internal bond',criterion=state,by_seed=target,test_parents=sum(a['test'] for a in rec['audits']),max_static_plateau_difference_vs_03=max_plateau_error,modal_usd=0,sequence_finished=True);(out/'decision.json').write_text(json.dumps(decision,indent=2,default=lambda x:x.item())+'\n');print(state,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);report(p.parse_args().out)
