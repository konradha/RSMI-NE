"""Parent-clustered physical contrasts. Ages never enter uncertainty as independent draws."""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from opening_history_match import MODES
from opening_history_stage import OUT, native, save_json


MEASURES = ('all_sites', 'marginal_low', 'marginal_high', 'matched_low_risk', 'matched_high_risk')


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


def finite(x):
    return None if not np.isfinite(x) else float(x)


def metrics(parent_delta, nparent, matched_fraction):
    d = parent_delta[np.isfinite(parent_delta)]
    ci = [None,None]; width = mean = None
    if len(d):
        sample = np.random.default_rng(531724).choice(d, size=(10000,len(d)))
        ci = np.percentile(sample.mean(1), [2.5,97.5]).tolist()
        mean, width = float(d.mean()), ci[1]-ci[0]
    sufficient = len(d) > .8*nparent and matched_fraction >= .05
    precise = width is not None and width <= .04
    meaningful = ci[0] is not None and ci[0] > .02
    if not sufficient: label='inadequate support'
    elif not precise: label='inadequate precision'
    elif meaningful: label='meaningful positive contrast'
    elif ci[1] <= .02: label='interval excludes meaningful positive contrast'
    else: label='interval straddles meaningful margin'
    return dict(all_parents=nparent, paired_supported=len(d), coverage=matched_fraction,
                high_minus_low=mean, ci95=ci, ci_width=width, support_pass=sufficient,
                precision_pass=precise, margin_pass=meaningful, pass_all=sufficient and precise and meaningful,
                interpretation=label)


def write_phase(phase, frames, grids, returned, lock):
    primary=10**6
    primary_rows=[]; parent_rows=[]; curve_rows=[]; results={}; parent_arrays={}
    for schedule,t in grids.items():
        j=int(np.flatnonzero(t==primary)[0]); parent_arrays[schedule+'_lags']=t
        for mode in MODES:
            block=sorted(((r,c) for r,c in frames if r['mode']==mode and r['schedule']==schedule),
                         key=lambda f:(f[0]['rank'],f[0]['age']))
            rr=[r for r,c in block]; ranks=np.array([r['rank'] for r in rr])[::2]
            p=len(ranks); expected=2*p
            if len(rr)!=expected or any(rr[2*i]['parent_id']!=rr[2*i+1]['parent_id'] or
                    [rr[2*i]['age'],rr[2*i+1]['age']] != [300000000,500000000] for i in range(p)):
                raise ValueError('not two paired ages')
            curves=np.array([[c[k] for k in MEASURES] for _,c in block]).reshape(p,2,len(MEASURES),len(t))
            means=curves.mean(1); delta=means[:,4,:]-means[:,3,:]; val=delta[:,j]
            parent_arrays[schedule+'_'+mode]=curves
            parent_arrays['parent_id']=np.array([r['parent_id'] for r in rr[::2]])
            parent_arrays['rank']=ranks
            masks={'all':np.ones(p,bool)}
            if phase in ('development','transfer'):
                for side in ('fit','dev','test'):
                    mask=np.array([r['side']==side for r in rr[::2]])
                    if mask.any(): masks[side]=mask
            for label,mask in masks.items():
                som=sum(r['matched_low']+r['matched_high'] for i,r in enumerate(rr) if mask[i//2] and np.isfinite(val[i//2]))
                den=sum(r['occupied'] for i,r in enumerate(rr) if mask[i//2]); key=f'{schedule}/{mode}/{label}'
                results[key]=metrics(val[mask],int(mask.sum()),som/den)
                for k,measure in enumerate(MEASURES):
                    ok=mask&np.isfinite(means[:,k,0]); mean=means[ok,k].mean(0) if ok.any() else np.full(len(t),np.nan)
                    for lag,v in zip(t,mean):
                        curve_rows.append(dict(schedule=schedule,mode=mode,roster=label,measure=measure,lag=int(lag),
                                               parents=int(ok.sum()),risk=finite(v)))
            for i in range(p):
                r=rr[2*i]
                per_age=curves[i,:,4,j]-curves[i,:,3,j]
                parent_rows.append(dict(parent_id=r['parent_id'],rank=r['rank'],side=r['side'],beta=r['beta'],
                        schedule=schedule,mode=mode,risk_difference_300M=finite(per_age[0]),
                        risk_difference_500M=finite(per_age[1]),paired_risk_difference=finite(val[i])))
            for i,(r,c) in enumerate(block):
                primary_rows.append(dict(**r,observed_returned_full_grid=returned[str(r['child_id'])],
                                         **{name:finite(c[name][j]) for name in MEASURES}))
    write_csv(OUT/f'{phase}-forks.csv',primary_rows)
    write_csv(OUT/f'{phase}-parents.csv',parent_rows)
    write_csv(OUT/f'{phase}-curves.csv',curve_rows)
    np.savez_compressed(OUT/f'{phase}-parent-curves.npz', **parent_arrays, measures=np.array(MEASURES),
                        ages=np.array([300000000,500000000]))
    stats=dict(protocol=lock['protocol'], cuts=lock['cuts_sha256'],phase=phase,primary_lag=primary,
        parent_curves_layout='for each schedule_mode: [parent ordered by rank, age 300M/500M, measure, lag]',
        results=results,primary_gate=all(results['full/'+mode+'/all']['pass_all'] for mode in MODES[:2]),
        observed_returned_sites_full_grid=sum(returned.values()),
        first_observed_by_terminal=sum(r['occupied']-r['censored'] for r in primary_rows if r['schedule']=='full' and r['mode']=='opening'),
        right_censored_through_terminal=sum(r['censored'] for r in primary_rows if r['schedule']=='full' and r['mode']=='opening'),
        max_within_stratum_vh_range=max(r['max_h_residual'] for r in primary_rows),
        note='Observed checkpoint mismatch only; sampled return never erases an earlier event. Same-parent ages paired. The aggregate site coverage includes only pairs with both ages supported.')
    save_json(OUT/f'{phase}-summary.json',stats)
    plot(phase,grids['full'],parent_arrays)
    if phase=='transfer' and (OUT/'held-source-summary.json').exists():
        source=json.loads((OUT/'held-source-summary.json').read_bytes())
        save_json(OUT/'transfer-decision.json',dict(pass_physical=stats['primary_gate'] and source['primary_gate'],
                  interval='parent percentile bootstrap, two-sided 95%',margin=.02,maximum_width=.04,
                  held_source={mode:source['results']['full/'+mode+'/all'] for mode in MODES},
                  true_transfer={mode:stats['results']['full/'+mode+'/all'] for mode in MODES},
                  unchanged_cuts_sha256=lock['cuts_sha256']))


def plot(phase,t,data):
    j=np.flatnonzero(t==10**6)[0]; fig,axes=plt.subplots(2,3,figsize=(13,6.5),constrained_layout=True)
    for col,mode in enumerate(MODES):
        arr=data['full_'+mode]; pm=arr.mean(axis=1); ax=axes[0,col]
        for k,style,color,name in [(0,':','gray','all occupied'),(1,'--','C0','low marginal'),(2,'--','C1','high marginal'),
                                    (3,'-','C0','low matched'),(4,'-','C1','high matched')]:
            ok=np.isfinite(pm[:,k,0]);
            if ok.any():ax.plot(t[1:],pm[ok,k,1:].mean(0),style,color=color,label=name)
        ax.set(xscale='log',title=mode,xlabel='Observed lag (sweeps)',ylabel='First observed mismatch risk',ylim=(-.02,1.02));ax.axvline(1e6,color='gray',lw=.6)
        if col==0:ax.legend(fontsize=7)
        ax=axes[1,col]; d=arr[:,:,4,j]-arr[:,:,3,j]; x=data['rank']; ax.vlines(x,d[:,0],d[:,1],alpha=.23,lw=.7)
        for age,c,name in [(0,'C2','300M'),(1,'C4','500M')]:ax.scatter(x,d[:,age],c=c,label=name,s=6,alpha=.6)
        ax.scatter(x,d.mean(1),c='k',s=5,label='paired');ax.axhline(.02,color='C3',ls=':',label='+2 pp');ax.axhline(0,color='gray',lw=.5)
        ax.set(xlabel='Independent parent (frozen rank)',ylabel='High − low risk at 1M');
        if col==0:ax.legend(fontsize=7)
    fig.suptitle(f'{phase}: native fork site histories (observations, not continuous persistence)')
    fig.savefig(OUT/f'{phase}.png',dpi=150);plt.close(fig)
