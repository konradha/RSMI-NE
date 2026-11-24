"""Seal outcome-blind, packing-matched boundary pairs; plot the archive-only decision."""
import argparse
import csv
from pathlib import Path
import json
import numpy as np
from rate import digest,json_out


def seal(a):
    o=a.out;inp=o/'issue-09.csv';src=json.loads((o/'issue-09.json').read_bytes());lock=o/'cold-lock.json';doc=json.loads(lock.read_bytes());meta=json.loads((a.train/'meta.json').read_bytes());receipt=o/'pairing-lock.json';tab=o/'prospective-pairs.csv'
    if receipt.exists() or tab.exists() or digest(inp)!=src['issue_sha256'] or src['lock_sha256']!=digest(lock) or meta['signature']!=doc['signature']:raise ValueError('invalid inputs or existing receipt')
    with inp.open() as f:data=list(csv.DictReader(f))
    ids=np.array([int(r['parent']) for r in data]);p=np.array([float(r['p_tripod']) for r in data]);q=np.array([float(r['comparator_p_tripod']) for r in data]);n=len(ids)
    if n!=1024 or len(set(ids))!=n or len(set((r['chain_seed'],r['chain_id']) for r in data))!=n or not np.isfinite(p+q).all() or min(p.min(),q.min())<=0 or max(p.max(),q.max())>=1:raise ValueError('invalid issue table')
    pairs=[]
    for b,g in enumerate(np.array_split(np.lexsort((ids,q)),8)):
        r=g[np.lexsort((ids[g],(p-q)[g]))];half=len(r)//2
        for lo,hi in zip(r[:half],r[-1:half-1:-1]):pairs.append((b,int(ids[lo]),int(ids[hi]),float(p[lo]),float(p[hi]),float(q[lo]),float(q[hi])))
    with tab.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(('packing_bin','low_residual_parent','high_residual_parent','low_p_tripod','high_p_tripod','low_packing','high_packing'));w.writerows(pairs)
    if len(set(z[1] for z in pairs)|set(z[2] for z in pairs))!=n:raise ValueError('reused a boundary pair')
    counts=meta['eligible_counts_T_tripod'];frac=sum(counts)/(len(meta['parent_ids'])*8000);power=[]
    for k in (16,32,64,128,256,512,1024):power.append(dict(replicas_per_boundary=k,expected_eligible_endpoints=n*k*frac,expected_boundaries_eligible_in_both_starts=n*(-np.expm1(k/2*np.log1p(-frac)))**2))
    qs=(0,.1,.25,.5,.75,.9,1);json_out(receipt,dict(issue_sha256=digest(inp),cold_model_lock_sha256=digest(lock),pair_table_sha256=digest(tab),source_sha256=meta['source_sha256'],cavity_protocol_sha256=digest(o/'cavity-protocol.md'),signature=meta['signature'],parents=n,disjoint_pairs=len(pairs),quantiles=qs,learned_forecast_quantiles=np.quantile(p,qs).tolist(),packing_forecast_quantiles=np.quantile(q,qs).tolist(),fraction_disagree_on_more_likely_arrangement=float(np.mean((p>.5)!=(q>.5))),matched_high_minus_low_prediction=float(np.mean([r[4]-r[3] for r in pairs])),matched_high_minus_low_baseline=float(np.mean([r[6]-r[5] for r in pairs])),train_counts_T_tripod=counts,train_eligible_per_anchor=float(frac),homogeneous_yield_scenarios=power,scenario_limit='No cavity simulations. Scenarios assume independent draws, homogeneous event rate and no equilibration cost. They are planning illustrations, not a power guarantee. Pairing depends only on issued shell forecasts and row IDs.'))
    print('BOUNDARY_ISSUE',len(pairs),'disjoint pairs; forecast quantiles',np.quantile(p,qs),flush=True)


def figure(o):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    allrows=[json.loads((o/f'score-{i:02d}.json').read_bytes()) for i in (0,8,9)];dest=o/'vacancy-selection.png';s=json.loads((o/'pairing-lock.json').read_bytes());
    if dest.exists():raise FileExistsError(dest)
    fig=plt.figure(figsize=(12,4.3),layout='constrained');gs=fig.add_gridspec(2,3,width_ratios=(1,1.6,1.6));off=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]);xy=off@np.array([[.87,-.45],[-.82,-.52],[0,1.]])
    for j,(title,idx) in enumerate([('Vacancy tripod',[0,1,3,5]),('Opposite pair + lateral (T)',[0,1,2,3])]):
        ax=fig.add_subplot(gs[j,0]);occupied=np.ones(7,dtype=bool);occupied[idx]=False
        for end in xy[1:]:ax.plot([0,end[0]],[0,end[1]],color='.65',lw=1,zorder=0)
        for m,fc,ec in [(occupied,'#326178','#326178'),(~occupied,'white','#bc5120')]:ax.scatter(*xy[m].T,s=85,facecolor=fc,edgecolor=ec,linewidths=1.6)
        ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.25),aspect='equal',title=title);ax.axis('off')
    fig.axes[1].legend(handles=[Line2D([],[],marker='o',mfc='white',mec='#bc5120',ls='',label='vacant'),Line2D([],[],marker='o',color='#326178',ls='',label='occupied')],loc='lower center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False,fontsize=8)
    from boundary_models import ratio_interval
    bx=fig.add_subplot(gs[:,1]);y=np.arange(3);templates=[]
    for r in allrows:
        with np.load(o/f"parents-{r['index']:02}.npz",allow_pickle=False) as z:
            templates.append(ratio_interval(z['log_score_sum_templates']-z['log_score_sum_intercept'],z['eligible_weight']))
    for vals,label,color,shift in [(templates,'Crystal templates − prevalence','#326178',-.12),([r['learned_minus_strongest'] for r in allrows],'Learned − selected control','#a54217',.12)]:
        v=np.array([[u['estimate'],u['lo'],u['hi']] for u in vals])*1000;m,l,h=v.T;bx.errorbar(m,y+shift,xerr=[m-l,h-m],fmt='o',c=color,capsize=3,label=label)
    bx.axvline(0,lw=.8,c='.45');bx.axvline(1,lw=.8,c='.5',ls='--');bx.set(yticks=y,yticklabels=['0\ncontrol','2.711\ntransfer','3.05\nprimary'],xlabel='Log-score gain (millinats / matched center)',ylabel='β',title='One archived interior per anchor');bx.legend(frameon=False,fontsize=8)
    cx=fig.add_subplot(gs[:,2]);q=np.array(s['quantiles']);cx.plot(q,np.array(s['learned_forecast_quantiles'])*100,'o-',label='Learned',c='#a54217');cx.plot(q,np.array(s['packing_forecast_quantiles'])*100,'s--',label='Selected control: '+allrows[-1]['strongest_baseline'],c='#326178');cx.set(xlabel='Quantile among issued boundaries',ylabel='P(tripod | either matched shape) (%)',title='1,024 predictions, no cavity outcomes');cx.legend(frameon=False,fontsize=8)
    for ax in (bx,cx):ax.spines[['top','right']].set_visible(False)
    fig.savefig(dest,dpi=165);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['seal','plot']);p.add_argument('--out',type=Path,required=True);p.add_argument('--train',type=Path);a=p.parse_args();seal(a) if a.mode=='seal' else figure(a.out)
