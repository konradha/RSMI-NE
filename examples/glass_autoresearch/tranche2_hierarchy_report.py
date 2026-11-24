"""Visualize all paired training ladders and lock a final checkpoint before audit."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tranche2_hierarchy import OUT, SEEDS, dump, intervals


def curves(r,n):return json.loads((OUT/f'25-curves-{r}-{n}.json').read_text())


def seedmean(c,arm,s):return np.mean([c['seeds'][str(seed)]['arms'][arm][s]['dev_score']['mean'] for seed in SEEDS])


def report():
    primary=curves(4096,4)
    slopes={a:float(seedmean(primary,a,4)-seedmean(primary,a,2)) for a in ('frozen','joint')}
    checkpoint=8 if max(slopes.values())>=.0001 else 4
    dev=np.load(OUT/f'25-dev-parents-4096-{checkpoint}.npz')
    cohorts={r:curves(r,checkpoint if r==4096 else 4) for r in (256,1024,4096)}
    armvec={a:np.mean([dev[f'{a}_{s}_{checkpoint}'] for s in SEEDS],axis=0) for a in ('raw','joint','frozen')}
    armvec['joint_minus_frozen']=armvec['joint']-armvec['frozen']
    stats={a:intervals(v) for a,v in armvec.items()};lo,hi=stats['joint_minus_frozen']['interval95']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for a in ('raw','frozen','joint') for s in SEEDS for p in [OUT/f'25-model-{a}-{s}-{checkpoint}.npz']}
    rawpositive=stats['raw']['interval95'][0]>0
    doc={'design_commit':'1036785','checkpoint':checkpoint,'seeds':list(SEEDS),'rung':4096,
         'extension_trigger_dev_change_2_to_4':slopes,'sha256':hashes,'dev':stats,
         'primary':'mean over three paired seed scores for each of 512 different archive TRAIN parents, then mean over parents; 95% normal parent interval, 80% approximate MDE; native chain keys disjoint.',
         'prediction':'Positive raw score.' if rawpositive else 'Raw advantage not established on development; no depth verdict.',
         'paired_prediction':('Above practical margin.' if lo>.001 else ('Below practical margin.' if hi<.001 else 'Practical-margin contrast unresolved.')),
         'margin_nats_per_face':.001,'audit':'positions 4608:5120 once, fixed hard rules and frozen decoders; no retention of best seeds or checkpoints; held-apart historical TRAIN, not pristine.'}
    aux=np.load(OUT/'25-rawlinear-dev-parents.npz')
    linear=intervals(np.mean([aux[f'r4096_{s}_8'] for s in SEEDS],axis=0))
    doc['diagnostic']={'arm':'post-small-pilot fit/development-only design: D4 pooled direct linear raw readout, 8 fixed Lloyd iterations',
                       'dev':linear,'prediction':'positive' if linear['interval95'][0]>0 else 'unresolved',
                       'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for s in SEEDS for p in [OUT/f'25-model-rawlinear-{s}-8.npz']}}
    dump('25-lock.json',doc)
    fig,axes=plt.subplots(2,3,figsize=(13,7.2),sharex='col')
    colors={'frozen':'#235c97','joint':'#ae4522','raw':'#19765c'}
    for col,(r,c) in enumerate(cohorts.items()):
        m=checkpoint if r==4096 else 4
        for arm in colors:
            for seed in SEEDS:
                obs=c['seeds'][str(seed)]['arms'][arm];x=np.arange(len(obs));color=colors[arm]
                axes[0,col].plot(x,[t['fit_score']*1000 for t in obs],color=color,lw=.9,alpha=.65)
                axes[1,col].plot(x,[t['dev_score']['mean']*1000 for t in obs],color=color,lw=.9,alpha=.65)
            for row,stat in ((0,'fit_score'),(1,'dev_score')):
                mean=[]
                for i in range(m+1):
                    a=[c['seeds'][str(s)]['arms'][arm][i][stat] for s in SEEDS]
                    mean.append(np.mean([z['mean'] if isinstance(z,dict) else z for z in a])*1000)
                axes[row,col].plot(range(m+1),mean,color=colors[arm],lw=2.1,label=arm)
        axes[0,col].set_title(f'{r:,} fit parents');axes[1,col].set_xlabel('Full fit sweeps');axes[1,col].axhline(0,color='0.5',lw=.7)
    axes[0,0].set_ylabel('Fit gain (millinats / face)');axes[1,0].set_ylabel('Development gain (millinats / face)');axes[0,2].legend(frameon=False)
    fig.suptitle('Cold archive • 64 microscopic sites → four hard states • three paired initializations');fig.tight_layout();fig.savefig(OUT/'25-learning.png',dpi=135);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for i,(r,c) in enumerate(cohorts.items()):
        m=checkpoint if r==4096 else 4;v=np.load(OUT/f'25-dev-parents-{r}-{m}.npz')
        for j,s in enumerate(SEEDS):
            stat=intervals(v[f'joint_{s}_{m}']-v[f'frozen_{s}_{m}']);ax.errorbar(i+(j-1)*.12,1000*stat['mean'],yerr=1960*stat['se'],fmt='o',color=['#1d447a','#a45535','#147666'][j],label=f'seed {s}' if i==0 else None)
    ax.axhline(1,color='0.4',ls=':',label='Practical gain');ax.axhline(0,color='0.4',lw=1);ax.set_xticks(range(3),list(cohorts));ax.set_ylabel('Joint − frozen (millinats / face)');ax.set_xlabel('Fit parents (independent development parent intervals)');ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(OUT/'25-paired.png',dpi=135);plt.close(fig)
    print(json.dumps(doc,indent=2))


if __name__=='__main__':report()
