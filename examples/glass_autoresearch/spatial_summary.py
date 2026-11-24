"""Publish registered reverse E→H held-apart development contrasts only."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P=Path(__file__).resolve().parents[2]/'results/l20-autoresearch/spatial-cycle'


def run():
    raw=json.loads((P/'space-dev.json').read_bytes()); phys=json.loads((P/'probe-dev.json').read_bytes())
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.5),layout='constrained'); scale=1000
    with (P/'learning.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['fit_parents','updates','metric (distinct endpoints)','estimate_nat','lower95_nat','upper95_nat','independent_dev_parents','independent_score_groups'])
        for n,color,offset in ((512,'#ca6e25',-.04),(2048,'#1667a5',.04)):
            ys=[];err=[[],[]]
            for u in (256,512,1024):
                q=raw['ladder'][str(n)]['checkpoints'][str(u)]['ensemble_mean_of_paired_seed_scores']['hard_difference']; p=q['estimate']; l,h=q['ci95']; ys.append(scale*p);err[0].append(scale*(p-l));err[1].append(scale*(h-p));w.writerow([n,u,'hard true-minus-broken 8-way retrieval',p,l,h,512,16]);r=phys['checkpoints'][f'n{n}u{u}']['paired']
                for arm in ('native','targeted','occupancy','random'):
                    v=r[arm+'-packing_shift'];w.writerow([n,u,arm+' minus calibrated packing Bernoulli log score',v['point'],v['lo'],v['hi'],512,512])
            a.errorbar(np.arange(3)+offset,ys,yerr=err,fmt='o-',color=color,capsize=3,label=f'{n:,} fit parents')
    a.set(xticks=range(3),xticklabels=['256','512','1,024'],xlabel='Optimizer updates',ylabel='True − mismatched shell/word score (millinats)',title='Can the shell code identify its central word?');a.axhline(0,c='.4',lw=.9);a.legend(frameon=False);a.grid(axis='y',alpha=.18)
    r=phys['checkpoints']['n2048u1024']['paired']; names=[('native','Label-free code pair'),('native_260511','  Label-free seed 1'),('native_260512','  Label-free seed 2'),('targeted','Supervised (collapsed)'),('random','Frozen random pair'),('occupancy','Shell vacancy quartiles')]
    for i,(arm,_) in enumerate(names):
        t=r[arm+'-packing_shift'];y=len(names)-1-i;x=t['point']*scale;b.errorbar(x,y,xerr=[[x-scale*t['lo']],[scale*t['hi']-x]],fmt='o',capsize=3,color='#1667a5' if arm.startswith('native') else '#555555')
    b.axvline(0,c='.4',lw=.9);b.axvline(1,c='#bb4c29',ls='--',label='Registered useful gain');b.set(yticks=list(reversed(range(len(names)))),yticklabels=[s for _,s in names],xlabel='Gain over calibrated packing (millinats / eligible anchor)',title='Does it predict the unfilled cycle?');b.legend(frameon=False,loc='lower right');b.grid(axis='x',alpha=.18)
    fig.suptitle('Reverse spatial code at β = 3.05  •  development only',fontsize=15,fontweight='bold');fig.savefig(P/'development.png',dpi=170);plt.close(fig)


if __name__=='__main__':run()
