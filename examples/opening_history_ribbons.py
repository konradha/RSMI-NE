"""Plot the already-published paired-parent risk curves, without refitting or selecting lags."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]/'results/opening-mechanism/history'


def interval(values):
    n,t=values.shape; samples=np.empty((10000,t)); rng=np.random.default_rng(531724)
    for i in range(0,10000,100):
        samples[i:i+100]=values[rng.choice(n,size=(100,n))].mean(1)
    return np.percentile(samples,[2.5,97.5],axis=0)


def main():
    fig,axes=plt.subplots(1,2,sharey=True,figsize=(10.8,4.8));fig.subplots_adjust(left=.078,right=.98,top=.76,bottom=.20,wspace=.12)
    fig.text(.078,.935,'Does opening predict the first observed site change?',fontsize=17,weight='bold');fig.text(.078,.86,'Same initial species, coordination and six signed exchange barriers  ·  source thresholds fixed before transfer',fontsize=10)
    rows=[]; decision=json.loads((OUT/'transfer-decision.json').read_text())
    for phase,label,side,ax in zip(('held-source','transfer'),('β = 3.20  ·  held source','β = 3.40  ·  unchanged transfer'),('held_source','true_transfer'),axes):
        with np.load(OUT/f'{phase}-parent-curves.npz') as z:
            lags=z['full_lags'];measures=z['measures'].tolist();lo,hi=map(measures.index,['matched_low_risk','matched_high_risk']);n=len(z['parent_id'])
            for mode,color,legend in [('opening+packing','#a24126','Opening, also matched on packing'),('packing','#32658c','Vacancy-count comparator')]:
                x=z['full_'+mode];delta=(x[:,:,hi,:].mean(1)-x[:,:,lo,:].mean(1))*100
                if not np.isfinite(delta).all():raise ValueError('unreported missing parent')
                mean=delta.mean(0);low,high=interval(delta);j=np.flatnonzero(lags==1000000).item()
                np.testing.assert_allclose([mean[j],low[j],high[j]],np.r_[decision[side][mode]['high_minus_low'],decision[side][mode]['ci95']]*100,rtol=0,atol=1e-12)
                ax.fill_between(lags,low,high,color=color,alpha=.21,linewidth=0);ax.plot(lags,mean,color=color,label=legend,lw=2);ax.scatter([10**6,10**9],mean[[j,-1]],s=16,color=color,zorder=5)
                rows.extend((phase,mode,int(t),n,a,b,c) for t,a,b,c in zip(lags,mean,low,high))
        ax.axvline(1e6,color='.4',ls=':',linewidth=1);ax.axhline(0,color='.47',lw=.65);ax.scatter([1e6],[2],marker='+',color='#222222',zorder=6,s=65,label='Locked 1M margin (+2 pp)'); ax.set_xscale('symlog',linthresh=1);ax.set(xlim=(0,1.5e9),xticks=[0,1e2,1e4,1e6,1e9],xticklabels=['0','100','10k','1M','1B'],xlabel='Recorded physical lag (local sweeps)');ax.set_title(label+f'  ·  {n} parents',loc='left',fontsize=10,weight='bold');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('High − low first-observed risk (percentage points)');axes[0].legend(fontsize=8,frameon=False,loc='upper left');fig.text(.078,.045,'Shading: pointwise 95% parent bootstrap, keeping both fork ages paired. Only the 1M lag is a decision endpoint.',fontsize=9)
    fig.savefig(OUT/'contrast-over-time.png',dpi=175);plt.close(fig)
    with (OUT/'contrast-over-time.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['cohort','mode','lag','paired_parents','mean_pp','ci95_low_pp','ci95_high_pp']);w.writerows(rows)
    for row in rows:
        if row[2]==10**9:print(row)


if __name__=='__main__':main()
