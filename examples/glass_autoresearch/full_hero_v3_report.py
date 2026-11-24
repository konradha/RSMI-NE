"""Render actual downloaded H100 training receipts; never loads specimen data."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PUB=Path(__file__).resolve().parents[2]/'results/l20-autoresearch/hero-full-v3'
BETA=[float(v['beta']) for v in json.loads((PUB/'split.json').read_text())['per_beta'].values()]
ARMS=[(a,l) for a in ('affine','tanh') for l in ('uniform','matched')]
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']


def curves(root):
    rows=[]; diagnostics=[]
    for p in sorted(root.glob('learning/*.json')):
        r=json.loads(p.read_text()); dev=r['device']; g=dev['gpu']
        if not (r['schema']=='hero-full-v3-learning-1' and r['source']=='archived-physical' and
                dev['effective_device']=='cuda' and 'H100' in g['name']):
            raise ValueError('Wrong physical/device evidence '+p.name)
        row={k:r[k] for k in ('arch','law','seed','denominator','updates','optimizer_steps','train_wall_sec','cuda_forward_backward_adam_ms','groups_per_wall_second','checkpoint_sha256','optimizer_state_sha256')}
        row.update(fit_nats=r['fit']['macro_common_hard_nats'],train_fold0_nats=r['dev']['macro_common_hard_nats'],enrolled_per_beta=6144//r['denominator'],
                   distinct_across_beta=sum(x['distinct_positive_parents'] for x in r['beta_positive_exposures'].values()),
                   positive_anchor_draws=sum(x['n_positive_slots'] for x in r['beta_positive_exposures'].values()),negative_group_size=8)
        rows.append(row)
        for b,c in r['beta_positive_exposures'].items():
            fr=r['fit']['per_beta'][b]; dr=r['dev']['per_beta'][b]
            d={k:row[k] for k in ('arch','law','seed','denominator','updates','enrolled_per_beta')}
            d.update(index=int(b),beta=BETA[int(b)],selection_parents=1024,drawn_fit_parents=c['distinct_positive_parents'],
                     positive_slots=c['n_positive_slots'],min_parent_exposure=c['min_exposure'],max_parent_exposure=c['max_exposure'],
                     hash_actual_draws=c['within_beta_ordered_draws_sha256'],minimum_class_share=min(dr['uniform']['hard_frequency']),
                     hard_entropy_bits=dr['uniform']['hard_H_bits'],code_information_bits=dr['uniform']['I_V_H_bits'])
            for k in ('uniform','matched'):
                d['fit_'+k+'_hard_nats']=fr[k]['hard_nats'];d['fold0_'+k+'_hard_nats']=dr[k]['hard_nats']
            diagnostics.append(d)
    if not rows: raise ValueError('No actual learning checkpoint available')
    return rows,diagnostics


def csvfile(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def plot(rows,dest):
    fig,axes=plt.subplots(2,3,figsize=(12.5,6.4),sharex='col',sharey='row',layout='constrained')
    for j,den in enumerate((16,4,1)):
        for a in axes[:,j]: a.axhline(0,c='#6f7377',lw=0.8);a.grid(axis='y',alpha=.18);a.set_xscale('log',base=2);a.spines[['top','right']].set_visible(False)
        axes[0,j].set_title(f'{6144//den:,} discovery parents / β');axes[1,j].set_xlabel('Optimizer updates')
        for i,(arch,law) in enumerate(ARMS):
            subset=[r for r in rows if (r['arch'],r['law'],r['denominator'])==(arch,law,den)];starts=sorted(set(r['seed'] for r in subset))
            for row,score in enumerate(('fit_nats','train_fold0_nats')):
                means=defaultdict(list)
                for seed in starts:
                    curve=sorted((r for r in subset if r['seed']==seed),key=lambda r:r['updates'])
                    x=[r['updates'] for r in curve]; y=[1000*r[score] for r in curve];
                    for u,v in zip(x,y): means[u].append(v)
                    axes[row,j].plot(x,y,c=COLORS[i],alpha=.20,lw=.7)
                x=sorted(means);axes[row,j].plot(x,[np.mean(means[k]) for k in x],'.-',lw=1.8,c=COLORS[i],label=f'{arch} · {law}')
    axes[0,0].set_ylabel('Fit gain (millinats)');axes[1,0].set_ylabel('TRAIN fold 0 gain (millinats)'); axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Does optimization buy held-parent prediction?\nEqual-temperature hard retrieval gain versus random guessing',fontsize=14)
    fig.savefig(dest,dpi=160);plt.close(fig)


def plot_held(report,dest):
    fig,ax=plt.subplots(figsize=(9.5,4.6),layout='constrained'); axs=report['hard_retrieval']
    for name,color in [('6101','#0072B2'),('site','#666666'),('coordination','#009E73'),('crystal','#D55E00'),('random_affine','#CC79A7')]:
        vals=[]
        for b in range(10):
            score=axs[str(b)]['models'][name]
            vals.append(1000*score['scores']['common']['hard_nats'])
        ax.plot(BETA,vals,'.-',color=color,label='Frozen learned code' if name=='6101' else name.replace('_',' '))
    ax.axhline(0,c='#666',lw=.8);ax.legend(frameon=False,ncols=2);ax.set(xlabel='Inverse temperature β',ylabel='Hard retrieval gain (millinats)',title='One held-out test: 8,192 parents per temperature');ax.spines[['top','right']].set_visible(False)
    fig.savefig(dest,dpi=160);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--result',type=Path,required=True);ap.add_argument('--out',type=Path,default=PUB);ap.add_argument('--assessment',type=Path)
    a=ap.parse_args();rows,details=curves(a.result);a.out.mkdir(parents=True,exist_ok=True)
    csvfile(a.out/'learning.csv',rows);csvfile(a.out/'learning-by-beta.csv',details);plot(rows,a.out/'learning.png')
    if a.assessment: plot_held(json.loads(a.assessment.read_text()),a.out/'held-temperature.png')
    print(f'{len(rows)} actual CUDA checkpoint receipts; {len(details)} per-beta rows.')
