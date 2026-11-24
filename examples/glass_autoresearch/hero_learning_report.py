"""Materialize complete HERO learning ladder and readable depth/data diagnostics from committed receipts."""
import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parents[2]/'results/l20-autoresearch/hero-central'
ARMS=[(a,l) for l in ('uniform','matched') for a in ('affine','tanh')]
LABELS=['Unconditional · affine','Unconditional · nonlinear','Conditional · affine','Conditional · nonlinear']
COLORS=['#3370a8','#6a389c','#248575','#bd651c']


def readj(p): return json.loads(p.read_bytes())


def run(stage):
    readj(P/'screen-receipt.json')
    if stage=='final': readj(P/'fixed-control-receipt.json')
    seeds=[6101,6102,6103]; records=[]; source=[]
    for f in sorted((P/'learning').glob('*.json')):
        v=readj(f);a,l,n,u,s=[v[k] for k in ('arm','law','parents','iteration','seed')]
        if stage=='screen' and ((a,l) not in ARMS or u>8192 or s not in seeds):continue
        source.append(f)
        for split in ('fit','tune'):
            e=v[split]; records.append(dict(arm=a,objective=l,parents=n,updates=u,seed=s,split=split,common_hard_nat=e['common_hard_nats'],
                unconditional_hard_nat=e['uniform']['hard_nats'],conditional_hard_nat=e['matched']['hard_nats'],unconditional_expected_nat=e['uniform']['expected_nats'],conditional_expected_nat=e['matched']['expected_nats'],
                hard_entropy_bits=e['hard_H_bits'],marginal_H_bits=e['H_bits'],conditional_H_given_V_bits=e['H_given_V_bits'],encoder_I_bits=e['I_V_H_bits'],
                minimum_hard_class_share=min(e['hard_frequency']),stage_train_cpu_s=v['train_cpu_sec'],stage_total_cpu_s=v['stage_cpu_sec'],stage_wall_s=v['stage_wall_sec']))
    if len(records)<360:raise ValueError('screen not complete')
    with (P/(stage+'-learning.csv')).open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    if stage=='final':
        with ZipFile(P/'learning-receipts.zip','x',compression=ZIP_DEFLATED,compresslevel=6) as z:
            for f in source:z.write(f,'learning/'+f.name)
    fig,axs=plt.subplots(1,3,figsize=(13.2,4.2),layout='constrained',sharex=False)
    for (a,l),lab,col in zip(ARMS,LABELS,COLORS):
        base=[r for r in records if (r['arm'],r['objective'])==(a,l)]
        for ax,dim,eligible in [(axs[0],'parents',[r for r in base if r['updates']==8192]),(axs[1],'updates',[r for r in base if r['parents']==4096])]:
            pts=sorted({r[dim] for r in eligible}); vals=[np.array([r['common_hard_nat'] for r in eligible if r[dim]==p and r['split']=='tune' and r['seed'] in seeds])*1000 for p in pts];ax.plot(pts,[np.median(v) for v in vals],'-o',ms=4,color=col,label=lab);ax.fill_between(pts,[v.min() for v in vals],[v.max() for v in vals],color=col,alpha=.10);ax.set_xscale('log',base=2)
        d=[r for r in base if r['parents']==4096];pts=sorted({r['updates'] for r in d});g=[]
        for p in pts:
            paired=[tuple(r['common_hard_nat'] for split in ('fit','tune') for r in d if r['seed']==s and r['split']==split and r['updates']==p) for s in seeds];g.append(np.median([f-t for f,t in paired])*1000)
        axs[2].plot(pts,g,'-o',ms=4,color=col);axs[2].set_xscale('log',base=2)
    axs[0].set(title='More independent parents',xlabel='Fit parents · 8,192 updates',ylabel='Hard development score (millinats)');axs[1].set(title='More updates at fixed data',xlabel='Updates · 4,096 fit parents');axs[2].set(title='Train − development',xlabel='Updates · 4,096 fit parents',ylabel='Hard-score difference (millinats)')
    for ax in axs:
        ax.axhline(0,c='.6',lw=.7);ax.grid(axis='y',alpha=.2);ax.tick_params(which='minor',bottom=False)
    axs[0].legend(fontsize=8,frameon=False);fig.suptitle('Forward central code at β = 2.7111 · same held-apart hard score for all arms',fontsize=14,fontweight='bold');fig.savefig(P/(stage+'-learning.png'),dpi=165);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('screen','final'));run(p.parse_args().stage)
