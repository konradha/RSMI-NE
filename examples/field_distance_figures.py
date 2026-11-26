"""Render directly from the registered independent-parent estimates; no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from field_distance_inputs import OUT

HELD=('source-test','transfer-320','transfer-340','beta-zero')
LABELS=('β=3.05 · source TEST','β=3.20 · transfer','β=3.40 · transfer','β=0 · control')
SEEDS=(6101,6102,6103);COL=('#24649b','#b05d32','#448461')

def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.axhline(0,color='.55',lw=.65);ax.grid(axis='y',alpha=.16)

def main():
    d=pd.read_csv(OUT/'distance-curves.csv.gz',dtype={'d':str});d=d[d.d!='pool'].copy();d['distance']=d.d.astype(int);p=pd.read_csv(OUT/'primary.csv');m=json.loads((OUT/'mapping-receipt.json').read_text())['seeds'];
    for mode,plot in [('total','total-distance'),('enriched_residual','residual-distance')]:
        fig,axes=plt.subplots(2,4,figsize=(13.3,6.6),sharex=True,sharey='row');fig.subplots_adjust(left=.08,right=.99,top=.78,bottom=.18,wspace=.13,hspace=.25)
        title='Total frozen RSMI field and ordinary packing' if mode=='total' else 'After removing ordinary directional packing'
        fig.text(.08,.94,title,fontsize=18,weight='bold');fig.text(.08,.87,'Signed two-star covariance  ·  in units of total source TRAIN variance per component  ·  shading: pointwise 95% parent intervals',fontsize=10)
        for j,(tag,label) in enumerate(zip(HELD,LABELS)):
            for i,pol in enumerate(('L','T')):
                ax=axes[i,j];clean(ax);ax.axvspan(4.85,6.15,color='#dedad4',alpha=.48,zorder=-2)
                sub=d[(d.cohort==tag)&(d.polarization==pol)&(d.channel=='observed')];
                for seed,color in zip(SEEDS,COL):
                    z=sub[sub.model==f'{mode}_s{seed}'].sort_values('distance');ax.fill_between(z.distance.to_numpy(),z.low.to_numpy(),z.high.to_numpy(),color=color,alpha=.13,lw=0);ax.plot(z.distance,z['mean'],color=color,marker='o',ms=2.6,lw=1.3,label=str(seed))
                if mode=='total':
                    z=sub[sub.model=='packing_enriched'].sort_values('distance');ax.plot(z.distance,z['mean'],color='.15',ls='--',lw=1.3,label='Ordinary packing')
                if i==0:ax.set_title(label,loc='left',fontsize=10,weight='bold')
                if j==0:ax.set_ylabel(('Longitudinal L' if i==0 else 'Transverse T')+'\nnormalized covariance')
                ax.set_xticks(range(3,10));ax.tick_params(labelsize=8)
        handles,leg=axes[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.51,.004));fig.text(.51,.10,'Periodic axial center separation d (lattice sites); shaded interval = registered d=5,6',ha='center',fontsize=10);fig.savefig(OUT/f'{plot}.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,5.3),sharey=True);fig.subplots_adjust(left=.155,right=.98,top=.77,bottom=.16,wspace=.12)
    fig.text(.155,.945,'Did an independent residual propagate across the gap?',fontsize=17,weight='bold');fig.text(.155,.875,'Pooled d=5,6  ·  simultaneous 95% parent intervals (72 comparisons)  ·  shaded area: ±0.01 of total source TRAIN variance',fontsize=10)
    ys=np.arange(8)[::-1];yt=[lab.split(' · ')[0]+'   '+pol for lab in LABELS for pol in ('L','T')]
    for ax,contrast,title in zip(axes,('R','R_minus_shuffle','R_minus_shuffle_minus_random_corrected'),('Residual','Minus shuffled sites','Also minus corrected random controls')):
        ax.axvspan(-.01,.01,color='#dce6df',alpha=.75);ax.axvline(0,color='.45',lw=.8);ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_title(title,loc='left',fontsize=10,weight='bold')
        for i,(s,c) in enumerate(zip(SEEDS,COL)):
            for j,tag in enumerate(HELD):
                z=p[(p.cohort==tag)&(p.seed==s)&(p.contrast==contrast)].set_index('polarization').loc[['L','T']];y=ys[2*j:2*j+2]+(i-1)*.19;ax.errorbar(z['mean'],y,xerr=z.halfwidth,fmt='o',color=c,markersize=3,lw=1,capsize=2,label=str(s) if j==0 else None)
        ax.set_yticks(ys,yt);ax.tick_params(axis='y',length=0);ax.set_xlabel('Signed covariance');ax.set_xlim(min(-.011,ax.get_xlim()[0]),max(.011,ax.get_xlim()[1]))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.savefig(OUT/'primary-intervals.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.8,4.4),layout='constrained');x=np.arange(3);ax=axes[0];ax.bar(x-.17,[100*m[str(s)]['basic_residual_fraction'] for s in SEEDS],.34,color='#7499b8',label='After arm differences');ax.bar(x+.17,[100*m[str(s)]['enriched_residual_fraction'] for s in SEEDS],.34,color='#3a7254',label='After enriched packing');ax.axhline(1,color='.35',ls=':',label='1% support floor');ax.set(xticks=x,xticklabels=SEEDS,ylabel='Percent of original component variance',title='What is left of the frozen vector field?');ax.legend(fontsize=8,frameon=False)
    ax=axes[1];r=pd.read_csv(OUT/'random-matching.csv');
    for j,s in enumerate(SEEDS):
        k=m[str(s)]; ax.plot([j,j],[k['power_T1u']+k['selected_mean_T1u_power_minus_true'],k['power_T1u']],color='.6',zorder=0);ax.scatter(j,k['power_T1u'],color=COL[j],s=45,label='Frozen encoder' if j==0 else None);ax.scatter(j,k['power_T1u']+k['selected_mean_T1u_power_minus_true'],marker='x',color='.15',s=45,label='Mean of 8 TRAIN-matched shuffles' if j==0 else None)
    ax.set(xticks=x,xticklabels=SEEDS,ylabel='Fraction of category span in T1u',title='Do random partitions match its symmetry?');ax.legend(fontsize=8,frameon=False);[a.spines[['top','right']].set_visible(False) for a in axes];fig.savefig(OUT/'train-specificity.png',dpi=175);plt.close(fig)

if __name__=='__main__':main()
