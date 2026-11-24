"""Plot the local-propensity experiment from parent-level scores, without refitting."""
import argparse
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
from scipy.stats import t as student

C={'species':'#767d83','local':'#317f99','geometry':'#699472','barrier':'#d05b26','rsmi':'#624285','random':'#aa8b33'}
KEY=['cohort','index','beta','lag','role','replicas']


def estimate(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)];n=len(x)
    if not n:return np.nan,np.nan,np.nan
    m=x.mean();off=student.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
    return m,m-off,m+off


def summary(data):
    rows=[]
    for k,g in data.groupby(KEY):
        b=g.pivot(index='parent_id',columns='model',values='brier')
        def add(metric,model,values):
            v,lo,hi=estimate(values); rows.append(dict(zip(KEY,k))|dict(metric=metric,model=model,parents=len(b),mean=v,lower=lo,upper=hi))
        for model in b:
            if b[model].isna().any():raise ValueError(f'missing scores for {model}: {k}')
            add('brier',model,b[model]);add('gain_vs_local',model,b.local-b[model]);add('gain_vs_barrier',model,b.barrier-b[model]);add('gain_vs_geometry',model,b.geometry-b[model]);add('gain_vs_random',model,b.random-b[model]);add('gain_vs_species',model,b.species-b[model]);add('correlation',model,g.loc[g.model==model,'correlation'])
        add('mobility','observed',g[g.model=='local'].mobility)
        add('split_half_correlation','observed',g[g.model=='local'].split_half_correlation)
    return pd.DataFrame(rows)


def cold_scale(out,data):
    """Express Brier differences in units of reproducible within-parent spatial variance."""
    with zipfile.ZipFile(out/'inputs.zip') as bundle, bundle.open('cold-replicated.npz') as member, np.load(member,allow_pickle=False) as f:
        ids=f['parent_id'];s=f['origin'];r=int(f['replicas']);lags=f['lags'];half=f['half_changed'];full=f['changed']
    rows=[]
    for j,t in enumerate(lags):
        g=data[(data.cohort=='cold-replicated')&(data.lag==t)]
        b=g.pivot(index='parent_id',columns='model',values='brier').sort_index()
        cov=[]
        for pid in b.index:
            i=np.flatnonzero(ids==pid).item();keep=s[i]!=0
            a=half[j,i,keep].astype(float)/(r//2);c=(full[j,i,keep]-half[j,i,keep]).astype(float)/(r-r//2)
            cov.append(np.mean((a-a.mean())*(c-c.mean())))
        cov=np.asarray(cov);ix=np.random.default_rng(99147).integers(0,len(b),(10000,len(b)))
        for ref,model in [('species','local'),('local','barrier'),('barrier','rsmi'),('random','rsmi')]:
            diff=(b[ref]-b[model]).to_numpy();draws=100*diff[ix].mean(1)/cov[ix].mean(1);lo,hi=np.quantile(draws,[.025,.975])
            rows.append(dict(lag=t,parents=len(b),reference=ref,model=model,split_covariance=cov.mean(),brier_gain=diff.mean(),percent=100*diff.mean()/cov.mean(),lower_percent=lo,upper_percent=hi))
    pd.DataFrame(rows).to_csv(out/'cold-scale.csv',index=False)


def proximity_summary(data):
    rows=[]; keys=['cohort','replicas','lag','map','distance'];
    for key,g in data.groupby(keys):
        for col in ('near_risk','far_matched_risk','difference'):
            v,lo,hi=estimate(g[col]);rows.append(dict(zip(keys,key))|dict(metric=col,parents=len(g),coverage=g.near_matched.sum()/g.near_all.sum(),mean=v,lower=lo,upper=hi))
        a,b=g.near_risk.to_numpy(),g.far_matched_risk.to_numpy()
        ix=np.random.default_rng(88223).integers(0,len(g),(10000,len(g)))
        lo,hi=np.quantile(a[ix].mean(1)/b[ix].mean(1),[.025,.975])
        rows.append(dict(zip(keys,key))|dict(metric='near_far_ratio',parents=len(g),coverage=g.near_matched.sum()/g.near_all.sum(),mean=a.mean()/b.mean(),lower=lo,upper=hi))
    return pd.DataFrame(rows)


def paint(ax,g,x,color,marker,label,scale=1,ls='-'):
    g=g.sort_values(x);y=g['mean'].to_numpy()*scale
    ax.errorbar(g[x],y,yerr=np.maximum(0,np.vstack((y-g.lower.to_numpy()*scale,g.upper.to_numpy()*scale-y))),color=color,marker=marker,ms=4.5,lw=1.5,capsize=2.3,ls=ls,label=label)


def finish(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=165,facecolor='white');plt.close(fig)


def performance(s,out):
    fig,ax=plt.subplots(2,2,figsize=(12,8.3),layout='constrained');cold=s[s.cohort=='cold-replicated'];arch=s[(s.cohort!='cold-replicated')&s.role.str.contains('alpha')]
    for model in ('local','barrier','rsmi'):
        z=cold[(cold.metric=='correlation')&(cold.model==model)];paint(ax[0,0],z,'lag',C[model],{'local':'s','barrier':'o','rsmi':'x'}[model],{'local':'Species + coordination','barrier':'Composition + swap barrier¹','rsmi':'Same + frozen RSMI'}[model])
    ax[0,0].set(xscale='log',xlabel='Local sweeps',ylabel='Within-parent Pearson r',title='A  |  Predicting the replicated cold maps');ax[0,0].legend()
    order=['geometry','local_rsmi','local_random','barrier','rsmi','random'];d=cold[(cold.lag==4096)&(cold.metric=='gain_vs_local')].set_index('model')
    for i,m in enumerate(order):
        row=d.loc[m];x,lo,hi=row[['mean','lower','upper']]*1000;ax[0,1].errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=C[m.removeprefix('local_')],capsize=3,ms=6)
    ax[0,1].axvline(0,color='0.5',lw=.8);ax[0,1].set(yticks=range(len(order)),yticklabels=['Cube counts','Frozen RSMI','Random filters','Counts + barrier','Counts + barrier + RSMI','Counts + barrier + random'],xlabel='Brier improvement vs species + coordination (×10⁻³)',title='B  |  β = 3.05 · 4,096 sweeps');ax[0,1].invert_yaxis()
    for m,mk in [('barrier','o'),('rsmi','x'),('geometry','s')]:
        z=arch[(arch.metric=='gain_vs_local')&(arch.model==m)];paint(ax[1,0],z,'beta',C[m],mk,{'geometry':'Cube composition','barrier':'With swap barrier','rsmi':'With frozen RSMI'}[m],1000)
    for m,mk in [('random','s'),('rsmi','o')]:
        z=arch[(arch.metric=='gain_vs_barrier')&(arch.model==m)];paint(ax[1,1],z,'beta',C[m],mk,{'random':'Random filters','rsmi':'Frozen RSMI'}[m],1000)
    for a in ax[1]:
        a.axvspan(2.55,3.15,color='0.92',zorder=-10);a.axhline(0,color='0.4',lw=.75);a.set(xlabel='Inverse temperature β',xlim=(-.1,3.15));a.legend()
    ax[1,0].set(ylabel='Brier gain vs local (×10⁻³)',title='C  |  Archived local paths near relaxation');ax[1,1].set(ylabel='Extra Brier gain vs barrier model (×10⁻³)',title='D  |  Do the learned filters add information?')
    fig.suptitle('From a static configuration to physical site mobility',size=17,fontweight='bold');fig.supxlabel('¹ Includes species and coordination. Intervals: 95% across whole held-out parents, conditional on the fit.\nShaded β range: one archived trajectory per parent; event prediction, not a measured propensity map.',fontsize=9,color='#44515b')
    finish(fig,out,'propensity')


def defects(p,out):
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained');g=p[(p.distance=='1-2')&(p['map']=='aligned')]
    for cohort,ls,mark in [('cold-replicated','-','o'),('archive-09','--','D')]:
        z=g[g.cohort==cohort]
        for metric,col,name in [('near_risk','#ce5c29','1–2 steps from a free site'),('far_matched_risk','#286e88','≥5 steps, matched')]:
            paint(ax[0],z[z.metric==metric],'lag',col,mark,name if ls=='-' else None,100,ls)
    for distance,col in [('1-2','#ce5c29'),('3-4','#7972ad')]:
        for cohort,ls,mark in [('cold-replicated','-','o'),('archive-09','--','D')]:
            z=p[(p.distance==distance)&(p['map']=='aligned')&(p.cohort==cohort)&(p.metric=='near_far_ratio')];paint(ax[1],z,'lag',col,mark,'Distance '+distance if ls=='-' else None,1,ls)
    for cohort,ls,mark in [('cold-replicated','-','s'),('archive-09','--','D')]:
        z=p[(p.distance=='1-2')&(p['map']=='translated')&(p.cohort==cohort)&(p.metric=='near_far_ratio')];paint(ax[1],z,'lag','#77838b',mark,'Translated map, 1–2' if ls=='-' else None,1,ls)
    for a in ax:a.set_xscale('log');a.set_yscale('log');a.set_xlabel('Local sweeps');a.legend()
    ax[1].axhline(1,lw=.8,color='0.45');ax[0].set(ylabel='Overlap-loss probability (%)',title='A  |  Equally blocked sites, different surroundings');ax[1].set(ylabel='Risk ratio vs matched far sites',title='B  |  Does the location of free sites matter?')
    fig.suptitle('β = 3.05  ·  Starting near a site with a non-uphill swap',fontweight='bold',size=15);fig.supxlabel('Same parent, species, coordination and exact minimum swap cost. Only initially uphill occupied sites.\nSolid: 64 parents × 16 local children. Dashed: 32 different parents × one child; not replicated propensities. Parent-level 95% intervals.',fontsize=9,color='#44515b')
    finish(fig,out,'soft-sites')


def maps(path,out):
    fig,axs=plt.subplots(1,4,figsize=(12,3.5),layout='constrained');norm=PowerNorm(.45,vmin=0,vmax=1);cmap=plt.colormaps['magma'].copy();cmap.set_bad('#d1dce1')
    with np.load(path,allow_pickle=False) as f:
        z=10;mask=f['species'][z]==0;pid=int(f['parent_id']);lag=int(f['lag']);rep=int(f['replicas'])
        for ax,key,name in zip(axs,['observed','local','barrier','rsmi'],['Observed across children','Species + coordination','+ composition and barrier','+ frozen RSMI']):
            im=ax.imshow(np.ma.masked_where(mask,f[key][z]),vmin=None,norm=norm,cmap=cmap,interpolation='nearest',origin='lower');ax.set_title(name,fontsize=10);ax.set(xticks=[],yticks=[])
    fig.colorbar(im,ax=axs,location='bottom',aspect=65,shrink=.65,ticks=[0,.05,.1,.25,.5,1],label='Probability of species-overlap loss at an initially occupied site');fig.suptitle(f'One held-out parent, same 20 × 20 slice  ·  β = 3.05, {lag:,} local sweeps',fontweight='bold',size=15);fig.supxlabel(f'Endpoint {pid} · z = {z} · {rep} independent local children · gray: initially vacant',fontsize=9,color='#44515b');finish(fig,out,'maps')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('results',type=Path);a=ap.parse_args();o=a.results
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlelocation':'left','axes.titleweight':'semibold','axes.titlesize':11,'legend.frameon':False,'legend.fontsize':8})
    files=sorted((o/'scores').glob('*.csv'));data=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    cold_scale(o,data);s=summary(data);s.to_csv(o/'performance.csv',index=False);p=proximity_summary(pd.read_csv(o/'proximity-parents.csv'));p.to_csv(o/'proximity.csv',index=False);performance(s,o);defects(p,o);maps(o/'scores/cold-replicated-example.npz',o)

if __name__=='__main__':main()
