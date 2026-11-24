"""Plots of the registered confined kinetic quantities, not bulk relaxation times."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stock_glass_local_kinetics import OUT,NAMES

LABEL=('stock 6101','stock 6102','stock 6103','center vacancy','03 site-linear','common slow','supported common');COL=('#008388','#cf5f25','#73599b','#888888','#285a9e','#1c1c1c','#9d7623');STY=('-','-','-',':','--','-.',':')

def series(ax,d,x,y,k):
    d=d.sort_values(x);xv=d[x].to_numpy(float);yv=d[y].to_numpy(float);ax.plot(xv,yv,STY[k],marker='o',ms=3,color=COL[k],label=LABEL[k]);
    if y+'_lo' in d:ax.fill_between(xv,d[y+'_lo'].to_numpy(float),d[y+'_hi'].to_numpy(float),alpha=.13,linewidth=0,color=COL[k]);
    if 'adequate' in d:
        bad=(~d.adequate.to_numpy())&np.isfinite(yv);ax.scatter(xv[bad],yv[bad],marker='x',s=45,color=COL[k],zorder=4)

def save(fig,path):fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)

def draw(out):
    read=lambda name:pd.read_csv(out/(name+'.csv'),dtype={'name':str,'reference':str});obs=read('observable-moments');lags=read('finite-lags');pair=read('paired-differences');g=read('spectrum-support');cold=obs[(obs['index']==9)&obs.valid];
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.7),constrained_layout=True)
    for k,name in enumerate(NAMES):
        d=obs[obs.name==name];series(axes[0],d,'beta','slope',k);series(axes[1],d,'beta','C100',k)
    axes[0].set(yscale='log',ylabel='Initial connected decay rate (per internal attempt)',xlabel='β');axes[1].set(ylabel='Connected memory at 100 internal attempts',xlabel='β');axes[1].legend(fontsize=8,ncol=2);fig.suptitle('Same clock, different observables · all eligible exteriors\nPointwise parent 95% bands; × = inadequate variance support. Constant stock codes at β=0 omitted.');save(fig,out/'slope-and-lag.png')
    fig,axs=plt.subplots(2,3,figsize=(14.7,7.8),sharex=True,sharey=True,constrained_layout=True)
    for j,(b,title) in enumerate([(1,'β=0.339'),(5,'β=1.694'),(9,'β=3.05')]):
        for k,name in enumerate(NAMES[:6]):
            d=lags[(lags['index']==b)&(lags.name==name)&lags.valid].sort_values('lag');
            if not len(d):continue
            start=d.iloc[:1].copy();start['lag']=0.;
            for metric in ['connected','raw']:
                for suf in ('','_lo','_hi'):start[metric+suf]=1.
            pp=pd.concat([start,d],ignore_index=True);series(axs[0,j],pp,'lag','connected',k);series(axs[1,j],pp,'lag','raw',k);axs[1,j].plot([1300,2900],[d.plateau.iloc[0]]*2,lw=3,c=COL[k]);
        axs[0,j].set_title(title);axs[1,j].set_xlabel('Attempted internal bonds');axs[1,j].set_xscale('symlog',linthresh=1);axs[1,j].set_xticks([0,1,10,100,1000],['0','1','10','100','1000']);
    axs[0,0].set_ylabel('Within-exterior connected memory');axs[1,0].set_ylabel('Global-mean-centered memory');axs[0,0].legend(fontsize=7);fig.suptitle('Subtract the mean of each exterior before interpreting a slow decay\nBottom: raw memory tends to a static plateau; horizontal bars at right are t→∞, not additional lags.');save(fig,out/'plateau-vs-decay.png')
    fig,axs=plt.subplots(2,2,figsize=(12.6,8.4),constrained_layout=True)
    ax=axs[0,0];x=g.beta.to_numpy();lo=g.gap_p50_lo.to_numpy();hi=g.gap_p50_hi.to_numpy();ok=lo>0;ax.semilogy(x[ok],hi[ok],'-o',c='#007c83',label='Median gap (resolved)');ax.scatter(x[~ok],hi[~ok],marker='v',c='#af5722',label='Median upper bound');ax.semilogy(x,g.resolution_max,':',c='gray',label='Maximum numerical floor');ax.set(xlabel='β',ylabel='Positive gap / internal attempt');ax.legend(fontsize=8)
    ax=axs[0,1];ax.plot(x,100*g.unresolved/g.n,'o-',color='#b45222',label='Unresolved gap');ax.plot(x,100*g.fraction_single_state_99,'s--',c='#4c708e',label='One state carries ≥99%');ax.set(xlabel='β',ylabel='Fraction of TEST exteriors (%)',ylim=(-3,103));ax.legend(fontsize=8)
    ax=axs[1,0];
    for k,name in enumerate(NAMES):series(ax,obs[(obs.name==name)&obs.valid],'beta','within_ESS',k)
    ax.axhline(100,color='gray',ls=':');ax.set(xlabel='β',ylabel='Effective parents for within variance',yscale='log');ax.legend(fontsize=7,ncol=2)
    ax=axs[1,1];xs=np.arange(len(cold));
    for j,(key,lab,color) in enumerate([('power0.0001','gap + 0.0001','#b9813c'),('power0.001','gap + 0.001','#007c83'),('power0.01','gap + 0.01','#405b98')]):
        d=cold[key].to_numpy();a=cold[key+'_lo'].to_numpy();b=cold[key+'_hi'].to_numpy();ax.errorbar(xs+.17*(j-1),d,yerr=np.array([d-a,b-d]),fmt='o',ms=4,label=lab,color=color,capsize=2)
    ax.set(xticks=xs,xticklabels=[LABEL[NAMES.index(n)] for n in cold.name],ylabel='Fraction of thermal variance in slow band',title='β=3.05 · nonstationary subspaces');ax.tick_params(axis='x',labelrotation=24,labelsize=8);ax.legend(fontsize=8);fig.suptitle('How much slow memory is numerically and statistically resolved?');save(fig,out/'spectral-support.png')
    fig,ax=plt.subplots(figsize=(9,3.8),constrained_layout=True);d=pair[(pair['index']==9)&(pair.metric=='C100')];
    for k,ref in enumerate(('site_linear','common_slow','center_vacancy')):
        q=d[d.reference==ref].sort_values('name');v=q.learned_minus_reference.to_numpy();a=q.learned_minus_reference_lo.to_numpy();b=q.learned_minus_reference_hi.to_numpy();yy=np.arange(3)+(.19*(k-1));c=COL[NAMES.index(ref)];ax.errorbar(v,yy,xerr=[v-a,b-v],fmt='o',capsize=3,c=c,label=LABEL[NAMES.index(ref)]);ax.scatter(v[~q.adequate],yy[~q.adequate],marker='x',color='red',s=45,zorder=5)
    ax.axvline(0,c='gray',lw=.8);ax.axvline(.05,c='k',ls=':',label='+.05 primary margin vs site-linear');ax.set(yticks=range(3),yticklabels=LABEL[:3],xlabel='Learned minus comparator · paired connected correlation at 100 attempts');ax.legend(fontsize=8);fig.suptitle('Cold primary endpoint · 8,187 held parents (× = inadequate support)');save(fig,out/'cold-paired.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);draw(p.parse_args().out)
