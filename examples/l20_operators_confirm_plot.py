"""Replot fresh-parent tests and normalize directional accuracy by sibling reproducibility."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student


def ci(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)];m=a.mean();d=student.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a));return m,m-d,m+d


def comparisons(datasets,block):
    rows=[]
    contrasts=([('kinetics','kinetics+learned'),('kinetics','kinetics+init'),('kinetics','kinetics+corners'),('kinetics+init','kinetics+learned')] if block else [('direction','barrier'),('direction','learned'),('barrier','barrier+learned'),('barrier','barrier+random'),('barrier','barrier+simple'),('barrier+random','barrier+learned')]);metrics=['brier','logloss'] if block else ['brier','logloss','direction_mse']
    if not block:contrasts += [('direction','neural'),('barrier','barrier+neural'),('barrier','barrier+neural-random'),('barrier+neural-random','barrier+neural')]
    for (beta,lag),g in datasets.groupby(['beta','lag']):
        for metric in metrics:
            f=g.pivot(index='parent_id',columns='model',values=metric)
            for base,model in contrasts:
                delta=(f[base]-f[model]).dropna();m,lo,hi=ci(delta);rows.append(dict(beta=beta,lag=lag,parents=len(delta),metric=metric,baseline=base,model=model,baseline_loss=f[base].mean(),model_loss=f[model].mean(),gain=m,lo=lo,hi=hi))
    return pd.DataFrame(rows)


def repeatability(data):
    rows=[]
    for (beta,lag),g in data.groupby(['beta','lag']):
        g=g.sort_values('parent_id');a=g.split_var_a.to_numpy();b=g.split_var_b.to_numpy();c=g.split_cov.to_numpy();ix=np.random.default_rng(418022).integers(0,len(g),(5000,len(g)));z=c[ix].mean(1)
        metrics={'halves_correlation':(c.mean()/np.sqrt(a.mean()*b.mean()),z/np.sqrt(a[ix].mean(1)*b[ix].mean(1)))}
        for name in ('barrier','learned','random'):
            v=g['residual_'+name].to_numpy();metrics['fraction_explained_'+name]=(1-v.mean()/c.mean(),1-v[ix].mean(1)/z)
        for name,(m,samp) in metrics.items():
            lo,hi=np.quantile(samp,[.025,.975]);rows.append(dict(beta=beta,lag=lag,metric=name,parents=len(g),value=m,lo=lo,hi=hi))
    return pd.DataFrame(rows)


def showpoint(ax,x,y,lo,hi,col,marker='o'):
    ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],color=col,fmt=marker,capsize=3,ms=6)


def plots(r,out):
    def load(directory,pattern):return pd.concat([pd.read_csv(f) for f in sorted(directory.glob(pattern))])
    d=load(r/'directional','confirm-??.csv');nn=load(r/'directional-neural','confirm-??.csv')
    d=pd.concat([d,nn[nn.model.isin(['neural','neural-random','barrier+neural','barrier+neural-random'])]])
    block=load(r,'block-??.csv');rel=repeatability(load(r/'directional','confirm-??-reliability.csv'));v=comparisons(d,False);bv=comparisons(block,True)
    nce=load(r/'directional-neural','confirm-??-nce.csv');summary=[]
    for (beta,lag),g in nce.groupby(['beta','lag']):
        pivot=g.pivot(index='parent_block',columns='model',values='nce_nats')
        for model in ('neural','neural-random','learned'):
            for reference in ('zero','neural-random'):
                m,lo,hi=ci(pivot[model]-pivot[reference]);summary.append(dict(beta=beta,lag=lag,model=model,reference=reference,blocks=len(pivot),gain=m,lo=lo,hi=hi))
    pd.DataFrame(summary).to_csv(r/'directional-nce.csv',index=False)
    v.to_csv(r/'directional-comparisons.csv',index=False);bv.to_csv(r/'block-comparisons.csv',index=False);rel.to_csv(r/'repeatability.csv',index=False)
    domains=[(3.05,4096,'Cold\n4,096'),(2.711111111111,116,'Transfer\n116'),(2.711111111111,462,'Transfer\n462')]
    fig,axs=plt.subplots(1,3,figsize=(12.8,4.7),layout='constrained');colors=['#885b88','#ae8d31','#097989']
    for i,(b,t,label) in enumerate(domains):
        for k,model in enumerate(('barrier+neural','barrier+neural-random','barrier+learned')):
            row=v[np.isclose(v.beta,b)&(v.lag==t)&(v.baseline=='barrier')&(v.model==model)&(v.metric=='logloss')].iloc[0];showpoint(axs[0],i+(k-1)*.16,row.gain*1000,row.lo*1000,row.hi*1000,colors[k],['o','s','D'][k]);
        for k,model in enumerate(('halves_correlation','fraction_explained_barrier')):
            row=rel[np.isclose(rel.beta,b)&(rel.lag==t)&(rel.metric==model)].iloc[0];showpoint(axs[1],i+(k-.5)*.17,row.value,row.lo,row.hi,['#5a6873',colors[2]][k],['D','o'][k]);
        for k,model in enumerate(('kinetics+learned','kinetics+init','kinetics+corners')):
            row=bv[np.isclose(bv.beta,b)&(bv.lag==t)&(bv.baseline=='kinetics')&(bv.model==model)&(bv.metric=='logloss')].iloc[0];showpoint(axs[2],i+(k-1)*.16,row.gain*1000,row.lo*1000,row.hi*1000,colors[k],['o','s','D'][k]);
    for ax in axs:
        ax.axhline(0,color='0.5',lw=.8);ax.set(xticks=range(len(domains)),xticklabels=[x[-1] for x in domains],xlabel='Independent cohort · local sweeps',xlim=(-.44,len(domains)-.55));ax.spines[['top','right']].set_visible(False)
    for ax,items in zip(axs,[[('Neural InfoNCE','o',colors[0]),('Untrained network','s',colors[1]),('Learned tree','D',colors[2])],[('Sibling-half correlation','D','#5a6873'),('Fraction explained by barriers','o',colors[2])],[('Frozen spatial code','o',colors[0]),('Untrained initialization','s',colors[1]),('Two-corner bits','D',colors[2])]]):
        for label,marker,col in items:ax.plot([],[],marker=marker,ls='none',color=col,label=label)
        ax.legend(fontsize=8,frameon=False)
    axs[0].set(title='A  |  Does the seed code add information?',ylabel='Log-loss gain over barrier model (×10⁻³)');axs[1].set(title='B  |  Is direction itself predictable?',ylabel='Directional reproducibility / explained fraction');axs[2].set(title='C  |  Does spatial RSMI predict block loss?',ylabel='Log-loss gain over kinetic baseline (×10⁻³)');fig.suptitle('Where the L20 glass moves · frozen operators on fresh parents',fontsize=15,fontweight='bold');fig.supxlabel('Cold: β = 3.05, 512 parents × 8 children. Transfer: β = 2.7111, 256 × 8, matched t/τα. Uncertainty: 95% across independent parents.',fontsize=9);fig.savefig(out,dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);a=p.parse_args();plots(a.results,a.results/'confirmation.png')
