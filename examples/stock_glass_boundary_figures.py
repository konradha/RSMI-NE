"""04 figures from published parent-level summaries only."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

STYLE={'vacancy':('Center vacancy','#222c38','s'),'linear':('Site-linear','#61944a','^'),'hard_6101':('Code 6101','#007d9c','o'),'hard_6102':('Code 6102','#cd602a','o'),'hard_6103':('Code 6103','#8556a1','o')}
LAW={'fields':('Fields only','#b57427','--'),'in_only':('Only $J_{in}$','#007c90','-.'),'out_only':('Only $J_{out}$','#95579c',':'),'full':('Both pairs','#253441','-')}
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'axes.titlesize':11,'savefig.dpi':155})


def curve(ax,d,x,col,label,color,mark='o',ls='-'):
    d=d.sort_values(x);v=d[x].to_numpy();lo=d[col+'_lo'].to_numpy();hi=d[col+'_hi'].to_numpy();ax.fill_between(v,lo,hi,color=color,alpha=.12,lw=0);ax.plot(v,d[col].to_numpy(),marker=mark,color=color,ls=ls,label=label,lw=1.5,ms=4)


def save(fig,out,name):fig.savefig(out/(name+'.png'),bbox_inches='tight');plt.close(fig)


def render(out):
    df=pd.read_csv(out/'memory.csv');ds=pd.read_csv(out/'distortion.csv');beta=df.drop_duplicates('index').sort_values('index').beta.to_numpy(); fig=plt.figure(figsize=(10.3,7.6),constrained_layout=True);gs=fig.add_gridspec(2,2);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1],sharex=a,sharey=a);c=fig.add_subplot(gs[1,:]);axes=[a,b,c]
    for ax,model,col,title in zip(axes,['fields','full','fields'],['T','T','full_minus_T'],['A   Boundary fields without induced pairs','B   Original Hamiltonian (fields + both pairs)','C   Full minus fields-only on paired TEST parents']):
        for op,(label,color,mk) in STYLE.items():curve(ax,df[(df.observable==op)&(df.law==model)],'beta',col,label,color,mk)
        ax.set(title=title,xlabel='Inverse temperature β',ylabel='Frozen memory T' if ax!=c else 'Signed difference in T');ax.grid(alpha=.16)
    a.set_ylim(-.035,1.04);c.axhspan(-.01,.01,color='#606e78',alpha=.1);c.axhline(0,c='gray',lw=.6);c.legend(ncol=3,frameon=False,fontsize=9);fig.suptitle('What changes when the two boundary-induced pair terms are restored?',fontsize=14);fig.text(.01,-.015,'Same empirical exteriors and same frozen scores in every panel. Bands: pointwise 95% parent intervals. At β=0 hard codes are constant (undefined T).',fontsize=8);save(fig,out,'field-vs-full')
    fig,axs=plt.subplots(2,5,figsize=(14.3,5.7),sharex=True,sharey=True,constrained_layout=True)
    for i,ax in enumerate(axs.flat):
        sub=df[(df['index']==i)&df.law.isin(['fields','joint25','joint50','joint75','full'])];ax.axhline(0,color='gray',lw=.6)
        for op,(label,color,mk) in STYLE.items():curve(ax,sub[sub.observable==op],'jin','minus_fields_T',label,color,mk)
        ax.set_title(f'β={beta[i]:.3f} · n={sub.parents.iloc[0]:,}');ax.set_xticks([0,.5,1]);ax.grid(alpha=.16)
    axs[0,0].legend(fontsize=7,frameon=False);fig.suptitle('Joint coupling profile · fields and intrinsic interactions retained');fig.supxlabel('λ on both induced pair terms');fig.supylabel('Paired change T(λ) − T(0)');save(fig,out,'coupling-profile')
    fig,axs=plt.subplots(2,2,figsize=(10.4,7.4),constrained_layout=True)
    for ax,law,title in zip(axs.flat,['fields','in_only','out_only'],['A   Delete both pair terms','B   Delete exterior-centered pairs; keep interior pairs','C   Delete interior-centered pairs; keep exterior pairs']):
        for op in ['state','hard_6101','hard_6102','hard_6103']:
            label,color,mk=('210-state law','#222c38','s') if op=='state' else STYLE[op];curve(ax,ds[(ds.space==op)&(ds.law==law)],'beta','JS_bits',label,color,mk)
        ax.set(title=title,xlabel='Inverse temperature β',ylabel='Mean conditional JS from full law (bits)',ylim=(-.03,1.025));ax.grid(alpha=.16)
    axs[0,0].legend(frameon=False,fontsize=8)
    ax=axs[1,1];sub=df[df.observable=='vacancy'];
    for law,(label,color,ls) in LAW.items():curve(ax,sub[sub.law==law],'beta','state_coincidence',label,color,'.',ls)
    ax.set(title='D   Independent replicas: whole stars match',xlabel='Inverse temperature β',ylabel='Mean conditional coincidence probability',ylim=(-.03,1.025));ax.grid(alpha=.16);ax.legend(frameon=False,fontsize=8);fig.suptitle('Microscopic distortion can persist even when a scalar barely changes');save(fig,out,'conditional-distortion')
    fig,axs=plt.subplots(1,3,figsize=(13,4.2),constrained_layout=True);cols=['T','between','total'];names=['A   Normalized memory','B   Between-exterior variance','C   Total variance'];x=np.arange(len(STYLE));cold=df[df['index']==9]
    for ax,col,title in zip(axs,cols,names):
        for j,(law,(label,color,ls)) in enumerate(LAW.items()):
            d=cold[cold.law==law].set_index('observable').loc[list(STYLE)];y=d[col].to_numpy();lo=d[col+'_lo'].to_numpy();hi=d[col+'_hi'].to_numpy();ax.errorbar(x+(j-1.5)*.095,y,yerr=[y-lo,hi-y],color=color,marker=['o','s','^','D'][j],ls='none',capsize=2,ms=4,label=label)
        ax.set(xticks=x,xticklabels=['Vac.','Linear','6101','6102','6103'],title=title,ylabel='T' if col=='T' else 'Variance of frozen score');ax.grid(alpha=.15)
    axs[0].legend(fontsize=8,frameon=False);fig.suptitle('β=3.05 · which pair term matters? Same five scores in all four laws');fig.text(.01,-.035,'Fields always present. Raw variance units are score-specific. Total variance is over both exterior and conditional interior; their ratio is panel A.',fontsize=8);save(fig,out,'cold-components')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);render(p.parse_args().out)
