"""Render experiment 07 directly from its held forecast tables (no model selection)."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stock_glass_forecast import SEEDS, OUT

COL={'prior':'#777777','packing':'#c56629','directional':'#007e85','rotated':'#79599f','oracle':'#151515'}
STYLE={'prior':':','packing':'--','directional':'-','rotated':'-.','oracle':'-'}

def line(ax,d,x,y,label,color,style='-',band=True):
    d=d.sort_values(x);a=d[x].to_numpy(float);b=d[y].to_numpy(float);ax.plot(a,b,style,color=color,marker='o',ms=3,label=label)
    if band and y+'_lo' in d:ax.fill_between(a,d[y+'_lo'].to_numpy(float),d[y+'_hi'].to_numpy(float),color=color,alpha=.13,linewidth=0)

def save(fig,path):fig.savefig(path,dpi=155,bbox_inches='tight');plt.close(fig)

def tail_panel(out,tail):
    fig,axs=plt.subplots(1,3,figsize=(12.5,4),sharey=True,constrained_layout=True)
    for ax,seed in zip(axs,SEEDS):
        d=tail[(tail.seed==seed)&(tail.view!='prior')].set_index('view')
        for j,v in enumerate(('packing','directional','rotated')):
            r=d.loc[v];xs=np.array([0,1])+.15*(j-1)
            for x,metric in zip(xs,['precision','recall']):
                y=r[metric];ax.errorbar(x,100*y,yerr=[[100*(y-r[metric+'_lo'])],[100*(r[metric+'_hi']-y)]],fmt='o',color=COL[v],capsize=3,label=v if metric=='precision' else None)
        ax.scatter([0,1],[100*r.events/r.ntest,10],marker='_',s=600,color='black',label='Uninformed (≈10% flagged)')
        ax.set(title=f'Seed {seed}: {int(r.events)}/{int(r.ntest)} ambiguous',xticks=[0,1],xticklabels=['Among flagged,\nactually ambiguous','Among ambiguous,\nflagged'],xlim=(-.4,1.4),ylim=(0,34))
    axs[0].set_ylabel('Held fraction (%) · Wilson 95%');axs[0].legend(fontsize=7)
    fig.suptitle('Cold weak pinning: ranking the weakest predicted tenth enriches it, but misses most cases\nAmbiguous = exact replica agreement < 0.8. Selection thresholds fixed on TRAIN.')
    save(fig,out/'cold-ambiguous-tail.png')

def draw(out):
    tab=lambda n:pd.read_csv(out/(n+'.csv'));sc=tab('held-scores');g=tab('paired-gains');sup=tab('support');cal=tab('calibration');desc=tab('descriptors');rep=tab('representatives');tail=tab('ambiguous-tail');
    tail_panel(out,tail)
    fig,axes=plt.subplots(2,3,figsize=(14,7.2),sharex=True,constrained_layout=True)
    for i,seed in enumerate(SEEDS):
        d=sc[sc.seed==seed];s=sup[sup.seed==seed];ax=axes[0,i];ax.set_title(f'Frozen code {seed}')
        for v in COL:
            if v=='oracle':line(ax,s,'beta','true_entropy_bits','exact oracle',COL[v],STYLE[v]);continue
            line(ax,d[d.view==v],'beta','bits',v,COL[v],STYLE[v])
        ax=axes[1,i];gg=g[(g.seed==seed)&(g.metric=='logloss_bits')];
        for poor,better,c in [('prior','packing',COL['packing']),('packing','directional',COL['directional']),('rotated','directional',COL['rotated'])]:line(ax,gg[(gg.baseline==poor)&(gg.forecast==better)],'beta','reduction',poor+' − '+better,c)
        ax.axhline(.01,color='gray',ls=':',lw=.8);ax.set_xlabel('Inverse temperature β')
    axes[0,0].set_ylabel('Held expected log loss (bits / cavity) ↓');axes[1,0].set_ylabel('Paired loss reduction (bits / cavity) ↑');axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8);fig.suptitle('Which label?  Independent exteriors, one frozen target per seed');save(fig,out/'which-label.png')
    fig,axes=plt.subplots(3,3,figsize=(14,10.2),constrained_layout=True)
    for i,seed in enumerate(SEEDS):
        ax=axes[0,i];d=cal[(cal.seed==seed)&(cal['index']==9)&(cal.kind=='strength')&(cal.n>0)];ax.plot([.25,1],[.25,1],':',c='k',lw=.8)
        for v in ('packing','rotated','directional','prior'):
            a=d[d.view==v];ax.plot(a.predicted,a.observed,STYLE[v],c=COL[v],lw=1);ax.scatter(a.predicted,a.observed,s=12+60*np.sqrt(a.n/a.total),c=COL[v],label=v);ax.vlines(a.predicted,a.observed_lo,a.observed_hi,color=COL[v],linewidth=1.2)
        ax.set(xlabel='Forecast agreement',ylabel='Exact agreement' if i==0 else '',title=f'Cold strength calibration · {seed}\n{int(d.total.iloc[0]):,} parents');
        ax=axes[1,i];gg=g[(g.seed==seed)&(g.metric=='strength_mse')&(g.baseline=='prior')];
        for v in ('packing','directional'):line(ax,gg[gg.forecast==v],'beta','reduction',v,COL[v]);
        ax.axhline(0,c='gray',lw=.65);ax.set_xlabel('β');s=sup[sup.seed==seed];ax=axes[2,i];line(ax,s,'beta','s_sd','SD of exact strength','#005488',band=False);line(ax,s,'beta','ambiguous_fraction','fraction with s < 0.8','#bf5b25');ax.set_xlabel('β')
    axes[0,0].legend(fontsize=8);axes[1,0].set_ylabel('MSE reduction vs TRAIN mean ↑');axes[1,0].legend(fontsize=8);axes[2,0].set_ylabel('Target dispersion / ambiguous fraction');axes[2,0].legend(fontsize=8);fig.suptitle('How strongly does a boundary pin the label? Separate bounded forecasts of replica agreement\nCold dots: fixed bins; area increases with support. Interval bands/bars: pointwise parent 95%.',fontsize=14);save(fig,out/'pinning-strength.png')
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),constrained_layout=True)
    for i,seed in enumerate(SEEDS):
        ax=axes[0,i];a=cal[(cal.seed==seed)&(cal['index']==9)&(cal.kind=='label')&(cal.n>0)];ax.plot([.25,1],[.25,1],':',c='k',lw=.8)
        for v in ('packing','directional','rotated','prior'):
            p=a[a.view==v];ax.plot(p.predicted,p.observed,STYLE[v],color=COL[v],label=v);ax.scatter(p.predicted,p.observed,s=12+55*np.sqrt(p.n/p.total),c=COL[v]);ax.vlines(p.predicted,p.observed_lo,p.observed_hi,color=COL[v]);
        ax.set(title=f'β=3.05 · seed {seed}',xlabel='Predicted modal-label probability');ax=axes[1,i];d=rep[(rep.seed==seed)&(rep['index']==9)].sort_values('quantile');matrix=d[[f'q{y}' for y in range(4)]].to_numpy();ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto');
        for row,(_,rec) in enumerate(d.iterrows()):
            for h in range(4):
                val=matrix[row,h];ax.text(h,row,f'{val:.2f}'+(' *' if int(rec.prediction)==h else ''),ha='center',va='center',color='white' if val>.6 else 'black')
        ax.set(xticks=range(4),xticklabels=['00','01','10','11'],yticks=range(3),yticklabels=['10th','50th','90th'],xlabel='Exact held label mass; * = forecast mode')
    axes[0,0].legend(fontsize=8);axes[0,0].set_ylabel('Exact mass of predicted label');axes[1,0].set_ylabel('TRAIN confidence percentile');fig.suptitle('Modal-label calibration and three preregistered confidence examples per code');save(fig,out/'modal-calibration.png')
    fig,axes=plt.subplots(1,3,figsize=(15,5.4),sharey=True,constrained_layout=True)
    for ax,seed in zip(axes,SEEDS):
        ds=desc[(desc['index']==9)&(desc.seed==seed)].sort_values('label');h=np.arange(len(ds));ax.plot(h,ds.positive_q,'o',color='#097f86',label='positive sign group');ax.plot(h,ds.negative_q,'s',color='#b85f22',label='negative sign group');ax.vlines(h,ds.positive_q_lo,ds.positive_q_hi,color='#097f86');ax.vlines(h,ds.negative_q_lo,ds.negative_q_hi,color='#b85f22');
        for j,r in enumerate(ds.itertuples()):ax.text(j,.017,f'{r.positive_n}/{r.negative_n}',ha='center',fontsize=7,rotation=27)
        ax.set(title=f'β=3.05 · seed {seed}',xticks=h,xticklabels=[f'{int(r.label):02b}\n{r.channel}\n{r.position}' for r in ds.itertuples()],ylim=(0,1),xlabel='Label / shell channel / reference position');ax.tick_params(axis='x',labelsize=8)
    axes[0].set_ylabel('Exact mass assigned to this label');axes[0].legend(fontsize=8);fig.suptitle('What distinguishes labels? Opposite-site contrasts chosen and oriented on TRAIN\nNumbers: held positive / negative counts; ties excluded. Signs and both sites in descriptors.csv.',fontsize=13);save(fig,out/'boundary-descriptors.png')
    cold=g[(g['index']==9)&(g.metric=='logloss_bits')&(g.baseline=='packing')&(g.forecast=='directional')].sort_values('seed');strength=g[(g['index']==9)&(g.metric=='strength_mse')&(g.baseline=='prior')&(g.forecast=='directional')].sort_values('seed');ctl=g[(g['index']==9)&(g.metric=='logloss_bits')&(g.baseline=='rotated')&(g.forecast=='directional')].sort_values('seed');
    receipt={'primary_beta':3.05,'margin_bits':.01,'seeds':cold.seed.tolist(),'directional_label_gain':cold.reduction.tolist(),'directional_gain_lower95':cold.reduction_lo.tolist(),'cold_all_clear_margin':bool((cold.reduction_lo>.01).all()),'cold_all_beat_matched_rotations':bool((ctl.reduction_lo>0).all()),'strength_mse_reduction_vs_mean':strength.reduction.tolist(),'strength_lower95':strength.reduction_lo.tolist(),'ambiguous_events':tail[tail.view=='directional'].sort_values('seed').events.tolist(),'modal_usd':0,'started_08':False};(out/'decision.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=__import__('pathlib').Path,default=OUT);draw(p.parse_args().out)
