"""Experiment 06 figures/decisions from the frozen comparison tables only."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from stock_glass_cubic import OUT,SEEDS,CHAR
import stock_glass_modal as stock

CL=['#007c94','#c05c2d','#8655a0']; STYLE={'identity':('Fixed frame','#67757e',':'),'proper':('24 rotations','#c36a31','--'),'full':('48 incl. inversion','#007b87','-')};IR=[k+p for k in CHAR for p in 'gu']
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'axes.titlesize':11,'savefig.dpi':155})


def band(ax,d,k,c,label,ls='-'):
    x=d.beta.to_numpy();ax.plot(x,d[k],ls,marker='o',ms=3,c=c,label=label);ax.fill_between(x,d[k+'_lo'].to_numpy(),d[k+'_hi'].to_numpy(),color=c,alpha=.15)


def save(fig,out,k):fig.savefig(out/(k+'.png'),bbox_inches='tight');plt.close(fig)


def render(out):
    read=lambda k:pd.read_csv(out/(k+'.csv'));pairs=read('seed-pairs');p=read('symmetry-power');npow=read('random-power');ov=read('reference-overlap');rn=read('random-overlap');su=read('support');cold=pairs[pairs['index']==9];dec={};by=[]
    for b,g in pairs.groupby('index'):
        for space,h in g.groupby('space'):
            informative=bool(h.nonconstant.all());exact=informative and bool((h.exact_count>0).all());near='collapsed' if not informative else 'shared' if (h.NMI_lo>.90).all() else 'ruled_out' if (h.NMI_hi<.90).any() else 'unresolved';by.append(dict(index=b,beta=h.beta.iloc[0],space=space,informative=informative,exact_all=exact,exact_pairs=int((h.exact_count>0).sum()),approximate=near,least_lower_NMI=h.NMI_lo.min(),largest_upper_NMI=h.NMI_hi.max()));
            if b==9:dec[space]=dict(exact_all=exact,approximate=near)
    pd.DataFrame(by).to_csv(out/'family-by-temperature.csv',index=False);a=p[(p['index']==9)&(p.function=='labels')].set_index('seed');n=npow[(npow['index']==9)&(npow.function=='labels')].groupby('seed')[IR].mean();tops=a[IR].idxmax(axis=1);delta=[a.loc[s,tops[s]]-n.loc[s,tops[s]] for s in SEEDS];shared=bool(tops.nunique()==1 and min(delta)>=.05);dec.update(cold_beta=3.05,dominant_label_irreps=tops.tolist(),excess_over_null_mean=delta,shared_character=shared,verdict='common partition up to cubic symmetry' if dec['full']['approximate']=='shared' else 'common symmetry character, distinct partitions' if shared and dec['full']['approximate']=='ruled_out' else 'no common family resolved',spend_usd=0,started_07=False);stock.write_json(out/'decision.json',dec)
    fig,axs=plt.subplots(2,3,figsize=(13.3,6.6),sharex=True,constrained_layout=True)
    for j,((s1,s2),h) in enumerate(pairs.groupby(['seed_A','seed_B'])):
        for space,(label,c,ls) in STYLE.items():
            d=h[h.space==space];band(axs[0,j],d,'NMI',c,label,ls);band(axs[1,j],d,'disagreement',c,label,ls)
        d=h[h.space=='full'];ax=axs[0,j];ax.fill_between(d.beta.to_numpy(),d.null_p025_NMI.to_numpy(),d.null_p975_NMI.to_numpy(),color='#969c90',alpha=.18);ax.plot(d.beta,d.null_mean_NMI,c='#6c7762',ls='-.',lw=1,label='Optimized null mean');ax.axhline(.9,lw=.8,ls=':',c='black');ax.set_title(f'Seeds {s1} vs {s2}');ax.set_ylim(-.03,1.04);axs[1,j].set_ylim(-.03,1.04);axs[1,j].set_xlabel('β');
        for a0 in axs[:,j]:a0.grid(alpha=.15)
    axs[0,0].legend(fontsize=8,frameon=False);axs[0,0].set_ylabel('Held normalized mutual information');axs[1,0].set_ylabel('Held probability of different labels');fig.suptitle('Same partition after a change of cubic frame? TRAIN chooses; TEST scores');fig.text(.01,-.012,'Bands on data: pointwise parent 95%. Green-gray band: 2.5–97.5% over 64 independently shuffled pairs with identical TRAIN search. β=0 trivial constants have no NMI.',fontsize=8);save(fig,out,'held-agreement')
    fig,axs=plt.subplots(1,3,figsize=(13.7,3.9),constrained_layout=True);full=pairs[pairs.space=='full'];
    for j,((sa,sb),d) in enumerate(full.groupby(['seed_A','seed_B'])):
        c=CL[j];lab=f'{sa}/{sb}';band(axs[0],d,'HA_given_B',c,lab);axs[0].plot(d.beta,d.HB_given_A,'--',c=c,lw=1);band(axs[1],d,'scalar_r',c,lab);band(axs[2],d,'gain_NMI',c,lab)
    for ax,title,y in zip(axs,['Residual label uncertainty','Frozen scalar agreement','Does adding inversion help?'],['Conditional entropy (nats)','TRAIN-sign Pearson r on TEST','Paired NMI(48) − NMI(24)']):ax.set(title=title,xlabel='β',ylabel=y);ax.grid(alpha=.15)
    axs[0].legend(frameon=False,fontsize=7);axs[1].set_ylim(-1,1);axs[2].axhline(0,c='gray',lw=.7);fig.suptitle('Partition and scalar agreement are different observables');fig.text(.01,-.026,'Left: solid H(A|B), dashed H(B|A); fixed orientation at each β. The 48-group includes reflections as inversion × proper rotation.',fontsize=8);save(fig,out,'entropy-scalar-inversion')
    coldp=p[p['index']==9];coldn=npow[npow['index']==9];fig,axs=plt.subplots(1,2,figsize=(13.3,5.5),constrained_layout=True,gridspec_kw={'width_ratios':[1.6,1]});names=[];rows=[]
    for fn in ('labels','scalar'):
        for s in SEEDS:names.append(f'{fn} {s}');rows.append(coldp[(coldp.function==fn)&(coldp.seed==s)][IR].iloc[0].to_numpy(float))
    for fn in ('packing','center_vacancy','band_0.01'):names.append(fn);rows.append(coldp[coldp.function==fn][IR].iloc[0].to_numpy(float))
    names.append('Shuffled labels (mean)');rows.append(coldn[coldn.function=='labels'][IR].mean().to_numpy(float));arr=np.stack(rows);ax=axs[0];im=ax.imshow(arr,aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set(xticks=range(10),xticklabels=IR,yticks=range(len(names)),yticklabels=names,title='β=3.05 · cubic character fractions');
    for (i,j),x in np.ndenumerate(arr):ax.text(j,i,f'{x:.2f}',ha='center',va='center',fontsize=7,color='white' if x>.58 else '#152332')
    fig.colorbar(im,ax=ax,shrink=.7);ax=axs[1];ref='band_0.01';d=ov[(ov.function=='labels')&(ov.reference==ref)];rnd=rn[(rn.function=='labels')&(rn.reference==ref)].groupby('beta').overlap.mean();
    for j,(s,g) in enumerate(d.groupby('seed')):ax.plot(g.beta,g.overlap,'o-',c=CL[j],ms=3,label=f'Labels {s}')
    ax.plot(rnd.index,rnd,ls='-.',c='#666',label='Shuffled labels');ax.plot(su.beta,su[ref]/209,'k:',label='Rank / 209');ax.set(xlabel='β',ylabel='Fraction of label-span power',title='Overlap with 03 near-leading band');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8);ax2=ax.twinx();ax2.step(su.beta,su[ref],where='mid',alpha=.35,c='#62855b');ax2.set_ylabel('Band rank (green step)');fig.suptitle('Sharing a representation sector does not make two functions identical');fig.text(.01,-.017,'g/u = even/odd under inversion. Fractions use orbit-averaged TRAIN mass. β=0 has a 209-dimensional flat zero band, not a physically preferred mode.',fontsize=8);save(fig,out,'symmetry-and-band')
    fig,axs=plt.subplots(2,3,figsize=(13.4,6.3),constrained_layout=True);betas=su.beta.to_numpy()
    for j,s in enumerate(SEEDS):
        for i,fn in enumerate(('labels','scalar')):
            d=p[(p.function==fn)&(p.seed==s)].sort_values('index');ax=axs[i,j];ax.imshow(d[IR].to_numpy().T,aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set(xticks=range(10),xticklabels=[f'{b:.2f}' for b in betas] if i else [],yticks=range(10),yticklabels=IR if j==0 else [],title=f'{fn.capitalize()} · {s}');ax.tick_params(length=0,labelsize=8);ax.set_facecolor('#ddd');ax.set_xlabel('β' if i else '')
    fig.suptitle('All ten temperatures · fractional cubic power (white 0 → blue 1; gray undefined)');save(fig,out,'symmetry-temperature')
    ex=read('cold-example').head(12);fig,axs=plt.subplots(1,2,figsize=(10,5.6),constrained_layout=True,gridspec_kw={'width_ratios':[1.5,1]});species=np.array([['.AB'.index(x) for x in pat] for pat in ex.pattern]);a=axs[0];codes=np.array([[int(v) for v in ex[k]] for k in ('A','B_native','B_matched')]).T;a.imshow(species,vmin=0,vmax=2,cmap=ListedColormap(['#e9ecec','#3994a7','#e39c60']),aspect='auto');a.set(yticks=range(12),yticklabels=[f'{int(k)} · {v:.2%}' for k,v in zip(ex.state,ex.raw_train_mass)],xticks=range(7),xticklabels=['ctr','−z','+z','−y','+y','−x','+x'],title='Labeled star in fixed A frame',ylabel='State index · TRAIN marginal');
    for (i,j),v in np.ndenumerate(species):a.text(j,i,'.AB'[v],ha='center',va='center',fontsize=8)
    ax=axs[1];ax.imshow(codes,vmin=0,vmax=3,cmap=ListedColormap(['#dae3df','#87bec9','#d6a0bc','#f5c991']),aspect='auto');ax.set(yticks=range(12),yticklabels=[],xticks=range(3),xticklabels=['A','B native','B aligned'],title='Actual two-bit categories');
    for (i,j),v in np.ndenumerate(codes):ax.text(j,i,f'{v:02b}',ha='center',va='center',fontsize=8)
    fig.suptitle('β=3.05 · 6101 vs 6102 · twelve most probable TRAIN states');fig.text(.01,-.015,'B aligned = B(gx) after the single TRAIN-selected full-group action and label bijection. Native labels are arbitrary. No patterns chosen on TEST.',fontsize=8);save(fig,out,'cold-maps')
    print(dec,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=OUT,type=Path);render(ap.parse_args().out)
