"""Render experiment 03 from published numerical results, no fits and no data selection."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from stock_glass_cavity_exact import X,C

COLORS=['#007998','#d1672f','#7854a1']; SEEDS=[6101,6102,6103]
plt.rcParams.update({'axes.spines.right':False,'axes.spines.top':False,'font.size':10,'axes.titlesize':11,'savefig.dpi':165})

def band(ax,d,col,color,label,style='-',offset=0,marker=None):
    x=d.beta.to_numpy()+offset;y=d[col].to_numpy();low=d[col+'_lo'].to_numpy();hi=d[col+'_hi'].to_numpy();ax.plot(x,y,style,marker=marker,color=color,label=label,markersize=4);ax.fill_between(x,low,hi,color=color,alpha=.12,linewidth=0)

def save(fig,out,name):fig.savefig(out/(name+'.png'),bbox_inches='tight');plt.close(fig)

def physical(out):
    bench=pd.read_csv(out/'benchmark.csv');spec=pd.read_csv(out/'spectrum.csv');m=pd.read_csv(out/'state-support.csv');rec=json.loads((out/'exact-receipt.json').read_bytes());au=pd.DataFrame(rec['by_beta']);halves=pd.read_csv(out/'train-halves.csv');betas=au.beta.to_numpy()
    fig,axs=plt.subplots(2,2,figsize=(10,7.4),constrained_layout=True);ax,ay,az,aw=axs.flat
    for k in [1,2,3,5,10]:
        d=spec[spec['mode']==k];ax.plot(d.beta,d.train_value,'o-',ms=3,label=f'#{k}')
    ax.set(ylabel='TRAIN eigenvalue',title='A   Conditional-replica spectrum');ax.legend(frameon=False,ncol=3)
    ax2=ay.twinx();ay.plot(betas,au.top_gap,'o-',c='#007998');ax2.plot(betas[1:],au.near_leading_count.to_numpy()[1:],'s--',c='#7a528a');ay.set(ylabel='λ₁ − λ₂ (teal)',title='B   Leading contrasts are nearly degenerate');ax2.set_ylabel('Modes within 0.01 of λ₁ (purple)');ax2.spines['right'].set_visible(True)
    for n,c in [('oracle','#242e34'),('linear','#52823f'),('best_simple','#a8721b'),('common_mass_oracle','#ad58a0')]:
        d=bench[bench.observable==n];band(az,d,'T',c,n.replace('_',' '),marker='.')
    for k,h in halves.groupby('half'):az.plot(h.beta,h.held_top,':',lw=1,color='#444444',label='TRAIN half → TEST' if k==0 else None)
    az.set(ylabel='Frozen TEST memory T',title='C   Packing competitors & support sensitivity');az.legend(frameon=False,fontsize=8)
    aw.semilogy(betas,au.test_mean_effective_states,'o-',label='Effective interiors / exterior¹');aw.semilogy(betas,au.train_entropy_states,'s--',label='Effective marginal states¹');aw.semilogy(betas[1:],au.half_variance_states.to_numpy()[1:],'d:',label='States carrying half mode variance');aw.set(ylabel='Number of states (log scale)',title='D   Cold exteriors individually select few interiors');aw.legend(frameon=False,fontsize=8)
    for a in axs.flat:a.set_xlabel('Inverse temperature β');a.grid(alpha=.15)
    fig.suptitle('Seven-site cavity · 210 inventory-preserving states · no critic in the reference',fontsize=14);fig.text(.02,-.015,'¹ exp(Shannon entropy), averaged after exponentiation for individual exteriors. Sensitivity gap at β=2.711: no state passes both fixed cutoffs.',fontsize=8);save(fig,out,'exact-spectrum')
    # Show opposite score extrema as well as a state with largest variance share (not necessarily an extremum).
    cold=m[m['index']==9];sel=[cold.loc[cold.mode_value.idxmax()],cold.loc[cold.mode_value.idxmin()],cold.loc[cold.variance_share.idxmax()]];col=['#fbfcfc','#268294','#dc7746'];fig=plt.figure(figsize=(10,4.25),layout='constrained')
    for j,(r,title) in enumerate(zip(sel,['Largest positive amplitude','Most negative amplitude','Largest variance contribution'])):
        a=fig.add_subplot(1,3,j+1,projection='3d');st=X[int(r.state)];pts=C[:,::-1];
        for t0 in pts[1:]:a.plot([0,t0[0]],[0,t0[1]],[0,t0[2]],c='#8c9396',lw=1.5)
        for k,(x,y,z) in enumerate(pts):a.scatter(x,y,z,s=175,c=col[st[k]],edgecolors='#25323b',depthshade=False);a.text(x,y,z+.12,'.AB'[st[k]],fontsize=10)
        a.set(xlabel='x',ylabel='y',zlabel='z',xticks=[-1,0,1],yticks=[-1,0,1],zticks=[-1,0,1]);a.view_init(elev=17,azim=-55);a.set_box_aspect((1,1,1));a.set_title(f'{title}\nstate {int(r.state)} · f={r.mode_value:+.2f}\nTRAIN p={r.train_mass:.3%} · TEST n={int(r.test_observed)}',fontsize=10)
    fig.suptitle('β=3.05 · TRAIN-selected leading contrast (one arbitrary sign)',fontsize=13);fig.text(.03,-.015,'White = vacancy; teal = A; orange = B. Near-degeneracy: these are examples from a frozen mode, not a unique ordered motif.',fontsize=9);save(fig,out,'star-patterns')


def learned(out,reference):
    df=pd.read_csv(out/'frozen-readout.csv');base=pd.read_csv(reference/'benchmark.csv');final=df[df.final];betas=base.drop_duplicates('index').beta.to_numpy();fig,axs=plt.subplots(2,2,figsize=(11,7.7),constrained_layout=True)
    for a,field,title in zip(axs.flat,['T','fraction','minus_linear','entropy_nats'],['A   Absolute memory','B   Fraction of frozen exact mode','C   Increment over linear site indicators','D   Code entropy (nats)']):
        if field in ['T','fraction']:
            for key,color,label in [('oracle','#23272c','210-state mode'),('best_simple','#a9680d','Best individual'),('linear','#5f873c','Best site-linear')]:band(a,base[base.observable==key],field,color,label,style='--')
        for k,(seed,c) in enumerate(zip(SEEDS,COLORS)):
            x=final[(final.channel=='hard')&(final.seed==seed)].sort_values('beta');xs=x.beta.to_numpy()+.015*(k-1)
            if field=='entropy_nats':a.plot(xs,x[field],'.-',lw=1,c=c,label=str(seed));y=final[(final.channel=='stochastic')&(final.seed==seed)];a.plot(y.beta,y.entropy_nats,':',color=c,lw=1)
            else:
                band(a,x,field,c,str(seed),offset=.015*(k-1),marker='o');zero=x[~x.valid_contrast];a.scatter(zero.beta+.015*(k-1),np.full(len(zero),.035+k*.025),marker='x',transform=a.get_xaxis_transform(),color=c,s=20,clip_on=False)
        if field=='entropy_nats':a.axhline(np.log(4),ls='--',c='gray',lw=.8);a.set_ylabel('H(code), nats · dotted = stochastic')
        elif field=='minus_linear':a.axhline(0,c='gray',lw=.8);a.axhline(.01,ls=':',c='gray',lw=.8);a.set_ylabel('Paired difference in T')
        else:a.set_ylabel('TEST '+field)
        a.set(title=title,xlabel='Inverse temperature β');a.grid(alpha=.15)
    axs[0,0].legend(frameon=False,ncol=2,fontsize=8);fig.suptitle('Final stock operator vs exact conditional memory · all starts · held parents',fontsize=14);fig.text(.01,-.015,'Points = frozen deterministic two-bit partition. Bands = pointwise parent t95. × at baseline = constant code (undefined T), not measured zero. β=0 fraction not defined.',fontsize=8);save(fig,out,'frozen-recovery')
    fig,axs=plt.subplots(2,5,figsize=(14,5.7),sharex=True,sharey=True,constrained_layout=True)
    for b,a in enumerate(axs.flat):
        q=df[df['index']==b];base0=base[(base['index']==b)&(base.observable=='oracle')]['T'].iloc[0];a.axhline(base0,lw=.8,ls='--',c='black')
        for k,(s,c) in enumerate(zip(SEEDS,COLORS)):
            for ch,ls in [('hard','-'),('stochastic',':')]:
                d=q[(q.seed==s)&(q.channel==ch)].sort_values('depth');a.plot(d.depth,d['T'],ls,marker='.' if ch=='hard' else None,color=c,lw=1);a.fill_between(d.depth.to_numpy(),d.T_lo.to_numpy(),d.T_hi.to_numpy(),color=c,alpha=.10);zero=d[~d.valid_contrast];a.scatter(zero.depth,[-.065-.027*k]*len(zero),s=17,marker='x',color=c)
        a.set(xscale='log',ylim=(-.16,1.07),title=f'β={betas[b]:.3f}',xticks=[850,3400,10000],xticklabels=['850','3.4k','10k']);a.minorticks_off();a.grid(alpha=.16)
    fig.supxlabel('Actual optimizer updates · solid hard / dotted independent noisy bits / dashed frozen physical mode');fig.supylabel('TEST memory T');fig.suptitle('Exact readout at every registered depth (× = undefined constant code)');axs[0,0].legend([Line2D([],[],color=c) for c in COLORS],list(map(str,SEEDS)),loc='upper left',fontsize=7,frameon=False);save(fig,out,'depth-and-noise')
    # Show the actual category assignment, not an aligned or post-selected filter.
    support=pd.read_csv(reference/'state-support.csv');mass=support[support['index']==9].sort_values('state').train_mass.to_numpy()
    fig,axes=plt.subplots(1,3,figsize=(10,3.4),sharey=True,constrained_layout=True)
    with np.load(out/'code-maps.npz') as maps:
        for ax,seed in zip(axes,SEEDS):
            d=final[(final['index']==9)&(final.seed==seed)&(final.channel=='hard')]
            ax.set(xlabel='Frozen two-bit code',xticks=range(4),xticklabels=['00','01','10','11'],yticks=range(3),yticklabels=['vacancy','A','B'])
            if d.empty:ax.set_title(f'Seed {seed} · not reached');continue
            r=d.iloc[0];q=maps[f'b09-s{seed}_d{r.depth}_hard_Q'];cm=np.stack([np.average(q[X[:,0]==k],axis=0,weights=mass[X[:,0]==k]) for k in range(3)])
            ax.imshow(cm,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_title(f'Seed {seed} · {int(r.output_support)} codes')
            for (i,j),v in np.ndenumerate(cm):ax.text(j,i,f'{v:.0%}',ha='center',va='center',color='white' if v>.55 else '#1d2730')
    axes[0].set_ylabel('Central species');fig.suptitle('β=3.05 · which code does each central species receive?');fig.text(.02,-.045,'All 210 states, weighted by the exact TRAIN marginal. Rows sum to 100% before rounding. Code labels are not matched across seeds.',fontsize=9);save(fig,out,'cold-code-species')


def coverage(out,cv,betas):
    from matplotlib.colors import ListedColormap,BoundaryNorm
    levels=['Unreached','Invalid map','Constant hard','Finite hard'];cmap=ListedColormap(['#dddddd','#c13f40','#e9ac57','#288586']);fig,ax=plt.subplots(figsize=(11.2,4.3),constrained_layout=True);grid=np.zeros((10,15),int)
    for (b,s),g in cv.groupby(['index','seed']):
        for j,r in enumerate(g.sort_values('depth').itertuples()):grid[b,5*SEEDS.index(s)+j]=0 if r.checkpoint=='unreached' else 1 if r.checkpoint=='invalid' else 2 if r.hard_codes<=1 else 3
    im=ax.imshow(grid,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,4.5,1),4),aspect='auto');ax.set(yticks=range(10),yticklabels=[f'{v:.3f}' for v in betas],xticks=range(15),xticklabels=['850','1700','3400','6800','final']*3,ylabel='β',xlabel='Depth within seed · 6101 / 6102 / 6103',title='Coverage · 10 temperatures × 3 starts × 5 locked checkpoints');ax.axvline(4.5,color='white',lw=3);ax.axvline(9.5,color='white',lw=3);cb=fig.colorbar(im,ax=ax,ticks=range(4),shrink=.7);cb.ax.set_yticklabels(levels);save(fig,out,'coverage')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--exact-only',action='store_true');p.add_argument('--reference',type=Path);a=p.parse_args()
    if a.reference is None or a.exact_only:physical(a.reference or a.out)
    if not a.exact_only:
        cv=pd.read_csv(a.out/'coverage.csv');betas=cv.drop_duplicates('index').beta.to_numpy();coverage(a.out,cv,betas)
        if cv.checkpoint.eq('finite').any():learned(a.out,a.reference or a.out)
