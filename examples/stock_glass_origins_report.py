"""CPU-only report from the immutable experiment-01 packets. No training or archive TEST access."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import t

from stock_glass_origins import DEPTHS, SEEDS, LEVELS, PARENTS, ROWS, BATCHES


def interval(vals, n):
    w=n/n.sum(); mu=float(w@vals); m=len(n)
    se=np.sqrt(m/(m-1)*np.sum((w*(vals-mu))**2)); half=t.ppf(.975,m-1)*se
    return (mu,mu-half,mu+half)


def packets(src,manifest):
    fits={}; scores=[]; pairs=[]; coverage=[]; batches=[]
    for b in range(10):
        beta=float(manifest['coverage'][b]['beta'])
        for s in SEEDS:
            for k in LEVELS:
                stem=f'b{b:02}-k{k}-s{s}';f=src/(stem+'.npz');err=src/(stem+'-failure.json');fits[b,s,k]=None
                if f.exists():
                    with np.load(f,allow_pickle=False) as z:
                        rec=json.loads(str(z['receipt']));assert (rec['index'],rec['seed'],rec['origins'])==(b,s,k)
                        filters={d:z[f'enc_{d}'] for d in DEPTHS if f'enc_{d}' in z};trace=z['train'].copy();fits[b,s,k]=(rec,trace,filters)
                    steps=rec['steps'];status=rec['status'];checks=rec['checkpoints'];examples=rec['examples']
                    assert examples==(steps//BATCHES)*ROWS+min((steps%BATCHES)*500,ROWS)
                    assert rec['design']==manifest['design'] and rec['train_identity']==manifest['coverage'][b]['train_identity'] and rec['test_identity']==manifest['coverage'][b]['test_identity']
                    # Positive coverage is exact after one complete no-drop epoch; never infer it from anchor draws.
                    parents=PARENTS if steps>=BATCHES else np.nan; origin_total=k*parents
                else:
                    status='partial / execution failure' if err.exists() else 'unrun'; steps=examples=parents=origin_total=np.nan;checks={}
                coverage.append(dict(index=b,beta=beta,seed=s,k=k,status=status,actual_updates=steps,actual_examples=examples,train_parents=parents,actual_unique_parent_origins=origin_total,eligible_test_parents=PARENTS,file=f.name if f.exists() else '',error=json.loads(err.read_bytes())['error'] if err.exists() else ''))
                for d in DEPTHS:
                    q=checks.get(str(d));valid=q is not None and q.get('valid',False);n=np.array([]);gain=np.array([]);score_status=q['status'] if q else 'unreached: '+status
                    record=dict(index=b,beta=beta,seed=s,k=k,depth=d,checkpoint_status=score_status,fit_status=status,fit_updates=steps,fit_examples=examples,checkpoint_updates=d if q else np.nan,checkpoint_examples=(d//BATCHES)*ROWS+min((d%BATCHES)*500,ROWS) if q else np.nan,test_scored=q.get('test_parents',0) if q else 0,train_parents=PARENTS if q else np.nan)
                    for j in ('gain','real','shuffle'):record.update({j:np.nan,j+'_lo':np.nan,j+'_hi':np.nan})
                    if q and 'batches' in q:
                        v=q['batches'];n=np.array([x['n'] for x in v]);assert n.tolist()==[500]*16+[192] and record['test_scored']==PARENTS
                        record['entropy_bits']=q['entropy_bits'];record.update({f'hard_{j:02b}':v for j,v in enumerate(q['counts'])});assert sum(q['counts'])==PARENTS
                        for i,x in enumerate(v): batches.append(dict(index=b,seed=s,k=k,depth=d,batch=i,**x))
                        for j in ('gain','real','shuffle'):
                            vals=np.array([x.get(j,np.nan) for x in v]);record['raw_'+j]=n@vals/PARENTS
                            if valid:record[j],record[j+'_lo'],record[j+'_hi']=interval(vals,n)
                        record['max_stock_ref_error_nats']=max((max(abs(x['real']-x['real64']),abs(x['shuffle']-x['shuffle64'])) for x in v if 'real' in x),default=np.nan)
                    scores.append(record)
            for d in DEPTHS:
                for ka,kb in ((8,1),(4,1),(8,4)):
                    la,lb=fits[b,s,ka],fits[b,s,kb]; qa=la[0]['checkpoints'].get(str(d),{}) if la else {};qb=lb[0]['checkpoints'].get(str(d),{}) if lb else {}
                    eligible=qa.get('valid',False) and qb.get('valid',False);mean=lo=hi=np.nan
                    if eligible:
                        va,vb=qa['batches'],qb['batches']; assert [(x['n'],x['perm']) for x in va]==[(x['n'],x['perm']) for x in vb]; n=np.array([x['n'] for x in va]);mean,lo,hi=interval(np.array([x['gain']-y['gain'] for x,y in zip(va,vb)]),n)
                    pairs.append(dict(index=b,beta=beta,seed=s,depth=d,high=ka,low=kb,paired_parents=PARENTS if eligible else 0,D=mean,lo95=lo,hi95=hi,high_status=qa.get('status','unreached'),low_status=qb.get('status','unreached')))
    return fits,pd.DataFrame(scores),pd.DataFrame(pairs),pd.DataFrame(coverage),pd.DataFrame(batches)


COLORS={1:'#777777',4:'#009E73',8:'#0072B2'}
SEEDSTYLE=('-', '--', ':')
SYMBOLS=('o','s','^')


def figures(out,fits,sc,pa,co):
    bs=sc.groupby('index').beta.first().to_numpy(); final=sc[sc.depth==DEPTHS[-1]]; main=pa[(pa.depth==DEPTHS[-1])&(pa.high==8)&(pa.low==1)]
    def save(fig,stem):
        for ext in ('png','pdf'):fig.savefig(out/f'{stem}.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for j,s in enumerate(SEEDS):
        a=final[(final.k==8)&(final.seed==s)];p=main[main.seed==s]; shift=(j-1)*.030
        for ax,df,key,lo,hi in ((axes[0],a,'gain','gain_lo','gain_hi'),(axes[1],p,'D','lo95','hi95')):
            ok=df[np.isfinite(df[key])];ax.errorbar(ok.beta+shift,ok[key],yerr=np.stack([ok[key]-ok[lo],ok[hi]-ok[key]]),ls='none',marker=SYMBOLS[j],capsize=2,label=f'seed {s}');ax.axhline(0,color='#888',lw=.5)
        for k in LEVELS:
            a=final[(final.k==k)&(final.seed==s)];axes[2].plot(a.beta,a.get('entropy_bits',pd.Series(np.nan,index=a.index)),SEEDSTYLE[j],c=COLORS[k],lw=1,marker=SYMBOLS[j],ms=3,label=f'K{k}' if j==0 else None)
    axes[0].set_title('Final frozen TEST · 10,032 optimizer updates · pointwise 95% paired-batch intervals');axes[0].set_ylabel('8 origins: true − shuffle\n(nats)');axes[1].set_ylabel('8 − 1 origin gain\n(nats)');axes[2].set_ylabel('Hard-code entropy\n(bits, out of 2)');axes[0].legend(ncol=3);axes[2].legend(ncol=3);axes[2].set_xticks(bs,[f'{v:.2f}' for v in bs]);axes[2].set_xlabel('Inverse temperature β');save(fig,'held-final')
    state_labels=['Unrun','Partial / execution failure','Stock nonfinite','Invalid frozen score','Complete'];state_colors=['#e3e3e3','#e69f00','#b74542','#a279b2','#24876c'];mat=np.zeros((9,10),int)
    for r in co.itertuples():
        typ=0 if r.status=='unrun' else 4 if r.status=='complete' else 3 if 'readout' in r.status else 2 if 'nonfinite' in r.status else 1;mat[3*SEEDS.index(r.seed)+LEVELS.index(r.k),r.index]=typ
    fig,ax=plt.subplots(figsize=(12,6.5),layout='constrained');ax.imshow(mat,cmap=ListedColormap(state_colors),norm=BoundaryNorm(np.arange(-.5,5),5),aspect='auto');ax.set_xticks(range(10),[f'{v:.3f}' for v in bs]);ax.set_yticks(range(9),[f'{s} · K{k}' for s in SEEDS for k in LEVELS]);ax.set_xlabel('β');ax.set_title('Full fit coverage · numbers are actual Adam updates');
    for r in co.itertuples():
        ax.text(r.index,3*SEEDS.index(r.seed)+LEVELS.index(r.k),f'{int(r.actual_updates):,}' if pd.notna(r.actual_updates) else '—',ha='center',va='center',fontsize=8,color='white' if r.status in ('complete','nonfinite','invalid stock readout') else 'black')
    ax.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(state_colors,state_labels)],loc='upper center',bbox_to_anchor=(.5,-.10),ncol=5,fontsize=8);save(fig,'coverage')
    with PdfPages(out/'learning-curves.pdf') as pdf:
        for b in range(10):
            fig,ax=plt.subplots(3,3,figsize=(13,8),layout='constrained',sharex='col');fig.suptitle(f'β={bs[b]:.3f} · TRAIN traces and prespecified frozen TEST readouts')
            for j,s in enumerate(SEEDS):
                ax[0,j].set_title(f'seed {s}')
                for k in LEVELS:
                    run=fits[b,s,k];a=sc[(sc['index']==b)&(sc.seed==s)&(sc.k==k)];color=COLORS[k]
                    if run:
                        y=run[1]; x=1+132*np.arange(len(y));ax[0,j].plot(x,y,c=color,lw=.8,alpha=.55);ax[0,j].plot(x,pd.Series(y).ewm(span=20).mean(),c=color,lw=1.5,label=f'K{k}')
                    ax[1,j].plot(a.depth,a.gain,c=color,marker='o',ms=3);ax[1,j].fill_between(a.depth.to_numpy(),a.gain_lo.to_numpy(),a.gain_hi.to_numpy(),color=color,alpha=.13)
                for ka,kb,cl in ((8,1,'#0072B2'),(4,1,'#009E73'),(8,4,'#CC79A7')):
                    p=pa[(pa['index']==b)&(pa.seed==s)&(pa.high==ka)&(pa.low==kb)];ax[2,j].plot(p.depth,p.D,c=cl,marker='.',label=f'{ka}−{kb}');ax[2,j].fill_between(p.depth.to_numpy(),p.lo95.to_numpy(),p.hi95.to_numpy(),color=cl,alpha=.12)
                for i in range(3):ax[i,j].axhline(0,c='#888',lw=.5)
                ax[2,j].set_xlabel('Actual optimizer updates');ax[2,j].set_xlim(0,10200)
            ax[0,0].set_ylabel('TRAIN InfoNCE (nats)');ax[1,0].set_ylabel('TEST gain (nats)');ax[2,0].set_ylabel('Paired TEST difference (nats)');ax[0,0].legend(fontsize=8);ax[2,0].legend(fontsize=8);pdf.savefig(fig)
            if b in (0,9):fig.savefig(out/f'learning-b{b:02}.png',dpi=135)
            plt.close(fig)
    weights=[run[2][DEPTHS[-1]] for (b,s,k),run in fits.items() if b in (0,9) and run and DEPTHS[-1] in run[2] and np.isfinite(run[2][DEPTHS[-1]]).all()]; lim=max((np.max(np.abs(w)) for w in weights),default=1)
    with PdfPages(out/'hot-cold-filters.pdf') as pdf:
        for b in (0,9):
            for s in SEEDS:
                fig,ax=plt.subplots(4,3,figsize=(10,5),layout='constrained');im=None;fig.suptitle(f'β={bs[b]:.3f} · seed {s} · signed final 3D filters, fixed scale')
                for j,k in enumerate(LEVELS):
                    run=fits[b,s,k];w=run[2].get(DEPTHS[-1]) if run else None
                    for i in range(4):
                        a,h=divmod(i,2);panel=ax[i,j];panel.set_xticks([]);panel.set_yticks([])
                        if i==0: panel.set_title(f'{k} origin'+('' if k==1 else 's'))
                        if j==0: panel.set_ylabel(f'{"AB"[a]} → H{h+1}',rotation=0,ha='right')
                        if w is None:panel.text(.5,.5,'No final checkpoint',ha='center',transform=panel.transAxes,fontsize=8);continue
                        tile=np.full((3,11),np.nan);tile[:,:3]=w[0,:,:,a,h];tile[:,4:7]=w[1,:,:,a,h];tile[:,8:]=w[2,:,:,a,h];im=panel.imshow(tile,origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
                        if i==3:panel.set_xticks([1,5,9],['z=−1','z=0','z=+1'])
                if im is not None:fig.colorbar(im,ax=ax.ravel().tolist(),shrink=.6,label='Signed weight');
                fig.supxlabel('xy slices; x right, y up. No sign or component alignment.',fontsize=9);pdf.savefig(fig)
                if s==SEEDS[0]:fig.savefig(out/f'filters-b{b:02}.png',dpi=135)
                plt.close(fig)


def comparison(out,fits,sc,pa):
    # Matched estimates of change, never SITE-level standard errors or seed pooling.
    final=sc[sc.depth==DEPTHS[-1]];paired=pa[(pa.depth==DEPTHS[-1])&(pa.high==8)&(pa.low==1)]
    complete=final.pivot(index=['index','seed'],columns='k',values='gain')
    early=sc[(sc.k==8)&(sc.depth==1716)].set_index(['index','seed']).gain
    matched=complete.join(early.rename('early8')).dropna(); rows=[]
    for (b,s),g in matched.iterrows():
        rows.append(dict(index=int(b),seed=int(s),effect_8_minus_1=g[8]-g[1],effect_8_minus_4=g[8]-g[4],depth_final_minus_1716_K8=g[8]-g.early8))
    varying=pd.DataFrame(rows,columns=['index','seed','effect_8_minus_1','effect_8_minus_4','depth_final_minus_1716_K8']);varying.to_csv(out/'origin-depth-effects.csv',index=False)
    seed_spreads=[]
    for b,df in final[final.k==8].groupby('index'):
        vals=df.gain.dropna()
        if len(vals)==len(SEEDS):seed_spreads.append(dict(index=int(b),range_final_K8=float(vals.max()-vals.min()),sd_final_K8=float(vals.std())))
    pd.DataFrame(seed_spreads,columns=['index','range_final_K8','sd_final_K8']).to_csv(out/'seed-spread.csv',index=False)
    trends=[]
    for (b,s,k),run in fits.items():
        if not run or run[0]['steps']!=DEPTHS[-1]:continue
        y=run[1]
        if len(y)==76 and np.isfinite(y).all():
            trends.append(dict(index=b,seed=s,k=k,train_mean_epochs_57_66=float(y[56:66].astype('d').mean()),train_mean_epochs_67_76=float(y[66:76].astype('d').mean())))
    pd.DataFrame(trends).to_csv(out/'last-train-trend.csv',index=False)
    def checks(b):
        a=final[(final['index']==b)&(final.k==8)].sort_values('seed');p=paired[paired['index']==b].sort_values('seed')
        assert a.seed.tolist()==p.seed.tolist()==list(SEEDS)
        return [{'seed':int(s),'gain':val(x),'gain_lo':val(l),'gain_hi':val(h),'D':val(d),'D_lo':val(dl),'D_hi':val(dh)} for s,x,l,h,d,dl,dh in zip(a.seed,a.gain,a.gain_lo,a.gain_hi,p.D,p.lo95,p.hi95)]
    def val(x):return float(x) if np.isfinite(x) else None
    hot,cold=checks(0),checks(9);hot_complete=all(r['gain_lo'] is not None and r['D_lo'] is not None for r in hot);cold_complete=all(r['gain_lo'] is not None and r['D_lo'] is not None for r in cold)
    hot_clean=hot_complete and all(max(r['gain_lo'],r['D_lo'])<=.01 for r in hot)
    present=cold_complete and all(min(r['gain_lo'],r['D_lo'])>.01 for r in cold)
    verdict=dict(cold=cold,hot=hot,cold_all_valid=cold_complete,hot_all_valid=hot_complete,clears_cold_margin=present,hot_control_clear=hot_clean,support=bool(hot_clean and present),complete_matched_rows=len(matched),median_absolute_effect={c:val(varying[c].abs().median()) for c in varying.columns[2:]},complete_seed_triplets=len(seed_spreads),median_three_seed_range_K8=val(np.median([x['range_final_K8'] for x in seed_spreads])) if seed_spreads else None,complete_traces_with_last10_above_previous10_by_point01=sum(r['train_mean_epochs_67_76']>r['train_mean_epochs_57_66']+.01 for r in trends),complete_train_traces=len(trends),adaptive_extension=False)
    (out/'decision.json').write_text(json.dumps(verdict,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    fits,sc,pa,co,bat=packets(a.results,json.loads(a.manifest.read_bytes()))
    for df,name in ((sc,'checkpoint-scores'),(pa,'paired-differences'),(co,'fit-coverage'),(bat,'test-batches')):df.to_csv(a.out/(name+'.csv'),index=False,float_format='%.10g')
    comparison(a.out,fits,sc,pa);figures(a.out,fits,sc,pa,co);print(co.status.value_counts().to_string());print('updates',co.actual_updates.sum(),'examples',co.actual_examples.sum())

if __name__=='__main__':main()
