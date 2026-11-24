"""05: render frozen readouts and classify the registered cold endpoints; no fit or selection."""
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stock_glass_inventory_banks import OUT,SEEDS,SECTORS,SOURCE,stock

COLORS=['#007d9c','#c05f2f','#80569b']; POS=np.arange(len(SECTORS));GROUP={'223':'source','214':'A→B','232':'B→A','133':'vac→A','124':'vac→B','313':'A→vac','322':'B→vac'}
plt.rcParams.update({'axes.spines.right':False,'axes.spines.top':False,'font.size':10,'axes.titlesize':11,'savefig.dpi':160})


def export(fig,out,name):fig.savefig(out/(name+'.png'),bbox_inches='tight');plt.close(fig)


def cold_decisions(scores,cat,support,ordinary):
    results=[]
    for s in SECTORS:
        d=scores[(scores.sector==s)&(scores['index']==9)];q=cat[(cat.sector==s)&(cat['index']==9)];u=support[(support.sector==s)&(support['index']==9)].iloc[0];o=ordinary[(ordinary.sector==s)&(ordinary['index']==9)].set_index('observable');assert len(d)==len(q)==3 and d.seed.nunique()==q.seed.nunique()==3
        if not (d.scalar_status=='identified').all() or min(u.train_parents,u.test_parents)<100 or not np.isfinite(d.fraction).all():r='unsupported'
        elif (d.T_lo>.05).all() and (d.fraction_lo>.5).all():r='transfers'
        elif ((d.T_hi<=.05)|(d.fraction_hi<=.5)).any():r='threshold contradicted'
        else:r='unresolved'
        results.append(dict(sector=s,change=GROUP[s],class_= 'source' if s==SOURCE else 'A/B only' if s[0]=='2' else 'vacancy changing',train_parents=int(u.train_parents),test_parents=int(u.test_parents),states=int(u.states),T_min=d['T'].min(),T_max=d['T'].max(),fraction_min=d.fraction.min(),fraction_max=d.fraction.max(),min_T_lower95=d.T_lo.min(),min_fraction_lower95=d.fraction_lo.min(),exact_T=o.loc['oracle','T'],linear_T=o.loc['linear','T'],vacancy_T=o.loc['vacancy','T'],collision=u.state_coincidence,transfer=r,beats_linear_all=bool((d.minus_linear_lo>.01).all()),beats_random_categorical_mean_all=bool((q.true_minus_random_normalized_lo>.01).all()),mass_mismatch_seeds=int((abs(q.entropy_difference)>.1).sum()),empirical_random_ge_min=q.random_fraction_ge_true_normalized.min(),empirical_random_ge_max=q.random_fraction_ge_true_normalized.max()))
    return pd.DataFrame(results)


def plot(out):
    load=lambda n:pd.read_csv(out/(n+'.csv'),dtype={'sector':str});sc=load('frozen-scores');ca=load('categorical');ordinary=load('ordinary');sp=load('support');dec=cold_decisions(sc,ca,sp,ordinary);dec.to_csv(out/'cold-decisions.csv',index=False,float_format='%.13g');beta=sp[sp.sector==SOURCE].sort_values('index').beta.to_numpy();xt=[f'{b:.2f}' for b in beta];lab=[s+' '+GROUP[s] for s in SECTORS]
    def matrix(d,k):return d.pivot(index='sector',columns='index',values=k).reindex(SECTORS).to_numpy()
    def image(ax,data,minv,maxv,pal,show,label_x=False,label_y=False,fmt='.2f',signed=False):
        cmap=plt.get_cmap(pal).copy();cmap.set_bad('#e3e5e6');im=ax.imshow(data,aspect='auto',vmin=minv,vmax=maxv,cmap=cmap,interpolation='nearest');ax.set_xticks(range(10),xt if label_x else []);ax.set_yticks(POS,lab if label_y else []);ax.tick_params(length=0,pad=3);ax.axhline(.5,c='white',lw=1.4);ax.axhline(2.5,c='white',lw=1.4)
        if show:
            for (i,j),v in np.ndenumerate(data):
                rgb=cmap(im.norm(v))[:3] if np.isfinite(v) else (1,1,1);fg='#101616' if np.dot(rgb,[.2126,.7152,.0722])>.52 else 'white';ax.text(j,i,format(v,fmt) if np.isfinite(v) else '—',ha='center',va='center',color=fg,fontsize=6.5)
        return im
    fig,axs=plt.subplots(2,3,figsize=(15,6.2),constrained_layout=True)
    for j,seed in enumerate(SEEDS):
        d=sc[sc.seed==seed];axs[0,j].set_title(f'Seed {seed}');
        for i,col in enumerate(['T','fraction']):
            hi=1 if i==0 else max(1.01,sc.fraction.max());im=image(axs[i,j],matrix(d,col),0,hi,'viridis',True,i==1,j==0);axs[i,j].set_xlabel('β' if i==1 else '')
            if j==2:fig.colorbar(im,ax=axs[i,:],shrink=.8,label='T' if i==0 else 'T / target mode')
    fig.suptitle('One frozen scalar per β and seed, applied unchanged to seven inventories');fig.text(.01,-.022,'Top: held-out replica memory. Bottom: fraction of the target TRAIN-selected mode on the same TEST parents. Gray = undefined or unsupported, not zero.',fontsize=8);export(fig,out,'transfer-grid')
    fig,axs=plt.subplots(2,2,figsize=(12,7.2),constrained_layout=True);cold=sc[sc['index']==9];co=ordinary[ordinary['index']==9];cc=ca[ca['index']==9]
    def scatter(ax,d,field,offset,color,mark,label):
        v=d.set_index('sector').reindex(SECTORS);y=v[field].to_numpy();lo=v[field+'_lo'].to_numpy();hi=v[field+'_hi'].to_numpy();ax.errorbar(POS+offset,y,yerr=[y-lo,hi-y],ls='none',marker=mark,color=color,label=label,capsize=2,ms=4)
    for j,(sd,c) in enumerate(zip(SEEDS,COLORS)):
        d=cold[cold.seed==sd];q=cc[cc.seed==sd];label=str(sd);dx=(j-1)*.10;scatter(axs[0,0],d,'T',dx,c,'o',label);scatter(axs[0,1],d,'minus_linear',dx,c,'o',label);scatter(axs[1,0],q,'true_minus_random_normalized',dx,c,'o',label);axs[1,1].scatter(POS+dx,q.set_index('sector').loc[list(SECTORS),'entropy_difference'],s=18,c=c,label=label)
    for key,c,mrk,dx in [('oracle','#202930','x',-.23),('linear','#548837','+',.22),('vacancy','#a77c13','^',.30)]:scatter(axs[0,0],co[co.observable==key],'T',dx,c,mrk,key)
    for a,title,y in zip(axs.flat,['A   Absolute held-out memory','B   Is the scalar better than site-linear packing?','C   Do the labels beat the mean random partition?','D   Are label masses comparable?'],['T','Paired T − target site-linear T','Paired normalized collision excess difference','Learned entropy − random mean (nats)']):
        a.set(xticks=POS,xticklabels=SECTORS,title=title,ylabel=y,xlabel='Inventory (vac, A, B)');a.grid(alpha=.14);a.axvline(.5,color='gray',alpha=.25,ls=':');a.axvline(2.5,color='gray',alpha=.25,ls=':')
    for a in (axs[0,1],axs[1,0]):a.axhline(0,c='gray',lw=.8);a.axhline(.01,c='gray',ls='--',lw=.8)
    axs[1,1].axhspan(-.1,.1,color='#dce4dc',alpha=.5);axs[1,1].axhline(0,c='gray',lw=.8);axs[0,0].legend(ncol=3,frameon=False,fontsize=8);fig.suptitle('β=3.05 · transfer, packing and random specificity are different questions');fig.text(.02,-.025,'Intervals in A–C: pointwise parent t95, fixed fits and fixed nulls. D: descriptive mass mismatch. Sectors have distinct exteriors; do not pair points across columns.',fontsize=8);export(fig,out,'cold-contrasts')
    fig,axs=plt.subplots(2,3,figsize=(15,6.7),constrained_layout=True)
    true=ca.groupby(['sector','index'],as_index=False).normalized.mean();nn=ca.groupby(['sector','index'],as_index=False).random_mean_normalized.mean();lens=ca.groupby(['sector','index'],as_index=False).true_minus_random_normalized.mean()
    plots=[(matrix(true,'normalized'),0,1,'viridis','Mean learned label signal'),(matrix(nn,'random_mean_normalized'),0,1,'viridis','Mean random label signal'),(matrix(lens,'true_minus_random_normalized'),None,None,'RdBu_r','Learned minus random'),(matrix(sp,'state_coincidence'),0,1,'cividis','Whole-star conditional coincidence'),(matrix(sp,'train_effective_states'),0,sp.states.max(),'cividis','Effective marginal states (TRAIN)'),(matrix(sp,'test_parents'),None,8192,'cividis','Eligible TEST parents / 8,192')]
    for j,(z,lo,hi,cm,title) in enumerate(plots):
        if hi is None:hi=np.nanmax(abs(z));lo=-hi
        if lo is None:lo=np.nanmin(z)
        ax=axs.flat[j];im=image(ax,z,lo,hi,cm,False,j>=3,j%3==0);fig.colorbar(im,ax=ax,shrink=.78);ax.set_title(title);ax.set_xlabel('β' if j>=3 else '')
    fig.suptitle('Pinning, nulls and available support across inventories');fig.text(.015,-.02,'Top row descriptive mean across starts (not independent parents); undefined cells gray. Random assignments preserve unweighted state counts, not Gibbs masses.',fontsize=8);export(fig,out,'pinning-and-nulls')
    fig,axs=plt.subplots(3,3,figsize=(14,9),constrained_layout=True)
    for j,(seed,k,title) in enumerate(zip(SEEDS,('vacancy','linear','oracle'),('Center vacancy','Target site-linear','Target exact mode'))):
        d=sc[sc.seed==seed];h=ca[ca.seed==seed]
        for i,(data,lo,hi,pal,lab0) in enumerate([(matrix(ordinary[ordinary.observable==k],'T'),0,1,'viridis',title+' · T'),(matrix(d,'total'),0,sc.total.max(),'cividis',f'Seed {seed} · total scalar variance'),(matrix(h,'entropy'),0,np.log(4),'viridis',f'Seed {seed} · label entropy (nats)')]):
            ax=axs[i,j];im=image(ax,data,lo,hi,pal,False,i==2,j==0);ax.set_title(lab0)
            if j==2:fig.colorbar(im,ax=axs[i,:],shrink=.8)
    fig.suptitle('Ordinary physical contrasts and the marginal spread of the transferred observables');fig.supxlabel('β · each column is a distinct temperature, not a training depth');export(fig,out,'ordinary-and-marginals')
    print(dec.to_string(index=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=type(OUT),default=OUT);plot(ap.parse_args().out)
