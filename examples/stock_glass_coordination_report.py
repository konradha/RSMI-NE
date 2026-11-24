"""Experiment 02: CPU report of frozen stock weights/scores; never fit anything on TEST."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_dist, t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from stock_glass_coordination import EARLY, SEEDS, N, groups, code


def stat(v,p,K=N):
    if not len(v):return (np.nan,)*3
    cnt=np.bincount(p,minlength=K);g=np.count_nonzero(cnt);mu=float(np.mean(v)); resid=np.bincount(p,weights=v,minlength=K)-cnt*mu
    half=t.ppf(.975,g-1)*np.linalg.norm(resid)/len(v)*np.sqrt(g/(g-1)) if g>1 else np.nan
    return mu,mu-half,mu+half


def bal(correct,lower,p):
    if np.unique(lower).size!=2:return (np.nan,)*3
    influence=np.zeros(N);means=[];g=len(np.unique(p))
    for mask in (lower,~lower):
        v=correct[mask];mu=v.mean();means.append(mu);influence+=.5*np.bincount(p[mask],weights=v-mu,minlength=N)/len(v)
    avg=np.mean(means);half=t.ppf(.975,g-1)*np.linalg.norm(influence)*np.sqrt(g/(g-1)) if g>1 else np.nan
    return avg,avg-half,avg+half


def put(r,key,vals):r[key],r[key+'_lo'],r[key+'_hi']=vals


def sides(path,b,side):
    with np.load(path/f'b{b:02}-{side}.npz') as z:
        m=z['V'].sum(1,dtype='int16');p=z['parent'].astype('i4'); y=z['Y'].astype('i4');return p,y,m[:,0]-m[:,1],m.sum(1)


def strata(d):
    yield 'all',0,np.ones(len(d),bool)
    yield 'non-tie',0,d!=0
    for k in range(7):yield 'k',k,np.abs(d)==k
    for x in range(-6,7):yield 'd',x,d==x


def phys(side,b,beta,p,y,d):
    out=[]
    for kind,k,mask in strata(d):
        if kind not in ('k','d'):continue
        yp=y[mask];pp=p[mask];dd=d[mask];n=len(yp);isfav=np.where(dd==0,yp==1,np.where(dd<0,yp==1,yp==0));target=float(expit(4*beta*abs(k))) if kind=='k' else float(expit(-4*beta*k));v=isfav if kind=='k' else yp==1
        rare=(~isfav);g=len(np.unique(pp));anyr=len(np.unique(pp[rare]));rev=int(rare.sum());ci=stat(v,pp);parent_u=beta_dist.ppf(.975,anyr+1,g-anyr) if anyr<g else (1. if g else np.nan);denlo=max(1,n/g-49*np.sqrt(np.log(40)/(2*g))) if g else np.nan
        # Simultaneous 95% upper bound for the pair-weighted rare-event probability: at most
        # 50 rare pairs per active parent; independently bound cluster incidence and mean support.
        rare_u=min(1,50*parent_u/denlo) if g else np.nan
        lo,hi=(max(0,ci[1]),min(1,ci[2])) if g>1 else (np.nan,np.nan)
        if n and (v.all() or not v.any()):
            missing_u=min(1,50*beta_dist.ppf(.975,1,g)/denlo)
            lo,hi=(1-missing_u,1.) if v.all() else (0.,missing_u)
        r=dict(side=side,index=b,beta=beta,kind=kind,level=k,candidate_parents=N,support_parents=g,pairs=n,iA=int(yp.sum()),jA=int(n-yp.sum()),i_lower=int((dd<0).sum()),j_lower=int((dd>0).sum()),favorable_or_tie_iA=int(isfav.sum()),reverse_or_tie_jA=rev,parents_with_reverse=anyr,expected_reverses=n*float(expit(-4*beta*abs(k))),parent_any_reverse_upper=parent_u,rare_pair_upper=rare_u,empirical=ci[0],lo95=lo,hi95=hi,oracle=target)
        out.append(r)
    return out


def empirical(y,d,tot):
    def lookup(ix):return (np.sum(y[ix])+.5)/(np.sum(ix)+1)
    occ=np.array([lookup(tot==u) for u in range(13)]);rate=np.full(7,.5)
    for k in range(1,7):
        m=np.abs(d)==k;rate[k]=(np.sum(y[m]==(d[m]<0))+.5)/(m.sum()+1)
    return occ,rate,lookup(np.ones(len(y),bool))


def components(u,y,grouping):
    # Exact row contributions on the SAME groups; each matrix column is one of two labels.
    ix=np.arange(len(y));z=u.astype('d');z-=z.max(1,keepdims=True);out=np.empty(len(y));null=out.copy()
    for _,g,perm in grouping:
        a=y[g];c=np.bincount(a,minlength=2);v=z[g];den=np.logaddexp(v[:,0]+np.log(c[0]),v[:,1]+np.log(c[1]))-np.log(len(g)) if c.min() else v[:,int(c[1]>0)]
        out[g]=v[np.arange(len(g)),a]-den;null[g]=v[np.arange(len(g)),a[perm]]-den
    return out,null


def metrics(b,bt,seed,dep,p,y,d,tot,ck,z,grouping,oracle,emp):
    h=z[f'h{dep}'];codes=code(h);u=z[f's{dep}'];q=expit(u[:,1].astype('d')-u[:,0]);lower=d<0;rule=np.asarray(ck['rule']);pi=rule[codes];right=np.where(lower,pi,1-pi);occrate,empf,prior=emp;pred=occrate[tot];exact=expit(-4*bt*d);et=np.where(d<0,empf[abs(d)],np.where(d>0,1-empf[abs(d)],.5));loss=(q-y)**2;occ_loss=(pred-y)**2;law_loss=(exact-y)**2;emp_loss=(et-y)**2
    real,shuffle=components(u,y,grouping);oref,onull=oracle;stock_groups=z[f'g{dep}']; assert len(stock_groups)==len(grouping)
    # Keep the raw TFP (not reconstructed) batch values too. For an all-row mean their
    # pooled result must agree with the row decomposition within the preregistered tolerance.
    batches=[]
    for (block,g,_),v in zip(grouping,stock_groups):
        assert tuple(v[:2])==(block,len(g));batches.append(dict(index=b,beta=bt,seed=seed,depth=dep,block=block,n=len(g),true_stock=v[2],shuffle_stock=v[3],true64=v[4],shuffle64=v[5],numeric_valid=bool(v[6]),oracle_true=oref[g].mean(),oracle_shuffle=onull[g].mean()))
    out=[];pblocks=p//256;decision=(q>.5)+.5*(q==.5);accuracy=np.where(y==1,decision,1-decision)
    for kind,k,mask in strata(d):
        pp=p[mask]; block=pblocks[mask];n=mask.sum();r=dict(index=b,beta=bt,seed=seed,depth=dep,kind=kind,level=k,checkpoint_valid=ck['valid'],n=n,support_parents=len(np.unique(pp)),iA=int(y[mask].sum()),jA=int(n-y[mask].sum()),lower_i=int(lower[mask].sum()),lower_j=int((d[mask]>0).sum()))
        counts=np.bincount(codes[mask],minlength=4);f=counts[counts>0]/max(n,1);r['entropy_bits']=-np.sum(f*np.log2(f)) if n else np.nan;r.update({f'code_{j:02b}':v for j,v in enumerate(counts)})
        fav=np.where(d[mask]==0,y[mask]==1,y[mask]==(d[mask]<0));r.update(favorable_or_tie_iA=int(fav.sum()),reverse_or_tie_jA=int(n-fav.sum()))
        put(r,'pred_favorable_or_tie_iA',stat(np.where(d[mask]<=0,q[mask],1-q[mask]),pp))
        pos=mask&(d!=0);put(r,'sign_balacc',bal(right[pos],lower[pos],p[pos]));put(r,'sign_accuracy',stat(right[pos],p[pos]));put(r,'assignment_acc',stat(accuracy[mask],pp));put(r,'learned_Brier',stat(loss[mask],pp));put(r,'brier_skill_over_occ',stat((occ_loss-loss)[mask],pp));put(r,'brier_gap_to_law',stat((loss-law_loss)[mask],pp));put(r,'brier_gap_to_empirical',stat((loss-emp_loss)[mask],pp));put(r,'brier_skill_over_intercept',stat(((prior-y)**2-loss)[mask],pp));put(r,'true64',stat(real[mask],block,32));put(r,'shuffle64',stat(shuffle[mask],block,32));put(r,'true_minus_shuffle',stat((real-shuffle)[mask],block,32));put(r,'oracle_true',stat(oref[mask],block,32));put(r,'oracle_shuffle',stat(onull[mask],block,32));put(r,'oracle_minus_stock',stat((oref-real)[mask],block,32));put(r,'pred_iA',stat(q[mask],pp));put(r,'empirical_lookup_iA',stat(et[mask],pp));put(r,'exact_iA',stat(exact[mask],pp));out.append(r)
    return out,batches


def collect(inputs,results,out,only_support=False):
    meta=json.loads((inputs/'prepared.json').read_bytes());ph=[];learn=[];batch=[];status=[];traces={};weights={};pol=[]
    for b in range(10):
        bt=float(meta['coverage'][b]['beta']);trp,try_,trd,trtot=sides(inputs,b,'train');p,y,d,tot=sides(inputs,b,'test'); ph+=phys('train',b,bt,trp,try_,trd)+phys('test',b,bt,p,y,d)
        if only_support:continue
        ref=empirical(try_,trd,trtot);grouping=list(groups(p,b));assert np.array_equal(np.sort(np.concatenate([g for _,g,_ in grouping])),np.arange(len(y)))
        # Stable analytic log P(y|d), including cold minority classes, without clipping.
        a=-4*bt*d.astype('d');ou=np.stack([-np.logaddexp(0,a),-np.logaddexp(0,-a)],axis=1);oracle=components(ou,y,grouping);B=math.ceil(len(try_)/500);end=B*math.ceil(10000/B)
        for s in SEEDS:
            f=results/f'b{b:02}-s{s}.npz'; depths=(*EARLY,end)
            if not f.exists():
                for dep in depths:status.append(dict(index=b,beta=bt,seed=s,depth=dep,planned=end,status='execution failure' if (results/f'b{b:02}-s{s}-failed.json').exists() else 'unrun',train_pairs=len(try_),test_pairs=len(y)))
                continue
            with np.load(f,allow_pickle=False) as z:
                r=json.loads(str(z['receipt']));assert (r['index'],r['seed'],r['design'],r['final'],r['batches'])==(b,s,meta['design'],end,B);assert r['examples']==r['steps']//B*len(try_)+min((r['steps']%B)*500,len(try_));traces[b,s]=(z['trace'],B)
                for dep in depths:
                    ck=r['checkpoints'].get(str(dep));st=ck['status'] if ck else 'unreached: '+r['fit_status'];status.append(dict(index=b,beta=bt,seed=s,depth=dep,planned=end,steps=r['steps'],examples=r['examples'],status=st,fit_status=r['fit_status'],train_parents=r['train_usable'],train_pairs=len(try_),test_pairs=len(y),test_parents=ck.get('test_usable',0) if ck else 0,train_s=r['train_s'],uuid=r['uuid']))
                    if not ck:continue
                    weights[b,s,dep]=z[f'w{dep}'].copy()
                    if 'rule' in ck:pol.append(dict(index=b,beta=bt,seed=s,depth=dep,**{f'P_i_lower_code_{i:02b}':v for i,v in enumerate(ck['rule'])},train_dneg=json.dumps(ck['train_code_dneg']),train_dpos=json.dumps(ck['train_code_dpos'])))
                    if f'g{dep}' not in z:continue
                    xx,bb=metrics(b,bt,s,dep,p,y,d,tot,ck,z,grouping,oracle,ref);learn.extend(xx);batch.extend(bb)
    ph=pd.DataFrame(ph);ph.to_csv(out/'physical-law.csv',index=False,float_format='%.12g');
    if only_support:return ph
    for arr,name in [(status,'status'),(learn,'learned-channel'),(batch,'contrastive-groups'),(pol,'frozen-rules')]:pd.DataFrame(arr).to_csv(out/(name+'.csv'),index=False,float_format='%.10g')
    score_rows=[]
    for key,f in pd.DataFrame(batch).groupby(['index','beta','seed','depth']):
        sr=dict(zip(('index','beta','seed','depth'),key));sr['valid']=bool(f.numeric_valid.all())
        f=f.copy();f['gain']=f.true_stock-f.shuffle_stock;f['oracle_gain']=f.oracle_true-f.oracle_shuffle;f['gap']=f.oracle_true-f.true_stock
        for col in ('true_stock','shuffle_stock','gain','oracle_true','oracle_shuffle','oracle_gain','gap'):
            # Different microgroup sizes; aggregate signed residuals within the 32 parent blocks.
            n=f.n.to_numpy();x=f[col].to_numpy();mu=n@x/n.sum();dev=np.bincount(f.block,weights=n*(x-mu),minlength=32)/n.sum();delta=t.ppf(.975,31)*np.linalg.norm(dev)*np.sqrt(32/31);put(sr,col,(mu,mu-delta,mu+delta))
        sr['max_matrix_reduction_discrepancy']=max(abs(f.true_stock-f.true64).max(),abs(f.shuffle_stock-f.shuffle64).max());score_rows.append(sr)
    pd.DataFrame(score_rows).to_csv(out/'stock-vs-oracle.csv',index=False,float_format='%.10g')
    signed=np.tile(np.array([-1.,1.]),6);signed/=np.linalg.norm(signed);align=[]
    for (b,s,dep),w in weights.items():
        x=w.reshape(12,2);norm=np.linalg.norm(x,axis=0);cs=(signed@x)/np.where(norm>0,norm,np.nan)
        align.append(dict(index=b,seed=s,depth=dep,cosine_H1=cs[0],cosine_H2=cs[1],norm_H1=norm[0],norm_H2=norm[1]))
    pd.DataFrame(align).to_csv(out/'filter-direction.csv',index=False,float_format='%.10g')
    return ph,pd.DataFrame(learn),pd.DataFrame(status),traces,weights


def plot(out,ph,df,st,traces,weights):
    colors=['#0072B2','#D55E00','#009E73'];beta=st.groupby('index').beta.first().to_numpy()
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    def band(ax,x,data,key,color,label):
        y=data[key].to_numpy();ax.plot(x,y,'.-',c=color,ms=4,lw=1,label=label);ax.fill_between(x,data[key+'_lo'].to_numpy(),data[key+'_hi'].to_numpy(),color=color,alpha=.12)
    pk=ph[(ph.side=='test')&(ph.kind=='k')]
    fig,axes=plt.subplots(2,4,figsize=(13,6),sharex=True,sharey=True,layout='constrained')
    for k,ax in enumerate(axes.flat):
        if k==7:ax.set_axis_off();ax.text(.03,.5,'Points: archive TEST pairs\nIntervals: independent parents\nLine: exact Hamiltonian odds\nNo point: absent stratum',fontsize=11);continue
        g=pk[pk.level==k];ax.plot(g.beta,g.oracle,c='#333',label='Hamiltonian');ok=g[g.pairs>0];ax.errorbar(ok.beta,ok.empirical,yerr=np.stack([ok.empirical-ok.lo95,ok.hi95-ok.empirical]).clip(0),fmt='o',color=colors[0],ms=3,capsize=2);ax.set_title(f'|d|={k}'+(' · tie: i=A' if k==0 else ''));ax.set_ylim(-.035,1.035)
    fig.supxlabel('Inverse temperature β');fig.supylabel('Probability A occupies lower-m site (tie: i=A)');fig.suptitle('Two-site physical assignment law · all supported |d|');save(fig,'physical-calibration')
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained');mat=pk.pivot(index='level',columns='index',values='pairs').to_numpy();ax.imshow(np.log10(1+mat),cmap='Blues',aspect='auto')
    for r in pk.itertuples():ax.text(r.index,r.level,f'{r.reverse_or_tie_jA:,} / {r.pairs:,}' if r.pairs else 'no pairs',ha='center',va='center',color='white' if r.pairs>1000 else '#222',fontsize=8)
    ax.set_xticks(range(10),[f'{b:.2f}' for b in beta]);ax.set_yticks(range(7));ax.set_xlabel('β');ax.set_ylabel('|d|');ax.set_title('Reverse assignments / eligible TEST pairs · tie row: j=A / all ties');save(fig,'rare-support')
    ff=st[st.depth==st.planned];available=df.merge(ff[['index','seed','planned']],on=['index','seed']);final=available[available.depth==available.planned]
    with PdfPages(out/'assignment-calibration.pdf') as pdf:
        for b,bt in enumerate(beta):
            fg,(ap,ar)=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');fg.suptitle(f'β={bt:.3f} · final frozen predictions (sign rule independent of contrastive reduction)')
            phys_rows=ph[(ph.side=='test')&(ph.kind=='d')&(ph['index']==b)&(ph.pairs>0)]
            xx=np.arange(-6,7);ap.plot(xx,expit(-4*bt*xx),'k:',label='Hamiltonian')
            ap.errorbar(phys_rows.level,phys_rows.empirical,yerr=np.maximum(0,np.array([phys_rows.empirical-phys_rows.lo95,phys_rows.hi95-phys_rows.empirical])),color='black',fmt='.',capsize=2,label='Observed TEST')
            for j,s in enumerate(SEEDS):
                g=final[(final['index']==b)&(final.seed==s)&(final.kind=='d')&(final.n>0)];flag='*' if len(g) and not g.checkpoint_valid.all() else '';band(ap,g.level.to_numpy()+.04*(j-1),g,'pred_iA',colors[j],f'stock {s}{flag}')
                g=final[(final['index']==b)&(final.seed==s)&(final.kind=='k')&(final.level>0)&(final.n>0)];band(ar,g.level.to_numpy()+.04*(j-1),g,'sign_balacc',colors[j],f'seed {s}')
            ap.set(xlabel='Signed d = mᵢ − mⱼ',ylabel='P(i=A)',ylim=(-.05,1.05),xticks=np.arange(-6,7,2));ar.set(xlabel='|d| (ties have no lower site)',ylabel='Lower-site balanced accuracy',ylim=(-.05,1.05),xticks=np.arange(1,7));ar.axhline(.5,color='#777',ls=':');ap.legend(fontsize=8);ar.legend(fontsize=8);fg.supxlabel('* Invalid contrastive reduction: classifier diagnostic only; excluded from joint gate.',fontsize=9);pdf.savefig(fg)
            if b==9:fg.savefig(out/'cold-assignment.png',dpi=150)
            plt.close(fg)
    fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for j,s in enumerate(SEEDS):
        g=final[(final.seed==s)&(final.kind=='non-tie')];x=g.beta.to_numpy()+.018*(j-1)
        for a,key in zip(ax,('sign_balacc','brier_skill_over_occ','oracle_minus_stock')):
            y=g.copy()
            if key=='oracle_minus_stock':y.loc[~y.checkpoint_valid,[key,key+'_lo',key+'_hi']]=np.nan
            band(a,x,y,key,colors[j],f'{s}')
            if key!='oracle_minus_stock':
                bad=~g.checkpoint_valid;a.plot(x[bad],g.loc[bad,key],'x',c='black',ms=5)
    for a,thr in zip(ax,(.5,0,0)):a.axhline(thr,c='#555',ls=':',lw=1)
    ax[0].legend(ncol=3);ax[0].set_ylabel('Lower-site balanced\naccuracy');ax[1].set_ylabel('Brier improvement\nover occupancy-only');ax[2].set_ylabel('Exact − learned\npaired InfoNCE (nats)');ax[2].set_xlabel('β');fig.suptitle('Frozen non-tie TEST · 95% intervals · × invalid InfoNCE, classifier diagnostic only');save(fig,'operator-recovery')
    fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained');data=np.array([[{'complete':2,'invalid frozen score':1}.get(ff[(ff.seed==s)&(ff['index']==b)].status.iloc[0],0) for b in range(10)] for s in SEEDS]);ax.imshow(data,cmap=matplotlib.colors.ListedColormap(['#b74542','#b47513','#24876c']),vmin=0,vmax=2,aspect='auto')
    for r in ff.itertuples():ax.text(r.index,SEEDS.index(r.seed),f'{r.steps:,.0f}' if np.isfinite(r.steps) else 'unrun',ha='center',va='center',color='white',fontsize=9)
    ax.set_xticks(range(10),[f'{b:.2f}' for b in beta]);ax.set_yticks(range(3),SEEDS);ax.set_xlabel('β');ax.set_ylabel('Optimizer seed');ax.set_title('Actual Adam steps · green valid, amber invalid frozen critic, red no finite final');save(fig,'fit-status')
    with PdfPages(out/'learning.pdf') as pdf, PdfPages(out/'signed-d-atlas.pdf') as atlas:
        for b,bt in enumerate(beta):
            fig,axs=plt.subplots(4,3,figsize=(12,9),sharex='col',layout='constrained');fig.suptitle(f'β={bt:.3f} · stock TRAIN vs frozen held-parent calibration')
            for j,s in enumerate(SEEDS):
                axs[0,j].set_title(f'seed {s}');g=available[(available['index']==b)&(available.seed==s)&(available.kind=='non-tie')];x=g.depth.to_numpy()
                if (b,s) in traces:
                    y,B=traces[b,s];xx=1+B*np.arange(len(y));axs[0,j].plot(xx,y,c='#9abccb',lw=.8);axs[0,j].plot(xx,pd.Series(y).ewm(span=20).mean(),c=colors[0],lw=1.4)
                good=g.copy();good.loc[~good.checkpoint_valid,['true64','true64_lo','true64_hi','shuffle64','shuffle64_lo','shuffle64_hi']]=np.nan
                band(axs[1,j],x,good,'true64',colors[0],'true');band(axs[1,j],x,good,'shuffle64',colors[1],'shuffled');axs[1,j].plot(x,g.oracle_true,c='black',ls=':',label='exact true')
                band(axs[2,j],x,g,'sign_balacc',colors[2],'sign rule');axs[2,j].axhline(.5,c='#555',ls=':');axs[3,j].plot(x,g.entropy_bits,'.-',c=colors[0]);axs[3,j].set_xlabel('Actual optimizer updates')
                f,aa=plt.subplots(1,3,figsize=(12,6),layout='constrained');f.suptitle(f'β={bt:.3f} · seed {s} · signed d=mᵢ−mⱼ; grey = no eligible score')
                dep=st[(st['index']==b)&(st.seed==s)].depth.tolist()
                for a,key,title in zip(aa,('pred_iA','assignment_acc','sign_accuracy'),('Predicted P(i=A)','Assignment accuracy','Lower-site rule accuracy')):
                    rows=available[(available['index']==b)&(available.seed==s)&(available.kind=='d')];imdata=rows.pivot(index='level',columns='depth',values=key).reindex(index=range(-6,7),columns=dep);cm=plt.get_cmap('viridis').copy();cm.set_bad('#ddd');im=a.imshow(imdata,vmin=0,vmax=1,cmap=cm,aspect='auto',origin='lower');a.set_title(title);a.set_xticks(range(5),dep,rotation=55,fontsize=8);a.set_yticks(range(13),range(-6,7));a.set_ylabel('signed d');a.set_xlabel('Optimizer updates')
                f.colorbar(im,ax=aa,shrink=.6);atlas.savefig(f);plt.close(f)
            for a,l in zip(axs[:,0],('TRAIN InfoNCE\nraw & EMA20','TEST InfoNCE\n(nats)','Balanced sign\naccuracy','Code entropy\n(bits)')):a.set_ylabel(l)
            axs[1,0].legend(fontsize=8);pdf.savefig(fig)
            if b in (0,9):fig.savefig(out/f'learning-b{b:02}.png',dpi=135)
            plt.close(fig)
    finalw={(r.index,r.seed):weights[r.index,r.seed,r.planned] for r in ff.itertuples() if (r.index,r.seed,r.planned) in weights};lim=max((np.max(np.abs(w[np.isfinite(w)])) for w in finalw.values() if np.isfinite(w).any()),default=1)
    with PdfPages(out/'filters.pdf') as pdf:
        for s in SEEDS:
            fig,axes=plt.subplots(2,5,figsize=(14,5.5),layout='constrained');fig.suptitle(f'Signed final weights · stock seed {s} · shared scale; raw orientation, no sign or bit alignment')
            for b,a in enumerate(axes.flat):
                a.set_title(f'β={beta[b]:.3f}');w=finalw.get((b,s));a.set_yticks(range(4),['i H1','i H2','j H1','j H2']);a.set_xticks(range(6),['−z','+z','−y','+y','−x','+x'],fontsize=8)
                if w is None:a.text(.5,.5,'No final',ha='center',transform=a.transAxes);continue
                im=a.imshow(w.reshape(6,4).T,vmin=-lim,vmax=lim,cmap='RdBu_r',aspect='auto')
            if finalw:fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-lim,lim),cmap='RdBu_r'),ax=axes,shrink=.7,label='Weight')
            pdf.savefig(fig)
            if s==6101:fig.savefig(out/'filters-seed6101.png',dpi=135)
            plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    for key in ('inputs','results','out'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True);data=collect(a.inputs,a.results,a.out);plot(a.out,*data)
    print(data[2].groupby('status').size().to_string())

if __name__=='__main__':main()
