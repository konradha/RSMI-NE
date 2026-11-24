"""07: read locked forecast parameters; predict without access to any interior or exact energy as a feature."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import t,spearmanr
import stock_glass_modal as stock
from stock_glass_forecast import OUT,U,OPP,CHANNELS,LAM,SEEDS,VIEWS,load,classify

LOG2=np.log(2.)

def avg(x):
    x=np.asarray(x,dtype=float);m=x.mean();n=x.size;h=t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
    return float(m),float(m-h),float(m+h)

def put(row,key,x):
    for suf,y in zip(('','_lo','_hi'),avg(x)):row[key+suf]=y

def wilson(k,n):
    if not n:return (np.nan,)*3
    z=1.9599639845;p=k/n;mid=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return p,max(0.,mid-h),min(1.,mid+h)

def entropy(q):return -xlogy(q,q).sum(axis=-1)/LOG2

def calibration(rows,meta,kind,forecast,actual,lower):
    cuts=np.linspace(lower,1.,11);ii=np.clip(np.searchsorted(cuts,forecast,side='right')-1,0,9)
    for j in range(10):
        ix=ii==j;r=dict(**meta,kind=kind,bin=j,low=cuts[j],high=cuts[j+1],n=ix.sum(),total=len(actual));
        if ix.any():r['predicted']=forecast[ix].mean();put(r,'observed',actual[ix]);put(r,'bias',forecast[ix]-actual[ix]);
        rows.append(r)

def compare(rows,base,kind,models,links):
    for poorer,better in links:
        row=dict(**base,metric=kind,baseline=poorer,forecast=better);put(row,'reduction',models[poorer]-models[better]);rows.append(row)

def score(banks,out):
    locks=json.loads((out/'fit-lock.json').read_bytes());assert len(locks)==10
    assert all(stock.sha(out/name)==sha for name,sha in locks.items())  # verify entire TRAIN fit set before opening any TEST target
    scores=[];contr=[];support=[];rel=[];slices=[];examples=[];tail=[];cv=[];heldhash={};max_reconstruction=0.
    for b in range(10):
        tag=f'b{b:02}';beta,ids,j,rot,xx,zz,qs,masks=load(b,'test',banks);data={'parent_ids':ids,'selected_roster_rows':j,'rotation':rot,'seeds':np.array(SEEDS),'exact_q':qs,'views':np.array(['prior',*VIEWS])};outputs=[];strengths=[]
        with np.load(out/f'fit-{tag}.npz') as m:
            train_ids=m['train_ids'];ns=len(ids);tr=len(train_ids);ii=np.arange(ns)
            assert {tuple(i) for i in ids}.isdisjoint(map(tuple,train_ids)) and len(np.unique(m['rows']))==tr
            for seed,q,mask in zip(SEEDS,qs,masks):
                k=f's{seed}_';base=dict(index=b,beta=beta,seed=seed,ntrain=tr,ntest=ns);prior=m[k+'prior'];s0=float(m[k+'mean_s']);s=(q*q).sum(1);h=entropy(q);marg=q.mean(0);info=entropy(marg)-h.mean();coll=float(marg@marg);true_top=q.argmax(1);assert np.all(q[:,~mask]==0) and np.all(prior[mask]>0)
                supp=dict(**base,active=mask.sum(),s_sd=s.std(ddof=1),prior_collision=prior@prior,held_marginal_collision=coll,marginal_entropy_bits=entropy(marg),oracle_information_bits=info,training_prior_mismatch_bits=float((marg[mask]*np.log2(marg[mask]/prior[mask])).sum()),ambiguous=int((s<.8).sum()))
                put(supp,'true_entropy_bits',h);put(supp,'true_s',s);put(supp,'ambiguous_fraction',s<.8);kl=-q[:,mask]@np.log2(marg[mask])-h;put(supp,'oracle_information_bits',kl);support.append(supp)
                assert abs(info-supp['oracle_information_bits'])<3e-14;loss={};err={};probs=[];spreds=[]
                for view,x in [('prior',None),*zip(VIEWS,xx)]:
                    meta=dict(**base,view=view);prefix=k+view+'_'
                    if view=='prior':pr=np.broadcast_to(prior,q.shape);lp=np.broadcast_to(np.log(prior[mask]),(ns,mask.sum()));st=np.full(ns,s0);cut=s0
                    else:
                        pr,lp=classify(x,m[prefix+'W'],mask);v=m[prefix+'v'];st=np.clip(x@v[:-1]+v[-1],.25,1.);cut=float(m[prefix+'low_strength_cut']);cvmat=m[prefix+'cv_sums'];selected=m[prefix+'lambda'];opts=m[prefix+'convergence'];fold=m['fold'];
                        for a,l in enumerate(LAM):
                            for f in range(3):
                                nv=np.count_nonzero(fold==f);tmp=opts[(opts[:,0]==f)&(opts[:,1]==a)];cv.append(dict(**meta,fold=f,parents=nv,lambda_=l,label_choice=l==selected[0],strength_choice=l==selected[1],valid_bits=cvmat[f,a,0]/(LOG2*nv),valid_mse=cvmat[f,a,1]/nv,iterations=int(tmp[0,2]) if len(tmp) else 0,max_gradient=tmp[0,3] if len(tmp) else 0))
                    if mask.sum()==1:
                        assert np.max(abs(st-1))<1e-10;st=np.ones(ns);cut=1.
                    loss[view]=-(q[:,mask]*lp).sum(1)/LOG2;err[view]=(st-s)**2;top=pr.argmax(1);cf=pr.max(1);mass=q[ii,top];pp=(pr*pr).sum(1);r=dict(**meta)
                    assert np.isfinite(loss[view]).all() and np.min(loss[view]-h)>-1e-11 and np.allclose(pr.sum(1),1.)
                    put(r,'bits',loss[view]);put(r,'gap_to_oracle_bits',loss[view]-h);put(r,'brier',((pr-q)**2).sum(1));put(r,'expected_modal_mass',mass);put(r,'true_modal_match',top==true_top);put(r,'predicted_modal_confidence',cf);put(r,'strength_mse',err[view]);put(r,'strength_bias',st-s);put(r,'plugin_collision_bias',pp-s);r['pearson_s']=r['spearman_s']=np.nan
                    if st.std()>1e-10 and s.std()>1e-10:r['pearson_s']=np.corrcoef(st,s)[0,1];r['spearman_s']=spearmanr(st,s).statistic
                    scores.append(r);calibration(rel,meta,'label',cf,mass,0);calibration(rel,meta,'strength',st,s,.25);probs.append(pr);spreds.append(st)
                    if b==9:
                        hit=st<cut;event=s<.8;tp=np.sum(hit&event);trw=dict(**meta,cut=cut,events=event.sum(),flagged=hit.sum(),true_positives=tp,recall_supported=event.sum()>=30 and (~event).sum()>=30,precision_supported=event.sum()>=30 and (~event).sum()>=30 and hit.sum()>=30)
                        for name,n in [('recall',event.sum()),('precision',hit.sum())]:
                            val=wilson(tp,n) if trw[name+'_supported'] else [np.nan]*3
                            for t1,t2 in zip(('','_lo','_hi'),val):trw[name+t1]=t2
                        tail.append(trw)
                    if view=='directional':
                        for fr,c in zip((.1,.5,.9),m[prefix+'quantiles']):
                            ix=np.argmin(abs(cf-c));examples.append(dict(**meta,quantile=fr,train_conf=c,test_conf=cf[ix],id_seed=str(ids[ix,0]),id_chain=str(ids[ix,1]),roster_row=j[ix],prediction=top[ix],true_mode=true_top[ix],true_strength=s[ix],**{f'p{y}':pr[ix,y] for y in range(4)},**{f'q{y}':q[ix,y] for y in range(4)}))
                        for label,(term,sgn,train_corr) in enumerate(m[k+'slices']):
                            ix,ch=divmod(int(term),len(CHANNELS));u,v=OPP[ix];d=sgn*(zz[:,u,ch]-zz[:,v,ch]);r=dict(**meta,label=label,position=str(tuple(U[u])),opposite=str(tuple(U[v])),channel=CHANNELS[ch],train_sign=int(sgn),zero=int((d==0).sum()),active_label=mask[label]);parts=[]
                            for nm,ix in [('positive',d>0),('negative',d<0)]:
                                r[nm+'_n']=ix.sum();parts.append(q[ix,label]);
                                if ix.any():put(r,nm+'_q',q[ix,label]);r[nm+'_pred']=pr[ix,label].mean()
                            r['supported']=all(len(a)>=50 for a in parts) and mask[label] and sgn!=0
                            if r['supported']:
                                a,c=parts;delta=a.mean()-c.mean();a2=a.var(ddof=1)/len(a);c2=c.var(ddof=1)/len(c);se=np.sqrt(a2+c2);df=(a2+c2)**2/(a2*a2/(len(a)-1)+c2*c2/(len(c)-1)) if se>0 else np.inf;hw=t.ppf(.975,df)*se;r.update(delta=delta,delta_lo=delta-hw,delta_hi=delta+hw)
                            slices.append(r)
                loss['oracle']=h;links=(('prior','packing'),('packing','directional'),('rotated','directional'),('prior','directional'));compare(contr,base,'logloss_bits',loss,links);compare(contr,base,'strength_mse',err,links);outputs.append(np.stack(probs));strengths.append(np.stack(spreds));gap=(loss['prior']-h).mean()-info-supp['training_prior_mismatch_bits'];max_reconstruction=max(max_reconstruction,abs(gap))
        data.update(label_forecasts=np.stack(outputs),strength_forecasts=np.stack(strengths));name=f'held-{tag}.npz';np.savez_compressed(out/name,**data);heldhash[name]=stock.sha(out/name);print('SCORED',b,tr,ns,flush=True)
    for name,rows in [('held-scores',scores),('paired-gains',contr),('support',support),('calibration',rel),('descriptors',slices),('representatives',examples),('ambiguous-tail',tail),('training-selection',cv)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False,float_format='%.13g')
    assert max_reconstruction<1e-11;(out/'held-receipt.json').write_text(json.dumps(dict(files=heldhash,fit_lock=stock.sha(out/'fit-lock.json'),design=stock.sha(out/'design.json'),entropy_identity_max_error=max_reconstruction,new_trajectories=0,new_rsmi_fits=0,modal_usd=0,started_08=False),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--banks',required=True,type=Path);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();score(a.banks,a.out)
