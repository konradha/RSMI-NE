"""Matched affine challenger to the four-state equilibrium encoder. CPU reproduction:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/l20_operators_affine.py --data ~/datasets/lattice-glass/l20-16384-v1
Never rewrites the original models. Select on cold validation, then open index 7 first.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from l20_operators_static import (BATCH, EPOCHS, OUT as STATIC, SEEDS, TREPS, VREPS,
    backward, banks, codes, dense, dump, forward, init, netfile, prob, readfit, softmax, source)

W, BUFFER = 3, 1
ORDER = (7, 8, 9, 0, 1, 2, 3, 4, 5, 6)
METRICS = ('raw','shuffle','conditional_raw','conditional_shuffle')


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def posterior(v,model):
    m,enc,_=model
    if m['method']=='affine': return softmax(dense(v)@enc[0]+enc[1])
    return prob(v,W,m['spec'],enc)


def val_score(vbank,ebank,A,c,critic):
    totals=[]
    for v,e in zip(vbank,ebank):
        p=softmax(dense(v)@A+c); z=forward(dense(e),critic)[1].reshape(-1,BATCH,4).astype(float)
        z-=z.max(1,keepdims=True); r=z-np.log(np.exp(z).mean(1,keepdims=True))
        totals.append((p.reshape(r.shape)*r).sum(2).mean())
    return float(np.mean(totals))


def train_one(train,val,seed,out):
    # Consume the identical initializer RNG stream, retaining its exact corner
    # shortcut/bias, omitting rather than freezing a hidden branch.
    A,c=init(np.random.default_rng(seed),2*W**3,True,W,True)[-2:]
    initial=A.copy(); corner=np.flatnonzero(initial.any(1))[0]//2
    critic=init(np.random.default_rng(9601),2*train[1].shape[-1]); weights=[A,c]+critic
    mom=[np.zeros_like(a) for a in weights]; vel=[np.zeros_like(a) for a in weights]
    best=-np.inf; history=[]; n=0
    for epoch in range(EPOCHS+1):
        if epoch:
            xs=dense(train[0][epoch-1]); es=dense(train[1][epoch-1]); total=0.
            for j in range(0,len(xs),BATCH):
                x=xs[j:j+BATCH]; e=es[j:j+BATCH]; p=softmax(x@A+c); he,z=forward(e,critic)
                z-=z.max(0); ez=np.exp(z); r=z-np.log(ez.mean(0)); expected=(p*r).sum(1,keepdims=True)
                total+=expected.sum(); dp=-p*(r-expected)/BATCH
                dz=((ez/ez.sum(0))*p.sum(0)-p)/BATCH
                grad=[x.T@dp,dp.sum(0)]+backward(e,he,critic,dz)
                n+=1; rate=(.002 if epoch<=24 else .00065)*np.sqrt(1-.999**n)/(1-.9**n)
                for a,g,m,s in zip(weights,grad,mom,vel):
                    m*=.9; m+=.1*g; s*=.999; s+=.001*g*g
                    a-=rate*m/(np.sqrt(s)+1e-8)
        if epoch%3==0:
            val_nats=val_score(*val,A,c,critic)
            history.append(dict(epoch=epoch,validation=val_nats,training=float(total/len(xs)) if epoch else None))
            if val_nats>best: best=val_nats; selected=epoch; ckpt=[a.copy() for a in weights]
    m=dict(method='affine',width=W,buffer=BUFFER,seed=seed,epoch=selected,validation=best,history=history,
           initial_corner_zyx=(np.array(np.unravel_index(corner,(W,)*3))-1).tolist(),
           encoder='softmax(dense(V) @ A + c); no hidden representation or interaction coefficients')
    dump(out/f'seed-{seed}.json',m)
    np.savez_compressed(out/f'seed-{seed}.npz',A=ckpt[0],c=ckpt[1],initial_A=initial,
                        **{f'crit{i}':a for i,a in enumerate(ckpt[2:])})
    return m,ckpt[:2],ckpt[2:]


def score(vbank,ebank,model,strata):
    """Average positives equally. Each conditional candidate set has K distinct parents.

    Given stratum g and its count K in an independent batch, the K pairs are
    draws from P(V,E|g). Their own expected hard-H InfoNCE is a bound for that
    conditional law. Weight by K/B; do not equal-weight imbalanced strata.
    A singleton supplies exactly zero bound (and neutral null), not a duplicate.
    """
    nb=vbank.shape[1]//BATCH; out=np.zeros((nb,4)); counts=np.zeros(4,int)
    singleton=0; ksum=0; maxprob=0.; occup=np.zeros(4); samples=vbank.shape[0]*vbank.shape[1]
    for v,e,gg in zip(vbank,ebank,strata):
        g=gg.reshape(nb,BATCH); p=posterior(v,model).reshape(nb,BATCH,4).astype(float)
        occup+=np.bincount(p.argmax(2).ravel(),minlength=4); maxprob+=p.max(2).sum()
        z=forward(dense(e),model[2])[1].reshape(-1,BATCH,4).astype(float); z-=z.max(1,keepdims=True); ez=np.exp(z)
        r=z-np.log(ez.mean(1,keepdims=True)); other=(r.sum(1,keepdims=True)-r)/(BATCH-1)
        for j,value in enumerate((r,other)):out[:,j]+=(p*value).sum(2).mean(1)
        for h in range(4):
            mask=(g==h)[:,:,None]; k=mask.sum(1,keepdims=True); counts[h]+=k.sum(); singleton+=np.count_nonzero(k==1); ksum+=(k*k).sum()
            denom=(ez*mask).sum(1,keepdims=True)/np.maximum(k,1)
            r=z-np.log(np.where(k>0,denom,1.)); other=((r*mask).sum(1,keepdims=True)-r)/np.maximum(k-1,1)
            for j,value in enumerate((r,other),start=2):out[:,j]+=(p*value*mask).sum((1,2))/BATCH
    return out/len(vbank),dict(stratum_counts=counts.tolist(),singleton_positives=int(singleton),mean_K_per_positive=float(ksum/samples),
                              mean_max_p=float(maxprob/samples),modal_fractions=(occup/samples).tolist())


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)


def stats(x,name):
    m=x.mean(); h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return {name:m,name+'_lo':m-h,name+'_hi':m+h}


def plot(out,sc,delta):
    fig,axes=plt.subplots(1,3,figsize=(14.2,4.8),layout='constrained'); a,b,c=axes
    colors={'nonlinear':'#126e79','affine':'#905c9b','corners':'#bd6231'}
    for name,marker in zip(colors,('o','s','D')):
        rows=sorted([r for r in sc if r['model']==name],key=lambda r:r['beta']); col=colors[name]
        for ax,metric in ((a,'raw'),(c,'conditional_raw')):
            x=[r['beta'] for r in rows]; y=np.array([r[metric] for r in rows]); lo=[r[metric+'_lo'] for r in rows]; hi=[r[metric+'_hi'] for r in rows]
            ax.errorbar(x,y,yerr=[y-lo,hi-y],color=col,label=name,marker=marker,ms=3.8,lw=1.2,capsize=2)
    for j,(name,col) in enumerate((('nonlinear-minus-affine','#126e79'),('affine-minus-corners','#905c9b'))):
        rows=sorted([r for r in delta if r['comparison']==name],key=lambda r:r['beta']);x=[r['beta'] for r in rows]
        y=np.array([r['raw'] for r in rows]); b.errorbar(x,y,yerr=[y-[r['raw_lo'] for r in rows],[r['raw_hi'] for r in rows]-y],
           color=col,label=name.replace('-minus-',' − '),marker=['o','s'][j],ms=3.8,lw=1.2,capsize=2)
    confirm=next(r['beta'] for r in sc if r['index']==7)
    for ax,title,ylabel in zip(axes,('a   Frozen absolute bounds','b   Paired operator differences','c   Within opposite-corner strata'),
            ('Raw InfoNCE (nats)','Δ raw InfoNCE (nats)','Conditional raw InfoNCE (nats)')):
        ax.axvline(confirm,color='.55',lw=.8,ls='--');ax.axhline(0,color='.35',lw=.8,ls=':');ax.set(title=title,xlabel='Evaluation β',ylabel=ylabel)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Does the encoder need hidden interactions?  |  Dashed line: prespecified β=2.3722 confirmation',fontsize=15,weight='bold')
    fig.savefig(out/'affine-transfer.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--static',type=Path,default=STATIC);parser.add_argument('--out',type=Path,default=STATIC.parent/'affine');a=parser.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=True);selection=json.loads((a.static/'selection.json').read_bytes()); f=selection['flagship']; assert (f['width'],f['buffer'])==(W,BUFFER)
    fixed=selection['fixed_by_geometry'][f'{W}-{BUFFER}']; assert fixed['method']=='bits-body-1'
    nonlinear=readfit(netfile(a.static/'fits',W,BUFFER,'learned',f['seed'])); corners=readfit(netfile(a.static/'fits',W,BUFFER,fixed['method'],fixed['seed']))
    state,training=source(a.data,9); train=banks(state,9,W,BUFFER,0,EPOCHS); val=banks(state,9,W,BUFFER,1,VREPS); del state
    dump(out/'design.json',dict(test_order=ORDER,primary_index=7,followups=[8,9],other_temperatures='descriptive (including beta=0 null)',
        primary='frozen nonlinear minus frozen validation-selected affine, raw matched-anchor InfoNCE; lower two-sided 95% paired parent-batch limit > 0 indicates gain for this architecture/training budget, not universal necessity',
        conditional='I(H;E | corner bits): first group the 128 distinct test parents by the two opposite-corner species-A bits, then compute denominators and off-diagonal shuffles within stratum; weight each positive equally. Same frozen critic, no retraining, singleton K=1 gives zero. This is a conditional lower bound, NOT a difference of lower bounds.',
        secondary='positive absolute conditional lower limit tests residual shell information beyond the two bits. Constant conditional critic gives exact zero for the corner-only code, independent of the fitted conditional score.',
        training=training,encoder='two centered species indicators per site x 27 sites, four affine logits, trainable biases. Exact matching A,c corner start for each seed; no hidden branch (thus full logits at epoch zero need not agree).',
        critic='identical trainable 24-tanh + affine skip; identical critic seed 9601, Adam rate 0.002 then 0.00065 after epoch 24, batch 128',
        training_pairs=EPOCHS*len(train[0][0]),parents_per_epoch=len(train[0][0]),epochs=EPOCHS,validation_reps=VREPS,test_reps=TREPS,
        validation='all 3 seeds and checkpoints every 3 epochs including 0, select by raw beta09 validation only',
        static_receipt_sha256=sha(a.static/'fit-receipts.zip'),static_selection_sha256=sha(a.static/'selection.json')))
    challengers=[train_one(train,val,s,out) for s in SEEDS]; del train,val
    affine=max(challengers,key=lambda fit:fit[0]['validation']); chosen=affine[0]['seed']
    dump(out/'freeze.json',dict(affine={k:affine[0][k] for k in ('seed','epoch','validation','initial_corner_zyx')},
        affine_weights_sha256=sha(out/f'seed-{chosen}.npz'),nonlinear=f,corners=fixed,
        test_evaluation_order=ORDER,selection='frozen before first held-out shell evaluation in this contrast'))
    evaluate_frozen(a.data,out,affine,challengers,nonlinear,corners)


def diagonal_sensitivity(v,e,models,local,ix,beta):
    """Exploratory symmetry check; do not optimize an orientation on confirmation."""
    rows=[]; batches={}
    for name,model in models.items():
        for i,j in ((0,26),(2,24),(6,20),(8,18)):
            if i==0:
                x=local[name][:,2:]
            else:
                g=2*(v[:,:,i]==1)+(v[:,:,j]==1)
                x=score(v,e,model,g)[0][:,2:]
            batches[f'i{ix}_{name}_corners{i}_{j}']=x
            margin=t.ppf(1-.05/(2*4),len(x)-1)*x[:,0].std(ddof=1)/np.sqrt(len(x))
            rows.append(dict(index=ix,beta=beta,model=name,flat_zyx_i=i,flat_zyx_j=j,
                  **stats(x[:,0],'conditional_raw'),**stats(x[:,1],'conditional_shuffle'),
                  simultaneous_lo=x[:,0].mean()-margin,simultaneous_hi=x[:,0].mean()+margin))
    return rows,batches


def evaluate_frozen(data,out,affine,challengers,nonlinear,corners):
    chosen=affine[0]['seed']
    scores=[];contrasts=[];all_batches={};sources=[];diagnostics={};cold_seeds=[]
    diagonals=[]; diagonal_batches={}
    for ix in ORDER:
        state,meta=source(data,ix); v,e=banks(state,ix,W,BUFFER,2,TREPS);del state; sources.append(meta)
        gg=[codes(vv,W,corners[0]['spec']) for vv in v]; local={}
        cohort='primary confirmation' if ix==7 else ('previously examined follow-up' if ix in (8,9) else 'descriptive')
        for name,model in [('nonlinear',nonlinear),('affine',affine),('corners',corners)]:
            values,diag=score(v,e,model,gg);local[name]=values;all_batches[f'i{ix}_{name}']=values; diagnostics[f'i{ix}_{name}']=diag
            row=dict(index=ix,beta=meta['beta'],cohort=cohort,model=name)
            for j,k in enumerate(METRICS):row.update(stats(values[:,j],k))
            row.update(stats(values[:,0]-values[:,1],'paired_minus_shuffle'))
            row.update(stats(values[:,2]-values[:,3],'conditional_paired_minus_shuffle')); scores.append(row)
        for first,second in [('nonlinear','affine'),('affine','corners'),('nonlinear','corners')]:
            delta=local[first]-local[second];row=dict(index=ix,beta=meta['beta'],cohort=cohort,comparison=first+'-minus-'+second)
            for j,k in enumerate(METRICS): row.update(stats(delta[:,j],k))
            contrasts.append(row)
        if ix in (7,8,9):
            rows,bs=diagonal_sensitivity(v,e,{'affine':affine,'nonlinear':nonlinear},local,ix,meta['beta'])
            diagonals.extend(rows);diagonal_batches.update(bs)
        if ix==9:
            for fit in challengers:
                seed=fit[0]['seed']; values=local['affine'] if seed==chosen else score(v,e,fit,gg)[0];all_batches[f'i9_affine_seed{seed}']=values
                cold_seeds.append(dict(seed=seed,validation=fit[0]['validation'],selected_epoch=fit[0]['epoch'],**stats(values[:,0],'raw'),**stats(local['nonlinear'][:,0]-values[:,0],'nonlinear_minus_affine')))
    write_csv(out/'bounds.csv',scores);write_csv(out/'contrasts.csv',contrasts);write_csv(out/'cold-affine-seeds.csv',cold_seeds)
    np.savez_compressed(out/'parent-batches.npz',**all_batches)
    write_csv(out/'conditional-diagonals.csv',diagonals)
    np.savez_compressed(out/'conditional-diagonal-batches.npz',**diagonal_batches)
    dump(out/'inference.json',dict(sources=sources,diagnostics=diagnostics,columns=METRICS,
        npz='per index/model: 32 disjoint groups of 128 independent test parents; average all 16 anchor repeats WITHIN each group. Conditional on fits, pointwise paired t intervals, not simultaneous bands or intervals for true mutual information.',
        conditional='K depends on stratum frequency, so conditional bounds not directly comparable to unconditional bounds by subtraction. Frozen marginal critic need not be optimal conditional critic. All affine/nonlinear contrasts use their own frozen trained critic.'))
    plot(out,scores,contrasts)


if __name__=='__main__': main()
