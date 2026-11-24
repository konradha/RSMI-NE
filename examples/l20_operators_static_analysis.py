"""Held-out scores, frozen-temperature transfer, motif distillation and spatial diagnostics."""
import csv
import itertools
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t
from sklearn.metrics import adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier, export_text

from l20_operators_static import (BATCH, EPOCHS, EYE, TREPS, args, banks, dense, dump, evaluate,
                                  forward, offsets, partition, physical, prob, readfit, source, table)


def csvout(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,rows[0]); writer.writeheader(); writer.writerows(rows)


def stat(x, name):
    x=np.asarray(x,dtype=float); h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)); m=x.mean()
    return {name: m, name+'_lo': m-h, name+'_hi': m+h}


def batchstats(a, b):
    return {**stat(a[:,0],'raw'),**stat(a[:,1],'shuffle'),**stat(a[:,0]-a[:,1],'paired_minus_shuffle'),
            **stat(a[:,0]-b[:,0],'minus_fixed')}


def transforms(w):
    cube=np.arange(w**3).reshape((w,)*3)
    for axes in itertools.permutations(range(3)):
        for signs in itertools.product((1,-1),repeat=3):
            yield axes,signs,cube.transpose(axes)[tuple(slice(None,None,s) for s in signs)].ravel()


def agreement(y,z):
    cm=np.bincount(4*z+y,minlength=16).reshape(4,4); a,b=linear_sum_assignment(-cm)
    return cm[a,b].sum()/len(y), b


def stability(w,b,val,test,learned):
    # Select one common orientation and one category relabeling on validation, not test.
    vv=val[:2].reshape(-1,w**3); vt=test[:4].reshape(-1,w**3); orbit=list(transforms(w)); output=[]
    def labels(v,fit): return prob(v,w,fit[0]['spec'],fit[1]).argmax(1)
    for i,j in itertools.combinations(range(len(learned)),2):
        f,g=learned[i],learned[j]; y=labels(vv,f); best=-1
        for k,(axes,sign,order) in enumerate(orbit):
            z=labels(vv[:,order],g); acc,perm=agreement(y,z)
            if k==0: perm0=perm
            if acc>best: best,chosen,perm1=acc,k,perm
        axes,sign,order=orbit[chosen]; yt=labels(vt,f); raw=labels(vt,g); z=perm1[labels(vt[:,order],g)]
        # Direct numpy-scalar conversion here keeps the orientation receipt portable.
        chance=np.dot(np.bincount(z,minlength=4),np.bincount(yt,minlength=4))/len(z)**2
        acc=(yt==z).mean(); output.append(dict(width=w,buffer=b,seed_a=f[0]['seed'],seed_b=g[0]['seed'],
            axes=str(axes),signs=str(sign),state_map=','.join(map(str,perm1)),validation_accuracy=best,
            test_accuracy=acc,test_chance=chance,test_kappa=(acc-chance)/(1-chance) if chance<1 else '',
            test_ARI=adjusted_rand_score(yt,z),test_unrotated_accuracy=(perm0[raw]==yt).mean(),
            test_support_A=np.count_nonzero(np.bincount(yt)),test_support_B=np.count_nonzero(np.bincount(z))))
    return output


def feature_bank(v,w):
    co=offsets(w,1)[0]; loc=[f'[{z},{y},{x}]' for z,y,x in co]
    # Treat species labels nominally, never as ordered magnitudes in tree thresholds.
    f=[(v==1),(v==2)]; names=['A'+s for s in loc]+['B'+s for s in loc]
    for s,name in enumerate(('vacancy','A','B')):
        a=(v==s); parity=np.where(co.sum(1)%2, -1, 1)
        f.extend([a.sum(1)[:,None],(a@parity)[:,None]]); names.extend([f'N_{name}',f'staggered_{name}'])
    f.append(physical(v,w)[:,None]); names.append('internal_occupied_vacant_bonds')
    return np.hstack(f).astype('f'),names


def distill(out,w,b,model,train,test,shell):
    tr=train.reshape(-1,w**3); te=test.reshape(-1,w**3); m,enc,critic=model
    y=prob(tr,w,m['spec'],enc).argmax(1)
    # Equal sample count per training epoch; no temperature transfer or test labels used.
    f,names=feature_bank(tr,w); tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=256,random_state=11).fit(f,y)
    best=-1; best_index=None
    for j in range(w**3):
        cm=np.bincount(4*tr[:,j].astype(int)+y,minlength=12).reshape(3,4); score=cm.max(1).sum()
        if score>best: best=score; best_index=j; mapping=cm.argmax(1)
    p=prob(te,w,m['spec'],enc); target=p.argmax(1); f,_=feature_bank(te,w); pred=tree.predict(f)
    # Descriptive additive template: a codebook of position-by-species frequencies,
    # trained only to mimic the frozen mode; no shell or test information in the rule.
    nh=np.bincount(y,minlength=4)
    counts=np.array([[(tr[y==h]==s).sum(0) for s in range(3)] for h in range(4)])
    lp=np.log((counts+1)/(nh[:,None,None]+3)); prior=np.log((nh+1)/(len(y)+4))
    template=(lp[:,te,np.arange(w**3)].sum(2).T+prior).argmax(1)
    single=mapping[te[:,best_index]]
    count=np.bincount(target,minlength=4); div=lambda x:(x*np.log(np.maximum(x,1e-30))).sum(axis=-1)
    # Tree is purely a surrogate for the frozen operator, never a replacement critic.
    tree_text=export_text(tree,feature_names=names,decimals=2)
    metrics=dict(width=w,buffer=b,seed=m['seed'],n_train=len(tr),n_test=len(te),
        train_source='four fresh-anchor epochs of training parents',test_source='first four anchor repeats of test parents',
        tree_depth=3,min_leaf=256,tree_rules=tree_text,tree_accuracy=float((pred==target).mean()),
        template_accuracy=float((template==target).mean()),template_prior=prior.tolist(),template_loglikelihood_h_species_site=lp.tolist(),
        tree_confusion=np.bincount(target*4+pred,minlength=16).reshape(4,4).tolist(),
        majority_accuracy=float((target==np.bincount(y).argmax()).mean()),
        best_single_site_zyx=offsets(w,b)[0][best_index].tolist(),single_site_map=mapping.tolist(),
        single_site_accuracy=float((mapping[te[:,best_index]]==target).mean()),
        modal_state_fraction=(count/len(target)).tolist(),state_probability_mean=p.mean(0,dtype=float).tolist(),
        mean_max_probability=float(p.max(1).mean()),conditional_entropy=float(-div(p).mean()),
        marginal_entropy=float(-div(p.mean(0,dtype=float))),feature_importance={n:float(a) for n,a in zip(names,tree.feature_importances_) if a>0})
    dump(out/'motif.json',metrics)
    # Cellwise P(species|modal state) minus unconditional P(species), held-out.
    co=offsets(w,b)[0]; rows=[]
    for state,n in enumerate(count):
        if not n: continue
        delta=np.stack([(te[target==state]==s).mean(0)-(te==s).mean(0) for s in range(3)],axis=1)
        for c,d in zip(co,delta): rows.append(dict(H=state,z=c[0],y=c[1],x=c[2],delta_vac=d[0],delta_A=d[1],delta_B=d[2]))
    csvout(out/'state-enrichment.csv',rows)
    # Use the very same four cold critic rows with these approximate category
    # labels. A bad substitution can have a negative bound; do not recalibrate.
    z=forward(dense(shell.reshape(len(te),-1)),critic)[1].reshape(len(test),-1,BATCH,4).astype(float)
    z-=z.max(2,keepdims=True); ratios=z-np.log(np.exp(z).mean(2,keepdims=True))
    shuffled=(ratios.sum(2,keepdims=True)-ratios)/(BATCH-1)
    bounds=[]
    def per_batch(probability,t):
        return (probability.reshape(t.shape)*t).sum(3).mean((0,2))
    raw_ref=per_batch(p,ratios)
    for name,pp in [('learned',p),('modal_state',EYE[target]),('species_template',EYE[template]),
                    ('depth3_tree',EYE[pred]),('best_site_surrogate',EYE[single])]:
        bound=per_batch(pp,ratios)
        bounds.append(dict(method=name,**stat(bound,'raw'),**stat(per_batch(pp,shuffled),'shuffle'),
                           **stat(bound-raw_ref,'minus_original')))
    csvout(out/'distilled-scores.csv',bounds)
    fig=plt.figure(figsize=(10,5.4),layout='constrained'); grid=fig.add_gridspec(1,2,width_ratios=[1.15,1])
    ax=fig.add_subplot(grid[0]); bx=fig.add_subplot(grid[1])
    patterns=np.full((4,w,w,w),np.nan)
    for row in rows: patterns[row['H'],row['z']+w//2,row['y']+w//2,row['x']+w//2]=row['delta_A']
    palette=plt.get_cmap('RdBu_r').copy(); palette.set_bad('white'); image=np.full((4*(w+1)-1,w*(w+1)-1),np.nan)
    for h,zidx in itertools.product(range(4),range(w)):
        image[h*(w+1):h*(w+1)+w,zidx*(w+1):zidx*(w+1)+w]=patterns[h,zidx]
    limit=np.nanmax(np.abs(image)); handle=ax.imshow(image,cmap=palette,vmin=-limit,vmax=limit,interpolation='nearest')
    ax.set(xticks=np.arange(w)*(w+1)+w//2,xticklabels=[f'z={i-w//2}' for i in range(w)],yticks=np.arange(4)*(w+1)+w//2,
           yticklabels=[f'H={i}' for i in range(4)],title='a   Species A in each learned state')
    fig.colorbar(handle,ax=ax,shrink=.7,label='P(A at site | modal H) − P(A at site)')
    for i,row in enumerate(bounds):
        lo,mid,hi=[row[k] for k in ('raw_lo','raw','raw_hi')]
        bx.errorbar(mid,i,xerr=[[mid-lo],[hi-mid]],fmt='o',capsize=3,color='#166b77' if i<2 else '#b56235')
    bx.axvline(0,color='.5',ls=':');bx.set(yticks=range(5),yticklabels=['Stochastic encoder','Modal encoder','Additive species template','Depth-three tree','One-site surrogate'],
        xlabel='Raw InfoNCE, unchanged cold critic (nats)',title='b   Substitute the distilled labels');bx.invert_yaxis();bx.grid(axis='x',alpha=.2)
    for axis in (ax,bx): axis.spines[['top','right']].set_visible(False)
    fig.suptitle('What does the four-state code keep?  |  Exploratory cold motif distillation',fontsize=14,weight='bold')
    fig.savefig(out/'static-motif.png',dpi=180);plt.close(fig)
    return metrics


def spatial(state,index,w,b,models):
    """Whole-parent average at 32 periodic random origins, every positive axial displacement."""
    g=partition(index)[0][2]; rng=np.random.default_rng([119823,index]); vtab=table(offsets(w,b)[0]);
    dis=[0]+list(range(w,11)); c=np.indices((20,)*3).reshape(3,-1).T
    lut=np.empty((len(dis)-1,3,20**3),np.int16)
    for k,r in enumerate(dis[1:]):
        for d in range(3):
            cc=c.copy(); cc[:,d]=(cc[:,d]+r)%20; lut[k,d]=np.ravel_multi_index(cc.T,(20,)*3)
    means=np.empty((2,len(g),4)); products=np.zeros((2,len(g),len(dis)))
    for start in range(0,len(g),64):
        ix=slice(start,start+64); rows=g[ix,None,None]; a=rng.integers(20**3,size=(len(g[ix]),32));
        def probabilities(at):
            raw=state[rows,vtab[at]].reshape(-1,w**3)
            return [prob(raw,w,m['spec'],enc).reshape(len(g[ix]),32,4).astype(float) for m,enc,_ in models]
        p=probabilities(a)
        for i,pi in enumerate(p):
            means[i,ix]=pi.mean(1); products[i,ix,0]=(pi**2).sum(2).mean(1)
        for k in range(len(lut)):
            for d in range(3):
                for i,other in enumerate(probabilities(lut[k,d][a])):
                    products[i,ix,k+1]+=(p[i]*other).sum(2).mean(1)/3
    corr=[]; loo=[]; covs=[]; n=len(g)
    for mu,pr in zip(means,products):
        # Independent-parent product U statistic removes finite-n self bias in |E q|².
        s=mu.sum(0); selfsq=(mu**2).sum(1); u=(s@s-selfsq.sum())/(n*(n-1)); cv=pr.mean(0)-u
        uloo=(((s-mu)**2).sum(1)-selfsq.sum()+selfsq)/((n-1)*(n-2)); cvloo=(pr.sum(0)-pr)/(n-1)-uloo[:,None]
        covs.append(cv); corr.append(cv/cv[0]); loo.append(cvloo/cvloo[:,:1])
    def se(l): return np.sqrt((n-1)/n*((l-l.mean(0))**2).sum(0))
    rows=[]
    for i,model in enumerate(models):
        err=se(loo[i]); delta=corr[i]-corr[1]; err_delta=se(loo[i]-loo[1])
        for k,r in enumerate(dis): rows.append(dict(index=index,method=model[0]['method'],separation=r,
            connected_inner_product=covs[i][k],correlation=corr[i][k],jackknife_se=err[k],
            contrast_to_fixed=delta[k],contrast_jackknife_se=err_delta[k]))
    return rows,means,products


def make_figure(out,scores,transfer,spatial_rows,flag,reference):
    w,b,s=flag['width'],flag['buffer'],flag['seed']; cols=['#166b77','#c1693e']; fig,axes=plt.subplots(2,2,figsize=(12,8.4),layout='constrained'); a,bx,c,d=axes.ravel()
    def err(ax,x,row,key,color,marker,label=None):
        ax.errorbar(x,row[key],[[row[key]-row[key+'_lo']],[row[key+'_hi']-row[key]]],fmt=marker,color=color,label=label,capsize=2,ms=5)
    geoms=list(itertools.product((3,5),(1,2)))
    for i,(ww,bb) in enumerate(geoms):
        sub=[r for r in scores if (r['width'],r['buffer'])==(ww,bb)]
        for j,r in enumerate([r for r in sub if r['method']=='learned']):
            err(a,i+(j-1)*.085-.10,r,'raw',cols[0],'o','Learned (3 starts)' if i==0 and j==0 else None)
        row=next(r for r in sub if r['method']==r['strongest_fixed']);err(a,i+.18,row,'raw',cols[1],'D','Strongest fixed' if i==0 else None)
    a.set(xticks=range(4),xticklabels=[f'{ww}³ / {bb}' for ww,bb in geoms],xlabel='Block / buffer',ylabel='Raw InfoNCE (nats)',title='a   Cold held-out bounds'); a.legend(frameon=False,fontsize=8)
    for i,index in enumerate((0,8,9)):
        for j,(m,marker) in enumerate([('learned','o'),(reference,'D')]):
            row=next(r for r in transfer if r['index']==index and r['method']==m)
            err(bx,i+[-.16,.16][j]-.04,row,'raw',cols[j],marker);err(bx,i+[-.16,.16][j]+.045,row,'shuffle',cols[j],'x')
    bx.set(xticks=range(3),xticklabels=['0','2.7111','3.05'],xlabel='Evaluation β  (frozen cold encoder AND critic)',ylabel='InfoNCE (nats)',title='b   Temperature transfer: points = raw, × = shuffle')
    for m,col,mark in [('learned',cols[0],'o'),(reference,cols[1],'D')]:
        for ix,ls in ((0,':'),(8,'--'),(9,'-')):
            rows=[r for r in spatial_rows if r['index']==ix and r['method']==m and r['separation']]
            c.errorbar([r['separation'] for r in rows],[r['correlation'] for r in rows],yerr=[1.96*r['jackknife_se'] for r in rows],
                color=col,marker=mark,ls=ls,ms=3,capsize=1,label=f'{m}, β={ {0:"0",8:"2.71",9:"3.05"}[ix]}')
    c.set(xlabel='Axial center separation (sites); disjoint visible supports',ylabel='Connected q correlation / C(0)',title='c   Frozen spatial operators'); c.legend(frameon=False,fontsize=7,ncol=2)
    rows=[r for r in scores if (r['width'],r['buffer'])==(w,b) and (r['method']!='learned' or r['seed']==s)]
    rows.sort(key=lambda r:r['raw']); labels=[r['method'] for r in rows]
    for i,r in enumerate(rows):err(d,i,r,'raw',cols[0] if r['method']=='learned' else cols[1], 'o')
    d.set(xticks=range(len(rows)),xticklabels=labels,ylabel='Raw InfoNCE (nats)',title=f'd   Frozen choices at {w}³ / buffer {b}'); d.tick_params(axis='x',labelrotation=52,labelsize=7)
    for ax in axes.ravel(): ax.axhline(0,color='#536168',ls=':',lw=.8);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16)
    fig.suptitle('Four-state equilibrium operators | L=20, 4,096 unseen parents per temperature',fontsize=15,weight='bold')
    fig.savefig(out/'static-operators.png',dpi=180);plt.close(fig)


def gates(out,scores,transfer):
    def decision(row):
        raw_lo,delta_lo=[float(row[k]) for k in ('raw_lo','minus_fixed_lo')]
        return dict(width=int(row['width']),buffer=int(row['buffer']),seed=int(row['seed']),
                    strongest_fixed=row['strongest_fixed'],positive_raw=raw_lo>0,
                    positive_and_above_fixed=raw_lo>0 and delta_lo>0)
    dump(out/'gates.json',dict(criterion='Pointwise lower 95% confidence limit > 0; superiority additionally requires positive paired contrast to the validation-selected fixed code. Conditional on trained models; not a multiple-testing guarantee.',
        learned_cold=[decision(row) for row in scores if row['method']=='learned'],
        frozen_flagship=[dict(beta=float(row['beta']),**decision(row)) for row in transfer if row['method']=='learned']))


def main():
    a=args(); out=a.out; files=sorted((out/'fits').glob('*.json'))
    if not files:
        with zipfile.ZipFile(out/'fit-receipts.zip') as archive:
            files=[out/s for s in sorted(archive.namelist()) if s.endswith('.json')]
    fits=[readfit(p.with_suffix('')) for p in files]
    # Seal model identities before opening ANY held-out banks or transfer snapshots.
    groups={}; selected={};learn=[f for f in fits if f[0]['method']=='learned'];
    assert len(learn)==12 and len(fits)==104, 'incomplete training receipts'
    for w,b in itertools.product((3,5),(1,2)):
        fs=[f for f in fits if (f[0]['width'],f[0]['buffer'])==(w,b)]; groups[w,b]=fs
        selected[w,b]=max([f for f in fs if f[0]['method'] not in ('learned','random')],key=lambda f:f[0]['validation'])
    flagship=max(learn,key=lambda f:f[0]['validation']); flag=flagship[0]; w0,b0=flag['width'],flag['buffer']; fixed=selected[w0,b0]
    dump(out/'selection.json',dict(flagship={k:flag[k] for k in ('width','buffer','method','seed','epoch','validation')},
        fixed_by_geometry={f'{w}-{b}':{k:f[0][k] for k in ('method','seed','epoch','validation')} for (w,b),f in selected.items()}))
    state,meta=source(a.data,9); src=[meta]; scores=[]; transfer=[]; stability_rows=[]; batches={}; spatial_rows=[]; parent_moments={}
    for (w,b),models in groups.items():
        test=banks(state,9,w,b,2,TREPS); val=banks(state,9,w,b,1,2)[0]
        arr=[]
        for m,en,cr in models: arr.append(evaluate(*test,w,m['spec'],en,cr))
        im=next(i for i,f in enumerate(models) if f is selected[w,b]); base=arr[im]
        for (m,en,cr),v in zip(models,arr):
            row={k:m[k] for k in ('width','buffer','method','seed','epoch','validation')}; row.update(dict(strongest_fixed=models[im][0]['method']),**batchstats(v,base));scores.append(row)
            batches[f'i9-w{w}-b{b}-{m["method"]}-s{m["seed"]}']=v
            if (w,b)==(w0,b0) and (m['method']!='learned' or m['seed']==flag['seed']): transfer.append(dict(index=9,beta=meta['beta'],**row))
        stability_rows.extend(stability(w,b,val,test[0],[f for f in models if f[0]['method']=='learned']))
        if (w,b)==(w0,b0): distill(out,w,b,flagship,banks(state,9,w,b,0,4)[0],test[0][:4],test[1][:4])
    for ix in (9,8,0):
        if ix!=9:
            del state; state,meta=source(a.data,ix); src.append(meta)
            test=banks(state,ix,w0,b0,2,TREPS); models=[f for f in groups[w0,b0] if f[0]['method']!='learned' or f is flagship]
            indexed=[evaluate(*test,w0,m['spec'],en,cr) for m,en,cr in models]; ref=next(x for f,x in zip(models,indexed) if f is fixed)
            for f,v in zip(models,indexed):
                m=f[0]; row={k:m[k] for k in ('width','buffer','method','seed','epoch','validation')}; row.update(dict(strongest_fixed=fixed[0]['method']),**batchstats(v,ref)); transfer.append(dict(index=ix,beta=meta['beta'],**row)); batches[f'i{ix}-w{w0}-b{b0}-{m["method"]}-s{m["seed"]}']=v
        rows,mu,pr=spatial(state,ix,w0,b0,[flagship,fixed]);spatial_rows.extend(rows);parent_moments[f'i{ix}_means']=mu;parent_moments[f'i{ix}_products']=pr
    csvout(out/'scores.csv',scores);csvout(out/'transfer.csv',transfer);csvout(out/'reproducibility.csv',stability_rows);csvout(out/'spatial.csv',spatial_rows)
    np.savez_compressed(out/'disjoint-batch-scores.npz',**batches);np.savez_compressed(out/'spatial-parent-moments.npz',**parent_moments)
    dump(out/'inference.json',dict(sources=src,training_epochs=EPOCHS,validation_score='raw InfoNCE',
        bounds='raw/unclipped expected hard-H InfoNCE on same-beta disjoint parent batches, evaluated after freezing both networks; not unbiased true MI',
        shuffle='exact average over the 127 mismatched positive labels, keeping same critic denominator (not a retrained null, not necessarily 0)',
        intervals='two-sided 95% t intervals over 32 paired disjoint parent batches, with all patch repetitions kept together; conditional on the trained fit',
        spatial='32 random origins per held-out parent, all +axis directions, r >= width apart, C(r) = E[<q(x),q(x+r)>] - |E q|^2; latter from distinct parents. Normalize by zero-lag posterior variance. Whole-parent jackknife. No shell input when evaluating q.',
        stability='Choose one of 48 cube isometries and optimal four-label permutation by argmax agreement on 2 validation anchors per parent. Test with first 4 test anchors. ARI and chance-corrected agreement accompany accuracy; soft-posterior equivalence is not asserted.',
        controls='site-zyx gives sorted nonnegative center offsets, one representative of every cubic site orbit. bits-geometry-s encodes the two indicators of species s at a corner and at a near neighbor or an edge-, face-, or body-separated corner. Each rule has <=4 states and its own trained critic; composition uses train medians of vacancy and A counts, parity train quartiles of signed occupancy, interfaces train quartiles of internal nearest neighbor occupied-vacant bonds. random has frozen random nonlinear softmax encoder, trained critic.',
        spatial_moments='npz arrays indexed [method (flagship, fixed), test parent in partition order, state or distance]; distances [0,width,...,10]',
        gate_interpretation='Positive raw lower bound is required for evidence by this critic. Paired-vs-shuffle alone is not positive MI evidence when raw is negative. Beating strongest deterministic rule is distinct from detecting static dependence. No test-based fit selection.'))
    make_figure(out,scores,transfer,spatial_rows,flag,fixed[0]['method'])
    gates(out,scores,transfer)


if __name__=='__main__': main()
