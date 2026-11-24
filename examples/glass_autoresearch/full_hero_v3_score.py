"""Pure per-beta V3 assessment. No TEST loader, retraining, state alignment or tuning.

The caller must first freeze all endpoint/forecast objects and obtain the separate
all-ten TEST ticket. This module accepts already opened arrays in stable roster order.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.stats import t
from hero_rule import hard_control
from hero_train import (BASELINES, C, K, LAWS, LOG_C, SEEDS, DrawLaw, center_logits, hbits,
                        logsoftmax, shell_logits, softmax, strata)

N,Q,GROUPS,BLOCKS,SEED=8192,8,8192,16,731903


def load_scoring_models(refit,checkpoint_dir):
    """Verify previously refitted immutable weights only. No specimen input."""
    if (refit['schema']!='hero-full-v3-refit-1' or refit['status']!='complete' or refit['primary_seed']!=6101
            or refit['indices']!=list(range(10)) or refit['seeds']!=list(SEEDS) or refit['allows_test'] is not False
            or set(refit['checkpoints'])!={str(s) for s in SEEDS} or set(refit['controls'])!=set(BASELINES)):
        raise PermissionError('Missing full all-TRAIN refit or comparison weights')
    def load(entry):
        name=entry['path']
        if Path(name).name!=name or not name.startswith('refit-') or not name.endswith('.npz'):
            raise ValueError('Unexpected final weight name')
        path=Path(checkpoint_dir)/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Frozen weights changed')
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k] for k in z.files}
    models={str(s):(load(refit['checkpoints'][str(s)]),None) for s in SEEDS}
    for name,e in refit['controls'].items():
        if e['seed']!=6101 or e['frozen_spec']['control']!=name:raise ValueError('Mislabelled fixed code')
        models[name]=(load(e),e['frozen_spec'])
    return dict(models=models,primary='6101',split_sha256=refit['split_sha256'],
                nomination_sha256=refit['nomination_sha256'],refit_release_sha256=refit['refit_release_sha256'],source=refit['source'])


def interval(value,rep,comparisons=1):
    rep=np.asarray(rep,dtype='d');var=float((BLOCKS-1)/BLOCKS*np.square(rep-rep.mean()).sum())
    se=math.sqrt(var);half=float(t.ppf(1-.025/comparisons,BLOCKS-1)*se)
    return dict(gain_nats=float(value),variance=var,se=se,ci95=[float(value-half),float(value+half)],
                delete_block_nats=rep.tolist(),df=BLOCKS-1,bonferroni_comparisons=comparisons)


def score_one_beta(bundle,beta_index,V,E,parent):
    """Return JSON-safe model/law scores and paired contrasts on one intact beta.

    All 8192 TEST parents have 8 anchors; each group contains 8 parents and all 8
    rows contribute as positives. Groups can share parents. All model comparisons
    use the identical Monte Carlo sample; intervals cluster whole parents.
    """
    b=int(beta_index);v=np.asarray(V);e=np.asarray(E);p=np.asarray(parent)
    if (b not in range(10) or v.shape!=(N,Q,27) or e.shape!=(N,Q,386) or v.dtype!='u1' or e.dtype!='u1'
            or p.shape!=(N,) or p.dtype.kind not in 'iu' or len(np.unique(p))!=N or v.max()>2 or e.max()>2):
        raise ValueError('Expected one complete registered beta, 8192 distinct parent rows x 8 ternary anchors')
    dl=DrawLaw(v);ix=np.array([dl.sample(np.random.default_rng([SEED,b,j]),GROUPS,l) for j,l in enumerate(LAWS)])
    rows,inv=np.unique(ix,return_inverse=True);ii=inv.reshape(ix.shape)
    blocks=np.random.default_rng([SEED,b,323]).permutation(N)%BLOCKS
    group_blocks=blocks[ix//Q];group_keep=[[(g!=d).all(axis=1) for d in range(BLOCKS)] for g in group_blocks]
    for masks in group_keep:
        if not all(m.any() for m in masks):raise ValueError('Nonestimable whole-parent replicate')
    ss=strata(v);valid=np.array([x['s'] for x in dl.support['strata'] if x['used']]);matched=np.isin(ss,valid)
    support={}
    for j,l in enumerate(LAWS):
        parents=ix[j]//Q
        if (np.diff(np.sort(parents,axis=1),axis=1)==0).any():raise ValueError('Within-group parent repeated')
        if l=='matched' and np.any(np.diff(ss.ravel()[ix[j]],axis=1)):raise ValueError('Nonmatched conditional group')
        counts=np.bincount(parents.ravel(),minlength=N)
        support[l]=dict(eligible_parents=N if l=='uniform' else int(matched.any(1).sum()),
               eligible_anchors=N*Q if l=='uniform' else int(matched.sum()),positive_slots=int(parents.size),
               unique_drawn_parents=int((counts>0).sum()),unique_drawn_opportunities=int(np.unique(ix[j]).size),
               min_draws_per_parent=int(counts.min()),max_draws_per_parent=int(counts.max()),
               ordered_opportunity_draw_sha256=hashlib.sha256(ix[j].astype('<i8',copy=False).tobytes()).hexdigest(),
               group_counts_after_deleting_parent_blocks=[int(k.sum()) for k in group_keep[j]])
    models={}
    for key,(m,spec) in bundle['models'].items():
        if spec is None:
            q=softmax(center_logits(m,v.reshape(-1,27)).astype('d'))
        else:
            q=np.eye(K,dtype='d')[hard_control(v,spec).ravel()]
        h=q.argmax(-1);freq=q.mean(0);hf=np.bincount(h,minlength=K)/len(h);ent=float(hbits(freq));cond=float(hbits(q).mean())
        cr=shell_logits(m,e.reshape(-1,386)[rows]).astype('d')
        scores={}
        for j,l in enumerate(LAWS):
            lp=LOG_C+logsoftmax(cr[ii[j]],1)
            hard=np.take_along_axis(lp,h[ix[j],None],axis=2)[...,0].mean(1)
            expected=(lp*q[ix[j]]).sum(-1).mean(1)
            scores[l]=dict(hard_nats=float(hard.mean()),expected_nats=float(expected.mean()),
                delete_parent_block_hard_nats=[float(hard[k].mean()) for k in group_keep[j]],
                delete_parent_block_expected_nats=[float(expected[k].mean()) for k in group_keep[j]])
        for metric in ('hard','expected'):
            scores.setdefault('common',{})[metric+'_nats']=float(np.mean([scores[l][metric+'_nats'] for l in LAWS]))
            jack='delete_parent_block_'+metric+'_nats'
            scores['common'][jack]=np.mean([scores[l][jack] for l in LAWS],axis=0).tolist()
        models[key]=dict(scores=scores,all_parent_rates=dict(n_parents=N,n_anchors=N*Q,hard_frequency=hf.tolist(),
                         hard_H_bits=float(hbits(hf)),stochastic_frequency=freq.tolist(),H_bits=ent,H_given_V_bits=cond,I_V_H_bits=ent-cond))
    paired={}
    central=models[bundle['primary']]['scores']
    for name in BASELINES:
        paired[name]={}
        for l in (*LAWS,'common'):
            paired[name][l]={}
            for metric in ('hard','expected'):
                k=metric+'_nats';jk='delete_parent_block_'+k
                val=central[l][k]-models[name]['scores'][l][k]
                delta=np.array(central[l][jk])-models[name]['scores'][l][jk]
                paired[name][l][metric]=interval(val,delta)
                if (l,metric)==('common','hard'):
                    paired[name][l][metric]['simultaneous_60_ci95']=interval(val,delta,60)['ci95']
    return dict(schema='hero-full-v3-frozen-score-beta-1',source=bundle['source'],split_sha256=bundle['split_sha256'],
                refit_release_sha256=bundle['refit_release_sha256'],index=b,n_parents=N,anchor_count=Q,
                n_groups_per_law=GROUPS,primary_seed=6101,group_seed=SEED,n_delete_parent_blocks=BLOCKS,
                parent_sha256=hashlib.sha256(p.astype('<i4',copy=False).tobytes()).hexdigest(),
                sampling=support,matched_stratum_support=dl.support,models=models,paired_primary_minus_control=paired)


def summarize_ten_beta(records):
    """Equal weights on all ten fixed temperatures; no best-seed/temperature selection."""
    rows={int(k):r for k,r in records.items()}
    if set(rows)!=set(range(10)) or any(r['index']!=b or r['n_parents']!=N for b,r in rows.items()):
        raise ValueError('All ten complete beta receipts required; no missing-beta renormalization')
    root=rows[0]
    if any(any(r[k]!=root[k] for k in ('split_sha256','refit_release_sha256','source','group_seed')) for r in rows.values()):
        raise ValueError('Assessment receipts are not one frozen pipeline')
    def summarize(points,variances,family):
        value=float(np.mean(points));v=np.asarray(variances)/100;total=float(v.sum());se=math.sqrt(total)
        df=float(total**2/np.sum(v*v/(BLOCKS-1))) if total>0 else float(BLOCKS-1)
        c=float(t.ppf(.975,df))*se;res=dict(mean_nats=value,se=se,welch_df=df,ci95=[value-c,value+c])
        if family:
            d=float(t.ppf(1-.025/family,df))*se;res['simultaneous_ci95']=[value-d,value+d];res['bonferroni_comparisons']=family
        return res
    macro_models={};macro_contrasts={}
    for key in (*map(str,SEEDS),*BASELINES):
        macro_models[key]={}
        for l in (*LAWS,'common'):
            macro_models[key][l]={}
            for metric in ('hard','expected'):
                k=metric+'_nats';jk='delete_parent_block_'+k
                stats=[r['models'][key]['scores'][l] for r in rows.values()]
                macro_models[key][l][metric]=summarize([s[k] for s in stats],[interval(s[k],s[jk])['variance'] for s in stats],0)
    for key in BASELINES:
        macro_contrasts[key]={}
        for l in (*LAWS,'common'):
            macro_contrasts[key][l]={}
            for metric in ('hard','expected'):
                s=[r['paired_primary_minus_control'][key][l][metric] for r in rows.values()]
                macro_contrasts[key][l][metric]=summarize([x['gain_nats'] for x in s],[x['variance'] for x in s],
                                                           6 if (l,metric)==('common','hard') else 0)
    vals=[macro_models[str(s)]['common']['hard']['mean_nats'] for s in SEEDS]
    return dict(schema='hero-full-v3-frozen-score-ten-beta-1',source=root['source'],split_sha256=root['split_sha256'],
                refit_release_sha256=root['refit_release_sha256'],primary_seed=6101,indices=list(range(10)),
                group_seed=SEED,absolute_macro_scores=macro_models,paired_primary_minus_control=macro_contrasts,
                primary_seed_beta0_common_scores=rows[0]['models']['6101']['scores']['common'],
                five_start_macro_common_hard_nats=vals,five_start_spread_sd_nats=float(np.std(vals,ddof=1)),
                inference='Finite-candidate predictive nats on seen temperatures, not MI or zero-shot transfer. Pointwise t intervals; six macro and 60 per-beta comparisons adjusted for common-hard differences. Paired whole-parent deletion uncertainty conditional on fixed Monte Carlo groups; seed spread separately descriptive.')
