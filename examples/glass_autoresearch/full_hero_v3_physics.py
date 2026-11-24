"""V3 central distillation and same-f forecast. No native TEST reader in this module."""
import argparse
from pathlib import Path

import numpy as np
import full_hero_forecast as fc
import full_hero_rule as cr
from hero_bank import readj, require, sha
from hero_forecast import exact_inventory_null
from full_hero_stream import cutoff, savej
from full_hero_v3_stream import PUB,WORK,INDICES
from hero_train import SEEDS


def weights(directory, record):
    name=record['path']; p=directory/name
    require(Path(name).name==name and sha(p)==record['sha256'],'checkpoint identity changed')
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def train_inputs(fields):
    split=readj(PUB/'split.json'); summary=readj(PUB/'stream-summary.json'); inputs={}
    require(summary['split_sha256']==sha(PUB/'split.json') and summary['test_species_rows_returned']==0 and
            summary['train_parents']==81920 and split['rosters_sha256']==sha(WORK/'rosters.npz'),'training provenance')
    with np.load(WORK/'rosters.npz',allow_pickle=False) as roster:
        for b in INDICES:
            file=PUB/f'stream-b{b:02}.json'; require(sha(file)==summary['stream_receipts_sha256'][str(b)],'wrong trained stream')
            rec=readj(file); root=WORK/'streams'/f'b{b:02}'/'train'; arr={}
            for key in ('parent',*fields):
                path=root/(key+'.npy'); require(sha(path)==rec['roles']['train']['files'][path.name]['sha256'],'array digest')
                arr[key]=np.load(path,mmap_mode='r',allow_pickle=False)
            require(np.array_equal(arr.pop('parent'),roster[f'b{b:02}_train']),'native parent order mismatch')
            inputs[b]=arr
    return inputs


def fit_blocks(data, ranks, rule, critic, counts=None):
    # Exactly 8 preselected translations per parent; index array is never interpreted as native ID.
    for k in range(0,len(ranks),64):
        take=ranks[k:k+64]; i=data['I'][take,::8]; e=data['E'][take,::8]; v=data['V'][take,::8]
        if counts is not None: fc.add_support(counts,i)
        yield i,fc.design(i,e,critic),fc.targets(v,rule)


def gram(inputs,ranks,rule,critic,counts=None):
    moment=None
    for data in inputs.values():
        for _,x,y in fit_blocks(data,ranks,rule,critic,counts):
            xy=np.column_stack((x,y)); g=xy.T@xy
            if moment is None: moment=g
            else: moment+=g
    return moment/(len(inputs)*len(ranks)*8)


def costs(data,ranks,rule,critic,coef,variance,counts,b):
    y=[]; p={}; hot=[]; covered=0
    for i,x,truth in fit_blocks(data,ranks,rule,critic):
        y.append(truth); pred=fc.predict(x,coef)
        for name,est in pred.items(): p.setdefault(name,[]).append(est)
        covered+=int(np.sum(counts[fc.inventory_codes(i)]>=8))
        if b==0: hot.append(i)
    truth=np.concatenate(y); ps={k:np.concatenate(v) for k,v in p.items()}
    result=fc.losses(truth,ps,variance,9)
    null=(exact_inventory_null(np.concatenate(hot),truth[:,0].reshape(-1,8),ps['m1']-ps['m0'],variance) if b==0 else None)
    return result, covered/len(truth), null


def fit_fixed_alphas(g, choices):
    p=g.shape[0]-len(fc.TARGETS); cov=g-np.outer(g[0],g[0]); out={}
    for method in fc.METHODS:
        ix=fc.method_cols(method); j=ix[1:]; sd=np.sqrt(cov[j,j].clip(min=0)); sd[sd<1e-7]=1
        a=cov[np.ix_(j,j)]/(sd[:,None]*sd[None,:]); rhs=cov[j,p:]/sd[:,None]
        coeff=np.zeros((p,len(fc.TARGETS)))
        for k,alpha in enumerate(choices[method]['selected_alpha']):
            require(alpha in fc.ALPHAS,'unregistered ridge penalty')
            c=np.linalg.solve(a+alpha*np.eye(len(j)),rhs[:,k])/sd
            coeff[0,k]=g[0,p+k]-g[0,j]@c; coeff[j,k]=c
        out[method]=coeff
    return out


def validate_nomination(n):
    require(n['schema']=='hero-full-v3-nomination-1' and n['scope']=='training-fold0-selection' and
            n['split_sha256']==sha(PUB/'split.json') and n['release_sha256']==sha(PUB/'training-release-1.json') and
            n['status']=='qualified' and n['primary_seed']==6101 and n['allows_test'] is False and
            set(n['checkpoints'])==set(map(str,SEEDS)) and n['indices']==list(INDICES),'no authentic complete nomination')


def discover(root):
    cutoff('fit-dev'); nom_path=PUB/'nomination.json'; n=readj(nom_path); validate_nomination(n)
    models=[weights(root,n['checkpoints'][str(s)]) for s in SEEDS]; pos=np.arange(8192)
    fit,selection,rule=pos[pos%8>=2],pos[::8],pos[1::8]
    ins=train_inputs(('V','I','E'))
    spec,detail=cr.distill({b:np.ascontiguousarray(x['V'][fit,::8]) for b,x in ins.items()},
        {b:np.ascontiguousarray(x['V'][rule,::8]) for b,x in ins.items()},models,list(SEEDS),
        dict(version='v3',nomination_sha256=sha(nom_path),folds='2..7 FIT, 1 rule',split_sha256=n['split_sha256']))
    spec['schema']='hero-full-v3-distilled-rule-1'; detail['schema']='hero-full-v3-rule-on-train-fold1-1'
    savej(PUB/'physical-rule.json',spec); savej(PUB/'physical-rule-development.json',detail)
    report=dict(schema='hero-full-v3-same-f-train-development-1',nomination_sha256=sha(nom_path),
                rule_sha256=sha(PUB/'physical-rule.json'),test_opened=False,rule_status=spec['status'],physical_test_enabled=False)
    if spec.get('conventional_model') and spec.get('fit_sd',0)>0 and spec.get('sign') in (-1,1):
        counts=np.zeros(fc.CELL**2,'i4'); critic=models[0]
        a=gram(ins,fit,spec,critic,counts); d=gram(ins,selection,spec,critic); coef,alpha=fc.ridge_grid(a,d)
        vf=float(a[-3,-3]-a[0,-3]**2); require(vf>1e-8 and np.isclose(vf,spec['fit_variance']),'mismatched rule scale')
        per={}; support=[]; null=None
        for b,data in ins.items():
            per[b],c,h=costs(data,rule,spec,critic,coef,vf,counts,b); support.append(c)
            if h is not None: null=h
        panel=fc.panel_summary(per,range(1,10)); effect=panel['f']; power=fc.projected_power([per[b]['f']['g'] for b in range(1,10)],[8192]*9)
        gates=dict(symbolic=bool(spec['eligible_for_forecast_gate']),material=effect['g']['point']>=.01 and fc.passes(effect['g'],0),
                   vs_inventory=effect['vs_inventory']['point']>=.01 and fc.passes(effect['vs_inventory'],0),
                   residual=panel['residual']['g']['point']>0,support=np.mean(support)>=.90,
                   precision=(power['probability_lower95_above_margin'] or 0)>=.80)
        report.update(normalization=vf,alpha=alpha,fold1_by_beta=per,fold1_panel=panel,fit_support_by_beta=support,
                      beta0_exact_inventory=null,assessment_power=power,gates={k:bool(v) for k,v in gates.items()},
                      physical_test_enabled=bool(all(gates.values())))
    savej(PUB/'physical-development.json',report)


def freeze(root):
    cutoff('fit-dev'); n=readj(PUB/'nomination.json'); validate_nomination(n); ref=readj(PUB/'refit.json')
    require(ref['schema']=='hero-full-v3-refit-1' and ref['status']=='complete' and ref['nomination_sha256']==sha(PUB/'nomination.json') and
            ref['split_sha256']==n['split_sha256'] and ref['refit_release_sha256']==sha(PUB/'refit-release.json') and
            list(map(int,ref['seeds']))==list(SEEDS) and all(ref['n_train_parents'][str(b)]==8192 for b in INDICES),'refit omitted TRAIN parents')
    spec=readj(PUB/'physical-rule.json'); dev=readj(PUB/'physical-development.json')
    require(dev['rule_sha256']==sha(PUB/'physical-rule.json') and dev['nomination_sha256']==sha(PUB/'nomination.json'),'rule or development substitution')
    frozen=dict(schema='hero-full-v3-pretest-freeze-1',model_sha256=ref['checkpoints']['6101']['sha256'],refit_sha256=sha(PUB/'refit.json'),
                split_sha256=n['split_sha256'],physical_development_sha256=sha(PUB/'physical-development.json'),
                rule_sha256=sha(PUB/'physical-rule.json'),rule_status=spec['status'],test_opened=False,
                physical_test_enabled=bool(dev['physical_test_enabled']),coefficients_sha256=None,
                retrieval_enabled=True,scope='One all-ten TEST assay after this freeze; no new seed/radius/model on TEST.')
    if frozen['physical_test_enabled']:
        critic=weights(root,ref['checkpoints']['6101']); ins=train_inputs(('V','I','E')); counts=np.zeros(fc.CELL**2,'i4')
        g=gram(ins,np.arange(8192),spec,critic,counts); out=fit_fixed_alphas(g,dev['alpha']); path=PUB/'physical-frozen-coefficients.npz'
        with path.open('xb') as f: np.savez_compressed(f,**out,inventory_support=counts)
        frozen['coefficients_sha256']=sha(path); frozen['normalization']=dev['normalization']
    savej(PUB/'test-freeze.json',frozen)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('discover','freeze'));p.add_argument('--weights',type=Path,required=True)
    a=p.parse_args(); (discover if a.stage=='discover' else freeze)(a.weights)
