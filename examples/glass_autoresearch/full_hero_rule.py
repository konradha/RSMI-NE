"""One pooled, central-only forward HERO v2 distillate. No assessment reader."""
import argparse
import hashlib
import json
import math
import re
import resource
import time
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy.stats import t

import hero_rule as pilot
from hero_train import BASELINES, SEEDS, center_logits

PUB = Path(__file__).resolve().parents[2] / 'results/l20-autoresearch/hero-full-v2'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/hero-full-v2'
INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9)
FEATURES = tuple(pilot.FEATURES.values())


def apply(v, spec):
    """Frozen signed feature; neither temperature nor a learned network is needed."""
    return pilot.apply_rule(v, spec)


def residual(v, spec):
    """Frozen f(V) minus the global conventional projection of that same f(V)."""
    return pilot.residual_rule(v, spec)


def finite(x):
    return float(x) if np.isfinite(x) else None


def entropy(p):
    return -(p * np.log2(np.maximum(p, 1.e-300))).sum(axis=-1)


def var(m, m2):
    return np.maximum(0., m2 - m*m)


def mean(xs):
    return np.mean(list(xs), axis=0)


def moments(x):
    return np.array([x.mean(dtype='d'), np.mean(np.square(x, dtype='d'))])


def variance(m):
    return float(var(*m))


def fit_alignment(logits, seeds):
    """Per-beta tensors of paired FIT logits. Integer arithmetic for hard tie breaks."""
    if len(seeds) != len(set(seeds)) or seeds != sorted(seeds) or not seeds:
        raise ValueError('expected nonempty ordered distinct seed list')
    base = {}
    for b, ls in logits.items():
        if len(ls) != len(seeds) or any(l.shape != ls[0].shape for l in ls) or not ls[0].size:
            raise ValueError('unaligned seed arrays or empty FIT')
        base[b] = (ls[0].argmax(-1).ravel(), pilot._prob(ls[0]).reshape(-1, 4))
    unit = math.lcm(*(len(h) for h, _ in base.values()))
    weights = {b:unit // len(h) for b, (h, _) in base.items()}
    counts = sum(np.bincount(h, minlength=4).astype(object)*weights[b] for b, (h, _) in base.items())
    mass = mean(p.mean(0) for _, p in base.values())
    ref = sorted(range(4), key=lambda i: (-counts[i], -mass[i], i))
    output = []
    for k, seed in enumerate(seeds):
        conf = np.zeros((4, 4), dtype=object); soft = np.zeros((4, 4))
        for b, ls in logits.items():
            h0, p0 = base[b]; l = ls[k]; h = l.argmax(-1).ravel()
            h0 = np.argsort(ref)[h0]
            conf += weights[b]*np.bincount(4*h0+h, minlength=16).reshape(4,4).astype(object)
            soft += p0[:,ref].T @ pilot._prob(l).reshape(-1,4) / h.size
        ix = np.arange(4)
        perm = ref if k == 0 else min(permutations(range(4)), key=lambda p:
                    (-conf[ix,p].sum(), -soft[ix,p].sum(), p))
        m = mean(moments(ls[k][...,perm[0]].astype('d')-ls[k][...,perm[1]]) for ls in logits.values())
        output.append(dict(seed=seed, aligned_to_raw=list(perm), mean=float(m[0]), sd=math.sqrt(variance(m)),
                           reference_hard_frequency_tie=len(set(counts)) < 4 if k == 0 else None))
    return output


def summarize_symbols(logits, maps, uncertainty=False):
    """Keep only additive per-temperature statistics for pooling; entropy is nonlinear."""
    states = {}; local = {}; accum = []
    parents={m['seed']:{} for m in maps}
    for b, ls in logits.items():
        zs=[]; hs=[]; sums=[]
        for l, m in zip(ls, maps, strict=True):
            perm = m['aligned_to_raw']; p = pilot._prob(l)[...,perm]; delta = l[...,perm[0]].astype('d')-l[...,perm[1]]
            h=np.argsort(perm)[l.argmax(-1)]; hs.append(h)
            zs.append((delta-m['mean'])/m['sd'] if m['sd']>0 else np.zeros_like(delta))
            rows=np.column_stack((*[(h==i).mean(1) for i in range(4)],p.mean(1),entropy(p).mean(1),
                                  ((l==l.max(-1,keepdims=True)).sum(-1)>1).mean(1),delta.mean(1),(delta*delta).mean(1)))
            sums.append(rows.mean(0))
            if uncertainty: parents[m['seed']][b]=rows
        states[b]=(mean(zs),zs,hs)
        a = np.array(sums); accum.append(a)
        local[b] = symbol_record(a, maps, hs)
    result=dict(per_beta=local,equal_beta=symbol_record(mean(accum),maps))
    if uncertainty:
        def rate(a):
            draw=entropy(a[...,4:8])
            return np.concatenate((a[...,:8],np.stack((entropy(a[...,:4]),draw,draw-a[...,8],
                       a[...,8],a[...,9],np.sqrt(var(a[...,10],a[...,11]))),axis=-1)),axis=-1)
        names=[f'{code}_mass_{k}' for code in ('hard','draw') for k in range(4)]+[
            'hard_entropy','draw_entropy','draw_information','conditional_entropy','argmax_ties','raw_contrast_sd']
        result['parent_intervals_by_seed']={seed:across(data,rate,names) for seed,data in parents.items()}
    return states,result


def symbol_record(values, maps, hs=None):
    out = []
    for j, (a, m) in enumerate(zip(values, maps, strict=True)):
        h, p = a[:4], a[4:8]
        out.append(dict(seed=m['seed'], hard_mass=h.tolist(), hard_counts=None if hs is None else
                        np.bincount(hs[j].ravel(), minlength=4).tolist(), hard_entropy_bits=float(entropy(h)),
                        draw_mass=p.tolist(), draw_entropy_bits=float(entropy(p)), mean_conditional_entropy_bits=a[8],
                        draw_information_bits=entropy(p)-a[8], raw_argmax_tie_fraction=a[9],
                        raw_log_odds_mean=a[10], raw_log_odds_sd=math.sqrt(variance(a[10:12]))))
    return out


def interval(rows, fn, names, alpha=.05, simultaneous=1, weights=None):
    """Stratified delete-one-parent uncertainty for a function of equal-stratum moments."""
    if weights is None: weights=[np.ones(len(a)) for a in rows]
    mus = [np.average(a,axis=0,weights=w) for a,w in zip(rows,weights,strict=True)]
    mix = mean(mus); B = len(rows)
    est = np.atleast_1d(fn(mix)); parts = []
    for a, mu, w in zip(rows, mus, weights, strict=True):
        n = len(a)
        if n < 2:
            parts.append(np.full(len(est), np.nan)); continue
        leave = fn(mix+(mu-a)*(w/(w.sum()-w)/B)[:,None])
        parts.append((n-1)*np.mean((leave-leave.mean(0))**2, axis=0))
    parts = np.asarray(parts); vs = parts.sum(0); se = np.sqrt(vs)
    df_den = np.sum(parts**2/np.array([max(1,len(a)-1) for a in rows])[:,None], axis=0)
    df = np.divide(vs**2, df_den, out=np.full_like(vs, np.inf), where=df_den>0)
    out = {}
    for j, key in enumerate(names):
        s = se[j]; x = est[j]; d = df[j]
        radius = t.ppf(1-alpha/2,d)*s; lower = x-t.ppf(1-alpha/simultaneous,d)*s
        out[key] = dict(estimate=finite(x), se=finite(s), df=finite(d) if np.isfinite(s) else None,
                        ci95=[finite(x-radius), finite(x+radius)], lower95_one_sided=finite(lower))
    return out


def across(rows, fn, names, simultaneous=1):
    return dict(equal_beta=interval(list(rows.values()),fn,names,simultaneous=simultaneous),
                per_beta={b:interval([a],fn,names,simultaneous=simultaneous) for b,a in rows.items()})


def ratio(a,b):
    a, b = np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>1e-14)


def pair_metric(a):
    chance = (a[...,6:10]*a[...,10:14]).sum(-1)
    den = np.sqrt(var(a[...,0],a[...,2])*var(a[...,1],a[...,3]))
    return np.stack([ratio(a[...,4]-a[...,0]*a[...,1],den),ratio(a[...,5]-chance,1-chance),a[...,5]],axis=-1)


def reproducibility(dev, seeds):
    pairs = list(combinations(range(len(seeds)),2)); passed = len(seeds)>=3; out=[]
    for i,j in pairs:
        rows={}; shift={}
        for b,(_,z,h) in dev.items():
            rows[b]=pilot._pair_row(z[i],z[j],h[i],h[j]); n=len(z[i])
            if n>1:
                shift[b]=pilot._pair_row(z[i],np.roll(z[j],1,axis=0),h[i],np.roll(h[j],1,axis=0)).mean(0)
        r=across(rows,pair_metric,('r','kappa','agreement'),simultaneous=2*len(pairs))
        for name, point, low in [('r',.60,.40),('kappa',.30,.10)]:
            m=r['equal_beta'][name]; passed &= (m['estimate'] is not None and m['estimate']>=point
                    and m['lower95_one_sided'] is not None and m['lower95_one_sided']>low)
        r['seeds']=[seeds[i],seeds[j]]
        r['shift_r_kappa_agreement_per_beta']={b:[finite(x) for x in pair_metric(shift[b])] if b in shift else None for b in dev}
        r['shift_r_kappa_agreement_equal_beta']=[finite(x) for x in pair_metric(mean(shift.values()))] if len(shift)==len(dev) else None
        out.append(r)
    return out, bool(passed)


def sufficient(words, states):
    """Raw moments; never average within-beta variances in place of a mixture variance."""
    result = {}
    for b,v in words.items():
        ym,yv,xm,xv,cxy = pilot._moments(v,states[b][0]); n = len(xm)
        result[b] = np.stack((np.full(n,ym),np.full(n,yv+ym*ym),xm,xv+xm*xm,cxy+ym*xm))
    return result


def scores(s,a,b):
    ym,y2,xm,x2,xy=s
    mse=np.maximum(0,y2+a*a+b*b*x2+2*a*b*xm-2*a*ym-2*b*xy)
    return 1-ratio(mse,var(ym,y2))


def rank(fit, dev):
    f=mean(fit.values()); ym,y2,xm,x2,xy=f; xv=var(xm,x2)
    slopes=np.divide(xy-ym*xm,xv,out=np.zeros_like(xm),where=xv>1e-14); a=ym-slopes*xm
    fs=scores(f,a,slopes); ds=scores(mean(dev.values()),a,slopes)
    local={b:scores(s,a,slopes) for b,s in dev.items()}
    rows=[]
    for j,feat in enumerate(FEATURES):
        ok=xv[j]>1e-14
        rows.append(dict(feature=feat.name,cost=feat.cost,conventional=feat.conventional,
                         intercept=float(a[j]),slope=float(slopes[j]),fit_mean=float(xm[j]),fit_variance=float(xv[j]),
                         fit_r2=finite(fs[j]) if ok else None,odd_dev_r2=finite(ds[j]) if ok else None,
                         odd_dev_r2_per_beta={b:finite(sc[j]) if ok else None for b,sc in local.items()}))
    valid=[r for r in rows if r['odd_dev_r2'] is not None]
    best_fit=min((r for r in rows if r['fit_r2'] is not None),key=lambda r:(-r['fit_r2'],r['cost'],r['feature']),default=None)
    if not valid: return rows,best_fit,None,None
    high=max(r['odd_dev_r2'] for r in valid)
    chosen=min((r for r in valid if r['odd_dev_r2']>=high-.02),key=lambda r:(r['cost'],-r['odd_dev_r2'],r['feature']))
    rival=min((r for r in valid if r is not chosen),key=lambda r:(-r['odd_dev_r2'],r['cost'],r['feature']),default=None)
    return rows,best_fit,chosen,rival


def fidelity(words,states,chosen,rival):
    rows={}
    for b,v in words.items():
        y=states[b][0]; cols=[y.mean(1),np.square(y).mean(1)]
        for row in (chosen,rival):
            if row is not None:
                error=y-row['intercept']-row['slope']*pilot.feature_value(v,row['feature'])
                cols.append(np.square(error).mean(1))
        rows[b]=np.column_stack(cols)
    def metric(m):
        v=var(m[...,0],m[...,1]); r2=1-ratio(m[...,2],v)
        return np.stack([r2,ratio(m[...,3]-m[...,2],v)] if rival is not None else [r2],axis=-1)
    return across(rows,metric,['selected_R2','selected_minus_best_other_R2'] if rival is not None else ['selected_R2'])


def projection(words, spec):
    """One set of coefficients across the entire panel; each test parent counts once."""
    folds = {b:[pilot._sufficient(v,spec,k) for k in (0,1)] for b,v in words.items()}
    def mix(folds):
        return tuple(mean(z[i]/z[0][0,0] for z in folds) for i in (0,1))
    total = [tuple(x+y for x,y in zip(*fold,strict=True)) for fold in folds.values()]
    gram, groups = mix(total)
    models=pilot._models(gram,groups); errors={key:0. for key in models}
    complete=all(z[0][0,0]>0 for fold in folds.values() for z in fold)
    if complete:
        for k in (0,1):
            validation=1-k
            for name,m in pilot._models(*mix([fold[k] for fold in folds.values()])).items():
                errors[name]+=mean(pilot._mse(*fold[validation],m)*fold[validation][0][0,0]/
                    (fold[0][0][0,0]+fold[1][0][0,0]) for fold in folds.values())
    else:
        errors={key:None for key in errors}
    if pilot.FEATURES[spec['feature']].conventional:
        chosen=dict(name='exact_dictionary_identity'); errors['exact_dictionary_identity']=0.
    elif complete:
        chosen=next(m for name,m in models.items() if errors[name]<=min(errors.values())+1e-12)
    else:
        chosen=None
    return chosen,dict(fit_parent_parity_oof_mse=errors,within_S_fit_variance=pilot._mse(gram,groups,models['stratum']),
                       populated_fit_strata=int(np.count_nonzero(groups[:,0])),complete_folds=complete)


def control_report(fit,dev,reference):
    """Weighted empirical CDF, all FIT parents, no sitewise splits or random tie jitter."""
    out={}
    unit=math.lcm(*(len(v)*8 for v in fit.values()))
    weights={b:unit//(len(v)*8) for b,v in fit.items()}; total=unit*len(fit)
    mass=sorted((sum(weights[b]*reference[b][0]['hard_counts'][i] for b in fit) for i in range(4)),reverse=True)
    for key in BASELINES:
        x={b:pilot._mapped(v,lambda w:pilot._control_value(w,key)).ravel() for b,v in fit.items()}
        sizes=mass if key.startswith('random_') else [1,1,1,1]
        target=[n/sum(sizes) for n in sizes]
        cuts=[]
        if key!='site':
            # Binary search the finite support; exact integer mixture masses. In
            # particular do not turn a CDF equality on a discrete atom into a gap.
            vals=np.unique(np.concatenate(list(x.values()))); ordered={b:np.sort(a) for b,a in x.items()}
            for k in (1,2,3):
                lo=0; hi=len(vals)-1; threshold=total*sum(sizes[:k])
                while lo<hi:
                    mid=(lo+hi)//2
                    cnt=sum(weights[b]*int(np.searchsorted(a,vals[mid],side='right')) for b,a in ordered.items())
                    if cnt*sum(sizes)>=threshold: hi=mid
                    else: lo=mid+1
                cuts.append(float(vals[lo]))
        def summarize(values):
            rates={}; counts={}
            for b,a in values.items():
                h=a.astype(int) if key=='site' else np.searchsorted(cuts,a,side='left')
                counts[b]=np.bincount(h.ravel(),minlength=4); rates[b]=counts[b]/h.size
            p=mean(rates.values())
            return dict(equal_beta=dict(mass=p.tolist(),entropy_bits=float(entropy(p))),
                        per_beta={b:dict(counts=counts[b].tolist(),mass=q.tolist(),entropy_bits=float(entropy(q))) for b,q in rates.items()})
        out[key]=dict(control=key,cuts=cuts,seed=71991,target_mass=target if key!='site' else None,
                      fit=summarize(x),odd_dev=summarize({b:pilot._mapped(v,lambda w:pilot._control_value(w,key)) for b,v in dev.items()}))
    return out


def output_moments(words,spec):
    stats={}; raw={}
    for b,v in words.items():
        f=apply(v,spec); r=residual(v,spec); raw[b]=np.r_[moments(f),moments(r)]
        stats[b]=dict(parents=len(v),anchors=f.size,f_mean=float(f.mean()),f_variance=float(f.var()),
                      residual_mean=float(r.mean()),residual_variance=float(r.var()),projection_mse=float(np.square(r).mean()))
    a=mean(raw.values())
    return dict(per_beta=stats,equal_beta=dict(f_mean=float(a[0]),f_variance=variance(a[:2]),
                 residual_mean=float(a[2]),residual_variance=variance(a[2:]),projection_mse=float(a[3])))


def conditional_variance(fit,dev,targets):
    """Descriptive within-S and genuinely held-apart MSE of E_fit[target|S]."""
    out={}; prediction=None
    for role,words,target in zip(('fit','odd_dev'),(fit,dev),targets,strict=True):
        raw={}
        for b,v in words.items():
            s=pilot._mapped(v,lambda w:w.stratum()).astype(int).ravel(); y=target[b].ravel()
            raw[b]=np.stack([np.bincount(s,weights=w,minlength=pilot.N_STRATA)/len(s)
                             for w in (None,y,y*y)])
        mix=mean(raw.values())
        if prediction is None:
            prediction=np.divide(mix[1],mix[0],out=np.full_like(mix[1],mix[1].sum()),where=mix[0]>0)
        def stats(a):
            p,u,v=a; local=np.divide(u,p,out=np.zeros_like(u),where=p>0)
            return dict(populated_strata=int((p>0).sum()),total_variance=float(var(u.sum(),v.sum())),
                        within_S_variance=float(max(0.,v.sum()-u@local)),
                        frozen_fit_stratum_MSE=float(max(0.,v.sum()-2*u@prediction+p@(prediction**2))))
        out[role]=dict(per_beta={b:stats(a) for b,a in raw.items()},equal_beta=stats(mix))
    return out


def distill(fit,dev,models,seeds,source,geometry=True):
    start=time.perf_counter(); cpu=time.process_time()
    if set(fit)!=set(dev) or not fit or len(models)!=len(seeds):
        raise ValueError('incomplete paired beta/seed panel')
    for cohort in (fit,dev):
        for v in cohort.values():
            if v.shape[1:]!=(8,27) or v.dtype!=np.uint8:
                raise ValueError('expected eight registered 27-site uint8 center words per parent')
    report=dict(schema='full-hero-v2-rule-development-1',source=source,
                supports={b:dict(fit_parents=len(v),odd_dev_parents=len(dev[b]),fit_anchors=v.shape[0]*8,
                    odd_dev_anchors=len(dev[b])*8,odd_dev_fewer_than_eight=len(dev[b])<8) for b,v in fit.items()},
                note='Selection-conditional development only; no exam, causal or independent confirmation claim.')
    spec=dict(schema='full-hero-v2-rule-1',source=source,feature=None,fit_variance=None,residual_fraction=None,
              symbolic_passed=False,rediscovery=False,eligible_for_forecast_gate=False,status='no_defined_teacher_or_candidate')
    if any(not len(v) for role in (fit,dev) for v in role.values()):
        spec['status']='required_temperature_missing_no_pooled_choice'
        return spec,report
    def logits(words): return {b:[center_logits(m,v) for m in models] for b,v in words.items()}
    lf=logits(fit); ld=logits(dev); maps=fit_alignment(lf,seeds)
    fs,fsymbol=summarize_symbols(lf,maps); ds,dsymbol=summarize_symbols(ld,maps,uncertainty=True)
    del lf,ld
    pairs,stable=reproducibility(ds,seeds)
    stable &= all(r['raw_log_odds_sd']>=.25 and min(r['hard_mass'])>=.02
                  for r in fsymbol['equal_beta']+dsymbol['equal_beta'])
    rows,bfit,chosen,rival=rank(sufficient(fit,fs),sufficient(dev,ds))
    report.update(alignment=maps,fit_symbols=fsymbol,odd_dev_symbols=dsymbol,odd_dev_pairwise=pairs,
                  teacher_reproducible=bool(stable),dictionary=rows,best_fit_only=bfit,selection=chosen,
                  comparison_best_other=rival,hard_controls=control_report(fit,dev,fsymbol['per_beta']),
                  teacher_given_S=conditional_variance(fit,dev,[{b:s[0] for b,s in states.items()} for states in (fs,ds)]))
    if geometry:
        report['geometry_secondary']=geometry_report(fit,dev,models,maps)
    if chosen is not None:
        name=chosen['feature']; f=pilot.FEATURES[name]; slope=chosen['slope']
        spec.update(feature=name,formula=f.formula,symmetry=f.symmetry,sign=int(np.sign(slope)),
                    fit_mean=chosen['fit_mean'],fit_sd=math.sqrt(chosen['fit_variance']),rediscovery=f.conventional,
                    units='one global equal-beta FIT population SD of the selected central feature',
                    domain='27 open sites in lexicographic (z,y,x), 0=vac,1=A,2=B; signed fixed frame, no beta argument',
                    teacher_fit_affine=dict(intercept=chosen['intercept'],slope=slope))
        report['odd_dev_fidelity']=fidelity(dev,ds,chosen,rival)
        if chosen['fit_variance']<1e-8 or not slope:
            spec.update(status='degenerate_feature_or_zero_sign_no_readout')
        else:
            m,info=projection(fit,spec); report['conventional_fit']=info
            spec['symbolic_passed']=bool(stable and chosen['odd_dev_r2']>=.50)
            spec['conventional_model']=m
            if m is None:
                spec['status']='insufficient_parent_support_for_conventional_crossfit'
            else:
                spec['conventional_feature_order']=[g.name for g in pilot.CONVENTIONAL] if m.get('coef') is not None else []
                fm=output_moments(fit,spec); dm=output_moments(dev,spec)
                vf=fm['equal_beta']['f_variance']; vr=fm['equal_beta']['residual_variance']; frac=vr/vf
                spec.update(fit_variance=vf,fit_residual_variance=vr,residual_fraction=frac,
                            eligible_for_forecast_gate=bool(spec['symbolic_passed'] and not f.conventional and frac>=.10))
                if not stable: status='unstable_teacher_descriptor_only'
                elif not spec['symbolic_passed']: status='insufficient_odd_dev_fidelity_descriptor_only'
                elif f.conventional: status='conventional_rediscovery_no_exam'
                elif frac<.10: status='conventional_projection_explains_descriptor_no_exam'
                else: status='symbolic_nominee_pending_boundary_gate'
                spec['status']=status
                report.update(fit_readout=fm,odd_dev_readout=dm)
                report['readout_given_S']=conditional_variance(fit,dev,[{b:apply(v,spec) for b,v in words.items()} for words in (fit,dev)])
    report.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-start,
                  peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,incremental_gpu_usd=0)
    return spec,report


def geometry_group():
    mats=[m*s for m in pilot.rotations() for s in (1,-1)]
    mats.sort(key=lambda m:(not np.array_equal(m,pilot.STEP),np.linalg.det(m)<0,*m.ravel()))
    return [(m,np.array([pilot.INDEX[tuple(r@m)] for r in pilot.XYZ])) for m in mats]


def match_strata(v,seed,beta):
    """Return recipient, donor and eligibility counts. No parent straddles a block."""
    rng=np.random.default_rng([seed,beta]); s=pilot._mapped(v,lambda w:w.stratum()).astype(int)
    q=s.shape[1]; recipient=[]; donors=[]
    for start in range(0,len(v),16):
        g=s[start:start+16].ravel()
        for cell in np.unique(g):
            ix=np.flatnonzero(g==cell); p=ix//q; parents=np.unique(p)
            if len(parents)<2: continue
            for i in ix:
                other=rng.choice(parents[parents!=i//q]); k=rng.choice(ix[p==other])
                recipient.append(start*q+i); donors.append(start*q+k)
    r=np.asarray(recipient,dtype=np.intp); d=np.asarray(donors,dtype=np.intp)
    count=np.bincount(r//q,minlength=len(v))
    return r,d,count


def pair_columns(l,perm):
    h=np.argsort(perm)[l.argmax(axis=-1)]
    return l[...,perm[0]].astype('d')-l[...,perm[1]],h


def search_geometry(train,model,refmodel,refperm,matches,modes,group):
    """Fix one transform and raw class permutation against a fixed reference."""
    perms=np.array(list(permutations(range(4)))); a=np.arange(4); refs={}
    for b,v in train.items():
        l=center_logits(refmodel,v)
        refs[b]=(np.argsort(refperm)[l.argmax(-1)].ravel(),pilot._prob(l)[...,refperm].reshape(-1,4))
    candidates={key:[] for key in modes}; source={}
    for key in modes:
        source[key]={}
        for b,v in train.items():
            r,d,c=matches[b]
            if key=='full':
                r=d=np.arange(v.shape[0]*8); c=np.full(len(v),8)
            elif key=='matched_genuine': d=r
            n=np.count_nonzero(c)
            source[key][b]=(r,d,(840//c[r//8]).astype('d'),n)
    for gi,(_,ix) in enumerate(group):
        hs={}; ps={}
        for b,v in train.items():
            l=center_logits(model,v[...,ix]); hs[b]=l.argmax(-1).ravel(); ps[b]=pilot._prob(l).reshape(-1,4)
        for key in modes:
            valid=source[key]; denom=math.lcm(*(n for _,_,_,n in valid.values()))
            conf=np.zeros((4,4),dtype=object); dot=np.zeros((4,4))
            for b,(r,d,w,n) in valid.items():
                h,p=refs[b]
                conf+=np.bincount(4*h[r]+hs[b][d],weights=w,minlength=16).astype('i8').reshape(4,4).astype(object)*(denom//n)
                dot+=(p[r]*(w/(840*n))[:,None]).T@ps[b][d]
            hard=conf[a,perms].sum(1); soft=dot[a,perms].sum(1)/len(valid)
            for pi,perm in enumerate(perms):
                candidates[key].append((int(hard[pi]),float(soft[pi]),gi,tuple(map(int,perm))))
    out={}
    for key,rows in candidates.items():
        denom=math.lcm(*(n for _,_,_,n in source[key].values()))*840*len(train)
        def choose(rr):
            h,s,gi,perm=min(rr,key=lambda row:(-row[0],-row[1],row[2],row[3]))
            m,_=group[gi]
            return dict(matrix=m.tolist(),det=int(round(np.linalg.det(m))),index=gi,aligned_to_raw=list(perm),
                        fit_hard_agreement=h/denom,fit_soft_dot=s,maximum_hard_fits=sum(c[0]==h for c in rr),
                        hard_and_soft_tied_fits=sum(c[0]==h and abs(c[1]-s)<=1e-12 for c in rr),search_size=len(rr))
        out[key]=dict(fixed_frame=choose(rows[:24]),geometry=choose(rows))
    return out


def geo_vector(x,y,h,k):
    return np.column_stack((x,y,x*x,y*y,x*y,h==k,*[h==i for i in range(4)],*[k==i for i in range(4)]))


GEO_NAMES=('fixed_r','fixed_kappa','fixed_agreement','geometry_r','geometry_kappa','geometry_agreement',
           'delta_r','delta_kappa','delta_agreement','remaining_disagreement','disagreement_fraction_removed')


def geo_metrics(a):
    x=pair_metric(a[...,:14]); y=pair_metric(a[...,14:]); gap=y-x
    return np.concatenate((x,y,gap,(1-y[...,2,None]),ratio(gap[...,2],1-x[...,2])[...,None]),axis=-1)


def geo_compare(v,lref,refperm,model,selection,match,group,mode):
    r,d,c=match
    if mode=='full':
        r=d=np.arange(len(v)*8); c=np.full(len(v),8)
    elif mode=='matched_genuine':
        d=r
    if not len(r): return None
    x,h=pair_columns(lref.reshape(-1,4)[r],refperm); cols=[]
    for method in ('fixed_frame','geometry'):
        m=selection[method]; l=center_logits(model,v[...,group[m['index']][1]])
        y,k=pair_columns(l.reshape(-1,4)[d],m['aligned_to_raw']); cols.append(geo_vector(x,y,h,k))
    z=np.column_stack(cols); parent=np.zeros((len(v),z.shape[1]))
    np.add.at(parent,r//8,z/c[r//8,None])
    if mode=='full': return parent,np.ones(len(v))
    # The null's reference and donor can reuse a parent inside a block. Delete the
    # whole block for *both* matched analyses and retain per-beta parent weights.
    ids=np.arange(0,len(v),16); weight=np.add.reduceat((c>0).astype(int),ids)
    block=np.add.reduceat(parent,ids,axis=0); ok=weight>0
    return block[ok]/weight[ok,None],weight[ok]


def geometry_report(fit,dev,models,maps):
    start=time.perf_counter(); train={b:v[:256] for b,v in fit.items()}; g=geometry_group()
    matches={}; support={}
    for role,cohort,seed in [('fit',train,26112531),('odd_dev',dev,26112532)]:
        matches[role]={b:match_strata(v,seed,b) for b,v in cohort.items()}
        support[role]={b:dict(total_parents=len(c),matched_parents=int((c>0).sum()),matched_anchors=len(r),
                    unmatched_anchors=len(v)*8-len(r),matched_independent_blocks=len(np.unique(r//(8*16))))
                    for b,v in cohort.items() for r,d,c in [matches[role][b]]}
    modes=['full']
    if all(len(r) for r,d,c in matches['fit'].values()): modes+=['matched_genuine','matched_null']
    refperm=maps[0]['aligned_to_raw']; ldev={b:center_logits(models[0],v) for b,v in dev.items()}
    out=[]
    for model,m in zip(models[1:],maps[1:],strict=True):
        searches=search_geometry(train,model,models[0],refperm,matches['fit'],modes,g)
        comparisons={}
        for mode in modes:
            by={}
            for b,v in dev.items():
                data=geo_compare(v,ldev[b],refperm,model,searches[mode],matches['odd_dev'][b],g,mode)
                if data is not None: by[b]=data
            macro=interval([z for z,w in by.values()],geo_metrics,GEO_NAMES,weights=[w for z,w in by.values()]) if len(by)==len(dev) else None
            comparisons[mode]=dict(equal_beta=macro,per_beta={b:interval([by[b][0]],geo_metrics,GEO_NAMES,weights=[by[b][1]]) if b in by else None for b in dev})
        out.append(dict(seed=m['seed'],fit_search=searches,odd_dev=comparisons))
    spread={}
    for mode in modes:
        spread[mode]={}
        for name in ('fixed_r','geometry_r','delta_r','delta_kappa','remaining_disagreement'):
            vals=[p['odd_dev'][mode]['equal_beta'] for p in out]
            x=[a[name]['estimate'] for a in vals if a and a[name]['estimate'] is not None]
            spread[mode][name]=dict(n=len(x),range=[min(x),max(x)] if x else None,
                                    descriptive_sd=float(np.std(x,ddof=1)) if len(x)>1 else None)
    return dict(reference_seed=maps[0]['seed'],support=support,available_modes=modes,comparisons=out,
                fixed_seed_spread_not_SE=spread,elapsed_seconds=time.perf_counter()-start,
                purpose='secondary only; no new teacher, f, gate, or physical symmetry claim')


def native(rev, v21):
    """One release only. Administrative gates precede any checkpoint/word decoding."""
    import subprocess
    from hero_bank import committed, readj, require, sha
    from full_hero_stream import cutoff, savej
    cutoff('fit-dev')
    require(re.fullmatch('[0-9a-f]{40}',rev) and subprocess.run(
            ['git','-C',str(PUB.parents[2]),'merge-base','--is-ancestor',rev,'HEAD']).returncode==0,
            'explicit reachable immutable release commit required')
    prefix='discovery-v21-' if v21 else ''
    output=PUB/(prefix+'pooled-rule.json'); detail=PUB/(prefix+'pooled-rule-development.json')
    require(not output.exists() and not detail.exists(),'rule stage already spent')
    release_file=PUB/(prefix+'pooled-rule-release-1.json'); nominee_file=PUB/(prefix+'nomination.json'); code=Path(__file__).resolve()
    pins={'design_sha256':PUB/'design.json','rule_design_sha256':PUB/'pooled-rule-design.json',
          'nomination_sha256':nominee_file,'lineage_sha256':PUB/'lineage.json'}
    if v21:
        pins.update(panel_design_sha256=PUB/'discovery-v21-design.json',panel_sha256=PUB/'discovery-v21-panel.json')
    for p in [release_file,*pins.values(),code,code.with_name('hero_rule.py'),code.with_name('hero_train.py'),
              code.with_name('hero_bank.py'),code.with_name('full_hero_stream.py'),PUB/'stream-summary.json',
              PUB/'stream-release-1.json',PUB/'support-lock.json',PUB/'pooled-rule-controls.json']:
        committed(p,rev)
    r=readj(release_file); n=readj(nominee_file); lin=readj(pins['lineage_sha256'])
    control=readj(PUB/'pooled-rule-controls.json')
    require(control['passed'] and control['implementation_sha256']==sha(code) and
            control['design_sha256']==sha(pins['rule_design_sha256']),'pooled rule controls not verified')
    require(r['stage']==prefix+'pooled-rule' and all(r[k]==sha(p) for k,p in pins.items())
            and r['rosters_sha256']==lin['rosters_sha256']==sha(WORK/'rosters.npz'),'release pins mismatch')
    schema='discovery-v21-nomination-1' if v21 else 'hero-full-v2-nomination-1'
    require(n.get('schema')==schema and n.get('status')=='qualified'
            and n.get('primary_seed')==6101 and n.get('arch') in ('affine','tanh')
            and n.get('law') in ('uniform','matched') and n.get('denominator')==1 and
            n.get('updates') in (1024,2048,4096,8192,16384,32768,65536)
            and set(n.get('checkpoints',{}))==set(map(str,SEEDS)),'no unique qualified full-panel final teacher')
    indices=INDICES
    if v21:
        panel=readj(pins['panel_sha256']); indices=tuple(panel['eligible_indices'])
        require(len(indices)>=2 and 0 in indices and all(type(i) is int and i in INDICES for i in indices)
                and indices==tuple(sorted(set(indices))) and list(indices)==n.get('eligible_indices')
                and n.get('scope')=='development-only' and n.get('panel_sha256')==r['panel_sha256']
                and panel.get('design_sha256')==r['panel_design_sha256'],
                'prospective discovery panel disagrees with nomination')
    summ=readj(PUB/'stream-summary.json'); receipts={}; stream_release=sha(PUB/'stream-release-1.json')
    require(summ['release_sha256']==stream_release and set(summ['stream_receipts_sha256'])==set(map(str,INDICES)),
            'not the complete registered stream')
    for b in indices:
        p=PUB/f'stream-b{b:02d}.json'; committed(p,rev); s=readj(p)
        require(sha(p)==summ['stream_receipts_sha256'][str(b)] and s['source']=='archived-physical'
                and s['index']==b and s['stage']=='fit-dev' and s['lineage_sha256']==r['lineage_sha256']
                and s['release_sha256']==stream_release and s['rosters_sha256']==r['rosters_sha256'],'wrong stream')
        require((s['roles']['dev']['parents']+1)//2>=8,
                f'index {b}: fewer than eight distinct even DEV parents, registered nominee is impossible')
        receipts[b]=s
    models=[]; checkpoints={}
    for seed in SEEDS:
        rec=n['checkpoints'][str(seed)]
        name=f"{n['arch']}-{n['law']}-d1-s{seed}-u{n['updates']}.npz"
        path=WORK/'checkpoints'/name
        require(rec['path']==name and path.resolve()==path and sha(path)==rec['sha256'],'wrong checkpoint identity')
        with np.load(path,allow_pickle=False) as z:
            keys=['vW','vb']+(['v1W','v1b'] if n['arch']=='tanh' else [])
            shapes=[(16,4),(4,),(54,16),(16,)] if n['arch']=='tanh' else [(54,4),(4,)]
            model={k:z[k] for k in keys}
            require(all(model[k].shape==sh and model[k].dtype==np.float32 and np.isfinite(model[k]).all()
                    for k,sh in zip(keys,shapes,strict=True)),'invalid K4 center')
            models.append(model); checkpoints[seed]=rec
    fit={}; dev={}
    with np.load(WORK/'rosters.npz',allow_pickle=False) as roster:
        for b in indices:
            ids=[]
            for role,cohort in [('fit',fit),('dev',dev)]:
                folder=WORK/'streams'/f'b{b:02d}'/role; meta=receipts[b]['roles'][role]
                parent,V=[np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('parent','V')]
                rows=roster[f'b{b:02d}_{role}']
                require(parent.shape==rows.shape==(meta['parents'],) and parent.dtype=='<i4'
                        and V.shape==(len(parent),64,27) and V.dtype==np.uint8,'unexpected stream shape')
                if role=='dev':
                    V=V[1::2]; parent=parent[1::2]; rows=rows[1::2]
                    for key,arr in [('parent',parent),('V',V)]:
                        digest=hashlib.sha256()
                        for j in range(0,len(arr),64): digest.update(np.ascontiguousarray(arr[j:j+64]))
                        require(digest.hexdigest()==meta['odd_sha256'][key],'odd role mismatch')
                else:
                    for key in ('parent.npy','V.npy'):
                        require(sha(folder/key)==meta['files'][key]['sha256'],'FIT hash mismatch')
                require(np.array_equal(rows,parent) and len(np.unique(parent))==len(parent),'wrong parent role/order')
                ids.append(parent); cohort[b]=np.ascontiguousarray(V[:,::8])
            require(not np.intersect1d(*ids).size,'overlapping FIT and odd DEV parents')
    source=dict(release_commit=rev,release_sha256=sha(release_file),nomination_sha256=sha(nominee_file),
                checkpoint_receipts=checkpoints,implementation_sha256=sha(code),rule_design_sha256=r['rule_design_sha256'],
                rosters_sha256=r['rosters_sha256'],stream_receipts_sha256=summ['stream_receipts_sha256'],
                excluded_indices={'6':'permanent embargo; no observations or inferred score'},
                roles=['fit','dev[1::2]'],anchor_ordinals=list(range(0,64,8)))
    if v21:
        source.update(scope='development-only',panel_indices=list(indices),panel_sha256=r['panel_sha256'],
                      panel_design_sha256=r['panel_design_sha256'],neural_fit_indices=list(INDICES))
        source['excluded_indices'].update({str(b):'trained but excluded by prespecified discovery support; no rule estimate'
                                           for b in INDICES if b not in indices})
    spec,report=distill(fit,dev,models,list(SEEDS),source)
    if v21:
        spec.update(schema='discovery-v21-rule-1',assessment_permitted=False)
        if spec['eligible_for_forecast_gate']:
            spec['status']='eligible_for_same_f_development_only_no_exam'
        report.update(schema='discovery-v21-rule-development-1',assessment_permitted=False)
    cutoff('fit-dev'); savej(detail,report); savej(output,spec)
    print(json.dumps({'feature':spec['feature'],'status':spec['status']}),flush=True)


def controls():
    """Independent finite-word fixtures only. No archive, feature stream, shell, GPU or MCMC."""
    start=time.perf_counter(); rng=np.random.default_rng(26112533); checks={}
    def check(name,ok,**data): checks[name]=dict(passed=bool(ok),**data)
    def words(sizes):
        return {b:rng.choice(3,size=(n,8,27),p=p).astype('u1') for b,(n,p) in enumerate(
                zip(sizes,[(.21,.39,.40),(.42,.30,.28)],strict=True))}
    fit=words([107,37]); dev=words([39,48]); seeds=[7,13,19]
    # Exactly one linear count contrast and two minor symbols accessible to a K4
    # categorical encoder. Site-specific events have no rotations in the primary.
    W=np.zeros((54,4),'f'); c=np.zeros(4,'f'); W[::2,0]=1; W[::2,1]=-1
    c[0]=27*.25-9.375; c[1]=-c[0]
    for j,ij in [(2,[0,2]),(3,[32,34])]: W[ij,j]=32; c[j]=-32
    # Simultaneous rare events must not tie in the gauge-invariance fixture.
    # Tied raw argmax is a separate deliberately non-invariant control below.
    c[3]-=.5
    models=[]
    for k,p in enumerate([(0,1,2,3),(2,0,3,1),(3,1,0,2)]):
        gauge=np.zeros((54,1),'f'); gauge[::5]=k*.125
        models.append(dict(vW=W[:,p]+gauge,vb=c[list(p)]+2*k))
    spec,rep=distill(fit,dev,models,seeds,dict(kind='abstract fixed K4 arithmetic fixture'),geometry=False)
    raw={b:(v==0).sum(-1) for b,v in fit.items()}
    avg=mean(x.mean() for x in raw.values()); vv=mean(((x-avg)**2).mean() for x in raw.values())
    check('equal_beta_count_variance',abs(spec['fit_mean']-avg)<1e-12 and abs(spec['fit_sd']**2-vv)<1e-10 and
          abs(avg-np.concatenate([x.ravel() for x in raw.values()]).mean())>.1,raw_mean=float(avg),raw_variance=float(vv),
          macro_f_variance=spec['fit_variance'],per_beta_counts={k:v.size for k,v in raw.items()})
    pairs=[p['equal_beta'] for p in rep['odd_dev_pairwise']]
    check('planted_categorical_gauge',spec['feature']=='count:vac' and spec['symbolic_passed'] and
          all(p['r']['estimate']>.99999 and p['kappa']['estimate']>.99999 for p in pairs)
          and rep['selection']['odd_dev_r2']>.99999,feature=spec['feature'],sign=spec['sign'],r2=rep['selection']['odd_dev_r2'],
          fit_maps=rep['alignment'],odd_dev_symbols=rep['odd_dev_symbols'],held_apart_fidelity=rep['odd_dev_fidelity'])
    check('exact_conventional_explanation',spec['fit_variance']>.99999 and spec['fit_residual_variance']==0.
          and not spec['eligible_for_forecast_gate'] and spec['status']=='conventional_rediscovery_no_exam'
          and all(np.array_equal(apply(v,spec),pilot.conventional_rule(v,spec)) and not residual(v,spec).any() for v in dev.values()),
          projection=spec['conventional_model'],status=spec['status'],fit_residual_variance=spec['fit_residual_variance'])
    non='AvB_signed:x'; mu,v2=mean(moments(pilot.feature_value(v,non)) for v in fit.values())
    alt=dict(feature=non,sign=-1,fit_mean=mu,fit_sd=math.sqrt(var(mu,v2)))
    alt['conventional_model'],rec=projection(fit,alt)
    ok=all(np.max(abs(residual(v,alt)-(apply(v,alt)-pilot.conventional_rule(v,alt))))<1e-12 for v in dev.values())
    residual_var=output_moments(dev,alt)['equal_beta']['residual_variance']
    check('nonconventional_signed_projection',ok and residual_var>.1,model=alt['conventional_model']['name'],
          odd_dev_residual_variance=residual_var,comparison=rec)
    ctr=rep['hard_controls']; ok=True
    for key,m in ctr.items():
        for role,words_ in [('fit',fit),('odd_dev',dev)]:
            for b,v in words_.items():
                counts=np.bincount(pilot.hard_control(v,m).ravel(),minlength=4).tolist()
                ok &= counts==m[role]['per_beta'][b]['counts']
        if key!='site':
            x={b:pilot._mapped(v,lambda w:pilot._control_value(w,key)) for b,v in fit.items()}
            for cut,tau in zip(m['cuts'],np.cumsum(m['target_mass'])[:3],strict=True):
                ok &= mean((a<cut).mean() for a in x.values())<=tau+1e-12 and mean((a<=cut).mean() for a in x.values())>=tau-1e-12
    check('hard_baselines_and_weighted_cuts',ok,hard_codes={k:dict(cuts=m['cuts'],fit_mass=m['fit']['equal_beta']['mass'],
          odd_dev_H=m['odd_dev']['equal_beta']['entropy_bits']) for k,m in ctr.items()})
    y={b:rng.normal(size=(n,1)) for b,n in [(0,27),(1,83)]}
    ci=interval(list(y.values()),lambda z:z,['mean'])['mean']
    sem=np.sqrt(sum(a.var(ddof=1)/len(a) for a in y.values()))/len(y)
    draws=mean(a[rng.integers(len(a),size=(4096,len(a))),0].mean(1) for a in y.values())
    naive=np.sqrt(sum(np.repeat(a,8,axis=0).var(ddof=1)/(len(a)*8) for a in y.values()))/len(y)
    single=interval([y[0][:1]],lambda z:z,['mean'])['mean']
    check('whole_parent_intervals',abs(ci['se']-sem)<1e-12 and abs(draws.std(ddof=1)/sem-1)<.12 and
          sem>2.8*naive and single['se'] is None and single['ci95']==[None,None],
          parent_jackknife=ci,analytic_se=sem,whole_parent_bootstrap_se=float(draws.std(ddof=1)),naive_site_se=naive,
          singleton=single,bootstrap_replicates=4096)
    blank,br=distill(fit,{0:dev[0],1:dev[1][:0]},models,seeds,{},geometry=False)
    check('empty_beta_no_selection',blank['feature'] is None and not blank['eligible_for_forecast_gate'],
          status=blank['status'],support=br['supports'])
    l={b:[center_logits(m,v) for m in models] for b,v in dev.items()}
    for z in l.values(): z[-1]=np.roll(z[-1],1,axis=0)
    state,_=summarize_symbols(l,rep['alignment']); _,ok=reproducibility(state,seeds)
    check('no_realignment_on_new_parents',not ok,shuffled_seed=seeds[-1])
    word=np.zeros((1,27),'u1'); word[0,pilot.INDEX[(0,0,-1)]]=1; word[0,pilot.INDEX[(0,0,1)]]=2
    group=geometry_group(); j=next(i for i,(m,_) in enumerate(group) if np.array_equal(m,-pilot.STEP))
    exact=pilot.feature_value(word,'AvB_signed:x').item()==1 and pilot.feature_value(word[...,group[j][1]],'AvB_signed:x').item()==-1
    tied=np.array([[[1.,1.,0.,0.]]]); perm=[1,0,2,3]
    _,h0=pair_columns(tied,[0,1,2,3]); _,h1=pair_columns(tied[...,perm],perm)
    check('exact_geometry_and_raw_argmax_tie',exact and len({tuple(ix) for _,ix in group})==48 and
          sum(np.linalg.det(m)>0 for m,_ in group)==24 and h0.item()==0 and h1.item()==1 and
          pilot.feature_value(np.ones((1,27),'u1'),'internal_oo').item()==54,
          matrices=len(group),proper=24,improper=24,all_A_internal_edges=54)
    # Planted inversion and independently permuted symbols: the fixed-frame
    # mismatch must vanish after one learned transform on disjoint test words.
    p=[3,1,0,2]; rotated={'vW':W.reshape(27,2,4)[group[j][1]].reshape(54,4)[:,p],'vb':c[p]}
    geom=geometry_report(fit,dev,[models[0],rotated],rep['alignment'][:2])
    g=geom['comparisons'][0]; point=g['odd_dev']['full']['equal_beta']
    null=g['odd_dev']['matched_null']['equal_beta']; mtrue=g['odd_dev']['matched_genuine']['equal_beta']
    null_k=null['geometry_kappa']['estimate']; real_k=mtrue['geometry_kappa']['estimate']
    check('held_apart_geometric_diagnosis',point['geometry_kappa']['estimate']>.99999 and point['delta_kappa']['estimate']>.02
          and real_k>null_k+.10 and g['fit_search']['matched_null']['geometry']['search_size']==48*24,
          support=geom['support'],fit_search=g['fit_search'],full=point,matched_true_kappa=real_k,
          matched_null_kappa=null_k,geometry_seconds=geom['elapsed_seconds'])
    return dict(kind='finite categorical arithmetic; no physical sources, no training, no exam',passed=all(x['passed'] for x in checks.values()),
                fixture_revision='Initial minor-symbol fixture allowed a 0.0078125 odd-DEV raw argmax tie fraction, making exact hard permutation agreement impossible (rotated kappa 0.98718). Offset only one minor logit by -0.5, retain the same fixture words/acceptance thresholds and explicitly test raw-tie behavior separately.',
                checks=checks,implementation_sha256=pilot._file_sha(__file__),design_sha256=pilot._file_sha(PUB/'pooled-rule-design.json'),
                elapsed_seconds=time.perf_counter()-start,cpu_seconds_primary=rep['cpu_seconds'],
                max_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,incremental_gpu_usd=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='cmd',required=True)
    for name in ('native','native-v21'):
        p=sub.add_parser(name); p.add_argument('--release-commit',required=True)
    sub.add_parser('controls')
    args=parser.parse_args()
    if args.cmd in ('native','native-v21'): native(args.release_commit,args.cmd=='native-v21')
    else:
        result=controls()
        print(json.dumps({k:v['passed'] for k,v in result['checks'].items()}),flush=True)
        if not result['passed']: raise SystemExit('failed synthetic controls; no release permitted')
        pilot._write_json(PUB/'pooled-rule-controls.json',result)
