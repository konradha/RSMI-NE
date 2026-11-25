"""Buffered static Gate 1: inference on frozen per-parent predictions, no fitting.

Scores are log probabilities (higher is better), in bits/star, already averaged
within each parent. All rows share equal parent weights within their cohort.
Use --self-test for the synthetic formula, decision and figure checks.
For release replay, pass --scores results/buffered-static-gate1/parent-scores.zip.
Keep the accompanying audit, freeze/DEV and calibration files in --out.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'results' / 'buffered-static-gate1'
B = 2048
MARGIN = .01
MODELS = ('prior', 'composition', 'base', 'rich')
VIEWS = ('all', 'first')
TARGETS = ('soft', 'native', 'soft_minus_native')
LABELS = ('physical', 'observed_fit', 'shuffle', 'plant', 'nullplant', 'beta0')
GAINS = ((1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2))
GAIN = 'rich_minus_base'
METRICS = MODELS + tuple(f'{MODELS[a]}_minus_{MODELS[b]}' for a, b in GAINS)


def rng(*parts):
    raw = json.dumps(['buffered-static-gate1-bootstrap-v1', *parts], separators=(',', ':')).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(raw).digest()[:16], 'little'))


@dataclass
class Arm:
    label: str
    b: int
    beta: float
    r: int
    ids: np.ndarray
    names: tuple
    # parent, view, target (soft/native/difference), metric
    values: np.ndarray

    @property
    def ctx(self):
        return dict(label=self.label, b=self.b, beta=self.beta, r=self.r)


def read_arm(path):
    with np.load(path, allow_pickle=False) as data:
        label, b, beta, r = str(data['label'].item()), int(data['b'].item()), float(data['beta'].item()), int(data['r'].item())
        if label not in LABELS or r not in (2, 4, 5) or not math.isfinite(beta):
            raise ValueError(f'{path}: invalid arm {(label,b,beta,r)}')
        ids = data['ids']
        if ids.ndim != 2 or ids.shape[1] != 2 or len(ids) < 2 or len(np.unique(ids, axis=0)) != len(ids):
            raise ValueError(f'{path}: expected >=2 unique two-column parent IDs')
        order = np.lexsort((ids[:, 1], ids[:, 0]))
        a = np.stack([data[t] for t in TARGETS[:2]], axis=2).astype('f8', copy=False)
        if a.shape != (len(ids), 2, 2, 4) or not np.isfinite(a).all() or a.max() > 1e-7:
            raise ValueError(f'{path}: malformed or non-log2 absolute parent scores')
        names, cols = list(METRICS), [a, np.stack([a[...,j]-a[...,i] for j,i in GAINS], axis=-1)]
        for key, metric in [('oracle','oracle_minus_nuisance'), ('oracle_base','oracle_minus_base')]:
            if key in data:
                x = data[key].astype('f8', copy=False)
                if x.shape != (len(ids),2,2) or not np.isfinite(x).all():
                    raise ValueError(f'{path}: malformed {key} (expected parent, view, soft/native)')
                cols.append(x[...,None]); names.append(metric)
        if 'oracle' in data and 'oracle_base' in data:
            captured = data['oracle'] - data['oracle_base']
            gain = a[...,3] - a[...,2]
            cols.extend([captured[...,None], (captured+gain)[...,None], (data['oracle_base']-gain)[...,None]])
            names.extend(['base_minus_nuisance', 'rich_minus_nuisance', 'oracle_minus_rich'])
        if label == 'beta0':
            cols.append((a + math.log2(210)))
            names.extend([f'{m}_minus_sector_uniform' for m in MODELS])
        v = np.concatenate(cols, axis=-1)
        return Arm(label, b, beta, r, ids[order], tuple(names),
                   np.concatenate([v, (v[:,:,0]-v[:,:,1])[:,:,None]], axis=2)[order])


def boot_means(x, generator, reps=B):
    """All coordinates share a multinomial parent resample. No resampling of views."""
    n = len(x)
    matrix = np.ascontiguousarray(x.reshape(n, -1))
    out = np.empty((reps, matrix.shape[1]))
    for start in range(0, reps, 64):
        k = min(64, reps-start)
        indices = generator.integers(n, size=(k,n)) + n*np.arange(k)[:,None]
        w = np.bincount(indices.ravel(), minlength=k*n).reshape(k,n).astype('f8')
        out[start:start+k] = (w @ matrix) / n
    return out.reshape(reps, *x.shape[1:])


def support(x):
    pos = np.maximum(x, 0).sum(0); neg = -np.minimum(x, 0).sum(0)
    gross, square = pos+neg, np.einsum('n...,n...->...', x, x)
    def ratio(a,b):
        return np.divide(a, b, out=np.zeros_like(a), where=b>0)
    return dict(positive_parents=(x>0).sum(0), negative_parents=(x<0).sum(0),
                positive_sum=pos, negative_magnitude_sum=neg, signed_sum=x.sum(0),
                gross_sum=gross, sum_squares=square, kish_abs=ratio(gross**2,square),
                signed_to_gross=ratio(x.sum(0),gross), largest_abs_share=ratio(np.abs(x).max(0),gross))


def summarize(x, draws, ctx, names, targets=TARGETS):
    point = x.mean(0)
    low, high = np.quantile(draws, [.025,.975], axis=0)
    extras = support(x)
    for i, view in enumerate(VIEWS):
        for t, target in enumerate(targets):
            for m, metric in enumerate(names):
                j = (i,t,m)
                yield dict(**ctx, view=view, target=target, metric=metric, n=len(x),
                           estimate=point[j].item(), lo=low[j].item(), hi=high[j].item(),
                           **{k:a[j].item() for k,a in extras.items()})


def paired_power(point, draws):
    """Plug-in detection probability at the recovered frozen-plant effect and N.

    A future mean is modeled as point + resampled centered parent error. Shift
    the resampled error interval along with that future mean (fixed CI width).
    This is conditional on the trained heads, not repeated-training power.
    """
    errors = draws - point
    lower = point + errors + np.quantile(errors, .025, axis=0)
    return [*(lower>0).mean(0).tolist(), float((lower>0).all(1).mean())]


def table(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


def all_results(arms):
    estimates, precision, contrasts, paired, powers = [], [], [], [], []
    cache = {}
    for key, arm in sorted(arms.items()):
        x, names = arm.values, arm.names
        draws = boot_means(x, rng('cohort',arm.b,len(x)))
        estimates.extend(summarize(x, draws, arm.ctx, names))
        if arm.label == 'physical':
            j = names.index(GAIN)
            cache[key] = (x[...,j].mean(0), draws[...,j])
            if arm.b in (8,9) and arm.r == 4:
                # Random order depends only on cohort/size and the sorted IDs.
                perm = rng('nested_parent_order',arm.b,len(x)).permutation(len(x))
                for den in (4,2,1):
                    n = max(1,len(x)//den)
                    v = x[perm[:n],:,:,j:j+1] if den != 1 else x[:,:,:,j:j+1]
                    z = boot_means(v, rng('precision',arm.b,n)) if den != 1 else draws[...,j:j+1]
                    precision.extend(summarize(v,z,dict(**arm.ctx, fraction=1/den), (GAIN,)))
        if arm.label in ('plant','nullplant'):
            for i, view in enumerate(VIEWS):
                for m in (GAIN, 'oracle_minus_nuisance', 'oracle_minus_base'):
                    if m in names:
                        j = names.index(m); v = x[:,i,:2,j].mean(0); d = draws[:,i,:2,j]
                        ps,pn,pj = paired_power(v,d)
                        powers.append(dict(**arm.ctx,view=view,metric=m,n=len(x),draws=len(d),
                                           soft=v[0],native=v[1],power_soft=ps,power_native=pn,power_joint=pj))
    for (label,b,r), (v,d) in cache.items():
        if b == 9 or ('physical',9,r) not in cache:
            continue
        # Streams differ by cohort; row j is an independent source/transfer draw.
        s,sd = cache['physical',9,r]; diff = sd-d; bounds = np.quantile(diff,[.025,.975],axis=0)
        src = arms['physical',9,r]; other = arms[label,b,r]
        for i,view in enumerate(VIEWS):
            for t,target in enumerate(TARGETS):
                contrasts.append(dict(source_beta=src.beta,transfer_beta=other.beta, b=b,r=r,view=view,
                                      target=target, metric=GAIN,n_source=len(src.ids),n_transfer=len(other.ids),
                                      estimate=s[i,t]-v[i,t],lo=bounds[0,i,t],hi=bounds[1,i,t]))
    for (label,b,r), arm in arms.items():
        if label not in ('observed_fit','nullplant'):
            continue
        reference,contrast = (('physical','soft_fit_minus_observed_fit') if label == 'observed_fit'
                              else ('plant','plant_on_minus_off'))
        other = arms[reference,b,r]
        # Both contrasts preserve the parent pairing, including on/off pseudo-observations.
        if not np.array_equal(other.ids,arm.ids) or other.names != arm.names:
            raise ValueError(f'{contrast}: parent IDs or metric names disagree at b={b}, r={r}')
        x = other.values - arm.values
        d = boot_means(x,rng('cohort',b,len(x)))
        paired.extend(summarize(x,d,dict(b=b,beta=arm.beta,r=r,contrast=contrast),arm.names))
    return estimates, precision, contrasts, paired, powers


def get_evidence(out):
    def flag(name,key):
        path = out/name
        obj = json.loads(path.read_text()) if path.exists() else {}
        return obj.get(key) is True
    return {'authenticated_crystals':flag('mask-audit.json','authenticated_crystals'),
            'hidden_feature_invariance':flag('mask-audit.json','masked_invariance'),
            'hidden_prediction_invariance':flag('predictor-invariance.json','success')}


def decide(estimates, powers, evidence):
    index = {(z['label'],z['b'],z['r'],z['view'],z['target'],z['metric']):z for z in estimates}
    checks = []
    def check(name,group,ok,rule,**numbers):
        checks.append(dict(name=name,group=group,pass_check=bool(ok),criterion=rule,**numbers))
    for key in ('authenticated_crystals','hidden_feature_invariance','hidden_prediction_invariance'):
        check(key,'control',evidence.get(key) is True,'audited flag must be true')
    def test(name, group, key, field, op, threshold):
        row = index.get(key); value = row[field] if row else None
        ops = {'>':lambda a,b:a>b, '<':lambda a,b:a<b, '>=':lambda a,b:a>=b, '<=':lambda a,b:a<=b}
        check(name, group, value is not None and ops[op](value,threshold), f'{field} {op} {threshold}',
              arm='/'.join(map(str,key)),value=value,n=row['n'] if row else None)
    for b in (9,8):
        k = ('physical',b,4,'all')
        for target, limit in [('soft',MARGIN),('native',0.)]:
            test(f'b{b:02}_{target}_gain','primary',(*k,target,GAIN),'lo','>',limit)
            test(f'b{b:02}_{target}_support','support',(*k,target,GAIN),'kish_abs','>=',100.)
        for field,cmp,limit in [('lo','>',-MARGIN),('hi','<',MARGIN)]:
            test(f'b{b:02}_agreement_{field}','agreement',(*k,'soft_minus_native',GAIN),field,cmp,limit)
    for target in ('soft','native'):
        test(f'contact_{target}','control',('physical',9,2,'all',target,'rich_minus_prior'),'lo','>',0.)
        # The matched shuffle may use a one-star arm; the all view of that
        # bank is its entire retained sample, not a multiply-counted sample.
        for label,b in [('shuffle',9),('nullplant',9),('beta0',0)]:
            test(f'{label}_{target}','control',(label,b,4,'all',target,GAIN),'hi','<',MARGIN)
        test(f'plant_{target}','control',('plant',9,4,'all',target,GAIN),'lo','>',0.)
    for cmp,lim,suffix in [('>=',.0075,'low'),('<=',.0125,'high')]:
        test(f'plant_oracle_margin_{suffix}','control',('plant',9,4,'all','soft','oracle_minus_nuisance'),
             'estimate',cmp,lim)
    p = next((z['power_joint'] for z in powers if (z['label'],z['b'],z['r'],z['view'],z['metric']) ==
              ('plant',9,4,'all',GAIN)),None)
    check('plant_joint_power','control',p is not None and p>=.8,'conditional joint detection proxy >= 0.8',value=p)
    failed = [c for c in checks if not c['pass_check']]
    if any(c['group']=='control' for c in failed):
        verdict = 'INCONCLUSIVE_FAILED_CONTROL'
    elif not failed:
        verdict = 'PASS'
    else:
        excluded = any(index['physical',b,4,'all','soft',GAIN]['hi']<MARGIN for b in (9,8)
                       if ('physical',b,4,'all','soft',GAIN) in index)
        adequate = not any(c['group'] in ('agreement','support') or c.get('value') is None for c in failed)
        verdict = 'MARGIN_EXCLUDED_FOR_THESE_FROZEN_PREDICTORS' if adequate and excluded else 'INCONCLUSIVE'
    return dict(verdict=verdict, disposition='STOP; Gate 2 not authorized', primary_gap=4,
                source_b=9, independent_transfer_b=8, margin_bits=MARGIN, bootstrap_replicates=B,
                interval='pointwise two-sided 95% whole-parent percentile; shared resamples within each cohort',
                power_scope='conditional detection proxy at zero gain null; not prospective refit power, and not power for lower bound >.01',
                physical_exclusion=False,
                primary_rule='intersection of all checks; descriptive arms cannot rescue a failed primary',
                failed_checks=[c['name'] for c in failed],checks=checks)




def rows_for(rows, **spec):
    return [z for z in rows if all(z[k]==v for k,v in spec.items())]


def draw(out, es, precision):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ('#24618c','#bc5525','#337355','#724884')
    def dots(ax, x, records, color, label, mark='o', connect=False):
        y = [z['estimate'] for z in records]
        ax.vlines(x,[z['lo'] for z in records],[z['hi'] for z in records],color=color,lw=1.15)
        ax.plot(x,y,marker=mark,ms=4,color=color,label=label,linewidth=.85,ls='-' if connect else 'none')
    def clean(ax):
        ax.grid(axis='y',color='0.9',lw=.6); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    def save(fig,name):
        fig.savefig(out/name,dpi=165,bbox_inches='tight',facecolor='white'); plt.close(fig)
    phys = rows_for(es,label='physical',view='all')
    rnames = {2:'Contact',4:'Gap R4 · primary',5:'Gap R5 · secondary'}
    # Distinct panel scales keep the contact effect from swallowing the buffered estimates.
    fig,axes = plt.subplots(2,3,figsize=(12.3,6.1),constrained_layout=True,squeeze=False)
    for j,r in enumerate((2,4,5)):
        for i,target in enumerate(TARGETS[:2]):
            ax = axes[i,j]; clean(ax); ax.axhline(0,c='0.5',lw=.8)
            ax.axhline(MARGIN,c='0.4',ls=':',lw=.8)
            z = sorted(rows_for(phys,r=r,metric=GAIN,target=target),key=lambda a:a['beta'])
            if z:
                dots(ax,[a['beta'] for a in z],z,colors[i],target,connect=True)
                for a in z:
                    if a['b'] in (8,9):
                        ax.plot(a['beta'],a['estimate'],marker='s' if a['b']==8 else 'D',mfc='none',mec='k',ms=8)
            if i==0: ax.set_title(rnames[r])
            if j==0: ax.set_ylabel(f'{target.capitalize()} gain (bits/star)')
            if i==1: ax.set_xlabel('Inverse temperature β')
    fig.suptitle('Detailed shell minus fitted conventional predictor')
    fig.supxlabel('Whole-parent 95% intervals; open diamond = source, square = prespecified transfer. Dotted = .01 bit/star.')
    save(fig,'inference-buffered-gains.png')

    fig,axes = plt.subplots(2,3,figsize=(12.3,6.3),constrained_layout=True)
    for j,r in enumerate((2,4,5)):
        for i,target in enumerate(TARGETS[:2]):
            ax = axes[i,j]; clean(ax)
            for m,name in enumerate(MODELS):
                z = sorted(rows_for(phys,r=r,metric=name,target=target),key=lambda a:a['beta'])
                if z: dots(ax,[a['beta'] for a in z],z,colors[m],name,mark=('o','s','^','D')[m],connect=True)
            if i==0: ax.set_title(rnames[r])
            if j==0: ax.set_ylabel(f'{target.capitalize()} log score (bits/star)')
            if i==1: ax.set_xlabel('Inverse temperature β')
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Absolute forecast scores on the same 210 categories')
    fig.supxlabel('Higher is better. All eligible stars averaged within each parent; whole-parent 95% intervals.')
    save(fig,'inference-model-scores.png')

    fig,axes = plt.subplots(2,2,figsize=(10.5,6.9),constrained_layout=True)
    for j,b in enumerate((9,8)):
        for i,view in enumerate(VIEWS):
            ax = axes[i,j]; clean(ax); ax.axhline(0,c='0.4',lw=.8); ax.axhline(MARGIN,c='0.4',ls=':',lw=.8)
            for k,target in enumerate(TARGETS):
                z = sorted(rows_for(precision,b=b,view=view,target=target,metric=GAIN),key=lambda a:a['n'])
                if z: dots(ax,np.arange(len(z))+(k-1)*.09,z,colors[k],
                           ('soft','native','paired soft − native')[k],mark=('o','s','^')[k],connect=True)
            if z: ax.set_xticks(range(len(z)),[str(a['n']) for a in z])
            if j==0: ax.set_ylabel(f'{view.capitalize()}-star view (bits/star)')
            if i==0: ax.set_title('Source · β=3.05' if b==9 else 'Frozen transfer · β≈2.711')
            if i==1: ax.set_xlabel('Nested assessment parents')
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Primary R4 increment: parent precision and first-star sensitivity')
    fig.supxlabel('Nested parent sets, same frozen heads. No additional tuning at any fraction.')
    save(fig,'inference-parent-precision.png')

    fig,axes = plt.subplots(1,2,figsize=(11.5,4.9),constrained_layout=True)
    controls = [('shuffle',9,GAIN,'Shuffle'),('beta0',0,GAIN,'β=0 fit'),('nullplant',9,GAIN,'Plant off'),('plant',9,GAIN,'Plant on')]
    plant = [('plant',9,m,l) for m,l in [('oracle_minus_nuisance','Oracle\n− nuisance'),('base_minus_nuisance','Base\n− nuisance'),('oracle_minus_base','Oracle\n− base'),(GAIN,'Rich\n− base')]]
    for ax,groups,title in zip(axes,(controls,plant),('Nulls and positive control: rich − base','What was left for the rich head?')):
        clean(ax); ax.axhline(0,c='0.4',lw=.8); ax.axhline(MARGIN,c='0.4',ls=':',lw=.8)
        for k,target in enumerate(TARGETS[:2]):
            x,records = [],[]
            for h,(label,b,m,_) in enumerate(groups):
                match = rows_for(es,label=label,b=b,r=4,view='all',target=target,metric=m)
                if match: x.append(h+(k-.5)*.13); records.extend(match)
            if x: dots(ax,x,records,colors[k],target,mark=('o','s')[k])
        ax.set_xticks(range(len(groups)),[g[-1] for g in groups]); ax.set_title(title); ax.set_ylabel('Bits/star')
        ax.legend(fontsize=8,frameon=False)
    fig.suptitle('R4 controls, not physical evidence')
    fig.supxlabel('Whiskers: 95% whole-parent intervals. Oracle − base is achievable residual, not necessarily plant-specific.')
    save(fig,'inference-controls.png')


def development(out, estimates):
    """Reconstruct the frozen DEV selection, without using held scores for choice."""
    path = out/'dev.csv'
    if not path.exists():
        return []
    with path.open() as fp:
        grid = list(csv.DictReader(fp))
    grouped = {}
    for z in grid:
        k = (z['label'],int(z['b']),int(z['r']),float(z['fraction']),z['model'])
        grouped.setdefault(k,[]).append(z)
    links = []
    for (label,b,r,frac,model),zs in sorted(grouped.items()):
        best = min(zs,key=lambda z:(float(z['dev_logloss_bits']),int(z['epoch'])))
        dev = float(best['dev_logloss_bits']); train = float(best['fit_logloss_bits'])
        arch = best['architecture']
        by_ep = {int(z['epoch']):float(z['dev_logloss_bits']) for z in zs if z['architecture']==arch}
        target = 'native' if label=='observed_fit' else 'soft'
        held = rows_for(estimates,label=label,b=b,r=r,view='all',target=target,metric=model) if frac==1 else []
        test = -held[0]['estimate'] if held else None
        links.append(dict(label=label,b=b,r=r,model=model,fraction=frac,fit_parents=int(best['parents']),
                          dev_parents=int(best['dev_parents']),target=target,selected_arch=arch,
                          selected_epoch=int(best['epoch']),active_parameters=int(best['active_parameters']),
                          selected_fit_loss=train,selected_dev_loss=dev,dev_minus_fit=dev-train,
                          dev_improvement_from_epoch0=by_ep[0]-dev,
                          dev_improvement_epoch8_to_16=by_ep[8]-by_ep[16],
                          frozen_held_loss=test,held_minus_dev=test-dev if held else None))
    table(out/'dev-held-linkage.csv',links)
    return links


def calibration(scores, out):
    def collect(stem):
        if scores.is_file():
            with gzip.open(scores.parent/(stem+'.csv.gz'),'rt',newline='') as fp:
                return list(csv.DictReader(fp))
        z = []
        for path in sorted(scores.glob(stem+'-b*.csv')):
            with path.open() as fp:
                z.extend(csv.DictReader(fp))
        return z
    masses = collect('calibration'); confidence = collect('confidence')
    if not masses:
        return []
    keys = ('label','b','beta','r'); groups = {}
    for z in masses:
        groups.setdefault(tuple(z[k] for k in keys),[]).append(z)
    rows = []
    for k,zs in sorted(groups.items()):
        ctx = dict(zip(keys,k))
        if len({int(z['state']) for z in zs})!=210 or len(zs)!=210:
            raise ValueError(f'Invalid sector calibration rows for {k}')
        def arr(field): return np.array([float(z[field]) for z in zs])
        native,teacher = arr('native_mass'),arr('exact_mass')
        parents = arr('obs_parents')
        def tv(a,b): return np.abs(a-b).sum()/2
        for model in ('base','rich'):
            mass = arr(model+'_mass')
            bins = rows_for(confidence,**ctx,model=model)
            if not bins:
                raise ValueError(f'Missing confidence bins for {k} {model}')
            def col(f): return np.array([float(z[f]) for z in bins])
            w = col('parent_weight')
            np.testing.assert_allclose([w.sum(),teacher.sum(),native.sum(),mass.sum()],1,atol=3e-5)
            pred,acc,exact = col('pred_confidence'),col('obs_accuracy'),col('exact_agreement')
            rows.append(dict(**ctx,model=model,observed_states=int((parents>0).sum()),
                             states_seen_in_1_to_5_parents=int(((parents>0)&(parents<=5)).sum()),
                             tv_native_exact=tv(native,teacher),tv_native_model=tv(native,mass),
                             tv_exact_model=tv(teacher,mass),largest_abs_native_exact=np.abs(native-teacher).max(),
                             native_top1=w@acc,exact_top1=w@exact,model_top1=w@pred,
                             binned_abs_native_cal_gap=w@np.abs(pred-acc),binned_abs_exact_cal_gap=w@np.abs(pred-exact)))
    table(out/'calibration-summary.csv',rows)
    return rows


def summary(out, es, contrasts, paired, powers, links, cal, decision):
    def effect(z):
        return f"{z['estimate']:+.5f} [{z['lo']:+.5f}, {z['hi']:+.5f}]"
    lines = [decision['verdict'], decision['disposition'], '',
             'All numbers below: signed bits/star [two-sided 95% whole-parent percentile interval].',
             'Parent is the unit. Soft and native are paired, not independent replications.',
             'Required R4 physical increments (rich minus frozen strongest conventional):']
    for b in (9,8):
        for z in rows_for(es,label='physical',b=b,r=4,view='all',metric=GAIN):
            lines.append(f"  b{b:02} {z['target']}: {effect(z)}, N={z['n']}, Kish(|increment|)={z['kish_abs']:.1f}.")
    for z in rows_for(contrasts,b=8,r=4,view='all'):
        lines.append(f"  b09 minus independent b08 increment, {z['target']}: {effect(z)}.")
    lines += ['','Required contact, null and recovery checks:']
    for label,b,r,m in [('physical',9,2,'rich_minus_prior'),('shuffle',9,4,GAIN),('beta0',0,4,GAIN),
                         ('nullplant',9,4,GAIN),('plant',9,4,GAIN)]:
        for t in TARGETS[:2]:
            for z in rows_for(es,label=label,b=b,r=r,view='all',metric=m,target=t):
                lines.append(f"  {label}, b{b:02} R{r}, {m}, {t}: {effect(z)} (N={z['n']}).")
    uni = -math.log2(210)
    lines.append(f'Sector-uniform oracle absolute log score is {uni:.8f} bits/star on every β=0 soft/native target.')
    for t in TARGETS[:2]:
        for z in rows_for(es,label='beta0',b=0,r=4,view='all',metric='rich_minus_sector_uniform',target=t):
            lines.append(f"  de novo β=0 rich minus exact sector-uniform, {t}: {effect(z)}.")
    lines += ['','Plant decomposition (soft scoring):']
    for m in ('oracle_minus_nuisance','base_minus_nuisance','oracle_minus_base',GAIN):
        for label in ('plant','nullplant'):
            for z in rows_for(es,label=label,r=4,view='all',target='soft',metric=m):
                lines.append(f"  {label} {m}: {effect(z)}.")
        for z in rows_for(paired,contrast='plant_on_minus_off',r=4,view='all',target='soft',metric=m):
            lines.append(f"  paired on minus off, {m}: {effect(z)}.")
    lines += ['Base − nuisance is signed performance relative to the known mass model. It can be negative.',
              'Oracle − fitted base includes nuisance approximation error; its on–off change is also reported.',
              'Thus neither the imposed oracle KL nor total residual is automatically a beyond-base plant increment.']
    for z in rows_for(powers,label='plant',r=4,view='all',metric=GAIN):
        lines.append(f"Conditional bootstrap detection proxy: soft {z['power_soft']:.3f}, noisy native {z['power_native']:.3f}, joint {z['power_joint']:.3f}.")
    recon = out/'descriptor-reconstruction.json'
    if recon.exists():
        metrics = json.loads(recon.read_bytes())['summary']
        for key in ('all_indicators','plant_minus_fit_bin_mean'):
            v = metrics[key]
            lines.append(f"  Conventional-input linear reconstruction, {key}, source DEV R² {v['r2']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}].")
        lines.append('This measures predictability of a shell feature, not the portion of its score gain the conventional decoder captured.')
    lines += ['One frozen fit and one held plant realization. This is not prospective power over data generation/refits;',
              'the detection event is both lower bounds >0, not a lower bound >.01 at a true effect of .01.',
              '', 'Benefit of exact training, identical R4 source parents (q-trained minus native-trained score):']
    for m in ('base','rich',GAIN):
        for z in rows_for(paired,contrast='soft_fit_minus_observed_fit',b=9,r=4,view='all',metric=m):
            lines.append(f"  {m}, {z['target']}: {effect(z)}.")
    lines += ['','Source R4 fit diagnostics (bits/star; lower loss is better).']
    for z in rows_for(links,label='physical',b=9,r=4):
        held = f"; frozen held {z['frozen_held_loss']:.5f}" if z['frozen_held_loss'] is not None else ''
        lines.append(f"  {z['model']}, {z['fit_parents']} training parents: {z['selected_arch']} epoch {z['selected_epoch']}, fit {z['selected_fit_loss']:.5f}, DEV {z['selected_dev_loss']:.5f}{held}.")
    lines += ['The nested fitting-parent curves use the same DEV population, not new evaluations.',
              'The epoch-16 cap limits any claim about the best achievable decoder.',
              '', 'R4 calibration: total variation (TV) of category marginals and parent-weighted top-state reliability gaps.']
    for label,b in [('physical',9),('physical',8),('beta0',0)]:
        for z in rows_for(cal,label=label,b=str(b),r='4',model='rich'):
            lines.append(f"  {label} b{b:02}: {z['observed_states']}/210 native states; TV(native, exact) {z['tv_native_exact']:.5f}, TV(native, rich) {z['tv_native_model']:.5f}; native binned absolute top-confidence gap {z['binned_abs_native_cal_gap']:.5f}.")
    lines += ['', 'Mandatory checks not satisfied: '+(', '.join(decision['failed_checks']) or 'none')+'.',
              'No physical null, length, or phase conclusion. Two gaps and a fitted score increment do not identify conditional MI.',
              '', 'Numerical sources: inference-estimates.csv (absolute, paired gains, signed/gross/squared support);',
              'independent-transfer.csv; nested-precision.csv; paired-fit-controls.csv; plant-detection.csv;',
              'parent-scores.zip (per-parent score banks; higher is better); calibration-summary.csv (category and confidence gaps);',
              'dev.csv: source checkpoint grid; dev-held-linkage.csv: selected fit/DEV loss and corresponding full frozen held loss.',
              'decision.json contains each frozen criterion and its measured value or explicit failure.']
    (out/'inference-summary.txt').write_text('\n'.join(lines)+'\n')


def read_arms(scores):
    if scores.is_dir():
        for path in sorted(scores.glob('*.npz')):
            yield read_arm(path)
    else:
        with ZipFile(scores) as archive:
            for name in sorted(n for n in archive.namelist() if n.endswith('.npz')):
                with archive.open(name) as fp:
                    yield read_arm(fp)


def run(scores, out):
    out.mkdir(parents=True,exist_ok=True)
    arms = {}
    for arm in read_arms(scores):
        key = arm.label,arm.b,arm.r
        if key in arms:
            raise ValueError(f'Duplicate score arm: {key}')
        arms[key] = arm
    if not arms:
        raise ValueError('No frozen score files')
    estimates, precision, contrasts, paired, powers = all_results(arms)
    decision = decide(estimates,powers,get_evidence(out))
    for name,rows in [('inference-estimates',estimates),('nested-precision',precision),
                      ('independent-transfer',contrasts),('paired-fit-controls',paired),('plant-detection',powers)]:
        table(out/f'{name}.csv',rows)
    draw(out,estimates,precision)
    summary(out,estimates,contrasts,paired,powers,development(out,estimates),calibration(scores,out),decision)
    (out/'decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n')
    return decision


def self_test():
    """Deterministic synthetic parent atoms, no access to archive or experiment outputs."""
    x = np.array([[-2.,4.],[1.,-2.],[4.,-8.]])
    s = support(x)
    np.testing.assert_allclose(s['signed_sum'],[3.,-6.])
    np.testing.assert_allclose(s['gross_sum'],[7.,14.])
    np.testing.assert_allclose(s['sum_squares'],[21.,84.])
    np.testing.assert_allclose(s['kish_abs'],[49/21,49/21])
    draws = boot_means(x,rng('test'),19)
    ref = x[rng('test').integers(3,size=(19,3))].mean(1)
    np.testing.assert_array_equal(draws,ref)
    np.testing.assert_array_equal(draws[:,0]*-2,draws[:,1])
    assert not np.array_equal(boot_means(x,rng('cohort',9,3),50),boot_means(x,rng('cohort',8,3),50))
    assert paired_power(np.ones(2),np.ones((10,2))) == [1.,1.,1.]
    assert paired_power(np.array([1.,0.]),np.tile([1.,0.],(10,1))) == [1.,0.,0.]
    with TemporaryDirectory(prefix='synthetic-gate1-') as tmp:
        tmp = Path(tmp); out = tmp/'out'; out.mkdir(); scr = tmp/'scores'; scr.mkdir()
        n = 160; ids = np.column_stack((np.arange(n), np.arange(n)[::-1]))
        specs = [('physical',9,4,.024),('physical',8,4,.018),('physical',9,2,.2),
                 ('physical',9,5,.005),('observed_fit',9,4,.012),('plant',9,4,.008),
                 ('nullplant',9,4,0.),('shuffle',9,4,0.),('beta0',0,4,0.)]
        for i,(label,b,r,gain) in enumerate(specs):
            a = np.broadcast_to(np.array([-8.,-7.9,-7.85,-7.85]),(n,2,4)).copy()
            # Shared non-constant parent variation cancels in the paired discrepancy.
            a[:,:,3] += gain + .0001*np.arange(n)[:,None]/n
            native = a.copy(); native[:,:,3] -= .001
            oracle = {}
            if label in ('plant','nullplant'):
                q = np.full((n,2,2),.01 if label=='plant' else 0.)
                oracle = dict(oracle=q,oracle_base=q*.9)
            np.savez(scr/f'{i}.npz',label=label,b=b,beta={9:3.05,8:2.711,0:0.}[b],r=r,
                     ids=ids,soft=a,native=native,**oracle)
        ctx = dict(label='physical',b=9,beta=3.05,r=4)
        table(scr/'calibration-b09.csv',[dict(**ctx,state=i,obs_parents=n,native_mass=1/210,exact_mass=1/210,
                                              base_mass=1/210,rich_mass=1/210) for i in range(210)])
        table(scr/'confidence-b09.csv',[dict(**ctx,model=m,parent_weight=1.,pred_confidence=.02,
                                             obs_accuracy=.01,exact_agreement=.015) for m in ('base','rich')])
        table(out/'dev.csv',[dict(**ctx,fraction=1.,model='rich',parents=n,dev_parents=n,architecture='tanh',
                                  epoch=ep,fit_logloss_bits=7.7,dev_logloss_bits=loss,active_parameters=17000)
                             for ep,loss in [(0,8.),(8,7.9),(16,7.9)]])
        mask_flags = {'authenticated_crystals':True,'masked_invariance':True}
        (out/'mask-audit.json').write_text(json.dumps(mask_flags))
        (out/'predictor-invariance.json').write_text(json.dumps({'success':True}))
        d = run(scr,out)
        assert d['verdict']=='PASS', d['failed_checks']
        with (out/'calibration-summary.csv').open() as fp:
            c = rows_for(list(csv.DictReader(fp)),model='rich')[0]
        np.testing.assert_allclose([float(c[k]) for k in ('tv_native_model','binned_abs_native_cal_gap')],[0,.01])
        with (out/'dev-held-linkage.csv').open() as fp:
            sel = next(csv.DictReader(fp))
        assert sel['selected_epoch']=='8' and float(sel['frozen_held_loss'])<float(sel['selected_dev_loss'])
        with (out/'inference-estimates.csv').open() as fp:
            exported = list(csv.DictReader(fp))
        a = rows_for(exported,label='plant',view='all',target='soft',metric='base_minus_nuisance')[0]
        np.testing.assert_allclose(float(a['estimate']),.001,atol=1e-12)
        a = rows_for(exported,label='physical',b='9',r='4',view='all',target='soft_minus_native',metric=GAIN)[0]
        np.testing.assert_allclose([float(a[k]) for k in ('estimate','lo','hi')],.001,atol=1e-12)
        with (out/'independent-transfer.csv').open() as fp:
            c = rows_for(list(csv.DictReader(fp)),b='8',r='4',view='all',target='soft')[0]
        np.testing.assert_allclose(float(c['estimate']),.006)
        assert float(c['hi'])>float(c['lo'])  # independent draws, despite matching parent array patterns
        for img in out.glob('*.png'):
            assert img.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
        assert len(list(out.glob('*.png'))) == 4
        archive = tmp/'parent-scores.zip'
        with ZipFile(archive,'w') as bundle:
            bundle.write(scr/'0.npz','synthetic.npz')
        np.testing.assert_array_equal(next(read_arms(archive)).values,read_arm(scr/'0.npz').values)
        bad = decide([],[],{})
        assert bad['verdict']=='INCONCLUSIVE_FAILED_CONTROL' and not bad['physical_exclusion']
    print('Synthetic whole-parent bootstrap, support, paired/independent contrasts, plant decomposition, decision, CSVs and figures: OK.')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--scores',type=Path,help='Directory of completed parent NPZs, or released parent-scores.zip')
    group.add_argument('--self-test',action='store_true',help='Test with synthetic inputs only')
    p.add_argument('--out',type=Path,default=OUT)
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        print(run(args.scores,args.out)['verdict'])
