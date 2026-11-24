"""HERO-v2: preissue one pooled signed boundary forecast, then score whole parents.

Only `fixtures` is unprivileged. All native entry points require separate committed
releases. No native archive reader or alternative observable in this module.
"""
import argparse
import re
import json
import subprocess
import resource
import time
from pathlib import Path

import numpy as np
from scipy.stats import nct, norm, t

from hero_bank import ROOT, committed, readj, require, sha
from hero_forecast import conventional_features, exact_inventory_null
from hero_train import shell_logits
from full_hero_stream import FIELDS, INDICES, PUB, STREAMS, WORK, cutoff, savej

NONZERO = tuple(b for b in INDICES if b)
TARGETS = ('f', 'conventional', 'residual')
METHODS = ('inventory', 'm0', 'm1', 'm0_noI', 'm1_noI')
ALPHAS = (1., .01, .0001)
MATERIAL = .01
Q = 8
CHUNK = 64  # parents, not anchors
D = 2 + 64 + 4
CELL = 344
NAMES = dict(rule='pooled-rule.json', forecast_design='pooled-forecast-design.json',
             design='design.json', lineage='lineage.json', nomination='nomination.json')
FREEZE = 'pooled-forecast-freeze.json'
MODEL = 'pooled-forecast-model.npz'
DISCOVERY = dict(NAMES, rule='discovery-v21-pooled-rule.json', nomination='discovery-v21-nomination.json',
                 amendment='discovery-v21-design.json', panel='discovery-v21-panel.json')


def key(method, target):
    return method if target == 'f' else f'{method}_{target}'


def numerical_rows(a, width):
    a = np.asarray(a, dtype=np.float64).reshape(-1, width)
    require(np.isfinite(a).all(), 'nonfinite input')
    return a


def design(I, E, critic):
    logits = shell_logits(critic, E)
    x = np.column_stack((numerical_rows(I, 2)/343., numerical_rows(conventional_features(E), 64),
                         numerical_rows(logits - logits.mean(-1, keepdims=True), 4)))
    return np.column_stack((np.ones(len(x)), x, x*x))


def targets(V, spec):
    from full_hero_rule import apply, residual
    f = apply(V, spec)
    r = residual(V, spec)
    return numerical_rows(np.stack((f, f-r, r), axis=-1), len(TARGETS))


def inventory_codes(I):
    return I[..., 0].astype(np.int32)*CELL + I[..., 1]


def add_support(counts, I):
    codes = inventory_codes(I)
    # Parent-local duplicate cells count once even when the same I occurs at 8 anchors.
    unique = np.unique(np.arange(len(I))[:, None]*CELL**2 + codes) % (CELL**2)
    np.add.at(counts, unique, 1)


def method_cols(method):
    cols = np.arange(D)
    if method == 'inventory':
        cols = cols[:2]
    else:
        if method.endswith('_noI'):
            cols = cols[2:]
        if method.startswith('m0'):
            cols = cols[cols < D-4]
    return np.r_[0, 1+cols, 1+D+cols]


def ridge_grid(fit, even):
    """Both moments have mass 1: [1,x,x²,f,conventional,residual]."""
    p = fit.shape[0]-len(TARGETS)
    ey = fit[0, p:]
    cov = fit - np.outer(fit[0], fit[0])
    result, report = {}, {}
    for m in METHODS:
        ix = method_cols(m)
        j = ix[1:]
        sd = np.sqrt(np.maximum(cov[j, j], 0.))
        sd[sd < 1e-7] = 1.
        a = cov[np.ix_(j,j)] / (sd[:,None]*sd[None,:])
        rhs = cov[j,p:]/sd[:,None]
        slopes, costs = [], []
        for alpha in ALPHAS:
            w = np.linalg.solve(a + alpha*np.eye(len(j)), rhs)/sd[:,None]
            coef = np.vstack((ey-fit[0,j]@w, w))
            # E[(y-Xb)^2], including absolute costs, not correlation.
            loss = np.diag(even[p:,p:]) - 2*(coef*even[ix,p:]).sum(0) + (coef*(even[np.ix_(ix,ix)]@coef)).sum(0)
            slopes.append(coef); costs.append(loss)
        costs = np.asarray(costs)
        selected = np.argmin(costs, axis=0)
        coef = np.zeros((p,len(TARGETS)))
        coef[ix] = np.column_stack([slopes[k][:,tgt] for tgt,k in enumerate(selected)])
        result[m] = coef
        report[m] = dict(even_dev_macro_mse=np.maximum(costs, 0.).tolist(), selected_alpha=[ALPHAS[k] for k in selected])
    return result, report


def predict(x, weights):
    return {key(m, target): a[:, j].reshape(-1, Q) for m in METHODS
            for a in (x@weights[m],) for j,target in enumerate(TARGETS)}


def interval(mu, se, df, family=1):
    if se is None:
        return dict(point=mu, se=None, df=None, lower95=None, upper95=None)
    # Zero within-parent variance is not a claim of infinite sample size. Represent
    # the Gaussian limiting critical value with df=None instead of JSON Infinity.
    crit = t.ppf(1-.025/family, df) if df is not None else norm.ppf(1-.025/family)
    return dict(point=mu, se=se, df=df, lower95=float(mu-crit*se), upper95=float(mu+crit*se))


def parents(x, family=1):
    x = np.asarray(x, dtype=float)
    require(x.ndim == 1 and np.isfinite(x).all(), 'invalid parent values')
    n = len(x)
    sd = float(x.std(ddof=1)) if n > 1 else None
    return dict(n=n, small_n=n<8, parent_sd=sd, **interval(float(x.mean()) if n else None,
                sd/np.sqrt(n) if n>1 else None, n-1 if n>1 else None, family))


def welch(summaries, n_project=None):
    """Condition on this fixed list of temperatures. Never renormalize a missing one."""
    require(summaries, 'empty finite panel')
    means = [r['point'] for r in summaries]
    mu = None if None in means else float(np.mean(means))
    n = np.array(n_project if n_project is not None else [r['n'] for r in summaries])
    if mu is None or np.any(n<2) or any(r['parent_sd'] is None for r in summaries):
        return interval(mu,None,None)
    v = np.array([r['parent_sd']**2 for r in summaries])/n
    denom = (v*v/(n-1)).sum()
    df = float(v.sum()**2/denom) if denom else None
    return interval(mu,float(np.sqrt(v.sum())/len(v)),df)


def projected_power(stats, exam_n, family=1):
    w = welch(stats, exam_n)
    se, df = w['se'], w['df']
    power = (None if se is None else 1. if se == 0 else
             float(nct.sf(t.ppf(1-.025/family,df),df,MATERIAL/se)))
    return dict(projected_se=se, projected_df=df, alternative_true_g=.02, margin=MATERIAL,
                probability_lower95_above_margin=power, projected_parent_n=list(exam_n))


def losses(y, pred, variance, family=len(NONZERO)):
    require(variance>1e-8, 'degenerate FIT f')
    result = {}
    for j,target in enumerate(TARGETS):
        a = y[:,j].reshape(-1,Q)
        by_method = {m: ((pred[key(m,target)]-a)**2).mean(1) for m in METHODS}
        row = {f'mse_{m}':parents(z) for m,z in by_method.items()}
        for name,baseline,new in (('g','m0','m1'),('vs_inventory','inventory','m1'),('noI_g','m0_noI','m1_noI')):
            effect = (by_method[baseline]-by_method[new])/variance
            row[name] = parents(effect)
            if name=='g':
                row['g_bonferroni_nonzero_family'] = parents(effect, family)
        result[target] = row
    return result


def panel_summary(per_beta, nonzero=NONZERO):
    return {target: {metric:welch([per_beta[b][target][metric] for b in nonzero])
            for metric in (*('mse_'+m for m in METHODS), 'g','vs_inventory','noI_g')}
            for target in TARGETS}


def passes(bounds, level):
    return bounds['lower95'] is not None and bounds['lower95'] > level


def pin(name, rev):
    path = PUB/name
    committed(path,rev)
    return path


def match(path, digest):
    require(sha(path)==digest, f'input digest differs: {path.name}')
    return path


def checkpoint(nomination, discovery=False):
    schema='discovery-v21-nomination-1' if discovery else 'hero-full-v2-nomination-1'
    require(nomination['schema']==schema and nomination['status']=='qualified'
            and nomination['primary_seed']==6101 and nomination['denominator']==1, 'no qualified full-data reference critic')
    name = '{arch}-{law}-d1-s6101-u{updates}.npz'.format(**nomination)
    ck = nomination['checkpoints']['6101']
    require(ck['path']==name,'wrong portable primary seed')
    path = match(WORK/'checkpoints'/name, ck['sha256'])
    with np.load(path,allow_pickle=False) as z:
        weights = {k:z[k] for k in z.files if k.startswith('e')}
    return weights, sha(path)


def common(rev, discovery=False):
    require(re.fullmatch(r'[0-9a-f]{40}',rev) is not None and
            subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',rev,'HEAD']).returncode==0,
            'expected immutable release commit in branch ancestry')
    for filename in ('full_hero_forecast.py','full_hero_rule.py','full_hero_stream.py','hero_forecast.py',
                     'hero_train.py','hero_rule.py','hero_bank.py'):
        committed(Path(__file__).resolve().with_name(filename), rev)
    paths = {k:pin(name,rev) for k,name in (DISCOVERY if discovery else NAMES).items()}
    indices=INDICES
    roles=('fit','dev','eval')
    if discovery:
        panel=readj(paths['panel'])
        indices=tuple(panel['eligible_indices'])
        require(panel['schema']=='hero-full-v2.1-prospective-panel-1' and tuple(panel['training_indices'])==INDICES
                and len(indices)>=2 and 0 in indices
                and tuple(b for b in INDICES if panel['by_index'][str(b)]['eligible'])==indices
                and panel['lineage_sha256']==sha(paths['lineage']) and panel['design_sha256']==sha(paths['amendment']),
                'invalid or ineligible frozen discovery panel')
        roles=('fit','dev')  # power uses counts only, not evaluation parent IDs
    lin = readj(paths['lineage'])
    require(lin['duplicate_excess_rows']==lin['unresolved_identities']==0 and lin['per_beta']['6']['eval']==0,
            'whole-parent distinctness or barred-beta seal not established')
    rp = match(WORK/'rosters.npz',lin['rosters_sha256'])
    with np.load(rp,allow_pickle=False) as z:
        rosters = {(b,s):z[f'b{b:02}_{s}'] for b in indices for s in roles}
    for b in indices:
        rows = [rosters[b,s] for s in roles]
        require(all(r.ndim==1 and len(r)==lin['per_beta'][str(b)][s] and len(r)>0
                    for r,s in zip(rows,roles)) and len(np.unique(np.concatenate(rows)))==sum(map(len,rows)),
                'overlap, missing parents or bad lineage counts')
    return paths,rosters,lin


def stream(b, stage, role, rev, roster, fields, roster_digest, release_digest):
    cutoff(stage)
    receipt = pin((f'stream-b{b:02}.json' if stage=='fit-dev' else f'stream-{stage}-b{b:02}.json'), rev)
    r = readj(receipt)
    require(b in INDICES and r['index']==b and r['stage']==stage and r['source']=='archived-physical'
            and r['rosters_sha256']==roster_digest and r['release_sha256']==release_digest,
            'incorrect stream receipt or wrong release')
    rec = r['roles'][role]
    n = len(roster); q = 64 if stage=='fit-dev' else Q
    require(rec['parents']==n and rec['opportunities']==n*q,'incomplete extraction')
    out = {}
    for k in ('parent',*fields):
        name = k+'.npy'
        path = match(STREAMS/f'b{b:02}'/role/name, rec['files'][name]['sha256'])
        a = np.load(path,allow_pickle=False,mmap_mode='r')
        shape = (n,) if k=='parent' else (n,q,*FIELDS[k][1])
        dtype = np.dtype('<i4' if k=='parent' else FIELDS[k][0])
        require(a.shape==shape and a.dtype==dtype, 'stream layout mismatch: '+name)
        out[k] = a
    require(np.array_equal(out['parent'],roster), 'stream parent order changed')
    return out, receipt


def slices(n, parity=None):
    step = 1 if parity is None else 2
    start = parity or 0
    for k in range(start, n, CHUNK*step):
        yield slice(k,min(k+CHUNK*step,n),step)


def discovery_blocks(data, spec, critic, parity):
    for s in slices(len(data['parent']),parity):
        cutoff('fit-dev')
        i,e,v = (data[k][s,::8] for k in ('I','E','V'))
        yield i,design(i,e,critic),targets(v,spec)


def freeze(rev, discovery=False):
    start, cpu = time.monotonic(), time.process_time()
    cutoff('fit-dev')
    paths,rosters,lin = common(rev,discovery)
    release = pin('discovery-v21-forecast-release.json' if discovery else 'pooled-forecast-release.json',rev)
    r = readj(release)
    stage='discovery-v21-forecast-fit' if discovery else 'forecast-fit'
    require(r['stage']==stage and r['rosters_sha256']==lin['rosters_sha256']
            and (not discovery or r['exam_authorized'] is False), 'missing forecast authorization')
    for k,path in paths.items(): match(path,r[k+'_sha256'])
    prep=pin('stream-release-1.json',rev)
    pr=readj(prep)
    require(pr['stage']=='fit-dev' and all(pr[k]==r[k] for k in ('design_sha256','lineage_sha256','rosters_sha256')),
            'wrong fit/dev stream release')
    prep_digest=sha(prep)
    dest = PUB/('discovery-v21-forecast.json' if discovery else FREEZE)
    artifact = PUB/('discovery-v21-forecast-coefficients.npz' if discovery else MODEL)
    require(not dest.exists() and not artifact.exists(), 'forecast freeze already spent')
    spec = readj(paths['rule']); nom = readj(paths['nomination'])
    indices=tuple(b for b in INDICES if (b,'fit') in rosters)
    nonzero=tuple(b for b in indices if b!=0)
    if discovery:
        panel_hash=sha(paths['panel'])
        require(nom['schema']=='discovery-v21-nomination-1' and nom['scope']=='development-only'
                and nom['panel_sha256']==panel_hash and tuple(nom['eligible_indices'])==indices,
                'wrong discovery nomination or panel')
        require(spec['schema']=='discovery-v21-rule-1' and spec['source']['panel_sha256']==panel_hash
                and tuple(spec['source']['panel_indices'])==indices
                and spec['source']['nomination_sha256']==sha(paths['nomination']), 'wrong discovery rule provenance')
    meta = dict(stage='forecast-freeze', release_sha256=sha(release), rule_sha256=sha(paths['rule']),
                nomination_sha256=sha(paths['nomination']), lineage_sha256=sha(paths['lineage']),
                forecast_design_sha256=sha(paths['forecast_design']), rosters_sha256=lin['rosters_sha256'],
                index6='barred; not identifiable', indices=indices, margin=MATERIAL, incremental_usd=0)
    if discovery:
        meta.update(stage='discovery-v21-forecast-development', scope='nonconfirmatory development only',
                    panel_sha256=panel_hash, amendment_sha256=sha(paths['amendment']), exam_authorized=False,
                    excluded_training_indices=[b for b in INDICES if b not in indices],
                    excluded_scope='eligible for original training only; no discovery or physical inference on excluded indices')
    if not (spec.get('feature') and spec.get('fit_sd',0)>0 and spec.get('sign') in (-1,1)
            and spec.get('conventional_model') and nom.get('status')=='qualified'):
        savej(dest,dict(**meta, passed=False, primary='no valid qualified prediction', development='no defined descriptor/projection or no nominated critic',
                        first_failed_gate=spec['status'], wall_seconds=time.monotonic()-start))
        return
    critic,critic_hash = checkpoint(nom,discovery)
    counts = np.zeros(CELL**2,dtype=np.int32)
    gram = [None,None]; dev_data = {}; coverage = {}; fit_seconds = {}; digests = {}
    for b in indices:
        tick = time.monotonic(); pilot = []
        for split,parity in (('fit',None),('dev',0)):
            data,receipt = stream(b,'fit-dev',split,rev,rosters[b,split],('I','E','V'),lin['rosters_sha256'],prep_digest)
            digests[b] = sha(receipt)
            if split=='dev': dev_data[b]=data
            accum = None; rows = 0
            for i,x,y in discovery_blocks(data,spec,critic,parity):
                xy = np.column_stack((x,y)); g = xy.T@xy
                if accum is None: accum=g
                else: accum+=g
                rows+=len(xy)
                if split=='fit': add_support(counts,i)
            expected_n = len(rosters[b,split][::1 if parity is None else 2])
            require(rows==Q*expected_n and rows>0, 'missing FIT/even DEV parents')
            j = int(split=='dev'); accum/=rows*len(indices)
            if gram[j] is None: gram[j]=accum
            else: gram[j]+=accum
            pilot.append(expected_n)
        coverage[b] = dict(beta=lin['per_beta'][str(b)]['beta'],fit_parents=pilot[0],even_dev_parents=pilot[1],
                           odd_dev_parents=len(rosters[b,'dev'][1::2]),exam_parents=lin['per_beta'][str(b)]['eval'],anchors_per_parent=Q)
        fit_seconds[b] = time.monotonic()-tick
    weights,alpha = ridge_grid(*gram)
    var = (np.diag(gram[0])[-3:]-gram[0][0,-3:]**2).clip(min=0)
    variance = float(var[0]); residual_fraction = float(var[2]/variance) if variance>1e-8 else None
    require(variance>1e-8 and np.isclose(variance,spec['fit_variance'],rtol=2e-6,atol=1e-8)
            and np.isclose(residual_fraction,spec['residual_fraction'],rtol=2e-6,atol=1e-8), 'rule and forecast FIT moments disagree')
    per = {}; support = {}; odd_seconds = {}; canonical=None
    for b,data in dev_data.items():
        tick = time.monotonic(); truth=[]; pp={}; supported=0; null_inventory=[]
        for i,x,y in discovery_blocks(data,spec,critic,1):
            truth.append(y); pred=predict(x,weights)
            for k,v in pred.items(): pp.setdefault(k,[]).append(v)
            supported+=int((counts[inventory_codes(i)]>=8).sum())
            if discovery and b==0: null_inventory.append(i)
        y = np.concatenate(truth); pred={k:np.concatenate(v) for k,v in pp.items()}
        require(len(y)==coverage[b]['odd_dev_parents']*Q, 'odd DEV incomplete')
        per[b]=losses(y,pred,variance,len(nonzero)); support[b]=supported/len(y)
        if null_inventory:
            canonical=exact_inventory_null(np.concatenate(null_inventory),y[:,0].reshape(-1,Q),pred['m1']-pred['m0'],variance)
        odd_seconds[b]=time.monotonic()-tick
    macro = panel_summary(per,nonzero); f=macro['f']
    power = projected_power([per[b]['f']['g'] for b in nonzero],[coverage[b]['exam_parents'] for b in nonzero])
    if discovery:
        meta.update(odd_dev_beta0_exact_inventory=canonical, projected_power_by_index={
                b:dict(pointwise=projected_power([per[b]['f']['g']],[coverage[b]['exam_parents']]),
                       simultaneous_nonzero=projected_power([per[b]['f']['g']],[coverage[b]['exam_parents']],len(nonzero)) if b else None)
                for b in indices})
    gates = dict(symbolic=bool(spec['eligible_for_forecast_gate'] and spec['symbolic_passed'] and not spec['rediscovery']),
                 distinct_residual=residual_fraction>=.10, dev_raw=passes(f['g'],0) and f['g']['point']>=MATERIAL,
                 dev_vs_inventory=passes(f['vs_inventory'],0) and f['vs_inventory']['point']>=MATERIAL,
                 dev_residual=macro['residual']['g']['point']>0, dev_support=np.mean(list(support.values()))>=.90,
                 precision=power['probability_lower95_above_margin'] is not None and power['probability_lower95_above_margin']>=.80)
    gates={k:bool(v) for k,v in gates.items()}; passed=all(gates.values())
    cutoff('fit-dev')
    with artifact.open('xb') as out:
        np.savez_compressed(out,**weights,fit_inventory_parent_counts=counts)
    primary=('development diagnostic only; no assessment or blind issuance is authorized' if discovery else
             'single frozen prediction eligible for blind issuance' if passed else 'no valid qualified physical prediction; no exam')
    savej(dest,dict(**meta, passed=passed and not discovery, development_thresholds_met=passed, gates=gates,
            first_failed_gate=next((k for k,v in gates.items() if not v),None), primary=primary,
            development='odd DEV; nonconfirmatory single selected descriptor, shared with symbolic selection',
            feature=spec['feature'], feature_sign=spec['sign'], fit_variances=dict(zip(TARGETS,var.tolist())),
            symbolic_status=spec['status'],
            residual_fraction=residual_fraction, critic_seed=6101, critic_sha256=critic_hash, model_file=artifact.name,
            model_sha256=sha(artifact), stream_sha256=digests, coverage_by_index=coverage, odd_dev_support=support,
            odd_dev_pointwise_and_simultaneous=per, odd_dev_macro=macro, projected_power=power,
            selection=alpha, alphas=ALPHAS, targets=TARGETS, methods=METHODS, fit_and_even_seconds=fit_seconds,
            odd_seconds=odd_seconds,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
            max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def frozen(rev):
    cutoff('hidden')
    paths,rosters,lin=common(rev)
    freeze_path=pin(FREEZE,rev); f=readj(freeze_path)
    require(f['stage']=='forecast-freeze' and f['passed'] is True and all(f['gates'].values()) and f['margin']==MATERIAL,
            'no exam: forecast not qualified')
    for k in ('lineage','nomination','rule','forecast_design'): match(paths[k],f[k+'_sha256'])
    require(f['rosters_sha256']==lin['rosters_sha256'] and f['model_file']==MODEL,'frozen lineage/model changed')
    return paths,rosters,lin,f,sha(freeze_path)


def stage_release(stage,b,rev,paths,freeze_sha):
    p=pin(f'{stage}-release-b{b:02}.json',rev); r=readj(p)
    require(r['stage']==stage and r['index']==b and r['forecast_file']==FREEZE and r['rule_file']==NAMES['rule']
            and r['prerequisites'][FREEZE]==freeze_sha and r['prerequisites'][NAMES['rule']]==sha(paths['rule'])
            and r['lineage_sha256']==sha(paths['lineage']) and r['design_sha256']==sha(paths['design']),
            'wrong blind/hidden release pins')
    return sha(p)


def issue(b,rev):
    start=time.monotonic(); paths,rosters,lin,f,freeze_sha=frozen(rev)
    lock=stage_release('blind',b,rev,paths,freeze_sha)
    dest=PUB/f'predictions-b{b:02}.npz'; rec=dest.with_suffix('.json')
    require(not rec.exists() and not dest.exists(), 'already issued')
    data,receipt=stream(b,'blind','blind',rev,rosters[b,'eval'],('anchor','E','I'),lin['rosters_sha256'],lock)
    critic,ck=checkpoint(readj(paths['nomination'])); require(ck==f['critic_sha256'],'critic changed')
    n=len(data['parent']); out={k:data[k] for k in ('parent','anchor')}; supported=0
    with np.load(match(pin(MODEL,rev),f['model_sha256']),allow_pickle=False) as z:
        weights={m:z[m] for m in METHODS}; counts=z['fit_inventory_parent_counts']
    for s in slices(n):
        cutoff('blind'); i=data['I'][s]; x=design(i,data['E'][s],critic)
        for k,a in predict(x,weights).items():
            if k not in out: out[k]=np.empty((n,Q))
            require(np.isfinite(a).all(),'nonfinite signed prediction'); out[k][s]=a
        supported+=int((counts[inventory_codes(i)]>=8).sum())
    cutoff('blind')
    with dest.open('xb') as fp: np.savez_compressed(fp,**out)
    savej(rec,dict(stage='predictions',index=b,role='eval',beta=lin['per_beta'][str(b)]['beta'],
            parents=n,opportunities=n*Q,prediction_sha256=sha(dest),blind_sha256=sha(receipt),
            rule_sha256=f['rule_sha256'],forecast_sha256=freeze_sha, support_fraction=supported/(n*Q),
            all_rows_issued=True, wall_seconds=time.monotonic()-start))


def score(rev):
    start=time.monotonic(); paths,rosters,lin,f,freeze_sha=frozen(rev); model=match(pin(MODEL,rev),f['model_sha256'])
    dest=PUB/'pooled-forecast-score.json'; require(not dest.exists(),'combined assessment already spent')
    # Global barrier: all files, blind coordinates, support and coefficients pinned BEFORE a single hidden V mmap.
    certified={}; supports={}; variance=f['fit_variances']['f']; spec=readj(paths['rule'])
    with np.load(model,allow_pickle=False) as z: counts=z['fit_inventory_parent_counts']
    for b in INDICES:
        path=pin(f'predictions-b{b:02}.npz',rev); p=readj(pin(f'predictions-b{b:02}.json',rev))
        hidden_lock=stage_release('hidden',b,rev,paths,freeze_sha)
        hr=readj(pin(f'stream-hidden-b{b:02}.json',rev)); require(hr['release_sha256']==hidden_lock,'unbound hidden stream')
        blind_lock=stage_release('blind',b,rev,paths,freeze_sha)
        blind,br=stream(b,'blind','blind',rev,rosters[b,'eval'],('anchor','I'),lin['rosters_sha256'],blind_lock)
        require(readj(br)['release_sha256']==blind_lock and p['stage']=='predictions' and p['index']==b and p['role']=='eval'
                and p['blind_sha256']==sha(br) and p['forecast_sha256']==freeze_sha and p['rule_sha256']==f['rule_sha256'], 'unbound issued panel')
        support=float((counts[inventory_codes(blind['I'])]>=8).mean()); supports[b]=support
        require(support==p['support_fraction'],'pre-outcome support changed')
        with np.load(match(path,p['prediction_sha256']),allow_pickle=False) as z:
            require(np.array_equal(z['parent'],blind['parent']) and np.array_equal(z['anchor'],blind['anchor']), 'different issued roster or opportunities')
            for m in METHODS:
                for target in TARGETS:
                    a=z[key(m,target)]
                    require(a.shape==(len(blind['parent']),Q) and np.isfinite(a).all(),'incomplete issued panel')
        certified[b]=(path,blind,p['prediction_sha256'],hidden_lock)
    require(np.mean(list(supports.values()))>=.90,'blind coverage gate failed')
    per={}; canonical=None; per_seconds={}; assessed={}
    for b,(path,blind,digest,hidden_lock) in certified.items():
        tick=time.monotonic(); data,hr=stream(b,'hidden','hidden',rev,rosters[b,'eval'],('anchor','V'),lin['rosters_sha256'],hidden_lock)
        require(np.array_equal(data['anchor'],blind['anchor']), 'opportunities changed after issuance')
        truth=[]
        for s in slices(len(data['parent'])):
            cutoff('hidden'); truth.append(targets(data['V'][s],spec))
        y=np.concatenate(truth)
        with np.load(path,allow_pickle=False) as z: pred={key(m,k):z[key(m,k)] for m in METHODS for k in TARGETS}
        per[b]=losses(y,pred,variance)
        if b==0:
            canonical=exact_inventory_null(blind['I'],y[:,0].reshape(-1,Q),pred['m1']-pred['m0'],variance)
        assessed[b]=dict(beta=lin['per_beta'][str(b)]['beta'],prediction_sha256=digest, hidden_sha256=sha(hr),
                          parents=len(data['parent']),opportunities=len(y),deployed_fraction=1.,inventory_support=supports[b])
        per_seconds[b]=time.monotonic()-tick
    macro=panel_summary(per)
    gates=dict(exact_parent_lineage=True, blind_support=True, macro_material=passes(macro['f']['g'],MATERIAL),
               distinct_residual=f['residual_fraction']>=.10 and passes(macro['residual']['g'],0), canonical_equivalence=bool(canonical['passed']))
    passed=all(gates.values()); individual={b:passed and passes(per[b]['f']['g_bonferroni_nonzero_family'],MATERIAL) for b in NONZERO}
    cutoff('hidden')
    savej(dest,dict(stage='combined-final-score',passed=passed,gates=gates,individual_positive=individual,
            claim='material distinct finite-panel gain' if passed else 'material distinct gain not demonstrated',
            forecast_sha256=freeze_sha,rule_sha256=f['rule_sha256'],fit_variance=variance,material_margin=MATERIAL,
            fit_residual_fraction=f['residual_fraction'],assessed=assessed,per_beta=per,macro=macro,beta0_exact_inventory=canonical,
            index6='all parents barred, not identifiable, not zero',null_index=0, nonzero_indices=NONZERO,
            per_index_seconds=per_seconds,wall_seconds=time.monotonic()-start, incremental_usd=0))


def fixtures():
    """Abstract numeric opportunities only; independently enumerable tests of the estimand and counterexamples."""
    start=time.monotonic(); rng=np.random.default_rng(26112513)
    a=np.array([1.,3.,5.,7.]); b=np.array([4.,6.]); w=welch([parents(a),parents(b)])
    v1=20/3/4; v2=2/2
    np.testing.assert_allclose((w['point'],w['se'],w['df']), (4.5,np.sqrt(v1+v2)/2,(v1+v2)**2/(v1**2/3+v2**2)))
    require(welch([parents(a),parents([1.])])['lower95'] is None,'singleton uncertainty invented')
    counts=np.zeros(CELL**2,dtype=int); i=np.zeros((3,Q,2),dtype=int); i[0,4:,1]=2; i[1,:,1]=2
    add_support(counts,i); require(counts[0]==counts[2]==2 and counts.sum()==4,'repeated anchors treated as parents')
    gs=[]; ys=[]; xs=[]
    for n,shift in ((32,-2.),(128,1.)):
        x=rng.normal(shift,.5,(n,D)); x=np.column_stack((np.ones(n),x,x*x)); y=np.column_stack((x[:,2]+.8*x[:,D],x[:,2],.8*x[:,D])); xy=np.column_stack((x,y))
        gs.append(sum(z.T@z for z in np.array_split(xy,4))/n); xs.append(x); ys.append(y)
    g=np.mean(gs,axis=0); weights,select=ridge_grid(g,g)
    mu=np.mean([y[:,0].mean() for y in ys]); var=np.mean([np.mean((y[:,0]-mu)**2) for y in ys]); np.testing.assert_allclose(var,g[-3,-3]-g[0,-3]**2)
    require(not np.isclose(mu,np.concatenate(ys)[:,0].mean()), 'pooled-row weighting not distinguished')
    for m in METHODS:
        p=np.concatenate([x@weights[m] for x in xs]); splits=np.split(p,[len(xs[0])]); rm=np.mean([np.mean((k-y)**2,axis=0) for k,y in zip(splits,ys)],axis=0)
        np.testing.assert_allclose(rm,np.array(select[m]['even_dev_macro_mse']).min(0),atol=1e-9)
    require(all(np.count_nonzero(weights['m0'][k])==0 for k in set(method_cols('m1'))-set(method_cols('m0'))),'critic leaks into m0')
    p=predict(xs[0],weights); loss=losses(ys[0],p,var)['f']['g']; require(loss['n']==4,'site pseudoreplication')
    exact=((p['m0']-ys[0][:,0].reshape(-1,Q))**2-(p['m1']-ys[0][:,0].reshape(-1,Q))**2).mean(1)/var
    np.testing.assert_allclose(loss['se'],exact.std(ddof=1)/2)
    require(parents(exact,len(NONZERO))['lower95']<=loss['lower95'],'anti-conservative simultaneous bound')
    # Independent empirical coverage of the Welch procedure with heterogeneous parent SD and n.
    n=np.array([9,32,50]); sd=np.array([.1,.2,.8]); reps=4096
    pts=[]; vs=[]
    for k,s in zip(n,sd):
        z=rng.normal(0,s,(reps,k)); pts.append(z.mean(1)); vs.append(z.var(1,ddof=1)/k)
    vs=np.array(vs); se=np.sqrt(vs.sum(0))/len(n); df=vs.sum(0)**2/(vs**2/(n[:,None]-1)).sum(0)
    covered=float((abs(np.mean(pts,axis=0)) <= t.ppf(.975,df)*se).mean()); require(.93<covered<.97,'Welch coverage control')
    f0=np.broadcast_to(np.arange(-3,5),(16,Q)).astype(float); same=np.zeros((16,Q,2),dtype=int)
    null=exact_inventory_null(same,f0,np.zeros_like(f0),1.); leak=exact_inventory_null(same,f0,f0,1.)
    require(null['passed'] and not leak['passed'],'canonical equivalence or leakage control failed')
    const=projected_power([parents([1.,1.]),parents([2.,2.])],[2,4]); require(const['probability_lower95_above_margin']==1,'zero-noise power')
    return dict(schema='hero-full-v2-pooled-forecast-fixtures-1',physical_data=False, seed=26112513, passed=True,
                mean_se_df=w, unequal_beta_fit_mean=float(mu),unequal_beta_fit_variance=float(var),support_cells=[int(counts[0]),int(counts[2])],
                opportunities_per_parent=Q,whole_parent_gain=loss, simulated_Welch_95_coverage=covered,replicates=reps,
                null_passed=bool(null['passed']), planted_leak_passed=bool(leak['passed']),ridge_grid_selected_exact_cost=True,
                elapsed_seconds=time.monotonic()-start)


def main():
    cli=argparse.ArgumentParser(description=__doc__); sub=cli.add_subparsers(dest='cmd',required=True)
    test=sub.add_parser('fixtures'); test.add_argument('--publish',action='store_true')
    for mode in ('freeze','discovery-v21','issue','score'):
        p=sub.add_parser(mode); p.add_argument('--release-commit',required=True)
        if mode=='issue': p.add_argument('--index',type=int,choices=INDICES,required=True)
    a=cli.parse_args()
    if a.cmd=='fixtures':
        result=fixtures(); print(json.dumps(result,indent=2,allow_nan=False),flush=True)
        if a.publish: savej(PUB/'pooled-forecast-fixtures.json',result)
    elif a.cmd=='freeze': freeze(a.release_commit)
    elif a.cmd=='discovery-v21': freeze(a.release_commit,discovery=True)
    elif a.cmd=='issue': issue(a.index,a.release_commit)
    else: score(a.release_commit)


if __name__=='__main__':
    main()
