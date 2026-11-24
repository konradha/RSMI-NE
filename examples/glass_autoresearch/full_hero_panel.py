"""HERO v2.1 nuisance counts before optimization. Never opens any shell or exam data."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nct, t
from hero_train import C, strata

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT/'results/l20-autoresearch/hero-full-v2'
WORK = Path.home()/'rsmi-work/l20-autoresearch-1795520725/hero-full-v2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixed(name, commit):
    p = PUB/name
    b = p.read_bytes()
    g = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{p.relative_to(ROOT)}'])
    if b != g:
        raise ValueError(f'{name} differs from registered version')
    return json.loads(b), sha(b)


def parent_support(s):
    n, q = s.shape
    stats, bins = {}, {}
    for x in np.unique(s):
        mask = (s == x)
        parents = np.flatnonzero(mask.any(axis=1))
        bins[int(x)] = parents
        stats[int(x)] = {'parents': int(len(parents)), 'opportunities': int(mask.sum())}
    legal = [x for x in bins if len(bins[x]) >= C]
    used = np.zeros(n, bool)
    # Deterministic lower bound: every emitted group's parents are disjoint from all prior groups.
    for x in sorted(legal, key=lambda x: (-len(bins[x]), x)):
        free = bins[x][~used[bins[x]]]
        used[free[:len(free)//C*C]] = True
    cov = sum(stats[x]['opportunities'] for x in legal)/(n*q)
    enough = n >= 4*C and cov >= 0.9 and used.sum() >= 4*C
    return dict(eligible=bool(enough), n=n, opportunities=n*q, uniform_disjoint_groups=n//C,
                matched_anchor_fraction=cov,
                matched_supported_parents=int(len(np.unique(np.concatenate([bins[x] for x in legal])))) if legal else 0,
                disjoint_matched_group_lower_bound=int(used.sum()//C), bins=stats)


def power(ns, alpha=.05):
    if not ns or min(ns)<2:
        return None
    ns = np.asarray(ns, dtype=float)
    weights = 1/ns
    df = weights.sum()**2 / ((weights**2)/(ns-1)).sum()
    critical = t.isf(alpha/2, df)
    scale = np.sqrt(weights.sum())/len(ns)
    nc80 = brentq(lambda d: nct.sf(critical, df, d)-.8, 0, 100)
    return dict(n=[int(x) for x in ns], df=float(df), alpha=alpha, gap=.01,
                max_equal_parent_sd_at_80_percent=.01/(nc80*scale),
                illustrative_power_by_sd={str(s): float(nct.sf(critical,df,.01/(s*scale))) for s in [.05,.10,.20,.50]})


def main(commit):
    _, designsha = fixed('discovery-v21-design.json', commit)
    lineage, linesha = fixed('lineage.json', commit)
    result = {}
    for idx, v in lineage['per_beta'].items():
        b = int(idx)
        row = result[idx] = dict(index=b, beta=v['beta'], fit=v['fit'], dev_even=(v['dev']+1)//2,
                                 dev_odd=v['dev']//2, evaluation_n=v['eval'], eligible=False)
        if not v['fit']:
            row['reason'] = 'native temperature permanently embargoed'
            continue
        receipt,_ = fixed(f'stream-b{b:02}.json', commit)
        role = WORK/'streams'/f'b{b:02}'/'dev'
        allowed_hash = receipt['roles']['dev']['even_sha256']
        even = {}
        for key in ['parent','V']:
            even[key] = np.ascontiguousarray(np.load(role/(key+'.npy'),allow_pickle=False,mmap_mode='r')[::2])
            if sha(memoryview(even[key]).cast('B')) != allowed_hash[key]:
                raise ValueError(f'Even development digest mismatch for {b} {key}')
        if len(np.unique(even['parent']))!=row['dev_even'] or even['V'].shape!=(row['dev_even'],64,27):
            raise ValueError('Unexpected repeated parent, shape or parity')
        row['support'] = parent_support(strata(even['V']))
        row['eligible'] = row['support']['eligible']
        row['reason'] = 'all three registered conditions passed' if row['eligible'] else 'independent parent or matched-stratum support below registered threshold'
    eligible = [int(i) for i,r in result.items() if r['eligible']]
    nonzero = [i for i in eligible if i!=0]
    for r in result.values():
        r['assessment_power_pointwise_plan_only'] = power([r['evaluation_n']])
        r['assessment_power_simultaneous_plan_only'] = power([r['evaluation_n']],.05/len(nonzero)) if r['index'] in nonzero else None
    out = dict(schema='hero-full-v2.1-prospective-panel-1',design_sha256=designsha,lineage_sha256=linesha,commit=commit,
               eligible_indices=eligible,training_indices=[int(k) for k,v in result.items() if v['fit']],by_index=result,
               nonzero_supported_macro_power_plan_only=power([result[str(i)]['evaluation_n'] for i in nonzero]),
               exposures='S(V) and native row IDs from even development only; aggregate counts. No estimate of parent-effect variance from outcomes and no physical scores, gradients, exams or new simulations. Planning is conditional on assumed SD, not evidence of actual assay power.',
               native_optimizer_steps_at_registration=0,incremental_modal_usd=0)
    target=PUB/'discovery-v21-panel.json'
    with target.open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False); f.write('\n')
    for i,r in result.items():
        s=r.get('support',{})
        print(i, r['beta'], r['dev_even'], s.get('matched_anchor_fraction'),s.get('disjoint_matched_group_lower_bound'),r['eligible'])
    print('Supported panel',eligible,'; power is planning only')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--design-commit',required=True)
    main(p.parse_args().design_commit)
