"""Distill a frozen four-state cold teacher, using ONLY cold train/validation visible banks.

Run: OPENBLAS_NUM_THREADS=1 python motif.py --train DIR --validation DIR --out DIR
observable(v, rule) is the standalone, purely visible scalar. No shell labels are loaded.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np

STATIC = Path(__file__).resolve().parents[2] / 'results/l20-operators/static'
SPECIES = ('vacancy', 'A', 'B')
# Site ordering and output-category mapping from the published cold VALIDATION alignment.
ALIGN = {11: ((0, 1, 2), (1, 1, 1), (0, 1, 2, 3)),
         29: ((2, 0, 1), (1, 1, -1), (3, 1, 2, 0)),
         47: ((1, 0, 2), (-1, 1, 1), (0, 1, 2, 3))}
CHUNK = 8192


def coord(i):
    return tuple(int(x) - 1 for x in np.unravel_index(i, (3, 3, 3)))


def observable(v, rule):
    """Evaluate the exact finite species expression on N visible cubes. No fit needed."""
    v = np.asarray(v).reshape(-1, 27)
    e = rule['expression']; s = e['sites']; op = e['op']; a = e['species']
    if op == 'indicator':
        r = v[:, s[0]] == a[0]
    elif op == 'difference':
        r = (v[:, s[0]] == a[0]).astype('i1') - (v[:, s[1]] == a[0])
    elif op == 'pattern':
        r = v[:, s[0]] == a[0]
        for i, t in zip(s[1:], a[1:]):
            r &= v[:, i] == t
    elif op == 'count' or op == 'gated-count':
        gate = op == 'gated-count'
        r = sum((v[:, i] == a[-1]).astype('i1') for i in s[int(gate):])
        if gate:
            r *= v[:, s[0]] == a[0]
    elif op == 'alternating':
        # s follows the perimeter. Either phase, two different named species.
        r = ((v[:, s[0]] == v[:, s[2]]) & (v[:, s[1]] == v[:, s[3]]) &
             (((v[:, s[0]] == a[0]) & (v[:, s[1]] == a[1])) |
              ((v[:, s[0]] == a[1]) & (v[:, s[1]] == a[0]))))
    elif op == 'tripod':
        n = sum((v[:, i] == a[1]).astype('i1') for i in s[1:])
        m = sum((v[:, i] == a[2]).astype('i1') for i in s[1:])
        r = (v[:, s[0]] == a[0]) & (n == 2) & (m == 1)
    else:
        raise ValueError(f'unknown motif operation: {op}')
    return r.astype(np.float32)


def dictionary():
    rules = []
    def add(op, sites, species, name, formula, low=0, high=1):
        rules.append(dict(name=name, formula=formula, levels=list(range(low, high+1)),
                          expression=dict(op=op, sites=list(sites), species=list(species)),
                          length=len(sites)))
    # One-site controls include the entire cube, not just the teacher's enriched corner.
    for i in range(27):
        for t in (0, 1, 2):
            name = f'{SPECIES[t]} at {coord(i)}'
            add('indicator', [i], [t], name, f'1 if {name}, otherwise 0.')
    # Directed first octant at each end of the historically identified body diagonal.
    for c, d in ((0, 1), (26, -1)):
        ns = [c+d*i for i in (9, 3, 1)]; cs = str(coord(c))
        for t in (0, 1):
            name = SPECIES[t]; locs = ', '.join(str(coord(i)) for i in ns)
            add('count', ns, [t], f'inward {name} count near {cs}',
                f'Number of {name} sites among {locs}.', high=3)
            add('gated-count', [c, *ns], [1, t], f'A-corner inward {name} count at {cs}',
                f'If {cs} is A count {name} among {locs}; otherwise 0.', high=3)
        for n in ns:
            for t in (0, 1, 2):
                desc = f'A at {cs} and {SPECIES[t]} at {coord(n)}'
                add('pattern', [c, n], [1, t], f'edge: {desc}', f'1 if {desc}, otherwise 0.')
            add('difference', [c,n], [1], f'A edge polarity {cs} to {coord(n)}',
                f'A indicator at {cs} minus A indicator at {coord(n)}.', -1, 1)
        for n, k in combinations(ns, 2):
            diag = n+k-c
            for t in (0, 2):
                ring = [c,n,diag,k]; locs = ', '.join(str(coord(i)) for i in ring)
                add('alternating', ring, [1,t], f'A/{SPECIES[t]} alternating plaquette {cs}:{coord(diag)}',
                    f'1 if walking {locs} gives alternating A and {SPECIES[t]} in either phase; otherwise 0.')
                desc = f'A at {cs} with {SPECIES[t]} at both {coord(n)} and {coord(k)}'
                add('pattern', [c,n,k], [1,t,t], f'right angle: {desc}', f'1 if {desc}, otherwise 0.')
        for t, u in ((1,0), (0,2), (2,0)):
            add('tripod', [c,*ns], [1,t,u], f'A tripod {cs}, two {SPECIES[t]}, one {SPECIES[u]}',
                f'1 if {cs} is A and its three inward nearest sites contain exactly two {SPECIES[t]} and one {SPECIES[u]}; otherwise 0.')
    for i in (2,6,8):
        add('difference', [i,26-i], [1], f'opposed A polarity {coord(i)} to {coord(26-i)}',
            f'A indicator at {coord(i)} minus A indicator at {coord(26-i)}.', -1, 1)
        desc = f'A at both {coord(i)} and {coord(26-i)}'
        add('pattern', [i,26-i], [1,1], f'opposed A pair {coord(i)}:{coord(26-i)}', f'1 if {desc}, otherwise 0.')
    return rules


def open_bank(path, fold):
    meta = json.loads((path/'meta.json').read_text())
    if meta['index'] != 9 or meta['fold'] != fold:
        raise ValueError(f'only beta index 9 fold {fold} may be opened')
    def load(name):
        return np.load(path/f'{name}.npy', mmap_mode='r', allow_pickle=False)
    ids = load('parent_id'); v = load('v'); m = load('m')
    if v.ndim != 3 or v.shape[-1] != 27 or v.shape != m.shape or len(ids) != len(v):
        raise ValueError('inconsistent parent-major visible bank')
    return v, m, ids


def teacher(v, seed):
    examples = str(Path(__file__).resolve().parents[1])
    if examples not in sys.path:
        sys.path.insert(0,examples)
    from l20_operators_static import prob
    axes, signs, perm = ALIGN[seed]
    order = np.arange(27).reshape(3,3,3).transpose(axes)[tuple(slice(None,None,s) for s in signs)].ravel()
    key = f'fits/w3-b1-learned-s{seed}'
    with zipfile.ZipFile(STATIC/'fit-receipts.zip') as archive:
        spec = json.loads(archive.read(key+'.json'))['spec']
        with np.load(io.BytesIO(archive.read(key+'.npz')), allow_pickle=False) as f:
            enc = [f[f'enc{i}'] for i in range(5)]
    q = np.empty((len(v),4), np.float32)
    for i in range(0,len(v),CHUNK):
        batch = v[i:i+CHUNK]; q[i:i+len(batch)] = prob(batch[:,order],3,spec,enc)[:,np.argsort(perm)]
    return q


def features(v, m, physical):
    if not physical:
        return np.column_stack((np.ones(len(v)), v == 1, v == 2))
    # Permutation-invariant species x exact coordination histograms, plus oriented corner coordination.
    code = 7*v + m
    return np.column_stack([np.ones(len(v)), *((code == k).sum(1) for k in range(21)),
                            m[:,0], m[:,26], (v[:,0] == 1)*m[:,0], (v[:,26] == 1)*m[:,26]])


def regression(train, banks, physical=False):
    # Projection of all four log posteriors onto an intercept + fixed feature basis on training rows.
    v,m,q = train; dim = 26 if physical else 55
    gram = np.eye(dim)*0.001; rhs = np.zeros((dim,4))
    for i in range(0,len(v),CHUNK):
        sl = slice(i,i+CHUNK); x = features(v[sl],m[sl],physical); gram += x.T@x
        rhs += x.T@np.log(np.maximum(q[sl],1e-20))
    coef = np.linalg.solve(gram,rhs); out = []
    for v,m in banks:
        p = np.empty((len(v),4),'f')
        for i in range(0,len(v),CHUNK):
            sl = slice(i,i+CHUNK); z = features(v[sl],m[sl],physical)@coef
            z = np.exp(z-z.max(1,keepdims=True)); p[sl] = z/z.sum(1,keepdims=True)
        out.append(p)
    return out


def sums(g, p, k):
    return np.column_stack([np.bincount(g, weights=p[:,h], minlength=k) for h in range(4)])


def table(g, q, k):
    n = sums(g,q,k) + 4*q.mean(0,dtype='d'); return (n/n.sum(1,keepdims=True)).astype('f')


def tilt(g, q, p, k):
    # Pseudocount toward no correction, including wholly empty strata.
    return ((sums(g,q,k)+1)/(sums(g,p,k)+1)).astype('f')


def corrected(g, p, r):
    a = p*r[g]; return a/a.sum(1,keepdims=True)


def measures(q,p,reps):
    ll = (q*np.log(np.maximum(p,1e-30))).sum(1)
    acc = q.argmax(1) == p.argmax(1)
    return np.column_stack((ll.reshape(-1,reps).mean(1,dtype='d'), acc.reshape(-1,reps).mean(1)))


def estimate(x):
    x = np.asarray(x,dtype='d'); mean = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x))
    return dict(mean=float(mean), se=float(se), lo95=float(mean-1.96*se), hi95=float(mean+1.96*se))


def contexts(v):
    return (3*v[:,0]+v[:,26]).astype('i2')


def rule_index(rule,v):
    return observable(v,rule).astype('i2')-rule['levels'][0]


def run(a):
    tr = open_bank(a.train,'train'); va = open_bank(a.validation,'validation')
    if np.intersect1d(tr[2],va[2]).size or len(np.unique(va[2])) != len(va[2]) or len(np.unique(tr[2])) != len(tr[2]):
        raise ValueError('banks must have distinct, disjoint archived parents')
    # Already-random archive split; make a deterministic discovery/audit split by hashed parent identity.
    order = sorted(range(len(va[2])), key=lambda i: hashlib.sha256(f'motif-validation:{va[2][i]}'.encode()).digest())
    tune, audit = np.array_split(np.array(order),2)
    def flatten(bank, ids=None):
        v,m,p = bank
        if ids is not None:
            v,m,p = v[ids],m[ids],p[ids]
        return (v.reshape(-1,27),m.reshape(-1,27)),p,v.shape[1]
    train, ti, rt = flatten(tr); valid, vi, rv = flatten(va,tune); confirm, ci, rc = flatten(va,audit)
    banks = [train,valid,confirm]; reps = [rt,rv,rc]; names = ['train','validation_selection','validation_audit']; rows = []
    groups = [contexts(v) for v,m in banks]; rules = dictionary(); effects = {}; examples = None
    def record(seed, label, j, scores, ref, what):
        stats = (estimate(scores[:,0]),estimate(scores[:,1]),estimate(scores[:,0]-ref[:,0]))
        rows.append(dict(seed=seed,split=names[j],model=label,reference=what,parents=len(scores),
                         **{k+'_'+s:val for k,stat in zip(('soft_log_fidelity','hard_agreement','paired_gain_nats'),stats) for s,val in stat.items()}))
        return stats[-1]
    for seed in (11,29,47):
        qs = [teacher(v,seed) for v,m in banks]; q = qs[0]; pack = (*train,q)
        if seed == 11:
            audit_teacher = qs[2]
        ps = regression(pack,banks); phys = regression(pack,banks,True)
        bits = [2*(v[:,0]==1)+(v[:,26]==1) for v,m in banks]; bit_tab = table(bits[0],q,4)
        ac = [(v[:,0]==1).astype('i1') for v,m in banks]; ac_tab = table(ac[0],q,2)
        corner_tab = table(groups[0],q,9); corner_sc = []; base_sc = []; bps = []
        cor = tilt(groups[0],q,ps[0],9)
        # Give the null exactly as many residual-calibration updates as each candidate.
        init = [corrected(g,p,cor) for g,p in zip(groups,ps)]
        null_ratio = tilt(groups[0],q,init[0],9)
        for j in range(3):
            gj = groups[j]; soft = qs[j]; r = reps[j]; cs = measures(soft,corner_tab[gj],r); corner_sc.append(cs)
            pj = corrected(gj,init[j],null_ratio); bps.append(pj); bs = measures(soft,pj,r); base_sc.append(bs)
            models = [('opposite A bits',measures(soft,bit_tab[bits[j]],r)), ('full opposite species',cs),
                      ('single A at (-1,-1,-1)',measures(soft,ac_tab[ac[j]],r)),
                      ('coordination and species group regression',measures(soft,phys[j],r)),
                      ('additive site-species regression',measures(soft,ps[j],r)), ('additive + full opposite species',bs),
                      ('teacher self (upper bound)',measures(soft,soft,r))]
            for name,s in models:
                record(seed,name,j,s,cs,'full opposite species')
            if seed != 11 and j == 2:
                record(seed,'seed-11 teacher as fixed predictor',j,measures(soft,audit_teacher,r),cs,'full opposite species')
        def score_rule(rule, parts, also_raw=False):
            k = len(rule['levels']); gt = k*groups[0]+rule_index(rule,train[0]); rat = tilt(gt,q,init[0],9*k)
            if also_raw:
                ct = table(gt,q,9*k)
                single_tab = table(gt%k,q,k)
            val = []
            for j in parts:
                g = gt if j == 0 else k*groups[j]+rule_index(rule,banks[j][0]); soft = qs[j]
                s = measures(soft,corrected(g,init[j],rat),reps[j]); diff = s[:,0]-base_sc[j][:,0]; val.append(diff)
                record(seed,rule['name']+' | additive + full opposite species',j,s,base_sc[j],'additive + full opposite species')
                if also_raw:
                    record(seed,rule['name']+' | full opposite species',j,measures(soft,ct[g],reps[j]),corner_sc[j],'full opposite species')
                    record(seed,rule['name']+' alone',j,measures(soft,single_tab[g%k],reps[j]),corner_sc[j],'full opposite species')
            return val
        if seed == 11:
            ranking = []
            for idx,rule in enumerate(rules):
                k = len(rule['levels']); gt = groups[0]*k+rule_index(rule,train[0])
                mass = np.bincount(gt,minlength=9*k).reshape(9,k)
                # A rule fixed by the corner states is not a residual field. Require
                # appreciable common support for at least one training level contrast.
                def overlap(l,h):
                    w = np.minimum(mass[:,l],mass[:,h]); return w[w>=32].sum()
                if max(overlap(l,h) for l,h in combinations(range(k),2)) < 128:
                    continue
                gains = score_rule(rule,(0,1)); means = [g.mean() for g in gains]
                ranking.append((min(means),idx,gains[1]))
            best = max(ranking,key=lambda row: row[0]); eligible = []
            for merit,idx,delta in ranking:
                # One paired standard error of the best, with a held-in training stability cap.
                allowance = estimate(best[2]-delta)['se']
                if merit >= best[0]-allowance:
                    eligible.append((rules[idx]['length'],len(rules[idx]['levels']),-merit,idx))
            ix = min(eligible)[-1]; selected = rules[ix]; selection = dict(candidates=len(rules),supported=len(ranking),
                one_se_eligible=len(eligible),best_name=rules[best[1]]['name'],best_merit=float(best[0]),
                selected_merit=float(next(row[0] for row in ranking if row[1]==ix)))
            score_rule(selected,(2,),True)
        else:
            score_rule(selected,(0,1,2),True)
        levels = selected['levels']; k = len(levels)
        # Freeze a direction using TRAIN only. Standardize across exact opposite-corner strata
        # in which both levels have support. Do not choose an extreme validation demonstration.
        l0 = rule_index(selected,train[0]); joint = k*groups[0]+l0
        mass = np.bincount(joint,minlength=9*k).reshape(9,k)
        resid = (sums(joint,q-bps[0],9*k).reshape(9,k,4)/np.maximum(mass[:,:,None],1))
        if seed == 11:
            options = []
            for l,h in combinations(range(k),2):
                weight = np.minimum(mass[:,l],mass[:,h]).astype('d'); weight[weight<32] = 0
                if weight.sum() < 128: continue
                weight /= weight.sum(); d = weight@(resid[:,h]-resid[:,l]); cat = int(np.argmax(abs(d)))
                options.append((abs(d[cat]),l,h,cat,weight))
            if not options: raise ValueError('selected field has no shared-corner training support')
            _,lo,hi,cat,weights = max(options,key=lambda row:row[0]); contrast = dict(low=levels[lo],high=levels[hi],class_index=cat,
                opposite_stratum_weights=weights.tolist(),stratum='3*species(corner 0)+species(corner 26); vacancy=0,A=1,B=2')
        la = rule_index(selected,confirm[0]); ga = groups[2]; qa = qs[2]; ba = bps[2]
        effects[str(seed)] = compare_levels(ga,la,qa,ba,cat,lo,hi,weights,rc)
        if seed == 11:
            examples = paired_examples(confirm[0],ci,rc,ga,la,lo,hi,weights,qa,ba,selected)
    audit_gain = {r['seed']:r['paired_gain_nats_lo95'] > 0 for r in rows if r['split']=='validation_audit'
                  and r['model']==selected['name']+' | additive + full opposite species'}
    selection.update(positive_nominal_audit_lower_bound_by_seed=audit_gain,
                     conclusion=('Conditional fidelity improvement across all three aligned teachers.'
                                 if all(audit_gain.values()) else
                                 'No consistently positive conditional fidelity improvement across all three aligned teachers.'))
    report = dict(**selected,species_key=SPECIES,coordinate_convention='centered (z,y,x), flat index x+3*y+9*z using 0..2',
                  coefficients='All nonzero terms have unit coefficients except the second term of a difference (-1).',
                  source='archived cold beta index 9; fitting uses visible species, conventional coordination control, four teacher probabilities only',
                  selection_procedure='For each fixed field use multiplicative cell calibration (sum teacher+1)/(sum prediction+1), then row normalize, on top of additive log-posterior projection calibrated within 9 exact corner pairs. Equal calibration passes for the no-field comparison. Require training overlap: min(low,high) counts >=32 in contributing corner strata, total >=128 for at least one level pair. Maximize smaller of train and selection-validation mean paired soft log gain. Within one paired selection-validation SE of best, prefer fewer referenced sites, then fewer levels, then gain, then dictionary order. Separate validation-audit parents never select the field.',
                  selection=selection,teacher_contrast=contrast,aligned_audit_contrast=effects,matched_observations=examples,
                  alignments=ALIGN,counts=dict(train_parents=len(ti),selection_parents=len(vi),audit_parents=len(ci)),
                  bank_provenance={fold:json.loads((path/'meta.json').read_text()) for fold,path in
                                   [('train',a.train),('validation',a.validation)]},
                  teacher_receipt_sha256=hashlib.sha256((STATIC/'fit-receipts.zip').read_bytes()).hexdigest(),
                  uncertainty='Normal 95% intervals over independent whole parents, paired for gains, conditional on trained fits. Selection-split intervals are descriptive. Audit intervals apply to the frozen field. Standardized probability contrasts use parent-cluster ratio influence functions and fixed train overlap weights.',
                  limits='An oriented teacher-fidelity measurement, not evidence of an irreducible many-body field, cooling, or dynamics. Matching conditions on corner species, not all local features.')
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'motif-rule.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (a.out/'motif-fidelity.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    plot(a.out,rows,selected,effects,examples,cat)


def compare_levels(g,l,q,b,cat,lo,hi,weights,r):
    out = {}; n = len(g)//r
    for key, val in [('teacher_probability',q[:,cat]),('teacher_minus_calibrated_additive_probability',q[:,cat]-b[:,cat])]:
        influence = np.zeros(n); delta = 0.; missing=[]
        for s in np.flatnonzero(weights):
            for lev,sgn in ((hi,1),(lo,-1)):
                hit = (g==s)&(l==lev); count = hit.reshape(n,r).sum(1)
                if not count.sum(): missing.append([int(s),int(lev)]); continue
                total = (hit*val).reshape(n,r).sum(1,dtype='d'); mean = total.sum()/count.sum()
                w = sgn*weights[s]; delta += w*mean; influence += w*(total-mean*count)/count.mean()
        out[key] = (dict(unsupported_audit_cells=missing) if missing else estimate(delta+influence))
    return out


def paired_examples(v,ids,r,g,l,lo,hi,weights,q,b,rule):
    # One observation per side, from the SAME real parent. Select by agreement on other
    # visible pixels then conventional probabilities, not teacher outcome. Demonstrations only.
    sites = rule['expression']['sites']; other = np.setdiff1d(np.arange(27),sites); choices=[]
    for p,pid in enumerate(ids):
        start=p*r; ix=np.arange(start,start+r); best=None
        for s in np.flatnonzero(weights):
            a=ix[(g[ix]==s)&(l[ix]==lo)]; z=ix[(g[ix]==s)&(l[ix]==hi)]
            if not len(a) or not len(z): continue
            dv=(v[a][:,None,other]!=v[z][:,other][None,:,:]).sum(2)
            ii,jj=np.where(dv==dv.min()); dist=np.abs(b[a[ii]]-b[z[jj]]).sum(1); t=dist.argmin()
            key=(int(dv[ii[t],jj[t]]),float(dist[t]),int(a[ii[t]]),int(z[jj[t]]))
            if best is None or key<best: best=key
        if best is not None: choices.append((best,p))
        if len(choices)>=96: break
    result=[]
    for (ham,dist,i,j),p in sorted(choices)[:3]:
        def rec(t):
            return dict(anchor_row=int(t-p*r),v=v[t].tolist(),value=float(observable(v[t:t+1],rule)[0]),
                        posterior=q[t].tolist(),additive_corner_posterior=b[t].tolist())
        result.append(dict(parent_id=int(ids[p]),off_support_hamming=ham,additive_corner_L1=dist,
                           changed_coordinates=[coord(t) for t in np.flatnonzero(v[i]!=v[j])],low=rec(i),high=rec(j)))
    return result


def plot(out,rows,rule,effects,examples,cat):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    fig=plt.figure(figsize=(10,7),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=(2,1.3))
    axes=[fig.add_subplot(grid[0,j]) for j in range(2)]
    for ax,(metric,title) in zip(axes,[('paired_gain_nats','Soft log-score gain beyond additive + corners'),
                                        ('teacher_minus_calibrated_additive_probability',f'Matched change in residual P(H={cat})')]):
        for x,seed in enumerate((11,29,47)):
            if metric=='paired_gain_nats':
                row=next(r for r in rows if r['split']=='validation_audit' and r['seed']==seed and r['model']==rule['name']+' | additive + full opposite species')
                m,se=row[metric+'_mean'],row[metric+'_se']
            else:
                stat=effects[str(seed)][metric]
                if 'mean' not in stat: continue
                m,se=stat['mean'],stat['se']
            ax.errorbar(x,m,1.96*se,fmt='o',capsize=4,color='#245783')
        ax.axhline(0,color='.5',lw=.7); ax.set_xticks(range(3),['11','29','47']); ax.set_xlabel('Aligned teacher seed'); ax.set_title(title,fontsize=10)
    axes[0].set_ylabel('nats per visible block'); axes[1].set_ylabel('probability difference'); fig.suptitle(rule['name']+'\nCold validation audit; whole-parent 95% intervals',fontsize=12)
    # Plot the first matched pair without replacing any site's archived species.
    if examples:
        example=examples[0]
        palette=ListedColormap(['#eeeeee','#eab468','#5984a8'])
        for j,key in enumerate(('low','high')):
            ax=fig.add_subplot(grid[1,j]); row=example[key]; v=np.array(row['v']).reshape(3,3,3)
            for z in range(3):
                x0=4*z
                ax.imshow(v[z],cmap=palette,vmin=0,vmax=2,extent=(x0-.5,x0+2.5,2.5,-.5),interpolation='nearest')
                ax.text(x0+1,-1,f'z={z-1:+d}',ha='center',fontsize=8)
                for y,x in np.ndindex(3,3):
                    ax.text(x0+x,y,('·','A','B')[v[z,y,x]],ha='center',va='center')
                    if 9*z+3*y+x in rule['expression']['sites']:
                        ax.add_patch(Rectangle((x0+x-.48,y-.48),.96,.96,fill=False,lw=2,edgecolor='#a22742'))
            ax.set(xlim=(-.6,10.6),ylim=(2.65,-1.4),aspect='equal')
            ax.set_title(f"Parent {example['parent_id']}, row {row['anchor_row']}: field={row['value']:g}, "
                         f"P(H={cat})={row['posterior'][cat]:.2f}",fontsize=9)
            ax.axis('off')
        fig.text(.5,.012,'Actual visible blocks from the same parent with identical corner species; red outlines = rule sites. · = vacancy.',
                 ha='center',fontsize=8)
    fig.savefig(out/'motif.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('train','validation','out'): parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())
