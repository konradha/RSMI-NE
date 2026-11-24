"""Nonphysical qualification for the full-archive forward optimizer and sampler."""
import argparse
import json
import time
from itertools import product

import numpy as np
import torch
from scipy.stats import norm
from hero_controls import CASES, E, N, V, evaluate, oracle, partitions
from hero_train import (ARMS, LAWS, _branch, center_logits, dump_new, indicators,
                        init_model, objective_grad, strata)
from full_hero_train import PUBLIC, Pool, Temperature, Trainer, branch, device_setup, loss_fn


def equality_timing():
    info = device_setup('cpu')
    rng = np.random.default_rng(100332)
    v = rng.integers(3,size=(64,27),dtype=np.uint8)
    e = rng.integers(3,size=(64,386),dtype=np.uint8)
    checks = {}
    for arch in ARMS:
        m = init_model(arch,6101)
        x,y = indicators(v),indicators(e)
        params = {k:torch.tensor(a,requires_grad=True) for k,a in m.items()}
        xt,yt = torch.from_numpy(x),torch.from_numpy(y)
        ref,g = objective_grad(m,v,e)
        loss = loss_fn(params,xt,yt)
        loss.backward()
        errs = {k:float(np.max(abs(g[k]-p.grad.numpy()))) for k,p in params.items()}
        for k,a in [('v',xt),('e',yt)]:
            errs[k+'_logits'] = float(np.max(abs(branch(params,a,k).detach().numpy()-_branch(m,a.numpy(),k)[0])))
        errs['objective'] = abs(ref-loss.detach().item())
        checks[arch] = errs
    # Real optimizer/timing, only generated ternary numbers; heterogeneous beta support + 64 anchors.
    ds = {b:Temperature(b,rng.integers(3,size=(n,64,27),dtype=np.uint8),
                         rng.integers(3,size=(n,64,386),dtype=np.uint8),np.arange(n))
          for b,n in enumerate((32,79,270,360))}
    timing = {}
    for arch,law in product(ARMS,LAWS):
        p = Pool(ds,law,6101)
        for _ in range(3):
            _,_,used = p.batch(32)
            for b,ix in used.items():
                pp = np.sort(ix//ds[b].q,axis=1)
                if np.any(np.diff(pp,axis=1)==0) or (law=='matched' and np.any(np.diff(ds[b].s.ravel()[ix],axis=1))):
                    raise AssertionError('Group membership broken')
        z = Trainer(ds,arch,law,6101,'cpu')
        timing[arch+'-'+law] = z.advance(256)
    passed = all(err<2e-6 for row in checks.values() for err in row.values())
    dump_new(PUBLIC/'optimizer-math-timing.json',dict(source='generated synthetic ternary arrays only',
             physical_parents=0,device=info,max_abs_tolerance=2e-6,checks=checks,passed=passed,
             timing=timing,synthetic_complete_sampler_group_invariants=True))
    if not passed:
        raise ArithmeticError('Gradient/logit identity check outside frozen tolerance')


def letters(case,seed):
    rng = np.random.default_rng(seed)
    z,x = rng.integers(2,size=(2,N),dtype='u1')
    beta = np.arange(N)%4
    rho = .6 if case=='shuffled' else CASES[case]
    y = np.bitwise_xor(x,rng.random(N)>(1+rho)/2).astype('u1')
    if case=='shuffled':
        for b,zz in product(range(4),range(2)):
            rows = np.flatnonzero((beta==b)&(z==zz))
            y[rows] = y[rng.permutation(rows)]
    return beta,z,x,y


def groups_by_beta(beta,z,seed):
    separate, merged = {}, {law:[] for law in LAWS}
    for b in range(4):
        loc = np.flatnonzero(beta==b)
        local = partitions(z[loc],seed+b*101)
        separate[b] = (loc,local)
        for l,ix in local.items():
            merged[l].append(loc[ix])
    return separate,{l:np.concatenate(xs) for l,xs in merged.items()}


def balanced_accuracy(pred,truth):
    # Simultaneous 95% coverage for both class recalls (Bonferroni Wilson).
    z = norm.ppf(1-.05/4)
    stats=[]
    for k in (0,1):
        use = truth==k
        n = int(use.sum()); p = float((pred[use]==k).mean())
        denom = 1+z*z/n
        mid = (p+z*z/(2*n))/denom
        w = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
        stats.append([p,mid-w,mid+w])
    mean,lo,hi = np.mean(stats,axis=0)
    return dict(mean=float(mean),lo=float(lo),hi=float(hi))


def direct_decode(m,z,x,b,hz,hx):
    symbols = center_logits(m,V).argmax(-1)
    def fit_map(h):
        return [int(np.count_nonzero((h==j)&(x==1))>np.count_nonzero((h==j)&(x==0))) for j in range(4)]
    mapping = fit_map(symbols[z,x])
    pred = np.array(mapping)[symbols[hz,hx]]
    nuisance = np.array(fit_map(z))[hz]
    perm = hx.copy()
    rng = np.random.default_rng(94831)
    for beta,zz in product(range(4),range(2)):
        mask = (b==beta)&(hz==zz)
        perm[mask] = rng.permutation(hx[mask])
    return dict(fit_category_to_X=mapping,dev=balanced_accuracy(pred,hx),
                matched_label_shuffle=balanced_accuracy(pred,perm),nuisance=balanced_accuracy(nuisance,hx))


def run(case,arch,law):
    t = time.monotonic()
    info = device_setup('cpu')
    beta,z,x,y = letters(case,99201)
    fit = {b:Temperature(b,V[z[beta==b],x[beta==b],None,:],E[y[beta==b],None,:],np.flatnonzero(beta==b))
           for b in range(4)}
    fit_groups = groups_by_beta(beta,z,8101)[1]
    hb,hz,hx,hy = letters(case,99203)
    perbeta,hold_groups = groups_by_beta(hb,hz,8102)
    zt = strata(V)
    if zt[0,0]!=zt[0,1] or zt[1,0]!=zt[1,1] or zt[0,0]==zt[1,0]:
        raise AssertionError('S leaked planted bit')
    model = Trainer(fit,arch,law,6101,'cpu')
    history=[]
    for end in (1024,2048,4096,8192):
        timing=model.advance(end)
        m=model.weights()
        val=evaluate(m,(hz,hx,hy),hold_groups,CASES[case])
        # One independent letter per pseudo-parent. Same-beta exhaustive disjoint partitions.
        history.append(dict(steps=end,timing=timing,fit=evaluate(m,(z,x,y),fit_groups,CASES[case]),dev=val))
    by_beta = {}
    for b,(ii,bg) in perbeta.items():
        scores = evaluate(m,(hz[ii],hx[ii],hy[ii]),bg,CASES[case])
        by_beta[b] = {k:scores[k] for k in (*LAWS,'mse_gain')}
    common=float(np.mean([b[l]['expected_nats']['mean'] for b in by_beta.values() for l in LAWS]))
    gain = val['mse_gain']
    passed = bool((common>.001 and gain['mean']>.004 and gain['lo']>0) if case=='weak' else
                  (common>.06 and gain['mean']>.15 and gain['lo']>0) if case=='strong' else
                  (common<=.0015 and gain['mean']<=.002))
    decode = direct_decode(m,z,x,hb,hz,hx)
    if case != 'shuffled':
        passed = passed and (decode['dev']['lo']>.75 and .49<=decode['matched_label_shuffle']['mean']<=.51
                             and .49<=decode['nuisance']['mean']<=.51)
    dump_new(PUBLIC/'controls'/f'{case}-{arch}-{law}.json',dict(source='nonphysical pseudo-parents',physical_parents=0,
             seed=6101,case=case,arch=arch,law=law,oracle=oracle(CASES[case]),
             comparator='Uniform global shell count, composition, central site, coordination and S contain no planted bit; zero normalized MSE gain and zero log-score gain in expectation.',
             hard_rule_recovery=decode,
             fit_seed=99201,dev_seed=99203,n_fit=N,n_dev=N,beta_groups=4,independent_unit='one letter per pseudo-parent',
             beta_counts=[N//4]*4,common_expected_nats=common,mse_gain=gain,dev_by_beta=by_beta,
             passed=passed,history=history,device=info,total_wall_seconds=time.monotonic()-t))


def summarize():
    rows = {}
    for arch,law in product(ARMS,LAWS):
        vals={c:json.loads((PUBLIC/'controls'/f'{c}-{arch}-{law}.json').read_text()) for c in CASES}
        checks={c:bool(vals[c]['passed']) for c in CASES}
        for c in ('strong','weak'):
            checks[c+'_separates_null'] = bool(vals[c]['mse_gain']['lo']>vals['shuffled']['mse_gain']['hi'] and
                vals[c]['common_expected_nats']>vals['shuffled']['common_expected_nats'])
        rows[arch+'-'+law] = dict(checks=checks,passed=all(checks.values()),
                      scores={c:dict(mse_gain=v['mse_gain'],common_expected_nats=v['common_expected_nats']) for c,v in vals.items()})
    dump_new(PUBLIC/'control-gate.json',dict(source='nonphysical pseudo-parents only',physical_parents=0,rows=rows,
             passed=all(row['passed'] for row in rows.values()) and json.loads((PUBLIC/'optimizer-math-timing.json').read_text())['passed']))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('math','plant','summary'))
    ap.add_argument('--case',choices=CASES); ap.add_argument('--arch',choices=ARMS); ap.add_argument('--law',choices=LAWS)
    args=ap.parse_args()
    if args.mode=='math': equality_timing()
    elif args.mode=='plant': run(args.case,args.arch,args.law)
    else: summarize()
