"""Finite nonphysical planted alphabet for the frozen forward HERO training control."""
import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import t
from hero_train import (ARMS, Bank, C, LAWS, LOG_C, Learner, PUBLIC, center_logits, dump_new,
                        hbits, logsoftmax, shell_logits, softmax, strata)

N = 131072
CASES = {'weak':.1, 'strong':.6, 'shuffled':0.}
V = np.full((2,2,27),2,dtype='u1')
V[:,0,0] = 1
V[:,1,26] = 1
V[1,:,[1,2,3]] = 0
OFF = np.array([r for r in product(range(-4,5),repeat=3) if max(map(abs,r))==4])
E = np.full((2,386),2,dtype='u1')
E[0,OFF[:,2]==-4] = E[1,OFF[:,2]==4] = 1
E[1,OFF[:,2]==-4] = E[0,OFF[:,2]==4] = 0


def alphabet(case,seed):
    rng = np.random.default_rng(seed)
    z, x = rng.integers(2,size=(2,N),dtype='u1')
    strength = .6 if case=='shuffled' else CASES[case]
    y = np.bitwise_xor(x,rng.random(N)>(1+strength)/2).astype('u1')
    if case=='shuffled':
        for nuisance in (0,1):
            ix = np.flatnonzero(z==nuisance)
            y[ix] = y[rng.permutation(ix)]
    return z,x,y


def exact(q,critic,rho):
    """Enumerate both central bits, both nuisance values, both axes and 8 candidate counts."""
    expected = hard = 0.
    for z,x,y,k in product(range(2),range(2),range(2),range(C)):
        py = (1+(1 if x==y else -1)*rho)/2
        weight = .25*py*math.comb(C-1,k)*.5**(C-1)
        n1 = y+k
        denom = (C-n1)*np.exp(critic[0])+n1*np.exp(critic[1])
        gain = LOG_C+critic[y]-np.log(denom)
        expected += weight*(q[z,x]@gain)
        hard += weight*gain[q[z,x].argmax()]
    return dict(expected_nats=float(expected),hard_nats=float(hard))


def oracle(rho):
    gold_q = np.broadcast_to(np.eye(2),(2,2,2))
    gold_critic = np.log(np.array([[1+rho,1-rho],[1-rho,1+rho]]))
    return {**exact(gold_q,gold_critic,rho),'normalized_mse_gain':rho*rho,
            'ordinary_and_nuisance_mse_gain':0.,'ordinary_and_nuisance_nats':0.}


def mean_ci(x):
    m,se = float(np.mean(x)),float(np.std(x,ddof=1)/math.sqrt(len(x)))
    w = float(t.ppf(.975,len(x)-1)*se)
    return dict(mean=m,lo=m-w,hi=m+w,se=se,n=len(x))


def partitions(z,seed):
    rng = np.random.default_rng(seed)
    groups = {'uniform':rng.permutation(len(z)).reshape(-1,C)}
    gs = []
    for j in (0,1):
        ii = rng.permutation(np.flatnonzero(z==j))
        gs.append(ii[:len(ii)//C*C].reshape(-1,C))
    groups['matched'] = np.concatenate(gs)
    return groups


def evaluate(m,data,groups,rho):
    z,x,y = data
    q,crit = softmax(center_logits(m,V).astype('d')),shell_logits(m,E).astype('d')
    # Known four equally probable central prototypes, no labelled regression or gradient.
    mu = q.mean(axis=(0,1))
    x_given_h = (q[:,1].sum(axis=0)-q[:,0].sum(axis=0))/np.maximum(4*mu,1e-300)
    ratio = np.exp(crit-np.log(np.exp(crit).mean(axis=0,keepdims=True)))
    h_given_y = ratio*mu
    h_given_y /= h_given_y.sum(axis=1,keepdims=True)
    pred = h_given_y@x_given_h
    sign = 2*x.astype(float)-1
    gain = 1-(sign-pred[y])**2
    # Fixed, analytic Var(X)=1 and baseline mean zero. Whole independent pseudo-parent.
    result = dict(exact=exact(q,crit,rho),mse_gain=mean_ci(gain),pred_x_given_y=pred.tolist(),
                  exact_mse_gain=float(rho*(pred[1]-pred[0])-.5*(pred@pred)),
                  q_by_nuisance_bit=q.tolist(),critic_by_shell=crit.tolist(),
                  freq=mu.tolist(),H_bits=float(hbits(mu)),H_given_V_bits=float(hbits(q).mean()),
                  I_bits=float(hbits(mu)-hbits(q).mean()),hard_freq=(np.bincount(q.argmax(-1).ravel(),minlength=4)/4).tolist())
    for law,idx in groups.items():
        lp = LOG_C+logsoftmax(crit[y[idx]],axis=1)
        prob = q[z[idx],x[idx]]
        result[law] = dict(expected_nats=mean_ci((lp*prob).sum(-1).mean(1)),
                           hard_nats=mean_ci(np.take_along_axis(lp,prob.argmax(-1)[...,None],axis=-1)[...,0].mean(1)))
    return result


def run(case,arch,law):
    train, hold = alphabet(case,99201), alphabet(case,99203)
    fit_groups, hold_groups = partitions(train[0],8101),partitions(hold[0],8102)
    z,x,y = train
    bank = Bank(V[z,x,None,:],E[y,None,:])
    ztra = strata(V)
    if ztra[0,0] != ztra[0,1] or ztra[1,0] != ztra[1,1] or ztra[0,0]==ztra[1,0]:
        raise ValueError('Nuisance must stratify without encoding the bit')
    learner = Learner(arch,6101,law,bank)
    hist=[]
    wall,cpu=time.monotonic(),time.process_time()
    for power in range(9,14):
        end=2**power
        hist.append(dict(updates=end, **learner.advance(end),
                         fit=evaluate(learner.m,train,fit_groups,CASES[case]),
                         dev=evaluate(learner.m,hold,hold_groups,CASES[case])))
    d=hist[-1]['dev']; common=np.mean([d[k]['expected_nats']['mean'] for k in LAWS]); m=d['mse_gain']
    gate=bool((common>.001 and m['mean']>.004 and m['lo']>0) if case=='weak' else
              (common>.06 and m['mean']>.15 and m['lo']>0) if case=='strong' else
              (common<=.0015 and m['mean']<=.002))
    path=PUBLIC/'controls'/f'{case}-{arch}-{law}.json'
    dump_new(path,dict(case=case,arch=arch,law=law,n_fit=N,n_dev=N,seed=6101,analytic_oracle=oracle(CASES[case]),
                       analytic_ordinary_zero_reason='Counts, site, coordination, nuisance do not vary with X; global shell counts do not vary with Y.',
                       stratum_by_nuisance_and_bit=ztra.tolist(), empirical_fit_support=bank.draws.support,
                       observations='Both evaluation laws use exhaustive disjoint groups, matched discards at most 14 residual pseudo-parents; group rather than candidate uncertainty for nats.',
                       checkpoints=hist, common_expected_nats=float(common),numerical_gate=gate,elapsed_sec=time.monotonic()-wall,cpu_sec=time.process_time()-cpu))
    print(case,arch,law,gate,common,m['mean'],flush=True)


def summarize():
    rows=[]
    for arch,law in product(ARMS,LAWS):
        arms={}
        for case in CASES:
            with (PUBLIC/'controls'/f'{case}-{arch}-{law}.json').open() as f:
                arms[case]=json.load(f)
        m={c:a['checkpoints'][-1]['dev']['mse_gain'] for c,a in arms.items()}
        # Same PRNG base deliberately pairs arms/cases; marginal CI separation is conservative.
        sep={c:bool(m[c]['lo']>m['shuffled']['hi'] and arms[c]['common_expected_nats']>arms['shuffled']['common_expected_nats']) for c in ('weak','strong')}
        rows.append(dict(arch=arch,law=law,gates={c:a['numerical_gate'] for c,a in arms.items()},
                         separation=sep,mse=m,common_expected_nats={c:a['common_expected_nats'] for c,a in arms.items()},
                         single_process_cpu_sec=sum(a['cpu_sec'] for a in arms.values()),
                         update_cpu_sec=sum(h['train_cpu_sec'] for a in arms.values() for h in a['checkpoints'])))
    dump_new(PUBLIC/'train-control-receipt.json',dict(passed=all(all(r['gates'].values()) and all(r['separation'].values()) for r in rows),
             control_mse_oracles={c:oracle(rho) for c,rho in CASES.items()}, rows=rows,
             synthetic_only=True, n_physical_inputs=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=CASES)
    ap.add_argument('--arch',choices=ARMS)
    ap.add_argument('--law',choices=LAWS)
    ap.add_argument('--summary',action='store_true')
    a=ap.parse_args()
    if a.summary:
        summarize()
    else:
        run(a.case,a.arch,a.law)
