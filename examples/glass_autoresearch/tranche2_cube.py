"""Archive-only three-bit pair projection and two-bond canonical cross ratio.

No simulated masks or label redraws: a corner substitutes counts on affected
coordination centers only. Coordinates in this module are (z,y,x).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from zipfile import ZipFile

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import t
from vacancy_free_energy import log_partition, verify as verify_coefficient

L, N_A, SAMPLE = 20, 2400, 512
ROOT = Path(__file__).resolve().parents[2] / 'results/l20-autoresearch/tranche2'
BONDS = {
    'cube': (((0,0,0),(0,0,-1)), ((0,1,0),(0,2,0)), ((1,0,0),(2,0,0))),
    'far': (((0,0,0),(0,0,-1)), ((0,8,0),(0,8,-1))),
    'near': (((0,0,0),(0,0,-1)), ((0,1,0),(0,1,-1))),
}
K = np.arange(7)
PARITY = np.array([1,-1,-1,1,-1,1,1,-1])
D = 4*K-16
B = (K-5)**2


def projection(logw):
    """Unique pair projection on each exterior, with no subtraction at roots.

    Normalization, three singles and three pairs span seven constraints. Their
    one-dimensional nullspace is parity s, hence q=p+lambda*s. Nonnegativity
    gives the complete bracket [-a,b] where a=min(p_even), b=min(p_odd).
    Write lambda=-a+(a+b)*sigmoid(v). Form both q classes from their positive
    endpoint distances, in logs, so even a root exponentially close to either
    boundary is attainable. d entropy/d lambda = -sum s log q, strictly
    decreasing on the open interval. No pseudocounts or probability clipping.
    """
    lp = logw-logsumexp(logw, axis=1, keepdims=True)
    even = PARITY > 0
    le, lo = lp[:,even], lp[:,~even]
    a, b = le.min(axis=1,keepdims=True), lo.min(axis=1,keepdims=True)
    span = np.logaddexp(a,b)
    with np.errstate(divide='ignore'):
        de = le + np.log(-np.expm1(a-le))
        do = lo + np.log(-np.expm1(b-lo))

    def corners(v):
        return (np.logaddexp(de,span-np.logaddexp(0.,-v)),
                np.logaddexp(do,span-np.logaddexp(0.,v)))

    def score(v):
        qe, qo = corners(v)
        return (qe.sum(1)-qo.sum(1))[:,None]

    hi = 16. + 8.*np.ptp(lp, axis=1,keepdims=True)
    low = -hi
    if np.any(score(low) > 0.) or np.any(score(hi) < 0.):
        raise ArithmeticError('pair projection boundary bracket failure')
    for _ in range(60):
        mid = (hi+low)*0.5
        left = score(mid)<0.
        low, hi = np.where(left,mid,low), np.where(left,hi,mid)
    lq = np.empty_like(lp)
    lq[:,even], lq[:,~even] = corners((hi+low)*0.5)
    p, q = np.exp(lp), np.exp(lq)
    err = max(np.max(np.abs(q.sum(1)-1.)), np.max(np.ptp((q-p)*PARITY,axis=1)))
    cubic_err = np.max(np.abs(lq@PARITY))
    if err > 3e-11 or cubic_err > 1e-9:
        raise ArithmeticError(('pair projection residual',err,cubic_err))
    kl = np.sum(p*(lp-lq),axis=1)
    if np.any(kl < -2e-11):
        raise ArithmeticError('invalid KL')
    return lp,lq,kl,float(err),float(cubic_err)


def geometry(bonds):
    ends = np.asarray(bonds,dtype='i2').reshape(-1,3)
    delta = np.concatenate((np.eye(3,dtype='i2'), -np.eye(3,dtype='i2')))
    sites = np.unique(np.concatenate((ends,(ends[:,None,:]+delta).reshape(-1,3))),axis=0)
    eq = np.all(sites[:,None,:]==ends[None,:,:],axis=2)
    eid = eq.argmax(0)
    adj = (np.abs(sites[:,None,:]-ends[None,:,:]).sum(2)==1).astype('i1')
    coords = np.indices((L,L,L),dtype='i2').reshape(3,-1).T
    ids = np.ravel_multi_index(np.moveaxis((coords[:,None,:]+sites)%L,2,0),(L,L,L)).astype('u2')
    bits = (np.arange(2**len(bonds))[:,None] >> np.arange(len(bonds))) & 1
    occ = np.stack((1-bits,bits),axis=2).reshape(len(bits),-1).astype('i1')
    if np.unique(eid).size != len(ends):
        raise ValueError('duplicate bond endpoint')
    return ids,eid,adj,occ


def substitutions(mask,coord,total,anchors,geom):
    ids,eid,adj,corners = geom
    mobile = mask[ids[:,eid]]
    eligible = ((mobile[:,::2]+mobile[:,1::2])==1).all(1)
    aid = anchors[eligible[anchors]]
    if not len(aid):
        raise ArithmeticError('empty per-parent evaluation; record zero denominators instead')
    local, current = ids[aid], mobile[aid]
    actual_k = coord[local]
    actual_o = mask[local]
    rest_k = actual_k - current @ adj.T
    base = np.stack([np.sum((actual_k==k)*actual_o,axis=1) for k in K],axis=1)
    # Subtract old contributions to get the unchanged histogram; each corner
    # adds only local coordination-center counts. No full candidate mask.
    candidate_k = rest_k[:,None,:] + (corners@adj.T)[None,:,:]
    candidate_o = np.broadcast_to(actual_o[:,None,:],candidate_k.shape).copy()
    candidate_o[:,:,eid] = corners
    dh = (np.stack([np.sum((candidate_k==k)*candidate_o,axis=2) for k in K],axis=2)
          - base[:,None,:])
    actual = current[:,1::2] @ (1 << np.arange(len(eid)//2))
    h = total+dh
    if np.any(dh[np.arange(len(aid)),actual]) or np.any(h<0) or np.any(dh.sum(2)):
        raise ArithmeticError('count substitution or bit convention failure')
    return int(eligible.sum()), aid, actual.astype('u1'), dh.astype('i1'), h


def common_bath(h,beta):
    # Center one invariant bath per exterior on the fractional midpoint census.
    mid = h.mean(1)
    lo = np.full(len(h),-100.)
    hi = -lo
    for _ in range(52):
        u = (lo+hi)*0.5
        lower = (mid*expit(u[:,None]-beta*D)).sum(1) < N_A
        lo,hi = np.where(lower,u,lo),np.where(lower,hi,u)
    u = (lo+hi)*0.5
    x = u[:,None]-beta*D
    p = expit(x)
    chi = (mid*p*expit(-x)).sum(1)
    da_hist = ((h[:,1]-h[:,0])+(h[:,3]-h[:,2]))*0.5
    db_hist = ((h[:,2]-h[:,0])+(h[:,3]-h[:,1]))*0.5
    da, db = (da_hist*p).sum(1), (db_hist*p).sum(1)
    l_sg = ((h[:,3]-h[:,1]-h[:,2]+h[:,0])*(np.logaddexp(0.,x)-beta*B)).sum(1)
    return u,chi,da,db,l_sg


def controls():
    bit = (np.arange(8)[:,None] >> np.arange(3))&1
    f = np.stack((bit[:,0]*bit[:,1],bit[:,1]*bit[:,2],bit[:,2]*bit[:,0]),axis=1)
    pair = bit@np.array([.4,-1.2,2.6]) + f@np.array([-.8,1.4,3.1])
    # Bracket a root vastly closer to a boundary than float64 epsilon times
    # its width, and exercise log q even when exp(log q) underflows.
    adversarial = np.array([0.,-1500.,-1200.,-3.,-1100.,-4.,-5.,-1000.])
    lp,lq,kl,err,ce = projection(np.stack((pair,130*pair,np.zeros(8),adversarial)))
    np.testing.assert_allclose(lp[:3],lq[:3],rtol=0.,atol=2e-12)
    return dict(coefficient=verify_coefficient(),pair_max_log_error=float(np.abs(lp[:3]-lq[:3]).max()),
                projection_moment_residual=err,projection_log_cubic_residual=ce,
                extreme_input_log_range=float(np.ptp(adversarial)), extreme_KL=float(kl[-1]))


def selected(bank,name,start,stop,size,shape):
    with bank.open(name+'.npy') as stream:
        v = np.lib.format.read_magic(stream)
        dims,col,dtype = {(1,0):np.lib.format.read_array_header_1_0,
                          (2,0):np.lib.format.read_array_header_2_0}[v](stream)
        if col or dims != (size,*shape) or dtype.kind not in 'ui':
            raise ValueError((name,dims,dtype))
        nbytes = int(np.prod(shape,dtype=np.int64))*dtype.itemsize
        stream.seek(start*nbytes,1)
        buf = stream.read((stop-start)*nbytes)
        return np.frombuffer(buf,dtype=dtype).reshape(stop-start,*shape)


def ratio(s,w):
    n = len(w)
    mean = s.sum()/w.sum()
    se = np.sqrt(n/(n-1)*np.sum(((s-mean*w)/w.sum())**2))
    rad = t.ppf(.975,n-1)*se
    return dict(mean=float(mean),se=float(se),ci95=[float(mean-rad),float(mean+rad)])


def hist_summary(v):
    q = [0.,.01,.1,.25,.5,.75,.9,.99,1.]
    return dict(quantiles=q,values=np.quantile(v,q).tolist())


def hashes(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def run(bank,out,index,phase,release):
    if (phase=='audit' and (index!=9 or release is None or not re.fullmatch(r'[a-f0-9]{40}',release))):
        raise ValueError('only cold audit, with explicitly released public SHA')
    size = 5120 if index==9 else 768
    if phase=='audit':
        start,stop = 4608,5120
    else:
        start = 4096 if index==9 else 256
        stop = start+128
    beta = 3.05 if index==9 else 0.
    suffix = f'{phase}-b{index:02d}'
    stem = '16-17-'+suffix
    paths = {k:out/f'{16 if k=="cube" else 17}-{k}-{suffix}.npz' for k in BONDS}
    receipt = out/(stem+'.json')
    if any(p.exists() for p in [receipt,*paths.values()]):
        raise FileExistsError('immutable completed evidence already exists')
    then = time.monotonic()
    c = controls()
    # 'species' is read only from the declared contiguous rows; do not load
    # resources for evaluation partitions under any development invocation.
    with ZipFile(bank) as z:
        rows = selected(z,'parent',start,stop,size,())
        ids = selected(z,'chain_id',start,stop,size,())
        seeds = selected(z,'chain_seed',start,stop,size,())
        boxes = selected(z,'species',start,stop,size,(L,L,L))
    if len(np.unique(rows))!=len(rows) or len(set(zip(ids,seeds)))!=len(rows):
        raise ValueError('duplicate parent/lineage')
    geom = {k:geometry(v) for k,v in BONDS.items()}
    raw = {g:{} for g in geom}; parent_records = {g:[] for g in geom}
    opportunity_fields = {}; total_h = []; worst_mom = worst_cubic = 0.
    max_sg = max_reverse = max_rev_pred = 0.
    for i,(row,box) in enumerate(zip(rows,boxes)):
        if not np.array_equal(np.bincount(box.ravel()),[2000,2400,3600]):
            raise ValueError('wrong parent species inventory')
        mask = (box!=0).astype('i1')
        coord = sum(np.roll(mask,step,axis=ax) for step in (-1,1) for ax in range(3)).ravel()
        mask = mask.ravel(); total = np.bincount(coord[mask!=0],minlength=7)
        total_h.append(total)
        anchors = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1617,index,int(row)]))).choice(L**3,SAMPLE,replace=False)
        for g in geom:
            eligible,aid,y,dh,h = substitutions(mask,coord,total,anchors,geom[g])
            lw = log_partition(h.reshape(-1,7),beta,N_A).reshape(h.shape[:2])
            n = len(aid); at = np.arange(n)
            d = dict(anchor=aid.astype('u2'),observed=y,delta_h=dh,logZ=lw)
            if g=='cube':
                lp,lq,kl,err,cerr = projection(lw-lw[:,:1])
                worst_mom = max(worst_mom,err); worst_cubic = max(worst_cubic,cerr)
                exact,matched = lp[at,y],lq[at,y]
                gain = exact-matched
                cubic = (lw-lw[:,:1])@PARITY
                d.update(logp=lp,logq=lq,KL=kl,gain=gain,cubic=cubic)
                metrics = dict(KL=kl,observed_gain=gain,exact_log_score=exact,pair_log_score=matched,
                               abs_cubic=np.abs(cubic),nonzero_cubic_over_1e_8=np.abs(cubic)>1e-8)
            else:
                u,chi,da,db,sg = common_bath(h,beta)
                cross = (lw[:,3]-lw[:,1])-(lw[:,2]-lw[:,0])
                pred = -da*db/chi
                # Re-evaluate both the coordinate differences and the bath
                # after reversing the first endpoint convention.
                rev = [1,0,3,2]
                ru,rc,ra,rb,rsg = common_bath(h[:,rev],beta)
                revcross = (lw[:,rev[3]]-lw[:,rev[1]])-(lw[:,rev[2]]-lw[:,rev[0]])
                max_reverse = max(max_reverse,float(np.abs(revcross+cross).max()))
                max_rev_pred = max(max_rev_pred,float(np.abs(pred-ra*rb/rc).max()))
                if g=='far':
                    if np.any(h[:,3]-h[:,1]-h[:,2]+h[:,0]):
                        raise ArithmeticError('far affected stencils overlap')
                    max_sg = max(max_sg,float(np.abs(sg).max()))
                d.update(L=cross,mu=u,chi=chi,delta_a=da,delta_b=db,semi_grand_L=sg,predicted=pred)
                metrics = dict(L=cross,abs_L=np.abs(cross),above_0_01=np.abs(cross)>=.01,
                               positive_above_1e_8=cross>1e-8,negative_below_minus_1e_8=cross< -1e-8,
                               semi_grand=sg,abs_semi_grand=np.abs(sg),abs_constraint_residual=np.abs(cross-sg),
                               predicted=pred,abs_predicted=np.abs(pred),prediction_MAE=np.abs(cross-pred),
                               prediction_bias=pred-cross,chi=chi)
            for k,v in d.items():
                raw[g].setdefault(k,[]).append(v)
            opportunity_fields[g] = list(metrics)
            parent_records[g].append([eligible,n,*[v.sum() for v in metrics.values()]])
    summaries = {}
    for g in geom:
        a = {k:np.concatenate(v) for k,v in raw[g].items()}
        p = np.array(parent_records[g]); m = opportunity_fields[g]
        np.savez_compressed(paths[g],native_parent=rows,chain_id=ids,chain_seed=seeds,
                            base_h=np.array(total_h,dtype='u2'),parent_columns=np.array(['all_eligible','sample_eligible',*m]),
                            parent_sums=p,offsets=np.r_[0,np.cumsum(p[:,1].astype('i8'))],**a)
        summary = {k:ratio(p[:,j+2],p[:,1]) for j,k in enumerate(m)}
        summary['sample_eligible'] = int(p[:,1].sum()); summary['all_eligible'] = int(p[:,0].sum())
        summary['opportunity_fraction'] = ratio(p[:,0],np.full(len(rows),L**3))
        summary['observed_corner_counts'] = np.bincount(a['observed'],minlength=a['logZ'].shape[1]).tolist()
        for k in (['KL','gain','cubic'] if g=='cube' else ['L','chi','predicted','semi_grand_L']):
            summary[k+'_distribution'] = hist_summary(a[k])
        if g=='cube':
            v = [summary[k] for k in ('KL','observed_gain')]
            summary['both_points_at_margin'] = all(s['mean']>=.001 for s in v)
            summary['practical_disposition'] = ('above' if all(s['ci95'][0]>=.001 for s in v) else
                                                'below' if any(s['ci95'][1]<.001 for s in v) else 'unresolved')
        if g=='far':
            lo,hi = summary['abs_L']['ci95']
            summary['practical_disposition'] = 'above' if lo>=.01 else 'below' if hi<.01 else 'unresolved'
        summaries[g] = summary
    c.update(archived_projection_moment_max=worst_mom,archived_projected_cubic_max=worst_cubic,
             far_semi_grand_max=max_sg,orientation_reversal_L_error=max_reverse,
             orientation_reversal_prediction_error=max_rev_pred)
    if max_reverse>2e-9 or max_rev_pred>2e-10 or max_sg!=0.:
        raise ArithmeticError('orientation or factorization control failed')
    if beta==0. and (max(abs(summaries['cube'][k]['mean']) for k in ['KL','observed_gain','abs_cubic'])>2e-12
                       or any(summaries[g]['abs_L']['mean']!=0. for g in ('far','near'))):
        raise ArithmeticError('beta-zero control failed')
    with receipt.open('x') as f:
        json.dump(dict(beta=beta,index=index,phase=phase,train_positions=[start,stop],parents=len(rows),
                       audit_release=release,all_anchors_per_parent=L**3,draws_per_parent=SAMPLE,
                       code_sha256=hashes(Path(__file__)),design_sha256=hashes(out/'16-17-design.json'),
                       coefficient_sha256=hashes(Path(__file__).with_name('vacancy_free_energy.py')),
                       evidence_sha256={p.name:hashes(p) for p in paths.values()},
                       elapsed_seconds=time.monotonic()-then,controls=c,stencils=summaries),f,indent=2,allow_nan=False)
        f.write('\n')
    print(receipt, 'sec',round(time.monotonic()-then,1),flush=True)
    print('cube',summaries['cube']['KL'],summaries['cube']['observed_gain'],'far',summaries['far']['abs_L'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=ROOT)
    parser.add_argument('--index',type=int,choices=[0,9],required=True)
    parser.add_argument('--phase',choices=['development','audit'],required=True)
    parser.add_argument('--audit-release')
    args = parser.parse_args()
    run(args.bank_dir/f'b{args.index:02d}.npz',args.out,args.index,args.phase,args.audit_release)
