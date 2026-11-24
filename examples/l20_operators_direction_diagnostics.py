"""Independent diagnostics on frozen predictions: direction reliability and kinetic identity."""
import argparse
import csv
from pathlib import Path
import numpy as np
from l20_operators_branch import branching,literal


def csvout(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def centered(x,mask):
    x=np.where(mask,x,0.0)
    return np.where(mask,x-x.sum(-1,keepdims=True)/np.maximum(1,mask.sum(-1,keepdims=True)),0.0)


def run(data,motifs,out):
    with np.load(data,allow_pickle=False) as f:
        parent_id=f['parent_id'];origin=f['origin'];half=f['half_changed'];counts=f['changed'];r=int(f['replicas']);beta=float(f['beta']);lags=f['lags'];row={int(pid):i for i,pid in enumerate(parent_id)}
    with np.load(motifs,allow_pickle=False) as f:
        pids=f['parent_id'];p=np.array([row[int(v)] for v in pids]);anchor=f['anchor'].astype(int);ax=f['axis'];sgn=f['sign'];ix=f['target_index'];n=f['n_occupied'];h=f['h'];pred=f['predicted'];observed=f['observed'];mlags=f['lags'];models=list(map(str,f['models']))
    ok=(ix>=0);np.testing.assert_array_equal(ok.sum(-1),n);ix=np.maximum(ix,0)
    release,lost,cost=branching(origin,p,anchor,ax,sgn)
    # Check an actual bond from each signed orientation with literal full-lattice energy differences.
    for axis in (0,1,2):
        for s in (-1,1):
            candidates=np.flatnonzero((ax==axis)&(sgn==s))
            if len(candidates):
                k=candidates[len(candidates)//2]; a,b,dh=literal(origin[p[k]],anchor[k],axis,s)
                np.testing.assert_array_equal(a,release[k]);np.testing.assert_array_equal(b,lost[k]);assert dh==cost[k]
    rows=[];code_stats=[];escape=[];opposite=np.array([1,0,3,2,5,4])
    for j,t in enumerate(mlags):
        step=np.flatnonzero(lags==t).item();frac=[]
        for val,children in [(half[step],r//2),(counts[step]-half[step],r-r//2)]:
            x=val.reshape(len(origin),-1)[p[:,None,None],ix];frac.append((x*ok).sum(-1)/(children*np.maximum(n,1)))
        A,B=frac;valid=n>0;obs=(A*(r//2)+B*(r-r//2))/r
        np.testing.assert_allclose(obs[valid],observed[j][valid],atol=3e-7,rtol=0)
        A=centered(A,valid);B=centered(B,valid);base=centered(pred[j,models.index('barrier')],valid)
        enc=centered(pred[j,models.index('barrier+learned')],valid);rand=centered(pred[j,models.index('barrier+random')],valid)
        def mean(x,mask):return float(x[mask].mean())
        hj=h if h.ndim==1 else h[j]
        for pid in np.unique(pids):
            sel=pids==pid;mask=sel[:,None]&valid&(valid.sum(1)[:,None]>=2)
            if not mask.any():continue
            a=mean(A*A,mask);b=mean(B*B,mask);ab=mean(A*B,mask)
            vals=dict(beta=beta,lag=int(t),parent_id=int(pid),motifs=int(sel.sum()),directions=int(mask.sum()),split_cov=ab,split_var_a=a,split_var_b=b,
                      residual_barrier=mean((A-base)*(B-base),mask),residual_learned=mean((A-enc)*(B-enc),mask),residual_random=mean((A-rand)*(B-rand),mask))
            rows.append(vals)
            opened=sel[:,None]&(release>0)&(release[:,opposite]==0)&valid&valid[:,opposite]
            if opened.any():
                # Same central bond and axis, opposite sign: identical target geometry.
                d_obs=mean(obs-obs[:,opposite],opened);d_control=mean(base-base[:,opposite],opened)
                escape.append(dict(beta=beta,lag=int(t),parent_id=int(pid),pairs=int(opened.sum()),
                                   open_risk=mean(obs,opened),opposite_risk=mean(obs[:,opposite],opened),
                                   difference=d_obs,predicted_difference=d_control,residual=d_obs-d_control))
            for symbol in (-1,0,1,2,3):
                u=sel if symbol<0 else sel&(hj==symbol)
                code_stats.append(dict(beta=beta,lag=int(t),parent_id=int(pid),symbol=symbol,n_bonds=int(u.sum()),n_opening=int(np.count_nonzero(release[u].sum(-1))),n_new=int(release[u].sum()),n_lost=int(lost[u].sum()),n_downhill=int(np.count_nonzero(cost[u]<0))))
    out.parent.mkdir(parents=True,exist_ok=True);csvout(out.with_name(out.name+'-reliability.csv'),rows);csvout(out.with_name(out.name+'-branching.csv'),code_stats)
    csvout(out.with_name(out.name+'-escape.csv'),escape)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('input',type=Path);a.add_argument('motifs',type=Path);a.add_argument('prefix',type=Path);v=a.parse_args();run(v.input,v.motifs,v.prefix)
