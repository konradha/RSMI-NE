"""Does proximity to a free bond predict loss at an initially uphill occupied site?"""
import argparse
import csv
from pathlib import Path

import numpy as np
from l20_propensity_predict import coordination_barrier


def four_shells(free):
    # Only distances 0..4 and >=5 are needed. Exact L1 distance on the torus.
    d=np.full(free.shape,5,np.int8);d[free]=0
    for _ in range(4):
        d=np.minimum.reduce([d]+[np.roll(d,s,ax)+1 for ax in range(3) for s in (-1,1)])
    return d


def collect(data):
    with np.load(data,allow_pickle=False) as f:
        cohort=str(f['cohort']); beta=float(f['beta']); lags=f['lags']; parents=f['parent_id']; reps=int(f['replicas'])
        origin=f['origin']; y=f['changed'].astype(np.float32)/reps
    if beta!=3.05: raise ValueError('Protocol only specifies beta 3.05')
    for i,pid in enumerate(parents):
        s=origin[i]; m,b=coordination_barrier(s); d=four_shells(b<=0)
        # +infinity is its own category; no unlike neighbor is not a zero barrier.
        keep=(s!=0)&(b>0); key=np.stack((s[keep],m[keep],b[keep]),axis=1)
        _,stratum=np.unique(key,axis=0,return_inverse=True); ns=stratum.max()+1
        yy=y[:,i,keep]
        for ctl,dist in [('aligned',d),('translated',np.roll(d,(10,10,10),axis=(0,1,2)))]:
            v=dist[keep]; far=v>=5
            nfar=np.bincount(stratum[far],minlength=ns)
            for label,near in [('1-2',(v>=1)&(v<=2)),('3-4',(v>=3)&(v<=4))]:
                nnear=np.bincount(stratum[near],minlength=ns);ok=(nnear>0)&(nfar>=8)
                used=near&ok[stratum]; n=used.sum()
                if not n:continue
                for j,t in enumerate(lags):
                    sfar=np.bincount(stratum[far],weights=yy[j,far],minlength=ns)
                    observed=float(yy[j,used].mean(dtype=np.float64)); matched=float(np.dot(nnear[ok],sfar[ok]/nfar[ok])/n)
                    yield dict(cohort=cohort,parent_id=int(pid),replicas=reps,lag=int(t),map=ctl,distance=label,free_sites=int((b<=0).sum()),
                               near_all=int(near.sum()),near_matched=int(n),near_risk=observed,far_matched_risk=matched,difference=observed-matched)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs',nargs='+',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    rows=[row for f in a.inputs for row in collect(f)];a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
