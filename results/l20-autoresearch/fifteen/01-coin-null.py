"""Direction-sensitive beta-zero TRAIN calibration; never score validation here.

The orientation-folded favored-end statistic is exactly half at forecast ties
by construction. This distinct, descriptive check retains the native positive
axes, so actual left-end occupancy is NOT forced to half. Recounts all shells
using full vectors and subtraction, independent of symmetric-difference LUTs.
"""
import argparse
import json
from itertools import product
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'examples'/'glass_autoresearch'))
from fifteen_mask import ratio, sha


def run(bank, out):
    meta=json.loads(bank.with_suffix('.json').read_text())
    if meta['index']!=0 or meta['fold']!='native train, first 256' or sha(bank)!=meta['bank_sha256']:
        raise ValueError('only the pinned beta-zero development roster')
    d=np.load(bank,allow_pickle=False); boxes=d['species']; n=len(boxes)
    left=np.zeros((n,3),'i4'); count=np.zeros_like(left)
    offsets=[tuple(v for v in product((-1,0,1),repeat=3) if sum(k*k for k in v)==r) for r in (1,2,3)]
    for i, box in enumerate(boxes):
        occupied=(box!=0).astype('i1')
        totals=[sum(np.roll(occupied,v,(0,1,2)) for v in shell) for shell in offsets]
        for axis in range(3):
            right=np.roll(occupied,-1,axis); ok=occupied!=right
            for shell,t in enumerate(totals):
                # Only shell 1 contains the other candidate, which must be removed.
                a=t-right if shell==0 else t
                tr=np.roll(t,-1,axis)
                b=tr-occupied if shell==0 else tr
                ok &= a==b
            count[i,axis]=ok.sum(); left[i,axis]=occupied[ok].sum()
    src=np.load(out/'01-development-b00-parents.npz',allow_pickle=False)
    assert np.array_equal(d['parent'],src['parent'])
    j=src['fields'].tolist().index('eligible')
    assert np.array_equal(count.sum(1),src['metrics'][:,j])
    nf=out/'01-coin-null-parents.npz'
    if nf.exists(): raise FileExistsError(nf)
    np.savez_compressed(nf,parent=d['parent'],chain_id=d['chain_id'],chain_seed=d['chain_seed'],eligible=count,positive_frame_left_occupied=left)
    L=left.sum(1); W=count.sum(1)
    r=dict(cohort='beta0 first 256 native TRAIN, not an independent transfer',item=1,
           contrast='Empirical P(left occupied | eligible,+axis) minus 1/2, raw native orientation, not tautological folded-at-tie diagnostic. No fitted orientation or reweighted sample.',
           independent_parent_counts_verified=True,eligible=int(W.sum()),left_occupied=int(L.sum()),parents=n,
           signed_residual=ratio(L-.5*W,W),per_axis=[ratio(a-.5*b,b) for a,b in zip(left.T,count.T)],
           independent_full_shell_lookup='All parent counts match symmetric-difference screen using an independent explicit full-shell census including every shared neighbor and removing the opposite endpoint from shell one.',
           decision='Descriptive beta0 calibration only; no alteration of the public transfer gate.',
           bank_sha256=sha(bank),per_parent_sha256=sha(nf),code_sha256=sha(Path(__file__)))
    with (out/'01-coin-null.json').open('x') as f:
        json.dump(r,f,indent=2);f.write('\n')
    print(r['eligible'],r['left_occupied'],r['signed_residual'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bank',type=Path);p.add_argument('--out',type=Path,default=Path(__file__).parent)
    a=p.parse_args();run(a.bank,a.out)
