"""Export orientation-resolved portraits of the four symbols using archived validation motifs.

Reads the --motifs development bank emitted by the fit CLI. The displayed raw example per
symbol is the first by parent ID and anchor coordinate (no outcome-based selection).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from l20_operators_directional_geometry import CORE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--motifs', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('results/l20-operators/directional')); args=p.parse_args()
    meta=json.loads((args.model/'freeze.json').read_text())
    with np.load(args.input, allow_pickle=False) as f:
        ids, state=f['parent_id'], f['origin']
    with np.load(args.motifs, allow_pickle=False) as f:
        keep=np.isin(f['parent_id'], meta['validation_parent_id']); pid,h,ank,axis,sign=(f[k][keep] for k in ('parent_id','h','anchor','axis','sign'))
        occ=f['n_occupied'][keep]; obs=f['observed'][:,keep]; pred=f['predicted'][:,:,keep]; models=f['models'].tolist(); lags=f['lags'].tolist()
    if set(pid.tolist())!=set(meta['validation_parent_id']): raise ValueError('wrong validation motif bank')
    source_rows={int(p):i for i,p in enumerate(ids)}
    rows=np.array([source_rows[int(v)] for v in pid])
    basis=np.eye(3,dtype=int)[(axis[:,None]+np.arange(3))%3]; basis[:,0]*=sign[:,None]
    xyz=(ank[:,None]+CORE@basis)%20
    raw=state[rows[:,None],xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]]
    _, inverse, counts=np.unique(pid,return_inverse=True,return_counts=True)
    w=1/counts[inverse]; symbols=[]
    for k in range(4):
        take=np.flatnonzero(h==k); pos=np.lexsort((sign[take],axis[take],ank[take,2],ank[take,1],ank[take,0],pid[take])); i=take[pos[0]]
        prob=np.stack([np.average(raw[take]==s,weights=w[take],axis=0) for s in range(3)],axis=-1)
        fw=w[take,None]*(occ[take]>0)/(occ[take]>0).sum(1,keepdims=True)
        rates=(np.nan_to_num(obs[:,take])*fw).sum(1)/fw.sum(0)
        symbols.append(dict(symbol=k, bonds=len(take), parents=len(set(pid[take])), core_species_probability=prob.tolist(),
                            observed_face_risk=rates.tolist(), example=dict(parent_id=int(pid[i]), anchor=ank[i].tolist(), axis=int(axis[i]),
                            sign=int(sign[i]), species=raw[i].tolist(), n_occupied=occ[i].tolist(),
                            observed=obs[:,i].tolist(), prediction=pred[:,:,i].tolist())))
    artifact=dict(split='archived validation', directions=['-u','+u','-v','+v','-w','+w'], core_offsets=CORE.tolist(), lags=lags,
                  models=models, weighting='equal parent, then equal bond; conditional on code (and nonempty face for risk)',
                  examples='first validation bond per code by parent_id, anchor zyx, axis, sign; not selected by outcomes', symbols=symbols)
    (args.model/'validation-motifs.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
