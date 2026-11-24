"""Positive control only: replace the physical environment by one visible label."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax, xlogy
from l20_fit import fit
from l20_pairs import load_banks


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    meta,*banks=load_banks(a.data/'beta-00.npz',0,24)
    planted=[(v,v[:,:,:1].copy()) for v,e in banks];del banks
    meta.update(environment='one copied visible label; not a physical shell',shell_sites=1)
    r=fit(meta,*planted,'learned',11)
    v,_=planted[-1]; w=np.array(r['weights']);v=v.reshape(-1,8)
    prob=softmax(np.stack([(v==s)@w for s in range(3)],axis=1),axis=1)
    joint=np.stack([prob[v[:,0]==s].sum(0) for s in range(3)])/len(v)
    r['plugin_MI_H_copied_label_nats']=float(xlogy(joint,joint/(joint.sum(0)[None,:]*joint.sum(1)[:,None])).sum())
    a.out.write_text(json.dumps(r,allow_nan=False,indent=2)+'\n');print('planted',r['test_nats'],r['test_shuffled_nats'],r['plugin_MI_H_copied_label_nats'])
