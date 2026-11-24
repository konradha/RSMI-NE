"""Ten-beta V3 selection. Only prospectively fixed TRAIN-fold0 scores enter."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t
from hero_train import ARMS, LAWS, SEEDS
from full_hero_train import sha


def stamp(arch,law,seed,depth):
    return f'{arch}-{law}-d1-s{seed}-u{depth}'


class Published:
    """Existing immutable packets; reject ambiguous runs or changed gates."""
    def __init__(self,roots,release_sha,split_sha):
        self.roots=tuple(dict.fromkeys(Path(r) for r in roots));self.release=release_sha;self.split=split_sha
        self.cache={}

    def read(self,arch,law,seed,depth):
        key=stamp(arch,law,seed,depth)
        if key not in self.cache:
            hits=[r for r in self.roots if (r/'learning'/(key+'.json')).is_file()]
            if len(hits)!=1:
                raise ValueError(f'Need exactly one immutable receipt for {key}, got {len(hits)}')
            root=hits[0];r=json.loads((root/'learning'/(key+'.json')).read_text())
            if (r['arch']!=arch or r['law']!=law or r['seed']!=seed or r['denominator']!=1 or r['updates']!=depth
                    or r['source']!='archived-physical' or r['release_sha256']!=self.release or r['split_sha256']!=self.split
                    or r['dev']['macro_common_hard_nats'] is None or set(r['dev']['per_beta'])!={str(b) for b in range(10)}):
                raise ValueError('Checkpoint metadata/estimand mismatch')
            if depth>=8192 and not r['all_enrolled_fit_parents_exposed']:
                raise ValueError('Cannot advance full-sample claim without complete positive exposure')
            if r['checkpoint']!=key+'.npz' or sha(root/'checkpoints'/r['checkpoint'])!=r['checkpoint_sha256']:
                raise ValueError('Checkpoint digest mismatch')
            self.cache[key]=root,r
        return self.cache[key]

    def stats(self,arch,law,seeds,depth):
        return np.array([self.read(arch,law,s,depth)[1]['dev']['macro_common_hard_nats'] for s in seeds])

    def primary_choice(self):
        scores={l:{a:self.stats(a,l,SEEDS[:3],8192).tolist() for a in ARMS} for l in LAWS}
        chosen={l:max(ARMS,key=lambda a:(np.mean(scores[l][a]),a=='affine')) for l in LAWS}
        return chosen,scores

    def change(self,arch,law,start,end):
        old=[self.read(arch,law,s,start)[1] for s in SEEDS]
        new=[self.read(arch,law,s,end)[1] for s in SEEDS]
        delta=np.array([n['dev']['macro_common_hard_nats']-o['dev']['macro_common_hard_nats'] for o,n in zip(old,new)])
        center=float(delta.mean());seed_se=float(delta.std(ddof=1)/math.sqrt(len(delta)))
        seed_w=float(t.ppf(.975,len(delta)-1)*seed_se)
        bybeta={};variances=[]
        for b in range(10):
            def jk(r):
                vals=[r['dev']['per_beta'][str(b)][l]['delete_parent_block_hard_nats'] for l in LAWS]
                if any(x is None for row in vals for x in row):
                    raise ValueError('Undefined delete-parent-block complement on locked panel')
                return np.mean(vals,axis=0)
            d=np.mean([jk(n)-jk(o) for o,n in zip(old,new)],axis=0)
            v=float((len(d)-1)/len(d)*np.square(d-d.mean()).sum())
            variances.append(v);bybeta[b]=dict(paired_delete_block_changes=d.tolist(),variance=v)
        variances=np.asarray(variances)/100.
        var=float(variances.sum());df=15. if var==0 else float(var*var/(np.square(variances).sum()/15))
        parent_w=float(t.ppf(.975,df)*math.sqrt(var))
        return dict(start=start,end=end,paired_seed_changes=delta.tolist(),mean_change=center,
                    seed_ci95=[center-seed_w,center+seed_w],whole_parent_ci95=[center-parent_w,center+parent_w],
                    parent_welch_df=df,per_beta=bybeta,
                    below_margin_both=bool(center+max(seed_w,parent_w)<.001),
                    deteriorating_both=bool(center+max(seed_w,parent_w)<0.))


def depth_status(changes,current,maxdepth):
    latest=changes[-1]
    if latest['deteriorating_both']:
        return 'overfit: both paired upper bounds below zero'
    if len(changes)>=2 and all(x['below_margin_both'] for x in changes[-2:]):
        return 'plateau: two paired upper bounds below .001 at each of two doublings'
    if current>=maxdepth:
        return 'capped while rising' if latest['mean_change']>=.001 else 'capped before plateau demonstrated'
    return None
