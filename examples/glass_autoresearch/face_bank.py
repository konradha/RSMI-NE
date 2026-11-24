"""Integer parent-group count bank of the SAME microscopic faces for all tile encoders."""
import argparse
from itertools import product
from pathlib import Path
import json
import numpy as np
from rate import source_rows, partition, digest, json_out
from tile_codec import TILE, POW

SHAPE=(6,6561,81)
FACES=[]
for axis in range(3):
    for normal in (-2,3):
        face=[]
        for uv in product(range(2),repeat=2):
            p=list(uv);p.insert(axis,normal);face.append(p)
        FACES.append(np.array(face))


def code(state,offsets,powers):
    z=np.zeros(state.shape,'i4')
    for xyz,w in zip(offsets,powers):z+=np.roll(state,tuple(-xyz),axis=(1,2,3)).astype('i4')*int(w)
    return z


def count(states):
    v=code(states,TILE,POW);out=np.empty(SHAPE,'u4')
    for j,face in enumerate(FACES):
        y=code(states,face,(1,3,9,27));out[j]=np.bincount((81*v+y).ravel(),minlength=6561*81).reshape(6561,81)
    if not np.array_equal(out.sum(2),np.broadcast_to(out[0].sum(1),(6,6561))):raise ValueError('orientations have unequal tile marginals')
    return out


def wrap_check(state):
    coord=np.array([[0,0,0],[19,19,19],[2,15,6],[19,0,19]])
    for offsets,powers in [(TILE,POW),*[(v,np.array([1,3,9,27])) for v in FACES]]:
        xyz=(coord[:,None,:]+offsets)%20;ref=state[tuple(xyz.transpose(2,0,1))]@powers
        if not np.array_equal(code(state[None],offsets,powers)[(0,*coord.T)],ref):raise ValueError('periodic digit convention mismatch')


def run(a):
    o=a.out;o.mkdir(parents=True,exist_ok=True);path=o/'meta.json';folds=(0,1) if a.index==9 else (1,)
    if a.index not in (9,4,8,0):raise ValueError('only registered cohorts')
    if a.index!=9:
        if not a.lock or not a.lock_sha256 or digest(a.lock)!=a.lock_sha256:raise ValueError('warm bank requires the published evaluation lock digest')
    if path.exists():raise FileExistsError('bank already sealed')
    (lookup,states),src=source_rows(a.data,a.index,folds);ids=partition(a.index)[0];wrap_check(states[0].reshape(20,20,20));sizes={}
    for f in folds:
        fn='validation' if f else 'train';g=32 if f else 1;shape=(g,*SHAPE) if f else SHAPE;tmp=o/(fn+'.partial.npy');out=np.lib.format.open_memmap(tmp,mode='w+',dtype='<u4',shape=shape);out[:]=0
        for i in range(0,len(ids[f]),64):
            s=states[lookup[ids[f][i:i+64]]].reshape(-1,20,20,20);c=count(s)
            if f:out[i//64]=c
            else:out+=c
        marg=out.reshape(g,*SHAPE).sum((2,3),dtype='i8');
        if not np.all(marg==8000*len(ids[f])//g):raise ValueError('incorrect accumulated groups')
        out.flush();del out;tmp.replace(o/(fn+'.npy'));sizes[fn]=len(ids[f])
    d=dict(index=a.index,beta=src['beta'],buffer=1,source_sha256=src['sha256'],train_parents=sizes.get('train',0),validation_parents=sizes['validation'],anchors_per_parent=8000,train_sha256=digest(o/'train.npy') if 0 in folds else None,validation_sha256=digest(o/'validation.npy'),lineage_sha256=src['lineage_sha256'],implementation_sha256=digest(Path(__file__)),groups=32,axis_order='-z,+z,-y,+y,-x,+x',face_normals=[-2,3],transfer_lock_sha256=a.lock_sha256,geometry='2x2x2 visible V digits z-major, x fastest, species 0/1/2; one unobserved layer, four registered face pixels as little-endian ternary word. All 8000 periodic anchors and six directions, whole parents kept together.');json_out(path,d);print('FACE_BANK',a.index,d['train_parents'],d['validation_parents'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True,type=int);p.add_argument('--data',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--lock',type=Path);p.add_argument('--lock-sha256');run(p.parse_args())
