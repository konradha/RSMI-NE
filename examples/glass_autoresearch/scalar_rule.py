"""Distill the scalar face operator into 11 readily countable block observables."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from face_bank import TILE,POW
from rate import digest,json_out


def scalar_counts(tile):
    """Array of (..., 8) ternary blocks; last axis z-major, x fastest."""
    if tile.shape[-1]!=8 or not np.isin(tile,(0,1,2)).all():raise ValueError('expected eight ordered ternary sites')
    names=['1','nA','nB']; cols=[np.ones(tile.shape[:-1],dtype='i1'),(tile==1).sum(-1),(tile==2).sum(-1)]
    for dist in (1,2,3):
        ij=np.array([(i,j) for i,j in combinations(range(8),2) if np.sum(np.abs(TILE[i]-TILE[j]))==dist]);left,right=tile[...,ij[:,0]],tile[...,ij[:,1]]
        for a,b,s in ((1,1,'AA'),(1,2,'AB'),(2,2,'BB')):
            hits=(left==a)&(right==b)
            if a!=b:hits|=(left==b)&(right==a)
            names.append(f'{s}{dist}');cols.append(hits.sum(-1))
    return names,np.stack(cols,axis=-1)


def main(o):
    lock=json.loads((o/'mode-lock.json').read_bytes());dest=o/'scalar-rule.json'
    if dest.exists() or digest(o/'modes.npz')!=lock['weights'] or lock['winning_sector']!='A1g':raise ValueError('expected original cold scalar lock')
    with np.load(o/'modes.npz',allow_pickle=False) as z:
        f=z['A1g_visible'][:,0];v=z['train_visible_mass'];target=z['A1g_pair'][:,0];linear=z['A1g_linear'][:,0]
    digits=np.arange(6561)[:,None]//POW%3;names,x=scalar_counts(digits);u=v>0;w=np.sqrt(v);coef,*_=np.linalg.lstsq(x*w[:,None],f*w,rcond=None);pred=x@coef
    if not np.allclose(pred[u],target[u],atol=1e-8,rtol=0):raise ValueError('invariant rule does not reproduce the general quadratic projection')
    a,*_=np.linalg.lstsq(x[:,:3]*w[:,None],f*w,rcond=None);lin=x[:,:3]@a
    if not np.allclose(lin[u],linear[u],atol=1e-8,rtol=0):raise ValueError('invariant composition rule does not reproduce additive projection')
    scales=[];tot=v@(f*f)
    for k in range(4):
        mat=x[:,:3+3*k];b,*_=np.linalg.lstsq(mat*w[:,None],f*w,rcond=None);scales.append(float(1-v@((f-mat@b)**2)/tot))
    json_out(dest,dict(lock_sha256=digest(o/'mode-lock.json'),observable='Scalar cage-composition score: invariant under all rotations and inversion of a 2x2x2 cube.',coordinates='Start with constant, nA, nB, then count unordered AA, AB, BB vertex pairs separated by Hamming distance 1 (edge), 2 (face diagonal), 3 (body diagonal). Each pair counted once; vacancy is the reference. Dot with the coefficients. Species labels 1=A, 2=B. Score sign follows the stored singular vector; simultaneous sign reversal of source and target changes no prediction.',coefficients=dict(zip(names,map(float,coef))),composition_only_coefficients=dict(zip(names[:3],map(float,a))),train_explained_variance_composition_then_distances_123=scales,support=int(u.sum()),max_abs_difference_from_full_quadratic_on_supported_tiles=float(abs(pred[u]-target[u]).max()),derivation='A cubic-invariant degree-two function is constant on the group orbits of single species sites and colored unordered vertex pairs. The cube group is transitive within each vertex pair Hamming distance. Cold-distribution weighted projection commutes with the group action. Labels on unsupported patterns are extrapolations; this is the score of the pair projection, not a proof an arbitrary tile, finite-state code, or full glass has only pair correlations.'))
    print('SCALAR_RULE',names,coef.tolist(),scales,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
