"""Independent L20 endpoints → periodic visible/shell label pairs. All splits are by parent."""
import hashlib
import json
from pathlib import Path

import numpy as np

L = 20
SPLIT = (10240, 2048, 4096)
PAIR_SEED = 769315


def offset_tables(buffer):
    outer = 2 + 2*(buffer+1)
    if buffer < 0 or outer > L:
        raise ValueError('shell wraps over itself')
    cube = np.indices((outer,)*3).reshape(3, -1).T - (buffer+1)
    shell = cube[(cube.min(axis=1) < -buffer) | (cube.max(axis=1) >= 2+buffer)]
    vis = np.indices((2,)*3).reshape(3,-1).T
    anchors = np.indices((L,)*3, dtype=np.int16).reshape(3,-1).T
    # 8,000 lookup rows. Build one axis at a time, not a (L^3, shell, 3) int64 temporary.
    def table(off):
        idx = np.zeros((L**3,len(off)), dtype=np.uint16)
        for d in range(3):
            idx *= L
            idx += ((anchors[:,d,None] + off[:,d].astype(np.int16)) % L).astype(np.uint16)
        return idx
    return table(vis),table(shell)


def partition(index):
    # Model seeds, buffers and operator choices never enter the split/anchor streams.
    split, *streams = map(np.random.default_rng, np.random.SeedSequence([PAIR_SEED,index]).spawn(4))
    return np.split(split.permutation(sum(SPLIT)), np.cumsum(SPLIT)[:-1]), streams


def load_banks(path, buffer, epochs):
    path = Path(path); index = int(path.stem.split('-')[-1]); receipt = json.loads(path.with_suffix('.json').read_bytes())
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if sha != receipt['endpoint_sha256']:
        raise ValueError(f'endpoint changed: {path}')
    with np.load(path,allow_pickle=False) as f:
        beta = float(f['beta']); species = f['species']; keys = np.column_stack((f['chain_seed'],f['chain_id']))
        if str(f['axes']) != 'chain,z,y,x' or float(receipt['beta']) != beta:
            raise ValueError('unexpected axes or beta')
    if species.dtype != np.uint8 or species.shape != (sum(SPLIT), L,L,L) or species.max() > 2 or len(np.unique(keys,axis=0)) != len(species):
        raise ValueError('invalid L20 release or duplicate lineage')
    state = species.reshape(len(species),-1); vt,et = offset_tables(buffer)
    parent_groups,streams = partition(index)
    banks = []
    for group,reps,rng,training in zip(parent_groups,(epochs,2,4),streams,(True,False,False)):
        v=np.empty((reps,len(group),8),np.uint8); e=np.empty((reps,len(group),et.shape[1]),np.uint8)
        for rep in range(reps):
            rows = rng.permutation(group) if training else group
            anchors = rng.integers(L**3,size=len(rows))
            v[rep] = state[rows[:,None],vt[anchors]]
            e[rep] = state[rows[:,None],et[anchors]]
        banks.append((v,e))
    meta=dict(beta=beta,index=index,buffer=buffer,source_sha256=sha,split=list(SPLIT),patch_seed=PAIR_SEED,
              shell_sites=et.shape[1],side=L,visible_side=2,nonlocal_sweeps=receipt['candidate_age'],
              parent_sha256=[hashlib.sha256(keys[g].astype('<u8').tobytes()).hexdigest() for g in parent_groups])
    return meta,*banks
