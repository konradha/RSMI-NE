"""Independent CPU local children from never-trained-on L20 parents (beta 3.05)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
import zipfile

import numpy as np
from l20_pairs import partition

PARENTS, REPLICAS = 64, 16
LAGS = (16, 256, 4096, 16384)
TAG = 'l20-propensity-cold-v1'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save_npz(path, **arrays):
    tmp=path.with_suffix('.partial'); path.parent.mkdir(exist_ok=True,parents=True)
    with tmp.open('xb') as fp: np.savez_compressed(fp, **arrays)
    tmp.replace(path)


def pack(species):
    out=species.copy(); occupied=np.uint8(species!=0)
    for dim,shift in ((3,2),(2,4),(1,6)):
        out |= np.roll(occupied,-1,axis=dim)<<shift
        out |= np.roll(occupied,1,axis=dim)<<(shift+1)
    return out.reshape(len(out),-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('simulation','data','fits','work','provenance'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.simulation.resolve()));from engine import Engine,audit,library
    raw=a.work/'raw'/TAG;raw.mkdir(parents=True,exist_ok=False)
    source=a.data/'beta-09.npz';sha=digest(source)
    if sha!=json.loads(source.with_suffix('.json').read_bytes())['endpoint_sha256']:raise ValueError('bad release checksum')
    with np.load(source,allow_pickle=False) as f:
        keys=np.column_stack((f['chain_seed'],f['chain_id']));heldout=partition(9)[0][-1]
        split_hash=hashlib.sha256(keys[heldout].astype('<u8').tobytes()).hexdigest()
        with zipfile.ZipFile(a.fits) as fits:
            for seed in (11,29,47):
                ref=json.loads(fits.read(f'fits/beta-09_b1_learned_{seed}.json'))['provenance']
                if (ref['parent_sha256'][-1],ref['source_sha256'])!=(split_hash,sha):raise ValueError('wrong frozen encoder split')
        seeds,ns=np.unique(keys[:,0],return_counts=True);pool=heldout[keys[heldout,0]==seeds[np.argmax(ns)]]
        rows=np.sort(np.random.default_rng(3098417).choice(pool,size=PARENTS,replace=False))
        origin=f['species'][rows]; expected_h=f['energy'][rows]; beta=float(f['beta']);assert beta==3.05
    packed=pack(origin);np.testing.assert_array_equal(audit(20,packed),expected_h)
    children=np.arange(PARENTS*REPLICAS,dtype=np.uint64);seed=int.from_bytes(hashlib.sha256((TAG+'/local').encode()).digest()[:8],'big')
    start=np.ascontiguousarray(np.repeat(packed,REPLICAS,axis=0)); source_ids=keys[rows]
    save_npz(raw/'origin.npz',origin=origin,release_row=rows,chain_seed=source_ids[:,0],chain_id=source_ids[:,1],energy=expected_h,child_id=children,child_seed=np.uint64(seed))
    sums={'origin.npz':digest(raw/'origin.npz')}; counts=[];halves=[];pos=0;before=monotonic();last_counts=np.zeros(len(children),np.uint64)
    with Engine('cpu',20,seed,children,['3.05']*len(children),start,'local') as eng:
        initial,h,acc=eng.read();np.testing.assert_array_equal(initial,start);np.testing.assert_array_equal(h,np.repeat(expected_h,REPLICAS));assert not np.any(acc)
        for t in LAGS:
            eng.step(pos,t-pos);state,h,acc=eng.read()
            for i in range(0,len(children),32):np.testing.assert_array_equal(audit(20,state[i:i+32]),h[i:i+32])
            if np.any(acc<last_counts) or np.any(acc-last_counts>(t-pos)*8000):raise ValueError('impossible acceptance counters')
            grid=(state&3).reshape(PARENTS,REPLICAS,20,20,20);diff=grid!=origin[:,None]
            counts.append(diff.sum(1,dtype=np.uint8));halves.append(diff[:,:REPLICAS//2].sum(1,dtype=np.uint8))
            name=f'sweep-{t}.npz';save_npz(raw/name,species=grid,energies=h,accepted=acc,child_id=children,sweep=np.int64(t));sums[name]=digest(raw/name)
            print(f'Committed local forks: {PARENTS} parents × {REPLICAS} children × {t} sweeps',flush=True);pos=t;last_counts=acc
    dest=a.work/'inputs/cold-replicated.npz'
    save_npz(dest,cohort='cold-replicated',index=9,beta=beta,replicas=REPLICAS,parent_id=rows.astype(np.uint64),origin=origin,lags=np.array(LAGS),roles=np.array(['primary' if t==4096 else 'short' for t in LAGS]),changed=np.stack(counts),half_changed=np.stack(halves))
    proof=dict(cohort='cold-replicated',beta=beta,parents=PARENTS,replicas=REPLICAS,lags=LAGS,primary_lag=4096,seconds=monotonic()-before,local_attempts=len(children)*pos*8000,
               selection='PCG64(3098417), 64 without replacement among bulk-only rows of the original beta-09 encoder test split',source_sha256=sha,encoder_test_sha256=split_hash,
               release_rows=rows.tolist(),source_identity=[[str(int(s)),str(int(i))] for s,i in source_ids],child_seed=str(seed),stream_layout='child_id = 16 * sorted-parent ordinal + sibling ordinal',
               raw=TAG,raw_file_sha256=sums,analysis_input_sha256=digest(dest),simulator_commit=subprocess.check_output(['git','-C',str(a.simulation),'rev-parse','HEAD'],text=True).strip(),cpu_library_sha256=digest(Path(library('cpu')._name)))
    a.provenance.parent.mkdir(exist_ok=True,parents=True);a.provenance.write_text(json.dumps(proof,indent=2)+'\n');print(dest,flush=True)

if __name__=='__main__':main()
