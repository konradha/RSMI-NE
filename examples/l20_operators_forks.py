"""Acquire untouched parents for the directional compression test; CPU local dynamics only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from l20_pairs import partition
from l20_propensity_forks import digest, pack, save_npz

# t/tau-matched transfer, then a separately labelled equal-clock diagnostic.
PROTOCOL = {9: (512, (4096, 16384)), 8: (256, (116, 462, 4096))}
REPLICAS = 8
DOMAIN = 'l20-directional-confirmation-v1'


def acquire(index, release, prior, simulation, work, receipts):
    sys.path.insert(0, str(simulation.resolve()))
    from engine import Engine, audit, library
    P, lags = PROTOCOL[index]; R = REPLICAS
    start_time = time.monotonic()
    source = release / f'beta-{index:02d}.npz'
    sha = digest(source)
    if sha != json.loads(source.with_suffix('.json').read_bytes())['endpoint_sha256']:
        raise ValueError('release checksum mismatch')
    with np.load(source, allow_pickle=False) as f:
        beta = str(f['beta']); keys = np.column_stack((f['chain_seed'], f['chain_id']))
        seed_values, n = np.unique(keys[:,0], return_counts=True)
        test = partition(index)[0][-1]; pool = test[keys[test,0] == seed_values[n.argmax()]]
        if index == 9:
            dev = json.loads(prior.read_bytes()); assert dev['source_sha256'] == sha
            pool = np.setdiff1d(pool, dev['release_rows'])
        selection_seed = 7621000 + index
        rows = np.sort(np.random.default_rng(selection_seed).choice(pool,P,replace=False))
        origin = f['species'][rows]; h0 = f['energy'][rows]
    if len({hashlib.sha256(s).digest() for s in origin}) != P: raise ValueError('duplicate initial lattice')
    parent_keys = keys[rows]; assert len(np.unique(parent_keys,axis=0)) == P
    packed = pack(origin); np.testing.assert_array_equal(audit(20,packed),h0)
    ids = np.arange(P*R,dtype=np.uint64); seed = int.from_bytes(hashlib.sha256(f'{DOMAIN}/{index}/local'.encode()).digest()[:8],'little')
    base = work/f'raw-{index:02d}'; base.mkdir(exist_ok=False,parents=True)
    save_npz(base/'origin.npz', species=origin, parent_id=rows, chain_seed=parent_keys[:,0], chain_id=parent_keys[:,1], energy=h0, child_id=ids, child_seed=np.uint64(seed))
    hashes = {'origin.npz':digest(base/'origin.npz')}; changed=[]; half=[]; cursor=0; previous=np.zeros(P*R,np.uint64)
    with Engine('cpu',20,seed,ids,[beta]*len(ids),np.repeat(packed,R,axis=0),'local') as eng:
        for t in lags:
            eng.step(cursor,t-cursor); state,h,acc=eng.read()
            for j in range(0,len(ids),64):np.testing.assert_array_equal(audit(20,state[j:j+64]),h[j:j+64])
            if np.any(acc<previous) or np.any(acc-previous>(t-cursor)*8000):raise ValueError('invalid native acceptance counter')
            delta=(state.reshape(P,R,20,20,20)&3)^origin[:,None]
            change=delta!=0;changed.append(change.sum(1,dtype=np.uint8));half.append(change[:,:R//2].sum(1,dtype=np.uint8))
            name=f'local-{t}.npz';save_npz(base/name,species_xor_origin=delta,energy=h.reshape(P,R),accepted=acc.reshape(P,R),sweep=t);hashes[name]=digest(base/name)
            previous=acc;cursor=t
            print(f'Archived beta={beta}: {P} parents x {R} siblings x {t} sweeps',flush=True)
    destination=work/f'confirm-{index:02d}.npz';save_npz(destination,origin=origin,parent_id=rows.astype(np.uint64),lags=np.array(lags,dtype=np.int64),replicas=R,beta=float(beta),changed=np.stack(changed),half_changed=np.stack(half))
    proof=dict(protocol=DOMAIN,index=index,beta=beta,selection_seed=selection_seed,source_sha256=sha,pool='test split, bulk preparation only; exclude old development fork parents for beta 09',parents=P,replicas=R,lags=lags,attempts=P*R*8000*cursor,seconds=time.monotonic()-start_time,child_seed=str(seed),child_id='8*parent ordinal + sibling ordinal',release_rows=rows.tolist(),raw_sha256=hashes,analysis_input_sha256=digest(destination),native_library_sha256=digest(Path(library('cpu')._name)),simulation_commit=subprocess.check_output(['git','-C',str(simulation),'rev-parse','HEAD'],text=True).strip())
    receipts.mkdir(parents=True,exist_ok=True);(receipts/f'forks-{index:02d}.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(f'Sealed {destination}',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('index',type=int,choices=PROTOCOL)
    for opt in ('release','simulation','work','receipts'):ap.add_argument('--'+opt,type=Path,required=True)
    ap.add_argument('--prior',type=Path,default=Path('results/l20-propensity/cold-provenance.json'));a=ap.parse_args()
    acquire(a.index,a.release,a.prior,a.simulation,a.work,a.receipts)
