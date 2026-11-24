"""Registered native TRAIN subsets. Read only these species rows for this tranche."""
import argparse,json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from vacancy_dipole import selected_member
from rate import partition,digest,json_out


def run(data,base):
    base.mkdir(parents=True,exist_ok=True);manifest=json.loads((data/'release-index.json').read_bytes());pins={q['release_index']:q for q in manifest['endpoints']}
    for i in (9,0,8,1,2,3,4,5,7):
        n=5120 if i==9 else 768 if i in (0,8) else 256;fit=4096 if i==9 else 256;dev=4608 if i==9 else 512 if i in (0,8) else 256;fold=partition(i)[0];ids=fold[0][:n];src=data/f'beta-{i:02}.npz';receipt=json.loads(src.with_suffix('.json').read_bytes());path=base/f'b{i:02}.npz'
        if path.exists():raise FileExistsError(path)
        if i==6 or np.intersect1d(ids,np.r_[fold[1],fold[2]]).size:raise ValueError('forbidden cohort')
        if digest(src)!=receipt['endpoint_sha256'] or pins[i]['npz_sha256']!=receipt['endpoint_sha256']:raise ValueError('archive checksum mismatch')
        with ZipFile(src) as zp:d={k:selected_member(zp,k,ids,shape) for k,shape in [('species',(20,20,20)),('energy',()),('chain_seed',()),('chain_id',())]}
        if np.unique(np.stack([d['chain_seed'],d['chain_id']],1),axis=0).shape[0]!=n or not np.all((d['species']==1).sum((1,2,3))==2400):raise ValueError('lineage/composition')
        tmp=path.with_suffix('.partial.npz');np.savez(tmp,parent=ids,beta=float(receipt['beta']),**d);tmp.replace(path);json_out(path.with_suffix('.json'),dict(index=i,beta=float(receipt['beta']),source_sha256=receipt['endpoint_sha256'],split='native TRAIN order only',fit_stop=fit,dev_stop=dev,audit_stop=n,rows=n,bank_sha256=digest(path)));print(path,n,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.data,a.out)
