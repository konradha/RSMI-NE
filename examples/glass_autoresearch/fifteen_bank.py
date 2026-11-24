"""One read per archive for the supervisor panel. No test or reserved issue rows."""
import argparse,json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from vacancy_dipole import selected_member
from rate import partition,digest,json_out
from boundary_bank import PINS


def extract(data,out,index,primary_lock):
    primary=index==7;pin=PINS.get(index,(2.372222222222,'123d2ec8a0592bac467493229c4c6f2f201f7dc04904692af852db9816bcf7bc'))
    if primary and not primary_lock:raise ValueError('separate public primary authorization needed for index 7')
    if index not in (0,8,9,7):raise ValueError('cohort not authorized')
    folds=partition(index)[0];ids=folds[1][:512] if primary else folds[0][:256];src=data/f'beta-{index:02d}.npz';dst=out/f'b{index:02d}.npz';receipt=out/f'b{index:02d}.json';doc=json.loads(src.with_suffix('.json').read_bytes())
    if dst.exists() or receipt.exists():raise FileExistsError(dst)
    if (float(doc['beta']),doc['endpoint_sha256'])!=pin or digest(src)!=pin[1] or np.intersect1d(ids,folds[2]).size:raise ValueError('incorrect archive or split')
    with ZipFile(src) as z:arr={k:selected_member(z,k,ids,shape) for k,shape in [('species',(20,20,20)),('chain_id',()),('chain_seed',())]}
    if len(np.unique(np.stack((arr['chain_seed'],arr['chain_id']),1),axis=0))!=len(ids) or not np.all((arr['species']==0).sum((1,2,3))==2000):raise ValueError('lineage/composition mismatch')
    tmp=dst.with_suffix('.partial.npz');np.savez(tmp,parent=ids,beta=pin[0],**arr);tmp.replace(dst);json_out(receipt,dict(index=index,beta=pin[0],parents=len(ids),fold='native validation, first 512' if primary else 'native train, first 256',native_file_sha256=pin[1],bank_sha256=digest(dst),scope='secondary reused archive; no native test or index 6',published_primary_design=primary_lock));print(dst,len(ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--primary-design-commit');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for index in ((7,) if a.primary_design_commit else (9,0,8)):extract(a.data,a.out,index,a.primary_design_commit)
