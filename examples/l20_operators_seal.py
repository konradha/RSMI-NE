"""Verify and package existing physical observations. No simulation dependency."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

import numpy as np


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def energy(s):
    occ = (s != 0).astype(np.int16)
    m = sum(np.roll(occ, sign, axis=axis) for axis in (-3, -2, -1) for sign in (-1, 1))
    return (occ * (m - np.where(s == 1, 3, 5))**2).sum(axis=(-3, -2, -1))


def seal(work, release, out):
    from l20_pairs import partition
    files = {}; summary = {}; seeds = set()
    for idx, analysis in [(8, work/'confirm-08.npz'), (9, work/'primary-4096/input.npz')]:
        tag = f'{idx:02d}'; raw = work/f'raw-{tag}'; source = release/f'beta-{tag}.npz'
        source_sha = digest(source)
        assert source_sha == json.loads(source.with_suffix('.json').read_bytes())['endpoint_sha256']
        components = {f'confirm-{tag}.npz': analysis, f'raw-{tag}/origin.npz': raw/'origin.npz'}
        with np.load(analysis,allow_pickle=False) as a, np.load(raw/'origin.npz',allow_pickle=False) as orig, np.load(source,allow_pickle=False) as release_data:
            rows=orig['parent_id']; s=orig['species']; r=int(a['replicas']); lags=a['lags']; ids=orig['child_id']; seed=int(orig['child_seed']); p=len(rows)
            assert (p,r)==({8:256,9:512}[idx],8) and len(np.unique(rows))==p and seed not in seeds; seeds.add(seed)
            assert set(rows)<=set(partition(idx)[0][2]) and len(set(map(lambda x: hashlib.sha256(x).digest(),s)))==p
            np.testing.assert_array_equal(ids,np.arange(p*r)); np.testing.assert_array_equal(a['parent_id'],rows)
            np.testing.assert_array_equal(a['origin'],s); np.testing.assert_array_equal(release_data['species'][rows],s)
            for key in ('chain_seed','chain_id','energy'): np.testing.assert_array_equal(orig[key],release_data[key][rows])
            np.testing.assert_array_equal(energy(s),orig['energy']); assert float(a['beta'])==float(release_data['beta'])
            old=np.zeros((p,r),np.uint64); previous=0
            for j,lag in enumerate(lags):
                leaf=f'local-{lag}.npz'; path=raw/leaf; components[f'raw-{tag}/{leaf}']=path
                with np.load(path,allow_pickle=False) as checkpoint:
                    assert int(checkpoint['sweep'])==lag
                    delta=checkpoint['species_xor_origin']; accepted=checkpoint['accepted']; h=checkpoint['energy']
                    assert np.all(accepted>=old) and np.all(accepted-old<=(lag-previous)*8000); old=accepted; previous=lag
                    np.testing.assert_array_equal(np.count_nonzero(delta,axis=1),a['changed'][j]); np.testing.assert_array_equal(np.count_nonzero(delta[:,:r//2],axis=1),a['half_changed'][j]);
                    for start in range(0,p,8):
                        sl=slice(start,start+8); v=s[sl,None]^delta[sl]; assert v.max()<=2
                        for k in (1,2): assert np.all((v==k).sum((-3,-2,-1)) == (s[sl,None]==k).sum((-3,-2,-1)))
                        np.testing.assert_array_equal(energy(v),h[sl])
            summary[tag]=dict(beta=float(a['beta']),source_sha256=source_sha,parents=p,replicas=r,child_seed=str(seed),complete_sweeps=lags.tolist(),requested_but_not_saved=[16384] if idx==9 else [],component_sha256={k:digest(v) for k,v in components.items()})
            if idx==8:
                receipt=json.loads((out/'forks-08.json').read_bytes()); assert receipt['raw_sha256']=={Path(k).name:digest(v) for k,v in components.items() if k.startswith('raw-')}; assert digest(analysis)==receipt['analysis_input_sha256']
        files.update(components)
    with ZipFile(out/'physical-observations.zip','w',compression=ZIP_STORED) as z:
        for name,path in files.items(): z.write(path,name)
    manifest=dict(cohorts=summary,bundle_sha256=digest(out/'physical-observations.zip'),encoding='raw species at time t = origin species XOR species_xor_origin; counts audited against snapshots',scope='Previously completed snapshots only. No new dynamics. Cold secondary was stopped before saving; not analyzed.')
    (out/'archive.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"Verified {len(files)} components: {manifest['bundle_sha256']}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('work','release','out'): p.add_argument('--'+a,type=Path,required=True)
    args=p.parse_args(); seal(args.work,args.release,args.out)
