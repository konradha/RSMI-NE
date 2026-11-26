"""Stage the final amended release as read-only native rows; no history outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'results/opening-mechanism'
ROSTER = ROOT / 'results/buffered-static-gate1/roster.npz'
N, L, FILLED, WIDTH = 16384, 20, 6000, 64


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage(runs, scratch, out):
    scratch.mkdir(parents=True, exist_ok=True)
    rmeta = json.loads(ROSTER.with_suffix('.json').read_text())
    require(sha(ROSTER) == rmeta['sha256'], 'original roster checksum')
    audit = runs / 'l20-final-verification/checks.json'
    manifest = (audit.parent / 'SHA256SUMS').read_text().splitlines()
    require(sha(audit) == next(line.split()[0] for line in manifest if line.split()[1] == 'checks.json'), 'final verification checksum')
    rows = []
    with np.load(ROSTER, allow_pickle=False) as bank:
        bid, keys, native, groups = [bank[k] for k in ('beta_index','ids','native_rows','group')]
    require(len(np.unique(keys[bid < 10],axis=0)) == 10*N, 'nonunique archive lineage')
    for b in range(10):
        study, i = ('l20-b2711-double-v1', 0) if b == 8 else ('l20-b305-production-v1', b)
        rel = Path(study) / 'samples' / f'beta-{i:02d}.npz'
        path = runs / rel
        receipt = json.loads(path.with_suffix('.json').read_text())
        digest = sha(path)
        require(digest == receipt['endpoint_sha256'] and receipt['independent_endpoints']==N, 'archive hash or declared count')
        idx = np.flatnonzero(bid == b)
        ordered = idx[np.argsort(native[idx])]
        require(np.array_equal(native[ordered],np.arange(N)), 'non-permutation of native rows')
        group = groups[ordered]
        require(np.array_equal(np.bincount(group),[6144,2048,4096,4096]), 'split drift')
        with np.load(path,allow_pickle=False) as a:
            actual = np.column_stack((a['chain_seed'], a['chain_id']))
            require(np.array_equal(actual,keys[ordered]) and actual.dtype == np.uint64, 'native uint64 IDs vs original split')
            require(str(a['axes'])=='chain,z,y,x' and str(a['beta']) == rmeta['beta_values'][b] and int(a['nonlocal_sweeps'])==receipt['candidate_age'], 'temperature, age or axes mismatch')
        # Stream uninterpreted bytes, not user-selected examples. Release checksum is verified above.
        dest = scratch / f'b{b:02d}.npy'; tmp = dest.with_suffix('.partial')
        with ZipFile(path) as z, z.open('species.npy') as source, tmp.open('wb') as target:
            shutil.copyfileobj(source,target,length=4<<20)
        os.replace(tmp,dest)
        arr = np.load(dest,allow_pickle=False,mmap_mode='r'); require(arr.shape==(N,L,L,L) and arr.dtype == np.uint8, 'input shape/dtype')
        np.save(scratch/f'g{b:02d}.npy',group,allow_pickle=False)
        rows.append(dict(beta_index=b, beta=receipt['beta'], source=str(rel), sweeps=receipt['candidate_age'], archive_sha256=digest, staged_sha256=sha(dest), group_sha256=sha(scratch/f'g{b:02d}.npy'), native_ids_sha256=hashlib.sha256(actual.astype('<u8',copy=False).tobytes()).hexdigest(), parents=N, train=8192, test=8192, test_reserve_descriptors_only=4096))
    dump(out/'inputs.json',dict(roster_sha256=rmeta['sha256'], release_audit_sha256=sha(audit), unique_parents=10*N, rows=rows))
    print(f'STAGED {10*N} unique release parents; ten archive SHA256s and all identities matched',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('runs',type=Path);p.add_argument('scratch',type=Path);p.add_argument('--out',type=Path,default=RESULT)
    a=p.parse_args();stage(a.runs,a.scratch,a.out)
