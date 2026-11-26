"""Parallel all-site reducer. Atomic, checksum-pinned shards; no trajectory input."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes as ct
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from opening_inputs import ROOT, RESULT, N, L, FILLED, WIDTH, dump, require, sha

# [0], (0,2^-128), [2^-128,2^-127), ..., [2^7,inf).
EDGES = np.exp2(np.arange(-128,8,dtype=float)); NB = len(EDGES)+2
MOMENTS = ('mean_Vh','mean_V2O','fraction_h_zero','fraction_O_zero')
STATE = {}


def bins(values):
    # For positive v, floor(log2(v)) == frexp(v).exponent - 1, even at exact powers of two.
    ix = np.clip(np.frexp(values)[1].astype(np.int32)+129, 1, NB-1)
    ix[values==0]=0
    return ix


def startworker(scratch,lib,manifest,signature):
    state = [np.load(scratch/f'b{i:02d}.npy',mmap_mode='r') for i in range(10)]
    group = [np.load(scratch/f'g{i:02d}.npy') for i in range(10)]
    fun = ct.CDLL(str(lib)).opening_oracle
    fun.argtypes = [ct.c_int,ct.c_double]+[ct.c_void_p]*5; fun.restype=ct.c_int
    STATE.update(state=state,group=group,fun=fun,manifest=manifest,scratch=scratch,sign=signature,
                 vh=np.empty(L**3),op=np.empty(L**3),de=np.empty((L**3,6),np.int8),coord=np.empty(L**3,np.uint8))


def shard(job):
    b,lo = job; begin=time.monotonic(); s=STATE; beta=float(s['manifest']['rows'][b]['beta'])
    groups=s['group'][b][lo:lo+WIDTH]; sparse_k=[]; sparse_n=[]; end=[0]
    rmom=np.zeros(4); group_sums=np.zeros((4,NB,NB),np.uint32)
    retained=np.flatnonzero(groups!=3).astype(np.uint8); moments=np.empty((len(retained),4))
    rv=np.zeros(4); r=0; vh,op,de,coord=(s[k] for k in ('vh','op','de','coord'))
    for j,g in enumerate(groups):
        x=s['state'][b][lo+j].reshape(-1); occupied=x!=0
        require(np.count_nonzero(occupied)==FILLED and x.max()<=2,'native species composition')
        rc=s['fun'](L,beta,*[a.ctypes.data for a in (x,vh,op,de,coord)])
        require(rc==0 and np.isfinite(vh).all() and np.isfinite(op).all() and vh.min()>=0 and op.min()>=0 and vh.max()<=2+1e-14,'oracle status/support')
        h,o=vh[occupied],op[occupied]; k=bins(h)*NB+bins(o)
        freq=np.bincount(k,minlength=NB*NB).astype(np.uint16); group_sums[g]+=freq.reshape(NB,NB)
        m=np.array([h.mean(),o.mean(),np.count_nonzero(h==0)/FILLED,np.count_nonzero(o==0)/FILLED])
        if g==3:
            rmom+=m;rv+=m*m
        else:
            nz=np.flatnonzero(freq).astype(np.uint16); sparse_k.append(nz);sparse_n.append(freq[nz]);end.append(end[-1]+len(nz));moments[r]=m;r+=1
    require(np.array_equal(group_sums.sum(axis=(1,2)),np.bincount(groups,minlength=4)*FILLED),'joint count loss')
    name=f'b{b:02d}-{lo:05d}'; path=s['scratch']/'shards'/f'{name}.npz'; tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f,retained=retained,moments=moments,group_joint=group_sums,reserve_sum=rmom,reserve_square_sum=rv,
                            offsets=np.array(end,np.uint32),cell=np.concatenate(sparse_k) if sparse_k else np.empty(0,np.uint16),count=np.concatenate(sparse_n) if sparse_n else np.empty(0,np.uint16))
    tmp.replace(path)
    seconds=time.monotonic()-begin
    rec=dict(beta_index=b,lo=lo,parents=WIDTH,occupied_sites=WIDTH*FILLED,seconds=seconds,sha256=sha(path),signature=s['sign']);dump(path.with_suffix('.json'),rec)
    return rec


def main(a):
    manifest=json.loads((a.out/'inputs.json').read_text()); qual=json.loads(a.qualification.read_text())
    require(qual[a.qualification_key] is True,'no positive native oracle qualification')
    exe=ROOT/'examples/opening_oracle.cpp'
    sig=hashlib.sha256(' '.join(sha(p) for p in (exe,Path(__file__),a.lib,a.qualification,a.out/'inputs.json')).encode()).hexdigest()
    for i,meta in enumerate(manifest['rows']):
        require(sha(a.scratch/f'b{i:02d}.npy')==meta['staged_sha256'] and sha(a.scratch/f'g{i:02d}.npy')==meta['group_sha256'],'staged source changed')
    d=a.scratch/'shards';d.mkdir(parents=True,exist_ok=True); hist={};t0=time.monotonic();initial=0
    # Interleave temperatures so the first full wave measures the actual heterogeneous workload.
    jobs=[(b,lo) for lo in range(0,N,WIDTH) for b in (0,9,1,8,2,7,3,6,4,5)]
    for job in jobs:
        b,lo=job; path=d/f'b{b:02d}-{lo:05d}.json'
        if path.exists():
            rec=json.loads(path.read_text());require(rec['signature']==sig and rec['sha256']==sha(path.with_suffix('.npz')) and rec['lo']==lo and rec['beta_index']==b,'untrusted resume shard');hist[job]=rec
    initial=len(hist);last=0
    def update():
        nonlocal last
        elapsed=time.monotonic()-t0;now=datetime.now(timezone.utc);completed=len(hist)*WIDTH
        by_beta=[sum(k[0]==b for k in hist)*WIDTH for b in range(10)]
        speed=(len(hist)-initial)*WIDTH/elapsed if len(hist)>initial else None
        eta=None;weighted_rate=None
        if all(by_beta) and speed:
            elapsed_shards=[sum(r['seconds'] for k,r in hist.items() if k[0]==b)/(by_beta[b]/WIDTH) for b in range(10)]
            progress_seconds=sum(r['seconds'] for k,r in hist.items() if k not in old)
            observed_parallelism=progress_seconds/elapsed
            if observed_parallelism>0:
                remaining=sum(((N-by_beta[b])/WIDTH)*v for b,v in enumerate(elapsed_shards))/observed_parallelism
                eta=(now+timedelta(seconds=remaining)).isoformat();weighted_rate=(10*N-completed)/remaining if remaining>0 else None
        physical_path=a.out/'history/decision.json'; numeric_path=a.out/'numerics/decision.json'
        numeric=(json.loads(numeric_path.read_text())['comparison_allowed'] if numeric_path.exists() else None)
        rec=dict(phase='census_complete' if completed==10*N else 'census_running',
                 timestamp_utc=now.isoformat(),completed_release_parents=completed,target_release_parents=10*N,
                 completed_release_occupied_sites=completed*FILLED,by_beta_parents=by_beta,workers=a.workers,
                 completed_shards=len(hist),elapsed_this_invocation_s=elapsed,throughput_parents_s=speed,
                 eta_utc=eta,remaining_mix_parents_s=weighted_rate,
                 eta_basis='Measured shard wall time per beta and observed parallelism; load can change.',
                 first_timed_shard=first[0],signature=sig,physical_gate='see history/decision.json' if physical_path.exists() else 'pending',
                 numerical_gate='no_go' if numeric is False else 'pass' if numeric is True else 'pending',
                 downstream_comparison='blocked_by_numerics' if numeric is False else 'locked_until_both_gates_pass',
                 terminal_decision=None)
        dump(a.out/'status.json',rec);last=time.monotonic()
        print(f'CENSUS {completed}/{N*10} parents; {completed*FILLED} occupied sites; {speed or 0:.2f} parents/s since invocation; ETA {eta}',flush=True)
    old=set(hist);first=[None]
    if hist:
        oldstatus=json.loads((a.out/'status.json').read_text());first[0]=oldstatus.get('first_timed_shard')
    pending=[j for j in jobs if j not in old]
    first_mix_pending=bool(pending)
    from concurrent.futures import wait, FIRST_COMPLETED
    with ProcessPoolExecutor(a.workers,initializer=startworker,initargs=(a.scratch,a.lib,manifest,sig)) as pool:
        it=iter(pending); active={pool.submit(shard,j) for j in [next(it,None) for _ in range(a.workers)] if j is not None}
        while active:
            done,active=wait(active,return_when=FIRST_COMPLETED)
            for f in done:
                rec=f.result();hist[(rec['beta_index'],rec['lo'])]=rec
                if first[0] is None:
                    first[0]=rec
                    print('FIRST_SHARD '+json.dumps(rec),flush=True)
                j=next(it,None)
                if j is not None: active.add(pool.submit(shard,j))
            if first_mix_pending and all(any(k[0]==b for k in hist) for b in range(10)):
                first_mix_pending=False;update();print('FIRST_MIX measured across all ten temperatures',flush=True)
            elif last==0 or time.monotonic()-last>=60:update()
    update()
    dump(a.out/'shard-digests.json',dict(signature=sig,shard_width=WIDTH,bin_edges=EDGES.tolist(),group_order=['FIT','DEV','TEST-assess','TEST-reserve tallies only'],moments=MOMENTS,shards=[hist[k] for k in sorted(hist)]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('scratch',type=Path);p.add_argument('lib',type=Path);p.add_argument('qualification',type=Path);p.add_argument('--qualification-key',default='qualified');p.add_argument('--out',type=Path,default=RESULT);p.add_argument('--workers',type=int,default=6)
    main(p.parse_args())
