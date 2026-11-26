"""One fixed frozen operator bank, original parent IDs, and one local label shuffle per parent."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone, timedelta
from itertools import zip_longest
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from field_distance_inputs import COHORTS, DISTS, OUT, make_bank, sha, save_json, need
from field_distance_pairs import prepare, score

BLOCK=32
SHARED={}

def work(job):
    tag,lo=job;s=SHARED;N=len(s['bank'][tag]);n=min(BLOCK,N-lo);nm=len(s['names']);d=len(DISTS);dest=s['scratch']/f'{tag}-{lo:05d}.npz';start=time.monotonic()
    corr=np.empty((n,2,nm,d,2),dtype=np.float64);counts=np.empty((n,d,3),np.int16);eligible=np.empty(n,np.uint16);ids=s['ids'][tag][lo:lo+n];b=COHORTS[tag][0]
    for j,(seed,cid) in enumerate(ids):
        grid=s['bank'][tag][lo+j];rng=np.random.default_rng([734982,b,int(seed),int(cid)])
        c,v,u,meta=score(grid,s['prepared'],DISTS,rng)
        need(np.array_equal(np.asarray(v).shape,(nm,d,2)) and (c>=0).all() and (c<=8000).all(),'pair result layout')
        ok=(c>=2).all(axis=1);need(np.isfinite(v[:,ok]).all() and np.isfinite(u[:,ok]).all() and np.isnan(v[:,~ok]).all() and np.isnan(u[:,~ok]).all(),'incorrect missing outcomes')
        counts[j]=c; corr[j,0]=v;corr[j,1]=u;eligible[j]=meta['n_eligible']
    tmp=dest.with_suffix('.partial');
    with tmp.open('wb') as f:np.savez(f,counts=counts,corr=corr,eligible=eligible,ids=ids)
    tmp.replace(dest);return dict(cohort=tag,lo=lo,parents=n,seconds=time.monotonic()-start,sha256=sha(dest))

def main(a):
    ad=json.loads((OUT/'admission.json').read_text());need(ad['held_spatial_scoring_permitted'] is True,'held access not admitted');out=a.scratch
    for name,digest in ad['pins'].items():need(sha(OUT/name)==digest,'changed frozen file: '+name)
    t=time.monotonic();cohorts=list(COHORTS)
    for tag in cohorts:
        if tag!='source-train':make_bank(tag,a.runs,a.staged,out)
    with np.load(OUT/'fields.npz',allow_pickle=False) as z:names=z['names'].astype(str);fields=z['fields']
    need(len(names)==len(set(names))==59 and fields.shape[:3]==(59,210,3),'wrong model bank')
    frozen=prepare(fields);banks={};ids={}
    for tag in cohorts:
        p=out/(tag+'.npy');r=json.loads((OUT/(tag+'-input.json')).read_text());need(sha(p)==r['star_bank_sha256'] and sha(out/(tag+'-parents.npz'))==r['metadata_sha256'],'row bank digest')
        banks[tag]=np.load(p,mmap_mode='r');
        with np.load(out/(tag+'-parents.npz')) as z:ids[tag]=z['ids']
    flat=np.concatenate(list(ids.values()));need(len(np.unique(flat,axis=0))==len(flat),'cohort lineage collision')
    SHARED.update(prepared=frozen,names=names,scratch=out,bank=banks,ids=ids)
    runs={tag:[] for tag in cohorts};t0=time.monotonic();complete=0;first=True;printed=False;total=len(flat);last=0
    order=['transfer-320','source-test','beta-zero','source-train','transfer-340'];lists=[[(tag,j) for j in range(0,len(ids[tag]),BLOCK)] for tag in order]; jobs=[x for row in zip_longest(*lists) for x in row if x]
    def update():
        nonlocal last,printed
        elapsed=time.monotonic()-t0;per={tag:sum(x['parents'] for x in runs[tag]) for tag in cohorts};bysec={tag:sum(x['seconds'] for x in runs[tag]) for tag in cohorts};par=sum(bysec.values())/elapsed;eta=None
        if min(per.values())>0 and par>0:
            left=sum((len(ids[tag])-per[tag])*bysec[tag]/per[tag] for tag in cohorts)/par;eta=(datetime.now(timezone.utc)+timedelta(seconds=left)).isoformat()
        save_json(OUT/'status.json',dict(phase='scored' if complete==total else 'scoring',source_train_normalization_parents=8192,spatial_parents_completed=complete,target_spatial_parents=total,by_cohort=per,throughput_parents_per_second=complete/elapsed,elapsed_seconds=elapsed,stage_and_prepare_seconds=t0-t,eta_utc=eta,workers=a.workers,gpu_seconds=0,new_trajectories=0))
        if not printed and min(per.values())>0:print(f'MIXED_RATE {complete}/{total}, {complete/elapsed:.2f} parents/s, mix-adjusted ETA {eta}',flush=True);printed=True
        print(f'SCORED_PARENTS {complete}/{total}',flush=True);last=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        it=iter(jobs);active={pool.submit(work,next(it)) for _ in range(min(a.workers,len(jobs)))}
        while active:
            done,active=wait(active,return_when=FIRST_COMPLETED)
            for f in done:
                r=f.result();runs[r['cohort']].append(r);complete+=r['parents'];j=next(it,None)
                if j:active.add(pool.submit(work,j))
            if first or (not printed and all(runs.values())) or time.monotonic()-last>=60:update();first=False
    update();save_json(OUT/'score-receipt.json',dict(complete=complete==total,parents=total,distances=DISTS,rows={tag:sorted(v,key=lambda x:x['lo']) for tag,v in runs.items()},field_sha256=sha(OUT/'fields.npz'),admission_sha256=sha(OUT/'admission.json'),input_receipts={tag:sha(OUT/(tag+'-input.json')) for tag in cohorts},cost='CPU only; no GPU or simulation'))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('runs',type=Path);ap.add_argument('staged',type=Path);ap.add_argument('scratch',type=Path);ap.add_argument('--workers',type=int,default=7);main(ap.parse_args())
