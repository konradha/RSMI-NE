"""One-shot V3 evaluation. Every learned object is fixed BEFORE the first TEST member opens."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
from hero_bank import DATA, INDEX_SHA, ROOT, committed, exact_keys, readj, require, sha
from hero_forecast import exact_inventory_null
import full_hero_forecast as fc
from full_hero_stream import anchors, cutoff, lookup, savej
from full_hero_v3_stream import PUB,WORK,INDICES,digest
from full_hero_v3_physics import weights


def project_test(path,b,rows,tables):
    """One forward pass per beta; reads only explicitly assigned native TEST parents."""
    n=len(rows); vt,et,it=tables
    V=np.empty((n,8,27),'u1'); E=np.empty((n,8,386),'u1'); I=np.empty((n,8,2),'<u2')
    with ZipFile(path) as z,z.open('species.npy') as f:
        ver=np.lib.format.read_magic(f); require(ver in ((1,0),(2,0)),'source version')
        shape,order,dtype=(np.lib.format.read_array_header_1_0 if ver==(1,0) else np.lib.format.read_array_header_2_0)(f)
        require(shape==(16384,20,20,20) and not order and dtype==np.uint8,'source geometry'); base=f.tell()
        for j,p in enumerate(np.argsort(rows)):
            if j%64==0: cutoff('hidden')
            r=int(rows[p]); addr=base+r*8000; require(f.tell()<=addr,'repeated or backward test row')
            f.seek(addr); buf=f.read(8000); require(len(buf)==8000 and buf.count(b'\x00')==2000 and buf.count(b'\x01')==2400 and buf.count(b'\x02')==3600,'source composition')
            x=np.frombuffer(buf,'u1'); a=anchors(b,r,False); V[p]=x[vt[a]]; E[p]=x[et[a]]; interior=x[it[a]]
            I[p,:,0]=(interior==0).sum(1); I[p,:,1]=(interior==1).sum(1)
    return V,E,I


def evaluate(rev,checkpoints):
    from full_hero_v3_score import load_scoring_models,score_one_beta,summarize_ten_beta
    cutoff('hidden'); receipt=PUB/'test-release.json'; release=readj(receipt)
    require(release['stage']=='one-final-all-ten' and len(rev)==40 and release['max_test_passes']==1,'invalid single use authorization')
    for name,digest_wanted in release['pins'].items():
        p=ROOT/name; require(not Path(name).is_absolute() and '..' not in Path(name).parts and sha(p)==digest_wanted,'frozen artifact changed: '+name); committed(p,rev)
    committed(receipt,rev)
    req=['design.json','split.json','nomination.json','refit-release.json','refit.json','physical-rule.json','physical-development.json','test-freeze.json','assessment-design.json']
    for name in req:
        require(str((PUB/name).relative_to(ROOT)) in release['pins'],'unpinned dependency '+name)
    freeze=readj(PUB/'test-freeze.json'); split=readj(PUB/'split.json'); ref=readj(PUB/'refit.json')
    require(freeze['schema']=='hero-full-v3-pretest-freeze-1' and freeze['refit_sha256']==sha(PUB/'refit.json') and
            freeze['split_sha256']==sha(PUB/'split.json')==ref['split_sha256'] and ref['status']=='complete' and
            ref['checkpoints']['6101']['sha256']==freeze['model_sha256'] and freeze['test_opened'] is False and
            freeze['rule_sha256']==sha(PUB/'physical-rule.json') and freeze['physical_development_sha256']==sha(PUB/'physical-development.json') and
            split['train']==split['test']==81920 and split['cross_side_intersections']==0 and split['rosters_sha256']==sha(WORK/'rosters.npz'),'archive, freeze or refit mismatch')
    require(sha(DATA/'release-index.json')==INDEX_SHA==split['source_manifest_sha256'],'source changed')
    models=load_scoring_models(ref,checkpoints); critic=weights(checkpoints,ref['checkpoints']['6101']); spec=readj(PUB/'physical-rule.json')
    can_forecast=freeze['physical_test_enabled']; params={}; inv=None
    if can_forecast:
        p=PUB/'physical-frozen-coefficients.npz'
        require(str(p.relative_to(ROOT)) in release['pins'] and sha(p)==freeze['coefficients_sha256'],'unfrozen fit')
        with np.load(p,allow_pickle=False) as z:
            params={m:z[m] for m in fc.METHODS}; inv=z['inventory_support']
    manifest=readj(DATA/'release-index.json')['endpoints']; sets={}; ids=set()
    with np.load(WORK/'rosters.npz',allow_pickle=False) as z:
        for item in manifest:
            b=item['release_index']; r=z[f'b{b:02d}_test']; rr=z[f'b{b:02d}_train']; k=exact_keys(item)
            require(digest(r)==split['ordered_native_int32le_roster_sha256'][f'b{b:02d}_test'] and
                    np.array_equal(np.sort(np.r_[r,rr]),np.arange(16384)) and len(r)==8192 and
                    digest(k[r])==split['per_beta'][str(b)]['test_identity_sha256'],'assessment identity mismatch')
            ids.update(map(tuple,k[r].tolist()));sets[b]=(r,DATA/item['npz'])
    require(len(ids)==81920 and sorted(sets)==list(INDICES),'incomplete or duplicated test key')
    require(not (PUB/'assessment.json').exists(),'test already evaluated'); tables=lookup()
    # Exclusive directory before ANY TEST species read. Crash is spent, never an automated rerun.
    lock=WORK/'test-once';lock.mkdir(exist_ok=False);savej(lock/'ticket.json',dict(sha256=sha(receipt),revision=rev))
    before=time.monotonic(); panels={}; phys={}; support={}; null=None
    for b in INDICES:
        rows,path=sets[b]; V,E,I=project_test(path,b,rows,tables)
        panels[b]=score_one_beta(models,b,V,E,rows)
        if can_forecast:
            truth=[]; collected={}
            for s in range(0,len(rows),64):
                x=fc.design(I[s:s+64],E[s:s+64],critic); pred=fc.predict(x,params)
                for name,value in pred.items(): collected.setdefault(name,[]).append(value)
                truth.append(fc.targets(V[s:s+64],spec))
            pred={k:np.concatenate(lst) for k,lst in collected.items()}; y=np.concatenate(truth)
            support[b]=float((inv[fc.inventory_codes(I)]>=8).mean()); phys[b]=fc.losses(y,pred,freeze['normalization'],9)
            if b==0: null=exact_inventory_null(I,y[:,0].reshape(-1,8),pred['m1']-pred['m0'],freeze['normalization'])
        savej(lock/f'b{b:02d}.json',dict(retrieval=panels[b],physical=phys.get(b),parent_count=len(rows)))
    macro=fc.panel_summary(phys,range(1,10)) if can_forecast else None
    physical=None if not can_forecast else dict(raw=fc.passes(macro['f']['g'],.01),
                 residual=spec['residual_fraction']>=.10 and fc.passes(macro['residual']['g'],0),
                 null=bool(null['passed']),support=bool(np.mean(list(support.values()))>=.90))
    proposal=dict(source='one V3 TEST evaluation; held out from v3, not a virgin archive',split_sha256=freeze['split_sha256'],
                  test_release_sha256=sha(receipt),refit_sha256=freeze['refit_sha256'],n_test=81920,n_per_beta=8192,
                  geometry='eight non-overlapping R4-cube opportunities per native TEST parent, fixed anchor grid and random shift',
                  indices=list(INDICES),hard_retrieval=panels,retrieval_summary=summarize_ten_beta(panels),
                  physical_test_enabled=can_forecast,physical_gates=physical,phys_by_beta=phys,physical_macro=macro,
                  passed_physical_mechanism=all(physical.values()) if can_forecast else None,beta0_exact_inventory=null,
                  first_failed_training_gate=None if can_forecast else readj(PUB/'physical-development.json'),
                  seconds=time.monotonic()-before,paid_gpu_seconds_for_this_final_score=0)
    cutoff('hidden');savej(PUB/'assessment.json',proposal)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--release-commit',required=True);ap.add_argument('--weights',required=True,type=Path)
    args=ap.parse_args();evaluate(args.release_commit,args.weights)
