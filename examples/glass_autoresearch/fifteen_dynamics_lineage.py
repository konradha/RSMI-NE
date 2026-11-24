"""Items 14–15 lineage gate. Reads public JSON and identity-only NPZ members, never frames."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l20_pairs import partition

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/l20-autoresearch/fifteen'
LOCK = '96a1e5289a04a5ee5354f22a2d094cd42d922c7c0c77fb48daa830e4345768f8'
LOCK_COMMIT = '811a60e2135d782b7cd7deb9d1cacc298dec3c7e'
LAGS = (0, 16, 256, 4096)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def readj(path):
    return json.loads(path.read_bytes())


def store(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def audit(data, runs, out):
    if sha(out/'physical-lock.json') != LOCK:
        raise ValueError('not the public calibration lock')
    propensity = ROOT/'results/l20-propensity'; directional = ROOT/'results/l20-operators/acquisition'
    paths = [propensity/'archive-provenance.json', propensity/'cold-provenance.json', directional/'archive.json', directional/'forks-08.json']
    hist, cold, directions, warm = map(readj, paths)
    rows = []; allowed = []; index9keys = None
    for c in hist['cohorts']:
        i = c['index']; r = c['replicas']; available = set(c['available_lags'])
        common = dict(cohort=c['cohort'],index=i,beta=c['beta'],parents=c['parents'],replicas=r,available_required_lags=sorted(available.intersection(LAGS)),ordered_distinct_child_pairs_per_parent=r*(r-1))
        if i == 6:
            rows.append(dict(**common,eligible_train_parents=0,reason='global index embargo; no identity or frame access'));continue
        native = data/f'beta-{i:02}.npz'; receipt = readj(native.with_suffix('.json'))
        metadata = {k:receipt[k] for k in ['endpoint_sha256','beta','candidate_age','independent_endpoints']}
        assert any(str(c['parent_checkpoint']).endswith(src['checkpoint']) for src in receipt['sources']) and c['candidate_age']==receipt['candidate_age']
        with np.load(native,allow_pickle=False) as f:
            seeds, ids = f['chain_seed'],f['chain_id']  # deliberately never access 'species'
        source = {(int(seed),int(pid)):j for j,(seed,pid) in enumerate(zip(seeds,ids))}
        assert len(source)==receipt['independent_endpoints']
        fold = np.empty(len(ids),np.int8); groups = partition(i)[0]
        for group,indices in enumerate(groups):fold[indices]=group
        matched = [source[int(c['preparation_seed']),int(p['parent_id'])] for p in c['lineage']]
        counts = np.bincount(fold[matched],minlength=3); train = int(counts[0]); eligible14 = train if r>=2 and 4096 in available else 0
        eligible15 = train if r>=2 and set(LAGS)<=available else 0
        rows.append(dict(**common,native=metadata,eligible_train_parents=train,validation_excluded=int(counts[1]),test_excluded=int(counts[2]),
                         support_14_same_clock=eligible14,support_15_exact_windows=eligible15,
                         analysis='locked secondary only' if i==7 else 'inventory only; no label access'))
        if i==7:
            for j,p in zip(matched,c['lineage']):
                if fold[j]==0:
                    allowed.append(dict(release_row=int(j), prep_seed=str(c['preparation_seed']),**p))
        if i==9:index9keys = (seeds,ids,groups,fold)
    allowed.sort(key=lambda p:int(p['parent_id'])); assert len(allowed)==21
    for j,p in enumerate(allowed):p['role']='fit' if j<11 else 'calibration'
    seeds,ids,groups,fold=index9keys; coldrows=np.array(cold['release_rows']); assert len(np.unique(coldrows))==64 and np.all(fold[coldrows]==2)
    assert [[str(int(seeds[i])),str(int(ids[i]))] for i in coldrows]==cold['source_identity']
    values,n=np.unique(seeds,return_counts=True);pool=groups[2][seeds[groups[2]]==values[np.argmax(n)]];pool=np.setdiff1d(pool,coldrows)
    d9=np.sort(np.random.default_rng(7621009).choice(pool,512,replace=False));assert np.all(fold[d9]==2) and not np.intersect1d(d9,coldrows).size
    d8=np.array(warm['release_rows']);assert len(np.unique(d8))==256 and np.isin(d8,partition(8)[0][2]).all()
    cohorts=directions['cohorts']; assert cohorts['09']['complete_sweeps']==[4096] and cohorts['08']['complete_sweeps']==[116,462,4096]
    assert cold['replicas']==16 and cohorts['09']['replicas']==cohorts['08']['replicas']==8
    summary=[]
    for name,idx,rr,rep,lags in [('cold propensity forks',9,coldrows,16,cold['lags']),('cold directional',9,d9,8,cohorts['09']['complete_sweeps']),('secondary directional',8,d8,8,cohorts['08']['complete_sweeps'])]:
        n=len(rr); s=set(lags)|{0};summary.append(dict(cohort=name,index=idx,parents=n,replicas=rep,completed_sweeps=lags,all_native_TEST=True,eligible_for_fit_or_scoring=0,
                 raw_support_14_parents=n if 4096 in s else 0,raw_support_15_parents=n if set(LAGS)<=s else 0,
                 raw_14_disjoint_sibling_pairs=n*(rep//2) if 4096 in s else 0,raw_15_disjoint_sibling_pairs=n*(rep//2) if set(LAGS)<=s else 0,
                 release_row_sha256=hashlib.sha256(np.array(rr,'<i8').tobytes()).hexdigest()))
    primary=hist['cohorts'][7];study=Path(primary['study']);plan=readj(runs/study/'physical/plan.json');head=readj(runs/study/'physical/batch-00000000/head.json');assert plan['move']=='local' and plan['replicas_per_beta'][7]==2
    selected={}
    for s in LAGS:
        fname,= [v for v in head['snapshots'] if v.startswith(f'{s:020d}-')]; selected[str(s)]=str(study/'physical/batch-00000000'/fname)
    assert len({x for p in allowed for x in p['child_stream_ids']})==42
    out_data=dict(schema='fifteen-siblings-lineage-v1',published_lock_commit=LOCK_COMMIT,lock_sha256=LOCK,input_metadata_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}, native_partition='(preparation_seed,parent_id) exact join to native (chain_seed,chain_id), candidate age and probe checkpoint verified; only native TRAIN eligible. All original index7 TRAIN identity matches included.',original_archive=rows,issued_forks=summary,source_for_cold_directional='examples/l20_operators_forks.py:29-40: replay the fixed without-replacement sample from production TEST excluding every old cold development parent. No test origin/trajectory arrays opened.', additional_physical_plan=dict(study=str(study),first_temperature_08_parents=int(plan['parent_counts'][8]),first_temperature_08_replicas=int(plan['replicas_per_beta'][8]),older_age_same_parent_ids='Older age origins are not extra independent parents; not included in this candidate-age lock.'),
        eligible_child_streams=allowed,frame_paths=selected,scope='Named cold primary and warm secondary: zero eligible replicated parents or sibling pairs, no finite power for either requested estimand. No cold frames or values accessed. Dynamic 32/32 split of the old 64 cannot override the static embargo. Verification uses metadata only, before opening any pilot frame.',
        partial_alternative='index7 at 4096 absolute sweeps, not matched t/tau; 21 TRAIN candidate parents with 2 true children, 11 fit/10 calibration. No controls at other beta or age authorized for scoring. Other temperature counts inventory only, not replications of the target beta.')
    store(out/'14-15-lineage.json',out_data)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=Path);p.add_argument('runs',type=Path);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();audit(a.release,a.runs,a.out)
