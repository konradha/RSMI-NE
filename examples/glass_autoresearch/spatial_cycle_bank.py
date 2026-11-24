"""Archive-only physical-channel addendum. 'lineage' never reads an outcome.

Physical extraction is fail-closed: no exam paths. The word learner receives only
unlabeled banks. Private recorded roster order is the anchor/OOF join contract.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

from cycle_bank import (CENTRE, DATA, N_ANCHOR, PINS, ROLES as OLD_ROLES, SHELL,
                        WEIGHTS, catalogue, geometry, pcg, require, table, write_np)
import numpy as np
from rate import digest, partition
from vacancy_dipole import selected_member

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'results/l20-autoresearch/spatial-cycle'
OLD = PUBLIC.with_name('cycle-boundary')
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/spatial-cycle'
OLD_WORK = WORK.with_name('cycle-boundary')
ROLES = {'fit': (9, 2048), 'calibration': (9, 384), 'dev': (9, 512),
         'cold': (9, 1024), 'hot': (0, 1024), 'transfer': (8, 1024)}


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


def readj(p):
    return json.loads(p.read_bytes())


def array_sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def pins():
    return {e['release_index']: e for e in readj(DATA/'release-index.json')['endpoints']}


def make_lineage():
    entries = pins(); local = readj(OLD/'lineage.json'); old = set()
    require(digest(OLD_WORK/'rosters.npz') == local['private_roster_sha256'], 'local roster pin')
    with np.load(OLD_WORK/'rosters.npz', allow_pickle=False) as z:
        for role in OLD_ROLES:
            key = np.column_stack((z[role+'_seed'], z[role+'_id'])).astype('<u8')
            require(array_sha(key) == local['cohorts'][role]['key_sha256'], 'local key pin')
            old.update((int(seed), int(cid)) for seed, cid in key)
    require(len(old) == 6912, 'wrong embargo')
    bulk = [int(e['bulk_seed_uint64']) for e in entries.values()]
    shared_diag = {int(e['physical_diagnostic_seed_uint64']) for e in entries.values()}
    # Diagnostic trajectories can name their originating endpoint seed: no such trajectory is an input.
    require(len(set(bulk)) == 10 and not set(bulk) & {int(e['prep_seed_uint64']) for e in entries.values()}, 'nonunique bulk lineage namespace')
    prior04 = set(); barred = set(); pools = {}; trainpos = {}; keys = {}; tuning = set()
    for i in (0,7,8,9):
        pin = entries[i]; path = DATA/pin['npz']; require(digest(path) == pin['npz_sha256'] == local['original_release_sha256'][str(i)], 'release hash mismatch')
        train,val,test = partition(i)[0]; trainpos[i] = {int(r):k for k,r in enumerate(train)}
        with np.load(path, allow_pickle=False) as z:
            key = list(zip(map(int,z['chain_seed']), map(int,z['chain_id'])))
        require(len(key) == len(set(key)) == 16384, 'duplicate source parent'); keys[i] = key
        old04 = np.r_[train[:256],val[:512]] if i == 7 else train[:256]
        prior04.update(key[r] for r in old04)
        later = train[4096:5120] if i == 9 else train[256:768] if i in (0,8) else []
        barred.update(key[r] for r in (*val,*test,*later))
        if i == 9:
            tuning.update(key[r] for r in np.random.default_rng(210491).permutation(train)[:len(train)//5])
    denied = old | prior04 | barred; assigned = set(); arrays = {}; cohorts = {}
    rng = np.random.default_rng(260518)
    for i in (9,0,8):
        seed = int(entries[i]['bulk_seed_uint64']); key = keys[i]
        t = partition(i)[0][0]
        rows = [int(r) for r in t if key[r][0] == seed and key[r] not in denied]
        rows = list(map(int,rng.permutation(rows))); pools[i] = len(rows)
        # Prefer the fresh unused tail for exams; exclude previous tripod tuning even on a different endpoint.
        if i == 9:
            order = sorted((r for r in rows if key[r] not in tuning), key=lambda r: trainpos[i][r] < 5120)
            exam, dev = order[:1024], order[1024:1536]
            used = set(exam + dev); other = [r for r in rows if r not in used]
            groups = [('cold',exam),('dev',dev),('fit',other[:2048]),('calibration',other[2048:2432])]
        else:
            groups = [('hot' if i==0 else 'transfer',[r for r in rows if trainpos[i][r]>=5120][:1024])]
        for role, selected in groups:
            r = np.array(selected, dtype='<i8'); k = np.array([key[x] for x in r], dtype='<u8'); unique = {key[x] for x in r}
            require(len(r) == len(unique) == ROLES[role][1] and not unique & (denied|assigned), 'incomplete or reused '+role)
            if role in ('cold','dev','transfer','hot'): require(not unique&tuning, 'reused old tuning in '+role)
            assigned.update(unique); arrays.update({role+'_parent':r,role+'_seed':k[:,0],role+'_id':k[:,1]})
            pos = np.array([trainpos[i][x] for x in r]); cohorts[role] = dict(index=i,beta=entries[i]['beta'],parents=len(r),bulk_seed=str(seed),
                prior_tranche2_fit=int(np.sum(pos < (4096 if i==9 else 256))),prior_tranche2_dev_or_audit=0,
                prior_tripod_fit_only=int(len(unique-tuning)) if i==9 else 0,prior_tripod_internal_tuning=len(unique&tuning),
                prior_cycle_endpoint=0,prior_locked_cycle_any_role=0,ids_sha256=array_sha(k),rows_sha256=array_sha(r))
    require(len(prior04)==1536 and len(assigned)==sum(n for _,n in ROLES.values()),'ancestry accounting')
    mask = np.load(OLD_WORK/'eligible.npy',allow_pickle=False)
    oldpilot=readj(OLD/'pilot.json'); require(len(mask)==22 and digest(OLD_WORK/'eligible.npy')==oldpilot['provenance']['eligible_sha256'],'support freeze pin')
    write_np(WORK/'rosters.npz',np.savez,**arrays); write_np(WORK/'eligible.npy',np.save,mask)
    save(PUBLIC/'lineage.json', dict(scope='Metadata only; no native species, word, label, shell or trajectory opened.',
        reused_training='All roles are native TRAIN and contributed to historical work on other observables. Not a globally new population. No role includes a prior exact #04 parent, local cycle trial parent, or a native validation/test or prior tranche2 development/audit lineage. Cold and new dev also exclude the earlier R=4 tripod internal tuning parents. Unrelated old TRAIN fitting counts are disclosed per role. Separate probe calibration is part of FIT and withheld from every encoder.',
        independent_parents=len(assigned),cohorts=cohorts,remaining_candidate_counts_before_assignment=pools,
        previous_local_embargo_size=len(old),prior_exact04_size=len(prior04),exact04_overlap=len(assigned&prior04),
        embargo_overlap=len(assigned&old),spent_exam_and_native_validation_test_overlap=len(assigned&barred),
        source_pins={i:local['original_release_sha256'][str(i)] for i in (0,7,8,9)},
        all_ten_release_bulk_seeds_distinct=True,bulk_prep_seed_overlap=False,bulk_diagnostic_seed_overlap_count=len(set(bulk)&shared_diag),
        unpublished_private_rosters_sha256=digest(WORK/'rosters.npz'),eligible_sha256=digest(WORK/'eligible.npy'),
        projected_80pct_MDE_nats_1024_perturbed_logit_from_old_center_only_pilot=.0007700,
        precision_limit='Not an empirical MDE for an actual raw or frozen-code comparison; report paired whole-parent resolution from independent dev for each fitted model.',
        exam_status='Cold/hot/adjacent roles reserved and CLOSED. Raw positive control failed; no confirmatory exam or size/temperature extrapolation. No recovery by reassignment.'))


def extract(role, release):
    require(role in ('fit','dev','calibration'), 'exam forbidden')
    lock = readj(PUBLIC/'lineage.json'); token = readj(PUBLIC/release)
    require(Path(release).name == release and token['stage']=='bank-extraction' and role in token['roles'] and
            token['lineage_sha256']==digest(PUBLIC/'lineage.json') and token['control_passed'] and
            token['control_sha256']==digest(PUBLIC/token['control_file']) and readj(PUBLIC/token['control_file'])['pass'], 'missing or invalid release')
    require(digest(WORK/'rosters.npz') == lock['unpublished_private_rosters_sha256'] and digest(WORK/'eligible.npy')==lock['eligible_sha256'], 'roster or support mismatch')
    if role=='calibration':
        require(token['hard_code_manifest_sha256']==digest(PUBLIC/token['hard_code_manifest']), 'hard encoder must be frozen before calibration')
    i,n = ROLES[role]; pin=pins()[i]; src=DATA/pin['npz']; ref=lock['cohorts'][role]
    require(digest(src)==PINS[i][1]==pin['npz_sha256'] and digest(DATA/pin['sidecar'])==pin['sidecar_sha256'], 'input hash')
    with np.load(WORK/'rosters.npz',allow_pickle=False) as r:
        parent,seed,cid=[r[role+'_'+k] for k in ('parent','seed','id')]
    require(len(parent)==n and array_sha(parent)==ref['rows_sha256'] and array_sha(np.column_stack((seed,cid)).astype('<u8'))==ref['ids_sha256'], 'wrong parents')
    with ZipFile(src) as z:
        require(np.array_equal(seed,selected_member(z,'chain_seed',parent,())) and np.array_equal(cid,selected_member(z,'chain_id',parent,())), 'archive identity mismatch')
        state=selected_member(z,'species',parent,(20,20,20)).reshape(n,8000)
    require(state.dtype==np.uint8 and all(np.all((state==v).sum(1)==k) for v,k in ((0,2000),(1,2400),(2,3600))), 'composition mismatch')
    geometry(); label,base,*_=catalogue(); patch=table(CENTRE); shell=table(SHELL)
    word=np.empty((n,N_ANCHOR),dtype=np.uint16); s=np.empty_like(word,dtype=np.int16); bits=np.empty((n,N_ANCHOR,49),np.uint8)
    for j,row in enumerate(parent):
        a=pcg(260519,i,row).choice(8000,N_ANCHOR,replace=False); c=state[j,patch[a]]
        w=(c!=0)@WEIGHTS; word[j]=w; s[j]=base[w]+64*(c==1).sum(1)
        v=state[j,shell[a]]==0; bits[j]=np.packbits(v,axis=1,bitorder='little')
        require(np.array_equal(np.unpackbits(bits[j],axis=1,bitorder='little',count=386),v), 'shell row order')
    require(not np.any(bits[:,:,-1]&252),'nonzero padding')
    bank=dict(parent=parent,stratum=s,shell=bits,word=word)
    if role != 'calibration': write_np(WORK/('unlabeled-'+role+'.npz'),np.savez,**bank)
    write_np(WORK/(role+'.npz'),np.savez,**bank,y=label[word]); receipt=dict(role=role,parents=n,anchors=n*N_ANCHOR,
        bank_sha256=digest(WORK/(role+'.npz')),release_sha256=digest(PUBLIC/release),raw_supervised_arm='not admitted')
    if role!='calibration':receipt['unlabeled_bank_sha256']=digest(WORK/('unlabeled-'+role+'.npz'))
    save(PUBLIC/('bank-'+role+'.json'),receipt)


if __name__=='__main__':
    os.umask(0o077); WORK.mkdir(mode=0o700,parents=True,exist_ok=True)
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('role',choices=['lineage','fit','dev','calibration']);parser.add_argument('--release');args=parser.parse_args()
    if args.role=='lineage':make_lineage()
    elif not args.release:parser.error('numbered public release required')
    else:extract(args.role,args.release)
