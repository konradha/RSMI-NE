"""Metadata-only gate for one distant-shell → frozen #04 experiment. Never opens species."""
import hashlib,json
from pathlib import Path
import numpy as np
from rate import partition,digest,json_out
from boundary_bank import PINS

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'results/l20-autoresearch/cycle-boundary'
PRIVATE=Path.home()/'rsmi-work/l20-autoresearch-1795520725/cycle-boundary'
RELEASE=Path.home()/'datasets/lattice-glass/l20-16384-v1'


def run():
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    path=PRIVATE/'rosters.npz'
    if path.exists():raise FileExistsError(path)
    manifest=json.loads((RELEASE/'release-index.json').read_bytes());entries={v['release_index']:v for v in manifest['endpoints']};folds={i:partition(i)[0] for i in (0,7,8,9)}
    # Retain uint64 throughout: converting a column of 64-bit seeds to float is lossy.
    keys={};prior_cycle=set();prior_audit=set();input_pins={}
    for i in (0,7,8,9):
        f=RELEASE/f'beta-{i:02}.npz';entry=entries[i];sha=digest(f)
        if sha!=entry['npz_sha256'] or (i in PINS and sha!=PINS[i][1]):raise ValueError('archive pin mismatch')
        with np.load(f) as z:
            seed=z['chain_seed'].astype('u8',copy=False);ids=z['chain_id'].astype('u8',copy=False);keys[i]=list(zip(map(int,seed),map(int,ids)))
        if len(set(keys[i]))!=16384:raise ValueError('source parents not unique')
        old04=np.r_[folds[i][0][:256],folds[i][1][:512]] if i==7 else folds[i][0][:256]
        prior_cycle.update(keys[i][int(row)] for row in old04)
        prior_audit.update(keys[i][int(row)] for row in np.r_[folds[i][1],folds[i][0][:5120]])
        input_pins[i]=sha
    arrays={};stats={};assigned=set();layout={9:[('pilot',256),('fit',3072),('dev',512),('cold',1024)],0:[('hot',1024)],8:[('transfer',1024)]}
    blocked=prior_cycle|prior_audit
    for i,scheme in layout.items():
        bulk=int(entries[i]['bulk_seed_uint64']);train=folds[i][0];pool=[(k,int(r)) for k,r in enumerate(train) if k>=5120 and keys[i][int(r)][0]==bulk and keys[i][int(r)] not in blocked and keys[i][int(r)] not in assigned];at=0
        for role,n in scheme:
            sub=pool[at:at+n];at+=n;rows=np.array([r for k,r in sub],dtype=np.int64);pair=[keys[i][r] for r in rows];uni=set(pair)
            if len(uni)!=n or assigned&uni or (prior_cycle|prior_audit)&uni or np.intersect1d(rows,np.r_[folds[i][1],folds[i][2],train[:5120]]).size:raise ValueError(f'forbidden or undersized {role}')
            assigned.update(uni);ss,cc=np.array(pair,dtype='<u8').T;arrays.update({role+'_parent':rows,role+'_seed':ss,role+'_id':cc,role+'_index':np.array(i)})
            stats[role]=dict(index=i,beta=float(entries[i]['beta']),n=n,train_position_inclusive=[sub[0][0],sub[-1][0]],bulk_seed=str(bulk),parent_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),key_sha256=hashlib.sha256(np.array(pair,dtype='<u8').tobytes()).hexdigest())
    np.savez(path,**arrays); report=dict(scope='Metadata and public receipts only; no patch, outcome or shell extracted.',cohorts=stats,parents=len(assigned),distinct_lineages=len(assigned),prior_exact_endpoint_lineages=len(prior_cycle),previous_outcome_overlap=len(assigned&prior_cycle),excluded_spent_or_native_validation_count=len(prior_audit),spent_overlap=len(assigned&prior_audit),index6_access=False,test_species_access=False,original_release_sha256=input_pins,private_roster_sha256=digest(path),exact_endpoint_provenance=['fifteen_topology.py: occupancy H1 mod 2 for open complex; marginal development only at indices 0,8,9','fifteen_secondary.py: marginal pair maxent test at index 7','boundary/design.json: same numerical R, different central target and conditioning; previously negative tripod transfer not reopened','tranche2/LEDGER.md: temperatures/parent folds previously used for other outcomes, no prior held-out cycle-shell contrast located in project code/designs'],incremental_exposure='Subject of #04 selected with historical prevalence information. No new measured parent previously supplied a #04 outcome. Many source configurations and the radius-four feature family have contributed to historical studies of other observables. Not an independent replication of selection.',fee_usd=0);json_out(PUBLIC/'lineage.json',report);print(json.dumps(stats),flush=True)

if __name__=='__main__':run()
