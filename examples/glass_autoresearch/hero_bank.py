"""HERO forward-only bank. `lineage` opens metadata only; `extract` requires a public release."""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples'))
from l20_pairs import partition

PUBLIC = ROOT / 'results/l20-autoresearch/hero-central'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/hero-central'
DATA = Path.home() / 'datasets/lattice-glass/l20-16384-v1'
INDEX_SHA = 'd4c591bab7a75e785fb749fa42cc78353dfc6d093c6742ae8d621ca2113040ff'
TRIALS = {'cycle-boundary': ('c17cf30ab80758cdda52de018fe3e5cc5fe2e666441b1f8ce759038ca80c8e13', 6912),
          'spatial-cycle': ('3c402d7e88ecd89e1d9581d2ca86e094c4a8130167fd5e871144319cf24220c7', 6016)}
INDICES = (0, 7, 8, 9)
ROLES = {'fit': (8, 4096), 'tune': (8, 512), 'rule': (8, 512), 'cold': (8, 1024),
         'issue': (8, 256), 'transfer': (7, 1024), 'hot': (0, 1024)}
STAGES = {'discovery': ('fit','tune','rule'), 'blind': ('cold','transfer','hot','issue'), 'hidden': ('cold','transfer','hot')}
ROW_SEED, ANCHOR_SEED = 26112401, 26112402
DISCOVERY_STOP = '2026-11-24T21:00:25+00:00'
HARD_STOP = '2026-11-24T21:45:25+00:00'


def cube(r):
    return np.array(list(product(range(-r, r + 1), repeat=3)), dtype=np.int16)


CENTER, INTERIOR = cube(1), cube(3)
C4 = cube(4)
SHELL = C4[np.max(abs(C4), axis=1) == 4]
SHELL5 = np.where(abs(SHELL) == 4, np.sign(SHELL) * 5, SHELL)
GRID = np.array(list(product((0, 10), repeat=3)), dtype=np.int16)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    with path.open('rb') as fd:
        return hashlib.file_digest(fd, 'sha256').hexdigest()


def array_sha(a, dtype):
    return hashlib.sha256(np.asarray(a, dtype=dtype).tobytes()).hexdigest()


def roster_sha(rows, anchors):
    h = hashlib.sha256(np.asarray(rows, '<i8').tobytes()); h.update(np.asarray(anchors, '<i8').tobytes())
    return h.hexdigest()


def readj(path):
    return json.loads(path.read_bytes())


def savej(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def savenp(path, arrays):
    with path.open('xb') as f:
        np.savez(f, **arrays)


def coords(flat):
    return np.stack(np.unravel_index(flat, (20,)*3), axis=-1).astype(np.int16)


def flatten(c):
    return (400*c[...,0] + 20*c[...,1] + c[...,2]).astype('<u2')


def sites(anchors, off):
    return flatten((coords(anchors)[..., None, :] + off) % 20)


def geometry():
    require((len(CENTER), len(INTERIOR), len(SHELL), len(SHELL5)) == (27,343,386,386), 'geometry sizes')
    out = {}; proper = 0
    for p in permutations(range(3)):
        for s in product((-1, 1), repeat=3):
            mat = np.eye(3, dtype=np.int8)[list(p)] * np.array(s)[:, None]
            proper += round(np.linalg.det(mat)) == 1
            for o in (CENTER, INTERIOR, SHELL, SHELL5):
                require(set(map(tuple, o @ mat)) == set(map(tuple, o)), 'cubic action')
    require(proper == 24, 'proper group'); own = np.r_[np.zeros((1,3), int), np.eye(3, dtype=int), -np.eye(3, dtype=int)]
    diam = int(abs(own[:,None] - own).sum(-1).max()); require(diam == 2, 'energy diameter')
    grid = flatten(GRID); word = sites(grid,CENTER)
    for off, radius in [(SHELL,4),(SHELL5,5)]:
        d = abs(CENTER[:,None] - off); dist = int(np.minimum(d, 20-d).sum(-1).min())
        require(dist == radius-1 and dist > diam, 'center-to-shell separation')
        boundary = sites(grid,off)
        require(not np.intersect1d(word, boundary).size and np.unique(word).size == 8*27,
                'other center in boundary or overlapping hidden cores')
        out[str(radius)] = dict(sites=len(off), offsets_i1_sha256=array_sha(off,'i1'), min_toroidal_L1=dist)
    require(np.array_equal(flatten(coords(np.arange(8000))), np.arange(8000)), 'C-order flat sites')
    # Roundtrip byte/channel convention, including a periodic wrap near (0,0,0).
    a = np.array([0, 7999, 341], dtype=np.uint16)
    for o in (CENTER,SHELL,SHELL5,INTERIOR):
        xyz = (coords(a)[:,None] + o) % 20; ref = np.ravel_multi_index(xyz.transpose(2,0,1), (20,)*3)
        require(np.array_equal(sites(a,o),ref), 'z,y,x strides')
        code = (ref % 3).astype(np.uint8); require(np.frombuffer(code.tobytes('C'),np.uint8).reshape(code.shape).tolist() == code.tolist(), 'unpacked ternary serialization')
    edges = int(np.triu(abs(CENTER[:,None] - CENTER).sum(axis=2) == 1,1).sum()); require(edges==54,'internal edge count')
    return dict(shells=out, center_i1_sha256=array_sha(CENTER,'i1'), interior_i1_sha256=array_sha(INTERIOR,'i1'),
                center_sites=27, interior_sites=343, internal_NN_edges=edges, energy_term_L1_diameter=diam,
                proper_group_size=int(proper), improper_group_size=48-int(proper), other_core_shell_intersections=0,
                species='C order z,y,x; uint8 per site; 0=vacancy,1=A,2=B; no bit or base-3 packing, no rotation applied')


def anchors(role, idx, rows):
    def draw(r):
        rng = np.random.default_rng([ANCHOR_SEED, idx, int(r)])
        if role in STAGES['discovery']:
            return rng.choice(8000, 64, replace=False)
        if role == 'issue':
            return rng.integers(8000, size=1)
        return flatten(GRID + rng.integers(10, size=3))
    return np.array([draw(r) for r in rows], dtype='<u2')


def pins():
    manifest = DATA / 'release-index.json'; require(sha(manifest) == INDEX_SHA, 'release-index pin')
    entries = {p['release_index']:p for p in readj(manifest)['endpoints']}
    bulk = {int(p['bulk_seed_uint64']) for p in entries.values()}; prep = {int(p['prep_seed_uint64']) for p in entries.values()}
    require(len(bulk)==10 and not bulk & prep, 'bulk namespaces not disjoint')
    return {i:entries[i] for i in INDICES}


def verify_source(p):
    path = DATA/p['npz']; side = DATA/p['sidecar']
    require(sha(path)==p['npz_sha256'] and sha(side)==p['sidecar_sha256'] and readj(side)['endpoint_sha256']==p['npz_sha256'], 'source bytes changed')
    return path


def exact_keys(p):
    with np.load(verify_source(p), allow_pickle=False) as z:
        require(str(z['axes'])=='chain,z,y,x' and float(z['beta'])==float(p['beta']), 'source coordinates or temperature')
        seed, cid = z['chain_seed'], z['chain_id']
        require(seed.dtype==cid.dtype==np.uint64 and seed.shape==cid.shape==(16384,), 'source uint64 identities')
        keys = np.column_stack((seed,cid)).astype('<u8',copy=False)
    require(len(np.unique(keys,axis=0))==16384 and set(map(int,np.unique(seed)))=={int(p['prep_seed_uint64']), int(p['bulk_seed_uint64'])}, 'release namespaces')
    return keys


def as_set(k):
    return {(int(s),int(c)) for s,c in k}


def old_trials(keys):
    groups = {}; by_role = {}; receipts = {}
    for study,(pin,n) in TRIALS.items():
        public = PUBLIC.with_name(study)/'lineage.json'; private = WORK.with_name(study)/'rosters.npz'; local = study=='cycle-boundary'
        require(sha(public)==pin, 'old metadata receipt pin'); meta=readj(public)
        require(sha(private)==meta['private_roster_sha256' if local else 'unpublished_private_rosters_sha256'], 'old private roster pin')
        union=set(); roles={}
        with np.load(private,allow_pickle=False) as z:
            for role,s in meta['cohorts'].items():
                idx=int(s['index']); rows=z[role+'_parent']; seed,cid=z[role+'_seed'],z[role+'_id']
                require(seed.dtype==cid.dtype==np.uint64 and rows.dtype.kind in 'iu' and np.all((rows>=0)&(rows<16384)), 'old roster types')
                k=np.column_stack((seed,cid)); require(np.array_equal(keys[idx][rows],k),'old roster does not match release row')
                require(array_sha(rows,'<i8')==s['parent_sha256' if local else 'rows_sha256'] and array_sha(k,'<u8')==s['key_sha256' if local else 'ids_sha256'], 'old role ordering changed')
                g=as_set(k); require(len(g)==len(k) and not g&union, 'duplicate old parent'); union.update(g); roles[role]=g
        require(len(union)==n,'incomplete prior trial'); groups[study]=union; by_role[study]=roles; receipts[study]=dict(lineage_sha256=pin,roster_sha256=sha(private),distinct=n)
    require(not groups['cycle-boundary'] & groups['spatial-cycle'], 'overlapping prior trials')
    return groups,by_role,receipts


def lineage():
    start=time.monotonic(); now=datetime.now(timezone.utc); src=pins(); folds={i:partition(i)[0] for i in INDICES}
    require(now < datetime.fromisoformat(DISCOVERY_STOP), 'discovery closed')
    for d in (WORK,PUBLIC): d.mkdir(parents=True,exist_ok=True)
    report=PUBLIC/'lineage.json'; priv=WORK/'rosters.npz'; require(not report.exists() and not priv.exists(),'roster already issued')
    geom=geometry(); keys={i:exact_keys(p) for i,p in src.items()}
    trials,oldroles,trial_pins=old_trials(keys); bans=dict(trials); metadata={}; fit_hist={}
    def mark(file):
        metadata[str(file.relative_to(ROOT) if file.is_relative_to(ROOT) else file)]=sha(file); return file
    def add(name,i,rr):
        bans.setdefault(name,set()).update(as_set(keys[i][np.array(rr,dtype=np.int64)]))
    fit_hist['first_pass_train']=set(); fit_hist['tranche2_train']=set(); fit_hist['preflight_train']=set(); trajectory=set()
    res=PUBLIC.parent; rootres=res.parent; f15=res/'fifteen'; pre=Path.home()/'rsmi-work/l20-preflight-f898730'; historical=readj(mark(rootres/'l20-propensity/archive-provenance.json'))
    for i in INDICES:
        train,val,test=folds[i]; k=keys[i]; e=src[i]; require(list(map(len,folds[i]))==[10240,2048,4096],'native split')
        # Identity fields only: no native outcome or old predictive scores are used.
        meta=readj(mark(rootres/f'l20-first-pass/fits/beta-{i:02}_b1_learned_11.json'))['provenance']
        require(meta['source_sha256']==e['npz_sha256'] and meta['parent_sha256']==[array_sha(k[r],'<u8') for r in folds[i]], 'first-pass native lineage mismatch')
        fit_hist['first_pass_train'].update(as_set(k[train])); add('native_validation',i,val); add('native_TEST',i,test)
        add('preparation',i,np.where(k[:,0]==int(e['prep_seed_uint64']))[0]);
        if i==7:
            with np.load(mark(f15/'04-10-parents.npz'),allow_pickle=False) as z:
                require(np.array_equal(z['train_ids'],train[:256]) and np.array_equal(z['val_ids'],val[:512]),'#04 transfer roster')
                add('selected_04',i,np.r_[z['train_ids'],z['val_ids']])
        else:
            with np.load(mark(f15/'topology-parents.npz'),allow_pickle=False) as z:
                r=z[f'b{i:02}_parent'];require(np.array_equal(r,train[:256]),'#04 roster'); add('selected_04',i,r)
        # Verify split and native identities directly from the compact historical bank metadata; never load species or outcomes.
        path=WORK.with_name('tranche2')/f'b{i:02}.npz'; receipt=readj(mark(path.with_suffix('.json')))
        nfit,ndev,ntotal = (4096,4608,5120) if i==9 else (256,512,768) if i in (0,8) else (256,256,256)
        require((receipt['fit_stop'],receipt['dev_stop'],receipt['audit_stop'])==(nfit,ndev,ntotal) and receipt['source_sha256']==e['npz_sha256'] and receipt['bank_sha256']==sha(path),'tranche2 metadata pin')
        with np.load(path,allow_pickle=False) as z:
            r=z['parent']; require(np.array_equal(r,train[:ntotal]) and np.array_equal(k[r],np.column_stack((z['chain_seed'],z['chain_id']))),'tranche2 ancestry')
        add('tranche2_dev',i,r[nfit:ndev]); add('tranche2_audit',i,r[ndev:]); fit_hist['tranche2_train'].update(as_set(k[r[:nfit]]))
        c,=[c for c in historical['cohorts'] if c['index']==i]; g={(int(c['preparation_seed']),int(a['parent_id'])) for a in c['lineage']}
        require(len(g)==c['parents'] and g <= as_set(k) and all(s==int(e['prep_seed_uint64']) for s,_ in g),'original archive diagnostic ancestry'); trajectory.update(g)
    bans['physical_parent_ancestry']=trajectory
    warm=readj(mark(rootres/'l20-operators/acquisition/forks-08.json')); cold=readj(mark(rootres/'l20-propensity/cold-provenance.json'))
    rr=np.asarray(cold['release_rows']); require(np.array_equal(keys[9][rr],np.asarray(cold['source_identity'],dtype='<u8')), 'old cold physical identities')
    pool=np.setdiff1d(folds[9][2][keys[9][folds[9][2],0]==int(src[9]['bulk_seed_uint64'])],rr)
    d9=np.sort(np.random.default_rng(7621009).choice(pool,512,replace=False)); d8=np.array(warm['release_rows']); proof=readj(mark(f15/'14-15-lineage.json'))['issued_forks']
    for i,r,pr in zip((9,9,8),(rr,d9,d8),proof):
        require(np.isin(r,folds[i][2]).all() and array_sha(r,'<i8')==pr['release_row_sha256'], 'forked TEST ancestry mismatch'); add('previous_fork_roster',i,r)
    add('cold_issue_reserve',9,folds[9][1][-1024:]); add('R4_tripod_tune',9,np.random.default_rng(210491).permutation(folds[9][0])[:2048]); mark(res/'boundary/design.json')
    # Original short smoke grid did not use native folds. Reconstruct heldouts, including its site-copy control.
    measurements,=pre.glob('measurements-*')
    paths=[pre/'deep-gate.json', *(p for p in sorted(measurements.glob('*.json')) if p.name!='summary.json')]
    for p in paths:
        m=readj(mark(p)); conf=m.get('config',m); nt,nf=int(conf['test']),int(conf['train']); i=int(Path(m['data']['path']).stem[-2:]); require(i in (0,9),'unregistered preflight')
        r=np.random.default_rng(np.random.SeedSequence(m['seed']).spawn(2)[0]).permutation(16384)
        require(m['data']['sha256']==src[i]['npz_sha256'] and m['data']['heldout_rows_sha256']==array_sha(r[:nt],'<i8') and m['data']['train_rows_sha256']==array_sha(r[nt:nt+nf],'<i8'), 'preflight split replay')
        add('preflight_heldout',i,r[:nt]); fit_hist['preflight_train'].update(as_set(keys[i][r[nt:nt+nf]]))
    require(len(bans['selected_04'])==1536,'incomplete old endpoint roster')
    mark(rootres/'l20-operators/static/design.json'); mark(rootres/'l20-operators/affine/design.json')
    deny=set().union(*bans.values()); assigned=set(); arr={}; roles={}; supply={}
    for i in INDICES:
        e=src[i]; k=keys[i]; train=folds[i][0]; trainset=as_set(k[train]); bulkseed=int(e['bulk_seed_uint64']); bulk={key for key in trainset if key[0]==bulkseed}
        cyclefree=bulk-trials['cycle-boundary']-trials['spatial-cycle']; elig=bulk-deny
        pool=np.array([int(r) for r in train if (int(k[r,0]),int(k[r,1])) in elig],dtype='<i8'); require(len(pool)==len(elig),'eligible pool count')
        pool=np.random.default_rng([ROW_SEED,i]).permutation(pool); supply[i]=dict(beta=e['beta'],native_train=len(trainset),bulk_TRAIN=len(bulk),prep_TRAIN=len(trainset-bulk),
             after_two_complete_trials=len(cyclefree),after_trials_04_and_tranche2=len(cyclefree-bans['selected_04']-bans['tranche2_dev']-bans['tranche2_audit']), eligible_after_all_exclusions=len(elig),
             TRAIN_intersections_nonadditive={n:len(trainset & v) for n,v in bans.items()},bulk_key_decimal=str(bulkseed), prep_key_decimal=e['prep_seed_uint64'])
        at=0
        for role,(idx,n) in ROLES.items():
            if idx!=i: continue
            rows=pool[at:at+n]; at+=n; g=as_set(k[rows]); require(len(rows)==n and len(g)==n and not g&(assigned|deny), 'insufficient or reused '+role)
            assigned.update(g); a=anchors(role,i,rows); arr.update({role+'_parent':rows,role+'_seed':k[rows,0],role+'_id':k[rows,1],role+'_anchor':a})
            roles[role]=dict(index=i,beta=e['beta'],parents=n,Q=a.shape[1],opportunities=a.size,rows_sha256=array_sha(rows,'<i8'),ids_sha256=array_sha(k[rows],'<u8'),
                anchors_u2_sha256=array_sha(a,'<u2'),roster_sha256=roster_sha(rows,a),forbidden_overlap={name:len(g & x) for name,x in bans.items()},
                full_trial_role_overlap={study:{name:len(g & x) for name,x in sub.items()} for study,sub in oldroles.items()},history_other_TRAIN={name:len(g & x) for name,x in fit_hist.items()})
        supply[i]['unused_after_assignment']=len(elig)-at
    savenp(priv,arr)
    savej(report,dict(utc_started=now.isoformat(),seconds=time.monotonic()-start,scope='metadata only; no native species, energy, time histories, near-halo, outcomes or previous cycle-model results opened',
        geometry=geom,parents=len(assigned),cohorts=roles,supply=supply,public_design_sha256=sha(PUBLIC/'bank-design.json'),rosters_sha256=sha(priv),
        metadata_receipt_pins=metadata,prior_trials=trial_pins,study_forbidden_distinct={n:len(x) for n,x in bans.items()},source_manifest_sha256=INDEX_SHA,
        source_pins={i:{v:src[i][v] for v in ('npz_sha256','sidecar_sha256','beta','study','study_index','prep_seed_uint64','bulk_seed_uint64','physical_diagnostic_seed_uint64')} for i in INDICES},
        history='All roles contributed to original first-pass 2^3 visible-to-shell RSMI fits at their beta; the parent folds are exactly the same. Historical beta9 operator/AFFINE training used 3^3 V, K4, six 5x5 shell face patches including distant R4; shared direction/scale, NOT the present 386-site species shell, hidden outcome, forecast or role. Prior R4 vacancy tripod used beta9 TRAIN, native val at 9/8/0; cycle rosters at all temperature/roles barred. Held-out from this HERO experiment by exact lineage, not pristine replication. No independent beta3.05 exam is claimed.',
        archive_scope='Only bulk namespace eligible. Monolithic compressed file hashes include all stored bytes but no denied physical arrays were decoded. Native metadata members were read at indices 0,7,8,9 only. Complete original archival fork lineages excluded without loading snapshots. Prior #04, tranche2, other study rosters and first-pass partition receipts read for identity only.',
        no_exam_open=True,incremental_fee_usd=0))
    print(json.dumps({i:supply[i] for i in INDICES}),flush=True)


def committed(path, revision):
    rel=str(path.relative_to(ROOT)); h=hashlib.sha256(); proc=subprocess.Popen(['git','-C',str(ROOT),'show',revision+':'+rel],stdout=subprocess.PIPE)
    with proc.stdout as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    require(proc.wait()==0 and h.hexdigest()==sha(path), 'not the public commit version: '+rel)


def authorize(stage, role, rev):
    require(role in STAGES.get(stage,()),'role not permitted in stage')
    require(datetime.now(timezone.utc)<datetime.fromisoformat(DISCOVERY_STOP if stage=='discovery' else HARD_STOP), 'native release cutoff passed')
    require(re.fullmatch(r'[0-9a-f]{40}',rev) is not None and subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',rev,'HEAD']).returncode==0,'require public commit on current history')
    p=PUBLIC/f'release-{stage}-{role}.json'; committed(p,rev); release=readj(p); lineage_path=PUBLIC/'lineage.json'
    require(release['stage']==stage and release['role']==role and release['lineage_sha256']==sha(lineage_path), 'wrong role or lineage release')
    for f in (lineage_path,PUBLIC/'bank-design.json',Path(__file__).resolve()): committed(f,rev)
    prereq=release['prerequisites']; require(isinstance(prereq,dict) and len(prereq)>0,'missing explicit prerequisites')
    for name,digest in prereq.items():
        require(Path(name).name==name and name not in ('.','..') and sha(PUBLIC/name)==digest,'release prerequisite changed'); committed(PUBLIC/name,rev)
    lock=readj(lineage_path); require(lock['public_design_sha256']==sha(PUBLIC/'bank-design.json') and lock['rosters_sha256']==sha(WORK/'rosters.npz'),'rosters or design changed')
    return p, lock


def issued(role, rows, a, rev):
    path=PUBLIC/f'predictions-{role}.npz'; meta=PUBLIC/f'predictions-{role}.json'; blind=WORK/f'blind-{role}.npz'; bank=PUBLIC/f'bank-blind-{role}.json'
    for p in (path,meta,bank): committed(p,rev)
    md=readj(meta); rec=readj(bank)
    require(md['role']==rec['role']==role and rec['stage']=='blind' and md['blind_sha256']==rec['bank_sha256']==sha(blind) and md['roster_sha256']==roster_sha(rows,a)
            and md['opportunities']==a.size and md['prediction_sha256']==sha(path),'issued prediction manifest does not cover blind bank')
    with np.load(path,allow_pickle=False) as z:
        require(np.array_equal(rows,z['parent']) and np.array_equal(a,z['anchor']), 'prediction row/anchor mismatch')
        forecast=[n for n in z.files if n.lower().startswith(('m0','m1'))]
        require(all(any(n.lower().startswith(stem) for n in forecast) for stem in ('m0','m1')),'missing signed forecast pair')
        for n in forecast:
            v=z[n]; require(v.shape[:2]==a.shape and v.dtype.kind=='f' and np.isfinite(v).all(),'incomplete or nonfinite '+n)
    return md['prediction_sha256']


def exterior_inventory(row, interior):
    # No decoded tensor of other anchors' centers, including during bin-counting.
    inside=np.sort(interior); splits=np.flatnonzero(np.diff(inside)!=1)+1
    lo=np.r_[0,inside[splits-1].astype(np.int32)+1,int(inside[-1])+1]; hi=np.r_[int(inside[0]),inside[splits],8000]
    outside=[(int(l),int(r)) for l,r in zip(lo,hi) if l<r]
    return (2000-sum(row.count(b'\x00',l,r) for l,r in outside),2400-sum(row.count(b'\x01',l,r) for l,r in outside))


def extract(stage,role,rev):
    start=time.monotonic(); release,lock=authorize(stage,role,rev); c=lock['cohorts'][role]; idx,n=ROLES[role]; src=pins()[idx]
    require(c['index']==idx and c['parents']==n and src['npz_sha256']==lock['source_pins'][str(idx)]['npz_sha256'], 'wrong source for role')
    dest=WORK/((role if stage=='discovery' else stage+'-'+role)+'.npz'); receipt=PUBLIC/f'bank-{stage}-{role}.json';require(not dest.exists() and not receipt.exists(),'stage already spent')
    with np.load(WORK/'rosters.npz',allow_pickle=False) as z:
        rows,seed,cid,a=(z[role+'_'+k] for k in ('parent','seed','id','anchor'))
    require(roster_sha(rows,a)==c['roster_sha256'] and array_sha(np.column_stack((seed,cid)),'<u8')==c['ids_sha256'] and np.isin(rows,partition(idx)[0][0]).all(), 'role drift')
    pred=issued(role,rows,a,rev) if stage=='hidden' else None
    k=exact_keys(src); require(np.array_equal(k[rows],np.column_stack((seed,cid))) and np.all(seed==int(src['bulk_seed_uint64'])),'role/source identity mismatch')
    data={'parent':rows,'anchor':a}; q=a.shape[1]
    if stage!='blind': data['V']=np.empty((n,q,27),np.uint8)
    if stage!='hidden':
        for key in ('E','E5'): data[key]=np.empty((n,q,386),np.uint8)
        data['I']=np.empty((n,q,2),dtype='<u2')
    with ZipFile(DATA/src['npz']) as z, z.open('species.npy') as f:
        ver=np.lib.format.read_magic(f); hdr={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
        shape,fortran,dtype=hdr[ver](f); require(shape==(16384,20,20,20) and not fortran and dtype==np.uint8,'bad species header'); base=f.tell()
        for j in np.argsort(rows):
            address=base+8000*int(rows[j]); f.seek(address)
            if stage=='hidden':
                # Read only registered center bytes in increasing archive order. ZIP may decompress intervening data internally.
                sites_v=sites(a[j],CENTER).ravel(); order=np.argsort(sites_v); ordered=sites_v[order]; v=data['V'][j].reshape(-1)
                for run in np.split(np.arange(len(order)),np.flatnonzero(np.diff(ordered)!=1)+1):
                    f.seek(address+int(ordered[run[0]])); chunk=f.read(len(run)); require(len(chunk)==len(run),'short center read'); v[order[run]]=np.frombuffer(chunk,np.uint8)
                continue
            row=f.read(8000); require(len(row)==8000,'short authorized row'); state=np.frombuffer(row,np.uint8)
            data['E'][j]=state[sites(a[j],SHELL)]; data['E5'][j]=state[sites(a[j],SHELL5)]; u=sites(a[j],INTERIOR)
            if stage=='discovery':
                require(row.count(b'\x00')==2000 and row.count(b'\x01')==2400 and row.count(b'\x02')==3600,'invalid fixed composition')
                data['V'][j]=state[sites(a[j],CENTER)]; inventory=state[u]
                data['I'][j,:,0]=(inventory==0).sum(-1); data['I'][j,:,1]=(inventory==1).sum(-1)
            else:
                for t,interior in enumerate(u):
                    nv,na=exterior_inventory(row,interior); require(0<=nv<=343 and 0<=na<=343-nv,'invalid blind aggregate'); data['I'][j,t]=(nv,na)
            del row,state
    require(all(int(data[k].max())<=2 for k in ('V','E','E5') if k in data),'invalid selected archive alphabet')
    savenp(dest,data)
    savej(receipt,dict(stage=stage,role=role,utc=datetime.now(timezone.utc).isoformat(),release_commit=rev,release_sha256=sha(release),
          lineage_sha256=sha(PUBLIC/'lineage.json'),source_sha256=src['npz_sha256'],roster_sha256=roster_sha(rows,a),
          bank_sha256=sha(dest),bank_bytes=dest.stat().st_size,prediction_sha256=pred,parents=n,opportunities=a.size,keys=list(data),
          elapsed_seconds=time.monotonic()-start,process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          native_scientific_access='only released parent and role; blind is two sparse shells plus aggregate stocks, not any microscopic interior labels',fee_usd=0))
    print(receipt,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('lineage'); ex=sub.add_parser('extract'); ex.add_argument('stage',choices=STAGES); ex.add_argument('role',choices=ROLES); ex.add_argument('--release-commit',required=True)
    cli=parser.parse_args(); lineage() if cli.mode=='lineage' else extract(cli.stage,cli.role,cli.release_commit)
