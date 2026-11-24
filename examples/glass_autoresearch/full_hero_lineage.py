"""Issue HERO v2 metadata-only rosters. Never open a native endpoint or trajectory."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples'))
from l20_pairs import partition

DATA = Path.home() / 'datasets/lattice-glass/l20-16384-v1'
RESULTS = ROOT / 'results'
PUB = RESULTS / 'l20-autoresearch/hero-full-v2'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/hero-full-v2'
INDEX_SHA = 'd4c591bab7a75e785fb749fa42cc78353dfc6d093c6742ae8d621ca2113040ff'
FIRST_SHA = '798ac54bc3b5dfd291a97d8a06c61d1da1d02acc7d76d70d5eae4108ebdbad8d'
HERO_SHA = '4b395b42d5bca1d7338a896786e6ea8a1b71d45f09ffb29768938c62c0f29589'
ROLES = ('fit', 'dev', 'eval')
N = 16384


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ah(a, dtype):
    return hashlib.sha256(np.asarray(a, dtype=dtype).tobytes()).hexdigest()


def readj(path):
    return json.loads(path.read_bytes())


def keys_at(k, r):
    return set(map(tuple, k[r].tolist()))


class Audit:
    def __init__(self):
        self.pins = {}
        self.keys = {}
        self.folds = {}
        self.fold_proof = {}
        self.held = defaultdict(set)
        self.history = defaultdict(set)
        self.trials = {}

    def pin(self, path, expected=None):
        h = digest(path)
        check(expected is None or h == expected, 'changed receipt: ' + str(path))
        self.pins[str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)] = h
        return path

    def register(self):
        man = readj(self.pin(DATA/'release-index.json', INDEX_SHA))
        self.entries = {e['release_index']: e for e in man['endpoints']}
        check(set(self.entries) == set(range(10)) and man['endpoint_count'] == 10*N, 'release cardinality')
        check(len({e['bulk_seed_uint64'] for e in self.entries.values()}) == 10 and
              not ({e['bulk_seed_uint64'] for e in self.entries.values()} &
                   {e[f] for e in self.entries.values() for f in ('prep_seed_uint64','physical_diagnostic_seed_uint64')}),
              'bulk namespace collision')
        protocols = {s: readj(self.pin(DATA/v['protocol'], v['protocol_sha256'])) for s,v in man['studies'].items()}
        with ZipFile(self.pin(RESULTS/'l20-first-pass/fit-receipts.zip', FIRST_SHA)) as archive:
            for i,e in sorted(self.entries.items()):
                m = protocols[e['study']]; idx = e['study_index']; prep = e['prep_seed_rows']
                check(m['parents'] == prep and m['chains'] == e['endpoints'] == N and e['bulk_seed_rows'] == N-prep and
                      m['prep_seed'] == int(e['prep_seed_uint64']) and m['production_seeds'][idx] == int(e['bulk_seed_uint64']) and
                      m['betas'][idx] == e['beta'], 'protocol/manifest mismatch')
                # production/export.py:_collect: diagnostic prefix strided by the generating study's beta count;
                # single-temperature production batches follow in chain-id order, starting at zero.
                k = np.empty((N,2), dtype='<u8'); k[:prep,0] = int(e['prep_seed_uint64']); k[prep:,0] = int(e['bulk_seed_uint64'])
                k[:prep,1] = idx + np.arange(prep, dtype='<u8') * len(m['betas']); k[prep:,1] = np.arange(N-prep, dtype='<u8')
                folds = partition(i)[0]; r = np.concatenate(folds)
                check(list(map(len, folds)) == [10240,2048,4096] and np.array_equal(np.sort(r),np.arange(N)), 'native partition closure')
                member = f'fits/beta-{i:02}_b1_learned_11.json'
                provenance = json.loads(archive.read(member))['provenance']
                cert = [ah(k[rr], '<u8') for rr in folds]
                check(cert == provenance['parent_sha256'] and provenance['source_sha256'] == e['npz_sha256'], 'independent native key proof '+str(i))
                self.keys[i] = k; self.folds[i] = folds
                self.fold_proof[i] = dict(public_member=member, native_train_val_test_ordered_key_sha256=cert)
                self.mark('preparation', i, np.arange(prep)); self.mark('native_validation', i, folds[1]); self.mark('native_TEST', i, folds[2]); self.mark('first_pass_generic_TRAIN', i, folds[0], history=True)
                if i == 6:
                    self.mark('index6_permanent', i, np.arange(N))

    def mark(self, name, i, rows, history=False):
        rows = np.asarray(rows)
        check(rows.ndim == 1 and rows.dtype.kind in 'iu' and np.all((rows >= 0) & (rows < N)), 'invalid old rows '+name)
        (self.history if history else self.held)[name].update(keys_at(self.keys[i], rows))

    def old_trials(self):
        v1 = readj(self.pin(PUB.with_name('hero-central')/'lineage.json', HERO_SHA)); distinct = set()
        for name,n,roster_field,row_field,key_field in [
                ('cycle-boundary',6912,'private_roster_sha256','parent_sha256','key_sha256'),
                ('spatial-cycle',6016,'unpublished_private_rosters_sha256','rows_sha256','ids_sha256'),
                ('hero-central',8448,'rosters_sha256','rows_sha256','ids_sha256')]:
            receipt = v1 if name=='hero-central' else readj(self.pin(PUB.with_name(name)/'lineage.json',v1['prior_trials'][name]['lineage_sha256']))
            confirmed = set(); role_counts = {}
            with np.load(self.pin(WORK.with_name(name)/'rosters.npz',receipt[roster_field]), allow_pickle=False) as z:
                for role,c in receipt['cohorts'].items():
                    i = c['index']; rows = z[role+'_parent']; seed,ids = z[role+'_seed'],z[role+'_id']; k = np.column_stack((seed,ids))
                    check(rows.ndim==1 and rows.dtype.kind in 'iu' and np.all((rows>=0)&(rows<N)) and
                          seed.dtype==ids.dtype==np.dtype('uint64') and np.array_equal(k,self.keys[i][rows]) and
                          ah(rows,'<i8')==c[row_field] and ah(k,'<u8')==c[key_field], 'historical roster mismatch '+name+'/'+role)
                    g = keys_at(self.keys[i],rows); check(len(g)==len(rows) and g.isdisjoint(confirmed|distinct),'prior roster duplicate')
                    confirmed.update(g); role_counts[role] = len(g)
                    self.mark(name+'_'+role,i,rows,history=name=='hero-central' and role=='fit')
            check(len(confirmed)==n,'incomplete old study '+name); distinct.update(confirmed)
            self.trials[name] = dict(distinct=n, roles=role_counts, private_roster_sha256=receipt[roster_field])
        self.v1 = v1

    def prior_holds(self):
        oldpins = self.v1['metadata_receipt_pins']; res=PUB.parent; f15=res/'fifteen'
        def old(path):
            return self.pin(path, oldpins[str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)])
        physical = readj(old(RESULTS/'l20-propensity/archive-provenance.json'))['cohorts']
        check(sorted(c['index'] for c in physical)==list(range(10)), 'original diagnostic cohorts')
        for c in physical:
            i=c['index']; e=self.entries[i]; n=e['prep_seed_rows']; k=self.keys[i]
            g={(int(c['preparation_seed']),int(a['parent_id'])) for a in c['lineage']}
            check(len(g)==c['parents']==n and g==keys_at(k,np.arange(n)), 'archive physical source ancestry')
            self.held['physical_fork_ancestor'].update(g)
        with np.load(old(f15/'topology-parents.npz'), allow_pickle=False) as z:
            for i in (0,8,9):
                rows=z[f'b{i:02}_parent']; check(np.array_equal(rows,self.folds[i][0][:256]), '#04 discovery ancestry'); self.mark('selected_04',i,rows)
                self.mark('fifteen_internal_calibration',i,rows[128:])
        with np.load(old(f15/'04-10-parents.npz'), allow_pickle=False) as z:
            for group,field,n in [(0,'train_ids',256),(1,'val_ids',512)]:
                rows=z[field]; check(np.array_equal(rows,self.folds[7][group][:n]), '#04 transfer ancestry'); self.mark('selected_04',7,rows)
        check(len(self.held['selected_04'])==1536,'old outcome roster')
        for i in sorted(set(self.entries)-{6}):
            path=WORK.with_name('tranche2')/f'b{i:02}.json'; m=readj(self.pin(path,oldpins.get(str(path))))
            x,y,n = (4096,4608,5120) if i==9 else (256,512,768) if i in (0,8) else (256,256,256)
            check((m['fit_stop'],m['dev_stop'],m['audit_stop'])==(x,y,n) and m['source_sha256']==self.entries[i]['npz_sha256'],'tranche2 contract')
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                rows=z['parent']; check(np.array_equal(rows,self.folds[i][0][:n]) and np.array_equal(self.keys[i][rows],np.column_stack((z['chain_seed'],z['chain_id']))),'tranche2 identity proof')
            self.mark('tranche2_fit',i,rows[:x],True); self.mark('tranche2_dev',i,rows[x:y]); self.mark('tranche2_audit',i,rows[y:])
        cold=readj(old(RESULTS/'l20-propensity/cold-provenance.json')); warm=readj(old(RESULTS/'l20-operators/acquisition/forks-08.json'))
        rr=np.array(cold['release_rows']); test=self.folds[9][2]; check(np.array_equal(self.keys[9][rr],np.asarray(cold['source_identity'],dtype='<u8')),'spent fork ancestry')
        pool=np.setdiff1d(test[self.keys[9][test,0]==int(self.entries[9]['bulk_seed_uint64'])],rr)
        issued=readj(old(f15/'14-15-lineage.json'))['issued_forks']
        for i,r,c in zip((9,9,8),(rr,np.sort(np.random.default_rng(7621009).choice(pool,512,replace=False)),np.array(warm['release_rows'])),issued,strict=True):
            check(np.isin(r,self.folds[i][2]).all() and ah(r,'<i8')==c['release_row_sha256'],'issued TEST exams'); self.mark('previous_fork_roster',i,r)
        train,val,_=self.folds[9]; self.mark('cold_issue_reserve',9,val[-1024:]); tripod=np.random.default_rng(210491).permutation(train)[:2048]
        lock=readj(self.pin(res/'boundary/cold-lock.json')); old(res/'boundary/design.json')
        check(ah(tripod,'<u4')==lock['tune_parent_ids_sha256'], 'old R4 tune hash'); self.mark('R4_tripod_tune',9,tripod)
        self.mark('R4_tripod_fit',9,np.setdiff1d(train,tripod),True)
        m=readj(self.pin(res/'mechanism/conventional-fits.json'))['reserved_calibration']; self.pin(res/'innovation/design.json'); self.pin(res/'transparent/design.json')
        check(m['first_training_row']==7680 and m['parents']==2560 and ah(train[7680:],'<u8')==m['calibration_row_id_sha256'], 'supervised calibration roster')
        self.mark('cold_supervised_calibration',9,train[7680:]); self.pin(res/'rate/design.json'); self.mark('cold_rate_internal_tune',9,train[-1024:])
        for name in ('static','affine'):
            old(RESULTS/f'l20-operators/{name}/design.json')
        self.mark('related_3cube_face_RSMI_TRAIN',9,train,True)
        pre=Path.home()/'rsmi-work/l20-preflight-f898730'; files=sorted([pre/'deep-gate.json',*filter(lambda p:p.name!='summary.json',pre.glob('measurements-*/*.json'))]); check(set(map(str,files))=={k for k in oldpins if k.startswith(str(pre)+'/')}, 'incomplete preflight roster')
        for path in files:
            m=readj(old(path)); conf=m.get('config',m); i=int(Path(m['data']['path']).stem[-2:]); check(i in (0,9), 'unregistered preflight beta')
            nt,nf=conf['test'],conf['train']; rows=np.random.default_rng(np.random.SeedSequence(m['seed']).spawn(2)[0]).permutation(N); tr=rows[nt:nt+nf]; te=rows[:nt]
            check(m['data']['sha256']==self.entries[i]['npz_sha256'] and m['data']['heldout_rows_sha256']==ah(te,'<i8') and m['data']['train_rows_sha256']==ah(tr,'<i8'),'preflight replay')
            self.mark('preflight_heldout',i,te); self.mark('preflight_fit',i,tr,True)

    def issue(self, temp, worktemp):
        keys=[(int(s),int(c)) for k in self.keys.values() for s,c in k]; counts=Counter(keys)
        # Native hashes fix this release exactly. New duplicates must not receive independent folds.
        duplicates={k for k,n in counts.items() if n>1}; self.held['duplicate_lineage'].update(duplicates)
        denied=set().union(*self.held.values()); prior=self.history['hero-central_fit']; assigned=set(); arrays={}; dest={}; beta_counts={}
        check(len(prior)==4096 and prior.isdisjoint(denied),'old FIT also has a mandatory hold')
        for i,fold in self.folds.items():
            train=fold[0]; k=self.keys[i]; admitted=np.array([r for r in train if tuple(k[r].tolist()) not in denied],dtype='<i8')
            if i==6:
                check(len(admitted)==0,'index 6 leak'); continue
            perm=np.random.Generator(np.random.PCG64([26112501,i])).permutation(admitted)
            previously_fit=np.array([tuple(k[r].tolist()) in prior for r in perm],dtype=bool); fresh=perm[~previously_fit]; n=len(fresh)//8
            check(n>0,'no disjoint exam supply '+str(i)); role=np.full(len(perm),'fit',dtype='U4'); positions=np.flatnonzero(~previously_fit); role[positions[:n]]='eval'; role[positions[n:2*n]]='dev'
            for label in ROLES:
                rows=perm[role==label]; g=keys_at(k,rows); check(len(g)==len(rows) and g.isdisjoint(assigned|denied) and (label=='fit' or g.isdisjoint(prior)), 'unsafe assignment '+str(i)+label)
                assigned.update(g); arrays[f'b{i:02}_{label}']=rows
                for rank,r in enumerate(rows): dest[i,int(r)]=(label,rank)
            beta_counts[i]={s:int((role==s).sum()) for s in ROLES}; beta_counts[i]['v1_fit_only']=int(previously_fit.sum())
        check(assigned==set(counts)-denied and prior<=assigned,'eligible lineage closure')
        with (worktemp/'rosters.npz').open('xb') as out: np.savez_compressed(out,**arrays)
        priority=('index6_permanent','preparation','native_TEST','native_validation','duplicate_lineage',
                  *(k for k in self.held if k.startswith(('hero-central_','cycle-boundary_','spatial-cycle_'))),
                  *sorted(k for k in self.held if k not in ('index6_permanent','preparation','native_TEST','native_validation','duplicate_lineage') and not k.startswith(('hero-central_','cycle-boundary_','spatial-cycle_'))))
        check(set(priority)==set(self.held) and len(priority)==len(self.held),'unmapped hold')
        reasons=defaultdict(list); histories=defaultdict(list)
        for name in priority:
            for g in self.held[name]: reasons[g].append(name)
        for name,group in sorted(self.history.items()):
            for g in group: histories[g].append(name)
        table=temp/'row-routing.csv.gz'; summary=Counter(); routecounts=Counter(); seen=set(); net=Counter(); flagcounts=defaultdict(Counter)
        with table.open('xb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz, io.TextIOWrapper(gz,encoding='ascii',newline='') as text:
            w=csv.writer(text,lineterminator='\n'); w.writerow(['beta','index','row','chain_seed','chain_id','native_fold','lineage_multiplicity','net_unique_first','route','roster_rank','hold_reasons','exposure_flags'])
            for i,k in self.keys.items():
                folds=np.empty(N,dtype='U5')
                for label,rows in zip(('TRAIN','VAL','TEST'),self.folds[i],strict=True): folds[rows]=label
                for row,key in enumerate(map(tuple,k.tolist())):
                    flags=reasons.get(key,()); history=histories.get(key,()); flagcounts[i].update(flags); flagcounts[i].update(history)
                    if flags:
                        check((i,row) not in dest,'held native row in roster'); route='exclude:'+flags[0]; rank=''
                    else:
                        route,rank=dest[i,row]
                    first=int(key not in seen); seen.add(key); summary[i,route]+=1; net[i,route]+=first; routecounts[route]+=1
                    w.writerow((self.entries[i]['beta'],i,row,str(key[0]),str(key[1]),folds[row],counts[key],first,route,rank,'|'.join(flags),'|'.join(history)))
        check(len(seen)==len(counts) and sum(routecounts.values())==10*N and sum(net.values())==len(counts),'raw/net count closure')
        for i in self.keys:
            check(sum(n for (idx,s),n in summary.items() if idx==i)==N,'per-beta closure')
        tsv=temp/'routing.tsv'
        with tsv.open('x',newline='') as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n'); w.writerow(['index','beta','group','category','raw_rows','net_unique_first'])
            for i,e in self.entries.items():
                for s in sorted(routecounts):
                    w.writerow([i,e['beta'],'primary_route',s,summary[i,s],net[i,s]])
                for s in (*priority,*sorted(self.history)):
                    w.writerow([i,e['beta'],'overlapping_hold' if s in self.held else 'overlapping_history',s,flagcounts[i][s],''])
        return dict(raw_rows=10*N, net_unique_lineages=len(counts), unresolved_identities=0,
                    duplicate_excess_rows=len(keys)-len(counts), multiplicity_histogram=dict(Counter(counts.values())),
                    per_beta={i:{'beta':e['beta'],'raw_rows':N,'distinct_within_beta':len(keys_at(self.keys[i],np.arange(N))),
                                 **beta_counts.get(i,dict(fit=0,dev=0,eval=0,v1_fit_only=0)),
                                 'primary_route':{s:n for (idx,s),n in summary.items() if idx==i},'overlapping_flags':dict(flagcounts[i])} for i,e in self.entries.items()},
                    total_fit=sum(v['fit'] for v in beta_counts.values()),total_dev=sum(v['dev'] for v in beta_counts.values()),total_eval=sum(v['eval'] for v in beta_counts.values()),
                    all_eligible=len(assigned), excluded_rows=10*N-len(assigned), primary_route=dict(routecounts),
                    hold_precedence=priority, global_nonadditive_hold_lineages={s:len(g) for s,g in self.held.items()},
                    global_nonadditive_exposure_lineages={s:len(g) for s,g in self.history.items()},
                    rosters_sha256=digest(worktemp/'rosters.npz'), rows_csv_gz_sha256=digest(table), routing_tsv_sha256=digest(tsv),
                    ordered_roster_i8le_sha256={k:ah(v,'<i8') for k,v in arrays.items()})


def main():
    now=datetime.now(timezone.utc); start=time.monotonic(); cpu=time.process_time()
    check(now.isoformat()<'2026-11-24T21:00:25+00:00','discovery closed')
    WORK.mkdir(parents=True,exist_ok=True); public=('row-routing.csv.gz','routing.tsv','lineage.json')
    check(not any((PUB/f).exists() for f in public) and not (WORK/'rosters.npz').exists(),'already issued')
    audit=Audit(); design=audit.pin(PUB/'lineage-design.json'); audit.pin(PUB/'design.json'); audit.register(); audit.old_trials(); audit.prior_holds()
    with tempfile.TemporaryDirectory(dir=PUB) as p, tempfile.TemporaryDirectory(dir=WORK) as w:
        p=Path(p); w=Path(w); report=audit.issue(p,w)
        report.update(schema='forward-hero.v2.row-routing.1',utc_started=now.isoformat(),source_manifest_sha256=INDEX_SHA,
                      metadata_receipt_pins=audit.pins,source_pins=audit.entries,prior_trials=audit.trials,native_key_proof=audit.fold_proof,
                      lineage_design_sha256=digest(design),implementation_sha256=digest(Path(__file__)),anchor_seed=26112502,fold_seed=26112501,
                      routing='Priority assigns one route per row; hold_reasons and exposure_flags overlap. rank is zero-based within beta/role. Per-key precision uses decimal uint64; net_unique_first is first in release-index then row order, not an extra specimen.',
                      scope='Archival administrative metadata, not synthetic and not physical measurement. No native NPZ, native sidecar, feature, trajectory, species, energy or model/forecast array accessed. Historical mixed receipts used for provenance only; mixed past NPZ read only parent/seed/ID members. Index 6 is completely embargoed: identity only from released manifest/protocol and previously public first-pass fold digest proof. No unresolved identities.',
                      independence='No assigned parent has a mandatory hold, across all temperatures. All three folds historically used for generic prior native TRAIN fitting; cold TRAIN also related 3-cube visible RSMI with different face target. Only original HERO FIT may recur in V2 and only in FIT. New exam is independent by assignment for this forward study, not pristine replication. No specimen transfer.',
                      synthetic=False,physical_features_read=False,index6_native_opened=False,incremental_usd=0,
                      timing_scope='physical archive metadata routing only; no physical feature scoring, no synthetic control',cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start)
        (p/'lineage.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        # The public receipt is the completion marker. No overwrite of an issued contract.
        (WORK/'rosters.npz').hardlink_to(w/'rosters.npz')
        for filename in public: (PUB/filename).hardlink_to(p/filename)
        print(json.dumps({key:report[key] for key in ('raw_rows','net_unique_lineages','all_eligible','excluded_rows','total_fit','total_dev','total_eval','rosters_sha256')},indent=2))


if __name__=='__main__':
    main()
