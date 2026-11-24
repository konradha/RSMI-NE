"""V3 only. There is no TEST species reader or alternate role switch in this program."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from zipfile import ZipFile
import numpy as np

from hero_bank import DATA, INDEX_SHA, ROOT, committed, coords, exact_keys, readj, require, sha
from full_hero_stream import FIELDS, anchors, benchmark, cutoff, file_manifest, io_counters, lookup, savej

PUB = ROOT/'results/l20-autoresearch/hero-full-v3'
WORK = Path.home()/'rsmi-work/l20-autoresearch-1795520725/hero-full-v3'
STREAMS = WORK/'streams'
INDICES = tuple(range(10))
Q = np.arange(8192); PILOT = Q[Q%8 >= 2][:64]; REST = np.setdiff1d(Q,PILOT)
KEYS = ('V','E','parent')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()


def rss():
    return int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmHWM:')))


def gate(rev):
    cutoff('fit-dev'); require(re.fullmatch('[0-9a-f]{40}',rev) and subprocess.run(
        ['git','-C',str(ROOT),'merge-base','--is-ancestor',rev,'HEAD']).returncode==0,'release revision')
    for p in (PUB/'design.json', PUB/'split.json',PUB/'stream-design.json',PUB/'stream-release-1.json',
              Path(__file__).resolve(),Path(__file__).with_name('full_hero_stream.py'),Path(__file__).with_name('hero_bank.py')):
        committed(p,rev)
    release = readj(PUB/'stream-release-1.json'); split = readj(PUB/'split.json')
    require(release['stage']=='train' and release['design_sha256']==split['design_sha256']==sha(PUB/'design.json') and
            release['split_sha256']==sha(PUB/'split.json') and release['rosters_sha256']==split['rosters_sha256']==sha(WORK/'rosters.npz')
            and split['source_manifest_sha256']==INDEX_SHA==sha(DATA/'release-index.json') and
            split['train']==split['test']==81920 and split['cross_side_intersections']==0,'wrong v3 split/release')
    return release,split


def train_rows(split,i):
    require(i in INDICES,'invalid beta')
    with np.load(WORK/'rosters.npz',allow_pickle=False) as z:
        train,test = (z[f'b{i:02d}_{r}'] for r in ('train','test'))
    for r,v in (('train',train),('test',test)):
        require(v.shape==(8192,) and v.dtype==np.dtype('<i4') and digest(v)==split['ordered_native_int32le_roster_sha256'][f'b{i:02d}_{r}'],'native row pin')
    require(np.array_equal(np.sort(np.r_[train,test]),np.arange(16384)),'not disjoint complete halves')
    return train


def folder(i): return STREAMS/f'b{i:02d}'/'train'


def partial(i): return STREAMS/f'b{i:02d}'/'pilot-subset.json'


def projection(i,rev,first):
    start = time.perf_counter(); release,split=gate(rev); rows=train_rows(split,i)
    src=next(e for e in readj(DATA/'release-index.json')['endpoints'] if e['release_index']==i)
    ids=exact_keys(src); s=split['per_beta'][str(i)]
    require(digest(ids)==s['native_identity_sha256'] and digest(ids[rows])==s['train_identity_sha256'], 'v3 native identity mismatch')
    d=folder(i); final=PUB/f'stream-b{i:02d}.json'; require(not final.exists(),'full native pass already spent')
    ppos=PILOT if first else REST; aux=partial(i); relsha=sha(PUB/'stream-release-1.json')
    if first:
        d.mkdir(parents=True,exist_ok=False)
        with (d/'parent.npy').open('xb') as f: np.save(f,rows); f.flush(); os.fsync(f.fileno())
        arr={k:np.lib.format.open_memmap(d/(k+'.npy'),mode='w+',dtype=dt,shape=(8192,64,*shape)) for k,(dt,shape) in FIELDS.items()}
    else:
        prev=readj(aux); require(prev['release_sha256']==relsha and prev['source_sha256']==src['npz_sha256'] and
                prev['native_pilot_rows_sha256']==digest(rows[PILOT]),'incompatible pilot projection')
        require(not (d.parent/'remainder-started.json').exists(),'remainder pass already attempted'); savej(d.parent/'remainder-started.json',dict(release_sha256=relsha))
        arr={k:np.load(d/(k+'.npy'),mmap_mode='r+',allow_pickle=False) for k in FIELDS}
        for k in arr: require(digest(arr[k][PILOT])==prev['pilot_raw_sha256'][k], 'original pilot subset changed')
        require(np.array_equal(np.load(d/'parent.npy',allow_pickle=False),rows),'row labels drift')
    vt,et,it=lookup(); io0=io_counters(); scan=time.perf_counter()
    with ZipFile(DATA/src['npz']) as z,z.open('species.npy') as f:
        ver=np.lib.format.read_magic(f); require(ver in ((1,0),(2,0)),'invalid species header version')
        shape,order,dt=(np.lib.format.read_array_header_1_0 if ver==(1,0) else np.lib.format.read_array_header_2_0)(f)
        require(shape==(16384,20,20,20) and not order and dt==np.uint8,'native shape'); base=f.tell()
        for j,p in enumerate(ppos[np.argsort(rows[ppos])]):
            if j%64==0: cutoff('fit-dev')
            row=int(rows[p]); addr=base+8000*row; require(f.tell()<=addr,'backward/duplicate native read')
            f.seek(addr); data=f.read(8000); require(len(data)==8000 and data.count(b'\x00')==2000 and data.count(b'\x01')==2400 and data.count(b'\x02')==3600,'invalid TRAIN species')
            x=np.frombuffer(data,'u1'); a=anchors(i,row,True); arr['anchor'][p]=coords(a)
            arr['V'][p]=x[vt[a]]; arr['E'][p]=x[et[a]]; core=x[it[a]]
            arr['I'][p,:,0]=(core==0).sum(1); arr['I'][p,:,1]=(core==1).sum(1)
    for k,a in arr.items():
        a.flush()
        with (d/(k+'.npy')).open('rb') as f: os.fsync(f.fileno())
    elapsed=time.perf_counter()-scan; io1=io_counters(); n=len(ppos)
    stats=dict(selected_train_parents=n,selected_native_species_bytes=n*8000,seconds_through_fsync=elapsed,
               parents_per_second=n/elapsed,logical_payload_MB_per_second=n*64*420/1e6/elapsed,
               proc_io_delta_bytes={k:io1[k]-io0[k] for k in io0},VmHWM_KiB=rss(),seconds_with_pins=time.perf_counter()-start)
    common=dict(index=i,beta=src['beta'],source='archived-physical',source_archive_bytes=src['npz_bytes'],source_sha256=src['npz_sha256'],
                release_commit=rev,release_sha256=relsha,split_sha256=release['split_sha256'],rosters_sha256=release['rosters_sha256'],
                train_only=True,test_species_rows_returned=0,incremental_fee_usd=0)
    if first:
        savej(aux,dict(**common,native_pilot_rows_sha256=digest(rows[PILOT]),pilot_raw_sha256={k:digest(v[PILOT]) for k,v in arr.items()},stats=stats))
    else:
        fold={}
        for k in KEYS:
            a=np.load(d/(k+'.npy'),mmap_mode='r',allow_pickle=False)
            for offset in range(8):
                hv=hashlib.sha256(); sub=a[offset::8]
                for lo in range(0,len(sub),32): hv.update(np.ascontiguousarray(sub[lo:lo+32]))
                fold.setdefault(str(offset),{})[k]=hv.hexdigest()
        fm=file_manifest(d); rec=dict(**common,stage='train',roles={'train':dict(parents=8192,opportunities=8192*64,files=fm)},
                train_identity_sha256=s['train_identity_sha256'],native_ordered_train_row_sha256=digest(rows),fold_offsets_sha256=fold,
                pilot_subset_receipt_sha256=sha(aux),pilot_projection=prev['stats'],complement_projection=stats,
                archive_scans=2,unique_train_species_returned=8192,rank_repeats=0,output_bytes=sum(x['bytes'] for x in fm.values()),
                utc=datetime.now(timezone.utc).isoformat(),total_seconds_per_beta=time.perf_counter()-start+prev['stats']['seconds_with_pins'],VmHWM_KiB=rss())
        cutoff('fit-dev'); savej(final,rec)
    print(json.dumps(dict(index=i,pass_name='pilot' if first else 'remainder',**stats)),flush=True)


def packet(rev):
    release,split=gate(rev); relsha=sha(PUB/'stream-release-1.json'); n=640
    data=dict(V=np.empty((n,64,27),'u1'),E=np.empty((n,64,386),'u1'),parent=np.empty(n,'<i4'),beta=np.empty(n,'<i2'))
    pins={}
    for i in INDICES:
        rec=readj(partial(i)); rows=train_rows(split,i); sl=slice(64*i,64*(i+1)); pins[i]=sha(partial(i))
        require(rec['index']==i and rec['release_sha256']==relsha and rec['native_pilot_rows_sha256']==digest(rows[PILOT]),'wrong pilot role')
        for k in ('V','E'):
            a=np.load(folder(i)/(k+'.npy'),mmap_mode='r',allow_pickle=False)[PILOT]
            require(digest(a)==rec['pilot_raw_sha256'][k],'pilot projection drift'); data[k][sl]=a
        data['parent'][sl]=rows[PILOT]; data['beta'][sl]=i
    p=STREAMS/'packet.npz'; raw=sum(a.nbytes for a in data.values())
    with p.open('xb') as f: np.savez(f,**data); f.flush(); os.fsync(f.fileno())
    require(max(raw,p.stat().st_size)<=64*1024**2, 'bounded pilot cap')
    md=dict(schema='hero-full-v3-fit-pilot-1',role='fit',source='archived-physical',train_only=True,fit_only=True,all_ten=True,
            train_folds=[2,3,4,5,6,7],beta_indices=list(INDICES),packet_sha256=sha(p),release_sha256=relsha,
            split_sha256=release['split_sha256'],rosters_sha256=release['rosters_sha256'],parents_per_beta={i:64 for i in INDICES},
            anchors_per_parent=64,partial_receipts_sha256=pins,raw_bytes=raw,stored_bytes=p.stat().st_size,test_species_rows_returned=0)
    savej(p.with_suffix('.json'),md); savej(PUB/'pilot-packet.json',md); print(json.dumps(md),flush=True)


def summary(rev):
    release,_=gate(rev); recs={}; pins={}
    for i in INDICES:
        p=PUB/f'stream-b{i:02d}.json'; r=readj(p); pins[i]=sha(p)
        require(r['unique_train_species_returned']==8192 and r['release_sha256']==sha(PUB/'stream-release-1.json'),'missing or mismatched source')
        recs[i]={k:r[k] for k in ('beta','output_bytes','total_seconds_per_beta','VmHWM_KiB','source_archive_bytes','pilot_projection','complement_projection')}
    savej(PUB/'stream-summary.json',dict(source='archived-physical',stage='train',beta_indices=list(INDICES),train_parents=81920,
            opportunities=81920*64,bytes=sum(r['output_bytes'] for r in recs.values()),per_beta=recs,stream_receipts_sha256=pins,
            split_sha256=release['split_sha256'],rosters_sha256=release['rosters_sha256'],release_sha256=sha(PUB/'stream-release-1.json'),
            packet_sha256=readj(PUB/'pilot-packet.json')['packet_sha256'],test_species_rows_returned=0,incremental_fee_usd=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('mode',choices=('pilot','remainder','packet','summary'))
    ap.add_argument('--index',type=int,choices=INDICES); ap.add_argument('--release-commit',required=True); a=ap.parse_args()
    if a.mode=='packet': packet(a.release_commit)
    elif a.mode=='summary': summary(a.release_commit)
    else: projection(a.index,a.release_commit,a.mode=='pilot')
