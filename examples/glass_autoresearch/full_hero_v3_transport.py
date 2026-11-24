"""Standalone v3 TRAIN-only ephemeral transport adapter. No cloud or TEST access.

Unpack E2 only after taking the permitted TRAIN-fold view. No arrays from other
studies are used. One compact shared input layout for all training roles.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zlib
import numpy as np

from full_hero_transport import h, header, need, pack2, regular, serialized

B = tuple(range(10)); FIELDS = ('V','E2','parent'); PART = 256; N = 8192
CAP = 2*1024**3; SHARD_CAP = 8*1024**2; SHAPES = {'V':(N,64,27),'E2':(N,64,97),'parent':(N,)}
ENCODING = 'E2-le2bit-v1'


def unpack_e(packed):
    """Expand caller-selected TRAIN E2 rows/anchors, low two bits are first site."""
    x=np.asarray(packed); need(x.dtype==np.uint8 and x.shape[-1]==97 and not np.any(x[...,-1] >> 4), 'invalid E2 layout')
    e=np.empty((*x.shape[:-1],386),'u1')
    for j in range(4):
        view=e[...,j::4]; view[:]=(x[...,:view.shape[-1]] >> (2*j)) & 3
    need(not np.any(e==3),'invalid ternary species'); return e


def name(i,j): return f'b{i:02d}-train-{j:03d}.z'


def unpack_shard(blob):
    n=PART; length=n*(64*(27+97)+4); d=zlib.decompressobj(); raw=d.decompress(blob,length+1)
    need(d.eof and not d.unused_data and not d.unconsumed_tail and len(raw)==length,'bounded frame mismatch')
    vlen=n*64*27; elen=n*64*97
    v=np.frombuffer(raw,'u1',vlen).reshape(n,64,27); e2=np.frombuffer(raw,'u1',elen,vlen).reshape(n,64,97)
    return dict(V=v,E2=e2,parent=np.frombuffer(raw,'<i4',n,vlen+elen))


def write_bank(md,src,out,i):
    destination=out/f'b{i:02d}'/'train'; destination.mkdir(parents=True,exist_ok=False)
    sh=md['per_beta'][str(i)]['shards']; need(len(sh)==N//PART,'incomplete TRAIN bank')
    files=md['input_manifest']['files']; hashes={}
    with ExitStack() as st:
        sinks={k:st.enter_context((destination/(k+'.npy')).open('xb')) for k in FIELDS}
        for k in FIELDS:
            head=bytes.fromhex(md['headers_hex'][k]); need(head==header(k,SHAPES[k]),'invalid array header')
            hashes[k]=hashlib.sha256(head); sinks[k].write(head)
        for j,item in enumerate(sh):
            n=name(i,j); need(item['name']==n and item['start']==j*PART,'unexpected shard/order')
            blob=regular(src/n,SHARD_CAP); need(len(blob)==item['bytes'] and h(blob)==item['sha256'],'corrupt TRAIN shard')
            arrays=unpack_shard(blob)
            for k in FIELDS: hashes[k].update(arrays[k]); sinks[k].write(arrays[k])
        for k in FIELDS:
            need(hashes[k].hexdigest()==files[f'b{i:02d}/train/{k}.npy'],'input NPY pin'); sinks[k].flush(); os.fsync(sinks[k].fileno())


def restore(manifest,sha256,shards,output,drop=False):
    raw=regular(manifest,1024**2); need(h(raw)==sha256,'unregistered transport')
    md=json.loads(raw); inp=md['input_manifest']; need(md['schema']=='hero-full-v3-offline-transport-1' and
        inp['schema']=='hero-full-v3-offline-training-input-1' and inp['beta_indices']==list(B) and inp['encoding']==ENCODING
        and inp['train_only'] is True and h(serialized(inp))==md['input_manifest_sha256'],'wrong v3 source roles')
    need(set(inp['files'])=={f'b{i:02d}/train/{k}.npy' for i in B for k in FIELDS} and
         set(md['per_beta'])==set(map(str,B)) and set(md['headers_hex'])==set(FIELDS),'unexpected input manifest')
    names={name(i,j) for i in B for j in range(N//PART)}
    need({p.name for p in shards.iterdir()}==names and sum(s['bytes'] for v in md['per_beta'].values() for s in v['shards'])==md['stored_bytes']<=CAP
         and md['restored_npy_bytes']==10*(N*(64*124+4)+sum(len(header(k,SHAPES[k])) for k in FIELDS))<=CAP, 'storage bound')
    output.mkdir(parents=True,exist_ok=False)
    for i in B: write_bank(md,shards,output,i)
    with (output/'manifest.json').open('xb') as f: f.write(serialized(inp)); f.flush(); os.fsync(f.fileno())
    if drop:
        for n in names: (shards/n).unlink()
    return md['input_manifest_sha256']


def prepare(rev,receipts_rev):
    import full_hero_v3_stream as s
    rel,split=s.gate(rev); start=time.perf_counter(); sm=s.PUB/'stream-summary.json'; s.committed(sm,receipts_rev); summary=s.readj(sm)
    op=s.WORK/'offline-transport'/'shards'; op.mkdir(parents=True,exist_ok=False); files={}; per_beta={}; stored=0
    head={k:header(k,SHAPES[k]) for k in FIELDS}; restored=10*(N*(64*124+4)+sum(map(len,head.values())))
    max_shard=0; readbytes=0
    for i in B:
        receipt=s.PUB/f'stream-b{i:02d}.json'; s.committed(receipt,receipts_rev); r=s.readj(receipt)
        need(s.sha(receipt)==summary['stream_receipts_sha256'][str(i)] and r['release_sha256']==summary['release_sha256'] and
             r['split_sha256']==rel['split_sha256'] and r['stage']=='train' and r['unique_train_species_returned']==N,'unreleased TRAIN bank')
        banks={k:np.load(s.folder(i)/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('V','E','parent')}
        need(np.array_equal(banks['parent'],s.train_rows(split,i)), 'TRAIN ranks drifted')
        pins={k:hashlib.sha256(head[k]) for k in FIELDS}; source={k:hashlib.sha256(header(k,a.shape)) for k,a in banks.items()}; parts=[]
        for j in range(N//PART):
            sl=slice(j*PART,(j+1)*PART); x={k:a[sl] for k,a in banks.items()}
            for k,a in x.items(): source[k].update(a); readbytes+=a.nbytes
            encoded=pack2(x['E']); blob=zlib.compress(b''.join((x['V'].tobytes(),encoded.tobytes(),x['parent'].tobytes())),1)
            need(len(blob)<=SHARD_CAP,'per-shard cap'); obj=unpack_shard(blob)
            need(np.array_equal(obj['V'],x['V']) and np.array_equal(unpack_e(obj['E2']),x['E']) and
                 np.array_equal(obj['parent'],x['parent']),'transport not lossless')
            for k in FIELDS: pins[k].update(obj[k])
            n=name(i,j)
            with (op/n).open('xb') as f: f.write(blob); f.flush(); os.fsync(f.fileno())
            parts.append(dict(name=n,start=j*PART,parents=PART,bytes=len(blob),sha256=h(blob))); stored+=len(blob); max_shard=max(max_shard,len(blob))
        for k in source: need(source[k].hexdigest()==r['roles']['train']['files'][k+'.npy']['sha256'],'native projection changed')
        for k in FIELDS: files[f'b{i:02d}/train/{k}.npy']=pins[k].hexdigest()
        per_beta[str(i)]=dict(parents=N,source_native_row_sha256=r['native_ordered_train_row_sha256'],shards=parts)
        # Actual indexed staging; no scoring, training or DEV/TEST access.
        rng=np.random.default_rng([26112507,i]); allowed=np.flatnonzero(np.arange(N)%8>=2)
        copied=0; t0=time.perf_counter()
        for _ in range(256):
            p=np.stack([rng.choice(allowed,8,replace=False) for _ in range(8)]); a=rng.integers(64,size=p.shape)
            copied+=banks['V'][p,a].nbytes+banks['E'][p,a].nbytes+banks['parent'][p].nbytes
        dt=time.perf_counter()-t0
        per_beta[str(i)]['screen_fit_cpu_staging']=dict(source='archived TRAIN folds 2..7',batches=256,shape=[8,8],
            seconds=dt,batches_per_second=256/dt,bytes_copied=copied,MB_per_second=copied/dt/1e6,no_device_transfer=True)
    need(max(stored,restored)<=CAP,'aggregate cap')
    inp=dict(schema='hero-full-v3-offline-training-input-1',encoding=ENCODING,train_only=True,beta_indices=list(B),
        rosters_sha256=rel['rosters_sha256'],split_sha256=rel['split_sha256'],stream_receipts=summary['stream_receipts_sha256'],files=files)
    elapsed=time.perf_counter()-start
    md=dict(schema='hero-full-v3-offline-transport-1',source='archived-physical',stage='offline TRAIN shipment preparation only',
        input_manifest=inp,input_manifest_sha256=h(serialized(inp)),headers_hex={k:b.hex() for k,b in head.items()},per_beta=per_beta,
        stored_bytes=stored,restored_npy_bytes=restored,source_payload_bytes=readbytes,shards=10*N//PART,largest_shard_bytes=max_shard,
        max_shard_bytes=SHARD_CAP,max_stored_bytes=CAP,max_restored_bytes=CAP,test_species_rows_returned=0,incremental_fee_usd=0,
        release_sha256=summary['release_sha256'],source_summary_sha256=s.sha(sm),source_receipts_commit=receipts_rev,
        encode_all_read_hash_round_trip_fsync_seconds=elapsed,source_MB_per_second=readbytes/elapsed/1e6,
        screening='Filter TRAIN ranks modulo 8 before E2 expansion; candidate fit 2..7, selection 0; rule diagnosis 1 only per separate authorization; final refit all TRAIN.',
        current_egress_authorized_bytes=0,future_main_aggregate_result_cap=64*1024**2,pilot_result_cap=128*1024,
        cold_only='Never a launch/spending approval. No stdout, volume, snapshot, additional data or labels; allowlisted regular outputs with size/SHA only under separate future Main release.')
    with tempfile.TemporaryDirectory(prefix='v3-restore-fixture-') as tmp:
        write_bank(md,op,Path(tmp),6)
        for k in FIELDS: need(s.sha(Path(tmp)/'b06'/'train'/(k+'.npy'))==files['b06/train/'+k+'.npy'],'nonreproducible restoration')
    md['complete_beta_disk_restore_check']=6
    target=s.PUB/'offline-transport.json'; s.savej(target,md)
    print(json.dumps(dict(manifest_sha256=s.sha(target),input_manifest_sha256=md['input_manifest_sha256'],stored=stored,
        restored=restored,shards=md['shards'],seconds=elapsed)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='task',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--release-commit',required=True); a.add_argument('--receipts-commit',required=True)
    a=sub.add_parser('restore'); a.add_argument('--manifest',type=Path,required=True); a.add_argument('--sha256',required=True)
    a.add_argument('--shards',type=Path,required=True); a.add_argument('--output',type=Path,required=True); a.add_argument('--drop',action='store_true')
    a=p.parse_args()
    if a.task=='prepare': prepare(a.release_commit,a.receipts_commit)
    else: print(restore(a.manifest,a.sha256,a.shards,a.output,a.drop))
