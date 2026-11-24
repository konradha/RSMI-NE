"""Offline encode/restore only. No network, credentials, volumes, cloud or trainer.

Standalone restore dependencies: Python stdlib + NumPy. Each shard carries contiguous
roster slices; only V/E lossless two-bit species and native parent IDs. The caller pins
the transport manifest and separately admits the immutable training release. Restore
only to a fresh ephemeral root; only manifest.json denotes a complete verified bank.
"""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import zlib

import numpy as np

INDICES = (0,1,2,3,4,5,7,8,9)
ROLES = ('fit','dev_even')
KEYS = ('V','E','parent')
PART = 256
SHARD_MAX = 8 * 1024**2
TOTAL_MAX = 2 * 1024**3
JSON_MAX = 1024**2


def need(test, msg):
    if not test: raise ValueError(msg)


def h(data):
    return hashlib.sha256(data).hexdigest()


def serialized(d):
    return (json.dumps(d, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('ascii')


def shapes(n):
    return {'V':(n,64,27), 'E':(n,64,386), 'parent':(n,)}


def header(k, shape):
    buf = io.BytesIO(); dt = np.dtype('<i4' if k == 'parent' else 'u1')
    np.lib.format.write_array_header_1_0(buf,dict(descr=dt.str,fortran_order=False,shape=shape))
    return buf.getvalue()


def pack2(a):
    w = a.shape[-1]; need(a.dtype == np.uint8 and int(a.max()) <= 2,'invalid ternary input')
    out = np.array(a[...,::4], order='C', copy=True)
    for j in range(1,4):
        sub = a[...,j::4]; out[...,:sub.shape[-1]] |= sub << (2*j)
    return out


def decode(blob, n):
    # The expected decoded size is fixed by n, independent of untrusted zlib headers.
    need(1 <= n <= PART and len(blob) <= SHARD_MAX,'shard bound')
    size = n*(64*(7+97)+4); z = zlib.decompressobj()
    raw = z.decompress(blob,size+1)
    need(len(raw) == size and z.eof and not z.unconsumed_tail and not z.unused_data,'invalid bounded zlib frame')
    out = {}; at = 0
    for key, width in (('V',27),('E',386)):
        packed = (width+3)//4; count = n*64*packed
        a = np.frombuffer(raw, np.uint8, count, at).reshape(n,64,packed); at += count
        v = np.empty((n,64,width),'u1')
        for j in range(4):
            s = v[...,j::4]; s[:] = (a[...,:s.shape[-1]] >> (2*j)) & 3
        need(int(v.max()) <= 2 and not np.any(a[...,-1] >> (2*(width%4))), 'invalid ternary or padding')
        out[key] = v
    out['parent'] = np.frombuffer(raw, '<i4', n, at)
    return out


def regular(path, size_limit):
    st = path.lstat()
    need(stat.S_ISREG(st.st_mode) and st.st_size <= size_limit, 'nonregular or oversized input '+path.name)
    return path.read_bytes()


def shard_name(idx, role, j):
    return f'b{idx:02d}-{role}-{j:03d}.z'


def write_role(directory, out, idx, role, role_meta, expected):
    n = role_meta['parents']; need(type(n) is int and 0 < n <= 10240,'invalid parent count')
    shape = shapes(n); shards = role_meta['shards']; need(len(shards) == (n+PART-1)//PART, 'coverage gap')
    hashes = {k:hashlib.sha256() for k in KEYS}; prefix = f'b{idx:02d}/{role}/'
    target = out/prefix; target.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        f = {k:stack.enter_context((target/(k+'.npy')).open('xb')) for k in KEYS}
        for k in KEYS:
            b = bytes.fromhex(role_meta['header_hex'][k]); reader = io.BytesIO(b)
            need(len(b) <= 512 and np.lib.format.read_magic(reader) == (1,0), 'invalid pinned NPY header')
            s, order, dtype = np.lib.format.read_array_header_1_0(reader)
            need(s == shape[k] and not order and dtype.str == ('<i4' if k == 'parent' else '|u1') and
                 reader.tell() == len(b), 'unexpected NPY header')
            hashes[k].update(b); f[k].write(b)
        for j, item in enumerate(shards):
            m = min(PART,n-j*PART); name = shard_name(idx,role,j)
            need(item['name'] == name and item['start'] == j*PART and item['parents'] == m,'shard substitution')
            compressed = regular(directory/name, SHARD_MAX)
            need(len(compressed) == item['bytes'] and h(compressed) == item['sha256'],'shard changed '+name)
            decoded = decode(compressed,m)
            for k in KEYS: hashes[k].update(decoded[k]); f[k].write(decoded[k])
        for k in KEYS:
            need(hashes[k].hexdigest() == expected[prefix+k+'.npy'],'restored array drift '+prefix+k)
            f[k].flush(); os.fsync(f[k].fileno())


def restore(manifest_path, manifest_sha256, shards, output, drop=False):
    """Trusted receiver. No network access. drop deletes only named verified transport copies."""
    blob = regular(manifest_path,JSON_MAX); need(h(blob) == manifest_sha256, 'unpinned transport'); md = json.loads(blob)
    need(md['schema'] == 'hero-full-v2-offline-transport-1' and md['beta_indices'] == list(INDICES), 'wrong transport')
    inp = md['input_manifest']; contents = serialized(inp)
    need(h(contents) == md['input_manifest_sha256'] and inp['schema'] == 'hero-full-v2-offline-training-input-1', 'wrong input manifest')
    expected = inp['files']; need(set(expected) == {f'b{i:02d}/{r}/{k}.npy' for i in INDICES for r in ROLES for k in KEYS},'unexpected role/file')
    need(set(md['roles']) == {f'b{i:02d}/{r}' for i in INDICES for r in ROLES},'unexpected roles')
    total_stored = total_raw = 0; names = []
    for i in INDICES:
        for r in ROLES:
            rm = md['roles'][f'b{i:02d}/{r}']; n = rm['parents']; need(type(n) is int and 0 < n <= 10240,'parent bound')
            total_raw += n * (64*413 + 4) + sum(len(bytes.fromhex(v)) for v in rm['header_hex'].values())
            for sh in rm['shards']:
                need(0 < sh['bytes'] <= SHARD_MAX,'shard exceeds cap'); total_stored += sh['bytes']; names.append(sh['name'])
    need(total_raw == md['restored_npy_bytes'] <= TOTAL_MAX and total_stored == md['stored_shard_bytes'] <= TOTAL_MAX and len(names) == len(set(names)), 'aggregate bound')
    need(set(p.name for p in shards.iterdir()) == set(names), 'unexpected/missing files in shard-only directory')
    output.mkdir(exist_ok=False,parents=True)
    for i in INDICES:
        for r in ROLES: write_role(shards,output,i,r,md['roles'][f'b{i:02d}/{r}'],expected)
    with (output/'manifest.json').open('xb') as f: f.write(contents); f.flush(); os.fsync(f.fileno())
    if drop:
        for name in names: (shards/name).unlink()
    return md['input_manifest_sha256']


def prepare(release_commit, receipts_commit):
    import full_hero_stream as s
    s.authorized('fit-dev',INDICES[0],release_commit)
    start = time.perf_counter(); summary_file = s.PUB/'stream-summary.json'; s.committed(summary_file,receipts_commit)
    summary = s.readj(summary_file); out = s.WORK/'offline-transport'/'shards'; out.mkdir(parents=True,exist_ok=False)
    receipt_path = s.PUB/'offline-transport.json'; need(not receipt_path.exists(),'transport already registered')
    total_stored = total_raw = 0; roles = {}; files = {}; biggest = 0; bytes_hashed = 0
    for idx in INDICES:
        receipt_path_beta = s.PUB/f'stream-b{idx:02d}.json'; s.committed(receipt_path_beta,receipts_commit)
        receipt = s.readj(receipt_path_beta); ordered = s.read_roles(idx)
        need(s.sha(receipt_path_beta) == summary['stream_receipts_sha256'][str(idx)] and
             receipt['rosters_sha256'] == summary['rosters_sha256'] and receipt['source'] == 'archived-physical'
             and receipt['stage'] == 'fit-dev' and receipt['release_sha256'] == summary['release_sha256'], 'mixed sources')
        for role in ROLES:
            isdev = role == 'dev_even'; src_role = 'dev' if isdev else 'fit'; prefix = f'b{idx:02d}/{role}'
            bank = s.FitInput(s.STREAMS/f'b{idx:02d}'/src_role, dev=isdev); n = len(bank.parent)
            rows = ordered[src_role][::2] if isdev else ordered[src_role]; need(np.array_equal(bank.parent,rows), 'wrong role order')
            headers = {k:header(k,shapes(n)[k]) for k in KEYS}
            hashes = {k:hashlib.sha256(headers[k]) for k in KEYS}; raw_h = {k:hashlib.sha256() for k in KEYS}
            rm = dict(parents=n, source_role=src_role, source_slice='0::2' if isdev else 'all',
                source_native_row_i8le_sha256=h(np.ascontiguousarray(rows,dtype='<i8')),
                header_hex={k:v.hex() for k,v in headers.items()}, shards=[]); roles[prefix] = rm
            total_raw += sum(len(v) for v in headers.values()) + n*(64*413+4)
            for j, at in enumerate(range(0,n,PART)):
                end = min(n,at+PART); arrays = {k:np.ascontiguousarray(getattr(bank,k)[at:end]) for k in KEYS}
                packed = b''.join((pack2(arrays['V']).tobytes(), pack2(arrays['E']).tobytes(), arrays['parent'].tobytes()))
                compressed = zlib.compress(packed,1); need(len(compressed) <= SHARD_MAX,'shard cap'); name = shard_name(idx,role,j)
                # Round trip EVERY emitted shard in bounded RAM and hash the actual decoder outputs, not an estimate.
                decoded = decode(compressed,end-at)
                for k in KEYS:
                    need(np.array_equal(arrays[k], decoded[k]), 'packing not lossless')
                    raw_h[k].update(arrays[k]); hashes[k].update(decoded[k]); bytes_hashed += decoded[k].nbytes
                with (out/name).open('xb') as f: f.write(compressed); f.flush(); os.fsync(f.fileno())
                rm['shards'].append(dict(name=name,start=at,parents=end-at,bytes=len(compressed),
                    raw_bytes=sum(a.nbytes for a in arrays.values()),sha256=h(compressed)))
                total_stored += len(compressed); biggest = max(biggest,len(compressed))
            for k in KEYS:
                value = hashes[k].hexdigest(); files[prefix+'/'+k+'.npy'] = value
                # Do not hash/reopen the unsliced DEV file: its odd physical entries are not part of transport.
                if isdev: need(raw_h[k].hexdigest() == receipt['roles']['dev']['even_sha256'][k], 'even-dev pin mismatch')
                else: need(value == receipt['roles']['fit']['files'][k+'.npy']['sha256'], 'full FIT pin mismatch')
    need(max(total_raw,total_stored) <= TOTAL_MAX,'transport aggregate cap')
    manifest = dict(schema='hero-full-v2-offline-training-input-1',rosters_sha256=summary['rosters_sha256'],
                    stream_receipts=summary['stream_receipts_sha256'], files=files)
    elapsed = time.perf_counter()-start
    md = dict(schema='hero-full-v2-offline-transport-1',source='archived-physical',beta_indices=list(INDICES),
        stage='local offline packing only; no launch or upload',roles=roles,input_manifest=manifest,
        input_manifest_sha256=h(serialized(manifest)),shards=len(list(out.iterdir())),stored_shard_bytes=total_stored,
        restored_npy_bytes=total_raw,restored_payload_bytes=bytes_hashed,largest_shard_bytes=biggest,
        max_compressed_shard_bytes=SHARD_MAX,max_stored_bytes=TOTAL_MAX,max_restored_npy_bytes=TOTAL_MAX,
        source_stream_summary_sha256=s.sha(summary_file),source_receipts_commit=receipts_commit,
        fit_dev_release_commit=release_commit,encode_hash_decode_and_fsync_seconds=elapsed,
        uncompressed_payload_MB_per_second=bytes_hashed/elapsed/1e6,one_thread=True,incremental_fee_usd=0,
        permitted_files='restore only V,E,parent per registered beta and fit/even-dev; no native, I, anchors, odd dev or eval',
        compression='zlib level 1 over packed V (7 bytes/anchor), packed E (97 bytes/anchor), raw little-endian int32 parent; first site in low two bits, pad zero',
        current_result_bytes_authorized=0,future_main_result_upper_bound_bytes=64*1024**2,
        pilot_result_upper_bound_bytes=128*1024,
        outputs='No execution/egress now. A separate future Main release must enumerate allowed regular checkpoint/receipt files, enforce per-file and aggregate 64MiB-or-tighter bounds, collect size and SHA; no stdout, snapshots or volumes. Existing pilot remains independently capped at 128 KiB.',
        offline_unpack='full_hero_transport.py restore --manifest FILE --sha256 DIGEST --shards EXCLUSIVE_DIR --output /tmp/hero-input --drop. Pin input_manifest_sha256 separately in training release.')
    # Exercise the file writer on one complete multishard role and the even-only tiny role. No full raw shadow bank.
    with tempfile.TemporaryDirectory(prefix='hero-restore-control-') as tmp:
        for i,r in ((0,'fit'),(9,'dev_even')):
            prefix = f'b{i:02d}/{r}'; write_role(out,Path(tmp),i,r,roles[prefix],files)
            for k in KEYS: need(s.sha(Path(tmp)/prefix/(k+'.npy')) == files[prefix+'/'+k+'.npy'],'failed on-disk restoration')
    md['selective_disk_restore_validated'] = ['b00/fit','b09/dev_even']; need(len(serialized(md)) <= JSON_MAX,'manifest cap')
    s.savej(receipt_path,md)
    print(json.dumps(dict(transport_sha256=s.sha(receipt_path),input_manifest_sha256=md['input_manifest_sha256'],
           shards=md['shards'],stored=total_stored,restored=total_raw,peak=biggest,seconds=elapsed)),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='action',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--release-commit',required=True); p.add_argument('--receipts-commit',required=True)
    p = sub.add_parser('restore'); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--sha256',required=True)
    p.add_argument('--shards',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--drop',action='store_true')
    a = ap.parse_args()
    if a.action == 'prepare': prepare(a.release_commit,a.receipts_commit)
    else: print(restore(a.manifest,a.sha256,a.shards,a.output,a.drop),flush=True)
