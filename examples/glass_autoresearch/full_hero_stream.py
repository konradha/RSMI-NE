"""HERO v2: project selected native TRAIN rows once; never open an eval role implicitly.

One union-sorted forward ZIP walk per beta for fit+dev (deflate may internally skip
unselected bytes). The full compressed-source SHA is a separate integrity read, not
a second decode. No decoded whole-archive cache. Parent order on disk is exactly
roster order, independent of I/O order. Only this module reads native species.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import tempfile
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from hero_bank import (CENTER, DATA, DISCOVERY_STOP, GRID, HARD_STOP, INDEX_SHA,
                       INTERIOR, ROOT, SHELL, committed, coords, exterior_inventory,
                       flatten, readj, require, sha, sites)

PUB = ROOT / 'results/l20-autoresearch/hero-full-v2'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/hero-full-v2'
STREAMS = WORK / 'streams'
INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9)
FIELDS = {'V': ('u1', (27,)), 'E': ('u1', (386,)), 'I': ('<u2', (2,)),
          'anchor': ('u1', (3,))}
FIELDS_BY_STAGE = {'fit-dev': ('V', 'E', 'I', 'anchor'), 'blind': ('E', 'I', 'anchor'),
                   'hidden': ('V', 'anchor')}


def savej(path, data):
    # Readers can use receipt presence as a completion barrier. Never replace a ticket.
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.link(tmp, path); tmp.unlink()


def cutoff(stage):
    bound = datetime.fromisoformat(DISCOVERY_STOP if stage == 'fit-dev' else HARD_STOP)
    require(datetime.now(timezone.utc) < bound, 'native archive cutoff passed')


def anchors(idx, row, discovery):
    rng = np.random.Generator(np.random.PCG64([26112502, idx, int(row)]))
    if discovery:
        return rng.choice(8000, 64, replace=False).astype('<u2')
    return flatten(GRID + rng.integers(10, size=3))


def lookup():
    # 12 MB, amortized across all parents; no temporary 8000 x 386 x 3 coordinate tensor.
    tables = []
    for off in (CENTER, SHELL, INTERIOR):
        table = np.empty((8000, len(off)), dtype='<u2')
        for start in range(0, 8000, 256):
            stop = min(8000, start + 256)
            table[start:stop] = sites(np.arange(start, stop), off)
        tables.append(table)
    return tables


class RowSink:
    """Positioned writes bound RSS even when native and registered orders differ."""
    def __init__(self, stack, path, dtype, shape):
        self.fd = stack.enter_context(path.open('xb', buffering=0))
        self.dtype = np.dtype(dtype)
        np.lib.format.write_array_header_1_0(self.fd, dict(descr=self.dtype.str,
                    fortran_order=False, shape=shape))
        self.base = self.fd.tell()
        self.block = self.dtype.itemsize * int(np.prod(shape[1:]))
        self.fd.truncate(self.base + self.block * shape[0])

    def put(self, p, a):
        a = np.ascontiguousarray(a, dtype=self.dtype)
        require(a.nbytes == self.block and os.pwrite(self.fd.fileno(), a, self.base + p * self.block)
                == self.block, 'short write or wrong row size')


def io_counters():
    return {key: int(val) for key, val in (line.split(':') for line in
            Path('/proc/self/io').read_text().splitlines()) if key in ('read_bytes', 'write_bytes')}


def file_manifest(directory):
    return {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(directory.glob('*.npy'))}


def project(srcfile, out, idx, roles, stage, tables, is_native):
    """Internal: explicit rows supplied *after* authorization; no fallback source/fold."""
    start = time.perf_counter(); disc = stage == 'fit-dev'; q = 64 if disc else 8
    require(set(roles) == ({'fit', 'dev'} if disc else {stage}), 'mixing stages')
    require(idx in INDICES, 'index 6 unavailable')
    selected = sorted((int(r), role, j) for role, rows in roles.items() for j, r in enumerate(rows))
    require(len(selected) and all(0 <= r < 16384 for r, _, _ in selected)
            and len({r for r, _, _ in selected}) == len(selected), 'invalid/overlapping selected rows')
    count = len(selected); io0 = io_counters(); observed = 0
    v_ix, e_ix, i_ix = tables
    with ExitStack() as stack:
        sinks = {}
        for role, rows in roles.items():
            require(len(rows) > 0, 'empty role')
            folder = out / role; folder.mkdir(parents=True, exist_ok=False)
            with (folder/'parent.npy').open('xb') as f:
                np.save(f, np.asarray(rows, dtype='<i4'), allow_pickle=False); f.flush(); os.fsync(f.fileno())
            sinks[role] = {k: RowSink(stack, folder/(k+'.npy'), FIELDS[k][0],
                        (len(rows), q, *FIELDS[k][1])) for k in FIELDS_BY_STAGE[stage]}
        z = stack.enter_context(ZipFile(srcfile)); info = z.getinfo('species.npy')
        f = stack.enter_context(z.open(info))
        ver = np.lib.format.read_magic(f); require(ver in ((1, 0), (2, 0)), 'bad NPY magic')
        hdr = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = hdr(f)
        require(shape[1:] == (20,20,20) and (not is_native or shape[0] == 16384)
                and selected[-1][0] < shape[0] and not fortran and dtype == np.uint8, 'bad source header')
        base = f.tell()
        for at, (r, role, p) in enumerate(selected):
            if is_native and at % 64 == 0: cutoff(stage)
            address = base + 8000 * r; a = anchors(idx, r, disc); dest = sinks[role]
            dest['anchor'].put(p, coords(a))
            if stage == 'hidden':
                # Export only registered center sites. Do not return a decoded whole row.
                ix = v_ix[a].ravel(); order = np.argsort(ix); ordered = ix[order]
                v = np.empty(ix.size, dtype='u1'); cuts = np.flatnonzero(np.diff(ordered) != 1) + 1
                for run in np.split(np.arange(len(order)), cuts):
                    addr = address + int(ordered[run[0]]); require(f.tell() <= addr, 'backward seek')
                    f.seek(addr); b = f.read(len(run)); require(len(b) == len(run), 'short center span')
                    v[order[run]] = np.frombuffer(b, 'u1'); observed += len(b)
                require(v.max() <= 2, 'invalid V alphabet'); dest['V'].put(p, v)
                continue
            require(f.tell() <= address, 'backward archive seek'); f.seek(address)
            row = f.read(8000); require(len(row) == 8000, 'short authorized row'); observed += len(row)
            state = np.frombuffer(row, np.uint8); e = state[e_ix[a]]
            require(e.max() <= 2, 'invalid E alphabet'); dest['E'].put(p, e)
            if disc:
                require(row.count(b'\x00') == 2000 and row.count(b'\x01') == 2400
                        and row.count(b'\x02') == 3600, 'source composition drift')
                dest['V'].put(p, state[v_ix[a]])
                core = state[i_ix[a]]
                inventory = np.column_stack(((core == 0).sum(-1), (core == 1).sum(-1)))
            else:
                # As in the preregistered pilot, only histograms of exterior runs,
                # never a microscopic cavity array or a center label in this stage.
                inventory = np.asarray([exterior_inventory(row, i_ix[anchor]) for anchor in a])
                require((inventory >= 0).all() and (inventory.sum(1) <= 343).all(), 'invalid aggregate')
            dest['I'].put(p, inventory)
        for dest in sinks.values():
            for sink in dest.values(): os.fsync(sink.fd.fileno())
    elapsed = time.perf_counter() - start
    roles_rec = {role: dict(parents=len(rows), opportunities=len(rows) * q,
                 files=file_manifest(out/role)) for role, rows in roles.items()}
    bytes_out = sum(m['bytes'] for r in roles_rec.values() for m in r['files'].values())
    io1 = io_counters()
    return dict(source='archived-physical' if is_native else 'synthetic-fixture', index=idx,
        stage=stage, roles=roles_rec, parents=count, opportunities=count*q,
        seconds_project_and_fsync=elapsed, seconds_through_output_hashes=time.perf_counter()-start,
        parents_per_second=count/elapsed, native_or_fixture_selected_bytes_returned=observed,
        species_member_compressed_bytes=info.compress_size, species_member_uncompressed_bytes=info.file_size,
        source_archive_bytes=srcfile.stat().st_size, output_bytes=bytes_out,
        logical_selected_input_MB_per_second=observed/1e6/elapsed,
        output_MB_per_second_including_fsync=bytes_out/1e6/elapsed,
        proc_io_delta_bytes_through_hashes={k:io1[k]-io0[k] for k in io0},
        process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        positive_only_seeks=True, species_member_open_count=1, incremental_fee_usd=0)


def sha_slice(folder, stride_start):
    out = {}
    for path in sorted(folder.glob('*.npy')):
        a = np.load(path, mmap_mode='r', allow_pickle=False)[stride_start::2]
        h = hashlib.sha256()
        for lo in range(0, len(a), 32):
            h.update(np.ascontiguousarray(a[lo:lo+32]))
        out[path.stem] = h.hexdigest()
    return out


class FitInput:
    """CPU/GPU staging: map only native row ids, V and E; no auxiliary predictors."""
    def __init__(self, folder, dev=False):
        self.parent, self.V, self.E = (np.load(folder / (k+'.npy'), mmap_mode='r',
                                      allow_pickle=False) for k in ('parent', 'V', 'E'))
        if dev:
            self.parent, self.V, self.E = self.parent[::2], self.V[::2], self.E[::2]
        n = len(self.parent)
        require(self.parent.shape == (n,) and self.parent.dtype == '<i4' and
                self.V.shape == (n,64,27) and self.E.shape == (n,64,386) and
                self.V.dtype == self.E.dtype == np.uint8, 'wrong staging layout')

    def gather(self, p, q):
        # Joining with (index,native row) is for coverage only, never encoder input.
        require(p.shape == q.shape and p.dtype.kind in 'iu' and q.dtype.kind in 'iu' and
                np.all((p >= 0) & (p < len(self.parent))) and np.all((q >= 0) & (q < 64)),
                'staging index outside registered slice')
        return self.V[p, q], self.E[p, q], self.parent[p]


def benchmark(folder):
    """Real copied batches, not an extrapolation from a single parent projection."""
    bank = FitInput(folder); batches = 256; rng = np.random.default_rng(26112503)
    group = min(len(bank.parent), 8); groups = 8; check = 0; copied = 0
    resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    for _ in range(batches):
        p = np.stack([rng.choice(len(bank.parent), group, replace=False) for _ in range(groups)])
        q = rng.integers(64, size=p.shape); v, e, parents = bank.gather(p, q)
        copied += v.nbytes + e.nbytes + parents.nbytes; check += int(parents[0,0])
    elapsed = time.perf_counter()-start
    return dict(stage='CPU fit mmap sampling + indexed gather; no training, no device transfer',
                batches=batches, groups=groups, distinct_parents_per_group=group,
                seconds=elapsed, batches_per_second=batches/elapsed,
                opportunities_per_second=batches*groups*group/elapsed, bytes_copied=copied,
                payload_MB_per_second=copied/elapsed/1e6, roster_sanity_checksum=check,
                peak_RSS_before_KiB=resident, process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def authorized(stage, idx, rev):
    cutoff(stage)
    require(stage in FIELDS_BY_STAGE and idx in INDICES, 'unpermitted native role/index')
    require(re.fullmatch(r'[0-9a-f]{40}', rev) is not None and
            subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',rev,'HEAD']).returncode == 0,
            'stage requires explicit immutable release revision')
    path = PUB / ('stream-release-1.json' if stage == 'fit-dev' else f'{stage}-release-b{idx:02d}.json')
    for p in (PUB/'design.json', PUB/'stream-design.json', PUB/'lineage.json', path,
              Path(__file__).resolve(), Path(__file__).with_name('hero_bank.py')):
        committed(p, rev)
    release = readj(path); lineage = readj(PUB/'lineage.json')
    require(release['stage'] == stage and release['design_sha256'] == sha(PUB/'design.json')
            and release['lineage_sha256'] == sha(PUB/'lineage.json')
            and release['rosters_sha256'] == lineage['rosters_sha256'] == sha(WORK/'rosters.npz'),
            'wrong design/lineage/roster authorization')
    require(sha(DATA/'release-index.json') == INDEX_SHA == lineage['source_manifest_sha256'],
            'native release manifest pin')
    if stage != 'fit-dev':
        require(release['index'] == idx, 'wrong eval index')
        pre = release['prerequisites']; require(isinstance(pre, dict) and pre, 'frozen prerequisites missing')
        for fname in ('rule_file', 'forecast_file'):
            require(release[fname] in pre, 'missing frozen artifact '+fname)
        for name, digest in pre.items():
            require(Path(name).name == name and name not in ('.','..') and sha(PUB/name) == digest,
                    'eval prerequisites changed'); committed(PUB/name,rev)
    return release, lineage, sha(path)


def read_roles(idx):
    from l20_pairs import partition
    require(idx in INDICES, 'forbidden index')
    with np.load(WORK/'rosters.npz', allow_pickle=False) as z:
        roles = {k:z[f'b{idx:02d}_{k}'] for k in ('fit','dev','eval')}
    all_rows = np.concatenate(list(roles.values()))
    require(all(v.ndim == 1 and v.dtype.kind in 'iu' and len(v) > 0 for v in roles.values())
            and len(np.unique(all_rows)) == len(all_rows)
            and np.isin(all_rows, partition(idx)[0][0]).all(), 'overlap/non-TRAIN/empty roster')
    ordered = readj(PUB/'lineage.json')['ordered_roster_i8le_sha256']
    for k, v in roles.items():
        require(hashlib.sha256(v.astype('<i8',copy=False).tobytes()).hexdigest() == ordered[f'b{idx:02d}_{k}'],
                'ordered native roster mismatch')
    return roles


def check_predictions(release, rev, rosters_sha):
    support = []
    for i in INDICES:
        pred = PUB/f'predictions-b{i:02d}.npz'; meta = pred.with_suffix('.json')
        blind = PUB/f'stream-blind-b{i:02d}.json'
        for p in (pred, meta, blind): committed(p, rev)
        br = readj(blind); md = readj(meta); rows = read_roles(i)['eval']; n = len(rows)
        require(md['index'] == br['index'] == i and md['stage'] == 'predictions'
                and br['stage'] == 'blind' and br['rosters_sha256'] == rosters_sha
                and br['roles']['blind']['parents'] == n and md['prediction_sha256'] == sha(pred)
                and md['blind_sha256'] == sha(blind), 'prediction/roster receipt mismatch')
        f = float(md['support_fraction']); require(np.isfinite(f) and 0 <= f <= 1, 'invalid prediction support')
        support.append(f)
        for key in ('rule','forecast'):
            require(md[key+'_sha256'] == br[key+'_sha256'] == release['prerequisites'][release[key+'_file']],
                    'inconsistent frozen '+key)
        a = np.stack([coords(anchors(i, r, False)).astype('u1') for r in rows])
        with np.load(pred, allow_pickle=False) as z:
            require({'parent','anchor','m0','m1'} <= set(z.files) and z['parent'].dtype.kind in 'iu'
                    and np.array_equal(z['parent'],rows) and z['anchor'].dtype == np.uint8
                    and np.array_equal(z['anchor'],a), 'predictions do not match all evaluation rows/anchors')
            for k in set(z.files)-{'parent','anchor'}:
                x = z[k]
                require(x.shape == (n,8) and x.dtype.kind == 'f' and np.isfinite(x).all(),
                        'incomplete signed prediction '+k)
    require(np.mean(support) >= .90, 'frozen macro forecast support gate failed')


def native(stage, idx, rev, tables):
    begin = time.perf_counter(); release, lock, release_sha = authorized(stage, idx, rev)
    rows = read_roles(idx)
    if stage == 'hidden': check_predictions(release, rev, lock['rosters_sha256'])
    roles = ({k:rows[k] for k in ('fit','dev')} if stage == 'fit-dev' else {stage:rows['eval']})
    source = next(s for s in readj(DATA/'release-index.json')['endpoints'] if s['release_index'] == idx)
    pin = lock['source_pins'][str(idx)]
    for k in ('npz', 'sidecar','npz_sha256','sidecar_sha256'):
        require(source[k] == pin[k], 'lineage source pin mismatch '+k)
    out = STREAMS/f'b{idx:02d}'; receipt = PUB / (f'stream-b{idx:02d}.json' if stage == 'fit-dev'
                                                   else f'stream-{stage}-b{idx:02d}.json')
    require(not receipt.exists() and not any((out/role).exists() for role in roles),
            'stream stage already started/spent')
    from hero_bank import exact_keys
    # Checks both exact compressed hashes and sidecar endpoint SHA, then only identity metadata.
    keys = exact_keys(source); used = np.concatenate(list(roles.values()))
    require(np.all(keys[used,0] == np.uint64(source['bulk_seed_uint64'])) and
            np.array_equal(keys[used,1], used - int(source['prep_seed_rows'])), 'native bulk identity drift')
    rec = project(DATA/source['npz'], out, idx, roles, stage, tables, True)
    rec.update(beta=source['beta'], release_commit=rev, release_sha256=release_sha,
               rosters_sha256=release['rosters_sha256'], source_sha256=source['npz_sha256'],
               sidecar_sha256=source['sidecar_sha256'], lineage_sha256=release['lineage_sha256'],
               utc=datetime.now(timezone.utc).isoformat())
    if stage == 'fit-dev':
        rec['roles']['dev'].update(even_sha256=sha_slice(out/'dev',0), odd_sha256=sha_slice(out/'dev',1))
        rec['batch_gather'] = benchmark(out/'fit')
    else:
        for k in ('rule','forecast'): rec[k+'_sha256'] = release['prerequisites'][release[k+'_file']]
    rec['seconds_end_to_end'] = time.perf_counter()-begin
    rec['process_peak_RSS_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cutoff(stage); savej(receipt, rec)
    print(json.dumps(dict(receipt=receipt.name, source=rec['source'], parents=rec['parents'],
              parents_per_second=rec['parents_per_second'], output_MB_per_second=rec['output_MB_per_second_including_fsync'],
              elapsed=rec['seconds_end_to_end'])), flush=True)
    return rec


def finish(rev):
    """No new archive reads. Packet and coverage require every admission, not a subset."""
    release, _, release_sha = authorized('fit-dev', INDICES[0], rev)
    path = STREAMS/'packet.npz'; meta = path.with_suffix('.json')
    require(not path.exists() and not meta.exists() and not (PUB/'stream-summary.json').exists(), 'already packed')
    records = {}; receipts = {}; layout = {}
    for i in INDICES:
        receipt = PUB/f'stream-b{i:02d}.json'; rec = readj(receipt); records[i] = rec; receipts[i] = sha(receipt)
        require(rec['stage'] == 'fit-dev' and rec['index'] == i and rec['source'] == 'archived-physical'
                and rec['release_sha256'] == release_sha and rec['rosters_sha256'] == release['rosters_sha256'],
                'mixed releases')
        rows = read_roles(i)
        for role in ('fit','dev'):
            require(rec['roles'][role]['parents'] == len(rows[role]), 'incomplete input')
        n = min(64,len(rows['fit'])); require(n >= 8, 'too few fit parents for within-beta pilot')
        layout[i] = n
    total = sum(layout.values())
    data = dict(V=np.empty((total,64,27),'u1'), E=np.empty((total,64,386),'u1'),
                parent=np.empty(total,'<i4'), beta=np.empty(total,'<i2'))
    raw_size = sum(a.nbytes for a in data.values()); limit = 64 * 1024**2
    require(raw_size < limit - 4096, 'packet exceeds byte ceiling')
    at = 0
    for i, n in layout.items():
        folder = STREAMS/f'b{i:02d}'/'fit'; rows = read_roles(i)['fit']
        for k in ('V','E','parent'):
            require(sha(folder/(k+'.npy')) == records[i]['roles']['fit']['files'][k+'.npy']['sha256'],
                    'fit input drift')
        bank = FitInput(folder); require(np.array_equal(bank.parent, rows), 'fit roster changed')
        sl = slice(at,at+n); data['V'][sl] = bank.V[:n]; data['E'][sl] = bank.E[:n]
        data['parent'][sl] = bank.parent[:n]; data['beta'][sl] = i; at += n
    with path.open('xb') as f:
        np.savez(f, **data); f.flush(); os.fsync(f.fileno())
    require(path.stat().st_size <= limit, 'stored packet exceeds ceiling')
    packet = dict(schema='hero-full-v2-fit-pilot-1', role='fit', source='archived-physical',
        packet_sha256=sha(path), release_sha256=release_sha, beta_indices=list(INDICES),
        index6_excluded=True, fit_only=True, stream_receipts_sha256=receipts,
        roster_sha256=release['rosters_sha256'], parents_per_beta=layout, anchors_per_parent=64,
        raw_bytes=raw_size, stored_bytes=path.stat().st_size)
    cutoff('fit-dev'); savej(meta,packet)
    seconds = sum(r['seconds_end_to_end'] for r in records.values())
    output = sum(r['output_bytes'] for r in records.values())
    summary = dict(source='archived-physical', stage='fit-dev', release_sha256=release_sha,
        rosters_sha256=release['rosters_sha256'], stream_receipts_sha256=receipts, packet=packet,
        per_beta={i:{k:r[k] for k in ('index','beta','parents','opportunities','output_bytes',
           'source_archive_bytes','native_or_fixture_selected_bytes_returned','seconds_project_and_fsync',
           'seconds_end_to_end','parents_per_second','output_MB_per_second_including_fsync',
           'proc_io_delta_bytes_through_hashes','process_peak_RSS_KiB','batch_gather')} for i,r in records.items()},
        parents_by_role={k:sum(r['roles'][k]['parents'] for r in records.values()) for k in ('fit','dev')},
        logical_output_bytes=output, archive_bytes=sum(r['source_archive_bytes'] for r in records.values()),
        selected_species_bytes=sum(r['native_or_fixture_selected_bytes_returned'] for r in records.values()),
        sum_end_to_end_seconds=seconds, max_process_RSS_KiB=max(r['process_peak_RSS_KiB'] for r in records.values()),
        physical_evaluation_rows_accessed=0, training=False, upload=False, incremental_fee_usd=0)
    savej(PUB/'stream-summary.json',summary); print(json.dumps(packet), flush=True)


def synthetic():
    """True compression + selected-row extraction + fsync + hash + indexed-batch control."""
    from hero_bank import geometry
    receipt = PUB/'stream-synthetic-control.json'
    require(not receipt.exists(), 'synthetic receipt already published')
    geom = geometry(); t = lookup(); size = 1632
    rng = np.random.default_rng(26112504); perm = rng.permutation(size)
    roles = dict(fit=perm[:1408], dev=perm[1408:1584]); exam = perm[1584:1592]
    template = np.repeat(np.arange(3,dtype='u1'), (2000,2400,3600))
    def row(r):
        return np.random.default_rng([26112505,int(r)]).permutation(template).reshape(20,20,20)
    deny = set(perm[1592:]); poison = b'\xff' * 8000
    with tempfile.TemporaryDirectory(prefix='hero-v2-synthetic-') as tmp:
        tmp = Path(tmp); physical = tmp/'fixture.npz'; out = tmp/'b00'
        with ZipFile(physical,'x',compression=ZIP_DEFLATED,compresslevel=1) as z, z.open('species.npy','w') as f:
            np.lib.format.write_array_header_1_0(f, dict(descr='|u1',fortran_order=False,shape=(size,20,20,20)))
            for r in range(size):
                f.write(poison if r in deny else row(r))
        rec = project(physical,out,0,roles,'fit-dev',t,False)
        require(set(p.name for p in out.iterdir()) == {'fit','dev'}, 'unrequested exam written')
        rec['batch_gather'] = benchmark(out/'fit')
        eval_modes = {s:project(physical,out,0,{s:exam},s,t,False) for s in ('blind','hidden')}
        for role, rows in {**roles,'blind':exam,'hidden':exam}.items():
            stage = 'fit-dev' if role in roles else role; folder = out/role
            fields = {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in folder.glob('*.npy')}
            require(set(fields) == set(FIELDS_BY_STAGE[stage]) | {'parent'} and
                    np.array_equal(fields['parent'],rows), 'role leak or order drift')
            for p in (0, len(rows)//2, len(rows)-1):
                r = int(rows[p]); src = row(r); a = fields['anchor'][p].astype(int)
                if stage == 'fit-dev':
                    g = np.random.Generator(np.random.PCG64([26112502,0,r]))
                    expected = np.column_stack(np.unravel_index(g.choice(8000,64,replace=False), (20,20,20)))
                    require(np.array_equal(a,expected), 'wrong seeded anchor layout')
                for k, off in (('V',CENTER),('E',SHELL),('I',INTERIOR)):
                    if k not in fields: continue
                    ix = (a[:,None,:] + off) % 20; cells = src[ix[:,:,0],ix[:,:,1],ix[:,:,2]]
                    ref = np.column_stack(((cells == 0).sum(1),(cells == 1).sum(1))) if k == 'I' else cells
                    require(np.array_equal(fields[k][p],ref), 'direct-reference mismatch '+role+'/'+k)
        require(eval_modes['hidden']['native_or_fixture_selected_bytes_returned'] == len(exam)*8*27,
                'hidden reader accessed more than the visible sites')
        savej(receipt,dict(source='synthetic-fixture', training=False,native_species_access=False,
            fixture='1632 synthetic fixed-composition or poison rows; 1408 fit, 176 dev, 8 simulated assessment, 40 forbidden',
            poison_rows_never_returned=len(deny),geometry=geom,stage_disjointness=True,
            fit_dev=rec, separate_synthetic_eval_controls=eval_modes,
            process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, incremental_fee_usd=0))
    print(json.dumps(dict(source=rec['source'],parents_per_second=rec['parents_per_second'],
                        batches_per_second=rec['batch_gather']['batches_per_second'])),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='action',required=True)
    sub.add_parser('synthetic')
    for stage in ('fit-dev','blind','hidden','pack'):
        cmd = sub.add_parser(stage); cmd.add_argument('--release-commit',required=True)
        if stage != 'pack': cmd.add_argument('--index',type=int,choices=INDICES,required=True)
    args = ap.parse_args()
    if args.action == 'synthetic': synthetic()
    elif args.action == 'pack': finish(args.release_commit)
    else: native(args.action,args.index,args.release_commit,lookup())
