"""Separate archive extraction RSS from launcher-inherited ru_maxrss, without a native reread."""
import argparse
import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np

import full_hero_stream as stream


def vmhwm():
    return int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines()
                    if s.startswith('VmHWM:')))


def one(idx, rev):
    # The physical control never opens the original NPZ, dev, I or examinations.
    _, _, release_sha = stream.authorized('fit-dev', idx, rev)
    rec = stream.readj(stream.PUB / f'stream-b{idx:02d}.json')
    stream.require(rec['release_sha256'] == release_sha and rec['stage'] == 'fit-dev', 'ineligible role')
    base = vmhwm(); b = stream.benchmark(stream.STREAMS / f'b{idx:02d}' / 'fit')
    return dict(source='archived-physical, released FIT arrays only', index=idx, vmhwm_after_import_KiB=base,
                fit_gather_peak_VmHWM_KiB=vmhwm(), gather_batches_per_second=b['batches_per_second'],
                batch_shape=[b['groups'],b['distinct_parents_per_group']], batches=b['batches'])


def matched_fixture(nfit, ndev):
    # Same native-sized member, output shape, union-sorted scatter, fsync and streaming SHA path.
    rng = np.random.default_rng(26112506); parents = rng.permutation(16384)
    roles = dict(fit=parents[:nfit], dev=parents[nfit:nfit+ndev]); base = vmhwm()
    template = np.repeat(np.arange(3,dtype='u1'), (2000,2400,3600)); tables = stream.lookup()
    with tempfile.TemporaryDirectory(prefix='hero-rss-synthetic-') as tmp:
        tmp = Path(tmp); src = tmp/'synthetic.npz'
        with ZipFile(src,'x',compression=ZIP_DEFLATED,compresslevel=1) as z, z.open('species.npy','w') as f:
            np.lib.format.write_array_header_1_0(f, dict(descr='|u1',fortran_order=False,shape=(16384,20,20,20)))
            for _ in range(16384): f.write(rng.permutation(template))
        before = vmhwm(); r = stream.project(src,tmp/'out',0,roles,'fit-dev',tables,False)
        after = vmhwm(); stream.benchmark(tmp/'out'/'fit')
        return dict(source='entirely synthetic fixed-composition rows; native-sized compressed-member fixture',
                    matched_fit_parents=nfit, matched_dev_parents=ndev, vmhwm_after_import_KiB=base,
                    vmhwm_after_fixture_and_tables_KiB=before, projection_and_hash_peak_VmHWM_KiB=after,
                    peak_after_gather_VmHWM_KiB=vmhwm(), output_bytes=r['output_bytes'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--release-commit',required=True)
    ap.add_argument('--worker',type=int); ap.add_argument('--fixture',nargs=2,type=int); args = ap.parse_args()
    if args.worker is not None:
        print(json.dumps(one(args.worker,args.release_commit)))
    elif args.fixture:
        print(json.dumps(matched_fixture(*args.fixture)))
    else:
        summary = stream.readj(stream.PUB/'stream-summary.json'); lineage = stream.readj(stream.PUB/'lineage.json')
        def run(*extra):
            return json.loads(subprocess.check_output([sys.executable,__file__,'--release-commit',args.release_commit,*extra]))
        physical = [run('--worker',str(i)) for i in stream.INDICES]
        biggest = max(stream.INDICES, key=lambda i:summary['per_beta'][str(i)]['parents'])
        cohort = lineage['per_beta'][str(biggest)]
        fixture = run('--fixture',str(cohort['fit']),str(cohort['dev']))
        stream.savej(stream.PUB/'stream-memory.json',dict(stage='fit-dev capacity measurement only',
            release_commit=args.release_commit, summary_sha256=stream.sha(stream.PUB/'stream-summary.json'),
            note='Original ru_maxrss includes a pre-exec launcher high-water mark; upper bound only. '
                 'Linux /proc/self/status VmHWM is the resident high-water mark for the current executable. '
                 'The native projection was not rerun; exact original extraction-only peak unavailable.',
            launcher_ru_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            launcher_python_VmHWM_KiB=vmhwm(), physical_fit_only=physical, synthetic_projection=fixture,
            native_species_reads=0, physical_eval_access=False, training=False, incremental_fee_usd=0))
        print(json.dumps(dict(staging_peak_range_KiB=[min(r['fit_gather_peak_VmHWM_KiB'] for r in physical),
            max(r['fit_gather_peak_VmHWM_KiB'] for r in physical)], synthetic_extraction=fixture)))
