"""Fit only the unchanged registered nuisance/packing offsets for the K4 diagnostic.

No raw residual, code, probe calibration, dev or exam access. Requires an issued
public packing-release.json with its digest and independently passing gates.
Frozen inference: spatial_raw.load(exact_model_file), then spatial_raw.packing_offsets.
"""
import argparse
import json
import resource
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import spatial_raw as raw
from spatial_raw import cm, joblib, np, threadpool_limits

STAGE = 'packing-only-fit'


def gate(release_sha):
    release = raw.PUBLIC/'packing-release.json'; raw.require(cm.digest(release) == release_sha, 'Incorrect signed-off release digest')
    r = json.loads(release.read_bytes()); raw.require(r['stage'] == STAGE, 'Not a packing-only release')
    raw.require(datetime.now(timezone.utc) < raw.CUTOFF, 'Discovery closed')
    raw.require(r['control_passed'] is True, 'Required native encoder control was not admitted')
    ctl = r['control_file']; raw.require(Path(ctl).name == ctl, 'Use public control basename')
    for path, key in [(raw.PUBLIC/'lineage.json', 'lineage_sha256'), (raw.PUBLIC/ctl, 'control_sha256'),
                      (raw.WORK/'fit.npz', 'fit_sha256'), (raw.WORK/'eligible.npy', 'eligible_sha256'),
                      (raw.WORK/'rosters.npz', 'rosters_sha256')]:
        raw.require(cm.digest(path) == r[key], 'Unissued input: '+path.name)
    ctl_result = json.loads((raw.PUBLIC/ctl).read_bytes())
    for name in r.get('control_pass_field', 'passed').split('.'):
        ctl_result = ctl_result[name]
    raw.require(ctl_result is True, 'Native spatial control did not pass')
    return r


def fit_prefix(site, conv, s, y, parent, levels):
    """Exact cycle_models.fit_arrays baseline stages; never initialize its encoder."""
    ids = raw.indices(levels, s); nlevels = len(levels)
    perm = np.random.default_rng(260405).permutation(np.unique(parent)); first = np.isin(parent, perm[::2])
    a = np.empty(len(y)); b = np.empty(len(y)); stages = []
    for held in (first, ~first):
        other = ~held; start = perf_counter()
        model = cm.packing_fit(site[other], conv[other], ids[other], y[other], nlevels)
        a[held], b[held] = cm.packing_logits(model, site[held], conv[held], ids[held])
        stages.append(dict(train_rows=int(other.sum()), held_rows=int(held.sum()), seconds=perf_counter()-start,
                           logistic_iterations=model['logistic_iterations'], tree_iterations=model['tree_iterations']))
    start = perf_counter(); model = cm.packing_fit(site, conv, ids, y, nlevels)
    stages.append(dict(train_rows=len(y), held_rows=0, seconds=perf_counter()-start,
                       logistic_iterations=model['logistic_iterations'], tree_iterations=model['tree_iterations']))
    return dict(levels=levels, packing=model), a, b, stages


def run(release_sha):
    start = perf_counter(); r = gate(release_sha)
    pub = raw.PUBLIC/'packing-fit.json'; marker = raw.WORK/'packing-started.json'
    raw.require(not pub.exists(), 'An immutable packing fit receipt already exists')
    cm.receipt(marker, dict(stage=STAGE, release_sha256=release_sha, started_utc=datetime.now(timezone.utc).isoformat()))
    levels = np.load(raw.WORK/'eligible.npy', allow_pickle=False)
    raw.require(levels.ndim == 1 and len(levels) and np.array_equal(levels, np.unique(levels)), 'Invalid frozen eligibility')
    shell, s, y, who, anchor, parents, supports = raw.read_bank(raw.WORK, 'fit', levels)
    report = dict(stage=STAGE, release_sha256=release_sha, raw_physical_fit=False, role='encoder-fit only',
                  outputs='OOF row-major eligible native-parent/anchor-ordinal and full-prefix frozen packing; no raw prediction.',
                  scope='Bounded descriptive label-free diagnostic only; no raw claim, no assessment or calibration bank access.',
                  implementation_sha256=cm.digest(Path(__file__)), reused_cycle_models_sha256=cm.digest(Path(cm.__file__)),
                  reused_spatial_raw_sha256=cm.digest(Path(raw.__file__)), native_threads=1, paid_calls=0,
                  inputs={k:v for k,v in r.items() if k.endswith('_sha256')}, support=supports, fits={})
    passed = all(m['passed'] for m in supports.values()); report['support_passed'] = passed
    if passed:
        now = perf_counter(); cm.rows(shell, s); site = cm.signed_sites(shell)
        conv = np.empty((len(s), cm.bm.D), 'f')
        for st in range(0, len(s), 256):
            sl = slice(st, st+256); conv[sl] = cm.summarize_shells(shell[sl])[:, :cm.bm.D]
        report['features_seconds'] = perf_counter()-now
        for size in raw.SIZES:
            sl = slice(int(np.searchsorted(who, size))); parent = parents[who[sl]]
            raw.require(np.unique(y[sl]).tolist() == [0,1], 'Need both labels within each prefix')
            model, a, b, stages = fit_prefix(site[sl], conv[sl], s[sl], y[sl], parent, levels)
            state = raw.WORK/f'packing-model-{size}.joblib'; off = raw.WORK/f'packing-oof-{size}.npz'
            raw.output_np(off, parent=parent, anchor=anchor[sl], stratum=s[sl], stratum_oof=a, packing_oof=b)
            with state.open('xb') as f:
                joblib.dump(model, f, compress=3)
            report['fits'][size] = dict(model=state.name, oof=off.name, model_sha256=cm.digest(state),
                    oof_sha256=cm.digest(off), eligible_rows=len(parent), eligible_parents=len(np.unique(parent)),
                    two_folds_then_all=stages)
    report['total_seconds'] = perf_counter()-start; report['max_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cm.receipt(pub, report); print(pub, flush=True)
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-sha256', required=True)
    with threadpool_limits(limits=1):
        if not run(p.parse_args().release_sha256):
            raise SystemExit('Support failed: no packing fit')
