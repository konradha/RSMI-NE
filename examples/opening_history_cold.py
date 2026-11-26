"""Frozen physical history workflow; --qualified-library must be the verified oracle build."""
import argparse
import json
from pathlib import Path

import numpy as np

import opening_history_match as m
from opening_history_stage import ARCHIVE, LOCK, OUT, SCRATCH, SPLIT, native, save_json


def feature_file(row):
    return SCRATCH / 'cold-features' / f'{row:03}.npz'


def stage_features(rows, oracle, states, meta):
    feature_file(0).parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        f = oracle(states[row], float(meta['betas'][row]))
        np.savez(feature_file(row), **f)


def freeze(args, origin, meta):
    rows, _ = native.roster(meta)
    fit = np.concatenate([rows[0, a][:48] for a in native.AGES])
    oracle = m.Oracle(args.qualified_library)
    stage_features(fit, oracle, origin, meta)
    forks = []
    for row in fit:
        with np.load(feature_file(row)) as f:
            forks.append({k: f[k] for k in ('key', 'op', 'shell')})
    cuts = m.fit_cuts(forks)
    np.savez_compressed(OUT / 'fit-cuts.npz', **cuts)
    save_json(OUT / 'fit-lock.json', dict(protocol=LOCK, source_beta=3.2, fit_parents=48, paired_forks=96,
        occupied_sites=576000, strata_with_at_least_24=len(cuts['key']),
        usable_cut_strata={k:int(np.count_nonzero(cuts[k+'_lo'] < cuts[k+'_hi'])) for k in ['opening','packing']},
        cuts_sha256=native.sha(OUT/'fit-cuts.npz'), library_sha256=native.sha(args.qualified_library),
        input_sha256=native.sha(SCRATCH/'origins.npy'), fit_source_rows=fit.astype(int).tolist(), outcome_rows_read=0))


def extract(idx, origin, local, prov, meta, lags):
    """Only selected rows of stored states.npy; fork baseline is never an observation of a change."""
    sent = len(lags)
    coarse_idx = sorted({0, *range(2, len(lags), 2), lags.index(10**6), sent - 1})
    fg, cg = np.full(origin.shape, sent, 'u1'), np.full(origin.shape, len(coarse_idx), 'u1')
    ever_returned = np.zeros(origin.shape, bool)
    coarse = {i: j for j, i in enumerate(coarse_idx)}
    occ = origin != 0
    for i, lag in enumerate(lags[1:], 1):
        nm = native.native_meta(local[lag], lag, prov['local-all-plan'])
        if any(not np.array_equal(nm[k], meta[k]) for k in native.META):
            raise ValueError(f'changed ancestry/order at lag {lag}')
        endpoint = native.state_rows(local[lag], idx, 512) & 3
        if np.any(endpoint == 3) or np.any((endpoint != 0).sum(1) != 6000):
            raise ValueError(f'invalid occupation at lag {lag}')
        changed = (endpoint != origin) & occ
        ever_returned |= (fg < i) & ~changed & occ
        fg[(fg == sent) & changed] = i
        if i in coarse:
            cg[(cg == len(coarse_idx)) & changed] = coarse[i]
    return fg, cg, ever_returned.sum(1), np.array(lags)[coarse_idx]


def score(args, phase, origin, meta):
    from opening_history_report import write_phase
    lock = json.loads((OUT/'fit-lock.json').read_bytes())
    if (native.sha(args.qualified_library), native.sha(OUT/'fit-cuts.npz'), native.sha(SCRATCH/'origins.npy')) != (
            lock['library_sha256'], lock['cuts_sha256'], lock['input_sha256']):
        raise ValueError('fit lock differs')
    cuts = dict(np.load(OUT/'fit-cuts.npz'))
    rows, roster = native.roster(meta)
    spec = {'development': (0, slice(0,64)), 'held-source': (0,slice(64,128)), 'transfer': (1,slice(None))}
    b, ranks = spec[phase]
    idx = np.concatenate([rows[b,a][ranks] for a in native.AGES])
    if phase == 'development':
        fresh = np.setdiff1d(idx, lock['fit_source_rows'])
    else:
        if not (OUT/'development-summary.json').is_file():
            raise ValueError('development must precede held release')
        fresh = idx
    stage_features(fresh, m.Oracle(args.qualified_library), origin, meta)
    staging = json.loads((OUT/'staging.json').read_bytes()); lags = staging['lag_sweeps']
    prov = staging['source']; paths = ARCHIVE/'runs'/native.RUN/'local-all/batch-00000000'
    local = {lag: paths/n for lag,n in zip(lags,prov['local_checkpoints'])}
    first, decimated, returned, coarse = extract(idx, origin[idx], local, prov, meta, lags)
    # Future bounded comparison, if qualified, can reuse the same targets without reopening trajectories.
    np.savez_compressed(SCRATCH/f'events-{phase}.npz', rows=idx, first=first, decimated=decimated,
                        coarse_lags=coarse, lags=lags, observed_returned_sites=returned)
    by_pid = {r['parent_id']:r for r in roster if r['beta']==native.BETAS[b]}
    frames = []
    for j,row in enumerate(idx):
        with np.load(feature_file(row)) as f:
            group = m.groups(f, cuts, float(native.BETAS[b]))
        pid = int(meta['parent_ids'][row]); occ = origin[row] != 0
        info = dict(beta=native.BETAS[b], parent_id=pid, age=int(meta['parent_age'][row]),
                    child_id=int(meta['ids'][row]), rank=by_pid[pid]['rank'], side=by_pid[pid]['side'])
        for schedule, t, e in [('full',lags,first), ('decimated',coarse,decimated)]:
            for mode, (sup, curves) in m.fork_risks(e[j,occ],group,len(t)).items():
                frames.append((dict(**info,schedule=schedule,mode=mode,**sup), curves))
    write_phase(phase, frames, {'full':np.array(lags),'decimated':coarse},
                {str(int(meta['ids'][row])):int(returned[j]) for j,row in enumerate(idx)}, lock)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['freeze','development','held-source','transfer'])
    ap.add_argument('--qualified-library',required=True,type=Path); a=ap.parse_args()
    qual = json.loads((OUT.parent/'oracle-qualification.json').read_bytes())
    if not qual['qualified'] or native.sha(Path(__file__).with_name('opening_oracle.cpp')) != qual['source_sha256']['opening_oracle.cpp']:
        raise ValueError('oracle source not qualified')
    if native.sha(OUT/'protocol.json') != LOCK or native.sha(SPLIT) != json.loads((OUT/'staging.json').read_bytes())['roster_sha256']:
        raise ValueError('invalid protocol or roster')
    origin = np.load(SCRATCH/'origins.npy',mmap_mode='r')
    with np.load(SCRATCH/'meta.npz') as meta:
        if a.action=='freeze': freeze(a,origin,meta)
        else: score(a,a.action,origin,meta)


if __name__=='__main__':
    main()
