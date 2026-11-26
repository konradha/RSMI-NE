"""Prepare matched-Q local-fork maps; no simulation, fitting or inference.

/usr/bin/python3 examples/stock_glass_history_maps.py train
Only after the design owner's assessment release and frozen depth:
/usr/bin/python3 examples/stock_glass_history_maps.py test --freeze FROZEN_DEPTH_RECEIPT

The local origin and all 82 later snapshots are addressed by pinned head digest.
Whole-file hashes cover opaque bytes. Only authorized native state rows are
indexed, including when checking preparation bytes and summing native overlap.
"""
import argparse
import csv
import hashlib
import io
import json
import struct
import subprocess
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REV = '1cff504a996508c4b1bc8cdb050abfe560fff911'
RUN = 'l20-b320-b340-128x2-500m-aging-v1'
GALLERY = 'results/l20-paper-intermediate'
BETAS = ('3.20', '3.40')
AGES = (300, 500)
LEVELS = (75, 80, 85)
SLICES = {'fit': slice(0, 48), 'dev': slice(48, 64), 'test': slice(64, 128)}
META = ('ids', 'betas', 'parent_ids', 'parent_age', 'parent_hash')
DEFAULT_OUT = Path(__file__).resolve().parents[1] / 'results/history-overlap-field'
BOOT_SEED = 852501
BOOT_DRAWS = 10000


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    raw = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        need(path.read_bytes() == raw, f'refusing to replace differing receipt: {path}')
    else:
        path.write_bytes(raw)


def write_csv(path, records):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(records)


def pinned(repo, rel):
    return subprocess.check_output(['git', '-C', str(repo), 'show', f'{REV}:{rel}'])


def load_sources(repo):
    root = repo / 'runs' / RUN
    data, provenance = {}, {}
    for stage in ('prep', 'local-all'):
        batch = root / stage / 'batch-00000000'
        docs = {}
        for label, path in [('plan', batch.parent / 'plan.json'), ('head', batch / 'head.json')]:
            raw = pinned(repo, f'{GALLERY}/inputs/{stage}-{label}.json')
            need(path.read_bytes() == raw, f'{stage} {label} differs from pinned source')
            docs[label] = json.loads(raw)
            provenance[f'{stage}-{label}'] = hash_bytes(raw)
        names = docs['head']['snapshots']
        lags = [int(n[:20]) for n in names]
        need(lags == sorted(set(lags)) and lags[0] == 0 and docs['head']['batch'] == 0 and
             docs['head']['plan_sha256'] == provenance[f'{stage}-plan'], f'invalid {stage} head')
        paths = {lag: batch / name for lag, name in zip(lags, names)}
        used = paths if stage == 'local-all' else {k: paths[k] for k in docs['plan']['save_at']}
        for lag, p in used.items():
            need(p.name == f'{lag:020d}-{sha(p)}.npz', f'checkpoint checksum mismatch: {p}')
        data[stage] = docs['plan'], paths
    pp, prep = data['prep']
    lp, local = data['local-all']
    corpus = hashlib.sha256()
    for lag in pp['save_at']:
        corpus.update((json.dumps([0, prep[lag].name], separators=(',', ':')) + '\n').encode())
    need((lp['move'], lp['L'], lp['betas'], lp['parent_counts'], lp['parent_ages'], lp['replicas'], lp['sweeps']) ==
         ('local', 20, list(BETAS), [128, 128], [a * 10**6 for a in AGES], 1, [10**9, 10**9]) and
         pp['move'] == 'nonlocal' and pp['chains'] == 128 and len(local) == 83 and max(local) == 10**9 and
         lp['source_plan_sha256'] == provenance['prep-plan'] and lp['source_corpus_sha256'] == corpus.hexdigest(),
         'not the full paired local cohort or wrong preparation corpus')
    need(sha(root / 'full-fork-zero.npz') == local[0].stem[21:], 'fork-zero artifact mismatch')
    provenance.update(prep_corpus=corpus.hexdigest(), full_fork_zero=local[0].stem[21:],
                      prep_attested_checkpoints=[prep[lag].name for lag in pp['save_at']],
                      local_checkpoints=[p.name for p in local.values()])
    return prep, local, provenance


def state_rows(path, indices, nrows):
    """Map an uncompressed .npy member, then gather only selected 8,000-byte rows.

    np.load(path)['states'][indices] would first materialize all TEST values.
    Use the local ZIP header (not the central-directory extra-field length).
    """
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo('states.npy')
        need(info.compress_type == zipfile.ZIP_STORED, 'expected stored, not compressed, state member')
    with path.open('rb') as f:
        f.seek(info.header_offset)
        header = f.read(30)
        need(header[:4] == b'PK\x03\x04', 'invalid local ZIP header')
        name_len, extra_len = struct.unpack_from('<HH', header, 26)
        f.seek(name_len + extra_len, 1)
        need(np.lib.format.read_magic(f) == (1, 0), 'unexpected .npy version')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        need(shape == (nrows, 8000) and dtype == np.uint8 and not fortran, 'invalid packed state schema')
        return np.memmap(f, dtype=dtype, mode='r', offset=f.tell(), shape=shape)[indices]


def native_meta(path, sweep, ph):
    with np.load(path, allow_pickle=False) as z:
        need(int(z['sweep']) == sweep and str(z['plan_sha256']) == ph, f'bad snapshot identity: {path}')
        return {k: z[k] for k in META}


def rank_key(b, pid):
    return hash_bytes(f'history1:{b}:{pid}'.encode('ascii'))


def roster(meta):
    pid, age, beta = meta['parent_ids'], meta['parent_age'], meta['betas']
    need(pid.dtype == np.uint64 and meta['ids'].dtype == np.uint64 and pid.size == 512 and
         len(set(meta['ids'])) == 512 and len(set(pid)) == 256, 'invalid native IDs')
    rows, records = {}, []
    for b, beta_str in enumerate(BETAS):
        ids = sorted(set(map(int, pid[beta == beta_str])), key=lambda n: rank_key(b, n))
        need(len(ids) == 128, 'expected 128 parents per beta')
        for a in AGES:
            subset = np.flatnonzero((age == a * 10**6) & (beta == beta_str))
            by_pid = {int(pid[i]): int(i) for i in subset}
            need(len(subset) == 128 and set(by_pid) == set(ids), 'non-paired parent ages')
            rows[b, a] = np.array([by_pid[p] for p in ids])
        for rank, p in enumerate(ids):
            side = 'fit' if rank < 48 else 'dev' if rank < 64 else 'test'
            records.append(dict(beta=beta_str, parent_id=p, rank=rank, side=side, split_sha256=rank_key(b, p),
                                **{f'child_{a}M': int(meta['ids'][rows[b, a][rank]]) for a in AGES}))
    return rows, records


def origins(prep, local, rows, meta, prov):
    """Verify both the packed local bytes and the native per-child row hash."""
    origin, audit = {}, []
    for a in AGES:
        t = a * 10**6
        pm = native_meta(prep[t], t, prov['prep-plan'])
        slots = {int(pid): i for i, pid in enumerate(pm['ids'])}
        need(len(slots) == 256 and np.array_equal(pm['parent_ids'], pm['ids']) and
             np.all(pm['parent_age'] == 0), 'invalid preparation row lineage')
        for b in range(2):
            idx = rows[b, a]
            ids = meta['parent_ids'][idx]
            src = [slots[int(pid)] for pid in ids]
            need(np.all(pm['betas'][src] == BETAS[b]), 'preparation beta mismatch')
            native = state_rows(local[0], idx, 512)
            need(np.array_equal(native, state_rows(prep[t], src, 256)), 'origin differs from prep bytes')
            for row, p, child, expected in zip(native, ids, meta['ids'][idx], meta['parent_hash'][idx]):
                h = hash_bytes(row)
                need(h == expected, 'fork parent_hash mismatch')
                audit.append(dict(beta=BETAS[b], age_M=a, parent_id=int(p), child_id=int(child),
                                  packed_fork_sha256=h, prep_snapshot=prep[t].name))
            origin[b, a] = native & 3
    return origin, audit


def encode(origin, packed):
    endpoint = packed & 3
    for sp, n in enumerate((2000, 2400, 3600)):
        need(np.all(np.count_nonzero(endpoint == sp, axis=1) == n), 'noncanonical species composition')
    occ = origin != 0
    field = occ.astype('u1')
    field += occ & (origin != endpoint)
    same = np.count_nonzero(field == 1, axis=1)
    # Independent native-state count: same-type diagonal of the actual species
    # transition table, not a second normalization of the already encoded field.
    native = sum(np.count_nonzero((origin == sp) & (endpoint == sp), axis=1) for sp in (1, 2))
    need(np.array_equal(same, native) and np.all(np.count_nonzero(field == 2, axis=1) == 6000 - native),
         'native same-occupied overlap count disagrees with ternary map')
    return field, native


def normalized(n):
    return (n / 6000 - 0.39) / 0.61


def curves(local, rows, meta, orig, ph):
    counts = {k: np.empty((len(idx), len(local)), np.int32) for k, idx in rows.items()}
    for j, (t, path) in enumerate(local.items()):
        cm = native_meta(path, t, ph)
        need(all(np.array_equal(meta[k], cm[k]) for k in META), f'child identity/order drift at {t}')
        for k, idx in rows.items():
            _, counts[k][:, j] = encode(orig[k], state_rows(path, idx, 512))
    return counts


def parent_tables(out, phase, counts, lags, rows, meta):
    n = len(next(iter(rows.values())))
    # Same parent resample at every lag and both ages within beta; never resample sites.
    records, parent_q, summaries = [], [], {}
    for b in range(2):
        rng = np.random.default_rng(BOOT_SEED + b)
        mult = rng.multinomial(n, np.full(n, 1/n), size=BOOT_DRAWS) / n
        for a in AGES:
            same = counts[b, a]
            q = normalized(same)
            mean = q.mean(axis=0)
            ci = np.percentile(mult @ q, [2.5, 97.5], axis=0)
            summaries[b, a] = mean, ci
            for j, t in enumerate(lags):
                records.append(dict(beta=BETAS[b], age_M=a, lag=t, parents=n, mean_Q=mean[j],
                                    parent_ci_low=ci[0,j], parent_ci_high=ci[1,j],
                                    mean_same_occupied_count=same[:,j].mean()))
            for i, pid in enumerate(meta['parent_ids'][rows[b, a]]):
                side = phase if phase == 'test' else 'fit' if i < 48 else 'dev'
                parent_q.extend(dict(beta=BETAS[b], age_M=a, side=side, parent_id=int(pid), lag=t,
                                     native_same=int(s), Q=v) for t, s, v in zip(lags, same[i], q[i]))
    write_csv(out / f'lag-curves-{phase}.csv', records)
    write_csv(out / f'parent-Q-{phase}.csv', parent_q)
    return summaries


def matches(lags, summaries):
    records = []
    for (b, a), (mean, ci) in summaries.items():
        for level in LEVELS:
            j = int(np.argmin(abs(mean - level / 100)))  # ascending saved lags; earlier wins an exact tie
            records.append(dict(b=b, beta=BETAS[b], age_M=a, level=level, lag=int(lags[j]),
                                train_parents=64, train_mean_Q=float(mean[j]),
                                train_low95=float(ci[0,j]), train_high95=float(ci[1,j])))
    return records


def render(out, phase, lags, summary, choices):
    fig, aa = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axs, a in zip(aa, (500, 300)):
        for ax in axs:
            for b, color in enumerate(('#136e75', '#984b13')):
                mean, ci = summary[b, a]
                ax.fill_between(lags, *ci, alpha=.20, color=color, linewidth=0)
                ax.plot(lags, mean, '.-', color=color, ms=2.8, lw=1.15, label=f'β = {BETAS[b]}')
                for rec in choices:
                    if (rec['b'], rec['age_M']) == (b, a):
                        t = rec['lag']
                        ax.scatter(t, mean[lags.index(t)], s=48, marker={75:'s',80:'o',85:'^'}[rec['level']],
                                   color=color, edgecolors='white', linewidth=.8, zorder=3)
            ax.grid(alpha=.19)
            ax.set_xlabel('physical lag (local sweeps)')
            ax.set_ylabel('normalized overlap Q')
        axs[0].set_xscale('symlog', linthresh=1)
        axs[0].set_title(f'{a}M prep · all 83 saved lags')
        axs[0].legend(loc='lower left')
        ts = [r['lag'] for r in choices if r['age_M'] == a and r['lag'] > 0]
        axs[1].set(xscale='log', xlim=(min(ts)/1.4, max(ts)*1.4), title='Near the matched lags')
        for level in LEVELS:
            axs[1].axhline(level/100, ls=':', lw=.8, color='#666666')
    fig.suptitle(f'{phase.upper()} parents · measured local relaxation\n95% pointwise whole-parent bootstrap; '
                 'TRAIN-selected Q targets: □ 0.75, ○ 0.80, △ 0.85', fontsize=12)
    fig.savefig(out / ('matched-Q.png' if phase == 'train' else 'matched-Q-test.png'), dpi=160)
    plt.close(fig)


def verify_gallery(repo, out, counts, lags):
    """Only after release, independently compare the full cohort with the pinned gallery."""
    train = list(csv.DictReader((out / 'lag-curves-train.csv').open()))
    train_same = {(r['beta'], int(r['age_M']), int(r['lag'])): float(r['mean_same_occupied_count']) for r in train}
    raw = pinned(repo, f'{GALLERY}/interim-lags.csv')
    gallery = {(float(r['beta']), int(r['parent_age']), int(r['lag'])): r for r in csv.DictReader(io.StringIO(raw.decode()))}
    need(len(gallery) == len(lags) * 4, 'incomplete gallery reference')
    qerr, rawerr = [], []
    for (b,a), same in counts.items():
        for j,t in enumerate(lags):
            full_same = (same[:,j].mean() + train_same[BETAS[b],a,t])/2
            r = gallery[float(BETAS[b]),a * 10**6,t]
            qerr.append(abs(normalized(full_same) - float(r['Q'])))
            rawerr.append(abs(full_same / 6000 - float(r['mean_raw_occupied_overlap'])))
    need(max(qerr + rawerr) < 1.e-12, 'full-cohort source Q mismatch')
    return dict(reference=f'{REV}:{GALLERY}/interim-lags.csv', sha256=hash_bytes(raw), comparisons=len(qerr),
                max_abs_Q_difference=max(qerr), max_abs_raw_difference=max(rawerr))


def prepare(args):
    phase, out, scratch = args.phase, args.out, args.scratch
    if phase == 'test':
        need(args.freeze is not None and args.freeze.is_file() and args.freeze.stat().st_size > 0,
             'TEST requires an explicit frozen-depth receipt and prior authorization')
        train_receipt = json.loads((out / 'lineage-train.json').read_bytes())
        for name in ('split.csv', 'matched-lags.json', 'lag-curves-train.csv'):
            need(sha(out / name) == train_receipt['results_sha256'][name], f'changed frozen input: {name}')
    prep, local, prov = load_sources(args.repo)
    meta = native_meta(local[0], 0, prov['local-all-plan'])
    all_rows, split = roster(meta)
    scratch.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if phase == 'train':
        write_csv(out / 'split.csv', split)
    active = slice(0,64) if phase == 'train' else SLICES['test']
    rows = {k: i[active] for k,i in all_rows.items()}
    orig, audit = origins(prep, local, rows, meta, prov)
    write_csv(out / f'fork-origins-{phase}.csv', audit)
    counts = curves(local, rows, meta, orig, prov['local-all-plan'])
    lags = list(local)
    summary = parent_tables(out, phase, counts, lags, rows, meta)
    if phase == 'train':
        choices = matches(lags, summary)
        write_json(out / 'matched-lags.json', choices)
    else:
        choices = json.loads((out / 'matched-lags.json').read_bytes())
    banks, tv = [], []
    for r in choices:
        k = b, a = r['b'], r['age_M']
        t = r['lag']
        field, same = encode(orig[k], state_rows(local[t], rows[k], 512))
        need(np.array_equal(same, counts[k][:, lags.index(t)]), 'bank disagrees with verified curve')
        if phase == 'train':
            # Equal parent weight in each split; this is a descriptive marginal,
            # not an additional gate or a proxy for patch independence.
            fractions = np.stack([(field == i).mean(axis=1) for i in range(3)], axis=1)
            fit, dev = (fractions[SLICES[side]].mean(axis=0) for side in ('fit', 'dev'))
            tv.append(dict(beta=BETAS[b], age_M=a, level=r['level'], lag=t, fit_parents=48, dev_parents=16,
                           **{f'fit_p{i}': p for i,p in enumerate(fit)}, **{f'dev_p{i}': p for i,p in enumerate(dev)},
                           total_variation=float(abs(fit - dev).sum() / 2)))
        for side in (('fit', 'dev') if phase == 'train' else ('test',)):
            s = SLICES[side] if phase == 'train' else slice(None)
            ids = meta['parent_ids'][rows[k][s]]
            path = scratch / f'b{b}-a{a}-q{r["level"]}-{side}.npz'
            q = normalized(same[s])
            np.savez(path, ids=ids, field=field[s].reshape(-1,20,20,20), lag=np.int64(t), Q=q)
            banks.append(dict(b=b, age_M=a, level=r['level'], side=side, lag=t, parents=len(q),
                              translations_per_parent=8000, measured_mean_Q=float(q.mean()),
                              path=str(path), sha256=sha(path), snapshot=local[t].name))
    write_csv(out / f'map-banks-{phase}.csv', banks)
    render(out, phase, lags, summary, choices)
    reference = verify_gallery(args.repo, out, counts, lags) if phase == 'test' else None
    result_names = ['split.csv','matched-lags.json', f'fork-origins-{phase}.csv', f'parent-Q-{phase}.csv',
                    f'lag-curves-{phase}.csv', f'map-banks-{phase}.csv']
    if tv:
        write_csv(out / 'ternary-TV-train.csv', tv)
        result_names.append('ternary-TV-train.csv')
    receipt = dict(source_repo=str(args.repo), source_revision=REV, source_run=RUN, source_files=prov,
                   native_rows=512, primary_age_M=500, paired_age_M=300,
                   split='within b order by SHA256 ASCII history1:{b}:{decimal prep parent_id}; first 48 FIT, next 16 DEV, last 64 TEST',
                   state_access='Only authorized row indices gathered; whole-file authentication uses opaque bytes.',
                   phase=phase, parent_ids_per_beta=64, authorized_child_origins=len(audit),
                   snapshot_count=len(lags), native_overlap_row_checks=len(audit)*len(lags),
                   all_row_counts_equal=True, canonical_state_counts=[2000,2400,3600],
                   lag_units='local sweeps of 8000 nearest-neighbor proposals',
                   selection='per beta and age: nearest of 83 measured equal-weight TRAIN means; earlier lag breaks a tie',
                   field='native low two bits species, C-order z,y,x. 0=initial vacancy; 1=initially occupied, same endpoint species; 2=initially occupied, different endpoint species (including vacancy). Net changes, not particle paths.',
                   overlap='Independent native same-species (1→1 plus 2→2) count == ternary state-1 count; Q=(count/6000-0.39)/0.61.',
                   uncertainty=dict(kind='pointwise 95% percentile CI for the mean, paired parent resamples over ages and lags',
                                    draws=BOOT_DRAWS, generator='numpy.default_rng multinomial equal weights', seed_per_beta='852501 + beta index'),
                   frozen_depth_receipt=None if phase == 'train' else dict(path=str(args.freeze), sha256=sha(args.freeze)),
                   full_cohort_gallery_crosscheck=reference,
                   results_sha256={name:sha(out/name) for name in result_names})
    write_json(out / f'lineage-{phase}.json', receipt)
    print(f'{phase}: verified {len(audit)} fork origins × {len(lags)} native overlap rows; {len(banks)} matched banks')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['train', 'test'])
    ap.add_argument('--repo', type=Path, default=Path.home() / 'code/lattice-exploration')
    ap.add_argument('--scratch', type=Path, default=Path.home() / 'rsmi-work/history-overlap-field')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--freeze', type=Path)
    prepare(ap.parse_args())
