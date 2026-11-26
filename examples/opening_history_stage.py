"""Stage only attested fork origins; no endpoint state rows. Run with /usr/bin/python3."""
import csv
import json
from pathlib import Path

import numpy as np

import stock_glass_history_maps as native

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/opening-mechanism/history'
SCRATCH = Path.home() / 'rsmi-work/opening-mechanism/history'
ARCHIVE = Path.home() / 'code/lattice-exploration'
SPLIT = ROOT / 'results/history-overlap-field/split.csv'
LOCK = 'fbca6a0587fd26fe3cb06d888cdde9a1b8a3c631b9084db57e15e07342c94694'


def save_json(path, data):
    raw = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(raw)
    tmp.replace(path)


def stage():
    native.need(native.sha(OUT / 'protocol.json') == LOCK, 'protocol lock mismatch')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    prep, local, prov = native.load_sources(ARCHIVE)
    meta = native.native_meta(local[0], 0, prov['local-all-plan'])
    rows, roster = native.roster(meta)
    with SPLIT.open() as f:
        canonical = list(csv.DictReader(f))
    native.need([{k: str(v) for k, v in r.items()} for r in roster] == canonical, 'parent split differs')
    origin, audit = native.origins(prep, local, rows, meta, prov)
    states = np.zeros((512, 8000), np.uint8)
    for key, ix in rows.items():
        states[ix] = origin[key]
    native.need(np.all((states <= 2)) and np.all((states != 0).sum(axis=1) == 6000), 'fork composition')
    np.save(SCRATCH / 'origins.npy', states)
    np.savez(SCRATCH / 'meta.npz', **meta)
    native.write_csv(OUT / 'cold-origins.csv', audit)
    save_json(OUT / 'staging.json', dict(protocol=LOCK, source_revision=native.REV,
        distinct_parents=256, cold_forks=512, occupied_origin_sites=3072000,
        lag_sweeps=list(local), source=prov, roster_sha256=native.sha(SPLIT),
        origin_array_sha256=native.sha(SCRATCH / 'origins.npy'), native_meta_sha256=native.sha(SCRATCH / 'meta.npz'),
        outcome_fields_extracted=0))


if __name__ == '__main__':
    stage()
