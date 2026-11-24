"""Extract occupied-site same-species change counts from released L20 local forks.

--runs and --paper are lattice-exploration's runs/ and results/l20-paper/.
--out is the directory of cohort NPZs; --provenance is the portable JSON receipt.
No simulation or model fit is involved. The low two bits encode site species;
compare species, not the packed coordination bits or individual particle IDs.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

import numpy as np


ORIGINAL = 'l20-b305-production-v1'
DOUBLED = 'l20-b2711-double-v1'
SNAP = re.compile(r'(\d{20})-([0-9a-f]{64})\.npz')
COMPOSITION = (2000, 2400, 3600)
N_OCCUPIED = sum(COMPOSITION[1:])
Q0 = sum(n*n for n in COMPOSITION[1:]) / (N_OCCUPIED * sum(COMPOSITION))
LABELS = ('ids', 'parent_ids', 'parent_age', 'parent_hash', 'betas')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Sources:
    """Record names relative to their source root, never workstation paths."""
    def __init__(self, root, receipt):
        self.root = root
        self.receipt = receipt

    def read(self, name, expected=None):
        data = (self.root / name).read_bytes()
        digest = sha(data)
        require(expected is None or expected == digest, f'checksum mismatch: {name}')
        self.receipt[str(name)] = digest
        return data

    def checkpoint(self, path, plan_hash):
        lag, digest = SNAP.fullmatch(path.name).groups()
        with np.load(io.BytesIO(self.read(path, digest)), allow_pickle=False) as z:
            require(str(z['plan_sha256']) == plan_hash and int(z['sweep']) == int(lag),
                    f'wrong plan or sweep: {path}')
            state = z['states']
            labels = {k: z[k] for k in LABELS}
        n = len(state)
        require(state.dtype == np.uint8 and state.shape == (n, 8000) and
                all(v.shape == (n,) for v in labels.values()) and
                all(labels[k].dtype == np.uint64 for k in ('ids', 'parent_ids')) and
                len(np.unique(labels['ids'])) == n, f'bad checkpoint rows: {path}')
        return state, labels


def study_inputs(source, study):
    root = Path(study)
    verdict = json.loads(source.read(root / 'evidence/verdict.json'))
    plans, paths = {}, {}
    for phase, plan_key in [('probe', 'prep_plan_sha256'), ('physical', 'local_plan_sha256')]:
        folder = root / phase
        plan = json.loads(source.read(folder / 'plan.json', verdict[plan_key]))
        # Both released validation runs used a single batch spanning all beta strata.
        heads = sorted((source.root / folder).glob('batch-*/head.json'))
        require([p.parent.name for p in heads] == list(verdict['head_sha256'][phase]) == ['batch-00000000'],
                f'unexpected committed {phase} batches: {root}')
        head_name = folder / heads[0].parent.name / 'head.json'
        head = json.loads(source.read(head_name, verdict['head_sha256'][phase][heads[0].parent.name]))
        require(head['batch'] == 0 and head['plan_sha256'] == verdict[plan_key], f'wrong head: {head_name}')
        snaps = {}
        for name in head['snapshots']:
            match = SNAP.fullmatch(name)
            require(match is not None, f'bad committed name: {name}')
            lag = int(match[1])
            require(not snaps or lag > next(reversed(snaps)), f'unordered committed sweep: {name}')
            snaps[lag] = head_name.parent / name
        require(0 in snaps and max(snaps) == max(plan['sweeps']), f'incomplete head: {head_name}')
        plans[phase], paths[phase] = plan, snaps
    prep, phys = plans['probe'], plans['physical']
    require(prep['move'] == 'nonlocal' and phys['move'] == 'local' and prep['L'] == phys['L'] == 20 and
            prep['betas'] == phys['betas'] and phys['source_plan_sha256'] == verdict['prep_plan_sha256'],
            f'wrong fork source: {study}')
    # Reproduce the campaign's attested probe corpus digest (endpoint/save_at receipts).
    corpus = hashlib.sha256()
    for t in sorted(set(prep['sweeps'] + prep['save_at'])):
        path = paths['probe'][t]
        source.read(path, SNAP.fullmatch(path.name)[2])
        corpus.update((json.dumps([0, path.name], separators=(',', ':')) + '\n').encode())
    require(corpus.hexdigest() == phys['source_corpus_sha256'] == verdict['source_corpus_sha256'],
            f'probe corpus mismatch: {study}')
    return verdict, plans, paths


def species(packed):
    values = packed & 3
    for s, count in enumerate(COMPOSITION):
        require(np.all(np.count_nonzero(values == s, axis=-1) == count), 'site composition mismatch')
    return values


def select(lags, tau_row):
    positive = [t for t in lags if t > 0]
    if float(tau_row['beta']) == 0:
        require(tau_row['status'] == 'before_first_positive' and int(tau_row['lag_high']) == 1 and
                min(positive) == 1, 'unexpected beta-zero censoring')
        return {1: 'early+alpha'}
    tau = float(tau_row['tau'])
    require(tau_row['status'] == 'interpolated' and tau > 0, 'expected a positive interpolated tau')
    early, alpha = [min(positive, key=lambda t: (abs(math.log(t / target)), t))
                    for target in (.01*tau, tau)]
    return {early: 'early+alpha'} if early == alpha else {early: 'early', alpha: 'alpha'}


def extract(index, study, source, inputs, zero, endpoints, tau_row, lag_rows, out):
    verdict, plans, paths = inputs
    prep, phys = plans['probe'], plans['physical']
    beta = float(tau_row['beta'])
    bi = 0 if index == 8 else index
    spec = verdict['by_beta'][bi]
    age, horizon = spec['candidate'], spec['physical']
    p, r = spec['parents'], spec['children_per_parent_per_age']
    require(tau_row['study'] == study and spec['index'] == bi and spec['verdict'] == 'PASS' and
            float(spec['beta']) == beta and spec['beta'] == phys['betas'][bi] and
            phys['parent_age_pairs'][bi] == [age, spec['older']] and age < spec['older'] and
            phys['sweeps'][bi] == horizon and phys['parent_counts'][bi] == p and
            0 < r == phys['replicas_per_beta'][bi] <= 255, f'inconsistent release at beta={beta}')
    for row in [tau_row, *lag_rows]:
        require(row['study'] == study and (int(row['candidate_age']), int(row['parents']),
                int(row['siblings_per_parent'])) == (age, p, r), f'wrong paper cohort for beta={beta}')
    archive_q = {int(row['lag']): float(row['Q']) for row in lag_rows}
    available = [t for t in paths['physical'] if t <= horizon]
    require(set(available) == archive_q.keys() and len(archive_q) == len(lag_rows) and
            available[-1] == horizon, f'paper lag inventory differs at beta={beta}')
    chosen = select(available, tau_row)
    packed, base = zero
    rows = np.flatnonzero((base['betas'] == spec['beta']) & (base['parent_age'] == age))
    rows = rows[np.lexsort((base['ids'][rows], base['parent_ids'][rows]))]
    parent_ids, multiplicity = np.unique(base['parent_ids'][rows], return_counts=True)
    require(len(parent_ids) == p and np.all(multiplicity == r), f'unbalanced candidate cohort: {beta}')
    rows = rows.reshape(p, r)
    ids = base['ids'][rows]
    require(np.all(ids >= np.uint64(2**63)) and np.all(parent_ids < np.uint64(2**63)),
            f'child and parent stream namespaces overlap: {beta}')
    parents = packed[rows[:, 0]]
    parent_sha = np.array([sha(state) for state in parents])
    require(np.all(base['parent_hash'][rows] == parent_sha[:, None]) and
            np.array_equal(packed[rows], np.broadcast_to(parents[:, None], (p, r, 8000))),
            f'siblings do not replicate their parent: {beta}')
    prep_path = paths['probe'][age]
    probe, lineage = source.checkpoint(prep_path, verdict['prep_plan_sha256'])
    slots = {int(pid): j for j, pid in enumerate(lineage['ids'])}
    loc = [slots[int(pid)] for pid in parent_ids]
    require(np.all(lineage['betas'][loc] == spec['beta']) and
            np.array_equal(lineage['parent_ids'][loc], parent_ids) and
            np.array_equal(probe[loc], parents), f'candidate preparation states differ: {beta}')
    origin = species(parents)
    lineage_rows = [dict(parent_id=str(int(pid)), packed_parent_sha256=str(h),
                         child_stream_ids=[str(int(k)) for k in children])
                    for pid, h, children in zip(parent_ids, parent_sha, ids)]
    changed = np.empty((len(chosen), p, 8000), dtype=np.uint8)
    half_changed = np.empty_like(changed)
    comparisons = []
    for j, (lag, role) in enumerate(chosen.items()):
        path = paths['physical'][lag]
        state, labels = source.checkpoint(path, verdict['local_plan_sha256'])
        # Check the *whole* compacted table, then index into this snapshot anew.
        # Hot strata retire after (not at) their terminal checkpoint.
        active = endpoints >= lag
        require(all(np.array_equal(labels[k], base[k][active]) for k in LABELS),
                f'compacted checkpoint lineage mismatch: {path}')
        slots = np.full(len(active), -1, dtype=np.int64)
        slots[active] = np.arange(len(state))
        loc = slots[rows]
        require(np.all(loc >= 0) and np.array_equal(labels['ids'][loc], ids), f'missing children: {path}')
        event = (species(state[loc]) != origin[:, None]) & (origin[:, None] != 0)
        np.sum(event, axis=1, out=changed[j], dtype=np.uint8)
        np.sum(event[:, :r//2], axis=1, out=half_changed[j], dtype=np.uint8)
        count = int(changed[j].sum())
        raw = 1 - count / (p*r*N_OCCUPIED)
        q = (raw - Q0) / (1 - Q0)
        delta = abs(q - archive_q[lag])
        require(delta < 1e-12, f'overlap mismatch at beta={beta}, lag={lag}: {q} vs {archive_q[lag]}')
        comparisons.append(dict(lag=lag, role=role, checkpoint=str(path), active_checkpoint_rows=len(state),
                                changed_child_sites=count, raw_same_species_fraction=raw,
                                Q=q, Q_paper=archive_q[lag], Q_absolute_difference=delta))
    cohort = f'archive-{index:02d}'
    target = out / f'{cohort}.npz'
    np.savez_compressed(target, cohort=cohort, index=index, beta=beta, replicas=r, parent_id=parent_ids,
                        origin=origin.reshape(p, 20, 20, 20), lags=np.array(list(chosen), np.int64),
                        roles=np.array(list(chosen.values())), changed=changed.reshape(-1, p, 20, 20, 20),
                        half_changed=half_changed.reshape(-1, p, 20, 20, 20))
    peak = max(v['Q_absolute_difference'] for v in comparisons)
    print(f'{cohort}: beta={beta:g} P={p} R={r} lags={chosen} max |ΔQ|={peak:.3g}', flush=True)
    return dict(cohort=cohort, index=index, beta=beta, study=study, study_beta_index=bi,
                candidate_age=age, physical_horizon=horizon, available_lags=available,
                tau=float(tau_row['tau']) if tau_row['tau'] else None, tau_status=tau_row['status'],
                parents=p, replicas=r, children=p*r, occupied_parent_sites=p*N_OCCUPIED,
                physical_seed=str(phys['seed']), preparation_seed=str(prep['seed']),
                parent_checkpoint=str(prep_path), origin_checkpoint=str(paths['physical'][0]),
                lineage=lineage_rows, selected=comparisons, output_file=target.name,
                output_sha256=sha(target.read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('runs', 'paper', 'out', 'provenance'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema='l20.archive-propensity.v1', runs_sha256={}, paper_sha256={}, cohorts=[],
                   composition=list(COMPOSITION), Q_independent=Q0,
                   event='Initially occupied site does not hold its original species at the saved lag.',
                   half_changed='First floor(R/2) siblings sorted by uint64 child stream ID within parent.',
                   horizons='Nearest available positive lag in log time to 0.01*tau and tau; lag 1 for censored beta zero.',
                   identity='IDs and seeds are decimal strings in this JSON to preserve unsigned 64-bit precision.')
    source = Sources(args.runs, receipt['runs_sha256'])
    paper = Sources(args.paper, receipt['paper_sha256'])
    def table(name):
        return list(csv.DictReader(io.StringIO(paper.read(name).decode())))
    taus = sorted(table('dynamics_tau.csv'), key=lambda row: float(row['beta']))
    lags = table('dynamics_lags.csv')
    require(len(taus) == len({t['beta'] for t in taus}) == 10, 'expected ten distinct published temperatures')
    for study in (ORIGINAL, DOUBLED):
        inputs = study_inputs(source, study)
        verdict, plans, paths = inputs
        zero = source.checkpoint(paths['physical'][0], verdict['local_plan_sha256'])
        endpoint = dict(zip(plans['physical']['betas'], plans['physical']['sweeps']))
        endpoints = np.array([endpoint[b] for b in zero[1]['betas']])
        for i, row in enumerate(taus):
            if study != (DOUBLED if i == 8 else ORIGINAL):
                continue
            same_beta = [v for v in lags if v['beta'] == row['beta']]
            receipt['cohorts'].append(extract(i, study, source, inputs, zero, endpoints, row, same_beta, args.out))
    receipt['cohorts'].sort(key=lambda v: v['index'])
    args.provenance.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
