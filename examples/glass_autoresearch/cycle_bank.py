"""One-role-only extraction and center-only pilot for the frozen #04 boundary question."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from scipy.special import xlogy
from scipy.stats import t

from boundary_bank import PINS, pcg
from boundary_models import SHELL
from fifteen_topology import BITS, PATCH, topology_table
from rate import digest, partition, table
from vacancy_dipole import selected_member

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
PUBLIC = ROOT / 'results/l20-autoresearch/cycle-boundary'
PRIVATE = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/cycle-boundary'
DATA = Path.home() / 'datasets/lattice-glass/l20-16384-v1'
ARCHIVED = PUBLIC.parent / 'fifteen/topology-parents.npz'
CENTRE = PATCH + [0, 0, -1]
WEIGHTS = 1 << np.arange(12, dtype=np.int16)
ROLES = {'pilot': (9, 256), 'fit': (9, 3072), 'dev': (9, 512), 'cold': (9, 1024),
         'hot': (0, 1024), 'transfer': (8, 1024)}
N_ANCHOR = 128
SIGN_SEED = 260405


def require(ok, message):
    if not ok:
        raise ValueError(message)


def geometry():
    ref = np.indices((9, 9, 9)).reshape(3, -1).T - 4
    require(np.array_equal(SHELL, ref[np.max(abs(ref), axis=1) == 4]) and len(SHELL) == 386,
            'not the registered z,y,x shell/bit order')
    require(np.array_equal(PATCH, np.indices((2, 2, 3)).reshape(3, -1).T), 'patch/word order')
    ds = abs(SHELL[:, None] - CENTRE)
    distance = int(np.minimum(ds, 20 - ds).sum(axis=2).min())
    owner = np.r_[np.zeros((1, 3), int), np.eye(3, dtype=int), -np.eye(3, dtype=int)]
    diameter = int(abs(owner[:, None] - owner).sum(axis=2).max())
    require(distance == 3 and diameter == 2 and distance > diameter, 'shell is not distant')
    # Set one vacancy at a time, including the two real bits in the final byte.
    single = np.packbits(np.eye(386, dtype=np.uint8), axis=1, bitorder='little')
    k = np.arange(386)
    require(single.shape == (386, 49) and np.all(single[k, k // 8] == 1 << (k % 8))
            and np.all(np.count_nonzero(single, axis=1) == 1), 'shell serialization')
    return dict(minimum_toroidal_L1=distance, interaction_diameter_L1=diameter,
                shell_sites=len(SHELL), shell_zyx_i1_sha256=hashlib.sha256(SHELL.tobytes()).hexdigest(),
                shell_bytes=49, vacancy_bit=1, bitorder='little')


def catalogue():
    b = topology_table()
    with np.load(ARCHIVED, allow_pickle=False) as z:
        require(b.shape == (4096,) and np.array_equal(b, z['betti']), 'frozen #04 LUT differs')
    y = (b > 0).astype(np.uint8)
    require(y.sum() == 142 and np.array_equal(BITS @ WEIGHTS, np.arange(4096)), 'word/label mismatch')
    n = BITS.sum(axis=1)
    i, j = np.nonzero(np.triu(abs(CENTRE[:, None] - CENTRE).sum(axis=2) == 1, 1))
    e = (BITS[:, i] * BITS[:, j]).sum(axis=1)
    imbalance = abs((1 - BITS) @ (1 - 2 * (CENTRE.sum(axis=1) % 2)))
    require(len(i) == 20 and e.max() < 32 and imbalance.max() < 8, 'nuisance radix overflow')
    # Id base is the n_A=0 id; adding 64 per A produces the final id.
    base = ((n * 13 * 16 + e // 2) * 4 + imbalance // 2).astype(np.int16)
    num = {}; den = {}; total = math.perm(8000, 12)
    for w in range(4096):
        k = int(n[w]); v = 12 - k
        for a in range(k + 1):
            # Count ordered 12-site draws from 2400 A, 3600 B, 2000 vacancies.
            # There are C(k,a) species assignments to this specific occupancy word.
            weight = math.comb(k, a) * math.perm(2400, a) * math.perm(3600, k-a) * math.perm(2000, v)
            s = int(base[w]) + 64 * a
            den[s] = den.get(s, 0) + weight
            num[s] = num.get(s, 0) + int(y[w]) * weight
    require(sum(den.values()) == total, 'finite-composition catalogue mass does not sum to one')
    ids = np.array(sorted(den), dtype=np.int16)
    p = np.array([num[s] / den[s] for s in ids]); mass = np.array([den[s] / total for s in ids])
    # A shell disjoint from the center sees fixed remaining counts conditional on inventory.
    return y, base, ids, p, mass, sum(num.values()) / total


def rosters(pins):
    path = PRIVATE / 'rosters.npz'; result = {}; seen = set(); last = {}
    with np.load(path, allow_pickle=False) as z:
        for role, (idx, n) in ROLES.items():
            index = z[role + '_index']
            require(index.shape == () and index.dtype.kind in 'iu' and int(index) == idx, 'wrong role index')
            parent, seed, chain = (z[role + '_' + k] for k in ('parent', 'seed', 'id'))
            require(all(a.shape == (n,) and a.dtype.kind in 'iu' for a in (parent, seed, chain)),
                    f'{role}: malformed roster')
            allowed = partition(idx)[0][0][5120:]
            positions = {int(row): i for i, row in enumerate(allowed)}
            require(np.isin(parent, allowed).all(), f'{role}: not permitted native TRAIN tail')
            pos = [positions[int(row)] for row in parent]
            require(all(a < b for a, b in zip([last.get(idx, -1), *pos[:-1]], pos)),
                    f'{role}: duplicate, reordered or overlapping parent')
            last[idx] = pos[-1]
            keys = set(zip(map(int, seed), map(int, chain)))
            require(np.all(seed == int(pins[idx]['bulk_seed_uint64'])) and len(keys) == n
                    and not seen.intersection(keys), f'{role}: not disjoint bulk lineages')
            seen.update(keys); result[role] = (parent, seed, chain)
    return result


def archive_rows(role, roster, pin):
    idx = ROLES[role][0]; beta, fixed_sha = PINS[idx]; path = DATA / pin['npz']
    receipt_path = DATA / pin['sidecar']; receipt = json.loads(receipt_path.read_bytes())
    require(digest(receipt_path) == pin['sidecar_sha256'] and receipt['endpoint_sha256'] ==
            pin['npz_sha256'] == fixed_sha == digest(path) and float(receipt['beta']) == beta,
            'source pin mismatch')
    parent, seed, chain = roster
    with ZipFile(path) as z:
        for name, val in [('beta', beta), ('axes', 'chain,z,y,x')]:
            with z.open(name + '.npy') as f:
                require(np.lib.format.read_array(f, allow_pickle=False).item() == val, 'archive metadata')
        for name, ref in [('chain_seed', seed), ('chain_id', chain)]:
            require(np.array_equal(selected_member(z, name, parent, ()), ref), 'roster/source lineage mismatch')
        state = selected_member(z, 'species', parent, (20, 20, 20)).reshape(len(parent), 8000)
    require(state.dtype == np.uint8 and np.all((state == 1).sum(axis=1) == 2400)
            and np.all((state == 2).sum(axis=1) == 3600) and np.all((state == 0).sum(axis=1) == 2000),
            'invalid species or finite composition')
    return state


def extract(role, parent, state, label, base):
    idx = ROLES[role][0]; patch = table(CENTRE); shell = None if role == 'pilot' else table(SHELL)
    for t, off in ((patch, CENTRE), (shell, SHELL)):
        if t is None:
            continue
        for a in (0, 19, 7999):
            site = (np.array(np.unravel_index(a, (20,)*3)) + off) % 20
            require(np.array_equal(t[a], np.ravel_multi_index(site.T, (20,)*3)), 'periodic site offset')
    y = np.empty((len(parent), N_ANCHOR), np.uint8); s = np.empty_like(y, dtype=np.int16)
    bank = dict(parent=parent, y=y, stratum=s)
    if shell is not None:
        bank['shell'] = np.empty((*y.shape, 49), np.uint8)
    for i, row in enumerate(parent):
        anchor = pcg(260404, idx, row).choice(8000, size=N_ANCHOR, replace=False)
        center = state[i, patch[anchor]]
        w = (center != 0) @ WEIGHTS
        y[i] = label[w]; s[i] = base[w] + 64 * (center == 1).sum(axis=1)
        if shell is not None:
            vacant = state[i, shell[anchor]] == 0
            packed = np.packbits(vacant, axis=1, bitorder='little')
            require(not np.any(packed[:, -1] & 252) and np.array_equal(
                np.unpackbits(packed, axis=1, bitorder='little', count=386), vacant), 'shell bit mismatch')
            bank['shell'][i] = packed
    return bank


def precision(parent, y, row, eligible):
    # Cross-fit only the stratum frequencies, never a shell model. Halves are whole roster blocks.
    p = len(parent); k = len(eligible); mid = p // 2
    keep = eligible[row]; n = keep.sum(axis=1); pair_sum = np.zeros(p)
    for tr, te in [(slice(None, mid), slice(mid, None)), (slice(mid, None), slice(None, mid))]:
        total = np.bincount(row[tr].ravel(), minlength=k)
        yes = np.bincount(row[tr].ravel(), weights=y[tr].ravel(), minlength=k)
        # Jeffreys smoothing fixed before pilot to handle either absent class in a reference half.
        logits = np.log((yes + .5) / (total - yes + .5))
        require(np.all(total[eligible] > 0), 'eligible stratum missing in reference half')
        for i in range(*te.indices(p)):
            sign = 2 * pcg(SIGN_SEED, 9, parent[i]).integers(2, size=N_ANCHOR) - 1
            eta = logits[row[i]]; delta = .15 * sign
            # Perturbed minus reference log probability, in nats per included origin.
            diff = y[i] * delta - (np.logaddexp(0., eta + delta) - np.logaddexp(0., eta))
            pair_sum[i] = diff[keep[i]].sum()
    if not n.sum():
        return dict(status='no eligible origins; nominal precision undefined')
    d = pair_sum.sum() / n.sum()
    influence = (pair_sum - d * n) / n.mean()
    se = influence.std(ddof=1) / math.sqrt(1024)
    return dict(status='nominal outcome-only diagnostic; not empirical final precision',
                contrast='log score of reference logit + independent Rademacher*0.15 minus reference log score',
                reference='Jeffreys (n1+0.5)/(n+1), other contiguous half of pilot parents',
                sign_stream='PCG64([260405,9,parent]).integers(2,size=128), independent of anchor stream',
                pilot_parents=p, assessment_parents=1024, estimate_nat_per_eligible=float(d),
                se_projected=float(se), halfwidth95_projected=float(t.ppf(.975, 1023)*se),
                mde80_projected=float(2.80*se),
                formula='sd_p((sum_diff_p - pooled_diff*n_eligible_p)/mean(n_eligible_p),ddof=1)/sqrt(1024); MDE=2.80*SE')


def pilot(bank, ids, hot, mass, prevalence, geo, provenance):
    parent, y, s = (bank[k] for k in ('parent', 'y', 'stratum'))
    row = np.searchsorted(ids, s)
    require(np.array_equal(ids[row], s), 'observed impossible stratum')
    code = 2 * row + y
    by_parent = np.stack([np.bincount(x, minlength=2*len(ids)) for x in code]).reshape(len(parent), len(ids), 2)
    counts = by_parent.sum(axis=0); diversity = (by_parent > 0).sum(axis=0)
    both = (counts > 0).all(axis=1); total = counts.sum(axis=1); observed = total > 0
    theoretical_both = (hot > 0) & (hot < 1); overlap = (hot >= .05) & (hot <= .95)
    require(not np.any((counts[:, 1] > 0) & (hot == 0)) and not np.any((counts[:, 0] > 0) & (hot == 1)),
            'observed outcome impossible under theoretical law')
    eligible = overlap & (counts >= 32).all(axis=1) & (diversity >= 16).all(axis=1)
    keep = eligible[row]; n = int(keep.sum()); yes = int(y[keep].sum()); positives = int(y.sum())
    probs = np.divide(counts, total[:, None], out=np.zeros_like(counts, dtype=float), where=total[:, None] > 0)
    ent = -xlogy(probs, probs).sum(axis=1)
    def ratio(x, denominator):
        return float(x / denominator) if denominator else None
    stats = dict(stratum=ids, hot_mass=mass, n0=counts[:, 0], n1=counts[:, 1], parents0=diversity[:, 0],
                 parents1=diversity[:, 1], parents_both=(by_parent > 0).all(axis=2).sum(axis=0),
                 eligible=eligible, empirical_entropy_nat=ent)
    report = dict(geometry=geo, provenance=provenance, label='betti1(open occupied #04 patch)>0',
                  topology_words_checked=4096, positive_words=142, anchors=y.size, positives=positives,
                  parent_count=len(parent), eligible_strata=int(eligible.sum()), eligible_anchors=n,
                  eligible_positive= yes, retained_fraction=n/y.size,
                  retained_fraction_of_positives=ratio(yes, positives), retained_positive_per_all=yes/y.size,
                  retained_parents=int(keep.any(axis=1).sum()), retained_positive_parents=int((keep & (y == 1)).any(axis=1).sum()),
                  retained_negative_parents=int((keep & (y == 0)).any(axis=1).sum()),
                  possible_strata=len(ids), possible_two_outcome_strata=int(theoretical_both.sum()),
                  possible_probability_overlap_strata=int(overlap.sum()), observed_strata=int(observed.sum()),
                  observed_two_outcome_strata=int(both.sum()), fraction_observed_strata_with_two_outcomes=ratio(both.sum(), observed.sum()),
                  fraction_all_anchors_in_observed_two_outcome_strata=float(total[both].sum()/y.size),
                  fraction_positives_in_observed_two_outcome_strata=ratio(counts[both,1].sum(), positives),
                  fraction_all_anchors_in_theoretical_overlap=float(total[overlap].sum()/y.size),
                  empirical_conditional_entropy_nat=float(total @ ent / y.size),
                  eligible_empirical_conditional_entropy_nat=ratio(total[eligible] @ ent[eligible], n),
                  beta0_unconditional_prevalence=prevalence, beta0_eligible_probability_mass=float(mass[eligible].sum()),
                  theoretical_law='C(k,a)*(2400)_a*(3600)_(k-a)*(2000)_(12-k)/(8000)_12 per occupancy word with k occupied/a A; (m)_j falling factorial. Exact integer sums before ratios.',
                  precision=precision(parent, y, row, eligible))
    return ids[eligible], stats, report


def write_np(path, writer, *args, **kwargs):
    with path.open('xb') as f:
        writer(f, *args, **kwargs)


def run(role, commit):
    design = PUBLIC / 'design.json'; lineage = PUBLIC / 'lineage.json'
    for file in (SCRIPT, design, lineage):
        locked = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{file.relative_to(ROOT)}'])
        require(locked == file.read_bytes(), f'{file.name} differs from release commit')
    expected = json.loads(lineage.read_bytes())['private_roster_sha256']
    require(digest(PRIVATE/'rosters.npz') == expected, 'roster not bound to public lineage freeze')
    geo = geometry(); label, base, ids, hot, mass, prevalence = catalogue()
    pins = {p['release_index']: p for p in json.loads((DATA/'release-index.json').read_bytes())['endpoints']}
    all_rosters = rosters(pins); idx = ROLES[role][0]; roster = all_rosters[role]
    bank_path = PRIVATE / (role + '.npz'); receipt = PUBLIC / 'pilot.json'
    outputs = [bank_path]
    if role == 'pilot':
        outputs.extend([PRIVATE/'eligible.npy', PUBLIC/'hot-law.npz', PUBLIC/'pilot-support.npz', receipt])
    else:
        require(receipt.is_file() and (PRIVATE/'eligible.npy').is_file(), 'pilot freeze absent')
        prior = json.loads(receipt.read_bytes())['provenance']
        require(prior['rosters_sha256'] == digest(PRIVATE/'rosters.npz') and
                prior['eligible_sha256'] == digest(PRIVATE/'eligible.npy'), 'changed roster or eligible list')
    for out in outputs:
        require(not out.exists(), f'refusing to overwrite {out}')
    state = archive_rows(role, roster, pins[idx]); bank = extract(role, roster[0], state, label, base)
    if role == 'pilot':
        provenance = dict(commit=commit, rosters_sha256=digest(PRIVATE/'rosters.npz'), design_sha256=digest(design),
                          source_sha256=PINS[idx][1], archived_lut_source_sha256=digest(ARCHIVED),
                          betti_le_i8_sha256=hashlib.sha256(topology_table().astype('<i8').tobytes()).hexdigest())
        kept, stats, report = pilot(bank, ids, hot, mass, prevalence, geo, provenance)
        write_np(PRIVATE/'eligible.npy', np.save, kept, allow_pickle=False)
        write_np(PUBLIC/'hot-law.npz', np.savez_compressed, stratum=ids, p=hot)
        write_np(PUBLIC/'pilot-support.npz', np.savez_compressed, **stats)
    write_np(bank_path, np.savez, **bank)
    if role == 'pilot':
        provenance.update(eligible_sha256=digest(PRIVATE/'eligible.npy'), pilot_bank_sha256=digest(bank_path),
                          hot_law_sha256=digest(PUBLIC/'hot-law.npz'), sufficient_statistics_sha256=digest(PUBLIC/'pilot-support.npz'))
        with receipt.open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=tuple(ROLES)); parser.add_argument('--release-commit', required=True)
    args = parser.parse_args(); run(args.role, args.release_commit)
