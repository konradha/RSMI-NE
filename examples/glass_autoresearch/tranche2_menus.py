"""Exact four-pair support on a registered one-direction empirical face table (item 22)."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from zipfile import ZipFile

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy.stats import t
from face_bank import code, FACES, TILE, POW

ROOT = Path(__file__).resolve().parents[2] / 'results/l20-autoresearch'
OUT = ROOT / 'tranche2'
WITNESS = ROOT / 'fifteen/09-witness.npz'
WITNESS_SHA = 'ce6de1b599bd8a7a19d3af88a54658a327264ebb5b3382fb1424f97c8c162363'
BANK_SHA = 'e1ce4a730cb8c79106bfb278803238ca2847eaa2c03479bfcf8e26e8520f0416'
BOUNDS = {'fit': (0, 4096), 'development': (4096, 4608), 'audit': (4608, 5120)}
V, Y, A = 6561, 81, 4
FACE_POW = (1, 3, 9, 27)
POLICY = OUT / '22-policy.npz'
LOCK = OUT / '22-lock.json'
DESIGN = OUT / '22-design.json'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def report(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def rows(z, name, start, stop, trailing):
    # Stored uncompressed by tranche2_bank. Read only the requested rows, even for species.
    with z.open(f'{name}.npy') as handle:
        if np.lib.format.read_magic(handle) != (1, 0):
            raise ValueError('unexpected bank NPY version')
        shape, order, dtype = np.lib.format.read_array_header_1_0(handle)
        if shape != (5120, *trailing) or order or dtype.kind not in 'iu':
            raise ValueError(f'unexpected {name} bank header: {shape}, {order}, {dtype}')
        width = int(np.prod(trailing)) * dtype.itemsize
        handle.seek(start * width, 1)
        return np.frombuffer(handle.read((stop-start) * width), dtype=dtype).reshape(-1, *trailing)


def cohort(path, split):
    meta = json.loads(path.with_suffix('.json').read_text())
    if (meta['index'], meta['beta'], meta['fit_stop'], meta['dev_stop'], meta['audit_stop'], meta['bank_sha256']) != (9, 3.05, 4096, 4608, 5120, BANK_SHA):
        raise ValueError('unregistered bank receipt')
    a, b = BOUNDS[split]
    with ZipFile(path) as z:
        # Lineage metadata, not species, are checked for independence across all splits.
        parent, chain, seed = (rows(z, name, 0, 5120, ()) for name in ('parent', 'chain_id', 'chain_seed'))
        if len(np.unique(parent)) != 5120 or len(np.unique(np.stack((chain, seed), axis=1), axis=0)) != 5120:
            raise ValueError('duplicated parent or lineage')
        states = rows(z, 'species', a, b, (20, 20, 20))
    if len(states) != b-a or states.max() > 2 or states.min() < 0:
        raise ValueError('incorrect species cohort')
    return parent[a:b], states


def frozen():
    if digest(WITNESS) != WITNESS_SHA:
        raise ValueError('wrong #09 witness')
    with np.load(WITNESS, allow_pickle=False) as data:
        return data['payoff'], data['maps'], data['policies'][[0, 1], [0, 1]]


def exact_scores(count, u):
    # Binary64 is dyadic: no rounding in sums, per-word choices, or between-menu ties.
    ratios = [float(x).as_integer_ratio() for x in u.ravel()]
    scale = max(den for _, den in ratios)
    coeff = np.array([num * (scale // den) for num, den in ratios], dtype=object).reshape(u.shape)
    return np.stack([count.astype(object) @ pay for pay in coeff]), scale


def enumerate_menus(scores, k):
    n_actions = scores.shape[-1]
    pair = (scores[0, :, :, None] + scores[1, :, None, :]).reshape(len(scores[0]), -1).T.copy()
    best, best_menu, best_lut, ties, runner, num = -1, None, None, 0, -1, 0
    for menu in itertools.combinations(range(n_actions**2), k):
        view = pair[list(menu)]
        value = int(view.max(axis=0).sum())
        num += 1
        if value > best:
            runner, best, best_menu, best_lut, ties = best, value, menu, view.argmax(axis=0).astype('u1'), 1
        elif value == best:
            ties += 1
        elif value > runner:
            runner = value
    policy = np.array(divmod(np.array(best_menu), n_actions), dtype='u1')
    return best, np.array(best_menu), best_lut, policy, num, ties, runner


def controls():
    # Asymmetric five-word, three-outcome toy (including one unsupported word).
    rng = np.random.default_rng(2200)
    count = rng.integers(1, 17, (5, 3)); count[-1] = 0
    payoff = rng.integers(0, 9, (2, 3, 2)) / 8
    s, scale = exact_scores(count, payoff)
    best, _, lut, _, n, _, _ = enumerate_menus(s, 2)
    brute = max(int(sum(s[t, np.array(h) == label].sum(axis=0).max() for label in range(2) for t in range(2)))
                for h in itertools.product(range(2), repeat=len(count)))
    zero, menu0, lut0, _, _, _, _ = enumerate_menus(s * 0, 2)
    padding_best = max(int((s[0, :, a] + s[1, :, b]).sum()) for a, b in itertools.product(range(2), repeat=2))
    if best != brute or n != 6 or padding_best > best or lut[-1] or zero or menu0.tolist() != [0, 1] or lut0.any():
        raise ArithmeticError('small enumeration, tie or padding mismatch')
    # Independent periodic scalar arithmetic: test every anchor of a deliberately asymmetric box.
    box = rng.integers(0, 3, (1, 20, 20, 20), dtype='u1')
    vv, yy = code(box, TILE, POW)[0], code(box, FACES[0], FACE_POW)[0]
    if FACES[0].tolist() != [[-2,0,0],[-2,0,1],[-2,1,0],[-2,1,1]]:
        raise ValueError('J=0 is not the registered -z face')
    for z, y, x in itertools.product(range(20), repeat=3):
        v = sum(3**(4*k+2*j+i) * int(box[0,(z+k)%20,(y+j)%20,(x+i)%20]) for k,j,i in itertools.product(range(2), repeat=3))
        f = sum(3**(2*j+i) * int(box[0,(z-2)%20,(y+j)%20,(x+i)%20]) for j,i in itertools.product(range(2), repeat=2))
        if v != vv[z,y,x] or f != yy[z,y,x]:
            raise ValueError('periodic direct-index parity failed')
    return dict(toy_menus=n, toy_encoders=2**len(count), toy_count_scale=scale, toy_brute=brute,
                toy_menu_support=best, toy_single_action_support=padding_best, unseen_label=int(lut[-1]),
                all_tied_first_menu=menu0.tolist(), direct_word_and_face_parity_sites=8000, mismatches=0)


def fit(bank):
    artifact = OUT / '22-fit-counts.npz'
    if any(path.exists() for path in (LOCK, POLICY, artifact)):
        raise FileExistsError('fit already frozen')
    nulls = controls()
    u, maps, fixed = frozen()
    parents, states = cohort(bank, 'fit')
    count = np.zeros((V, Y), dtype='int64')
    for batch in range(0, len(states), 32):
        box = states[batch:batch+32]
        v = code(box, TILE, POW); y = code(box, FACES[0], FACE_POW)
        count += np.bincount((Y*v+y).ravel(), minlength=V*Y).reshape(V, Y)
    opportunities = int(count.sum())
    if opportunities != 8000 * len(parents):
        raise ValueError('opportunity denominator mismatch')
    scores, scale = exact_scores(count, u[:, 0])
    support, menu, lut, policy, nm, ties, runner = enumerate_menus(scores, A)
    vals = [int(scores[t, np.arange(V), policy[t, lut]].sum()) for t in range(2)]
    targets = [int(scores[t, np.arange(V), fixed[t, 0, maps[t]]].sum()) for t in range(2)]
    if sum(vals) != support or nm != 1820:
        raise ArithmeticError('deployed code does not attain enumerated support')
    den = opportunities * scale
    # Second, independent count-weighted scoring path, separate utilities on the actual face outcomes.
    discrepancy = max(abs(np.einsum('vy,vy->', count, u[t, 0, :, policy[t, lut]], optimize=False)/opportunities - vals[t]/den) for t in range(2))
    if discrepancy > 1e-10:
        raise ArithmeticError('direct utility mismatch')
    np.savez_compressed(POLICY, lut=lut, policies=policy, menu=menu.astype('u1'))
    np.savez_compressed(artifact, count=count.astype('u4'), parent=parents)
    gap = (sum(targets)-support)/(2*den)
    deficits = [(a-b)/den for a,b in zip(targets, vals)]
    report(LOCK, dict(question=22, split='fit', direction='J=0, -z, normal -2, one unseen layer',
            utility_order=['discrete-target task', 'affine-target task'], parents=len(parents), opportunities=opportunities,
            nonempty_visible_words=int((count.sum(1)>0).sum()), nonempty_face_words=int((count.sum(0)>0).sum()),
            nonempty_joint_cells=int(np.count_nonzero(count)), design_sha256=digest(DESIGN), witness_sha256=WITNESS_SHA,
            bank_receipt_sha256=digest(bank.with_suffix('.json')), bank_sha256=BANK_SHA, script_sha256=digest(Path(__file__)),
            policy_file=POLICY.name, policy_sha256=digest(POLICY), count_file=artifact.name, count_sha256=digest(artifact),
            menu=menu.tolist(), policies=policy.tolist(), menu_count=nm, maximizing_menus=ties,
            runner_up_strictly_smaller_by=(support-runner)/(2*den), symbol_counts=np.bincount(lut, weights=count.sum(1), minlength=4).astype(int).tolist(),
            exact_support_numerator=str(support), exact_support_denominator=str(2*den), payoff_integer_scale=str(scale),
            achievable_utilities=[v/den for v in vals], matched_target_utilities=[v/den for v in targets],
            separate_deficits=deficits, achieved_weighted_lower=support/(2*den), exhaustive_weighted_upper=support/(2*den),
            matched_target_average=sum(targets)/(2*den), target_average_minus_support=gap,
            finite_table_exact_obstruction=bool(gap>0), finite_table_practical_obstruction=bool(gap>0.001),
            fitted_compromise_meets_both_within_margin=max(deficits)<=0.001, direct_utility_parity_error=float(discrepancy), controls=nulls,
            scope='Exact support only on this fit empirical table; does not exclude other population rules. Reused historical TRAIN witnesses.'))


def interval(x, level):
    sd = x.std(ddof=1); se = sd / np.sqrt(len(x)); width = t.ppf((1+level)/2, len(x)-1) * se
    return dict(mean=float(x.mean()), sd=float(sd), se=float(se), level=level, lo=float(x.mean()-width), hi=float(x.mean()+width))


def summarize(values):
    # axes: parent, utility, implementation (compromise / matched target)
    gaps = values[:,:,1] - values[:,:,0]
    paired = [interval(q, .975) for q in gaps.T]
    return dict(achieved=[interval(q, .95) for q in values[:,:,0].T], target=[interval(q, .95) for q in values[:,:,1].T],
                paired_target_minus_achieved=paired, mean_paired_deficit=interval(gaps.mean(1), .95),
                joint_within_margin=all(q['hi']<=.001 for q in paired), material_miss=any(q['lo']>.001 for q in paired))


def evaluate(bank, split, release):
    output = OUT / f'22-{split}.json'; evidence = OUT / f'22-{split}-parents.npz'
    if output.exists() or evidence.exists():
        raise FileExistsError('split already scored')
    if split == 'audit' and (not release or digest(LOCK) != release):
        raise ValueError('explicit audit release with published policy-lock SHA required')
    lock = json.loads(LOCK.read_text())
    if (lock['policy_sha256'],lock['design_sha256'],lock['script_sha256']) != (digest(POLICY),digest(DESIGN),digest(Path(__file__))):
        raise ValueError('frozen artifact changed')
    u, incumbent, fixed = frozen()
    with np.load(POLICY, allow_pickle=False) as p:
        lut, policy = p['lut'], p['policies']
    parent, states = cohort(bank, split)
    values = np.empty((len(parent), 6, 2, 2))
    vmass = np.zeros(V, dtype='i8'); ymass = np.zeros(Y, dtype='i8')
    for i in range(0, len(parent), 8):
        box = states[i:i+8]; v = code(box, TILE, POW).reshape(len(box), -1)
        vmass += np.bincount(v.ravel(), minlength=V)
        for j, face in enumerate(FACES):
            y = code(box, face, FACE_POW).reshape(v.shape)
            if j == 0:
                ymass += np.bincount(y.ravel(), minlength=Y)
            for k in range(2):
                for m, actions in enumerate([policy[k,lut], fixed[k,j,incumbent[k]]]):
                    values[i:i+len(box),j,k,m] = u[k,j,y,actions[v]].mean(axis=1)
    np.savez_compressed(evidence, parent=parent, values=values)
    result = summarize(values[:,0]); hwidth = [(r['hi']-r['lo'])/2 for r in result['paired_target_minus_achieved']]
    report(output, dict(split=split, design_sha256=digest(DESIGN), policy_lock_sha256=digest(LOCK),
        parents=len(parent), direction_opportunities=int(vmass.sum()), nonempty_visible_words=int(np.count_nonzero(vmass)),
        nonempty_face_words=int(np.count_nonzero(ymass)), utility_order=lock['utility_order'], matched_direction=result,
        equal_six_directions_with_unchanged_policy_constructive_only=summarize(values.mean(axis=1)),
        parent_values_file=evidence.name, parent_values_sha256=digest(evidence),
        evidence_axes=['parent', 'direction: -z,+z,-y,+y,-x,+x', 'utility: discrete,affine', 'deployment: compromise,matched target'],
        same_size_projected_simultaneous_halfwidth=hwidth if split=='development' else None,
        inference='Parent-clustered Student t. Separate paired deficit intervals simultaneously 95% within J=0 (Bonferroni). This evaluates only the frozen solution; no held-out upper bound. Historical #09 data reuse.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('split', choices=list(BOUNDS)); p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--release-lock-sha256'); args = p.parse_args()
    if args.split == 'fit':
        fit(args.bank)
    else:
        evaluate(args.bank, args.split, args.release_lock_sha256)
