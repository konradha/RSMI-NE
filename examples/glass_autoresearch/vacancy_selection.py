"""Score vacancy-only tripod selection among matched, geometrically ambiguous cages."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import boundary_bank as bank
import near_cage_law as colored
import vacancy_free_energy as species
from boundary_models import ratio_interval
from near_cage_test import load_rows, read_lock, require, scored_logs, write_json
from rate import digest, table


ORDER = (7, 9, 0)
CAP, SEED = 4, 772191
SIGNATURE = (3, 3, 5)
N_OCC, N_A = 6000, 2400
MODELS = ('vacancy', 'feasible', 'colored')
SRC = Path(__file__).resolve()
SHAPE = (bank.L,) * 3
HALO = np.asarray(colored.OFFSETS)
INNER = np.abs(HALO).sum(axis=1) == 2
ARM_SETS = np.asarray(list(combinations(range(6), 3)))
TRIPOD = np.abs(bank.NN[ARM_SETS].sum(axis=1)).sum(axis=1) == 3
# Rows index occupied distance-two sites; columns index arms or outer sites.
INCIDENT = (np.abs(HALO[INNER, None] - bank.NN).sum(axis=2) == 1).astype(np.int8)
ARM_TOUCH = INCIDENT[:, ARM_SETS].sum(axis=2).T
TO_OUTER = (np.abs(HALO[~INNER, None] - HALO[INNER]).sum(axis=2) == 1).astype(np.int8)
ARM_HIST = np.bincount(SIGNATURE, minlength=7)


class Exterior:
    """Coordinates on the punctured torus. No slots exist for center/arm bits."""

    def __init__(self, nn):
        xyz = np.indices(SHAPE, dtype=np.int16).reshape(3, -1)
        radius = np.minimum(xyz, bank.L - xyz).sum(axis=0)
        sites = np.flatnonzero(radius >= 2)
        index = np.full(bank.VOLUME, -1, dtype=np.int16)
        index[sites] = np.arange(len(sites))
        far = np.flatnonzero(radius > 2)
        self.far = index[far]
        self.neighbors = index[nn[far]].T.copy()
        self.halo = index[np.ravel_multi_index(HALO.T, SHAPE, mode='wrap')]
        # Structural invariance proof on all sites, not a check on observed fillings:
        # no bit in the hidden cross occurs in ANY closed far-site neighborhood.
        require(len(sites) == bank.VOLUME-7 and np.all(self.far >= 0) and
                np.all(self.neighbors >= 0) and np.all(self.halo >= 0), 'stencil reaches hidden cross')
        self.size = len(sites)
        self.shift = [((np.arange(bank.L, dtype=np.int16)[:, None] + xyz[d, sites]) % bank.L) * stride
                      for d, stride in enumerate((bank.L**2, bank.L, 1))]

    def extract(self, masks, parents, centers):
        require(masks.dtype == np.bool_, 'extract expects only occupied bits')
        z, y, x = np.unravel_index(centers, SHAPE)
        source = self.shift[0][z] + self.shift[1][y] + self.shift[2][x]
        return masks[parents[:, None], source]

    def unaffected_histogram(self, exterior):
        require(exterior.dtype == np.bool_ and exterior.ndim == 2 and exterior.shape[1] == self.size,
                'expected boolean exterior only, with the central cross removed')
        k = np.zeros((len(exterior), len(self.far)), np.uint8)
        for nb in self.neighbors:
            k += exterior[:, nb]
        occupied = exterior[:, self.far]
        return histograms(k, occupied)


def histograms(k, occupied):
    bins = k + 7*np.arange(len(k))[:, None]
    return np.bincount(bins[occupied], minlength=7*len(k)).reshape(-1, 7)


def admissible(inner):
    require(inner.dtype == np.bool_, 'feasibility must not receive labels')
    a = inner @ INCIDENT
    return (np.sort(a[:, ARM_SETS], axis=2) == SIGNATURE).all(axis=2)


def vacancy_forecast(exterior, geometry, beta):
    """Mask-only predictor; output shape probabilities and both log scores per model."""
    unaffected = geometry.unaffected_histogram(exterior)
    halo = exterior[:, geometry.halo]
    inner = halo[:, INNER]
    ok = admissible(inner)
    nt, no = ok[:, TRIPOD].sum(axis=1), ok[:, ~TRIPOD].sum(axis=1)
    require(np.all((nt > 0) & (no > 0)), 'predict only on the registered ambiguous subset')
    row, orient = np.nonzero(ok)
    m0 = halo[:, ~INNER] @ TO_OUTER
    k = m0[row] + ARM_TOUCH[orient]
    require(k.min() >= 0 and k.max() <= 6, 'invalid candidate coordination')
    census = unaffected[row] + ARM_HIST + histograms(k, inner[row])
    require(np.all(census >= 0) and np.all(census.sum(axis=1) == N_OCC), 'invalid complete candidate census')
    # Duplicates still contribute their orientation multiplicity after inversion.
    unique, inv = np.unique(census, axis=0, return_inverse=True)
    lnz = species.log_partition(unique, beta, N_A)[inv.reshape(-1)]
    require(lnz.shape == row.shape and np.isfinite(lnz).all(), 'invalid canonical ln Z')
    # Subtract the largest complete ln Z before summing either class.
    weights = np.full(ok.shape, -np.inf)
    weights[row, orient] = lnz
    weights -= weights.max(axis=1, keepdims=True)
    tripod, other = (logsumexp(weights[:, c], axis=1) for c in (TRIPOD, ~TRIPOD))
    normalizer = np.logaddexp(tripod, other)
    lp, lt = tripod-normalizer, other-normalizer
    p = np.exp(lp)
    uniform = nt/(nt+no)
    base = (uniform, np.log(uniform), np.log1p(-uniform))
    if beta == 0:
        require(np.all(weights[ok] == 0.) and np.allclose(p, uniform, atol=2e-15, rtol=0.),
                'beta zero did not give equal orientation weights')
        # Mathematically equal; use the identical rational evaluation for bitwise score parity.
        p, lp, lt = base
    return (p, lp, lt), base, nt, no, unaffected


def check_far(mask, center, nn, expected):
    """Independent observed-parent check; not an input to any forecast."""
    xyz = np.indices(SHAPE).reshape(3, -1)
    delta = np.abs(xyz - np.asarray(np.unravel_index(center, SHAPE))[:, None])
    d = np.minimum(delta, bank.L-delta).sum(axis=0)
    occupied_far = mask & (d > 2)
    direct = np.bincount(mask[nn[occupied_far]].sum(axis=1), minlength=7)
    np.testing.assert_array_equal(direct, expected)
    # For each occupied far site the largest possible coordination change between
    # ANY two cross fillings is bounded by the number of incident cross neighbors.
    require(not np.any(d[nn[occupied_far]] <= 1), 'far coordinator depends on cross filling')


def select_centers(parents, n_amb, ids, index):
    """Rows are already sorted by parent and native flat center; never accepts y."""
    keep = []
    end = 0
    for i in np.flatnonzero(n_amb):
        n = n_amb[i]
        idx = np.arange(n) if n <= CAP else np.sort(bank.pcg(SEED, index, ids[i]).choice(n, CAP, replace=False))
        keep.append(end + idx)
        end += n
    require(end == len(parents), 'incorrect ambiguous census')
    return np.concatenate(keep) if keep else np.empty(0, np.intp)


def collect(states, ids, index, beta, nn, shell, geometry):
    count_keys = ('matches', 'tripods', 'ambiguous', 'ambiguous_tripods', 'drawn', 'drawn_tripods', 'colored_log_repairs')
    data = {key: np.zeros(len(ids), np.int32) for key in count_keys}
    sums = ('forecast', 'log_score', 'observed_minus_forecast', 'accuracy')
    data.update({key: np.zeros((len(ids), len(MODELS))) for key in sums})
    checked = None
    for start in range(0, len(ids), 32):
        block = states[start:start+32]
        rows = slice(start, start+len(block))
        mask = block != 0
        require(np.all(mask.sum(axis=1) == N_OCC) and np.all((block == 1).sum(axis=1) == N_A),
                'archive composition differs from registration')
        pp, aa, code, yy = bank.candidates(mask, nn)
        match = code == (49*SIGNATURE[0] + 7*SIGNATURE[1] + SIGNATURE[2])
        pp, aa, yy = pp[match], aa[match], yy[match]
        # Only the outcome census sees the actual arrangement. Ambiguity uses
        # exterior occupancy, with exactly the same enumeration for both orbits.
        eligible = admissible(mask[pp[:, None], shell[aa][:, INNER]])
        nt, no = eligible[:, TRIPOD].sum(axis=1), eligible[:, ~TRIPOD].sum(axis=1)
        require(np.all(np.where(yy, nt, no) > 0), 'observed orbit not feasible')
        amb = (nt > 0) & (no > 0)
        def census(p, v=None):
            return np.bincount(p, weights=v, minlength=len(block))
        data['matches'][rows], data['tripods'][rows] = census(pp), census(pp, yy)
        p, a, y = pp[amb], aa[amb], yy[amb]
        n = census(p)
        data['ambiguous'][rows], data['ambiguous_tripods'][rows] = n, census(p, y)
        take = select_centers(p, n, ids[rows], index)
        if not len(take):
            continue
        p, a, y = p[take], a[take], y[take]
        w = n[p]/np.minimum(n[p], CAP)
        data['drawn'][rows], data['drawn_tripods'][rows] = census(p), census(p, y)
        ext = geometry.extract(mask, p, a)
        vacancy, feasible, nt2, no2, far = vacancy_forecast(ext, geometry, beta)
        require(np.array_equal(nt2, nt[amb][take]) and np.array_equal(no2, no[amb][take]),
                'exterior registration differs between screen and prediction')
        if checked is None:
            check_far(mask[p[0]], a[0], nn, far[0])
            checked = dict(native_row=int(ids[start+p[0]]), center=int(a[0]), far_histogram=far[0].tolist(),
                           max_cross_filling_effect_on_far_k=0)
        # Only the separate comparator receives species, never the central cross.
        oracle = colored.predict(block[p[:, None], shell[a]], beta, signature=SIGNATURE)
        require(np.array_equal(oracle['count_tripod'], nt2) and np.array_equal(oracle['count_T'], no2),
                'colored oracle uses different orientations')
        q = oracle['exact']
        lq, lT, repair = scored_logs(q, np.ones(len(q), bool))
        data['colored_log_repairs'][rows] = census(p, repair)
        forecasts = (vacancy, feasible, (q, lq, lT))
        if beta == 0:
            for triple in forecasts:
                for u, v in zip(triple, feasible):
                    np.testing.assert_array_equal(u, v)
        for j, (prob, lp, lt) in enumerate(forecasts):
            require(np.all((0 <= prob) & (prob <= 1)) and np.isfinite(lp).all() and np.isfinite(lt).all(),
                    'invalid interior-support forecast')
            vals = (prob, np.where(y, lp, lt), y-prob, (prob >= .5) == y)
            for name, val in zip(sums, vals):
                data[name][rows, j] = census(p, w*val)
    return data, checked


def summarize(data):
    n, y, match, all_y = (data[k] for k in ('ambiguous', 'ambiguous_tripods', 'matches', 'tripods'))
    score = data['log_score']
    return dict(matched_centers=int(match.sum()), ambiguous_centers=int(n.sum()), scored_centers=int(data['drawn'].sum()),
                native_parents=len(n), matched_parents=int(np.count_nonzero(match)), ambiguous_parents=int(np.count_nonzero(n)),
                scored_parents=int(np.count_nonzero(data['drawn'])), forced_centers=int((match-n).sum()),
                forced_tripods=int((all_y-y).sum()), tripod_prevalence_all=ratio_interval(all_y, match),
                ambiguous_tripods=int(y.sum()), tripod_prevalence_ambiguous=ratio_interval(y, n),
                forced_fraction=ratio_interval(match-n, match), colored_log_repairs=int(data['colored_log_repairs'].sum()),
                models={model: {key: ratio_interval(data[key][:, j], n) for key in
                                ('log_score', 'forecast', 'observed_minus_forecast', 'accuracy')}
                        for j, model in enumerate(MODELS)},
                paired_gain_nat_per_ambiguous_center={f'{MODELS[i]}_minus_{MODELS[j]}':
                    ratio_interval(score[:, i]-score[:, j], n) for i, j in ((0, 1), (2, 0), (2, 1))})


def plot(path, reports, margin):
    import matplotlib.pyplot as plt
    fig, (gain, cal) = plt.subplots(1, 2, figsize=(9.1, 3.1), layout='constrained', sharey=True)
    for i, r in enumerate(reports):
        s = r['scores']
        for ax, v, dy, color, text in (
                (gain, s['paired_gain_nat_per_ambiguous_center']['vacancy_minus_feasible'], -.10, '#186674', 'Mask − feasible'),
                (gain, s['paired_gain_nat_per_ambiguous_center']['colored_minus_vacancy'], .10, '#a34a25', 'Colored − mask'),
                (cal, s['models']['vacancy']['observed_minus_forecast'], -.10, '#186674', 'Mask'),
                (cal, s['models']['colored']['observed_minus_forecast'], .10, '#a34a25', 'Colored')):
            if v['lo'] is not None:
                x = v['estimate']
                ax.errorbar(x, i+dy, xerr=[[x-v['lo']], [v['hi']-x]], fmt='o', color=color, capsize=3,
                            label=text if i == 0 else None)
    gain.axvline(margin, ls='--', lw=1, color='.5', label='Primary margin')
    gain.set(xlabel='Paired gain (nat / ambiguous center)', ylabel='β', yticks=range(3),
             yticklabels=['2.372 (primary)', '3.05', '0'])
    cal.set_xlabel('Observed tripod − forecast')
    gain.invert_yaxis()
    for ax in (gain, cal):
        ax.axvline(0, color='.45', lw=.7); ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='best', fontsize=8, frameon=False)
    fig.suptitle('Matched 3,3,5 cages permitting both shapes · parent-clustered 95% intervals')
    with path.open('xb') as f:
        fig.savefig(f, format='png', dpi=160)
    plt.close(fig)


def run(args):
    out = args.out
    plan = read_lock(out/'vacancy-design.json', args.lock_sha256)
    near = read_lock(out/'near-design.json', plan['near_design_sha256'])
    margin = plan['margin_nat_per_ambiguous_center']
    require(plan['schema'] == 'l20.vacancy-selection-design.v1' and tuple(plan['signature']) == SIGNATURE and
            tuple(plan['cohort_order']) == tuple(near['cohort_order']) == ORDER and plan['ambiguous_cap_per_parent'] == CAP
            and plan['center_sample_seed'] == SEED and margin == .01 and
            plan['composition'] == dict(volume=bank.VOLUME, occupied=N_OCC, n_a=N_A), 'incorrect score registration')
    outputs = ['vacancy-run.json', 'vacancy-results.json', 'vacancy-ambiguous.png'] + [
        path for i in ORDER for path in (f'vacancy-score-{i:02}.json', f'vacancy-parents-{i:02}.npz')]
    if any((out/name).exists() for name in outputs):
        raise FileExistsError('vacancy selection already claimed or output exists; no replacement scoring')
    nn, shell = table(bank.NN), table(HALO)
    geometry = Exterior(nn)
    require(TRIPOD.sum() == 8 and (~TRIPOD).sum() == 12, 'unexpected cubic orbits')
    species.verify()
    files = [SRC, Path(species.__file__), Path(colored.__file__), Path(bank.__file__)] + [
        SRC.with_name(name+'.py') for name in ('near_cage_test', 'vacancy_dipole', 'rate', 'boundary_models')] + [
        SRC.parents[1]/(name+'.py') for name in ('l20_pairs', 'l20_operators_static')]
    claim = dict(schema='l20.vacancy-selection-run.v1', vacancy_design_sha256=args.lock_sha256,
                 near_design_sha256=plan['near_design_sha256'], analytic_partition_check='passed',
                 far_stencils_disjoint_from_cross=len(geometry.far), implementation_sha256={p.name: digest(p) for p in files})
    write_json(out/'vacancy-run.json', claim)
    run_sha = digest(out/'vacancy-run.json')
    hashes, reports = {}, []
    for index in ORDER:
        pin = near['release'][str(index)]
        ids, keys, states, split = load_rows(args.data, index, pin)
        data, checked = collect(states, ids, index, pin['beta'], nn, shell, geometry)
        del states
        tab = out/f'vacancy-parents-{index:02}.npz'
        with tab.open('xb') as f:
            np.savez_compressed(f, parent=ids, chain_seed=keys[:, 0], chain_id=keys[:, 1], models=MODELS, **data)
        hashes[tab.name] = digest(tab)
        scores = summarize(data)
        primary_gain = scores['paired_gain_nat_per_ambiguous_center']['vacancy_minus_feasible']
        decision = ('descriptive only' if index != 7 else 'inconclusive' if primary_gain['lo'] is None
                    else 'pass' if primary_gain['lo'] >= margin else 'did not pass')
        report = dict(schema='l20.vacancy-selection-score.v1', index=index, **pin, split=split,
                      vacancy_design_sha256=args.lock_sha256, vacancy_run_sha256=run_sha, scores=scores,
                      direct_far_census_check=checked, assessment=decision, parent_table_sha256=hashes[tab.name],
                      lineage_sha256=hashlib.sha256(keys.tobytes()).hexdigest())
        path = out/f'vacancy-score-{index:02}.json'
        write_json(path, report); hashes[path.name] = digest(path)
        reports.append(report)
    figure = out/'vacancy-ambiguous.png'
    plot(figure, reports, margin); hashes[figure.name] = digest(figure)
    result = dict(schema='l20.vacancy-selection-results.v1', vacancy_design_sha256=args.lock_sha256,
                  vacancy_run_sha256=run_sha, primary_index=7, assessment=reports[0]['assessment'],
                  primary_vacancy_minus_feasible=reports[0]['scores']['paired_gain_nat_per_ambiguous_center']['vacancy_minus_feasible'],
                  margin_nat_per_ambiguous_center=margin, interpretation=plan['question'], artifacts_sha256=hashes)
    write_json(out/'vacancy-results.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='Boundary directory containing the design locks')
    parser.add_argument('--lock-sha256', required=True, help='Prior SHA256 of vacancy-design.json')
    run(parser.parse_args())
