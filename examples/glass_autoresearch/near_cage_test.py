"""One-shot near-halo positive control. See results/l20-autoresearch/boundary/near-design.json."""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import boundary_bank as bank
import near_cage_law as law
from boundary_models import ratio_interval
from rate import digest, table
from vacancy_dipole import validation


ORDER = (7, 9, 0)
SIGNATURE = (3, 3, 5)
MODELS = ('exact', 'feasible', 'linear', 'cold_prevalence')
SUBSETS = ('all', 'ambivalent')
NBINS = 10
SRC = Path(__file__).resolve()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def read_lock(path, expected):
    blob = path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == expected, f'{path}: lock digest mismatch')
    return json.loads(blob)


def frozen_plan(out, lock):
    plan = read_lock(out/'near-design.json', lock)
    require(plan['schema'] == 'l20.near-cage-design.v1' and tuple(plan['cohort_order']) == ORDER
            and plan['margin_nat_per_matched_center'] == .01 and tuple(plan['signature']) == SIGNATURE,
            'unexpected near-cage decision or registration')
    cold = read_lock(out/'cold-lock.json', plan['cold_lock_sha256'])
    pairing = read_lock(out/'pairing-lock.json', plan['pairing_lock_sha256'])
    require(pairing['cold_model_lock_sha256'] == plan['cold_lock_sha256'] and cold['index'] == 9
            and pairing['source_sha256'] == cold['source_sha256'] == plan['release']['9']['endpoint_sha256']
            and pairing['signature'] == cold['signature'] == list(SIGNATURE), 'incorrect cold lock or census')
    t, tripod = pairing['train_counts_T_tripod']
    require([t, tripod] == [77991, 48533], 'incorrect cold training census')
    return plan, tripod / (t + tripod)


def load_rows(data, index, pin):
    require(index in ORDER, 'unregistered validation cohort')
    ids, split = bank.native_ids(index, 'validation')
    require(len(ids) == pin['native_validation_parents'] and np.unique(ids).size == len(ids), 'invalid native rows')
    if index == 9:
        # Unlike the full-fold loader, cold_subset never materializes reserved center arrays.
        path = data/'beta-09.npz'
        require(digest(path.with_suffix('.json')) == pin['receipt_sha256'] and
                bank.PINS[9] == (pin['beta'], pin['endpoint_sha256']), 'incorrect cold release registration')
        bank.check_receipt(path, index)
        states, seeds, chains, _ = bank.cold_subset(path, ids)
        keys = np.column_stack((seeds, chains)).astype('<u8')
    else:
        rows, keys, states = validation(data, index, pin)
        require(np.array_equal(ids, rows), 'native validation order differs')
    require(np.unique(keys, axis=0).shape[0] == len(ids), 'repeated parent lineage')
    return ids, keys, states.reshape(len(ids), bank.VOLUME), split


def scored_logs(p, ambivalent):
    """Keep true endpoints; repair ONLY rounded interior-support endpoints for the log."""
    repair = ambivalent & ((p == 0) | (p == 1))
    if repair.any():
        p = p.copy()
        eps = np.finfo(np.float64).eps / 2
        p[repair] = np.clip(p[repair], eps, 1-eps)
    # log(0) is legitimate on a forbidden shape. Neither observed nor expected
    # scoring selects it (skip zero-mass terms explicitly; never multiply 0*-inf).
    with np.errstate(divide='ignore'):
        return np.log(p), np.log1p(-p), repair


def accumulators(n):
    shape = (n, len(SUBSETS))
    out = {k: np.zeros(shape, np.int32) for k in ('n', 'tripod', 'feasible_tripod_masks', 'feasible_T_masks')}
    for k in ('log_score', 'oracle_expected_log_score', 'accuracy', 'oracle_expected_accuracy', 'forecast',
              'repaired_log_endpoint'):
        out[k] = np.zeros((*shape, len(MODELS)), np.int32 if k == 'repaired_log_endpoint' else np.float64)
    for k in ('bin_count', 'bin_tripod', 'bin_forecast', 'bin_oracle'):
        out[k] = np.zeros((*shape, len(MODELS), NBINS), np.int32 if k in ('bin_count', 'bin_tripod') else np.float64)
    return out


def collect(states, beta, prevalence, nn, shell):
    data = accumulators(len(states))
    match = SIGNATURE[0]*49 + SIGNATURE[1]*7 + SIGNATURE[2]
    for first in range(0, len(states), 32):
        block = states[first:first+32]
        p, a, codes, y = bank.candidates(block, nn)
        matched = codes == match
        p, a, y = p[matched], a[matched], y[matched]
        if not len(y):
            continue
        # The only microscopic argument to predict is colored EXTERIOR bytes.
        pred = law.predict(block[p[:, None], shell[a]], beta, signature=SIGNATURE)
        nt, nT = pred['count_tripod'], pred['count_T']
        require(np.all((nt >= 0) & (nt <= 8) & (nT >= 0) & (nT <= 12)) and
                np.all(np.where(y, nt, nT) > 0), 'observed matched arrangement outside geometric support')
        ambivalent = (nt > 0) & (nT > 0)
        oracle = pred['exact']
        forecast = {k: pred[k] for k in MODELS[:-1]}
        forecast['cold_prevalence'] = np.full(len(y), prevalence)
        require(np.allclose(pred['feasible'], nt/(nt+nT), rtol=0, atol=1e-15), 'wrong packing multiplicities')
        if beta == 0:
            require(all(np.allclose(pred[k], pred['feasible'], rtol=0, atol=1e-14) for k in ('exact', 'linear')),
                    'beta=0 does not recover uniform feasible masks')
        rows = slice(first, first+len(block))
        for s, mask in enumerate((np.ones(len(y), bool), ambivalent)):
            pp, yy, q = p[mask], y[mask], oracle[mask]
            def sums(w=None):
                return np.bincount(pp, weights=w, minlength=len(block))
            data['n'][rows, s] = sums()
            data['tripod'][rows, s] = sums(yy)
            data['feasible_tripod_masks'][rows, s] = sums(nt[mask])
            data['feasible_T_masks'][rows, s] = sums(nT[mask])
            for m, name in enumerate(MODELS):
                f = forecast[name][mask]
                require(np.isfinite(f).all() and np.all((f >= 0) & (f <= 1)), 'invalid probability')
                both = ambivalent[mask]
                if name != 'cold_prevalence':
                    require(np.array_equal(f[~both], yy[~both]), 'forced shape must receive unit mass')
                lp, lT, repaired = scored_logs(f, both)
                actual = np.where(yy, lp, lT)
                expected = np.zeros_like(f)
                pos, neg = q > 0, q < 1
                expected[pos] += q[pos] * lp[pos]
                expected[neg] += (1-q[neg]) * lT[neg]
                require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'invalid matched log score')
                call = f >= .5
                for key, value in [('log_score', actual), ('oracle_expected_log_score', expected),
                                   ('accuracy', call == yy), ('oracle_expected_accuracy', np.where(call, q, 1-q)),
                                   ('forecast', f), ('repaired_log_endpoint', repaired)]:
                    data[key][rows, s, m] = sums(value)
                # p==1 goes to the closed final bin, not a separate support class.
                bins = np.minimum((f*NBINS).astype(np.intp), NBINS-1)
                for key, value in [('bin_count', None), ('bin_tripod', yy), ('bin_forecast', f), ('bin_oracle', q)]:
                    data[key][rows, s, m] = np.bincount(pp*NBINS+bins, weights=value,
                                                        minlength=len(block)*NBINS).reshape(-1, NBINS)
    return data


def summarize(data):
    result = {}
    for s, subset in enumerate(SUBSETS):
        n, y = data['n'][:, s], data['tripod'][:, s]
        scores = data['log_score'][:, s]
        contrasts = {f'{a}_minus_{b}': ratio_interval(scores[:, i]-scores[:, j], n)
                     for i, a in enumerate(MODELS) for j, b in enumerate(MODELS) if i < j}
        models = {}
        for m, name in enumerate(MODELS):
            v = {key: data[key][:, s, m] for key in ('log_score', 'oracle_expected_log_score', 'accuracy',
                                                        'oracle_expected_accuracy', 'forecast')}
            row = {key: ratio_interval(value, n) for key, value in v.items()}
            row['observed_minus_forecast'] = ratio_interval(y-v['forecast'], n)
            for k in ('log_score', 'accuracy'):
                row[f'observed_minus_oracle_expected_{k}'] = ratio_interval(v[k]-v[f'oracle_expected_{k}'], n)
            row['rounded_endpoints_repaired_for_logs'] = int(data['repaired_log_endpoint'][:, s, m].sum())
            bins = []
            for b in range(NBINS):
                w, obs, pred, oracle = (data[k][:, s, m, b] for k in ('bin_count','bin_tripod','bin_forecast','bin_oracle'))
                bins.append(dict(lo=b/NBINS, hi=(b+1)/NBINS, n=int(w.sum()), parents=int(np.count_nonzero(w)),
                                 mean_forecast=ratio_interval(pred, w)['estimate'], mean_oracle=ratio_interval(oracle, w)['estimate'],
                                 observed=ratio_interval(obs, w), observed_minus_forecast=ratio_interval(obs-pred, w)))
            row['reliability'] = bins
            models[name] = row
        result[subset] = dict(matches=int(n.sum()), tripod=int(y.sum()), T=int((n-y).sum()),
                              contributing_parents=int(np.count_nonzero(n)), actual_tripod_prevalence=ratio_interval(y,n),
                              models=models, paired_log_score_gains=contrasts)
    n, amb = data['n'].T
    y, y_amb = data['tripod'].T
    result['geometrically_forced'] = dict(fraction=ratio_interval(n-amb, n), tripod=int((y-y_amb).sum()),
                                          T=int(((n-amb)-(y-y_amb)).sum()))
    return result


def plot(path, reports, margin):
    fig, (gain, forced, energy) = plt.subplots(1, 3, figsize=(10, 3.5), layout='constrained', sharey=True)
    colors = ('#216b82', '#ad4829')
    for j, r in enumerate(reports):
        for ax, subset, contrast, shift, color, label in [
                (gain, 'all', 'exact_minus_cold_prevalence', 0, colors[0], None),
                (energy, 'ambivalent', 'exact_minus_feasible', .1, colors[0], 'Exact − feasible'),
                (energy, 'ambivalent', 'exact_minus_linear', -.1, colors[1], 'Exact − linear')]:
            ci = r['scores'][subset]['paired_log_score_gains'][contrast]
            if ci['lo'] is not None:
                x = ci['estimate']
                ax.errorbar(x, j+shift, xerr=[[x-ci['lo']], [ci['hi']-x]], fmt='o', capsize=3,
                            color=color, label=label if j == 0 else None)
        fraction = r['scores']['geometrically_forced']['fraction']['estimate']
        if fraction is not None:
            forced.barh(j, fraction, height=.5, color='#647b63')
    for ax in (gain, energy):
        ax.axvline(0, lw=.7, color='.45'); ax.set_xlabel('Gain (nat / matched center)')
    gain.axvline(margin, ls='--', lw=1, color='.45', label='Primary margin'); gain.legend(fontsize=8, frameon=False)
    energy.legend(fontsize=8, frameon=False)
    gain.set(yticks=range(len(reports)), yticklabels=['2.372 (primary)', '3.05', '0'], ylabel='β', title='Exact − cold prevalence\nAll matches')
    forced.set(title='Forced by geometry\nAll matches', xlabel='Fraction', xlim=(0, 1))
    energy.set_title('Energetic discrimination\nAmbivalent matches only')
    gain.invert_yaxis()
    for ax in (gain, forced, energy):
        ax.spines[['top','right']].set_visible(False)
    with path.open('xb') as f:
        fig.savefig(f, format='png', dpi=155)
    plt.close(fig)


def run(args):
    out = args.out
    plan, prevalence = frozen_plan(out, args.lock_sha256)
    outputs = ['near-run.json', 'near-results.json', 'near-cage.png'] + [
        f'near-{suffix}' for i in ORDER for suffix in (f'parents-{i:02}.npz', f'score-{i:02}.json')]
    if any((out/name).exists() for name in outputs):
        raise FileExistsError('near-cage run already claimed or output exists; no replacement scoring')
    offsets = np.asarray(law.OFFSETS)
    expected = [v for v in product(range(-3, 4), repeat=3) if 2 <= sum(map(abs, v)) <= 3]
    require(np.array_equal(offsets, expected), 'changed near-halo geometry/order')
    nn, shell = table(bank.NN), table(offsets)
    law.verify()  # independent finite analytic cases; no endpoint access yet
    files = [SRC, Path(law.__file__), Path(bank.__file__), SRC.with_name('vacancy_dipole.py'),
             SRC.with_name('boundary_models.py'), SRC.with_name('rate.py'), SRC.parents[1]/'l20_pairs.py',
             SRC.parents[1]/'l20_operators_static.py']
    claim = dict(schema='l20.near-cage-run.v1', near_design_sha256=args.lock_sha256,
                 cold_lock_sha256=plan['cold_lock_sha256'], pairing_lock_sha256=plan['pairing_lock_sha256'],
                 implementation_sha256={f.name: digest(f) for f in files}, analytic_verification='passed')
    write_json(out/'near-run.json', claim)  # persistent exclusive claim, including on interrupted runs
    run_sha = digest(out/'near-run.json')
    reports, hashes = [], {}
    for index in ORDER:
        pin = plan['release'][str(index)]
        ids, keys, states, split = load_rows(args.data, index, pin)
        nums = collect(states, pin['beta'], prevalence, nn, shell)
        del states
        tab = out/f'near-parents-{index:02}.npz'
        with tab.open('xb') as f:
            np.savez_compressed(f, parent=ids, chain_seed=keys[:, 0], chain_id=keys[:, 1],
                                subsets=SUBSETS, models=MODELS, **nums)
        hashes[tab.name] = digest(tab)
        report = dict(schema='l20.near-cage-score.v1', index=index, **pin, split=split, origins_per_parent=bank.VOLUME,
                      lineage_sha256=hashlib.sha256(keys.tobytes()).hexdigest(), parent_table_sha256=hashes[tab.name],
                      near_run_sha256=run_sha, near_design_sha256=args.lock_sha256, frozen_cold_prevalence=prevalence,
                      signature=SIGNATURE, scores=summarize(nums))
        primary = report['scores']['all']['paired_log_score_gains']['exact_minus_cold_prevalence']
        report['assessment'] = ('descriptive only' if index != 7 else 'inconclusive' if primary['lo'] is None
                                else 'pass' if primary['lo'] >= plan['margin_nat_per_matched_center'] else 'did not pass')
        score_path = out/f'near-score-{index:02}.json'
        write_json(score_path, report)
        hashes[score_path.name] = digest(score_path)
        reports.append(report)
    figure = out/'near-cage.png'
    plot(figure, reports, plan['margin_nat_per_matched_center'])
    hashes[figure.name] = digest(figure)
    primary = reports[0]
    result = dict(schema='l20.near-cage-results.v1', near_design_sha256=args.lock_sha256, near_run_sha256=run_sha,
                  primary_index=7, cold_prevalence=prevalence, margin_nat_per_matched_center=plan['margin_nat_per_matched_center'],
                  primary_exact_minus_cold_prevalence=primary['scores']['all']['paired_log_score_gains']['exact_minus_cold_prevalence'],
                  assessment=primary['assessment'], interpretation=plan['question'], artifacts_sha256=hashes)
    write_json(out/'near-results.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='Existing boundary directory containing the three frozen locks')
    parser.add_argument('--lock-sha256', required=True, help='Prior SHA256 of near-design.json')
    run(parser.parse_args())
