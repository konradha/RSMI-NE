"""Cold-lock and independently score fixed tile codes on the same physical exterior face."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from scipy.stats import t

DESIGN = Path(__file__).resolve().parents[2] / 'results/l20-autoresearch/faces/evaluation-design.json'
SHAPE = (6, 6561, 81)
PACKETS, PARENTS, ANCHORS = 32, 64, 8000
MARGIN = 0.001
FULL = 'full_V'
# Y digits are 0=vacancy, 1=A, 2=B, little-endian within the registered face.
Y_DIGITS = np.arange(81)[:, None] // 3**np.arange(4) % 3
COMPOSITIONS, Y_COMP = np.unique(np.stack([(Y_DIGITS == k).sum(1) for k in (1, 2)], axis=1),
                                 axis=0, return_inverse=True)
COMP_ORDER = np.argsort(Y_COMP, kind='stable')
COMP_START = np.r_[0, np.cumsum(np.bincount(Y_COMP))[:-1]]
SCORE_KEYS = ('rate_bits', 'target_entropy', 'marginal_ce', 'conditional_entropy', 'conditional_ce',
              'lower', 'upper', 'plugin_mi', 'composition_lower', 'composition_upper',
              'arrangement_lower', 'arrangement_upper', 'composition_ce', 'arrangement_ce')
CE_KEYS = ('marginal_ce', 'conditional_ce', 'marginal_composition_ce', 'composition_ce')
SCRIPT = Path(__file__).resolve()


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def checked(path, expected):
    require(sha(path) == expected, f'SHA256 mismatch: {path}')


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def bank_header(bank, cold):
    m = read_json(bank / 'meta.json')
    i = m['index']
    require(i in ((9,) if cold else (0, 4, 8)), 'wrong archive/fold for job')
    beta, pin = read_json(DESIGN)['source_pins'][str(i)]
    require(abs(float(m['beta'])-beta) < 1e-10 and m['source_sha256'] == pin, 'wrong release source')
    require(m['anchors_per_parent'] == ANCHORS and m['buffer'] == 1 and
            m['validation_parents'] == PACKETS*PARENTS, 'incompatible bank geometry or parent count')
    require(cold or (m['train_sha256'] is None and not (bank/'train.npy').exists()),
            'warm bank must be validation-only')
    return m


def load_maps(base, optimized):
    maps = {}
    for path, trained in ((base, False), (optimized, True)):
        with np.load(path, allow_pickle=False) as z:
            names = {k[:-4] for k in z.files if k.endswith('_lut')}
            if trained:
                require(names == {'affine', 'discrete'}, 'optimized.npz lacks the two declared winners')
            else:
                require({'rsmi', 'species-A', 'species-B', 'vacancy', 'composition', 'checkerboard',
                         'surface', 'energy', 'tetra_A', 'tetra_B', 'tetra_vacancy',
                         'kmeans_A_dipole'} <= names, 'incomplete comparator bundle')
            for name in sorted(names):
                lut = z[name + '_lut']
                require(name not in maps and name != FULL and lut.shape == (6561,) and
                        lut.dtype.kind in 'ui' and lut.min() >= 0 and lut.max() < 4, f'invalid map: {name}')
                maps[name] = lut.astype(np.uint8, copy=False)
    return maps


def compress(counts, bins):
    """Exactly marginalize integer joint counts over the deterministic map."""
    return np.stack([counts[:, v, :].sum(1, dtype=np.int64) for v in bins], axis=1)


def groups(lut):
    return [np.flatnonzero(lut == i) for i in range(4)]


def compose(a):
    dtype = np.int64 if a.dtype.kind in 'iu' else np.float64
    return np.add.reduceat(np.take(a, COMP_ORDER, axis=-1), COMP_START, axis=-1, dtype=dtype)


def conditional_entropy(a):
    # H(last axis | all preceding axes) under the pooled empirical joint.
    rows = a.sum(axis=-1, dtype=np.int64)
    return float((xlogy(rows, rows).sum()-xlogy(a, a).sum())/rows.sum())


def q_marginal(a):
    return (a + 0.5)/(a.sum(-1, keepdims=True) + 40.5)


def log_weights(cold_rows, marginal):
    # The frozen integer rows + fixed prior in decoders.npz losslessly encode the decoder;
    # no parameters from the evaluation distribution enter any frozen probability.
    q = (cold_rows + 81.*marginal[:, None, :])/(cold_rows.sum(-1, keepdims=True) + 81.)
    return -np.log(marginal), -np.log(q), -np.log(compose(marginal)), -np.log(compose(q))


def ce(c, c_comp, weights):
    m, q, mc, qc = weights
    total_y = c.sum(1, dtype=np.int64)
    total_comp = c_comp.sum(1, dtype=np.int64)
    return np.array([np.einsum('ij,ij->', total_y, m), np.einsum('ijk,ijk->', c, q),
                     np.einsum('ij,ij->', total_comp, mc), np.einsum('ijk,ijk->', c_comp, qc)])/total_y.sum()


def scores(c, cc, frozen_ce):
    marginal, cond, marginal_comp, cond_comp = frozen_ce
    hy = conditional_entropy(c.sum(1, dtype=np.int64))
    hc = conditional_entropy(cc.sum(1, dtype=np.int64))
    hyh, hch = conditional_entropy(c), conditional_entropy(cc)
    rate = conditional_entropy(c.sum(axis=(0, 2), dtype=np.int64))/np.log(2.)
    return np.array([rate, hy, marginal, hyh, cond, hy-cond, marginal-hyh, hy-hyh,
                     hc-cond_comp, marginal_comp-hch, (hy-hc)-(cond-cond_comp),
                     (marginal-marginal_comp)-(hyh-hch), cond_comp, cond-cond_comp])


def algebra_checks(train, lut):
    # Known channels, made by *integer* substitutions of the measured cold (J,V) masses.
    mass = train.sum(-1, dtype=np.int64)
    controls = {}
    for dependent in (False, True):
        toy = np.zeros((*mass.shape, 81), dtype=np.int64)
        if dependent:
            toy[:, np.arange(lut.size), lut] = mass
        else:
            toy[:, :, 0] = mass
            toy[:, :, 1] = mass
        c = compress(toy, groups(lut))
        q = log_weights(c, q_marginal(c.sum(1)))
        cc = compose(c)
        vals = dict(zip(SCORE_KEYS, scores(c, cc, ce(c, cc, q))))
        h = conditional_entropy(c.sum(-1))
        expected = h if dependent else 0.
        require(abs(vals['plugin_mi']-expected) < 1e-10 and vals['lower'] <= expected+1e-10 and
                expected <= vals['upper']+1e-10 and abs(vals['composition_lower']+
                vals['arrangement_lower']-vals['lower']) < 1e-10, 'algebraic channel check failed')
        controls['Y_is_H' if dependent else 'independent_Y'] = {
            'known_mi': expected, 'measured_plugin': vals['plugin_mi'], 'lower': vals['lower'], 'upper': vals['upper']}
    return controls


def freeze(train, maps, path):
    # Count parameterization compresses much better than saving 3.2M dense floats;
    # the unit-count rows and shared prior specify every probability exactly.
    require(train.dtype.kind in 'ui' and train.min() >= 0 and train.max() < 2**32, 'invalid train counts')
    rows = {FULL: train.astype(np.uint32, copy=False)}
    for name, lut in maps.items():
        c = compress(train, groups(lut))
        require(c.max() < 2**32, 'decoder count exceeds uint32')
        rows[name] = c.astype(np.uint32)
    prior = q_marginal(train.sum(1, dtype=np.int64))
    np.savez_compressed(path, marginal=prior, **rows)


def score_model(packets, weights):
    # Only c_total and one omitted packet are needed for the dense full-tile table.
    total = np.zeros(packets.shape[1:], dtype=np.int64)
    packet_scores = []
    for pkt in packets:
        total += pkt
        packet_scores.append(ce(pkt, compose(pkt), weights))
    packet_scores = np.stack(packet_scores)
    mean = packet_scores.mean(0)
    total_comp = compose(total)
    pooled = scores(total, total_comp, mean)
    leave = np.stack([scores(total-pkt, total_comp-compose(pkt),
                             (PACKETS*mean-loss)/(PACKETS-1)) for pkt, loss in zip(packets, packet_scores)])
    return pooled, leave, packet_scores


def se(replicates):
    return np.sqrt((PACKETS-1)*np.mean((replicates-replicates.mean(axis=0))**2, axis=0))


def interval(value, err, tails=.025):
    half = t.ppf(1-tails, PACKETS-1)*err
    return {'estimate': float(value), 'se': float(err), 'lo': float(value-half), 'hi': float(value+half)}


def compare(ce_a, ce_b, tails=.025):
    # Gain in log score of A over B. Equal parent counts and face counts per packet.
    d = ce_b-ce_a
    return interval(d.mean(), d.std(ddof=1)/np.sqrt(PACKETS), tails)


def score(bank, out, m, maps, locked_comparator=None):
    path = bank/'validation.npy'
    checked(path, m['validation_sha256'])
    val = np.load(path, mmap_mode='r', allow_pickle=False)
    require(val.shape == (PACKETS, *SHAPE) and val.dtype == np.dtype('uint32'), 'invalid validation array')
    for pkt in val:
        nv = pkt.sum(-1, dtype=np.int64)
        require(np.array_equal(nv, np.broadcast_to(nv[0], nv.shape)) and
                nv[0].sum() == ANCHORS*PARENTS, 'misregistered direction or packet count')
    models, evidence, leave_bounds = {}, {}, {}
    with np.load(out/'decoders.npz', allow_pickle=False) as dec:
        prior = dec['marginal']
        for name in [*maps, FULL]:
            if name == FULL:
                pkts = val
            else:
                bins = groups(maps[name])
                pkts = np.stack([compress(pkt, bins) for pkt in val])
            rows = dec[name]
            weights = log_weights(rows, prior)
            pooled, leave, packet_ce = score_model(pkts, weights)
            model = {k: interval(v, e) for k, v, e in zip(SCORE_KEYS, pooled, se(leave))}
            model['cold_decoder_rows_sha256'] = hashlib.sha256(rows.tobytes(order='C')).hexdigest()
            model['encoder_lut_sha256'] = None if name == FULL else hashlib.sha256(maps[name].tobytes()).hexdigest()
            models[name] = model
            evidence[name] = packet_ce
            leave_bounds[name] = leave[:, [SCORE_KEYS.index('lower'), SCORE_KEYS.index('upper')]]
            del rows, weights
    marginal_h = models[FULL]['target_entropy']['estimate']
    for model in models.values():
        require(abs(model['target_entropy']['estimate']-marginal_h) < 1e-12, 'unequal target entropy')
    # Familywise conservative retention: numerator LCB, denominator UCB; establish positive full MI first.
    critical = t.ppf(1-.05/(len(models)+1), PACKETS-1)
    full = models[FULL]
    upper_full = full['upper']['estimate'] + critical*full['upper']['se']
    lower_full = full['lower']['estimate'] - critical*full['lower']['se']
    for name, model in models.items():
        lower = model['lower']['estimate']-critical*model['lower']['se']
        model['simultaneous_lower_endpoint'] = lower
        if name != FULL:
            model['retained_fraction_lower_95'] = (max(0., lower)/upper_full
                                                   if lower_full > 0 and upper_full > 0 else None)
    ordinary = sorted(set(maps)-{'rsmi', 'discrete'})
    # At transfer compare only to the cold-locked target; never select or rank warm competitors.
    comparator = locked_comparator if locked_comparator is not None else min(
        ordinary, key=lambda n: (models[n]['conditional_ce']['estimate'], n))
    require(comparator in ordinary, 'invalid locked comparator')
    cold = m['index'] == 9
    refs = [*ordinary, 'rsmi'] if cold else [comparator, 'rsmi']
    # Include the possible historical-vs-ordinary choices in the cold selection family.
    n_contrasts = 2*len(ordinary)+1 if cold else 3
    def contrast(name, ref):
        a, b = evidence[name], evidence[ref]
        ma, mb = models[name], models[ref]
        da, db = leave_bounds[name], leave_bounds[ref]
        # Cross the endpoints to bound a true MI difference. Never subtract two lower bounds.
        low = ma['lower']['estimate']-mb['upper']['estimate']
        high = ma['upper']['estimate']-mb['lower']['estimate']
        q = t.ppf(1-.025/n_contrasts, PACKETS-1)
        return {'face': compare(a[:, 1], b[:, 1]),
                'face_simultaneous': compare(a[:, 1], b[:, 1], .025/n_contrasts),
                'composition': compare(a[:, 3], b[:, 3]),
                'within_composition': compare(a[:, 1]-a[:, 3], b[:, 1]-b[:, 3]),
                'mi_gap_directional_scores': [low, high],
                'mi_gap_simultaneous_outer_95': [low-q*se(da[:, 0]-db[:, 1]), high+q*se(da[:, 1]-db[:, 0])]}
    contrasts = {ref: contrast('discrete', ref) for ref in refs}
    primary = contrasts[comparator]['face']
    status = ('supported' if primary['lo'] >= MARGIN else 'excluded' if primary['hi'] < MARGIN else 'unresolved')
    result = {
        'bank': m, 'bank_meta_sha256': sha(bank/'meta.json'), 'models': models,
        'strongest_ordinary_selected_on_cold': comparator, 'discrete_minus_reference_log_score': contrasts,
        'historical_rsmi_minus_locked_ordinary': contrast('rsmi', comparator),
        'contrast_family_size': n_contrasts,
        'conditional_mi_units': 'nats per physical face given J; lower/upper are directional scores, not a realized CI',
        'simultaneous_retention': {'lower_full': lower_full, 'upper_full': upper_full,
                                   'one_sided_t': critical, 'endpoint_count': len(models)+1},
        'cold_superior_to_all_ordinary': (all(contrasts[n]['face_simultaneous']['lo'] > 0 for n in ordinary)
                                         if cold else None),
        'index4_prediction': status if m['index'] == 4 else 'not designated primary',
        'packet_columns': list(CE_KEYS), 'intervals': 'pointwise two-sided 95% packet t; multiplicity only as labeled'}
    tag = f"{m['index']:02d}"
    np.savez_compressed(out/f'packets-{tag}.npz', **evidence)
    return result


def plot(path, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = result['models']
    chosen = result['strongest_ordinary_selected_on_cold']
    names = sorted(models, key=lambda n: (models[n]['rate_bits']['estimate'], n))
    fig, (ax, rate) = plt.subplots(1, 2, sharey=True, figsize=(10.2, 6.7),
                                   gridspec_kw={'width_ratios': [2.8, 1.]}, constrained_layout=True)
    for y, name in enumerate(names):
        color = {'discrete': '#00856a', 'rsmi': '#9d4c9b', chosen: '#bd5715', FULL: '#111111'}.get(name, '#688497')
        m = models[name]
        lo, hi = (m[k]['estimate'] for k in ('lower', 'upper'))
        ax.plot([lo, hi], [y, y], lw=3.2, color=color)
        ax.scatter(lo, y, color=color, s=21, zorder=2)
        ax.scatter(hi, y, edgecolor=color, facecolor='white', s=21, zorder=2)
        bits = m['rate_bits']['estimate']
        rate.barh(y, bits, height=.44, color=color)
        rate.text(bits+.08, y, f'{bits:.2f}', ha='left', va='center', fontsize=8)
    ax.set_yticks(range(len(names)), [n + (' (cold comparator)' if n == chosen else '') for n in names])
    ax.axvline(0, color='0.7', lw=.8)
    ax.grid(axis='x', alpha=.18)
    ax.set_xlabel('Conditional information (nats per physical face)\nSolid: lower score   ○: upper score   •   not a confidence interval')
    rate.set_xlabel('Empirical tile entropy\n(bits per emitted label)')
    rate.set_xlim(0, max(2, models[FULL]['rate_bits']['estimate'])*1.18)
    rate.axvline(2, color='0.5', ls=':', lw=1)
    fig.suptitle(f"Same physical face, β={float(result['bank']['beta']):.4g}\nCold-frozen decoders · 2-bit maps and uncompressed V reference")
    fig.savefig(path, dpi=155)
    plt.close(fig)


def cold_job(a):
    require(not (a.out/'cold-lock.json').exists(), 'cold evidence already locked; do not overwrite')
    m = bank_header(a.bank, True)
    train_path = a.bank/'train.npy'
    checked(train_path, m['train_sha256'])
    train = np.load(train_path, mmap_mode='r', allow_pickle=False)
    require(train.shape == SHAPE and train.dtype.kind in 'ui', 'wrong training array')
    nv = train.sum(-1, dtype=np.int64)
    require(np.array_equal(nv, np.broadcast_to(nv[0], nv.shape)) and
            nv[0].sum() == ANCHORS*m['train_parents'], 'wrong cold training mass')
    checked(a.base, read_json(DESIGN)['base_sha256'])
    optimization = a.optimized.with_name('optimization.json')
    fit = read_json(optimization)
    checked(a.optimized, fit['optimized_sha256'])
    checked(a.base, fit['base_sha256'])
    checked(SCRIPT.with_name('face_optimizer.py'), fit['optimizer_sha256'])
    source_keys = ('index', 'beta', 'buffer', 'source_sha256', 'train_sha256', 'train_parents', 'anchors_per_parent')
    require(fit['source'] == {k: m[k] for k in source_keys}, 'optimizer used a different training bank')
    maps = load_maps(a.base, a.optimized)
    require(not a.lock_sha256, 'cold cannot take a transfer lock')
    controls = algebra_checks(train, maps['discrete'])
    a.out.mkdir(exist_ok=True, parents=True)
    freeze(train, maps, a.out/'decoders.npz')
    result = score(a.bank, a.out, m, maps)
    result['algebraic_train_controls'] = controls
    write_json(a.out/'beta-09.json', result)
    plot(a.out/'physical-09.png', result)
    artifacts = ('decoders.npz', 'beta-09.json', 'packets-09.npz', 'physical-09.png')
    lock = {'artifacts': {n: sha(a.out/n) for n in artifacts}, 'script_sha256': sha(SCRIPT),
            'design_sha256': sha(DESIGN), 'base_sha256': sha(a.base), 'optimized_sha256': sha(a.optimized),
            'bank_meta_sha256': result['bank_meta_sha256'], 'cold_bank': m,
            'optimization_report_sha256': sha(optimization), 'optimizer_sha256': fit['optimizer_sha256'],
            'strongest_ordinary': result['strongest_ordinary_selected_on_cold'],
            'prediction': read_json(DESIGN)['primary_prediction']}
    write_json(a.out/'cold-lock.json', lock)
    print('Cold comparator:', lock['strongest_ordinary'], 'cold-lock.json SHA256:', sha(a.out/'cold-lock.json'))


def transfer_job(a):
    require(a.lock_sha256 is not None, 'must supply the previously published --lock-sha256')
    lock_path = a.out/'cold-lock.json'
    checked(lock_path, a.lock_sha256)
    lock = read_json(lock_path)
    for path, key in ((SCRIPT, 'script_sha256'), (DESIGN, 'design_sha256'), (a.base, 'base_sha256'),
                      (a.optimized, 'optimized_sha256')):
        checked(path, lock[key])
    checked(a.optimized.with_name('optimization.json'), lock['optimization_report_sha256'])
    checked(SCRIPT.with_name('face_optimizer.py'), lock['optimizer_sha256'])
    for name, digest in lock['artifacts'].items():
        checked(a.out/name, digest)
    cold = read_json(a.out/'beta-09.json')
    require(cold['bank'] == lock['cold_bank'] and cold['bank_meta_sha256'] == lock['bank_meta_sha256'] and
            cold['strongest_ordinary_selected_on_cold'] == lock['strongest_ordinary'], 'inconsistent cold lock')
    maps = load_maps(a.base, a.optimized)
    # No warm bank, including its header, is inspected before the cold evidence verifies.
    m = bank_header(a.bank, False)
    require(m['transfer_lock_sha256'] == a.lock_sha256 and
            m['implementation_sha256'] == cold['bank']['implementation_sha256'],
            'warm extraction not bound to published cold lock and frozen face geometry')
    tag = f"{m['index']:02d}"
    require(not (a.out/f'beta-{tag}.json').exists(), 'transfer already published at this index')
    result = score(a.bank, a.out, m, maps, lock['strongest_ordinary'])
    result['cold_lock_sha256'] = a.lock_sha256
    plot(a.out/f'physical-{tag}.png', result)
    write_json(a.out/f'beta-{tag}.json', result)
    print('Transfer index:', tag, 'primary:', result['index4_prediction'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', choices=('cold', 'transfer'))
    for key in ('bank', 'base', 'optimized', 'out'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--lock-sha256', help='external SHA256 of the published cold lock, required for transfer')
    args = p.parse_args()
    (cold_job if args.job == 'cold' else transfer_job)(args)
