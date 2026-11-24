"""Distill one frozen static channel into explicit local rules. See transparent/design.json."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[var] = '1'
import numpy as np
from scipy.special import rel_entr, xlogy
from scipy.stats import t

import vacancy_dipole as stream

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))
import l20_pairs as pairs
import l20_operators_static as static

ROOT = EXAMPLES.parent
OUT = ROOT / 'results/l20-autoresearch/transparent'
B = static.BATCH
REPS = 128
CUT = 7680
FEATURES = ['A(c-)', 'A(c+)', 'B(c-)', 'B(c+)', 'k-', 'k+', 'k- minus k+',
            'diagonal vacancy moment', 'center occupied coordination']
CONTROL = ['area', 'corner_A_bits', 'exact_corners']
PRIMARY = 'exact_corners'
MODES = ['hard', 'stochastic']
DIAGONAL = static.offsets(3, 1)[0].sum(axis=1).astype(np.int8)
C_MINUS, C_PLUS, K_MINUS, K_PLUS, POL, COORD = np.indices((3, 3, 4, 4, 31, 7), dtype=np.int16).reshape(6, -1)
GRID = np.column_stack((C_MINUS == 1, C_PLUS == 1, C_MINUS == 2, C_PLUS == 2,
                        K_MINUS, K_PLUS, K_MINUS - K_PLUS, POL-15, COORD))
CELLS = np.arange(len(GRID))


def verify(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def arr_sha(a, dtype):
    return hashlib.sha256(np.asarray(a, dtype=dtype, order='C').tobytes()).hexdigest()


def read_json(p):
    return json.loads(p.read_bytes())


def save_json(p, obj):
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); tmp.replace(p)


def load_teacher(design):
    pins = design['pins']
    for key, module in [('pair_code', pairs), ('static_code', static), ('row_reader', stream)]:
        verify(sha(module.__file__) == pins[key], f'{key} changed')
    stem = 'fits/w3-b1-learned-s11'
    # Use the published archive, not private extracted or alternative seed fits.
    with zipfile.ZipFile(static.OUT/'fit-receipts.zip') as z:
        meta, weights = (z.read(stem + suffix) for suffix in ('.json', '.npz'))
    verify(hashlib.sha256(meta).hexdigest() == pins['teacher_metadata'] and
           hashlib.sha256(weights).hexdigest() == pins['teacher_weights'], 'teacher receipt mismatch')
    with np.load(io.BytesIO(weights), allow_pickle=False) as f:
        enc, critic = ([f[f'{name}{i}'] for i in range(5)] for name in ('enc', 'crit'))
    return json.loads(meta), enc, critic


def cohort(data, index, fold, design):
    verify((index, fold) in ((9, 0), (9, 1), (8, 1), (0, 1)), 'forbidden cohort')
    path = data / f'beta-{index:02d}.npz'
    side = path.with_suffix('.json')
    beta, sourcehash, sidehash = design['sources'][str(index)]
    verify(sha(side) == sidehash and sha(path) == sourcehash, 'release hash mismatch')
    receipt = read_json(side)
    verify(float(receipt['beta']) == beta and receipt['endpoint_sha256'] == sourcehash and
           receipt['index'] == (0 if index == 8 else index), 'release/production index mismatch')
    ids = pairs.partition(index)[0][fold]
    with zipfile.ZipFile(path) as z:
        for name, value in [('axes', 'chain,z,y,x'), ('beta', beta)]:
            with z.open(name + '.npy') as f:
                item = np.lib.format.read_array(f, allow_pickle=False).item()
            verify((float(item) if name == 'beta' else item) == value, 'changed axes or beta')
        keys = np.column_stack([stream.selected_member(z, k, ids, ()).astype('<u8', copy=False)
                                for k in ('chain_seed', 'chain_id')])
        state = stream.selected_member(z, 'species', ids, (20, 20, 20))
    verify(state.dtype == np.uint8 and state.max() <= 2 and len(np.unique(keys, axis=0)) == len(ids),
           'invalid selected rows')
    info = dict(index=index, beta=beta, fold=('training', 'validation')[fold], parents=len(ids),
                endpoint_sha256=sourcehash, receipt_sha256=sidehash, ids_sha256=arr_sha(ids, '<u8'),
                lineage_sha256=arr_sha(keys, '<u8'))
    if not fold:
        for name, rows in [('structure', slice(None, CUT)), ('calibration', slice(CUT, None))]:
            info[name] = dict(parents=len(ids[rows]), ids_sha256=arr_sha(ids[rows], '<u8'),
                              lineage_sha256=arr_sha(keys[rows], '<u8'))
    return state.reshape(len(ids), 8000), info


def cell_index(v):
    # Canonical flat z,y,x. k- and k+ use exactly the three *internal* bonds at each corner.
    lo, hi = v[:, 0].astype(int), v[:, 26].astype(int)
    km = (v[:, [1, 3, 9]] == 0).sum(axis=1)
    kp = (v[:, [25, 23, 17]] == 0).sum(axis=1)
    pol = (v == 0) @ DIAGONAL  # signed sum in [-15,15], no halo
    coord = (v[:, [4, 10, 12, 14, 16, 22]] != 0).sum(axis=1)
    return ((((lo*3 + hi)*4 + km)*4 + kp)*31 + pol+15)*7 + coord


def local(v):
    lo, hi = v[:, 0].astype(int), v[:, 26].astype(int)
    return cell_index(v), {'area': static.physical(v, 3), 'corner_A_bits': 2*(lo == 1) + (hi == 1),
                           'exact_corners': lo*3 + hi}


def posterior(v, enc):
    q = static.prob(v, 3, {'kind': 'learned'}, enc).astype(float)
    return q / q.sum(axis=1, keepdims=True)


def add_mass(total, code, prob):
    for k in range(4):
        total[:, k] += np.bincount(code, weights=prob[:, k], minlength=len(total))


def training_tables(data, enc, design):
    state, info = cohort(data, 9, 0, design)
    vt = static.table(static.offsets(3, 1)[0])
    rng = np.random.default_rng([726101, 9]); h = hashlib.sha256()
    structure = np.zeros((len(GRID), 4)); calibration = np.zeros_like(structure)
    controls = {k: np.zeros((n, 4)) for k, n in zip(CONTROL, (55, 4, 9))}
    rows = np.arange(len(state))[:, None]
    for _ in range(32):
        a = rng.integers(8000, size=len(state)); h.update(a.astype('<u2').tobytes())
        v = state[rows, vt[a]]  # No shell table, shell responses or critic in this function.
        p = posterior(v, enc); cell, codes = local(v)
        add_mass(structure, cell[:CUT], p[:CUT]); add_mass(calibration, cell[CUT:], p[CUT:])
        for k in CONTROL:
            add_mass(controls[k], codes[k][CUT:], p[CUT:])
    info.update(repeats=32, origins_sha256=h.hexdigest())
    return structure, calibration, controls, info


def leaf(mass):
    p = mass + .5; p /= p.sum()
    return dict(calibration_records=int(round(mass.sum())), posterior=p.tolist(),
                hard_label=int(np.argmax(p)))


def impurity(mass):
    n = mass.sum()
    return float(xlogy(n, n) - xlogy(mass, mass).sum())


def grow(structure, calibration, cells, depth):
    parent = structure[cells].sum(0)
    node = leaf(calibration[cells].sum(0))
    node['structure_records'] = int(round(parent.sum()))
    if depth == 5 or node['structure_records'] < 512:
        return node
    best = impurity(parent); split = None
    # Deterministic tie order: listed feature, then smallest integer threshold.
    for j, name in enumerate(FEATURES):
        values = GRID[cells, j]
        for cut in np.unique(values)[:-1]:
            left = values <= cut
            lm, rm = (structure[cells[mask]].sum(0) for mask in (left, ~left))
            if min(round(lm.sum()), round(rm.sum())) < 256:
                continue
            loss = impurity(lm) + impurity(rm)
            if loss < best - 1e-10:
                best = loss; split = (name, int(cut), left)
    if split:
        name, cut, left = split
        node.update(feature=name, le=cut, yes=grow(structure, calibration, cells[left], depth+1),
                    no=grow(structure, calibration, cells[~left], depth+1))
    return node


def tree_table(node, depth):
    table = np.empty((len(GRID), 4))
    def walk(n, cells, left):
        if not left or 'feature' not in n:
            table[cells] = n['posterior']; return
        mask = GRID[cells, FEATURES.index(n['feature'])] <= n['le']
        walk(n['yes'], cells[mask], left-1); walk(n['no'], cells[~mask], left-1)
    walk(node, CELLS, depth)
    return table


def pack_labels(labels):
    """Four emitted symbols per byte, first symbol in bits 0..1; pad final byte with zeros."""
    packed = np.zeros((len(labels)+3)//4, dtype=np.uint8)
    for j in range(4):
        group = labels[j::4].astype(np.uint8, copy=False)
        packed[:len(group)] |= group << (2*j)
    return packed


def unpack_labels(packed, n):
    return ((np.asarray(packed, np.uint8)[:, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3).ravel()[:n]


def compress(blocks, rules, depth):
    """Deploy the selected JSON tree on n ternary 3-cubes. No neural net, critic or shell."""
    v = np.asarray(blocks)
    verify(depth in range(1, 6), 'no eligible locked tree depth')
    verify(v.shape[1:] in ((3, 3, 3), (27,)) and v.dtype.kind in 'ui' and
           not np.any((v < 0) | (v > 2)), 'expected n ternary 3-cubes in z,y,x order')
    cells = cell_index(v.reshape(len(v), 27)); labels = np.empty(len(v), dtype=np.uint8)
    def walk(node, rows, remaining):
        if not remaining or 'feature' not in node:
            labels[rows] = node['hard_label']; return
        take = GRID[cells[rows], FEATURES.index(node['feature'])] <= node['le']
        walk(node['yes'], rows[take], remaining-1); walk(node['no'], rows[~take], remaining-1)
    walk(rules['tree'], np.arange(len(v)), depth)
    return pack_labels(labels)


def fit(data, enc, design):
    s, c, control, info = training_tables(data, enc, design)
    return dict(train=info, fixed_frame='Coordinates z,y,x about center, species vacancy=0, A=1, B=2.',
                features=FEATURES, tripod_sites={'c-': [1, 3, 9], 'c+': [25, 23, 17]},
                tree_reader='Start at root. At test feature <= le follow yes, else no. At depth d or earlier terminal node emit hard_label (or sample posterior). Root depth zero. Equal scores choose lowest label.',
                tree=grow(s, c, CELLS, 0),
                lookup_index=dict(area='number of internal vacancy/particle bonds, 0..54',
                                  corner_A_bits='2*A(c-)+A(c+)', exact_corners='3*species(c-)+species(c+)'),
                lookups={k: [leaf(row) for row in mat] for k, mat in control.items()})


def ci(x):
    x = np.asarray(x, float); n = len(x)
    m = float(x.mean()); h = float(t.ppf(.975, n-1)*x.std(ddof=1)/np.sqrt(n))
    return dict(mean=m, lo95=m-h, hi95=m+h)


def bits(p):
    return -xlogy(p, p).sum(axis=-1)/np.log(2.)


def ent_interval(mass, noise):
    n = len(mass)
    point = bits(mass.mean(axis=0)) - noise.mean()
    deleted = bits((mass.sum(axis=0)-mass)/(n-1)) - (noise.sum()-noise)/(n-1)
    h = t.ppf(.975, n-1)*deleted.std(ddof=0)*np.sqrt(n-1)
    return dict(mean=float(point), lo95=float(point-h), hi95=float(point+h))


def normalized(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).mean(axis=1, keepdims=True))


def evaluate(data, index, design, model, rules, depths):
    meta, enc, critic = model
    state, info = cohort(data, index, 1, design)
    vt, et = map(static.table, static.offsets(3, 1))
    rng = pairs.partition(index)[1][1]; row = np.arange(len(state))[:, None]
    pks = len(state)//B
    verify(len(state) == 2048 and B == 128 and et.shape == (8000, 150), 'reference geometry changed')
    tables = {k: np.array([r['posterior'] for r in rules['lookups'][k]]) for k in CONTROL}
    tables.update({f'tree{d}': tree_table(rules['tree'], d) for d in depths})
    names = ['teacher', *tables]; count = len(names)
    # Axes repeat, whole-parent packet, method, mode (hard, stochastic), diagonal/offdiagonal.
    scores = np.empty((REPS, pks, count, 2, 2))
    packed = np.empty((REPS, count, len(state)//4), dtype=np.uint8)
    mass = np.zeros((pks, count, 2, 4)); noise = np.zeros((pks, count, 2))
    agree = np.zeros((pks, count)); kl = np.zeros_like(agree)
    h = hashlib.sha256(); uniform_error = ref_error = 0.
    for rep in range(REPS):
        a = rng.integers(8000, size=len(state)); h.update(a.astype('<u2').tobytes())
        v, e = state[row, vt[a]], state[row, et[a]]
        z = static.forward(static.dense(e), critic)[1].reshape(pks, B, 4).astype(float)
        r = normalized(z); wrong = (r.sum(axis=1, keepdims=True)-r)/(B-1)
        uniform_error = max(uniform_error, float(np.max(np.abs((r-wrong).mean(axis=(1, 2))))))
        p = posterior(v, enc); y = p.argmax(axis=1)
        cell, codes = local(v)
        for m, name in enumerate(names):
            q = p if name == 'teacher' else tables[name][codes[name] if name in CONTROL else cell]
            arg = q.argmax(axis=1)
            packed[rep, m] = pack_labels(arg)
            if rep == 0:
                verify(np.array_equal(unpack_labels(packed[rep, m], len(state)), arg), 'hard code packing mismatch')
                if name.startswith('tree'):
                    verify(np.array_equal(compress(v, rules, int(name[4:])), packed[rep, m]),
                           'deployable rule differs from scored lookup')
            agree[:, m] += (arg == y).reshape(pks, B).mean(1)
            kl[:, m] += rel_entr(p, q).sum(1).reshape(pks, B).mean(1)
            for j, emission in enumerate((static.EYE[arg], q)):
                pp = emission.reshape(pks, B, 4)
                mass[:, m, j] += pp.mean(axis=1)
                if j:
                    noise[:, m, j] += bits(pp).mean(axis=1)
                scores[rep, :, m, j, 0] = (pp*r).sum(2).mean(1)
                scores[rep, :, m, j, 1] = (pp*wrong).sum(2).mean(1)
        if rep < static.VREPS:
            direct = static.evaluate(v[None], e[None], 3, meta['spec'], enc, critic)
            ref_error = max(ref_error, float(np.max(np.abs(direct-scores[rep, :, 0, 1, :]))))
    info.update(origin_sha256=h.hexdigest(), origins_per_parent=REPS, negative_packet_size=B, packets=pks)
    mass /= REPS; noise /= REPS; agree /= REPS; kl /= REPS
    avg = scores.mean(axis=0); im = names.index(PRIMARY)
    results = {}
    for m, name in enumerate(names):
        d = dict(hard_teacher_agreement=ci(agree[:, m]), teacher_to_soft_KL_nats=ci(kl[:, m]),
                 excess_KL_over_exact_corners=ci(kl[:, m]-kl[:, im]),
                 agreement_minus_exact_corners=ci(agree[:, m]-agree[:, im]))
        for j, mode in enumerate(MODES):
            raw = avg[:, m, j, 0]; shuf = avg[:, m, j, 1]
            teacher = avg[:, 0, j, 0]; corner = avg[:, im, j, 0]
            d[mode] = dict(raw=ci(raw), shuffle=ci(shuf), raw_minus_shuffle=ci(raw-shuf),
                           teacher_loss=ci(teacher-raw), gain_over_exact_corners=ci(raw-corner),
                           gain_over_A_bits=ci(raw-avg[:, names.index('corner_A_bits'), j, 0]),
                           gain_over_area=ci(raw-avg[:, names.index('area'), j, 0]),
                           half_recovery=ci((raw-corner)-.5*(teacher-corner)),
                           class_mass=[ci(col) for col in mass[:, m, j].T],
                           H_bits=ent_interval(mass[:, m, j], np.zeros(pks)),
                           H_given_V_bits=ci(noise[:, m, j]),
                           empirical_channel_rate_bits=ent_interval(mass[:, m, j], noise[:, m, j]),
                           fixed_width_bits_per_emitted_symbol=2)
        results[name] = d
    first6 = float(scores[:static.VREPS, :, 0, 1, 0].mean())
    zero_error = float(abs(normalized(np.zeros((1, B, 4)))).max())
    algebra = ref_error < 2e-6 and zero_error == 0 and uniform_error < 1e-10
    calibrated = algebra and (index != 9 or abs(first6-meta['validation']) < 2e-6) and all(
        results['teacher'][m][stat]['lo95'] > 0 for m in MODES for stat in ('raw', 'raw_minus_shuffle'))
    packet_file = OUT / f'packets-{index:02d}.npz'
    np.savez_compressed(packet_file, scores=scores, names=np.array(names), modes=np.array(MODES),
                        packed_hard_symbols=packed,
                        order=np.array(['diagonal', 'offdiagonal']), packet_agreement=agree,
                        packet_teacher_to_soft_KL=kl, packet_class_mass=mass, packet_conditional_bits=noise)
    return dict(source=info, packet_sha256=sha(packet_file), calibration=dict(
        calibrated_positive_reference=bool(calibrated), direct_reference_max_error=ref_error,
        zero_logit_chance_error=zero_error, constant_code_raw_minus_shuffle_max_error=uniform_error,
        first_six_teacher_stochastic=first6, reference_cold_only=meta['validation'] if index == 9 else None),
        methods=results)


def select(doc):
    m = doc['methods']; g = m['teacher']['hard']['gain_over_exact_corners']
    if not doc['calibration']['calibrated_positive_reference']:
        return None, 'reference calibration failed'
    if g['lo95'] <= 0:
        return None, 'hard teacher gain over calibrated exact corners unresolved'
    for d in range(1, 6):
        r = m[f'tree{d}']; h = r['hard']
        if (r['hard_teacher_agreement']['mean'] >= .55 and h['gain_over_exact_corners']['mean'] >= .005
                and h['half_recovery']['mean'] >= 0):
            return d, 'shallowest eligible depth (cold selection; not independent evidence)'
    return None, 'no depth meets all three fixed cold criteria'


def decision(doc, depth):
    if depth is None:
        return 'no locked compressor; no transfer prediction tested'
    if not doc['calibration']['calibrated_positive_reference']:
        return 'unresolved: teacher/reference calibration failed'
    teacher = doc['methods']['teacher']['hard']['gain_over_exact_corners']
    if teacher['lo95'] <= 0:
        return 'unresolved: teacher gain over calibrated exact corners not established'
    r = doc['methods'][f'tree{depth}']['hard']
    material, recovery = r['gain_over_exact_corners'], r['half_recovery']
    if material['lo95'] > .005 and recovery['lo95'] > 0:
        return 'supports both locked gain margins'
    if material['hi95'] < .005 or recovery['hi95'] < 0:
        return 'excludes at least one locked gain margin'
    return 'unresolved at locked gain margins'


def figure(docs, depth, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(docs)+1, figsize=(4.8*(len(docs)+1), 4.2), layout='constrained')
    for ax, doc in zip(axs, docs):
        names = list(doc['methods']); ys = np.arange(len(names))
        for k, mode in enumerate(MODES):
            stats = [doc['methods'][name][mode]['raw'] for name in names]
            means = np.array([r['mean'] for r in stats]); err = np.array([[r['mean']-r['lo95'], r['hi95']-r['mean']] for r in stats]).T
            ax.errorbar(means, ys+(k-.5)*.24, xerr=err, fmt=('o', 's')[k], ms=4, capsize=2, label=mode)
        ax.axvline(0, lw=.7, color='.5'); ax.invert_yaxis(); ax.set(yticks=ys, yticklabels=names,
            xlabel='Frozen log bound (nats/origin)', title=f"β={doc['source']['beta']:.3g} · validation")
        ax.legend(frameon=False, fontsize=8)
    ax = axs[-1]; labels = []; x = []; bars = []
    for doc in docs:
        for name, r in doc['methods'].items():
            if name.startswith('tree') and (len(docs) == 1 or name == f'tree{depth}'):
                stat = r['hard']['half_recovery']; labels.append(f"β {doc['source']['beta']:.3g} · {name}")
                x.append(stat['mean']); bars.append([stat['mean']-stat['lo95'], stat['hi95']-stat['mean']])
    if x:
        ax.errorbar(x, np.arange(len(x)), xerr=np.array(bars).T, fmt='o', color='#177267', capsize=2)
    else:
        ax.text(.5, .5, 'No eligible cold rule', transform=ax.transAxes, ha='center')
    ax.axvline(0, lw=.8, color='.4'); ax.set(yticks=np.arange(len(labels)), yticklabels=labels,
        xlabel='(rule − corners) − ½(teacher − corners)\npaired nats/origin', title='Hard-code half-recovery margin'); ax.invert_yaxis()
    for ax in axs:
        ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(path, dpi=145); plt.close(fig)


def cold(args, design):
    lockfile = OUT/'cold-lock.json'
    if lockfile.exists():
        raise FileExistsError(lockfile)
    model = load_teacher(design); rules = fit(args.data, model[1], design)
    save_json(OUT/'rules.json', rules)
    doc = evaluate(args.data, 9, design, model, rules, range(1, 6)); depth, reason = select(doc)
    doc.update(selected_depth=depth, selection_reason=reason); save_json(OUT/'cold.json', doc)
    figure([doc], depth, OUT/'cold.png')
    artifacts = {n: sha(OUT/n) for n in ('design.json', 'rules.json', 'cold.json', 'packets-09.npz', 'cold.png')}
    lock = dict(selected_depth=depth, reason=reason, code_sha256=sha(__file__), artifacts=artifacts,
                prediction=design['transfer'], primary_comparator=PRIMARY, source_pins=design['sources'],
                reference_pins=design['pins'], interpretation=design['scope'])
    save_json(lockfile, lock)
    print(f'Cold selection={depth}: {reason}. Publish freeze before transfer. lock SHA256 {sha(lockfile)}', flush=True)


def public_lock(commit):
    verify(re.fullmatch(r'[0-9a-f]{40}', commit or ''), 'transfer requires --lock-commit <40 lowercase hex>')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args])
    refs = git('for-each-ref', '--contains', commit, '--format=%(refname)', 'refs/remotes/origin/').decode().splitlines()
    verify(refs, 'lock commit not reachable from a published origin tracking ref; push/fetch first')
    path = OUT/'cold-lock.json'
    committed = git('show', f'{commit}:{path.relative_to(ROOT)}')
    verify(committed == path.read_bytes(), 'working lock differs from public commit')
    lock = json.loads(committed)
    expected = {OUT/n: digest for n, digest in lock['artifacts'].items()}
    expected[Path(__file__).resolve()] = lock['code_sha256']
    for f, digest in expected.items():
        verify(sha(f) == digest and hashlib.sha256(git('show', f'{commit}:{f.relative_to(ROOT)}')).hexdigest() == digest,
               f'working or committed bytes differ from freeze: {f.name}')
    return lock, refs


def transfer(args, design):
    if (OUT/'transfer.json').exists():
        raise FileExistsError(OUT/'transfer.json')
    lock, refs = public_lock(args.lock_commit)  # Before any warm or specificity archive access.
    model = load_teacher(design); depth = lock['selected_depth']; rules = read_json(OUT/'rules.json')
    docs = [evaluate(args.data, i, design, model, rules, [] if depth is None else [depth]) for i in (8, 0)]
    null = docs[1]['methods']['teacher']; specificity = {}
    for mode in MODES:
        stat = null[mode]['raw_minus_shuffle']; specificity[mode] = ('inside ±0.01 nats' if stat['lo95'] > -.01 and stat['hi95'] < .01
            else 'outside ±0.01 nats' if stat['lo95'] > .01 or stat['hi95'] < -.01 else 'unresolved')
    verdict = decision(docs[0], depth)
    figure(docs, depth, OUT/'transfer.png')
    save_json(OUT/'transfer.json', dict(commit=args.lock_commit, origin_tracking_refs=refs,
              lock_sha256=sha(OUT/'cold-lock.json'), selected_depth=depth, warm_decision=verdict,
              beta_zero_specificity=specificity, temperatures=docs, interpretation=design['scope']))
    print(f'{verdict}; beta=0: {specificity}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('cold', 'transfer')); parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--lock-commit'); args = parser.parse_args()
    design = read_json(OUT/'design.json')
    (cold if args.mode == 'cold' else transfer)(args, design)
