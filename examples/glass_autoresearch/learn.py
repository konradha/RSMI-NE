"""CPU-only discrete bottleneck tournament. See --out/design.json for the locked gates."""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

for _variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l20_operators_static import CODE, offsets, source, table
from l20_pairs import L, PAIR_SEED, SPLIT, partition

B = 64
H = 12
ROOT_SEED = 181124
N_ROUNDS = 16
PROTOCOL = {
    'hypothesis': 'A nonlinear categorical code of a 3^3 block retains shell information across two empty layers beyond a matched affine encoder and fixed codes.',
    'primary_metric': 'Raw exact expected hard-state InfoNCE lower bound, nats/anchor, 64 distinct same-temperature parents per contrastive batch.',
    'geometry': 'Canonical z,y,x; w=3. Six 5x5 faces centered at signed axis distance 4 (b=2); b=1 calibration uses distance 3. 150 distinct sites for both, no overlap with visible sites.',
    'models': '2 and 4 hard output states sampled from a categorical posterior; sum the hard-state log density ratios exactly (never feed the soft vector to a critic). Invariant mean over eight corners; equivariant tied scalar score for two parity tetrahedra or four unoriented body diagonals; oriented untied corner readout. All local models compose a shared nonlinear map over 2x2x2 corner cells. Affine comparator uses 54 independent site/species coefficients per logit.',
    'critics': 'Flat MLP+linear skip vs a shared nonlinear face-quadrant MLP pooled over the three incident quadrants of each corner with oriented readout. Same 150 raw targets, 12 hidden channels per local unit, not identical parameter counts.',
    'baselines': 'Best validation affine and best validation deterministic code of the champion cardinality, allowing either critic for each. Also report the exactly critic-matched fits. Fixed codes: occupancy of species A or vacancy at a corner (or two body-diagonal corners); block composition. Raw and analytic average of all off-diagonal parent shuffles use the same scores and same negatives.',
    'discovery': 'Release index 9 train and validation only. Four independent-seed brackets of four stages; fresh training and screening anchors at each stage. In each cardinality keep ceil(half) nonlinear architectures by fresh-anchor raw validation bound; tie by model name. Halve fixed codes within (cardinality, critic); keep each affine comparator and calibration. No early epoch selection. End-of-stage checkpoints compete on a common four-anchor validation bank across all rounds. Final selection maximizes its raw bound, not a gain over controls. Disjoint test parents are not independent replications of model seeds.',
    'control': 'Fixed b=1 oriented four-state local / flat model; weak planted b=2 oriented two-state local / flat model. Start with uniformly permuted complete shells inside each same-fold parent batch; within a random 20% subset rank-match corner A to first-shell-site A. No configuration or species value generated or edited; both marginals preserved exactly. Synthetic pairs are a pipeline diagnostic, not physical MI.',
    'confirmation': 'Freeze one common-validation winner and cardinality-matched baselines before scoring. Test indices 9,8,7 (8 and 7 absent entirely from this search); historical public results mean none is a new physical release. Average 24 fresh anchors inside each of 64 disjoint batches of 64 test parents. Paired t intervals conditional on fits; no site- or seed-level pseudoreplication.',
    'gate': 'Family alpha=.05/3 allocated to this lane. One-sided Bonferroni lower bounds for 16 endpoints: raw, raw-minus-shuffled, raw-minus-affine, raw-minus-fixed at each of 9,8,7; b1 and plant raw and raw-minus-shuffled at index 9. All 16 lower bounds must exceed zero for a positive primary claim. Also report unadjusted two-sided 95% intervals, without clipping negative bounds. Only a variational-bound comparison, not proof of MI differences, order or scale invariance.',
    'references': ['https://arxiv.org/abs/2101.11633', 'https://arxiv.org/abs/2103.16887', 'https://arxiv.org/abs/2301.11934'],
}
ALPHA = .05 / (3 * 16)
CORNERS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=int)
VCELL = np.array([[np.ravel_multi_index(tuple(1 + c*np.array(u)), (3,)*3) for u in itertools.product((0, 1), repeat=3)] for c in CORNERS])
# Binary coordinates enumerate center (0), 3 face centres, 3 edge centres, corner (7).
GROUP = {2: (CORNERS.prod(1) > 0).astype(int), 4: np.minimum(np.arange(8), 7-np.arange(8))}


def face_indices():
    # Within each incident face, only tangent-axis permutations are pooled. Cube
    # isometries now permute the 8 corners and their three incident observations.
    a, b = [], []
    for c in CORNERS:
        for d in range(3):
            first = 25*(2*d + int(c[d] > 0))
            y, x = np.delete(c, d)
            rings = list(itertools.combinations_with_replacement(range(3), 2))
            a.append([first + 5*(2+y*i) + 2+x*j for i, j in rings])
            b.append([first + 5*(2+y*j) + 2+x*i for i, j in rings])
    return np.array(a), np.array(b)


FACE_A, FACE_B = face_indices()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def random_stream(*parts):
    return np.random.default_rng(np.random.SeedSequence([ROOT_SEED, *parts]))


def draw(state, parents, rng, tables):
    anchor = rng.integers(L**3, size=len(parents))
    return tuple(state[parents[:, None], tab[anchor]] for tab in tables)


def make_tables():
    v, e2 = offsets(3, 2)
    _, e1 = offsets(3, 1)
    for e in (e1, e2):
        if len(np.unique(np.concatenate([v, e]) % L, axis=0)) != 177:
            raise ValueError('visible/shell overlap or repeated target')
    return tuple(map(table, (v, e2, e1)))


def plant(v, e, rng):
    # Derangement is unnecessary: start with exchangeable random pairings.
    donor = rng.permutation(B)
    take = rng.permutation(B)[:round(.2*B)]
    dst = take[np.argsort(v[take, 0] == 1, kind='stable')]
    src = donor[take]
    donor[dst] = src[np.argsort(e[src, 0] == 1, kind='stable')]
    return e[donor]


def visible_features(v):
    site = CODE[v]
    cell = site[:, VCELL]
    corner = cell[:, :, 7]
    edge = (cell[:, :, 3] + cell[:, :, 5] + cell[:, :, 6]) / 3
    face = (cell[:, :, 1] + cell[:, :, 2] + cell[:, :, 4]) / 3
    # Include species-resolved local bond products as well as trainable nonlinear compositions.
    products = (corner[..., :, None] * edge[..., None, :]).reshape(len(v), 8, 4)
    return site.reshape(len(v), 1, 54), np.concatenate((corner, edge, face, cell[:, :, 0], products), axis=-1)


def target_features(e, geometry):
    if geometry == 'flat':
        return CODE[e].reshape(len(e), 1, 300)
    return ((CODE[e[:, FACE_A]] + CODE[e[:, FACE_B]])*.5).reshape(len(e), 24, 12)


def probabilities(logits):
    p = np.exp(logits-logits.max(axis=1, keepdims=True))
    return p/p.sum(axis=1, keepdims=True)


class Network:
    """Shared hidden map, geometry-specific linear pooling, then discrete logits."""
    def __init__(self, mode, k, rng, critic=False):
        self.mode, self.k = mode, k
        din = {'affine': 54, 'flat': 300}.get(mode, 12)
        dout = din + (0 if mode == 'affine' else H)
        sites = 8 if mode in ('oriented', 'faces') else 1
        outputs = 1 if mode == 'equivariant' else k
        norm = lambda shape, scale: (rng.normal(size=shape)*scale).astype('f')
        self.params = [] if mode == 'affine' else [norm((din, H), 1/np.sqrt(din)), np.zeros(H, 'f')]
        w = norm((dout*sites, outputs), (.012 if critic else 1.4)/np.sqrt(dout*sites))
        if not critic and mode in ('oriented', 'affine'):
            # Same species-A corner-bit skip for unrestricted additive and nonlinear codes.
            bits = np.arange(k)
            w[0] += .85*(2*(bits % 2)-1)
            if k == 4:
                w[52 if mode == 'affine' else 7*dout] += .85*(2*(bits//2)-1)
        self.params.extend([w, np.zeros(outputs, 'f')])

    def forward(self, x):
        mode = self.mode
        if mode == 'affine':
            h, base = None, x
        else:
            h = np.tanh(x @ self.params[0] + self.params[1])
            base = np.concatenate((x, h), axis=-1)
        if mode == 'faces':
            feat = base.reshape(len(x), 8, 3, -1).mean(axis=2).reshape(len(x), -1)
        elif mode == 'invariant':
            feat = base.mean(axis=1)
        elif mode == 'equivariant':
            feat = np.stack([base[:, GROUP[self.k] == j].mean(axis=1) for j in range(self.k)], axis=1)
        else:
            feat = base.reshape(len(x), -1)
        z = feat @ self.params[-2] + self.params[-1]
        return z.reshape(len(x), self.k), (x, h, feat)

    def backward(self, cached, dz):
        x, h, feat = cached
        dz = dz.reshape(*feat.shape[:-1], -1)
        d = dz.reshape(-1, dz.shape[-1]); g = [feat.reshape(-1, feat.shape[-1]).T @ d, d.sum(0)]
        if h is None:
            return g
        tail = dz @ self.params[-2].T
        if self.mode == 'faces':
            dh = np.repeat(tail.reshape(len(x), 8, -1)[:, :, -H:], 3, axis=1)/3
        elif self.mode == 'invariant':
            dh = tail[:, None, -H:]/8
        elif self.mode == 'equivariant':
            dh = tail[:, GROUP[self.k], -H:]*(self.k/8)
        else:
            dh = tail.reshape(len(x), x.shape[1], -1)[:, :, -H:]
        dh = (dh*(1-h*h)).reshape(-1, H)
        return [x.reshape(-1, x.shape[-1]).T @ dh, dh.sum(0), *g]


def role(architecture):
    if architecture == 'affine':
        return 'affine'
    if architecture in ('bits-A', 'bits-vacancy', 'composition'):
        return 'fixed'
    return 'nonlinear'


def specs():
    arms = [(a, k, g, 'physical') for a, k, g in itertools.product(
        ('invariant', 'equivariant', 'oriented', 'affine', 'bits-A', 'bits-vacancy', 'composition'),
        (2, 4), ('faces', 'flat'))]
    return [*arms, ('oriented', 4, 'flat', 'near'), ('oriented', 2, 'flat', 'planted')]


def arm_name(spec):
    return '-'.join(map(str, spec))


class Arm:
    def __init__(self, spec, bracket, medians):
        self.a, self.k, self.g, self.task = spec
        self.spec = spec
        self.name = arm_name(spec)
        self.medians = medians
        # Critic starts identical at matched k/geometry; encoder starts identical across critics.
        aidx = sorted({s[0] for s in specs()}).index(self.a)
        self.enc = None if role(self.a) == 'fixed' else Network(self.a, self.k, random_stream(11, bracket, aidx, self.k))
        self.crit = Network(self.g, self.k, random_stream(12, bracket, int(self.g == 'flat'), self.k), True)
        self.params = ([] if self.enc is None else self.enc.params) + self.crit.params
        self.mom, self.vel = [[np.zeros_like(x) for x in self.params] for _ in range(2)]
        self.steps = 0

    def channel(self, v, vis):
        if self.enc is not None:
            logits, cached = self.enc.forward(vis[0 if self.a == 'affine' else 1])
            return probabilities(logits), cached
        if self.a == 'composition':
            code = ((v == 0).sum(1) > self.medians[0]).astype(int)
            if self.k == 4:
                code += 2*((v == 1).sum(1) > self.medians[1])
        else:
            species = int(self.a == 'bits-A')
            code = (v[:, 0] == species).astype(int)
            if self.k == 4:
                code += 2*(v[:, -1] == species)
        return np.eye(self.k, dtype='f')[code], None

    def compute(self, v, vis, shell, train):
        p, ec = self.channel(v, vis)
        z, cc = self.crit.forward(shell)
        if not train:
            p, z = p.astype('d'), z.astype('d')
        z -= z.max(axis=0)
        ez = np.exp(z); ratio = z - np.log(ez.mean(axis=0))
        raw = (p*ratio).sum(axis=1, keepdims=True)
        if not train:
            shuffled = (p*((ratio.sum(axis=0)-ratio)/(len(v)-1))).sum(axis=1)
            return np.array([raw.mean(), shuffled.mean(), p.max(axis=1).mean(), *p.mean(axis=0)])
        grads = [] if self.enc is None else self.enc.backward(ec, -p*(ratio-raw)/len(v))
        grads.extend(self.crit.backward(cc, ((ez/ez.sum(axis=0))*p.sum(axis=0)-p)/len(v)))
        self.steps += 1
        lr = .0015 * np.sqrt(1-.999**self.steps)/(1-.9**self.steps) / np.sqrt(1+self.steps/1200)
        for w, m, s, grad in zip(self.params, self.mom, self.vel, grads):
            m *= .9; m += .1*grad
            s *= .999; s += .001*grad*grad
            w -= lr*m/(np.sqrt(s)+1e-8)

    def save(self, path):
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, steps=self.steps, medians=self.medians, **{
                f'{key}{j}': x for key, arrays in [('w', self.params), ('m', self.mom), ('s', self.vel)] for j, x in enumerate(arrays)})
        os.replace(path.with_suffix('.tmp'), path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as f:
            self.steps = int(f['steps']); self.medians = f['medians'].tolist()
            for key, arrays in [('w', self.params), ('m', self.mom), ('s', self.vel)]:
                for j, x in enumerate(arrays):
                    x[:] = f[f'{key}{j}']


def traverse(arms, banks, seed, train):
    """All arms see exactly the same rows, centers, physical negatives and target budget."""
    rng = random_stream(*seed)
    outputs = {a.name: [] for a in arms}
    for start in range(0, len(banks[0]), B):
        v, e2, e1 = (bank[start:start+B] for bank in banks)
        vis = visible_features(v)
        shells = {'physical': e2, 'near': e1}
        features = {}
        for a in arms:
            key = (a.task, a.g)
            if key not in features:
                if a.task == 'planted' and 'planted' not in shells:
                    shells['planted'] = plant(v, e2, rng)
                features[key] = target_features(shells[a.task], a.g)
            stat = a.compute(v, vis, features[key], train)
            if not train:
                outputs[a.name].append(stat)
    return {} if train else {k: np.array(v) for k, v in outputs.items()}


def evaluate(arms, state, group, tables, reps, seed):
    total = {}
    for anchor in range(reps):
        banks = draw(state, group, random_stream(*seed, anchor), tables)
        stats = traverse(arms, banks, [*seed, anchor, 72], False)
        for key, rows in stats.items():
            if key not in total:
                total[key] = rows.copy()
            else:
                total[key] += rows
    return {k: v/reps for k, v in total.items()}


def budget(round_id):
    bracket, stage = divmod(round_id, 4)
    parents = (4096, 6144, 8192, 10240)[stage]
    epochs = (2, 3, 5, 8)[stage] + 2*bracket
    return parents, epochs


def score_summary(rows):
    val = rows.mean(axis=0)
    return dict(raw=float(val[0]), shuffle=float(val[1]), excess=float(val[0]-val[1]),
                mean_max_probability=float(val[2]), state_fractions=val[3:].tolist())


def advance_stage(records):
    groups = {}
    for name, rec in records.items():
        a, k, g, task = rec['spec']
        family = role(a)
        key = (task, k, family, g if family != 'nonlinear' else '')
        groups.setdefault(key, []).append(name)
    keep = []
    for key, names in sorted(groups.items()):
        ranked = sorted(names, key=lambda m: (-records[m]['screen']['raw'], m))
        count = (len(names)+1)//2
        keep.extend(ranked[:count])
        for j, name in enumerate(ranked):
            records[name]['pruning'] = dict(stratum=list(key), rank=j+1, candidates=len(names), keep=count, kept=j < count)
    return sorted(keep)


def leaders(records, old, round_id):
    best = dict(old)
    for name, rec in records.items():
        a, k, g, task = rec['spec']
        item = dict(name=name, spec=rec['spec'], round=round_id, raw=rec['common']['raw'],
                    sha256=rec['sha256'], steps=rec['steps'])
        r = role(a)
        keys = [task] if task != 'physical' else [f'{r}-{k}-{g}', f'{r}-{k}']
        if task == 'physical' and r == 'nonlinear':
            keys.append('champion')
        for key in keys:
            if key not in best or (-item['raw'], item['round'], name) < (-best[key]['raw'], best[key]['round'], best[key]['name']):
                best[key] = item
    return best


def label_round(r):
    return f'round-{r:04d}'


def write_trend(out, receipts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for key, desc in [('champion', 'nonlinear b2'), ('affine', 'matched affine b2'), ('fixed', 'matched fixed b2'), ('near', 'nonlinear b1'), ('planted', '20% shell matching')]:
        values = []
        for rec in receipts:
            lead = rec['leaders']; k = lead['champion']['spec'][1]
            values.append(lead[f'{key}-{k}' if key in ('affine', 'fixed') else key]['raw'])
        ax.plot(range(len(values)), values, '.-', label=desc)
    ax.axhline(0, color='0.5', lw=.8); ax.set(xlabel='completed discovery round', ylabel='selected validation bound (nats)')
    ax.legend(fontsize=8); fig.tight_layout(); tmp = out/'validation.tmp.png'
    fig.savefig(tmp, dpi=130); plt.close(fig); os.replace(tmp, out/'validation.png')


def discover(args, design, tables):
    r = args.round
    completed = args.out/(label_round(r)+'.json')
    if completed.exists():
        return
    if (args.work/'freeze.json').exists() or (args.out/'confirmation.json').exists():
        raise ValueError('confirmation already frozen: discovery closed')
    if not 0 <= r < N_ROUNDS:
        raise ValueError('round outside preregistered bracket schedule')
    previous = [read_json(args.out/(label_round(j)+'.json')) for j in range(r)]
    if any(rec['design_id'] != design for rec in previous):
        raise ValueError('tournament design changed between rounds')
    state, meta = source(args.data, 9)
    if previous and previous[-1]['data']['sha256'] != meta['sha256']:
        raise ValueError('discovery source changed')
    train, val, _ = partition(9)[0]
    bracket, stage = divmod(r, 4)
    all_specs = {arm_name(s): s for s in specs()}
    active = sorted(all_specs) if stage == 0 else previous[-1]['advance']
    arm_dir = args.work/label_round(r); arm_dir.mkdir(exist_ok=True)
    ref = draw(state, train, random_stream(20), tables[:1])[0]
    medians = [float(np.median((ref == s).sum(1))) for s in (0, 1)]
    arms = [Arm(all_specs[name], bracket, medians) for name in active]
    if stage:
        for a in arms:
            checkpoint = args.work/label_round(r-1)/(a.name+'.npz')
            if digest(checkpoint) != previous[-1]['arms'][a.name]['sha256']:
                raise ValueError('damaged parent checkpoint')
            a.load(checkpoint)
    n, epochs = budget(r)
    for epoch in range(epochs):
        rng = random_stream(21, r, epoch)
        group = rng.permutation(train)[:n]
        banks = draw(state, group, rng, tables)
        traverse(arms, banks, [22, r, epoch], True)
    screen = evaluate(arms, state, val, tables, 2, [23, r])
    common = evaluate(arms, state, val, tables, 4, [24])
    records = {}
    for a in arms:
        path = arm_dir/(a.name+'.npz'); a.save(path)
        records[a.name] = dict(spec=a.spec, steps=a.steps, parameters=sum(int(w.size) for w in a.params),
                               screen=score_summary(screen[a.name]), common=score_summary(common[a.name]), sha256=digest(path))
    advance = advance_stage(records) if stage < 3 else []
    lead = leaders(records, previous[-1]['leaders'] if r else {}, r)
    rec = dict(mode='discover', round=r, status='ok' if lead['champion']['raw'] > 0 else 'inconclusive',
               hypothesis=PROTOCOL['hypothesis'], metric=PROTOCOL['primary_metric'],
               estimate=lead['champion']['raw'], selection_basis=PROTOCOL['discovery'], design_id=design,
               data=meta, params=dict(bracket=bracket, stage=stage, batch=B, hidden=H, root_seed=ROOT_SEED, split_seed=PAIR_SEED,
                                    optimizer='Adam 0.0015 / sqrt(1+step/1200), beta=(.9,.999); persisted across bracket stages'),
               counts=dict(parents_per_epoch=n, training_epochs_this_stage=epochs, validation_parents=len(val), fresh_screen_anchors=2,
                           common_validation_anchors=4, train_pairs_per_active_arm=n*epochs, active_arms=len(arms)),
               provenance=dict(source_path=str(args.data.resolve()), checkpoint_dir=label_round(r), draw_streams='21,round,epoch: rows/centres; 22,round,epoch: plant; 23,round: fresh validation; 24: common validation'),
               arms=records, active=active, advance=advance, advancement_reason=('bracket complete; next bracket starts all architectures from a fresh initialization' if stage == 3 else 'Within each reported stratum: top ceil(half) on fresh validation raw, name-order tie break'),
               leaders=lead, search_exhausted=r == N_ROUNDS-1)
    write_trend(args.out, [*previous, rec]); atomic_json(completed, rec)


def interval(x, primary=False):
    avg = float(x.mean()); se = float(x.std(ddof=1)/np.sqrt(len(x)))
    q95 = float(t.ppf(.975, len(x)-1))
    stat = dict(mean=avg, se=se, ci95=[avg-q95*se, avg+q95*se], independent_parent_batches=len(x))
    if primary:
        stat['simultaneous_lower'] = avg-float(t.ppf(1-ALPHA, len(x)-1))*se
    return stat


def confirm(args, design, tables):
    marker = args.out/'confirmation.json'
    if marker.exists():
        rec = read_json(marker)
        if rec['round'] != args.round or rec['design_id'] != design:
            raise ValueError('confirmation cannot be reopened on a different prefix/design')
        return
    if (args.out/(label_round(args.round+1)+'.json')).exists():
        raise ValueError('confirmation requires the highest completed discovery round')
    rec = read_json(args.out/(label_round(args.round)+'.json'))
    if rec['design_id'] != design:
        raise ValueError('confirmation design differs from discovery')
    leads = rec['leaders']; win = leads['champion']; _, k, g, _ = win['spec']
    selected = {key: leads[key] for key in ('champion', f'affine-{k}', f'fixed-{k}', f'affine-{k}-{g}', f'fixed-{k}-{g}', 'near', 'planted', 'nonlinear-2', 'nonlinear-4')}
    freeze = dict(round=args.round, design_id=design, selected=selected, indices=[9, 8, 7], gates=PROTOCOL['gate'])
    lock = args.work/'freeze.json'
    if lock.exists():
        if read_json(lock) != freeze:
            raise ValueError('frozen prefix or models changed')
    else:
        atomic_json(lock, freeze)
    models = {}; ids = {}
    for key, item in selected.items():
        ident = (item['name'], item['round']); ids[key] = ident
        if ident not in models:
            a = Arm(item['spec'], item['round']//4, [0, 0]); path = args.work/label_round(item['round'])/(a.name+'.npz')
            if digest(path) != item['sha256']:
                raise ValueError('checkpoint hash mismatch at freeze')
            a.load(path)
            # Separate checkpoints of the same architecture can be selected for different comparisons.
            a.name += '@'+str(item['round']); models[ident] = a
    sources, scores, primary = [], {}, {}
    batch_rows = []
    for index in freeze['indices']:
        state, meta = source(args.data, index); sources.append(meta)
        if index == 9 and meta['sha256'] != rec['data']['sha256']:
            raise ValueError('training release changed before confirmation')
        group = partition(index)[0][2]
        used = [a for a in models.values() if index == 9 or a.task == 'physical']
        bank = evaluate(used, state, group, tables, 24, [25, index]); del state
        stats = {key: bank[models[ident].name] for key, ident in ids.items() if models[ident].name in bank}
        raw, shuffle = stats['champion'][:, :2].T
        gates = {'raw': raw, 'excess': raw-shuffle, 'vs_affine': raw-stats[f'affine-{k}'][:, 0], 'vs_fixed': raw-stats[f'fixed-{k}'][:, 0]}
        if index == 9:
            for ctrl in ('near', 'planted'):
                value = stats[ctrl]
                gates[f'{ctrl}_raw'] = value[:, 0]; gates[f'{ctrl}_excess'] = value[:, 0]-value[:, 1]
        primary[str(index)] = {name: interval(rows, primary=True) for name, rows in gates.items()}
        scores[str(index)] = {name: dict(**score_summary(rows), raw_interval=interval(rows[:, 0]), shuffled_interval=interval(rows[:, 1])) for name, rows in stats.items()}
        for name, rows in stats.items():
            for j, row in enumerate(rows):
                batch_rows.append((index, j, name, row[0], row[1]))
    gates = [x for grp in primary.values() for x in grp.values()]
    assert len(gates) == 16
    success = all(x['simultaneous_lower'] > 0 for x in gates)
    tmp = args.out/'confirmation-batches.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(('release_index', 'parent_batch', 'frozen_model', 'raw_nats', 'shuffled_nats')); writer.writerows(batch_rows)
    os.replace(tmp, args.out/'confirmation-batches.csv')
    atomic_json(marker, dict(mode='confirm', round=args.round, design_id=design, status='ok' if success else 'inconclusive',
                             hypothesis=PROTOCOL['hypothesis'], metric='Frozen nonlinear raw b=2 InfoNCE (nats), index 9; joint decision requires all 16 adjusted contrasts',
                             estimate=primary['9']['raw']['mean'], selection_basis=PROTOCOL['discovery'], data=sources,
                             params=dict(batch=B, hidden=H, buffer=2, width=3, per_contrast_one_sided_alpha=ALPHA),
                             counts=dict(parents_per_temperature=SPLIT[2], batches_per_temperature=SPLIT[2]//B, anchors_per_parent=24, confirmatory_contrasts=len(gates)),
                             provenance=dict(freeze_sha256=digest(lock), source_path=str(args.data.resolve()), anchor_stream='25,release_index,anchor'),
                             freeze=freeze, primary=primary, scores=scores, gate=PROTOCOL['gate'], passed=success,
                             interpretation='Conditional on selected fits. A failed gate is not evidence of zero physical MI and a passing gate is not a phase identification.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['discover', 'confirm']); parser.add_argument('--round', type=int, required=True)
    for name in ('data', 'work', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); args.work.mkdir(parents=True, exist_ok=True); args.out.mkdir(parents=True, exist_ok=True)
    protocol = dict(**PROTOCOL, code_sha256=digest(Path(__file__)), split_seed=PAIR_SEED, root_seed=ROOT_SEED,
                    stage_budgets=[dict(round=i, parents=budget(i)[0], epochs=budget(i)[1]) for i in range(N_ROUNDS)])
    design = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    with (args.work/'writer.lock').open('a') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX)
        sheet = args.out/'design.json'
        if sheet.exists():
            if read_json(sheet) != protocol:
                raise ValueError('immutable protocol has changed; use independent output and working directories')
        else:
            atomic_json(sheet, protocol)
        (discover if args.mode == 'discover' else confirm)(args, design, make_tables())


if __name__ == '__main__':
    main()
