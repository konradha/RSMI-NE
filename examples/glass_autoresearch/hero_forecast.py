"""HERO forward-central stage: predict ONE frozen signed central rule from E and I.

No archive access: physical commands consume only separately released bank artifacts.
See results/l20-autoresearch/hero-central/forecast-design.json.
"""
import argparse
import hashlib
import itertools
import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import norm, t

PUBLIC = Path(__file__).resolve().parents[2] / 'results/l20-autoresearch/hero-central'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/hero-central'
SPLITS = ('fit', 'tune', 'rule')
ROLES = ('cold', 'transfer', 'hot', 'issue')
SHAPES = {'fit': (4096, 64), 'tune': (512, 64), 'rule': (512, 64),
          'cold': (1024, 8), 'transfer': (1024, 8), 'hot': (1024, 8), 'issue': (256, 1)}
TARGETS = ('f', 'residual', 'conventional')
METHODS = ('inventory', 'm0', 'm1', 'm0_noI', 'm1_noI')
# Larger first: argmin breaks exact ties in favor of stronger regularization.
ALPHAS = (1., .01, .0001)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with open(path, 'rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def roster_hash(parent, anchor):
    digest = hashlib.sha256()
    for a in (parent, anchor):
        digest.update(np.ascontiguousarray(a, dtype='<i8'))
    return digest.hexdigest()


def write_json(path, data):
    with open(path, 'x') as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write('\n')


def write_npz(path, data):
    with open(path, 'xb') as out:
        np.savez_compressed(out, **data)


@lru_cache(None)
def shell_coords(radius):
    require(radius in (4, 5), 'only the two locked radii exist')
    raw = np.array([p for p in itertools.product(range(-4, 5), repeat=3) if max(map(abs, p)) == 4])
    return np.where(abs(raw) == 4, np.sign(raw)*radius, raw)


@lru_cache(8)
def _crystals(coords_tuple):
    coords = np.array(coords_tuple)
    result = []
    for proto, period in (((1, 1, 1), 3), ((1, 2, 3), 7), ((1, 0, 0), 5)):
        masks = set()
        for perm in itertools.permutations(range(3)):
            parity = (-1)**sum(perm[i] > perm[j] for i in range(3) for j in range(i+1, 3))
            for signs in itertools.product((-1, 1), repeat=3):
                if np.prod(signs)*parity != 1:
                    continue
                phases = coords @ (np.array(proto)[list(perm)] * signs) % period
                for phase in range(period):
                    mask = phases == phase
                    if 0 < mask.sum() < len(mask):
                        masks.add(mask.tobytes())
        col = np.column_stack([np.frombuffer(m, dtype=bool) for m in sorted(masks)])
        contrast = col / col.sum(0) - (~col) / (~col).sum(0)
        result.append(np.ascontiguousarray(contrast.astype('f')))
    return result


def crystal_features(E, coords):
    """Analytic A3/B7 trial crystals and generic P5; (..., site) -> (...,6)."""
    masks = _crystals(tuple(map(tuple, np.asarray(coords).tolist())))
    E = np.asarray(E)
    flat = E.reshape(-1, len(coords))
    out = np.empty((len(flat), 6), dtype='f')
    for start in range(0, len(flat), 512):
        e = flat[start:start+512]
        vac = (e == 0).astype('f')
        for j, matrix in enumerate(masks):
            channel = vac if j == 2 else vac - (e == j+1)
            contrasts = channel @ matrix
            out[start:start+len(e), 2*j] = contrasts.max(axis=1)
            out[start:start+len(e), 2*j+1] = np.sqrt(np.mean(contrasts**2, axis=1))
    return out.reshape(*E.shape[:-1], 6)


@lru_cache(None)
def geometry(radius):
    coords = shell_coords(radius)
    lookup = {tuple(k): i for i, k in enumerate(coords)}
    edges = []
    for step in (1, 2):
        for axis in range(3):
            direction = step*np.eye(3, dtype=int)[axis]
            pairs = [(i, lookup[tuple(k+direction)]) for i, k in enumerate(coords) if tuple(k+direction) in lookup]
            edges.append(np.array(pairs).T)
    return coords, edges


def conventional_features(E, radius=4):
    coords, edges = geometry(radius)
    E = np.asarray(E)
    eall = E.reshape(-1, 386)
    result = np.empty((len(eall), 2+12+30+14+6), dtype='f')
    for s in range(0, len(eall), 512):
        e = eall[s:s+512]
        cols = [(e == k).mean(1) for k in (0, 1)]
        for j in range(3):
            for sign in (-1, 1):
                face = e[:, coords[:, j] == sign*radius]
                cols.extend((face == k).mean(1) for k in (0, 1))
        degree = np.zeros_like(e)
        for d, (i, j) in enumerate(edges):
            a, b = e[:, i], e[:, j]
            if d < 3:
                degree[:, i] += b != 0
                degree[:, j] += a != 0
            cols.extend(mask.mean(1) for mask in ((a == 0) & (b == 0), (a == 1) & (b == 1),
                        (a != 0) & (b != 0), (a == 0) & (b == 1), (a == 1) & (b == 0)))
        cols.extend(((e == k) & (degree == n)).mean(1) for k in (1, 2) for n in range(7))
        result[s:s+len(e)] = np.column_stack([*cols, crystal_features(e, coords)])
    return result.reshape(*E.shape[:-1], result.shape[-1])


def _flat(a):
    a = np.asarray(a)
    return a.reshape(-1, a.shape[-1])


def _features(I, conventional, critic, name):
    terms = [] if name.endswith('_noI') else [_flat(I)]
    if name != 'inventory':
        terms.append(_flat(conventional))
    if name.startswith('m1'):
        w = _flat(critic)
        terms.append(w - w.mean(axis=1, keepdims=True))
    return np.concatenate(terms, axis=1, dtype=np.float64)


def _scale(x):
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-7] = 1.
    return mu, sd


def _expand(x, state):
    z = (x-state['mu']) / state['sd']
    return np.column_stack((z, (z*z-state['sqmu'])/state['sqsd']))


def grid_predict(x, model):
    """Return (row, alpha, target). Constants and calibrations are FIT-only."""
    z = _expand(x, model)
    p = z @ model['weights'].reshape(z.shape[1], -1)
    return p.reshape(len(z), len(ALPHAS), -1) + model['mean']


def selected(pred, model):
    return pred[..., model['selected'], np.arange(len(model['selected']))]


def grid_fit(x, y, xt, yt):
    y, yt = _flat(y), _flat(yt)
    mu, sd = _scale(x)
    sqmu, sqsd = _scale(np.square((x-mu)/sd))
    model = dict(mu=mu, sd=sd, sqmu=sqmu, sqsd=sqsd, mean=y.mean(0))
    z = _expand(x, model)
    gram = z.T @ z / len(z)
    rhs = z.T @ (y-model['mean']) / len(z)
    model['weights'] = np.stack([np.linalg.solve(gram + a*np.eye(len(gram)), rhs) for a in ALPHAS], axis=1)
    p = grid_predict(xt, model)
    model['selected'] = np.argmin(np.mean((p-yt[:, None, :])**2, axis=0), axis=0)
    m = selected(p, model)
    model['quantiles'] = np.quantile(m, [.2,.4,.6,.8], axis=0)
    model['abs90'] = np.quantile(abs(m-yt), .90, axis=0)
    return model


def parent_stats(v):
    v = np.asarray(v, dtype=float)
    require(v.ndim == 1 and len(v) >= 2 and np.isfinite(v).all(), 'invalid parent statistic')
    n, m, sd = len(v), v.mean(), v.std(ddof=1)
    se = sd/np.sqrt(n)
    t95 = t.ppf(.975, n-1)*se
    return dict(point=float(m), lower=float(m-t95), upper=float(m+t95),
                lower_one_sided=float(m-t.ppf(.95, n-1)*se), sd_parent=float(sd), n=n)


def cluster_ratio(numerator, count):
    numerator, count = np.asarray(numerator), np.asarray(count)
    if count.sum() == 0:
        return None
    m = numerator.sum()/count.sum()
    # First-order cluster influence for a ratio; retain parents with zero weight.
    return parent_stats(m + (numerator - m*count)/count.mean())


def _rule_eval(preds, truth, fit_truth, names):
    v = float(np.var(fit_truth[...,0]))
    require(v > 1e-8, 'collapsed signed central rule')
    losses = {k: (a-truth)**2 for k,a in preds.items()}
    results = {}
    for j, name in enumerate(names):
        den = float(np.var(fit_truth[...,j])) if name == 'conventional' else v
        if den <= 1e-8:
            results[name] = {'collapsed': True}
            continue
        results[name] = {f'vs_{k}': parent_stats(((losses[k][...,j]-losses['m1'][...,j])/den).mean(1)) for k in ('m0','inventory')}
        results[name]['absolute_mse'] = {k: parent_stats(a[...,j].mean(1)) for k,a in losses.items()}
    raw, res = results['f'], results['residual']['vs_m0']
    fraction = float(np.var(fit_truth[...,1])/v)
    passed = all(raw['vs_'+k]['point'] >= .01 and raw['vs_'+k]['lower_one_sided'] > 0 for k in ('m0','inventory')) and res['point'] > 0 and fraction >= .10
    delta = (losses['m0'][...,:1]-losses['m1'][...,:1])[...,0]/v
    sd8 = delta[:,:8].mean(1).std(ddof=1)
    power = float(norm.cdf(.01*np.sqrt(1024)/sd8-norm.ppf(.975))) if sd8 else 1.
    results.update(passed=bool(passed), variance_f=v, residual_fraction=fraction, projected_cold_power=power, rule_sd8=float(sd8))
    return results


def _fit_methods(feats, targets, names, methods):
    require(tuple(names[:2]) == ('f','residual') and len(names) == targets['fit'].shape[-1], 'single named rule and central residual required')
    mf, preds, grids = {}, {}, {}
    for m in methods:
        x = {s: _features(*feats[s], m) for s in SPLITS}
        model = grid_fit(x['fit'], targets['fit'], x['tune'], targets['tune'])
        grid = grid_predict(x['rule'], model).reshape(*targets['rule'].shape[:2], len(ALPHAS), len(names))
        mf[m], grids[m], preds[m] = model, grid, selected(grid, model)
    info = _rule_eval(preds, targets['rule'], targets['fit'], names)
    info['selected_alphas'] = {m: [ALPHAS[k] for k in model['selected']] for m,model in mf.items()}
    info['matched_alphas_f'] = {str(a): parent_stats(((((grids['m0'][...,k,0]-targets['rule'][...,0])**2 - (grids['m1'][...,k,0]-targets['rule'][...,0])**2)/info['variance_f']).mean(1))) for k,a in enumerate(ALPHAS)}
    return mf, info


def nominate(banks, targets, shell, names):
    feats = {s: (banks[s]['I'], conventional_features(banks[s]['E']), shell[s]) for s in SPLITS}
    return _fit_methods(feats, targets, names, ('inventory','m0','m1'))[1]


def control():
    """No site configurations. Finite alphabet sufficient-statistic recovery check."""
    start = time.monotonic()
    path = PUBLIC / 'forecast-control.json'
    require(not path.exists(), 'numerical ticket already spent')
    rng = np.random.default_rng(1795526371)
    report = {}
    for g in (.02, .125, 0.):
        feats, targets = {}, {}
        for split,n in zip(SPLITS, (8192,2048,8192)):
            u, b, i2 = rng.choice([-1,1], size=(3,n,8))
            c = u * np.where(rng.random((n,8)) < (1+np.sqrt(2*(g or .02)))/2, 1, -1)
            if g == 0:
                c = c[rng.permutation(n)]
            ordinary = rng.choice([-1,1], size=(n,8,12))
            feats[split] = (np.stack((b,i2), axis=-1), ordinary,
                            np.stack((c,-c,c,-c), axis=-1))
            targets[split] = np.stack(((u+b)/np.sqrt(2),u/np.sqrt(2),b), axis=-1)
        _, row = _fit_methods(feats,targets,TARGETS,('inventory','m0','m1'))
        gain = row['f']['vs_m0']
        if g == 0:
            ok = gain['lower'] >= -.005 and gain['upper'] <= .005 and not row['passed']
        else:
            ok = abs(gain['point']-g) <= (.007 if g == .02 else .015) and gain['lower'] > .01 and row['passed']
        report[str(g)] = dict(oracle_mse_m0=.5, oracle_mse_m1=.5-g, oracle_gain=g,
                              criterion_passed=bool(ok), score=row)
    write_json(path, dict(stage='numerical calibration only; no physical arrays or forward-encoder training',
                          seed=1795526371, n_parents=[8192,2048,8192], anchors=8,
                          design_sha256=sha(PUBLIC/'forecast-design.json'), cases=report,
                          passed=all(r['criterion_passed'] for r in report.values()),
                          wall_sec=time.monotonic()-start))


def _inventory_codes(I):
    return I[...,0].astype(np.int32)*344 + I[...,1]


def fit_support(I):
    codes = _inventory_codes(I)
    unique = np.unique(np.arange(len(I))[:, None]*(344**2) + codes)
    return np.bincount(unique % (344**2), minlength=344**2)


def matching(parent, I, m0, m1, sd):
    """Pre-outcome greedy positive ordering on one opportunity per parent."""
    code = _inventory_codes(I[:,0])
    d = m1[:,0] - m0[:,0]
    group = np.full(m0.shape, -1, dtype='i4')
    sign = np.zeros(m0.shape, dtype='i1')
    count = 0
    for cell in np.unique(code):
        ids = np.flatnonzero(code == cell)
        candidates = []
        for i,j in itertools.combinations(ids, 2):
            hi,lo = (i,j) if d[i] > d[j] else (j,i)
            if d[hi] > d[lo] and m1[hi,0] > m1[lo,0] and abs(m0[hi,0]-m0[lo,0]) <= .25*sd:
                candidates.append((d[lo]-d[hi], int(parent[hi]), int(parent[lo]), hi,lo))
        for _,__,___,hi,lo in sorted(candidates):
            if group[hi,0] < 0 and group[lo,0] < 0:
                group[[hi,lo],0] = count
                sign[hi,0], sign[lo,0] = 1,-1
                count += 1
    return group, sign


def _public_pin(path):
    """A release on disk is insufficient: compare it to its committed Git blob."""
    root = PUBLIC.parents[2]
    rel = str(path.relative_to(root))
    rev = subprocess.check_output(['git','rev-parse',f'HEAD:{rel}'], cwd=root, text=True).strip()
    obj = subprocess.check_output(['git','hash-object','--',rel], cwd=root, text=True).strip()
    require(obj == rev, f'uncommitted input: {path.name}')


def _release(stage, lock):
    from datetime import datetime, timezone
    import signal
    cutoff = '2026-11-24T21:00:25+00:00' if stage == 'freeze' else '2026-11-24T21:45:25+00:00'
    remaining = (datetime.fromisoformat(cutoff)-datetime.now(timezone.utc)).total_seconds()
    require(remaining > 0, 'physical deadline expired')
    # Kill the CLI even if a long native/BLAS operation overruns its authorized window.
    signal.setitimer(signal.ITIMER_REAL, remaining)
    path = PUBLIC/f'forecast-release-{stage}.json'
    require(sha(path) == lock, 'wrong release digest')
    _public_pin(path)
    release = json.loads(path.read_bytes())
    require(release['stage'] == stage and release['mode'] in ('confirmatory','nonconfirmatory'), 'invalid release')
    require(release['mode'] == 'confirmatory' or stage == 'freeze', 'engineering cannot spend an exam')
    require(release['design_sha256'] == sha(PUBLIC/'forecast-design.json'), 'protocol changed')
    require(release['control_sha256'] == sha(PUBLIC/'forecast-control.json'), 'forecast control changed')
    require(json.loads((PUBLIC/'forecast-control.json').read_bytes())['passed'] or release['mode'] == 'nonconfirmatory',
            'forecast control failed')
    _public_pin(PUBLIC/'forecast-design.json')
    _public_pin(PUBLIC/'forecast-control.json')
    return release


def _input(release, key, specific=None, public=False):
    spec = release['inputs'][key]
    base = PUBLIC if public else WORK
    path = (base/spec['path']).resolve(strict=True)
    require(path.is_relative_to(base.resolve()) and (specific is None or path == (base/specific).resolve()), 'unexpected input path')
    if public:
        _public_pin(path)
    require(sha(path) == spec['sha256'], f'{key}: pinned bytes changed')
    return path


def _reference_critic(release):
    path = _input(release,'critic',public=True)
    nomination = PUBLIC/'nomination.json'
    _public_pin(nomination)
    rows = [r for r in json.loads(nomination.read_bytes())['models'] if r['seed'] == 6101]
    require(len(rows) == 1 and (PUBLIC.parents[2]/rows[0]['public']).resolve() == path and
            rows[0]['sha256'] == release['inputs']['critic']['sha256'], 'wrong reference critic')
    return path


def _validate_keys(path, role, stage):
    with np.load(path, allow_pickle=False) as z:
        required = {'parent','anchor'}
        if stage != 'blind':
            required.add('V')
        if stage != 'hidden':
            required.update(('I','E','E5'))
        require(set(z.files) == required, 'wrong stage fields')
        a = {k:z[k] for k in required}
    p,q = SHAPES[role]
    require(a['parent'].shape == (p,) and a['parent'].dtype.kind in 'iu' and len(np.unique(a['parent'])) == p, 'incomplete parent roster')
    require(a['anchor'].shape == (p,q) and a['anchor'].dtype.kind in 'iu' and (a['anchor'] < 8000).all() and (a['anchor'] >= 0).all(), 'bad opportunities')
    require((np.diff(np.sort(a['anchor'],axis=1).astype('i4'),axis=1) != 0).all(), 'repeated opportunity')
    for key, n in (('V',27),('E',386),('E5',386)):
        if key in a:
            require(a[key].shape == (p,q,n) and a[key].dtype == np.uint8 and a[key].max() <= 2, 'wrong ternary bank')
    if 'I' in a:
        require(a['I'].dtype == np.uint16 and a['I'].shape == (p,q,2) and (a['I'].sum(-1) <= 343).all(), 'wrong aggregate inventory')
    return a


def _critic(model, E):
    from hero_train import shell_logits
    flat = E.reshape(-1,386)
    out = np.empty((len(flat),4), dtype='f')
    for s in range(0,len(flat),512):
        out[s:s+512] = shell_logits(model,flat[s:s+512])
    require(np.isfinite(out).all(), 'nonfinite shell evidence')
    return out.reshape(*E.shape[:2],4)


def _targets(V, spec):
    from hero_rule import apply_rule, conventional_rule, residual_rule
    result = np.stack([fn(V,spec) for fn in (apply_rule,residual_rule,conventional_rule)],axis=-1)
    require(result.shape == (*V.shape[:2],3) and np.isfinite(result).all(), 'invalid frozen central target')
    return result


def _check_ticket(stage):
    path = WORK/f'forecast-spent-{stage}.json'
    require(not path.exists(), f'{stage} ticket already spent')
    return path


def freeze(release, lock):
    ticket = _check_ticket('freeze')
    receipt = PUBLIC/'forecast-freeze.json'
    model_path = WORK/'frozen-forecasts.npz'
    require(not receipt.exists() and not model_path.exists(), 'forecast freeze already exists')
    require(release['critic_seed'] == 6101, 'use fixed lowest reference seed of nominated arm')
    rule_path = _input(release,'rule_spec','rule.json',public=True)
    spec = json.loads(rule_path.read_bytes())
    require(spec['schema'] == 'hero-rule-1' and spec['symbolic_passed'] is True,
            'no reproducible, adequately distilled symbolic nominee; refuse before bank access')
    nominee = (spec['status'] == 'symbolic_nominee_only_pending_boundary_gate' and
               spec['rediscovery'] is False and spec['residual_fraction'] >= .10)
    conventional = spec['status'] == 'conventional_rediscovery_no_exam' and spec['rediscovery'] is True
    require(nominee or (conventional and release['mode'] == 'nonconfirmatory'),
            'only a nominee, or explicit dev-only conventional rediscovery, may enter freeze')
    critic_path = _reference_critic(release)
    banks_paths = {s:_input(release,s,f'{s}.npz') for s in SPLITS}
    write_json(ticket,dict(stage='freeze',release_sha256=lock))
    start = time.monotonic()
    banks = {s:_validate_keys(path,s,'discovery') for s,path in banks_paths.items()}
    for a,b in itertools.combinations(SPLITS,2):
        require(not np.intersect1d(banks[a]['parent'],banks[b]['parent']).size, 'discovery parent leak')
    targets = {s:_targets(banks[s]['V'],spec) for s in SPLITS}
    counts = fit_support(banks['fit']['I'])
    coverage = float((counts[_inventory_codes(banks['rule']['I'])] >= 8).mean())
    arrays = dict(fit_inventory_parent_counts=counts, fit_variances=targets['fit'].var((0,1)))
    scores = {}
    with np.load(critic_path,allow_pickle=False) as critic:
        for radius,key in ((4,'E'),(5,'E5')):
            feats = {s:(a['I'],conventional_features(a[key],radius),_critic(critic,a[key])) for s,a in banks.items()}
            models, info = _fit_methods(feats,targets,TARGETS,METHODS)
            for m,model in models.items():
                arrays.update({f'r{radius}_{m}_{k}':v for k,v in model.items()})
            scores[f'r{radius}'] = info
    primary = scores['r4']
    gates = dict(symbolic=bool(spec['symbolic_passed']), novel=not bool(spec['rediscovery']),
                 rule_forecast=primary['passed'], support=coverage >= .90, power=primary['projected_cold_power'] >= .80,
                 confirmatory=release['mode'] == 'confirmatory')
    write_npz(model_path,arrays)
    write_json(receipt, dict(stage='freeze', release_sha256=lock, mode=release['mode'],
                             model_sha256=sha(model_path), critic_sha256=sha(critic_path), rule_sha256=sha(rule_path),
                             rule_feature=spec['feature'], rule_support=coverage, variance_units='FIT population Var(f)',
                             target_order=TARGETS, alpha_order=ALPHAS, scores=scores, gates=gates, passed=all(gates.values()),
                             wall_sec=time.monotonic()-start))


def _frozen_model(z, radius, name):
    prefix = f'r{radius}_{name}_'
    return {k[len(prefix):]:z[k] for k in z.files if k.startswith(prefix)}


def issue(release, role, lock):
    ticket = _check_ticket('issue-'+role)
    report = PUBLIC/f'predictions-{role}.json'
    dest = PUBLIC/f'predictions-{role}.npz'
    require(not report.exists() and not dest.exists(), 'issuance already exists')
    freeze_path = _input(release,'freeze','forecast-freeze.json',public=True)
    freeze_info = json.loads(freeze_path.read_bytes())
    require(freeze_info['passed'], 'failed freeze cannot issue')
    preceding = {'hot':None, 'cold':'hot', 'transfer':'cold', 'issue':'transfer'}[role]
    if preceding is not None:
        p = _input(release,'preceding',f'forecast-score-{preceding}.json',public=True)
        require(json.loads(p.read_bytes())['passed'], 'previous gate failed')
    blind = _input(release,'blind',f'blind-{role}.npz')
    model_path = _input(release,'model','frozen-forecasts.npz')
    critic_path = _reference_critic(release)
    require(sha(model_path) == freeze_info['model_sha256'] and sha(critic_path) == freeze_info['critic_sha256'], 'model was not frozen')
    write_json(ticket,dict(stage='issue',role=role,release_sha256=lock))
    start = time.monotonic()
    bank = _validate_keys(blind,role,'blind')
    out = {k:bank[k] for k in ('parent','anchor','I')}
    with np.load(model_path,allow_pickle=False) as z, np.load(critic_path,allow_pickle=False) as critic:
        variances = z['fit_variances']
        out['support'] = z['fit_inventory_parent_counts'][_inventory_codes(bank['I'])] >= 8
        for radius,key in ((4,'E'),(5,'E5')):
            feat = conventional_features(bank[key],radius)
            h = _critic(critic,bank[key])
            for m in METHODS:
                model = _frozen_model(z,radius,m)
                grid = grid_predict(_features(bank['I'],feat,h,m),model).reshape(*SHAPES[role],len(ALPHAS),len(TARGETS))
                pred = selected(grid,model)
                prefix = f'r{radius}_{m}'
                out[prefix+'_grid'] = grid
                out[prefix] = pred
                out[prefix+'_bins'] = (pred[...,None,:] > model['quantiles']).sum(-2).astype('u1')
                if m == 'm1':
                    out[prefix+'_lo90'] = pred - model['abs90']
                    out[prefix+'_hi90'] = pred + model['abs90']
    out['pair_group'],out['pair_sign'] = matching(out['parent'],out['I'],out['r4_m0'][...,0],out['r4_m1'][...,0],np.sqrt(variances[0]))
    require(all(np.isfinite(v).all() for v in out.values()), 'nonfinite issuance')
    write_npz(dest,out)
    group_count = int((out['pair_sign'] == 1).sum())
    write_json(report,dict(role=role, stage='boundary-only issuance, no interior opened', release_sha256=lock,
                           blind_sha256=sha(blind), roster_sha256=roster_hash(out['parent'],out['anchor']),
                           prediction_sha256=sha(dest), opportunities=out['anchor'].size, parents=len(out['parent']),
                           target_order=TARGETS, alpha_order=ALPHAS, fit_variances=variances.tolist(),
                           freeze_sha256=sha(freeze_path), rule_sha256=freeze_info['rule_sha256'],
                           support=float(out['support'].mean()), matched_pairs=group_count, matching_supported=group_count>=16,
                           uncertainty='TUNE marginal absolute-error 90th percentile, not a thermal variance or pointwise coverage guarantee',
                           wall_sec=time.monotonic()-start))


def exact_inventory_null(I, f, d, variance):
    """Within-exact-inventory covariance U-statistic excluding same-parent pairs.

    Each supported inventory cell is weighted by its number of opportunities.
    Delete a whole parent and recompute those weights, cell eligibility and U-stat.
    """
    p,q = f.shape
    codes = _inventory_codes(I).ravel()
    _, cells = np.unique(codes,return_inverse=True)
    cells = cells.ravel()
    ncell = int(cells.max())+1
    group, inv = np.unique(np.repeat(np.arange(p),q)*ncell + cells,return_inverse=True)
    inv = inv.ravel()
    par, cell = np.divmod(group,ncell)
    n = np.bincount(inv)
    F = np.bincount(inv,weights=f.ravel())
    D = np.bincount(inv,weights=d.ravel())
    FD = np.bincount(inv,weights=(f*d).ravel())
    # Column 5 removes the within-one-parent part of the naive pair numerator.
    pieces = np.column_stack((n,F,D,FD,n*n,n*FD-F*D))
    totals = np.zeros((ncell,6))
    np.add.at(totals,cell,pieces)

    def effect(table):
        nc,fc,dc,fd,n2,intra = table.T
        weights = nc*nc-n2
        keep = weights > 0
        if not keep.any():
            return None,0
        cov = (nc*fd-fc*dc-intra)[keep]/weights[keep]
        return float(2*np.average(cov,weights=nc[keep])/variance),int(nc[keep].sum())

    value,count = effect(totals)
    if value is None:
        return dict(point=None, eligible_fraction=0., passed=False)
    start = np.searchsorted(par,np.arange(p+1))
    leave = np.empty(p)
    for i in range(p):
        s = slice(start[i],start[i+1])
        tmp = totals.copy()
        tmp[cell[s]] -= pieces[s]
        v,_ = effect(tmp)
        if v is None:
            return dict(point=value, eligible_fraction=count/(p*q), passed=False)
        leave[i] = v
    se = np.sqrt((p-1)/p * ((leave-leave.mean())**2).sum())
    hw = float(t.ppf(.975,p-1)*se)
    return dict(point=value, lower=value-hw,upper=value+hw, eligible_fraction=count/(p*q),
                parents=p, exact_cells=ncell, passed=bool(count/(p*q) >= .90 and abs(value)+hw <= .01))


def _matched_result(truth, pred):
    group,sign = pred['pair_group'],pred['pair_sign']
    ng = int((sign == 1).sum())
    if not ng:
        return dict(matched_pairs=0,supported=False)
    keep = group >= 0
    contrast = lambda a: np.bincount(group[keep],weights=(sign*a)[keep],minlength=ng)
    actual = contrast(truth[...,0])
    return dict(matched_pairs=ng,supported=ng >= 16,
                predicted_gap=float(contrast(pred['r4_m1'][...,0]).mean()),
                baseline_gap=float(contrast(pred['r4_m0'][...,0]).mean()),
                observed_gap=parent_stats(actual) if ng > 1 else float(actual[0]),
                fraction_ordered=parent_stats((actual > 0) + .5*(actual == 0)) if ng > 1 else None)


def _calibration(pred, y, prefix, j):
    bins = pred[prefix+'_bins'][...,j]
    point = pred[prefix][...,j]
    rows = []
    for b in range(5):
        mask = (bins == b)
        if not mask.any():
            continue
        counts = mask.sum(1)
        rows.append(dict(bin=b,n=int(counts.sum()),parents=int(np.count_nonzero(counts)),
                         predicted=float(point[mask].mean()),
                         actual=cluster_ratio((mask*y).sum(1),counts)))
    return rows


def _calibration_plot(role, result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2,figsize=(8.5,3.5),constrained_layout=True)
    for pane,name in zip(ax,('f','residual')):
        for m, style in (('inventory','s'),('m0','o'),('m1','^')):
            rows = result['r4'][name]['calibration'][m]
            x = [r['predicted'] for r in rows]
            y = np.array([r['actual']['point'] for r in rows])
            lo = [r['actual']['lower'] for r in rows]
            hi = [r['actual']['upper'] for r in rows]
            pane.errorbar(x,y,yerr=[y-lo,hi-y],fmt=style+'-',lw=1,ms=4,capsize=2,label=m)
        bounds = pane.get_xlim()
        pane.plot(bounds,bounds,':',color='0.6',lw=1)
        pane.set(xlabel='Issued signed prediction',ylabel='Mean observed target (95% CI)',title=f'{role}: {name}')
        pane.legend(frameon=False,fontsize=8)
    with open(path,'xb') as out:
        fig.savefig(out,format='png',dpi=150)
    plt.close(fig)


def score(release,role,lock):
    require(role in ('hot','cold','transfer'), 'no unused-boundary interior may be opened')
    ticket = _check_ticket('score-'+role)
    report = PUBLIC/f'forecast-score-{role}.json'
    figure = PUBLIC/f'forecast-calibration-{role}.png'
    require(not report.exists() and not figure.exists(),'score already present')
    receipt_path = _input(release,'receipt',f'predictions-{role}.json',public=True)
    receipt = json.loads(receipt_path.read_bytes())
    ppath = _input(release,'predictions',f'predictions-{role}.npz',public=True)
    require(receipt['role'] == role and receipt['prediction_sha256'] == sha(ppath), 'unbound signed table')
    require(role == 'hot' or receipt['support'] >= .90, 'poor pre-outcome inventory support; do not open labels')
    rpath = _input(release,'rule_spec','rule.json',public=True)
    require(sha(rpath) == receipt['rule_sha256'], 'rule changed after freeze')
    spec = json.loads(rpath.read_bytes())
    with np.load(ppath,allow_pickle=False) as z:
        pred = {k:z[k] for k in z.files}
    require(roster_hash(pred['parent'],pred['anchor']) == receipt['roster_sha256'] and pred['anchor'].size == receipt['opportunities'], 'issuance lacks complete opportunities')
    write_json(ticket,dict(stage='score',role=role,release_sha256=lock))
    start = time.monotonic()
    # No path to any V exists in the issuance path. Here the public commit is verified first.
    hpath = _input(release,'hidden',f'hidden-{role}.npz')
    bank = _validate_keys(hpath,role,'hidden')
    require(np.array_equal(bank['parent'],pred['parent']) and np.array_equal(bank['anchor'],pred['anchor']), 'hidden opportunities differ')
    y = _targets(bank['V'],spec)
    v = receipt['fit_variances'][0]
    require(v > 1e-8,'degenerate frozen rule')
    result = {}
    for radius in (4,5):
        stats = {}
        for j,name in enumerate(TARGETS):
            den = v if j < 2 else receipt['fit_variances'][2]
            yt = y[...,j]
            loss = {m:(pred[f'r{radius}_{m}'][...,j]-yt)**2 for m in METHODS}
            row = dict(absolute_mse={m:parent_stats(a.mean(1)) for m,a in loss.items()})
            row['calibration'] = {m:_calibration(pred,yt,f'r{radius}_{m}',j) for m in ('inventory','m0','m1')}
            if den > 1e-8:
                for k,m0,m1 in (('gain','m0','m1'),('vs_inventory','inventory','m1'),('noI_gain','m0_noI','m1_noI')):
                    delta = (loss[m0]-loss[m1])/den
                    row[k] = parent_stats(delta.mean(1))
                    if k == 'gain':
                        support = pred['support']
                        row['supported_gain'] = cluster_ratio((support*delta).sum(1),support.sum(1))
                row['matched_alphas'] = {}
                for k,alpha in enumerate(ALPHAS):
                    e0 = pred[f'r{radius}_m0_grid'][...,k,j]-yt
                    e1 = pred[f'r{radius}_m1_grid'][...,k,j]-yt
                    row['matched_alphas'][str(alpha)] = parent_stats(((e0**2-e1**2)/den).mean(1))
            stats[name] = row
        result[f'r{radius}'] = stats
    raw, residual = (result['r4'][name]['gain'] for name in ('f','residual'))
    null = None
    if role == 'hot':
        null = exact_inventory_null(pred['I'],y[...,0],pred['r4_m1'][...,0]-pred['r4_m0'][...,0],v)
        passed = null['passed']
        disposition = 'conditional equivalence' if passed else ('insufficient exact-I support' if null['eligible_fraction'] < .9 else 'conditional equivalence not demonstrated')
    elif role == 'cold':
        passed = raw['lower'] > .01 and residual['lower'] > 0
        disposition = ('material incremental and distinct' if passed else
                       'below material margin' if raw['upper'] <= .01 else
                       'precision does not establish material gain' if raw['lower'] <= .01 else 'conventional central residual gate failed')
    else:
        passed = raw['lower'] > 0 and raw['point'] >= .005
        disposition = 'positive zero-shot transfer' if passed else 'frozen transfer sign or size failed'
    _calibration_plot(role,result,figure)
    write_json(report,dict(role=role, passed=bool(passed), disposition=disposition, release_sha256=lock,
                           prediction_sha256=sha(ppath), hidden_sha256=sha(hpath), roster_sha256=receipt['roster_sha256'],
                           figure_sha256=sha(figure), parents=len(y), opportunities=receipt['opportunities'],
                           support=receipt['support'], fit_variances=receipt['fit_variances'], scores=result,
                           null=null, matched_ordering=_matched_result(y,pred), wall_sec=time.monotonic()-start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('control','freeze','issue','score'))
    parser.add_argument('--role',choices=ROLES)
    parser.add_argument('--lock',help='SHA256 of committed public forecast-release-{stage[-role]}.json')
    args = parser.parse_args()
    if args.stage == 'control':
        require(args.role is None and args.lock is None,'numerical control uses no native release')
        return control()
    require(bool(args.role) == (args.stage != 'freeze') and args.lock,'explicit role/release required')
    stage = args.stage + ('-'+args.role if args.role else '')
    release = _release(stage,args.lock)
    if args.stage == 'freeze':
        freeze(release,args.lock)
    elif args.stage == 'issue':
        issue(release,args.role,args.lock)
    else:
        score(release,args.role,args.lock)


if __name__ == '__main__':
    main()
