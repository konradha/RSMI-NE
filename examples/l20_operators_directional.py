"""Fit/freeze or score a four-state directional predictive bottleneck on independent L20 parents.

From the repository root (set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1):
  python examples/l20_operators_directional.py fit --input /path/to/cold-replicated.npz
  python examples/l20_operators_directional.py score --input /path/to/confirm-09.npz --out /path/to/scores.csv [--motifs /path/to/motifs.npz]

The sampled outcome is a Bernoulli species-change event: uniform parent, uniform free bond,
uniform nonempty face D, uniform initially occupied site on that face, uniform child. The
critic q(Y|H,D) bounds I(Y;H|D) from below by H(Y|D) + E log q(Y|H,D), with entropies
in nats. This is predictive, not equilibrium RSMI. Aggregated sibling counts suffice for
the expected proper scores. They are not independent observations for inference.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.special import expit, logit, xlogy
from scipy.stats import t as student
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from l20_operators_directional_geometry import CORE, FACES, extract, raw_tests, simple_code

OUT = Path('results/l20-operators/directional')
SEED = 8247991
MODELS = ('direction', 'simple', 'random', 'learned', 'barrier', 'barrier+simple', 'barrier+random', 'barrier+learned')
LAGS = (4096, 16384)
PRIOR = 12.  # pseudo-face events, not child x site multiplicity
TREE = dict(max_iter=70, max_leaf_nodes=9, min_samples_leaf=80, learning_rate=.07, l2_regularization=2.,
            max_bins=63, early_stopping=False, random_state=SEED)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def labels(g, changed, reps):
    obs = changed.ravel()[g['site']].astype(float)/reps
    return face_average(g, obs), obs


def face_average(g, values):
    n = g['n_occupied']
    sums = np.bincount(g['bond']*6 + g['face'], weights=values, minlength=n.size).reshape(n.shape)
    return np.divide(sums, n, out=np.zeros_like(sums, dtype=float), where=n > 0)


def weights(g):
    valid = g['n_occupied'] > 0
    return valid / (valid.sum(1, keepdims=True) * len(valid))


def split_tree(x, y, weights_face, random=False):
    """Two levels, a scalar raw-species predicate at each node; four hard leaves.

    Greedy splits maximize the resulting direction-specific variational log score. The
    untrained tree draws uniformly from feasible features and then feasible cuts, using
    identical depth, candidate bank and minimum leaf mass but no future labels.
    """
    rng = np.random.default_rng(SEED+1)
    w = weights_face
    m = w.sum(1)
    # Each mask is small on the archived fitting set; reuse across all three nodes.
    masks, predicates = [], []
    for j in range(x.shape[1]):
        for v in np.unique(x[:, j])[:-1]:
            masks.append(x[:, j] > v)
            predicates.append([j, int(v)])
    a = np.array(masks, dtype=float)
    pred = np.array(predicates)
    tree = []
    parents = [np.arange(len(x))]
    for depth in range(2):
        children = []
        for ids in parents:
            b = a[:, ids]
            mass = b @ m[ids]
            eligible = (mass > .09*m[ids].sum()) & (mass < .91*m[ids].sum())
            if not eligible.any():
                raise ValueError('no feasible hard split')
            if random:
                j = rng.choice(np.unique(pred[eligible, 0]))
                choice = rng.choice(np.flatnonzero(eligible & (pred[:, 0] == j)))
            else:
                ny, nw = w[ids]*y[ids], w[ids]
                y1, w1 = b @ ny, b @ nw
                y0, w0 = ny.sum(0)-y1, nw.sum(0)-w1
                def score(pos, total):
                    rate = np.clip(pos/np.maximum(total, 1e-12), 1e-9, 1-1e-9)
                    return (xlogy(pos, rate) + xlogy(total-pos, 1-rate)).sum(1)
                choice = int(np.argmax(np.where(eligible, score(y0, w0) + score(y1, w1), -np.inf)))
            tree.append(predicates[choice])
            gt = b[choice].astype(bool)
            children.extend((ids[~gt], ids[gt]))
        parents = children
    return tree


def apply_tree(x, tree):
    def test(node):
        j, threshold = tree[node]
        return x[:, j] > threshold
    first = test(0)
    return (2*first + np.where(first, test(2), test(1))).astype(np.int8)


def fit_ensemble(x, y, w):
    """Fractional binomial log loss with a standard histogram-boosted logistic critic."""
    pos = y > 0
    neg = y < 1
    h = HistGradientBoostingClassifier(**TREE)
    h.fit(np.concatenate((x[neg], x[pos])), np.repeat([0, 1], [neg.sum(), pos.sum()]),
          sample_weight=np.r_[w[neg]*(1-y[neg]), w[pos]*y[pos]])
    # Portable numeric trees rather than a version-fragile executable pickle. No
    # missing nor categorical features here. sklearn leaf values include shrinkage.
    ntrees = len(h._predictors)
    nodes = np.zeros((ntrees, 2*TREE['max_leaf_nodes']-1, 5))
    for t, (tree,) in enumerate(h._predictors):
        s = tree.nodes
        nodes[t, :len(s)] = np.column_stack((np.where(s['is_leaf'], -1, s['feature_idx'].astype(int)),
                                            s['num_threshold'], s['left'], s['right'], s['value']))
    bias = h._baseline_prediction.item()
    if not np.allclose(ensemble((nodes, bias), x[:1024]), h.predict_proba(x[:1024])[:, 1], rtol=1e-12, atol=1e-12):
        raise ValueError('numeric tree export disagrees with fitted model')
    return nodes, bias


def ensemble(model, x):
    nodes, bias = model
    out = np.full(len(x), float(bias))
    rows = np.arange(len(x))
    for t in nodes:
        ix = np.zeros(len(x), dtype=int)
        while True:
            f = t[ix, 0].astype(int)
            take = f >= 0
            if not take.any():
                break
            i = ix[take]
            right = x[rows[take], f[take]] > t[i, 1]
            ix[take] = t[i, 2+right.astype(int)]
        out += t[ix, 4]
    return expit(out)


def clipped_logit(p):
    return logit(np.clip(p, 1e-7, 1-1e-7))


def offsets(h, y, w, base):
    # MAP/ridge-type shrinkage via PRIOR pseudo-events per cell at its base mean;
    # same correction family and same prior for barrier with and without a code.
    delta = np.zeros((4, 6))
    for k in np.unique(h):
        take = h == k
        a, b, c = y[take], w[take], base[take]
        wc = b.sum(0)
        ref = np.divide((b*c).sum(0), wc, out=np.full(6, .5), where=wc>0)
        target, raw, prior = (b*a).sum(0) + PRIOR*ref, clipped_logit(c), clipped_logit(ref)
        lo, hi = np.full(6, -15.), np.full(6, 15.)
        for _ in range(42):
            mid = (lo+hi)*.5
            high = (b*expit(raw+mid)).sum(0) + PRIOR*expit(prior+mid) > target
            lo, hi = np.where(high, lo, mid), np.where(high, mid, hi)
        delta[k] = (lo+hi)*.5
    return delta


def codes(raw, model):
    x, _ = raw_tests(raw)
    return dict(simple=simple_code(raw), random=apply_tree(x, model['random']), learned=apply_tree(x, model['learned']))


def fit_model(geom, counts, replicas):
    raw = np.concatenate([g['raw'] for g in geom])
    x = np.concatenate([g['x'] for g in geom])
    w = np.concatenate([weights(g) for g in geom])
    w *= (w > 0).sum()/w.sum()
    sw = np.concatenate([weights(g)[g['bond'], g['face']]/g['n_occupied'][g['bond'], g['face']] for g in geom])
    sw *= len(sw)/sw.sum()
    z, names = raw_tests(raw)
    model = dict(test_names=names)
    artifacts = {}
    for j, lag in enumerate(LAGS):
        yy = [labels(g, a, replicas) for g, a in zip(geom, counts[j])]
        y = np.concatenate([v[0] for v in yy])
        if j == 0:
            model.update(learned=split_tree(z, y, w), random=split_tree(z, None, w, random=True))
            hs = codes(raw, model)
            hs['direction'] = np.zeros(len(raw), np.int8)
        trees = fit_ensemble(x, np.concatenate([v[1] for v in yy]), sw)
        artifacts[f'trees{lag}'], artifacts[f'bias{lag}'] = trees
        pooled = np.clip((w*y).sum(0)/w.sum(0), 1e-7, 1-1e-7)
        model[str(lag)] = {'direction': pooled.tolist()}
        phys = np.concatenate([face_average(g, ensemble(trees, g['x'])) for g in geom])
        for name, h in hs.items():
            if name != 'direction':
                model[str(lag)][name] = offsets(h, y, w, np.broadcast_to(pooled, y.shape)).tolist()
            key = 'barrier' if name == 'direction' else 'barrier+'+name
            model[str(lag)][key] = offsets(h, y, w, phys).tolist()
    return model, artifacts


def prediction(g, hs, lag, model, trees):
    table = model[str(lag)]
    ray = np.broadcast_to(np.array(table['direction']), g['n_occupied'].shape)
    phys = face_average(g, ensemble((trees[f'trees{lag}'], trees[f'bias{lag}']), g['x']))
    pred = []
    for name in MODELS:
        if name == 'direction':
            p = ray
        else:
            h = hs[name.split('+')[-1]] if name != 'barrier' else np.zeros(len(ray), dtype=int)
            p = expit(clipped_logit(phys if name.startswith('barrier') else ray) + np.array(table[name])[h])
        pred.append(p)
    return np.array(pred)


def metrics(y, p, w):
    pc = np.clip(p, 1e-12, 1-1e-12)
    u = w/w.sum(1, keepdims=True)
    residual = (y-p) - (u*(y-p)).sum(1, keepdims=True)
    # Expected Brier over a random binary event, not the error on a noisy map.
    return dict(observed=float((w*y).sum()), predicted=float((w*p).sum()),
                logloss=float(-(w*(xlogy(y, pc) + xlogy(1-y, 1-pc))).sum()),
                brier=float((w*(y*(1-p)**2 + (1-y)*p**2)).sum()), direction_mse=float((w*residual**2).sum()))


def score_rows(origin, ids, counts, lags, fits, beta, reps, model, trees, split, motif_path=None, cached=None):
    pieces = {k: [] for k in ('parent_id', 'anchor', 'axis', 'sign', 'h', 'target_index', 'n_occupied', 'observed', 'predicted')}
    rows = []
    for i, (s, pid) in enumerate(zip(origin, ids)):
        g = extract(s) if cached is None else cached[i]
        hs = codes(g['raw'], model)
        n = len(hs['learned'])
        if motif_path:
            for k in ('anchor', 'axis', 'sign', 'target_index', 'n_occupied'):
                pieces[k].append(g[k])
            pieces['parent_id'].append(np.full(n, pid, dtype=np.uint64))
            pieces['h'].append(hs['learned'])
        obs, ps = [], []
        for j, (lag, fit_lag) in enumerate(zip(lags, fits)):
            y, _ = labels(g, counts[j, i], reps)
            pp = prediction(g, hs, fit_lag, model, trees)
            for k, (name, p) in enumerate(zip(MODELS, pp)):
                row = dict(parent_id=int(pid), beta=beta, lag=int(lag), fit_lag=int(fit_lag), replicas=reps, split=split[i],
                           model=name, n_bonds=n, n_faces=int((g['n_occupied']>0).sum()), n_sites=len(g['site']))
                row.update(metrics(y, p, weights(g)) if n else dict.fromkeys(('observed','predicted','logloss','brier','direction_mse'), float('nan')))
                rows.append(row)
            if motif_path:
                obs.append(np.where(g['n_occupied'] > 0, y, np.nan).astype(np.float32))
                ps.append(np.where(g['n_occupied'] > 0, pp, np.nan).astype(np.float32))
        if motif_path:
            pieces['observed'].append(np.stack(obs))
            pieces['predicted'].append(np.stack(ps))
    if motif_path:
        arr = {k: np.concatenate(parts, axis=1 if k=='observed' else 2 if k=='predicted' else 0) for k, parts in pieces.items()}
        motif_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(motif_path, **arr, lags=lags, models=MODELS, fit_lags=fits)
    return rows


def summarize(rows):
    summary = []
    comparisons = [('direction', 'simple'), ('direction', 'random'), ('direction', 'learned'), ('simple', 'learned'),
                   ('random', 'learned'), ('barrier', 'barrier+learned'), ('barrier+simple', 'barrier+learned'),
                   ('barrier+random', 'barrier+learned')]
    for split in ('fit', 'validation'):
        for lag in LAGS:
            for base, mod in comparisons:
                def choose(name):
                    return {r['parent_id']: r for r in rows if r['split']==split and r['lag']==lag and r['model']==name and r['n_bonds']}
                a, b = choose(base), choose(mod)
                for metric in ('logloss', 'brier', 'direction_mse'):
                    dif = np.array([v[metric]-b[i][metric] for i,v in a.items()]); n = len(dif)
                    mean = dif.mean(); half = student.ppf(.975,n-1)*dif.std(ddof=1)/np.sqrt(n)
                    summary.append(dict(split=split, lag=lag, base=base, model=mod, metric=metric, n_parents=n,
                                        gain=mean, lo=mean-half, hi=mean+half))
    return summary


def load_input(path):
    with np.load(path, allow_pickle=False) as f:
        ids, origin, lags, changed = (f[k] for k in ('parent_id', 'origin', 'lags', 'changed'))
        beta, reps = float(f['beta']), int(f['replicas'])
    if (ids.dtype != np.uint64 or ids.ndim != 1 or len(np.unique(ids))!=len(ids) or origin.dtype!=np.uint8 or
        origin.shape!=(len(ids),20,20,20) or origin.max()>2 or lags.dtype!=np.int64 or lags.ndim!=1 or len(set(lags))!=len(lags)
        or changed.dtype!=np.uint8 or changed.shape!=(len(lags), *origin.shape) or not 0<reps<=255 or changed.max()>reps):
        raise ValueError('malformed trajectory input')
    return ids, origin, lags, changed, beta, reps


def fit(args):
    out = args.model; out.mkdir(parents=True, exist_ok=True)
    if (out/'freeze.json').exists():
        raise FileExistsError('refusing to overwrite freeze; use --model with a new directory to reproduce')
    ids, origin, lags, changed, beta, reps = load_input(args.input)
    if args.input.name != 'cold-replicated.npz' or (len(ids),beta,reps)!=(64,3.05,16):
        raise ValueError('only archived cold-replicated development may be used to fit')
    lookup = [list(lags).index(lag) for lag in LAGS]
    perm = np.random.default_rng(SEED).permutation(len(ids)); tr, val = perm[:48], perm[48:]
    geom = [extract(s) for s in origin]
    selected = changed[lookup]
    model, trees = fit_model([geom[i] for i in tr], selected[:, tr], reps)
    dump(out/'channel.json', model); np.savez_compressed(out/'barrier.npz', **trees)
    split = np.where(np.isin(np.arange(len(ids)), tr), 'fit', 'validation')
    rows = score_rows(origin, ids, selected, LAGS, LAGS, beta, reps, model, trees, split, args.motifs, cached=geom)
    csv_write(out/'development.csv', rows); csv_write(out/'development-gains.csv', summarize(rows))
    leaf_counts = {k: np.bincount(np.concatenate([codes(geom[i]['raw'], model)['learned'] for i in ii]), minlength=4).tolist()
                   for k,ii in [('fit',tr),('validation',val)]}
    here = Path(__file__)
    dump(out/'freeze.json', dict(schema=1, input_sha256=sha(args.input), beta=beta, replicas=reps,
          fit_parent_id=ids[tr].tolist(), validation_parent_id=ids[val].tolist(), seed=SEED, fit_lags=LAGS,
          confirm_lag_mapping={'cold':{'4096':4096,'16384':16384}, 'transfer':{'116':4096,'462':16384,'4096':4096}},
          models=MODELS, learned_leaf_counts=leaf_counts, core=CORE.tolist(), faces=[f.tolist() for f in FACES],
          min_core_target_L1=2, barrier_critic=TREE, pseudo_events_per_code_direction=PRIOR,
          entropy_objective='parent/bond/nonempty direction/occupied site/child uniform; greedy depth-two primary-lag raw tree; six Bernoulli logits per symbol; no validation selection or refit',
          direction_mse='parent mean within-bond direction-demeaned squared error on six region risks; paired differences cancel the map noise term',
          versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
          source_sha256={p.name:sha(p) for p in (here, here.with_name('l20_operators_directional_geometry.py'),
                                                here.with_name('l20_propensity_predict.py'),here.with_name('l20_propensity_proximity.py'))},
          files={k:sha(out/k) for k in ('channel.json','barrier.npz','development.csv','development-gains.csv')}))
    print(f'Frozen at {out}/freeze.json', flush=True)


def score(args):
    out = args.model; meta = json.loads((out/'freeze.json').read_text())
    for k in ('channel.json','barrier.npz'):
        if sha(out/k)!=meta['files'][k]:
            raise ValueError(f'changed frozen artifact {k}')
    here = Path(__file__)
    if any(sha(here.with_name(k)) != v for k,v in meta['source_sha256'].items()):
        raise ValueError('scoring source no longer matches model freeze')
    model = json.loads((out/'channel.json').read_text())
    with np.load(out/'barrier.npz', allow_pickle=False) as f:
        trees = dict(f)
    ids, origin, lags, changed, beta, reps = load_input(args.input)
    if np.isclose(beta, 3.05):
        group = 'cold'
        if set(ids.tolist()) & set(meta['fit_parent_id']+meta['validation_parent_id']):
            raise ValueError('confirmation overlaps archived parents')
    elif np.isclose(beta, 3.05*8/9):
        group = 'transfer'
    else:
        raise ValueError(f'unknown confirmation beta {beta}')
    fit_lags = [meta['confirm_lag_mapping'][group][str(t)] for t in lags]
    rows = score_rows(origin, ids, changed, lags, fit_lags, beta, reps, model, trees, ['confirmation']*len(ids), args.motifs)
    csv_write(args.out, rows)
    dump(args.out.with_suffix('.json'), dict(input_sha256=sha(args.input), freeze_sha256=sha(out/'freeze.json'),
         csv_sha256=sha(args.out), motifs_sha256=sha(args.motifs) if args.motifs else None, beta=beta, lags=lags.tolist(),
         fit_lags=fit_lags, replicas=reps, n_parents=len(ids)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    for name in ('fit','score'):
        p = sp.add_parser(name); p.add_argument('--input', type=Path, required=True)
        p.add_argument('--model', type=Path, default=OUT); p.add_argument('--motifs', type=Path)
        if name=='score': p.add_argument('--out', type=Path, required=True)
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        (fit if args.cmd=='fit' else score)(args)

if __name__=='__main__':
    main()
