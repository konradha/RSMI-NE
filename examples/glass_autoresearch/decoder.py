"""Archived L20 categorical Barber–Agakov experiment. Search never scores the test split.

One search cell per directory; increase --steps there to deepen. Confirmation reads a
sealed frozen.pt and never constructs an optimizer. Load only trusted local artifacts.
"""
import argparse
import copy
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l20_pairs import L, SPLIT, partition

K, J, S, GROUP = 4, 54, 3, 64
LEARNED = ('additive', 'full_soft', 'full_hard', 'shuffled', 'plant')
ARMS = LEARNED + ('isotropic', 'corner')
COMPARATORS = ('additive', 'isotropic', 'corner')
DIAGS = ((0, 26), (2, 24), (6, 20), (8, 18))
SCHEMA = 'l20-categorical-decoder-v1'
CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFAULT_END = 1795556725


class Deadline(Exception):
    pass


def time_left(end):
    if time.time() >= end:
        raise Deadline


def atomic(path, obj):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    if path.suffix == '.pt':
        torch.save(obj, tmp)
    else:
        tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def file_sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bank(root, index, folds, device):
    path = root / f'beta-{index:02d}.npz'
    side = json.loads(path.with_suffix('.json').read_bytes())
    sha = file_sha(path)
    if sha != side['endpoint_sha256']:
        raise ValueError(f'source checksum mismatch: {path}')
    groups, _ = partition(index)
    with np.load(path, allow_pickle=False) as f:
        beta = float(f['beta'])
        if str(f['axes']) != 'chain,z,y,x' or beta != float(side['beta']):
            raise ValueError('wrong archive axes or temperature')
        # NPZ compresses the entire species member; select permitted parent IDs
        # before inspecting any species or transferring observations to device.
        states = f['species']
        if states.shape != (sum(SPLIT), L, L, L) or states.dtype != np.uint8:
            raise ValueError('wrong endpoint array')
        permitted = [states[groups[g]].reshape(-1, L**3) for g in folds]
        del states
        keys = np.stack((f['chain_seed'], f['chain_id']), axis=1)
        if len(np.unique(keys, axis=0)) != sum(SPLIT):
            raise ValueError('duplicate parent identities')
    n = np.bincount(permitted[0][0], minlength=3)
    if len(n) != 3 or min(n) == 0:
        raise ValueError('expected three species')
    for a in permitted:
        for chunk in np.array_split(a, max(1, len(a)//128)):
            if any(np.any((chunk == s).sum(1) != n[s]) for s in range(S)):
                raise ValueError('archive composition is not fixed')
    meta = dict(index=index, beta=beta, source_sha256=sha, species_counts=n.tolist(),
                entropy_nats=float(-np.sum(n/L**3 * np.log(n/L**3))),
                split_parent_sha256={str(g): hashlib.sha256(keys[groups[g]].astype('<u8').tobytes()).hexdigest() for g in folds})
    return [torch.from_numpy(a).to(device) for a in permitted], meta


def geometry(buffer, device):
    cube = np.indices((3, 3, 3)).reshape(3, -1).T - 1
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            for pair in np.ndindex(3, 3):
                v = np.empty(3, dtype=int)
                v[axis] = sign * (buffer + 2)
                v[np.arange(3) != axis] = np.asarray(pair) - 1
                faces.append(v)
    pos = np.indices((L,)*3).reshape(3, -1).T
    # int32 offsets on device; only the selected addresses are cast for indexing.
    off = np.vstack((cube, faces))
    addr = np.ravel_multi_index(((pos[:, None] + off) % L).transpose(2, 0, 1), (L,)*3)
    return torch.from_numpy(addr.astype(np.int32)).to(device)


def observe(rows, geom, parents, anchors):
    vals = rows[parents[:, None], geom[anchors].long()]
    return vals[:, :27], vals[:, 27:]


def species(x):
    return torch.stack([x == k for k in range(S)], dim=-1).float()


def entropy(prob):
    return -(prob * prob.clamp_min(1e-20).log()).sum(-1)


def controls(v, y):
    # Each chunk contains 64 distinct parents. The cyclic permutation has no
    # fixed point. For the planted diagnostic only, rank-match the A bits for
    # each target position within the first 1/8 of this shuffled chunk. This is
    # a permutation of existing labels at each j, NOT a generated physical shell.
    # A half-chunk shift also keeps all eight planted recipients disjoint
    # from all eight donors after the additional rank permutation.
    g = y.reshape(-1, GROUP, J).roll(GROUP//2, dims=1)
    m = GROUP // 8
    sub = g[:, :m]
    order = (v.reshape(-1, GROUP, 27)[:, :m, 0] == 1).int().argsort(dim=1, stable=True)
    labels = (sub == 1).int().argsort(dim=1, stable=True)
    plant = g.clone()
    plant[:, :m].scatter_(1, order[:, :, None].expand_as(sub), sub.gather(1, labels))
    return g.reshape_as(y), plant.reshape_as(y)


class Encoder(nn.Module):
    def __init__(self, nonlinear):
        super().__init__()
        self.affine = nn.Linear(81, K)
        self.residual = (nn.Sequential(nn.Linear(81, 64), nn.SiLU(), nn.Linear(64, K))
                         if nonlinear else None)
        # Start with a usable, spatially distributed four-way code, identical
        # across learned arms. All 27 x 3 coefficients remain trainable.
        with torch.no_grad():
            self.affine.weight.mul_(3)
            self.affine.bias.zero_()
            self.affine.weight[:, 1] += torch.tensor([-1., -1., 1., 1.])
            self.affine.weight[:, 26*3+1] += torch.tensor([-1., 1., -1., 1.])
            if nonlinear:
                nn.init.zeros_(self.residual[-1].weight)
                nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features, hard):
        logits = self.affine(features)
        if self.residual is not None:
            logits = logits + self.residual(features)
        soft = logits.softmax(-1)
        if not hard:
            return soft
        # Straight-through derivative only during training. Forward and inference
        # have exactly a single hard category, never continuous logits.
        disc = F.one_hot(soft.argmax(-1), K).float()
        return disc + (soft - soft.detach()) if torch.is_grad_enabled() else disc


class Models(nn.Module):
    def __init__(self, counts):
        super().__init__()
        self.register_buffer('p', torch.tensor(counts, dtype=torch.float32)/L**3)
        full = Encoder(True)
        self.enc = nn.ModuleDict({k: copy.deepcopy(full) for k in LEARNED if k != 'additive'})
        self.enc['additive'] = Encoder(False)
        self.enc['additive'].affine.load_state_dict(full.affine.state_dict())
        self.register_buffer('corner_code', torch.arange(9) % K)
        self.register_buffer('corner_sites', torch.tensor(DIAGS[0]))
        self.register_buffer('counts', torch.zeros(len(ARMS), K, J, S))
        self.register_buffer('corners', torch.zeros(len(DIAGS), 9, J, S))

    def features(self, v):
        return (species(v) - self.p).flatten(1)

    def code(self, arm, v, x):
        if arm in LEARNED:
            return self.enc[arm](x, arm == 'full_hard')
        if arm == 'corner':
            pair = v.index_select(1, self.corner_sites).long()
            labels = self.corner_code[3*pair[:, 0]+pair[:, 1]]
        else:
            # Rotation- and reflection-invariant physical composition quantizer:
            # above/below mean number of vacancies and of A particles in the cube.
            labels = 2*((v == 0).sum(1) > 27*self.p[0]).long() + ((v == 1).sum(1) > 27*self.p[1]).long()
        return F.one_hot(labels, K).float()

    def logratios(self):
        # The lookup is q_j(s|h). Shrink to exact p with 64 pseudo-observations
        # per code and pixel. log1p preserves precision near the marginal.
        n = self.counts + 64*self.p
        expected = n.sum(-1, keepdim=True)*self.p
        return torch.log1p((n - expected)/expected)


@functools.lru_cache(maxsize=1)
def corner_masks():
    # Four nonempty cells suffice: refining a cell cannot decrease fitted MI.
    codes = []
    def extend(xs, high):
        if len(xs) == 9:
            if high == K-1:
                codes.append(xs)
        else:
            for h in range(min(high+1, K-1)+1):
                extend(xs+(h,), max(high, h))
    extend((0,), 0)
    codes = np.array(codes, dtype=np.int64)
    masks = np.stack([((codes == h)*(1 << np.arange(9))).sum(1) for h in range(K)], 1)
    return codes, masks


@torch.no_grad()
def solve_corner(model):
    # Exact train objective over all 4 diagonals and 7,770 unlabeled partitions
    # per diagonal. Sum conditional log likelihood over the registered 54 sites,
    # not MI with a joint pair of exterior pixels as in quotient.py.
    raw = model.corners.cpu().numpy().astype(np.float64)
    codes, masks = corner_masks()
    subsets = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(float)
    best = -np.inf
    for d, c in enumerate(raw):
        n = (subsets @ c.reshape(9, -1)).reshape(512, J, S)
        logq = np.log(np.maximum(n, 1e-30)) - np.log(np.maximum(n.sum(-1, keepdims=True), 1e-30))
        scores = (n*logq).sum((1, 2))[masks].sum(1)
        i = scores.argmax()
        if scores[i] > best:
            best, winner, diag = scores[i], codes[i], d
    model.corner_sites.copy_(torch.as_tensor(DIAGS[diag]))
    model.corner_code.copy_(torch.from_numpy(winner))
    pooled = np.stack([raw[diag, winner == h].sum(0) for h in range(K)])
    model.counts[-1].copy_(torch.from_numpy(pooled))


def training_batch(rows, geom, batch, generator):
    p = torch.randperm(len(rows), device=rows.device, generator=generator)[:batch]
    a = torch.randint(L**3, (batch,), device=rows.device, generator=generator)
    return observe(rows, geom, p, a)


def fit_batch(model, v, y, opt, step, calibrating=False):
    x = model.features(v)
    shuffled, plant = controls(v, y)
    targets = [species(a).flatten(1) for a in (y, shuffled, plant)]
    ratios = model.logratios() if not calibrating else None
    tables, loss = [], 0.
    for i, arm in enumerate(ARMS):
        if arm == 'corner' and not calibrating:
            tables.append(torch.zeros_like(model.counts[i]))
            continue
        weights = model.code(arm, v, x)
        tags = targets[1 if arm == 'shuffled' else 2 if arm == 'plant' else 0]
        joint = (weights.T @ tags).view(K, J, S)
        tables.append(joint.detach())
        if not calibrating and arm in LEARNED:
            gain = (joint * ratios[i]).sum() / (len(y)*J)
            # Train-only warm start discourages both dead categories and a
            # perfectly balanced yet input-independent stochastic channel.
            bonus = 0.
            if step < 128:
                hm = entropy(weights.mean(0))
                useful = hm if arm == 'full_hard' else hm - entropy(weights).mean()
                bonus = .0005 * (1-step/128) * (hm - F.relu(.25 - useful))
            loss = loss - (gain + bonus)
    if not calibrating:
        opt.zero_grad(set_to_none=True)
        (1000*loss).backward()
        opt.step()
    with torch.no_grad():
        # Identical online conditional M-step across all arms, then a separate
        # identical, frozen-encoder calibration pass with no forgetting.
        decay = 1. if calibrating else 1.-1./1024
        model.counts.mul_(decay).add_(torch.stack(tables))
        if not calibrating:
            pairs = torch.stack([3*v[:, u] + v[:, w] for u, w in DIAGS], 1).long()
            paircodes = F.one_hot(pairs, 9).flatten(1).float()
            model.corners.mul_(decay).add_((paircodes.T @ targets[0]).reshape_as(model.corners))


def generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def interval(a):
    a = np.asarray(a, dtype=float)
    mean = a.mean()
    se = a.std(ddof=1)/math.sqrt(len(a))
    d = t.ppf(.975, len(a)-1)*se
    return dict(mean=float(mean), se=float(se), lo=float(mean-d), hi=float(mean+d))


@torch.no_grad()
def measure(model, rows, geom, index, split, reps, arms, end, diagnostics):
    # Same anchors for EVERY arm/seed/buffer. Rows are in immutable partition
    # order. A shuffle never exits an inference block. Full-translation option
    # uses all anchors once per parent (not 8,000 independent replicates).
    rng = generator(rows.device, 867154 + 32*index + split)
    count = len(rows)//GROUP
    ratios = model.logratios()[[ARMS.index(k) for k in arms]]
    metrics = ('raw', 'shuffled', 'planted')
    result = np.zeros((count, len(arms), len(metrics)), np.float64)
    hist = torch.zeros((len(arms), K, 27, S), device=rows.device, dtype=torch.float64)
    hcond = torch.zeros(len(arms), device=rows.device, dtype=torch.float64)
    for g in range(count):
        group = rows[g*GROUP:(g+1)*GROUP]
        acc = torch.zeros((len(arms), len(metrics)), device=rows.device, dtype=torch.float64)
        for lo in range(0, reps, 32):
            time_left(end)
            nr = min(32, reps-lo)
            parents = torch.arange(GROUP, device=rows.device).repeat(nr)
            if reps == L**3:
                anchors = torch.arange(lo, lo+nr, device=rows.device).repeat_interleave(GROUP)
            else:
                anchors = torch.randint(L**3, (nr*GROUP,), device=rows.device, generator=rng)
            v, y = observe(group, geom, parents, anchors)
            shuf, plant = controls(v, y)
            x = model.features(v)
            weights = torch.cat([model.code(k, v, x) for k in arms], 1)
            # Use exact log(q/p) scores instead of H(p)+sample mean log(q):
            # known translation-uniform p supplies a zero-mean control variate.
            # Accumulate already-centered scores and blocks in double precision.
            for j, labels in enumerate((y, shuf, plant)):
                tab = (weights.T @ species(labels).flatten(1)).reshape(len(arms), K, J, S)
                acc[:, j] += (tab * ratios).sum((1, 2, 3), dtype=torch.float64)
            if diagnostics:
                hist += (weights.T @ species(v).flatten(1)).reshape_as(hist)
                hcond += entropy(weights.reshape(len(v), len(arms), K)).sum(0, dtype=torch.float64)
        result[g] = (acc / (GROUP*reps*J)).cpu().numpy()
    blocks = {arm: {metric: result[:, i, j].tolist() for j, metric in enumerate(metrics)} for i, arm in enumerate(arms)}
    summaries = {}
    for arm, block in blocks.items():
        summaries[arm] = {metric: interval(a) for metric, a in block.items()}
        task = 'planted' if arm == 'plant' else 'raw'
        summaries[arm]['minus_shuffle'] = interval(np.subtract(block[task], block['shuffled']))
        # Approximate one-sided alpha=.05, 80%-power minimum detectable
        # change at these observed cluster variances, not power for true MI.
        mde = float(t.ppf(.95, count-1) + .8416212336)
        summaries[arm]['mde80_nats'] = {k: mde*summaries[arm][k]['se'] for k in (task, 'minus_shuffle')}
        summaries[arm]['paired_vs'] = {k: interval(np.subtract(block['raw'], blocks[k]['raw'])) for k in COMPARATORS if k in blocks and arm != k}
    interpret = {}
    if diagnostics:
        for i, arm in enumerate(arms):
            joint = hist[i].cpu().numpy()
            mass = joint[:, 0, :].sum(1)/(len(rows)*reps)
            h = float(-np.dot(mass, np.log(mass.clip(1e-30))))
            hc = float(hcond[i].item()/(len(rows)*reps))
            means = joint / joint.sum(-1, keepdims=True).clip(1e-30)
            interpret[arm] = dict(p_h=mass.tolist(), effective_states=math.exp(h), h_given_v=hc,
                                  i_h_v=h-hc, collapsed=bool(mass.min() < .01 or h-hc < .01),
                                  p_species_given_code_at_visible_sites=means.tolist())
    return dict(index=index, split=('validation' if split == 1 else 'test'), parents=len(rows), blocks=count,
                anchors_per_parent=reps, block_size=GROUP, summaries=summaries,
                parent_block_scores=blocks, interpretation=interpret)


def config(a, meta):
    return dict(schema=SCHEMA, source=meta['source_sha256'], code_sha256=CODE_SHA, index=meta['index'],
                buffer=a.buffer, batch=a.batch, seed=a.seed, lr=a.lr, device=a.device)


def snapshot(path, signature, model, opt, rng, step, curve):
    atomic(path, dict(config=signature, model=model.state_dict(), optimizer=opt.state_dict(),
                      step=step, rng=rng.get_state(), cpu_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,
                      curve=curve))


def searchable(a):
    index = a.indices[0]
    (train, val), meta = bank(a.data, index, (0, 1), a.device)
    geom = geometry(a.buffer, a.device)
    torch.manual_seed(a.seed)
    model = Models(meta['species_counts']).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = generator(a.device, a.seed+10001)
    signature = config(a, meta)
    ck = a.out/'checkpoint.pt'
    step, curve = 0, []
    if ck.exists():
        obj = torch.load(ck, map_location='cpu', weights_only=True)
        if obj['config'] != signature:
            raise ValueError('incompatible cell/config/source; use another --out')
        model.load_state_dict(obj['model'])
        opt.load_state_dict(obj['optimizer'])
        rng.set_state(obj['rng'])
        torch.set_rng_state(obj['cpu_rng'])
        if obj['cuda_rng'] is not None:
            torch.cuda.set_rng_state(obj['cuda_rng'])
        step, curve = obj['step'], obj['curve']
        if a.steps < step:
            raise ValueError(f'checkpoint already at {step}; --steps is absolute')
    elif any(a.out.iterdir()):
        raise ValueError('nonempty output directory lacks a matching checkpoint')
    try:
        while step < a.steps:
            time_left(a.end_epoch)
            v, y = training_batch(train, geom, a.batch, rng)
            fit_batch(model, v, y, opt, step)
            step += 1
            if step % 16 == 0 and a.device == 'cuda':
                torch.cuda.synchronize()
            if step % a.eval_every == 0 and step < a.steps:
                solve_corner(model)
                result = measure(model, val, geom, index, 1, a.curve_anchors, ARMS, a.end_epoch, False)
                curve.append(dict(step=step, phase='online', estimates=result['summaries']))
                atomic(a.out/'curve.json', curve)
            if step % a.save_every == 0:
                snapshot(ck, signature, model, opt, rng, step, curve)
        # Do not contaminate continuation with calibration or the frozen readout.
        snapshot(ck, signature, model, opt, rng, step, curve)
        sealed = copy.deepcopy(model).eval()
        solve_corner(sealed)
        sealed.counts.zero_()
        cal = a.calibrate_steps or min(step, 256)
        rg = generator(a.device, a.seed+20001)
        with torch.no_grad():
            for _ in range(cal):
                time_left(a.end_epoch)
                v, y = training_batch(train, geom, a.batch, rg)
                fit_batch(sealed, v, y, None, step, True)
        result = measure(sealed, val, geom, index, 1, a.eval_anchors, ARMS, a.end_epoch, True)
        # A collapsed code is a negative diagnostic, not an excluded result.
        winner = max(('full_soft', 'full_hard'), key=lambda k: result['summaries'][k]['raw']['mean'])
        desc = dict(config=signature, step=step, selected=winner, source=meta, calibration_batches=cal,
                    validation_anchors=a.eval_anchors, corner_sites=sealed.corner_sites.tolist(),
                    corner_partition=sealed.corner_code.tolist())
        atomic(a.out/'frozen.pt', dict(design=desc, model=sealed.state_dict()))
        final = dict(step=step, phase='calibrated', estimates=result['summaries'])
        atomic(a.out/'curve.json', curve+[final])
        atomic(a.out/'validation.json', dict(design=desc, frozen_sha256=file_sha(a.out/'frozen.pt'), **result))
        print(json.dumps(dict(mode='search', step=step, selected=winner,
                              raw=result['summaries'][winner]['raw'], out=str(a.out))), flush=True)
    except Deadline:
        snapshot(ck, signature, model, opt, rng, step, curve)
        raise


def confirm(a):
    digest = file_sha(a.frozen)
    packet = torch.load(a.frozen, map_location='cpu', weights_only=True)
    design = packet['design']
    if design['config']['schema'] != SCHEMA or design['config']['code_sha256'] != CODE_SHA:
        raise ValueError('frozen design/code mismatch')
    if a.frozen.resolve().parent == a.out.resolve():
        raise ValueError('write confirmation separately from search artifacts')
    # Record the model digest, index order and resolution BEFORE touching test
    # species; refuse to change any of these on resumption in this directory.
    plan = dict(frozen_sha256=digest, design=design, indices=a.indices, anchors=a.eval_anchors)
    lock = a.out/'confirmation-plan.json'
    if lock.exists() and json.loads(lock.read_bytes()) != plan:
        raise ValueError('confirmation directory bound to another frozen plan')
    if not lock.exists() and any(a.out.iterdir()):
        raise ValueError('nonempty directory without confirmation plan')
    atomic(lock, plan)
    model = Models(design['source']['species_counts']).to(a.device).eval()
    model.load_state_dict(packet['model'])
    geom = geometry(design['config']['buffer'], a.device)
    arms = (design['selected'],) + COMPARATORS + ('shuffled', 'plant')
    reports = {}
    for index in a.indices:
        time_left(a.end_epoch)
        target = a.out/f'test-{index:02d}.json'
        if not target.exists():
            (rows,), meta = bank(a.data, index, (2,), a.device)
            if meta['species_counts'] != design['source']['species_counts']:
                raise ValueError('frozen composition mismatch')
            if index == design['config']['index'] and meta['source_sha256'] != design['config']['source']:
                raise ValueError('search and confirmation source mismatch')
            result = measure(model, rows, geom, index, 2, a.eval_anchors, arms, a.end_epoch, True)
            atomic(target, dict(frozen_sha256=digest, selected=design['selected'], source=meta, **result))
            del rows
        reports[str(index)] = json.loads(target.read_bytes())['summaries']
    atomic(a.out/'confirmation.json', dict(plan=plan, summaries=reports))
    print(json.dumps(dict(mode='confirm', frozen_sha256=digest, out=str(a.out))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('search', 'confirm'), default='search')
    p.add_argument('--data', type=Path, default=Path('~/datasets/lattice-glass/l20-16384-v1').expanduser())
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frozen', type=Path)
    p.add_argument('--indices', nargs='+', type=int)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--seed', type=int, default=68239)
    p.add_argument('--buffer', type=int, choices=(1, 2, 3), default=2)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--save-every', type=int, default=128)
    p.add_argument('--eval-every', type=int, default=256)
    p.add_argument('--eval-anchors', type=int)
    p.add_argument('--curve-anchors', type=int, default=8)
    p.add_argument('--calibrate-steps', type=int, default=0, help='0 selects min(steps,256)')
    p.add_argument('--end-epoch', type=float, default=DEFAULT_END, help='absolute Unix wall-clock deadline')
    a = p.parse_args()
    a.indices = a.indices or ([9] if a.mode == 'search' else [9, 7, 0])
    if a.eval_anchors is None:
        a.eval_anchors = 64 if a.mode == 'search' else 512
    if not math.isfinite(a.end_epoch) or min(a.steps, a.eval_every, a.save_every) < 1 or a.calibrate_steps < 0:
        p.error('positive step counts and a finite absolute deadline required')
    if min(a.curve_anchors, a.eval_anchors) < 1 or max(a.curve_anchors, a.eval_anchors) > L**3:
        p.error('anchor repetitions must be in [1,8000]')
    if not GROUP <= a.batch <= SPLIT[0] or a.batch % GROUP:
        p.error('--batch must be a multiple of 64 up to 10240 for distinct-parent controls')
    if a.mode == 'search' and (a.indices != [9] or a.frozen):
        p.error('search protocol uses only index 9 and cannot load a confirmation artifact')
    if a.mode == 'confirm' and (a.indices != [9, 7, 0] or not a.frozen):
        p.error('confirmation requires one --frozen path, and fixed indices 9 7 0')
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        time_left(a.end_epoch)
        (searchable if a.mode == 'search' else confirm)(a)
    except Deadline:
        print(json.dumps(dict(status='deadline', mode=a.mode, out=str(a.out))), flush=True)
        return 75
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
