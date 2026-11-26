"""Six paired history fits. Only the stock optimiser applies gradients. No TEST I/O."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np

SEEDS = (6101, 6102, 6103)
ARMS = ('true', 'displaced')
CELLS = [(arm, seed) for seed in SEEDS for arm in ARMS]
DEPTHS = (8192, 16384, 32768, 65536)
BATCH = 48
EPOCH = 8192
OUT = Path('checkpoints')


def atomic(path, data):
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def encode(raw, dtype=np.float32):
    out = np.empty((*raw.shape, 2), dtype)
    np.equal(raw, 1, out=out[..., 0]); np.equal(raw, 2, out=out[..., 1])
    return out


def frozen_logits(ck, V):
    return encode(V, np.float64).reshape(len(V), 54) @ ck['encoder'].reshape(54, 2)


def hard_labels(logits):
    return 2 * (logits[:, 0] >= 0).astype('u1') + (logits[:, 1] >= 0).astype('u1')


def relaxed_draw(ck, logits, u):
    from scipy.special import expit
    if np.any(u <= 0) or np.any(u >= 1):
        raise ValueError('RelaxedBernoulli requires uniforms inside (0,1)')
    return expit((logits + np.log(u) - np.log1p(-u)) / float(ck['tau']))


def frozen_scores(ck, E, h):
    def tower(x, start):
        for j in range(start, start + 8, 2):
            x = x @ ck[f'c{j:02}'] + ck[f'c{j+1:02}']
            if j < start + 6:
                np.maximum(x, 0, out=x)
        return x
    env = tower(encode(E, np.float64).reshape(len(E), 772), 0)
    return tower(np.asarray(h, dtype=np.float64), 8) @ env.T


def dev_draws(n, seed):
    """128 origin panels, one origin/parent. Fixed across all depths and both arms."""
    rng = np.random.default_rng([811901, seed, n]); ix = np.arange(n)
    origins = np.array([rng.choice(8000, 128, replace=False) for _ in ix]).T
    for row in origins:
        perm = rng.permutation(n)
        while np.any(perm == ix):
            perm = rng.permutation(n)
        yield row, np.maximum(rng.random((n, 2)), np.finfo(float).tiny), perm


def offsets(radius):
    d = np.indices((2 * radius + 1,) * 3).reshape(3, -1).T - radius
    return d if radius == 1 else d[np.abs(d).max(axis=1) == radius]


def lookup(radius, shift=0):
    org = np.indices((20,) * 3, dtype='i2').reshape(3, -1).T
    d = offsets(radius).astype('i2')
    # Construct one axis at a time; flat toroidal C order (z, y, x).
    pos = (org[:, None, 0] + d[:, 0] + shift) % 20
    for ax in (1, 2):
        pos *= 20; pos += (org[:, None, ax] + d[:, ax] + shift) % 20
    return pos


def layout(ids, seed):
    org = np.empty((EPOCH, len(ids)), 'u2')
    for k, pid in enumerate(ids):
        p = np.random.default_rng([710311, seed, int(pid)]).permutation(8000)
        org[:8000, k] = p; org[8000:, k] = p[:192]
    rng = np.random.default_rng([710312, seed]); order = np.empty_like(org, dtype='u1')
    for row in order:
        row[:] = rng.permutation(len(ids))
    return order, np.take_along_axis(org, order, axis=1)


def train_inputs(fields, ids, seed, arm):
    parents, org = layout(ids, seed)
    vix, eix = lookup(1), lookup(4, 10 if arm == 'displaced' else 0)
    raw = fields.reshape(BATCH, 8000)
    # Hold the training epoch on host. Construct bounded chunks, not an Nx386x3 coordinate cube.
    v = np.empty((EPOCH, BATCH, 27, 2), 'f'); e = np.empty((EPOCH, BATCH, 386, 2), 'f')
    for l in range(0, EPOCH, 32):
        sl = slice(l, l + 32); p = parents[sl, :, None]; o = org[sl]
        for dest, idx in ((v, vix), (e, eix)):
            raw_patch = raw[p, idx[o]]
            np.equal(raw_patch, 1, out=dest[sl, ..., 0]); np.equal(raw_patch, 2, out=dest[sl, ..., 1])
    dg = hashlib.sha256(parents.tobytes() + org.astype('<u2', copy=False).tobytes()).hexdigest()
    return v.reshape(-1, 3, 3, 3, 2, 1), e.reshape(-1, 772), dg


def dev_assess(ck, critic, tf, panels):
    import tensorflow_probability as tfp
    from scipy.special import logsumexp
    n = len(panels[0][0]); ix = np.arange(n); ln2 = np.log(2.)
    bound = tfp.vi.mutual_information.lower_bound_info_nce
    @tf.function
    def measure(x, h, perm):
        s = critic(x, h, training=False)
        return (s, bound(logu=s, joint_sample_mask=tf.eye(n, dtype=tf.bool)),
                bound(logu=s, joint_sample_mask=tf.equal(tf.range(n)[None, :], perm[:, None])))
    results = {}; portable = {}; valid = True
    logits = np.stack([frozen_logits(ck, v) for v, *_ in panels]); labels = hard_labels(logits.reshape(-1, 2))
    for ei, arm in enumerate(ARMS, start=1):
        rows = []
        for k, panel in enumerate(panels):
            E, u, perm = panel[ei], panel[3], panel[4]; h = relaxed_draw(ck, logits[k], u)
            s, real, shuffled = measure(encode(E).reshape(n, 772), h.astype('f'), perm.astype('i4'))
            x = s.numpy().astype('d'); p = frozen_scores(ck, E, h)
            def stable(a):
                z = a - a.max(1, keepdims=True); den = logsumexp(z, axis=1) - math.log(n)
                return np.array([(z[ix, j] - den).mean() for j in (ix, perm)]) / ln2
            row = np.r_[float(real)/ln2, float(shuffled)/ln2, stable(x), stable(p)]
            rows.append(row)
        data = np.array(rows); ok = bool(np.isfinite(data).all() and data.max() <= np.log2(n) + 1e-5
                and np.abs(data[:, :2] - data[:, 2:4]).max() <= .001)
        valid &= ok; portable['dev_' + arm + '_scores'] = data
        results[arm] = dict(valid=ok, means_bits=data.mean(axis=0).tolist() if np.isfinite(data).all() else None,
                            stable_gain_bits=float((data[:, 2] - data[:, 3]).mean()) if ok else None)
    counts = np.bincount(labels, minlength=4); mass = counts[counts > 0] / len(labels)
    portable.update(dev_logits=logits, dev_hard_code=labels.reshape(-1, n))
    return dict(valid=valid, parents=n, panels=len(panels), rows='same fixed panel order for each depth',
                score_columns=['stock_real', 'stock_shuffle', 'stable_real', 'stable_shuffle', 'portable_real', 'portable_shuffle'],
                code_counts=counts.tolist(), entropy_bits=float(-mass @ np.log2(mass)), shell=results), portable


def checkpoint(cg, critic, adam, step, panels, tf, base):
    weights = critic._g.get_weights() + critic._h.get_weights()
    shapes = [(q, 32) if j == 0 else (32, 8) if j == 6 else (32, 32) if j in (2,4) else (8,) if j == 7 else (32,)
              for q in (772, 2) for j in range(8)]
    if [w.shape for w in weights] != shapes:
        raise ValueError('Unexpected stock critic topology')
    w, = cg._Λ.get_weights(); state = dict(encoder=w, tau=float(cg.tau), global_step=int(cg.global_step),
                                            adam_iterations=int(adam.iterations), **{f'c{j:02}': x for j, x in enumerate(weights)})
    if w.shape != (3,3,3,2,2) or state['adam_iterations'] != step or state['global_step'] != step - 1:
        raise ValueError('Unexpected checkpoint shape or stock iteration')
    finite = all(np.isfinite(a).all() for a in state.values()) and all(np.isfinite(v.numpy()).all() for v in adam.variables())
    report = dict(step=step, weights_and_adam_finite=bool(finite), valid=False)
    if finite:
        diag, arrays = dev_assess(state, critic, tf, panels); state.update(arrays); report.update(diag)
    path = OUT / f'{base}-u{step}.npz'; tmp = path.with_suffix('.part')
    with tmp.open('xb') as f: np.savez_compressed(f, **state)
    tmp.replace(path); report['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic(path.with_suffix('.json'), report)
    if not report['valid']: raise FloatingPointError(f'invalid frozen checkpoint {step}: see {path}')


def make_panels(fields, seed):
    n = len(fields); ix = np.arange(n)[:, None]; fields = fields.reshape(n, 8000)
    look = (lookup(1), lookup(4), lookup(4, 10))
    return [tuple(fields[ix, lut[orig]] for lut in look) + (u, perm) for orig, u, perm in dev_draws(n, seed)]


class DepthStop(Exception):
    """Common graceful stop, raised only after the last requested update succeeded."""


def worker(slot):
    cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, {cpus[slot % len(cpus)]})
    import tensorflow as tf
    from rsmine.coarsegrainer import cg_optimisers as core
    import stock_glass_modal as stock
    import tqdm
    arm, seed = CELLS[slot]; base = f'{arm}-s{seed}'; start = time.monotonic(); ticks = 0; evaluations_s = 0.
    gpu, = tf.config.list_physical_devices('GPU'); tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    device = tf.config.experimental.get_device_details(gpu)['device_name']; uid = stock.cuda_uuid()
    if 'H100' not in device: raise ValueError(device)
    tf.keras.utils.set_random_seed(seed); adam = tf.keras.optimizers.Adam(.009)
    with np.load('b0-a500-q80-fit.npz') as z: v, e, stream_sha = train_inputs(z['field'], z['ids'], seed, arm)
    with tf.device('/CPU:0'): V = tf.convert_to_tensor(v); E = tf.convert_to_tensor(e)
    del v, e
    with np.load('b0-a500-q80-dev.npz') as z: panels = make_panels(z['field'], seed)
    refs = {}; mi_min = math.inf; mi_max = -math.inf; train_start = time.monotonic(); first = None
    def capture(k, f):
        def ctor(*args, **kw):
            refs[k] = f(*args, **kw); return refs[k]
        return ctor
    def progress():
        return dict(arm=arm, seed=seed, adam_iterations=int(adam.iterations), callback_steps=ticks,
                    epoch=EPOCH, positives=ticks*BATCH, parents=BATCH, unique_origins_per_parent=min(ticks,8000),
                    stream_sha256=stream_sha, pid=os.getpid(), cpu=sorted(os.sched_getaffinity(0)),
                    device=device, uuid=uid, training_start=train_start, first_cuda_monotonic=first,
                    now_monotonic=time.monotonic(), setup_s=train_start-start, elapsed_s=time.monotonic()-train_start,
                    checkpoint_evaluation_and_wait_s=evaluations_s,
                    train_mi_range_nats=[mi_min,mi_max] if np.isfinite([mi_min,mi_max]).all() else None)
    def observe(bar, n=1):
        nonlocal ticks, mi_min, mi_max, first, evaluations_s
        ticks += n
        # The stock callback is after apply_gradients and the NaN guard. Inspect the actual scalar,
        # not rounded progress text; the pinned call site also owns the actual CUDA hidden draw.
        frame = sys._getframe(1).f_locals; mi = frame['mi']; value = float(mi)
        if not np.isfinite(value) or value > math.log(BATCH) + 1e-5:
            raise FloatingPointError(f'invalid stock training bound {value} at Adam step {int(adam.iterations)}')
        mi_min = min(mi_min, value); mi_max = max(mi_max, value)
        if ticks == 1:
            first = time.monotonic(); placement = [mi.device, frame['h'].device] + [x.device for x in refs['cg'].trainable_variables + refs['critic'].trainable_variables]
            # Stock explicitly places its critic on GPU. Returned scalar/hidden tensors may be copied to host.
            if not all('GPU:0' in s.upper() for s in placement[2:]) or int(adam.iterations) != 1: raise RuntimeError(placement)
            atomic(OUT / f'{base}-first.json', dict(progress(), placement=placement))
        if ticks == 1 or ticks % 1024 == 0:
            if int(adam.iterations) != ticks: raise ValueError('Adam callback mismatch')
            atomic(OUT / f'{base}-progress.json', progress())
        if ticks in DEPTHS:
            t = time.monotonic(); checkpoint(refs['cg'], refs['critic'], adam, ticks, panels, tf, base)
            if ticks == 32768:
                while not (OUT / 'depth-decision.json').exists(): time.sleep(.1)
                decision = json.loads((OUT / 'depth-decision.json').read_text()); evaluations_s += time.monotonic() - t
                if not decision['extend']: raise DepthStop()
            else: evaluations_s += time.monotonic() - t
        return True
    cg = dict(ll=(3,3,3), visible_dim=2, hidden_dim=2, h_embed=True, nonlinearCG=None, init_temperature=.75,
              min_temperature=.1, relaxation_rate=.0001, Nq=None, conv_activation=None, use_probs=False, use_logits=True)
    critic = dict(layers=3, embed_dim=8, hidden_dim=32, activation='relu', use_dropout=True, dropout_rate=.2)
    opt = dict(iterations=8, batch_size=BATCH, shuffle=1, learning_rate=.009)
    try:
        with patch.object(core, 'CoarseGrainer', capture('cg', core.CoarseGrainer)), patch.object(core, 'SeparableCritic', capture('critic', core.SeparableCritic)), patch.object(tqdm.tqdm, 'update', observe), patch.object(tqdm.tqdm, 'set_description', lambda *a, **k: None):
            try: core.train_RSMI_optimiser(cg, critic, opt, dict(N_samples=EPOCH*BATCH, dimension=3), V=V, E=E, optimizer=adam, use_GPU=True)
            except DepthStop: pass
        selected = json.loads((OUT / 'depth-decision.json').read_text())['selected_depth'] if (OUT / 'depth-decision.json').exists() else None
        if ticks != selected or int(adam.iterations) != ticks: raise FloatingPointError('premature stock return or numerical failure')
        atomic(OUT / f'{base}-final.json', dict(progress(), complete=True))
    except BaseException as exc:
        atomic(OUT / f'{base}-failure.json', dict(progress(), complete=False, error=repr(exc), traceback=traceback.format_exc())); raise


def decision(remaining_s, can_extend):
    by_seed = []; minimum_rate = math.inf
    for seed in SEEDS:
        gains = {}
        for arm in ARMS:
            stem = f'{arm}-s{seed}'; p = json.loads((OUT / f'{stem}-progress.json').read_text())
            minimum_rate = min(minimum_rate, 32768 / max(1, p['elapsed_s'] - p['checkpoint_evaluation_and_wait_s']))
            gains[arm] = []
            for d in (16384,32768):
                r = json.loads((OUT/f'{stem}-u{d}.json').read_text()); assert r['valid'] and r['weights_and_adam_finite']
                gains[arm].append(r['shell'][arm]['stable_gain_bits'])
        paired = np.subtract(gains['true'],gains['displaced']); by_seed.append(dict(seed=seed, own_arm_gains=gains, true_minus_displaced=paired.tolist(), improvement=float(paired[1]-paired[0])))
    deltas = [x['improvement'] for x in by_seed]; material = np.mean(deltas) > .001 and sum(x>0 for x in deltas)>=2
    needed = 32768/minimum_rate*1.12 + 45.; afford = bool(can_extend and remaining_s >= needed)
    extend = bool(material and afford)
    return dict(by_seed=by_seed, material=bool(material), expected_extension_wall_s=needed, remaining_wall_s=remaining_s,
                extend=extend, underconverged_due_to_cap=bool(material and not afford), selected_depth=65536 if extend else 32768,
                evidence='Frozen source DEV, stable paired own-arm (diagonal minus derangement) in bits. No TEST.')


def wave(seconds, can_extend):
    OUT.mkdir(exist_ok=False); start=time.monotonic(); end=start+seconds; jobs=[]; logs=[]; last=0.; traces=[]
    announced=set()
    try:
        for slot, (arm,seed) in enumerate(CELLS):
            log=(OUT/f'{arm}-s{seed}.log').open('x'); logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-u',__file__,'worker',str(slot)],stdout=log,stderr=subprocess.STDOUT))
        while True:
            elapsed=time.monotonic()-start; remaining=end-time.monotonic()
            if remaining<20: raise TimeoutError('hard wall protection; preserving available receipts')
            if elapsed-last>=10:
                info=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                traces.append(dict(elapsed_s=elapsed, gpu=info)); atomic(OUT/'telemetry.json',traces); last=elapsed
            states=[p.poll() for p in jobs]
            for a,s in CELLS:
                p=OUT/f'{a}-s{s}-first.json'
                if p not in announced and p.exists():
                    rr=json.loads(p.read_text()); print('FIRST_CUDA',a,s,rr['device'],rr['uuid'],rr['adam_iterations'],rr['placement'][0],flush=True); announced.add(p)
            for d in DEPTHS:
                if d not in announced and all((OUT/f'{a}-s{s}-u{d}.json').exists() for a,s in CELLS):
                    print('CHECKPOINT',d,'six complete',round(elapsed,1),'s',flush=True); announced.add(d)
            if any(rc not in (0,None) for rc in states): raise RuntimeError(f'worker failure: {states}')
            if all(rc==0 for rc in states): break
            if not (OUT/'depth-decision.json').exists() and all((OUT/f'{a}-s{s}-u32768.json').exists() for a,s in CELLS):
                firsts=[json.loads((OUT/f'{a}-s{s}-first.json').read_text()) for a,s in CELLS]
                if len({p['uuid'] for p in firsts})!=1 or any(firsts[k]['stream_sha256']!=firsts[k+1]['stream_sha256'] for k in (0,2,4)): raise ValueError('UUID/paired stream mismatch')
                choice=decision(remaining,can_extend); atomic(OUT/'depth-decision.json',choice); print('DEPTH_DECISION',json.dumps(choice),flush=True)
            elif not (OUT/'depth-decision.json').exists():
                for a,s in CELLS:
                    path=OUT/f'{a}-s{s}-progress.json'
                    if not path.exists(): continue
                    p=json.loads(path.read_text()); t=p['callback_steps']
                    if 4096<=t<32768:
                        core_s=p['elapsed_s']-p['checkpoint_evaluation_and_wait_s']
                        if (32768-t)*core_s/t > remaining-25: raise TimeoutError('minimum no longer fits measured wall cap')
            time.sleep(.5)
        atomic(OUT/'wave.json',dict(complete=True,workers=len(jobs),wall_s=time.monotonic()-start))
    except BaseException as exc:
        atomic(OUT/'wave.json',dict(complete=False,error=repr(exc),wall_s=time.monotonic()-start)); raise
    finally:
        for p in jobs:
            if p.poll() is None: p.terminate()
        for p in jobs:
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        for f in logs: f.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('worker').add_argument('slot',type=int,choices=range(6)); w=sub.add_parser('wave'); w.add_argument('--seconds',required=True,type=int); w.add_argument('--extension',action='store_true'); a=ap.parse_args()
    if a.cmd=='worker': worker(a.slot)
    else: wave(a.seconds,a.extension)
