"""Locked score-level diagnostic; run from repo root: PYTHONPATH=. python examples/opening_numerics.py"""
import csv
import hashlib
import inspect
import json
import math
import os
import re
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='-1', TF_USE_LEGACY_KERAS='1', WANDB_MODE='disabled',
                  OPENBLAS_NUM_THREADS='1', TF_NUM_INTEROP_THREADS='1', TF_NUM_INTRAOP_THREADS='1',
                  MPLBACKEND='Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from rsmine.mi_estimator.VBMI_bounds import infonce_lower_bound

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/opening-mechanism/numerics'
WORKER = ROOT / 'results/history-overlap-field/checkpoints'
N = 48
CAP = math.log(N)
SIGNAL = CAP - math.log1p((N - 1) * math.exp(-16))
EYE = np.eye(N)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(s):
    d = s.astype(np.float64)
    z = d - d.max(axis=1, keepdims=True)
    expz = np.exp(z)
    norm = expz.sum(axis=1, keepdims=True)
    return CAP + (np.diag(z) - np.log(norm[:, 0])).mean(), (expz / norm - EYE) / N, z


def identity_critic(x, residual):
    return x + residual


class Probe:
    """One zero-start Adam step on a score residual, never a sequence of training steps."""
    def __init__(self, graph=False):
        self.residual = tf.Variable(tf.zeros([N, N], tf.float32))
        self.adam = tf.keras.optimizers.Adam(.009)
        self.adam.build([self.residual])
        self.advance = tf.function(self._advance, autograph=False) if graph else self._advance

    def reset(self):
        for v in [self.residual, *self.adam.variables()]:
            v.assign(tf.zeros_like(v))

    def _advance(self, x):
        with tf.GradientTape() as tape:
            value = infonce_lower_bound(x, self.residual, identity_critic)
            loss = -value
        g = tape.gradient(loss, self.residual)
        self.adam.apply_gradients([(g, self.residual)])
        return value, g

    def state(self):
        it, m, v = self.adam.variables()
        assert int(it) == 1 and m.shape == v.shape == self.residual.shape
        return [x.numpy() for x in (self.residual, m, v)]


def stress_cases():
    rng = np.random.default_rng(20261126)
    draws = []
    for _ in range(32):
        x, y = rng.choice([-1., 1.], size=(2, N, 8))
        draws.append((x @ y.T / 4).astype(np.float32))
    for name, raw, analytic in [('zero', np.zeros((N, N)), 0.), ('independent', draws[0], None),
                                ('independent_shuffle', np.roll(draws[0], 1, axis=1), None),
                                ('signal', 16 * EYE, SIGNAL), ('signal_shuffle', np.roll(16 * EYE, 1, axis=1), SIGNAL - 16)]:
        for mode, exponent in [('none', 0)] + [(m, k) for k in (12, 18, 20, 22, 24) for m in ('constant', 'alternating')]:
            s = raw.astype(np.float64)
            if mode != 'none':
                signs = np.ones(N)
                if mode == 'alternating':
                    signs[1::2] = -1
                s += signs[:, None] * 2**exponent
            yield name, 0, mode, exponent, s.astype(np.float32), raw, analytic
    for idx, d in enumerate(draws[1:], 1):
        for name, x in [('independent', d), ('independent_shuffle', np.roll(d, 1, axis=1))]:
            yield name, idx, 'none', 0, x, x, None


def retained_evidence():
    records, ports = [], []
    for seed in (6101, 6102, 6103):
        for arm in ('true', 'displaced'):
            stem = f'{arm}-s{seed}'
            for kind in ('first', 'progress', 'failure'):
                path = WORKER / f'{stem}-{kind}.json'
                if not path.is_file():
                    if kind != 'failure':
                        raise FileNotFoundError(path.name)
                    continue
                data = json.loads(path.read_bytes())
                assert (arm, seed) == (data['arm'], data['seed'])
                update = data['adam_iterations']
                assert update == data['callback_steps']
                row = dict(source=path.name, sha256=sha(path), record=kind, arm=arm, seed=seed,
                           adam_iteration=update, accepted_range_nats=data['train_mi_range_nats'])
                if kind == 'failure':
                    m = re.fullmatch(r"FloatingPointError\('invalid stock training bound (\S+) at Adam step (\d+)'\)", data['error'])
                    assert m and int(m[2]) == update
                    row.update(rejected_nats=float(m[1]), excess_nats=float(m[1]) - CAP)
                records.append(row)
            for depth in (8192, 16384, 32768, 65536):
                p = WORKER / f'{stem}-u{depth}.npz'
                if p.is_file():
                    ports.append(p.name)
    return records, ports


def maxerr(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - b).max())


def run():
    if (tf.__version__, tfp.__version__) != ('2.18.1', '0.25.0') or tf.config.get_visible_devices('GPU'):
        raise RuntimeError('Pinned TensorFlow/TFP on CPU required')
    retained, ports = retained_evidence()
    rows = []
    with tf.device('/CPU:0'):
        baseline = Probe()
        probes = [('eager', Probe()), ('graph', Probe(graph=True))]
        for name, draw, mode, k, s, raw, analytic in stress_cases():
            ref, grad64, shifted = reference(s)
            base, _, unshifted = reference(raw)
            changed = not np.array_equal(shifted, unshifted)
            reference_ok = bool(np.isfinite(ref) and np.isfinite(grad64).all() and ref <= CAP + 1e-12
                                and (analytic is None or abs(ref - analytic) <= 1e-12)
                                and (changed or abs(ref - base) <= 1e-12))
            baseline.reset()
            baseline.adam.apply_gradients([(tf.constant(grad64, dtype=tf.float32), baseline.residual)])
            ref_param, ref_m, ref_v = baseline.state()
            for runtime, probe in probes:
                probe.reset()
                val, g = probe.advance(tf.constant(s))
                stock, grad = float(val), g.numpy()
                param, m, v = probe.state()
                finite = all(np.isfinite(a).all() for a in [param, m, v, ref_param, ref_m, ref_v])
                ge = maxerr(grad, grad64)
                pe, me, ve = maxerr(param, ref_param), maxerr(m, ref_m), maxerr(v, ref_v)
                rows.append(dict(fixture=name, draw=draw, offset=mode, power=k, runtime=runtime,
                                 quantized_differences=changed, reference_nats=ref, ref_minus_unshifted_nats=ref-base,
                                 reference_ok=reference_ok, stock_nats=stock, stock_minus_reference_nats=stock-ref,
                                 scalar_ok=bool(np.isfinite(stock) and stock <= CAP + 1e-5 and abs(stock-ref) <= 1e-4),
                                 gradient_abs_error=ge, gradient_finite=bool(np.isfinite(grad).all()),
                                 gradient_ok=bool(np.isfinite(grad).all() and ge <= 1e-7),
                                 updated_residual_absmax=float(np.abs(param).max()), first_moment_absmax=float(np.abs(m).max()),
                                 second_moment_absmax=float(np.abs(v).max()), updated_residual_error=pe,
                                 first_moment_error=me, second_moment_error=ve, adam_finite=bool(finite),
                                 adam_ok=bool(finite and max(pe, me, ve) <= 1e-6)))
    return rows, retained, ports


def null_summary(rows):
    def vector(label):
        return np.array([r['reference_nats'] for r in rows if r['fixture'] == label and r['offset'] == 'none' and r['runtime'] == 'graph'])
    a, b = vector('independent'), vector('independent_shuffle')
    assert len(a) == len(b) == 32
    return {k: dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)), min=float(x.min()), max=float(x.max()))
            for k, x in [('independent', a), ('deranged', b), ('paired_minus_deranged', a-b)]}


def plot(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for ax, mode in zip(axes, ('constant', 'alternating')):
        panel = [r for r in rows if r['fixture'] == 'signal' and r['offset'] in ('none', mode)]
        for column, runtime, symbol, name in [('stock_nats','eager','o','stock eager'), ('stock_nats','graph','x','stock graph'),
                                             ('reference_nats','graph','.','shifted float64')]:
            data = [r for r in panel if r['runtime'] == runtime]
            ax.plot(range(len(data)), [r[column] for r in data], marker=symbol, lw=1.4, label=name)
        ax.set_xticks(range(6), ['0', '$2^{12}$', '$2^{18}$', '$2^{20}$', '$2^{22}$', '$2^{24}$'])
        ax.set(xlabel='Row offset magnitude', ylabel='Measured bound (nats)', title=f'Signal, {mode} offsets')
    axes[0].legend(fontsize=8)
    pair = [[r['reference_nats'] for r in rows if r['fixture'] == name and r['offset'] == 'none' and r['runtime'] == 'graph']
            for name in ('independent', 'independent_shuffle')]
    axes[2].plot([0, 1], pair, 'o-', lw=.6, ms=3, alpha=.35, color='#16656a')
    axes[2].set_xticks([0, 1], ['Independent pairs', 'Deranged'])
    axes[2].set(xlim=(-.3, 1.3), ylabel='Shifted float64 bound (nats)', title='32 synthetic null panels')
    fig.savefig(OUT / 'controls.png', dpi=180)
    plt.close(fig)


def save():
    rows, retained, ports = run()
    with (OUT / 'controls.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(rows)
    historical = dict(batch_size=N, cap_nats=CAP, records=retained, scheduled_portables_present=ports,
                      prior_score_matrices_available=False, prior_failing_model_and_adam_state_available=False)
    if ports:
        raise RuntimeError('Unexpected historical portable; inspect before deciding gate')
    (OUT / 'worker-evidence.json').write_text(json.dumps(historical, indent=2, allow_nan=False)+'\n')
    checks = {k: sum(not r[k] for r in rows) for k in ('reference_ok', 'scalar_ok', 'gradient_ok', 'adam_ok')}
    rejects = sum(r['record'] == 'failure' for r in retained)
    resolved = not rejects
    numeric_pass = not any(checks.values())
    bound_src = Path(inspect.getsourcefile(infonce_lower_bound))
    src = [bound_src, ROOT/'rsmine/coarsegrainer/cg_optimisers.py', ROOT/'examples/stock_glass_history_gpu.py',
           ROOT/'examples/opening_numerics.py', OUT/'lock.json']
    result = dict(comparison_allowed=bool(numeric_pass and resolved), stock_numeric_gate_pass=numeric_pass,
                  historical_failure_resolved=resolved, retained_rejections=rejects, cap_nats=CAP,
                  measured_runtime_cases=len(rows), failed_tests=checks,
                  stock_above_cap=sum(r['stock_nats'] > CAP+1e-5 for r in rows),
                  stock_equals_4=sum(r['stock_nats'] == 4 for r in rows),
                  nonfinite_gradients=sum(not r['gradient_finite'] for r in rows),
                  nonfinite_parameter_or_adam=sum(not r['adam_finite'] for r in rows),
                  independent_controls_nats=null_summary(rows),
                  versions={'tensorflow':tf.__version__, 'tensorflow_probability':tfp.__version__, 'numpy':np.__version__,
                            'tf_keras':tf.keras.optimizers.Adam.__module__, 'device':'CPU'},
                  sha256={str(p.relative_to(ROOT)):sha(p) for p in src})
    result['sha256']['tfp/mutual_information.py'] = sha(inspect.getsourcefile(tfp.vi.mutual_information.lower_bound_info_nce))
    (OUT / 'decision.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('comparison_allowed','failed_tests','measured_runtime_cases','stock_above_cap','stock_equals_4')}))


if __name__ == '__main__':
    save()
