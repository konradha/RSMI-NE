"""Fit one categorical L20 operator; the caller owns sampling, splits and I/O."""
import math

import numpy as np
import tensorflow as tf

from rsmine.coarsegrainer.cg_layers import Conv3DSingle
from rsmine.mi_estimator.critics import SeparableCritic


_BATCH = 128
_STATES = 3
_LL = (2, 2, 2)
_RATE = 0.003


def fit(meta, train, val, test, method, seed):
    """Select both networks on validation InfoNCE, then score the held-out parents.

    Each bank is (visible, environment), two uint8 arrays with axes
    (anchor or training epoch, parent, site). Parent order across evaluation
    anchors is fixed by the caller. Returned scores are in nats, not bits.
    """
    if method not in ('learned', 'uniform', 'staggered', 'site', 'axis', 'face', 'site-start-learned'):
        raise ValueError(f'Unknown operator {method!r}')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    n_shell = train[1].shape[-1]
    for v, e in (train, val, test):
        if (v.ndim != 3 or e.ndim != 3 or v.shape[:2] != e.shape[:2]
                or min(v.shape[:2]) < 1 or v.shape[1] % _BATCH or v.shape[2] != 8
                or e.shape[2] != n_shell or v.dtype != np.uint8 or e.dtype != np.uint8):
            raise ValueError('Expected aligned uint8 banks with complete parent batches')

    if method == 'learned':
        init = np.random.default_rng(seed).normal(0., 0.5, _LL)
    elif method == 'uniform':
        init = np.ones(_LL)
    elif method in ('site', 'site-start-learned'):
        # Match the L2 norm sqrt(8) of the extended unit-amplitude patterns.
        init = np.zeros(_LL)
        init[0, 0, 0] = math.sqrt(8) * (0.5 if method == 'site-start-learned' else 1.0)
    else:
        z, y, x = np.indices(_LL)
        parity = {'axis': z, 'face': z + y, 'staggered': z + y + x}[method]
        init = 1 - 2 * (parity % 2)
    kernel = Conv3DSingle(hidden_dim=1, input_shape=_LL,
                          init_rule=init.astype(np.float32)[..., None, None])
    kernel.trainable = method in ('learned', 'site-start-learned')
    # Fixed patterns also fit the stochastic encoder's sharpness and sign.
    scale = None if kernel.trainable else tf.Variable(0.5, name='pattern_scale')
    critic = SeparableCritic(hidden_dim=32, embed_dim=16, layers=1, activation='relu')
    eye = tf.eye(_STATES)
    critic(tf.zeros((1, n_shell * _STATES)), eye)  # build variables; no data or update
    variables = ([kernel.ws] if kernel.trainable else [scale]) + critic.trainable_variables
    optimizer = tf.keras.optimizers.Adam(learning_rate=_RATE)
    optimizer.build(variables)
    log_batch = tf.constant(math.log(_BATCH), tf.float32)

    def terms(v, e):
        v = tf.reshape(tf.one_hot(v, _STATES), (_BATCH, *_LL, 1, _STATES))
        logits = kernel(v)[:, 0, :]
        if scale is not None:
            logits = logits * scale
        prob = tf.transpose(tf.nn.softmax(logits, axis=1))  # q(h | V_i), [h, i]
        e = tf.reshape(tf.one_hot(e, _STATES), (_BATCH, n_shell * _STATES))
        # Each hard h has its own denominator over candidate environments j.
        # Average *after* log-softmax: no relaxed H ever enters the critic.
        log_ratio = log_batch + tf.nn.log_softmax(critic(e, eye), axis=1)
        return prob, log_ratio

    signature = [tf.TensorSpec((None, _BATCH, n), tf.uint8) for n in (8, n_shell)]

    @tf.function(input_signature=signature, autograph=False)
    def train_epoch(v, e):
        def update(i, total):
            with tf.GradientTape() as tape:
                p, r = terms(v[i], e[i])
                score = tf.reduce_sum(p * r) / _BATCH
                loss = -score
            optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            return i + 1, total + score

        count = tf.shape(v)[0]
        _, total = tf.while_loop(lambda i, _: i < count, update,
                                 (tf.constant(0), tf.constant(0.)), parallel_iterations=1)
        return total / tf.cast(count, tf.float32)

    @tf.function(input_signature=signature, autograph=False)
    def evaluate(v, e):
        def batch_scores(batch):
            p, r = terms(*batch)
            # For each i, average over all j != i without materializing B x B.
            other = (tf.reduce_sum(r, axis=1, keepdims=True) - r) / (_BATCH - 1)
            return tf.stack((tf.reduce_sum(p * r), tf.reduce_sum(p * other))) / _BATCH

        return tf.map_fn(batch_scores, (v, e), fn_output_signature=tf.TensorSpec((2,), tf.float32),
                         parallel_iterations=1)

    def batches(v, e):
        return (tf.convert_to_tensor(x.reshape(-1, _BATCH, x.shape[-1])) for x in (v, e))

    val_batches = tuple(batches(*val))
    best_nats = -math.inf
    history = []
    for epoch, (v, e) in enumerate(zip(*train), start=1):
        train_nats = float(train_epoch(*batches(v, e)).numpy())
        val_nats = float(evaluate(*val_batches).numpy()[:, 0].mean(dtype=np.float64))
        if not math.isfinite(train_nats) or not math.isfinite(val_nats):
            raise FloatingPointError(f'Nonfinite training or validation score at epoch {epoch}')
        history.append(dict(epoch=epoch, train_nats=train_nats, val_nats=val_nats))
        if val_nats > best_nats:
            best_nats, best_epoch = val_nats, epoch
            best_values = [w.numpy() for w in variables]

    for w, value in zip(variables, best_values):
        w.assign(value)
    weights = kernel.ws.numpy().ravel() * (1.0 if scale is None else scale.numpy())
    # Preserve parent-batch ID: aggregate only the repeated anchors, not parents.
    scores = evaluate(*batches(*test)).numpy().reshape(test[0].shape[0], -1, 2)
    scores = scores.mean(axis=0, dtype=np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(weights).all():
        raise FloatingPointError('Nonfinite selected operator or held-out score')
    paired, shuffled = scores.mean(axis=0)
    return dict(beta=float(meta['beta']), index=int(meta['index']), buffer=int(meta['buffer']),
                method=method, seed=int(seed), provenance=dict(meta),
                settings=dict(batch_size=_BATCH, h_states=_STATES, n_shell=int(n_shell),
                              encoder='Conv3DSingle', kernel_order='zyx C', temperature=1.,
                              kernel_init=('iid Normal(0,0.5^2)' if method == 'learned' else
                                           'single-site weight sqrt(8)/2' if method == 'site-start-learned' else 'fixed unit pattern with learned scale'),
                              critic='SeparableCritic, 1 x 32 ReLU, 16-dimensional embeddings',
                              objective='categorical-expected InfoNCE over candidate environments',
                              optimizer='Adam', learning_rate=_RATE, selection='max raw validation nats each epoch; earliest tie',
                              control='all nonidentity cyclic shifts within each held-out batch',
                              train_parents=train[0].shape[1], val_parents=val[0].shape[1],
                              test_parents=test[0].shape[1], val_anchors=val[0].shape[0],
                              test_anchors=test[0].shape[0], tensorflow=tf.__version__),
                optimizer_steps=int(optimizer.iterations.numpy()), best_epoch=best_epoch,
                best_val_nats=best_nats, epochs=history, weights=weights.tolist(),
                test_nats=float(paired), test_shuffled_nats=float(shuffled),
                test_minus_shuffled_nats=float(paired - shuffled),
                test_batches=[dict(paired=float(p), shuffled=float(s)) for p, s in scores])
