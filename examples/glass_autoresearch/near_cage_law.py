"""Tripod probability conditional on an empty center and a matched three-arm cage.

This is a nearby-exterior positive control, not a cavity or a prediction from
an independent distant shell. The matched event itself is assumed, not inferred.
Fix the entire exterior. At fixed composition the number of A labels on the
three occupied arms is fixed, but need not be observed. For *each* possible A
count the decoration sum is symmetric in the three occupied coordination
numbers. Their multiset is the signature, so this factor cancels between
eligible masks. Integrating out the unobserved distant exterior cannot change
the resulting odds: all nonconstant exterior energies use only the supplied
L1=2,3 species. No labels or species from the central cross enter predict().
"""
from itertools import combinations, permutations, product
from operator import index

import numpy as np


OFFSETS = tuple(p for p in product(range(-3, 4), repeat=3) if 2 <= sum(map(abs, p)) <= 3)
_COORD = np.asarray(OFFSETS, dtype=np.int8)
_RADIUS = np.abs(_COORD).sum(axis=1)
_INNER = np.flatnonzero(_RADIUS == 2)
_OUTER = np.flatnonzero(_RADIUS == 3)
_ARMS = np.concatenate((np.eye(3, dtype=np.int8), -np.eye(3, dtype=np.int8)))
_CHOICES = np.asarray(list(combinations(range(6), 3)))
_MASK = np.zeros((6, 20), dtype=np.int16)
_MASK[_CHOICES.T, np.arange(20)] = 1
_TRIPOD = np.abs(_ARMS[_CHOICES].sum(axis=1)).sum(axis=1) == 3
if (int(_TRIPOD.sum()), int((~_TRIPOD).sum())) != (8, 12):
    raise AssertionError('incorrect cubic mask partition')


def _incident(left, right):
    return (np.abs(left[:, None, :] - right).sum(axis=2) == 1).astype(np.int16)


_INNER_ARM = _incident(_COORD[_INNER], _ARMS)
_OUTER_INNER = _incident(_COORD[_OUTER], _COORD[_INNER])
_S = _INNER_ARM @ _MASK
# s(s-1) = 2 per occupied pair sharing this exterior neighbor. Opposite
# arms share no site here; perpendicular arms share only their face diagonal.
_PAIR = _S * (_S - 1)
_ELL = np.array([0, 3, 5], dtype=np.int16)


def _terms(halo):
    species = halo[:, _INNER]
    occupied = (species != 0).astype(np.int16)
    a = occupied @ _INNER_ARM
    d = (halo[:, _OUTER] != 0) @ _OUTER_INNER - _ELL[species]
    # (d+s)^2 - d^2 = (2d+1)s + s(s-1).
    fields = (occupied * (2*d + 1)) @ _INNER_ARM
    linear = fields @ _MASK
    return a, linear, linear + occupied @ _PAIR


def _ratio(tripod, other):
    total = tripod + other
    return np.divide(tripod, total, where=total != 0,
                     out=np.full(total.shape, np.nan, dtype=np.float64))


def _weighted(energy, eligible, beta):
    # Shared max-log-weight shift (as in logsumexp); shift integer energies
    # *before* multiplying by beta so even extreme finite beta has a unit term.
    # All energies and all differences are much smaller than int16's range.
    floor = np.where(eligible, energy, 1000).min(axis=1, keepdims=True)
    gap = np.where(eligible, energy - floor, 0)
    with np.errstate(over='ignore', under='ignore'):
        weights = np.exp(-beta * gap)
    weights[~eligible] = 0.
    return _ratio(weights[:, _TRIPOD].sum(axis=1), weights[:, ~_TRIPOD].sum(axis=1))


def predict(halo, beta, signature=(3, 3, 5)):
    """Return P(tripod | matched, halo) and both feasible-mask multiplicities.

    Rows must be uint8 (0=vacancy, 1=A, 2=B), in lexicographic z,y,x OFFSETS
    order. `linear` omits shared-neighbor penalties; `feasible` ignores all
    energies. A missing class has zero weight; no eligible mask gives NaN.
    """
    halo = np.asarray(halo)
    if halo.dtype != np.uint8 or halo.ndim != 2 or halo.shape[1] != len(OFFSETS):
        raise ValueError(f'expected uint8 halo of shape (n, {len(OFFSETS)})')
    if halo.size and halo.max() > 2:
        raise ValueError('invalid species outside {0,1,2}')
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    sig = tuple(index(k) for k in signature)
    if len(sig) != 3 or not 0 <= sig[0] <= sig[1] <= sig[2] <= 5:
        raise ValueError('expected a sorted three-arm coordination signature in [0,5]')
    n = len(halo)
    result = {k: np.empty(n, dtype=np.float64) for k in ('exact', 'linear', 'feasible')}
    for k in ('count_tripod', 'count_T'):
        result[k] = np.empty(n, dtype=np.int64)
    # Bound temporary mask/energy storage independently of caller batch size.
    for start in range(0, n, 4096):
        batch = slice(start, start + 4096)
        a, linear, exact = _terms(halo[batch])
        eligible = (np.sort(a[:, _CHOICES], axis=2) == sig).all(axis=2)
        nt = eligible[:, _TRIPOD].sum(axis=1)
        no = eligible[:, ~_TRIPOD].sum(axis=1)
        result['count_tripod'][batch], result['count_T'][batch] = nt, no
        uniform = _ratio(nt, no)
        result['feasible'][batch] = uniform
        for key, energy in (('linear', linear), ('exact', exact)):
            result[key][batch] = _weighted(energy, eligible, beta) if beta else uniform
    return result


def _fixture(name):
    # Starting with all 18 distance-two sites occupied gives every arm a=5.
    # Axial deletions affect only their own arm; a diagonal deletion affects two.
    axes = {p for p in OFFSETS if 2 in map(abs, p) and sum(map(abs, p)) == 2}
    if name == 'T':
        removed = (axes - {(2, 0, 0), (-2, 0, 0)}) | {(-1, 0, -1), (-1, 0, 1)}
    else:
        removed = (axes - {(2, 0, 0)}) | {(0, 1, 1)}
        if name == 'mixed':
            removed.remove((0, 0, -2))
    row = np.zeros(len(OFFSETS), dtype=np.uint8)
    for i, p in enumerate(OFFSETS):
        if name != 'impossible' and p not in removed and (sum(map(abs, p)) == 2 or i % 3):
            row[i] = 1 + (i % 5 < 2)
    return row


def _literal(row, beta):
    """Independent, finite before/after NH-energy sums; no lattice generation."""
    steps = [p for p in product(range(-1, 2), repeat=3) if sum(map(abs, p)) == 1]
    halo = {p: int(s) for p, s in zip(OFFSETS, row) if s}

    def coord(p, state):
        return sum(tuple(x+y for x, y in zip(p, step)) in state for step in steps)

    def energy(state):
        # The omitted exterior contributes only constants to energy differences.
        return sum((coord(p, state) - (3 if species == 1 else 5))**2
                   for p, species in state.items())

    base = energy(halo)
    single = {p: energy({**halo, p: 1}) - base - (coord(p, halo)-3)**2 for p in steps}
    full = [[] for _ in range(4)]
    lines, classes = [], []
    a, linear, exact = _terms(row[None, :])
    production_arms = list(map(tuple, _ARMS))
    np.testing.assert_array_equal(a[0], [coord(p, halo) for p in production_arms])
    for arms in combinations(steps, 3):
        matched = sorted(coord(p, halo) for p in arms) == [3, 3, 5]
        j = np.flatnonzero((_MASK[:, :].T == [p in arms for p in production_arms]).all(axis=1)).item()
        line = sum(single[p] for p in arms)
        np.testing.assert_equal(line, linear[0, j])
        tripod = not any(tuple(-x for x in p) in arms for p in arms)
        np.testing.assert_equal(tripod, _TRIPOD[j])
        if matched:
            classes.append(tripod)
            lines.append(line)
        for labels in product((1, 2), repeat=3):
            state = {**halo, **dict(zip(arms, labels))}
            delta = energy(state) - base
            central = sum((coord(p, state) - (3 if state[p] == 1 else 5))**2 for p in arms)
            np.testing.assert_equal(delta - central, exact[0, j])
            if matched:
                full[labels.count(1)].append((tripod, delta))

    def chance(flags, energies):
        if not energies:
            return np.nan
        weights = np.exp(-beta * (np.asarray(energies) - min(energies)))
        return weights[np.asarray(flags)].sum() / weights.sum()

    actual = predict(row[None, :], beta)
    for key, value in [('count_tripod', sum(classes)), ('count_T', len(classes)-sum(classes)),
                       ('feasible', chance(classes, [0]*len(lines))), ('linear', chance(classes, lines))]:
        np.testing.assert_allclose(actual[key][0], value, rtol=2e-13, atol=0.)
    for states in full:  # Separate checks for 0, 1, 2, 3 arm A particles.
        np.testing.assert_allclose(actual['exact'][0], chance([c for c, _ in states], [e for _, e in states]),
                                   rtol=2e-13, atol=0.)


def verify():
    """Check 20 masks × 8 decorations per diagnostic halo; no archive or RNG."""
    names = ('tripod', 'T', 'mixed', 'impossible')
    halos = np.stack([_fixture(name) for name in names])
    betas = (0., 0.25, 3.05)
    for row in halos:
        for beta in betas:
            _literal(row, beta)
    zero = predict(halos, 0.)
    for k in ('exact', 'linear'):
        np.testing.assert_array_equal(zero[k], zero['feasible'])
    expected = predict(halos, 0.25)
    np.testing.assert_array_equal(expected['count_tripod'], [1, 0, 1, 0])
    np.testing.assert_array_equal(expected['count_T'], [0, 3, 1, 0])
    for k in ('exact', 'linear', 'feasible'):
        p = expected[k]
        if p.dtype != np.float64 or not (p[0] == 1 and p[1] == 0 and 0 < p[2] < 1 and np.isnan(p[3])):
            raise AssertionError(f'invalid support or dtype: {k}')
    if expected['exact'][2] == expected['linear'][2]:
        raise AssertionError('mixed diagnostic must distinguish pair term')
    lookup = {p: i for i, p in enumerate(OFFSETS)}
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            cols = [lookup[tuple(s*p[k] for k, s in zip(perm, signs))] for p in OFFSETS]
            changed = predict(halos[:, cols], 0.25)
            for k in expected:
                np.testing.assert_allclose(changed[k], expected[k], rtol=2e-14, atol=0.)
    return {'diagnostic_halos': list(names), 'masks': 20, 'decorations_per_mask': 8,
            'betas': list(betas), 'cubic_symmetries': 48}


if __name__ == '__main__':
    print(verify())
