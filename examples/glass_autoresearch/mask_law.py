"""Exact fixed-composition NH species law for a given occupied coordination histogram.

At each occupied site k, E_B=(k-5)^2 and E_A-E_B=d_k=4k-16. Independent
Bernoulli labels of odds exp(u-beta*d_k), conditioned on sum A=n_a, give the
canonical law: u cancels. We choose u to center the independent sum, NOT as an
approximation to the constraint. Fourier projection extracts the coefficient of
z^n_a of the resulting probability polynomial. The midpoint grid has strictly
more points than the polynomial degree (8192 for 6000 sites), so there is no
aliasing and no division at z=-1. Conjugacy halves the grid.
"""
from functools import lru_cache
from itertools import combinations
from operator import index

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


_K = np.arange(7)
_E_A = (_K - 3)**2
_E_B = (_K - 5)**2
_DELTA = _E_A - _E_B


@lru_cache(maxsize=8)
def _grid(size):
    theta = (np.arange(size // 2) + 0.5) * (2 * np.pi / size)
    return theta, np.expm1(1j * theta)


def _tilt(counts, d, n_a):
    # Center logits on the energy at the fill boundary (midgap at a closed
    # shell). Then +/-log(2N) brackets the root, regardless of beta. Here d
    # already has beta and the fill-boundary shift applied.
    def residual(u):
        x = u - d
        high = x > 0
        # Subtract integers first: near a closed shell both remaining terms
        # can be far below one ulp at the scale of n_a.
        return (int(counts[high].sum()) - n_a
                + counts[~high] @ expit(x[~high])
                - counts[high] @ expit(-x[high]))

    bound = np.log(2.0 * counts.sum()) + 1
    u = brentq(residual, -bound, bound, xtol=2e-14)
    return expit(u - d), expit(d - u)


def _project(counts, beta, n_a, energy):
    sites = np.flatnonzero(counts)
    c = counts[sites]
    n = int(c.sum())
    d = _DELTA[sites]
    p = np.zeros(7)
    variance = 0.0
    if n_a == 0 or n_a == n or beta == 0 or len(c) == 1:
        if n:
            p[sites] = n_a / n
        if energy and n > 1 and beta == 0 and n_a not in (0, n):
            # Uniform sample without replacement.
            variance = n_a * (n - n_a) / (n - 1) * (c @ (d - c @ d / n)**2 / n)
    else:
        cumulative = c.cumsum()
        edge = cumulative.searchsorted(n_a)
        pivot = d[edge] if cumulative[edge] != n_a else 0.5 * (d[edge] + d[edge+1])
        with np.errstate(over='ignore'):  # +/-infinite logits saturate in expit
            a, b = _tilt(c, beta * (d - pivot), n_a)
        v = a * b
        grand_var = c @ v
        size = 1 << n.bit_length()
        theta, zm1 = _grid(size)
        # |b+a*z| <= exp(-2*a*b*sin(theta/2)^2). All required numerator
        # integrands are sums with at most TWO tagged factors removed, hence
        # retain >= grand_var-1/2 in the exponent. Centered energy numerators
        # have coefficient L1 norm <= (24*N)^2 (the squared maximal range).
        # Omit only frequencies bounded by 1e-32 EVEN AFTER that prefactor,
        # in the normalized DFT before division by P(sum=n_a). Otherwise use
        # the whole grid. This is a floating-point tolerance, not a change of law.
        log_bound = 32 * np.log(10.) + 2 * np.log(24. * n)
        if 2 * (grand_var - 0.5) > log_bound:
            cutoff = 2 * np.arcsin(np.sqrt(log_bound / (2 * (grand_var - 0.5))))
            stop = theta.searchsorted(cutoff, side='right')
            theta, zm1 = theta[:stop], zm1[:stop]
        z = 1 + zm1
        # Factor z out of high-occupancy sites before raising to a large
        # power. The remaining polynomial factors have small phase at zero.
        high = a > b
        shifted = np.where(high[:, None], zm1.conj(), zm1)
        factors = 1 + np.minimum(a, b)[:, None] * shifted
        shift = int(c[high].sum()) - n_a
        fourier = np.exp(1j * shift * theta + (c[:, None] * np.log(factors)).sum(axis=0))
        norm = fourier.real.sum()
        if not np.isfinite(norm) or norm <= 0:
            raise ArithmeticError('nonpositive fixed-composition coefficient')
        inverse = 1 / factors
        # Project the rarer of A and B for numerical accuracy near endpoints.
        tagged = np.minimum(a, b) * (inverse * (1 + shifted) * fourier).real.sum(axis=1) / norm
        p[sites] = np.where(high, 1 - tagged, tagged)
        if energy:
            # A constant times sum A is deterministic after projection. Center
            # the d's first to reduce cancellation, then measure E about its
            # grand mean. With r=a*z/(b+a*z), r-a=v*(z-1)/(b+a*z).
            # E[(E-E_grand)^2] integrand = (sum d*(r-a))^2 + sum d^2*r*(1-r).
            # Thus both distinct-site products and each Bernoulli diagonal
            # are included BEFORE conditioning, not an independent-site variance.
            h = d - ((c * v) @ d / grand_var if grand_var else 0.)
            inv_denom = np.where(high[:, None], inverse * z.conj(), inverse)
            first = ((c * h * v)[:, None] * inv_denom).sum(axis=0) * zm1
            diagonal = ((c * h**2 * v)[:, None] * inv_denom**2).sum(axis=0) * z
            offset = (fourier * first).real.sum() / norm
            variance = (fourier * (first**2 + diagonal)).real.sum() / norm - offset**2
            # Only discard negative roundoff, never a materially negative result.
            if not np.isfinite(variance) or variance < -2e-12 * max(1., (c * v) @ h**2):
                raise ArithmeticError('invalid fixed-composition energy variance')
            variance = max(0., variance)
    if (not np.all(np.isfinite(p)) or p.min() < -2e-12 or p.max() > 1 + 2e-12
            or abs(counts @ p - n_a) > 2e-12 * max(1, n)):
        raise ArithmeticError('fixed-composition probability or sum rule failed')
    np.clip(p, 0., 1., out=p)
    mean = counts @ (p * _E_A + (1 - p) * _E_B) if energy else 0.
    return p, float(mean), float(variance)


def moments(counts, beta, n_a, center=3):
    """Return per-site P(A), distinct-target P(A|center A), mean H and Var H.

    H is the TOTAL energy: divide mean by lattice volume and variance by
    volume squared for per-lattice-site units. Absent target classes get zero.
    If there is no such center or no A at all the entire conditional vector
    is zero (no eligible events). A pinned A consumes one site AND one A;
    computing its reduced ensemble separately avoids division by a rare event.
    """
    c = np.asarray(counts)
    if (c.shape != (7,) or c.dtype.kind not in 'iuf' or not np.all(np.isfinite(c))
            or np.any(c < 0) or (c.dtype.kind == 'f' and np.any(c != np.floor(c)))):
        raise ValueError('counts must contain seven nonnegative integer multiplicities')
    total = sum(map(int, c))
    if total > np.iinfo(np.intp).max // 32:
        raise ValueError('coordination histogram too large for native indexing')
    c = c.astype(np.int64, copy=False)
    n_a, center = index(n_a), index(center)
    if not 0 <= n_a <= total or not 0 <= center < 7:
        raise ValueError('invalid composition or center coordination')
    if np.ndim(beta) != 0 or not np.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    p, mean, variance = _project(c, float(beta), n_a, True)
    conditional = np.zeros(7)
    if n_a and c[center]:
        remaining = c.copy()
        remaining[center] -= 1
        conditional, _, _ = _project(remaining, float(beta), n_a - 1, False)
    return p, conditional, mean, variance


def verify():
    """Enumerate a tiny diagnostic histogram; no archive access or random draws."""
    counts = np.array([0, 1, 0, 3, 2, 1, 1])
    k = np.repeat(_K, counts)
    n = len(k)
    n_a = 4
    states = np.zeros((len(list(combinations(range(n), n_a))), n), dtype=bool)
    for row, active in zip(states, combinations(range(n), n_a)):
        row[list(active)] = True
    energies = np.where(states, (k-3)**2, (k-5)**2).sum(axis=1)
    # The enumerator sums only 70 positive terms; tolerances allow ordinary
    # double-precision Fourier/root/summation error, not fit to archive data.
    def check(actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=2e-12)

    for beta in (0., 3.05):
        weights = np.exp(-beta * (energies - energies.min()))
        weights /= weights.sum()
        mu = weights @ energies
        joint = np.zeros((n, n))
        for center in np.unique(k):
            p, conditional, mean, variance = moments(counts, beta, n_a, center)
            i = np.flatnonzero(k == center)[0]
            given = weights * states[:, i]
            given /= given.sum()
            others = np.arange(n) != i
            check(p[k], weights @ states)
            check(conditional[k[others]], given @ states[:, others])
            targets = counts.copy()
            targets[center] -= 1
            check(conditional[targets == 0], 0.)
            check(p[counts == 0], 0.)
            check(counts @ p, n_a)
            check(targets @ conditional, n_a - 1)
            check(mean, mu)
            check(variance, weights @ (energies - mu)**2)
            joint[k == center] = p[center] * conditional[k]
        np.fill_diagonal(joint, p[k])  # at the identical site A_i^2=A_i
        cov = joint - np.outer(p[k], p[k])
        check(joint, joint.T)
        check(cov.sum(axis=1), 0.)  # negative off-diagonals cancel diagonals
        if np.max(cov[~np.eye(n, dtype=bool)]) > 2e-12:
            raise AssertionError('positive distinct-site covariance')
        check(_DELTA[k] @ cov @ _DELTA[k], variance)
    for n_a, expected in ((0, 0.), (n, 1.)):
        p, conditional, mean, variance = moments(counts, 3.05, n_a, center=1)
        check(p[k], expected)
        check(conditional, expected * ((counts - (_K == 1)) > 0))
        check(mean, counts @ (_E_A if n_a else _E_B))
        check(variance, 0.)
    check(moments(counts, 3.05, 3, center=0)[1], 0.)
    return {'enumerated_states': len(states), 'betas': [0., 3.05]}


if __name__ == '__main__':
    print(verify())
