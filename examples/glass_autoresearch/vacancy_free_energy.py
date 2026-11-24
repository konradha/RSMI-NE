"""Exact canonical log species partition function of a fixed vacancy mask.

For occupied-site coordination counts h, E_B=(k-5)^2, E_A=(k-3)^2:
Z = exp(-beta*h@E_B) [z^n_a] product_k (1+z*exp(-beta*(4*k-16)))^h_k.
A real tilt makes n_a typical, but does NOT replace the coefficient: we
project onto the exact count by a midpoint DFT larger than the total degree.
No vacancy probability, kinetic model or fitted parameter enters this sum.
"""
from itertools import combinations
from math import comb, log
from operator import index

import numpy as np
from scipy.special import expit, gammaln, logsumexp


_K = np.arange(7)
_A = (_K - 3)**2
_B = (_K - 5)**2


def _tilts(h, beta, n_a, total):
    """Group nearby exact projections on shared, well-conditioned contours."""
    cumul = h.cumsum(axis=1)
    # At a filled shell center the chemical potential in the gap to the next
    # NONEMPTY shell. Use twice the pivot k to keep the grouping key integral.
    pivot2 = (cumul >= n_a).argmax(axis=1) + (cumul > n_a).argmax(axis=1)
    with np.errstate(over='ignore'):
        delta = beta * (2 * (2*_K - pivot2[:, None]))
    hi = np.log(2. * total) + 1.
    lo = -hi
    # At an open shell these bounds put <1/2 defect on the saturated side;
    # at a closed shell the +/-log(2N) displacement balances even N:1 counts
    # across the gap. Factor out saturated A sites before summing the residual.
    step = 1. / np.sqrt(total.max())
    while np.max(hi-lo) > step * 0.01:
        mid = (lo+hi) * 0.5
        x = mid[:, None] - delta
        high = x > 0.
        q = expit(-np.abs(x))
        excess = (h*high).sum(axis=1) - n_a + (h*np.where(high, -q, q)).sum(axis=1)
        lo = np.where(excess < 0., mid, lo)
        hi = np.where(excess >= 0., mid, hi)
    # Quantizing the tilt only shares a Fourier basis, never changes the law.
    # d²(-log P_u(S=n_a))/du² = Var(S) <= N/4. The loss in log mass from
    # the optimal contour is <= N*(0.505*step)^2/8, well below one nat.
    bin_id = np.rint((lo+hi) / (2*step)).astype(np.int64)
    return pivot2, bin_id, step


def _projection(h, beta, n_a, x, size):
    """Common contour for this block; positive inverse tilt times P(S=n_a)."""
    high = x > 0.
    q = expit(-np.abs(x))  # rare label probability; never form 1-p for p near 1
    variance = h @ (q*(1.-q))
    theta = (np.arange(size//2) + 0.5) * (2*np.pi/size)
    # |1-p+p*exp(it)|² = 1-4*p*(1-p)*sin²(t/2). Thus all discarded
    # terms have magnitude <= 1e-30; after division by grid size the
    # absolute error in P is <= 1e-30 (before floating-point rounding).
    bound = 30.*np.log(10.)
    if 2.*variance.min() > bound:
        cutoff = 2.*np.arcsin(np.sqrt(bound / (2.*variance.min())))
        theta = theta[:theta.searchsorted(cutoff, side='right')]
    sin2 = np.sin(0.5*theta)**2
    re = 0.5*np.log1p(-4.*(q*(1.-q))[:, None]*sin2)
    signed = np.where(high, -q, q)
    im = np.arctan2(signed[:, None]*np.sin(theta), 1.-2.*q[:, None]*sin2)
    # Remove the linear phase before multiplying exponents by extensive counts.
    im -= signed[:, None]*theta
    shift = h[:, high].sum(axis=1) - n_a + h @ signed
    # Reductions over seven strata; bound row x frequency storage separately
    # from the number of candidates, without starting threaded BLAS per block.
    real = np.einsum('ik,kj->ij', h, re)
    phase = np.einsum('ik,kj->ij', h, im) + shift[:, None]*theta
    prob = (2./size)*(np.exp(real)*np.cos(phase)).sum(axis=1)
    # With an integer tilted mean, n_a is a mode (Darroch), hence its mass
    # is >= 1/(N+1). The tilt grid above loses <1 nat. Fail, do not clip,
    # if the projection could have extracted a tail by cancellation instead.
    if not np.all(prob >= 1./(3.*(1.+h.sum(axis=1)))):
        raise ArithmeticError('ill-conditioned canonical coefficient')

    # Use one fixed-count, minimum-energy species assignment as reference.
    # log Z = -beta*E_ref + log P(S=n_a) - log P(labels=ref). The latter
    # log probability is a sum of positive costs, not extensive subtractions.
    g = np.minimum(h, np.maximum(0, n_a - (h.cumsum(axis=1)-h)))
    energy = g @ _A + (h-g) @ _B
    mismatch = np.where(high, h-g, g)
    penalty = np.zeros(h.shape)
    np.multiply(mismatch, np.abs(x), where=mismatch != 0, out=penalty)
    return (-beta*energy + h @ np.log1p(np.exp(-np.abs(x)))
            + penalty.sum(axis=1) + np.log(prob))


def log_partition(hist, beta, n_a):
    """Return float64 ln Z_species for each integer row (N_0,...,N_6).

    n_a is fixed; endpoints and empty strata are supported. Absolute Fourier
    truncation error in ln Z is below 3*(N+1)*1e-30; ordinary float64
    arithmetic also rounds the returned extensive log (allow ~1e-9 at L20).
    """
    hist = np.asarray(hist)
    n_a = index(n_a)
    beta = float(beta)
    if hist.ndim != 2 or hist.shape[1] != 7 or hist.dtype.kind not in 'iu' or np.any(hist < 0):
        raise ValueError('hist must be nonnegative integer counts of shape (n,7)')
    if not np.isfinite(beta) or beta < 0. or n_a < 0:
        raise ValueError('expected nonnegative finite beta and nonnegative n_a')
    h, inv = np.unique(hist.astype(np.int64, copy=False), axis=0, return_inverse=True)
    total = h.sum(axis=1)
    if np.any(total < n_a):
        raise ValueError('n_a exceeds the number of occupied sites')
    inv = inv.ravel()
    if beta == 0.:
        return ((gammaln(total+1.) - gammaln((total-n_a)+1.)) - gammaln(n_a+1.))[inv]
    if n_a == 0:
        return (-beta*(h @ _B))[inv]
    closed = total == n_a
    result = np.empty(len(h))
    result[closed] = -beta*(h[closed] @ _A)
    interior = np.flatnonzero(~closed)
    if len(interior):
        pivot, bins, step = _tilts(h[interior], beta, n_a, total[interior])
        order = np.lexsort((bins, pivot))
        cuts = np.flatnonzero((np.diff(pivot[order]) != 0) | (np.diff(bins[order]) != 0)) + 1
        for group in np.split(order, cuts):
            j = group[0]
            with np.errstate(over='ignore'):
                x = bins[j]*step - beta*(2*(2*_K - pivot[j]))
            rows = interior[group]
            size = 1 << int(total[rows].max()).bit_length()
            chunk = max(1, min(512, 2**19 // (size//2)))
            for start in range(0, len(rows), chunk):
                ids = rows[start:start+chunk]
                result[ids] = _projection(h[ids], beta, n_a, x, size)
    return result[inv]


def verify():
    """Check against labeled subsets and an analytic, mixed-stratum L20 census."""
    sites = [[0, 1, 2, 3, 4, 5, 6], [2, 2, 3, 3, 3, 5, 5], [1, 5, 5, 1, 5, 3, 3]]
    sites.append(sites[1][::-1])
    hist = np.array([np.bincount(row, minlength=7) for row in sites])
    for beta in (0., 0.25, 3.05):
        for n_a in range(8):
            want = []
            for row in sites:
                energies = [sum((k-(3 if i in subset else 5))**2 for i, k in enumerate(row))
                            for subset in combinations(range(len(row)), n_a)]
                want.append(logsumexp(-beta*np.asarray(energies)))
            got = log_partition(hist, beta, n_a)
            np.testing.assert_allclose(got, want, rtol=0., atol=2e-12)
            np.testing.assert_allclose(got-got[0], np.subtract(want, want[0]), rtol=0., atol=2e-12)
            np.testing.assert_array_equal(got[-1], got[1])
            np.testing.assert_array_equal(log_partition(hist[::-1], beta, n_a), got[::-1])
    # 5999 ordinary sites (k=4, energy 1 regardless of label), one site
    # k=r. Split the exact sum by that one label. Python's comb is integer.
    large = np.zeros((2, 7), dtype=np.int64)
    large[:, 4] = 5999
    large[0, 0] = large[1, 6] = 1
    for beta in (0., 3.05):
        want = [np.logaddexp(log(comb(5999, 2399)) - beta*(5999+(r-3)**2),
                             log(comb(5999, 2400)) - beta*(5999+(r-5)**2)) for r in (0, 6)]
        got = log_partition(large, beta, 2400)
        np.testing.assert_allclose(got, want, rtol=0., atol=1e-9)
        np.testing.assert_allclose(np.diff(got), np.diff(want), rtol=0., atol=1e-9)
    return {'small_histograms': len(sites), 'n_a': list(range(8)), 'betas': [0., 0.25, 3.05],
            'L20_mixed_counts': [1, 5999], 'L20_n_a': 2400}


if __name__ == '__main__':
    print(verify())
