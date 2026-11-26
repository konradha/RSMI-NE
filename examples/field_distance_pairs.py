"""Signed finite-aperture vector covariance on the L=20 native z,y,x grid.

prepare each frozen TRAIN bank once. Fields must already have the chosen units
and copy weights (a weight w in the trace corresponds to sqrt(w) in the field).
There is no normalization, regression, model choice or fitting on spatial pairs.
"""
from dataclasses import dataclass
from functools import lru_cache
from operator import index

import numpy as np
from scipy.linalg import qr
from scipy.sparse import csr_matrix

L, STATES = 20, 210
GRID = (L, L, L)
_UPPER = np.triu_indices(STATES)
_PAIR = np.empty((STATES, STATES), dtype=np.int32)
_PAIR[_UPPER] = _PAIR[_UPPER[::-1]] = np.arange(len(_UPPER[0]), dtype=np.int32)


@dataclass(frozen=True)
class _EndpointMetric:
    basis: np.ndarray
    i: np.ndarray
    j: np.ndarray
    off: np.ndarray
    coeff: np.ndarray

    def bilinear(self, hist_left, hist_right):
        left = hist_left @ self.basis
        right = hist_right @ self.basis
        terms = left[:, self.i] * right[:, self.j]
        terms[:, self.off] += left[:, self.j[self.off]] * right[:, self.i[self.off]]
        return terms @ self.coeff


@dataclass(frozen=True)
class _Bank:
    gram: np.ndarray
    endpoints: tuple


def prepare(fields):
    """Cache the exact copy-trace Gram lookup and endpoint-mean bilinear maps."""
    f = np.asarray(fields, dtype=np.float64)
    if f.ndim != 4 or f.shape[1:3] != (STATES, 3) or not f.shape[0] or not f.shape[-1]:
        raise ValueError('fields must have shape (models,210,3,copies)')
    if not np.isfinite(f).all():
        raise ValueError('fields contain non-finite values')
    models = len(f)
    gram = np.empty((len(_UPPER[0]), 3, models))
    endpoints = []
    for c in range(3):
        # A constant shift cancels from both-endpoint covariance. It prevents
        # catastrophic subtraction when, e.g., a copy has a large constant offset.
        v = np.ascontiguousarray(f[:, :, c, :] - f[:, :1, c, :])
        for m in range(models):
            gram[:, c, m] = (v[m] @ v[m].T)[_UPPER]
        # These equivariant column spaces are small and shared between models.
        # Use only linear dependence (including zero-padded copies), never a
        # variance cutoff on physical configurations. Check each model's scale.
        flat = np.ascontiguousarray(v.transpose(1, 0, 2).reshape(STATES, -1))
        scale = np.max(abs(flat), axis=0)
        active = scale > 0
        if np.any(active):
            unit = flat[:, active] / scale[active]
            q, r, _ = qr(unit, mode='economic', pivoting=True, check_finite=False)
            rank = int(np.sum(abs(np.diag(r)) > 1e-14))
            basis = np.ascontiguousarray(q[:, :rank])
            if np.max(abs(unit - basis @ (basis.T @ unit))) > 5e-14:
                basis = np.eye(STATES)
        else:
            basis = np.empty((STATES, 0))
        i, j = np.triu_indices(basis.shape[1])
        coeff = np.empty((len(i), models))
        for m in range(models):
            w = basis.T @ v[m]
            coeff[:, m] = (w @ w.T)[i, j]
        endpoints.append(_EndpointMetric(basis, i, j, np.flatnonzero(i != j), coeff))
    return _Bank(gram.reshape(len(_UPPER[0]), -1), tuple(endpoints))


@lru_cache(maxsize=16)
def _neighbors(distances):
    if not distances or len(set(distances)) != len(distances) or not all(3 <= d < L // 2 for d in distances):
        raise ValueError('distinct distances from 3 through 9 required (disjoint non-antipodal stars)')
    grid = np.arange(L**3, dtype=np.int16).reshape(GRID)
    return np.stack([np.roll(grid, -d, a).ravel() for d in distances for a in range(3)])


def score(state_index_grid, fields, distances, rng):
    """Return counts[D,3], true[M,D,L/T], shuffled[M,D,L/T], metadata.

    Flat input is C-order z,y,x, same as 03, without a component reversal.
    One rng.permutation on eligible labels in flat increasing-site order is
    shared across all models and offsets. Every retained bond is directed
    toward increasing native coordinate modulo 20; do not add reverse bonds.
    """
    dist = tuple(index(d) for d in distances)
    neighbors = _neighbors(dist)
    bank = fields if isinstance(fields, _Bank) else prepare(fields)
    grid = np.asarray(state_index_grid)
    if grid.shape not in (GRID, (L**3,)) or grid.dtype.kind != 'i':
        raise ValueError('expected a flat or 20-cubed signed integer state grid')
    grid = grid.ravel(order='C')
    if grid.min() < -1 or grid.max() >= STATES:
        raise ValueError('grid label outside -1,0,...,209')
    eligible = np.flatnonzero(grid >= 0)
    permuted = np.full(L**3, -1, dtype=grid.dtype)
    permuted[eligible] = np.random.default_rng(rng).permutation(grid[eligible])
    targets = neighbors[:, eligible]
    row, origin = np.nonzero(grid[targets] >= 0)
    dest = targets[row, origin]
    origin = eligible[origin]
    groups = 3 * len(dist)
    counts = np.bincount(row, minlength=groups)
    models = bank.gram.shape[1] // 3
    out = np.full((2, len(dist), 2, models), np.nan)
    if len(row):
        left = np.concatenate((grid[origin], permuted[origin]))
        right = np.concatenate((grid[dest], permuted[dest]))
        n = np.tile(counts, 2)
        # CSR rows sum the frozen dot-product table over actual state pairs,
        # without materializing a pair x model temporary or rebuilding fields.
        indptr = np.r_[0, np.cumsum(n, dtype=np.int32)].astype(np.int32, copy=False)
        weights = csr_matrix((np.ones(len(left)), _PAIR[left, right], indptr),
                             shape=(2*groups, len(bank.gram)), copy=False)
        sums = (weights @ bank.gram).reshape(2*groups, 3, models)
        base = STATES * np.concatenate((row, row + groups))
        lh, rh = (np.bincount(base + labels, minlength=2*groups*STATES).reshape(-1, STATES)
                  for labels in (left, right))
        denom = np.maximum(n, 1)[:, None]
        for c, metric in enumerate(bank.endpoints):
            sums[:, c] -= metric.bilinear(lh, rh) / denom
        sums /= np.maximum(n-1, 1)[:, None, None]
        sums[n < 2] = np.nan
        by_axis = sums.reshape(2, len(dist), 3, 3, models)
        out[:, :, 0] = sum(by_axis[:, :, a, a] for a in range(3)) / 3
        out[:, :, 1] = sum(by_axis[:, :, a, c] for a in range(3) for c in range(3) if a != c) / 6
    return (counts.reshape(-1, 3), out[0].transpose(2, 0, 1),
            out[1].transpose(2, 0, 1), {'n_eligible': len(eligible)})
