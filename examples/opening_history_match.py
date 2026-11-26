"""Physical oracle decoding, source empirical cuts, exactly matched per-fork contrasts."""
import ctypes as ct
from itertools import product

import numpy as np

MODES = ('opening', 'opening+packing', 'packing')
V = 8000
KEY = np.dtype('V8')


class Oracle:
    def __init__(self, library):
        self.lib = ct.CDLL(str(library)).opening_oracle
        self.lib.argtypes = [ct.c_int, ct.c_double] + [ct.c_void_p] * 5
        self.lib.restype = ct.c_int

    def __call__(self, state, beta):
        state = np.ascontiguousarray(state, dtype=np.uint8)
        if state.shape != (V,):
            raise ValueError('expected one L20 fork')
        h = np.empty(V, np.float64)
        op = np.empty_like(h)
        de = np.empty((V, 6), np.int8)
        coord = np.empty(V, np.uint8)
        rc = self.lib(20, beta, *(a.ctypes.data for a in (state, h, op, de, coord)))
        if rc or not np.all(np.isfinite(h + op)) or np.any(op < 0):
            raise ValueError(f'oracle failure {rc}')
        occ = state != 0
        k = np.column_stack((state[occ], coord[occ], np.sort(de[occ], axis=1).view(np.uint8)))
        sites = state.reshape(20, 20, 20)
        nn1 = sum(np.roll(sites, sign, axis=axis) == 1 for axis in range(3) for sign in (-1, 1))
        shell = sum(np.roll(sites, shift, axis=(0, 1, 2)) == 0
                    for shift in product(range(-2, 3), repeat=3) if sum(map(abs, shift)) == 2)
        return dict(key=k.view(KEY).ravel(), h=h[occ], op=op[occ], nn1=nn1.ravel()[occ].astype('u1'),
                    shell=shell.ravel()[occ].astype('u1'))


def scores(op, beta):
    out = np.full(len(op), -np.inf)
    np.log(op, out=out, where=op > 0)
    return out / beta


def fit_cuts(forks):
    arrays = {name: np.concatenate([f[name] for f in forks]) for name in ('key', 'op', 'shell')}
    keys, ix, n = np.unique(arrays['key'], return_inverse=True, return_counts=True)
    ends = n.cumsum()
    starts = ends - n
    bin_lo, bin_hi = starts + (n-1)//3, starts + (2*n-1)//3
    viable = n >= 24
    result = dict(key=keys[viable], n=n[viable])
    for col, values in [('opening', scores(arrays['op'], 3.2)), ('packing', arrays['shell'])]:
        sorted_val = values[np.lexsort((values, ix))]
        result[col + '_lo'] = sorted_val[bin_lo[viable]]
        result[col + '_hi'] = sorted_val[bin_hi[viable]]
    return result


def groups(feat, cuts, beta):
    k = feat['key']
    bins = np.searchsorted(cuts['key'], k)
    bins = np.minimum(bins, len(cuts['key']) - 1)
    seen = k == cuts['key'][bins]
    # Keep unknown keys out of matching even when their insertion position collides.
    out = {}
    for name, score in [('opening', scores(feat['op'], beta)), ('packing', feat['shell'])]:
        lo, hi = cuts[name + '_lo'][bins], cuts[name + '_hi'][bins]
        valid = seen & (lo < hi)
        a = valid & (score <= lo)
        b = valid & (score >= hi)
        strata = bins
        out[name] = weights(strata, a, b, feat['h'])
        if name == 'opening':
            pack = feat['nn1'].astype(int) * 3 + np.digitize(feat['shell'], [4, 6])
            out['opening+packing'] = weights(bins * 21 + pack, a, b, feat['h'])
    return out


def weights(key, low, high, h):
    _, ix = np.unique(key, return_inverse=True)
    n = ix.max() + 1
    na, nb = [np.bincount(ix[mask], minlength=n) for mask in (low, high)]
    keep = (na >= 2) & (nb >= 2)
    w = np.minimum(na, nb) * keep
    wa, wb = [(w / np.maximum(count, 1))[ix] * arm for count, arm in ((na, low), (nb, high))]
    hmin, hmax = np.full(n, np.inf), np.zeros(n)
    subset = (low | high) & keep[ix]
    np.minimum.at(hmin, ix[subset], h[subset]); np.maximum.at(hmax, ix[subset], h[subset])
    residual = (hmax[keep] - hmin[keep]).max(initial=0.)
    if residual > 1e-14:
        raise ValueError(f'inconsistent h inside exact stratum: {residual}')
    return dict(low=low, high=high, wa=wa, wb=wb, overlap=int(w.sum()), strata=int(keep.sum()),
                matched_low=int(np.count_nonzero(wa)), matched_high=int(np.count_nonzero(wb)),
                max_h_residual=float(residual))


def cdf(first, nlag, w=None):
    den = len(first) if w is None else w.sum()
    if not den:
        return np.full(nlag, np.nan)
    return np.bincount(first, weights=w, minlength=nlag + 1)[:nlag].cumsum() / den


def fork_risks(first, groups_, nlag):
    result = {}
    for mode, g in groups_.items():
        curves = {'all_sites': cdf(first, nlag)}
        for key, name in [('low', 'marginal_low'), ('high', 'marginal_high'),
                          ('wa', 'matched_low_risk'), ('wb', 'matched_high_risk')]:
            curves[name] = cdf(first, nlag, g[key])
        support = {k: g[k] for k in ('overlap', 'strata', 'matched_low', 'matched_high', 'max_h_residual')}
        support.update(low=int(g['low'].sum()), high=int(g['high'].sum()), occupied=len(first),
                       censored=int((first == nlag).sum()))
        result[mode] = (support, curves)
    return result
