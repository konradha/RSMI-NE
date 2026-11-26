"""Qualify compiled single-parent predictions before running any outcome analysis.

Run from the repository root after compiling both shared libraries (oracle-method.md).
Only preassigned open-development rows (group zero) of the final archive are touched.
"""
import argparse
import ctypes as C
from decimal import Decimal, localcontext
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

SCALE = 1 << 64
TEMPS = ['0', '0.338888888889', '0.677777777778', '1.016666666667', '1.355555555556',
         '1.694444444444', '2.033333333333', '2.372222222222', '2.711111111111', '3.05', '3.10', '3.20', '3.40']


def thresholds(text):
    with localcontext() as c:
        c.prec = 100
        return np.array([0 if d == -128 else SCALE if d <= 0 or not Decimal(text) else
                         int((-Decimal(text)*d).exp()*SCALE) for d in range(-128, 128)], dtype=object)


def bind(lib, name, *types):
    f = getattr(lib, name)
    f.argtypes = types
    f.restype = C.c_int
    return f


def pts(*values):
    return [v.ctypes.data for v in values]


def lattice_neighbors(L):
    g = np.arange(L**3).reshape((L,)*3)
    return np.stack([np.roll(g, shift, ax).ravel() for ax in (2, 1, 0) for shift in (-1, 1)], axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('staged', type=Path); p.add_argument('exploration', type=Path)
    p.add_argument('--lib', type=Path, default=Path('/tmp/opening_oracle.so'))
    p.add_argument('--reference', type=Path, default=Path('/tmp/opening_oracle_reference.so'))
    p.add_argument('--output', type=Path, default=Path('results/opening-mechanism/oracle-qualification.json'))
    args = p.parse_args()
    sys.path.insert(0, str(args.exploration.resolve()))
    legacy = importlib.import_module('pilots.l20-b310.paper_barriers')
    lib, ref = (C.CDLL(str(x.resolve())) for x in (args.lib, args.reference)); ptr = C.c_void_p
    oracle = bind(lib, 'opening_oracle', C.c_int, C.c_double, *([ptr]*5))
    gains = bind(lib, 'opening_oracle_gains', C.c_int, C.c_double, ptr, ptr, C.c_int, ptr)
    thresh = bind(lib, 'opening_threshold', C.c_double, C.c_int, ptr, ptr)
    native = bind(ref, 'opening_native_thresholds', C.c_char_p, ptr)
    replay = bind(ref, 'opening_replay', C.c_int, ptr, ptr, C.c_int, ptr, ptr)
    table_results, fixtures, timings = [], [], []
    for text in TEMPS:
        tab = thresholds(text); nt = np.zeros(128, dtype=np.uint64)
        empty = native(text.encode(), nt.ctypes.data)
        for d in range(-128, 128):
            hi, lo = C.c_uint64(), C.c_uint64()
            assert not thresh(float(text), d, C.byref(hi), C.byref(lo))
            expected = tab[d+128]
            assert hi.value*(1 << 32)+lo.value == expected, (text, d, hi.value, lo.value, expected)
            if d > 0:
                assert (SCALE if empty else int(nt[d])) == expected, (text, d)
        table_results.append(dict(beta=text, compared_signed_bins=256, native_positive_thresholds=127,
                                  first_forbidden=next((d for d in range(1, 128) if not tab[d+128]), None)))
    rng = np.random.Generator(np.random.PCG64(148033))
    cases = []
    for L, text in [(4, '0'), (4, '3.40'), (5, '3.05'), (6, '0.338888888889')]:
        s = rng.choice(np.uint8([0, 1, 2]), size=(L,)*3, p=[.25, .45, .30]); probes = np.flatnonzero(s)
        cases.append((f'random-L{L}-{text}', s, text, probes, None))
    for v in (0, 1, 2):
        s = np.full((4,)*3, v, dtype=np.uint8)
        cases.append((f'uniform-{v}', s, '3.40', [0], None))
    for i in (0, 4, 9):
        row = int(np.flatnonzero(np.load(args.staged/f'g{i:02}.npy') == 0)[0]); states = np.load(args.staged/f'b{i:02}.npy', mmap_mode='r')
        s = states[row]; occ = np.flatnonzero(s); probes = [occ[0], occ[len(occ)//2]]
        cases.append((f'development-b{i:02}-row{row}', s, TEMPS[i], probes, row))
    zeros_lost, far_examples = 0, 0
    for name, s, text, probes, row in cases:
        L = s.shape[0]; V = s.size; b = float(text); probes = np.asarray(probes, dtype=np.int32); nq = len(probes)
        t0 = time.perf_counter(); tab = thresholds(text); nn = lattice_neighbors(L)
        h, o = np.empty(V), np.empty(V); de, n = np.empty((V, 6), 'i1'), np.empty(V, 'u1')
        assert not oracle(L, b, *pts(s, h, o, de, n))
        np.testing.assert_array_equal(n, np.count_nonzero(s.ravel()[nn], axis=1))
        orig, after = np.empty((V, 6), 'i2'), np.full((3*V, nq, 6), -128, dtype='i2')
        H = replay(L, *pts(s, probes), nq, *pts(orig, after)); np.testing.assert_array_equal(orig, de)
        assert H == legacy.whole_h(s)
        _, direct = legacy.score(s)
        for j, forward in enumerate(reversed(direct)):
            np.testing.assert_array_equal(np.where(forward == legacy.INF, -128, forward).ravel(), de[:, j*2])
        base = tab[orig+128].sum(axis=1)
        href = np.array([int(x)/(3*SCALE) if t else 0 for x, t in zip(base, s.ravel())]); np.testing.assert_allclose(h, href, rtol=3e-16, atol=0)
        # Python arbitrary-size integers: exact products, signs and absolute zeros.
        e0 = np.repeat(np.arange(V), 3); e1 = nn[:, ::2].ravel()
        valid = ((e0[:, None] != probes) & (e1[:, None] != probes) &
                 (s.ravel()[probes] != 0) & (s.ravel()[e0] != s.ravel()[e1])[:, None])
        afterweights = tab[after+128]; gain = np.where(valid, np.maximum(0, afterweights.sum(axis=2)-base[probes]), 0)
        want = np.asarray(gain, dtype=float)*2.**-64
        got = np.empty(want.shape); assert not gains(L, b, *pts(s, probes), nq, *pts(got))
        np.testing.assert_array_equal(got, want)
        env = tab[orig[:, ::2].ravel()+128]; opening = (env[:, None]*gain).sum(axis=0)
        oref = np.array([int(x)/(9*SCALE**2) for x in opening]); np.testing.assert_allclose(o[probes], oref, rtol=6e-14, atol=0)
        assert np.all(o[s.ravel()==0] == 0)
        naive = np.maximum(0., np.asarray(afterweights,dtype=float).sum(axis=2)-np.asarray(tab[orig[probes]+128],dtype=float).sum(axis=1))
        hard = valid & (want > 0) & (naive == 0) & (env > 0)[:, None]; zeros_lost += int(hard.sum())
        distance = np.zeros((V,nq),np.int8)
        for dim in range(3):
            q = np.arange(V)//(L**dim)%L; dx = np.abs(q[:, None]-q[probes]); distance += np.minimum(dx,L-dx).astype('i1')
        far = ((np.minimum(distance[e0],distance[e1]) >= 3) & (want>0) & (env>0)[:,None]); far_examples+=int(far.sum())
        numerator = int(gain[hard].min()) if hard.any() else None
        fixtures.append(dict(name=name, beta=text, input_sha256=hashlib.sha256(s.tobytes()).hexdigest(), dev_row=row,
                             L=L, H=H, raw_directed_barriers=int(orig.size), sites_full_replay=nq, proposals_each=3*V,
                             probe_indices=probes.tolist() if row is not None else 'all occupied' if nq>1 else [0],
                             full_replay_gain_checks=got.size, positive_openings=int(np.count_nonzero(oref)),
                             positive_gains_lost_by_naive_subtraction=int(hard.sum()), smallest_lost_numerator_2pow64=numerator,
                             contributing_edges_min_distance_at_least_3=int(far.sum()),
                             max_relative_opening_error=float(np.max(abs(oref-o[probes])/np.where(oref==0,1.,oref))),
                             elapsed_seconds=round(time.perf_counter()-t0, 4)))
        print(f'{name}: all {got.size} per-proposal gains exact; whole-H and legacy barriers agree', flush=True)
    assert zeros_lost and far_examples, (zeros_lost, far_examples)
    for i, text in enumerate(TEMPS[:10]):
        st = np.load(args.staged/f'b{i:02}.npy',mmap_mode='r'); rows = np.flatnonzero(np.load(args.staged/f'g{i:02}.npy')==0)[:16]
        h, o = np.empty(8000), np.empty(8000); de, n = np.empty((8000,6),'i1'), np.empty(8000,'u1')
        assert not oracle(20,float(text),*pts(st[rows[0]],h,o,de,n)); start = time.perf_counter()
        for row in rows:
            assert not oracle(20,float(text),*pts(st[row],h,o,de,n))
        dt = time.perf_counter()-start
        timings.append(dict(beta=text, rows=rows.tolist(), wall_seconds=dt, parents_per_second=len(rows)/dt,
                            mode='single thread; 16 distinct development parents, warmed geometry/table; no I/O reduction'))
    report = dict(qualified=True, threshold_comparisons=table_results, full_H_replays=fixtures,
                  exact_gains_lost_if_naive=zeros_lost, contributions_at_distance_at_least_three=far_examples,
                  timings=timings, source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in
                  (Path(__file__), Path(__file__).with_name('opening_oracle.cpp'), Path(__file__).with_name('opening_oracle_reference.cpp'))})
    tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(args.output)


if __name__ == '__main__':
    main()
