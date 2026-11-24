"""Canonical unlike free bonds on the periodic L20 torus; outcome-blind extraction."""
import itertools

import numpy as np

from l20_propensity_predict import IDEAL, NO_SWAP, neighbor_sum, verify_barrier
from l20_propensity_proximity import four_shells


DIRECTIONS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=np.int16)
CORE = np.array(list(itertools.product(range(-1, 3), range(-1, 2), range(-1, 2))), dtype=np.int16)
FACES = []
for d, direction in enumerate(DIRECTIONS):
    k = d // 2
    surface = CORE[CORE[:, k] == (CORE[:, k].max() if d % 2 else CORE[:, k].min())]
    FACES.append(surface + 2 * direction)
# Explicitly check the geometric gap and disjoint faces, including under PBC.
_dist = lambda x: np.abs((x + 10) % 20 - 10).sum(axis=-1)
assert all(_dist(f[:, None] - CORE).min() == 2 for f in FACES)
assert len(np.unique(np.concatenate(FACES), axis=0)) == sum(map(len, FACES))


def costs(state):
    occ = (state != 0).astype(np.int16)
    m = neighbor_sum(occ)
    r = (m - IDEAL[state]) * occ
    moment, h = neighbor_sum(r), r*r
    edges = []
    for axis in range(3):
        sb, ob, mb, rb, moment_b, hb = (np.roll(a, -1, axis) for a in (state, occ, m, r, moment, h))
        d = ob - occ
        dh = ob*(m-d-IDEAL[sb])**2 + occ*(mb+d-IDEAL[state])**2 - h - hb
        dh += 2*d*(moment-rb-moment_b+r) + d*d*(m-ob+mb-occ)
        dh[state == sb] = NO_SWAP
        edges.extend((np.roll(dh, 1, axis), dh))
    edges = np.stack(edges, axis=-1)
    minimum = edges.min(axis=-1)
    verify_barrier(state, minimum)
    return m, edges, minimum


def components(free):
    """Connected components of all endpoints of free bonds, by six-neighbor contact."""
    unseen = set(map(tuple, np.argwhere(free)))
    result = {}
    while unseen:
        seed = unseen.pop()
        todo, sites = [seed], [seed]
        while todo:
            p = np.array(todo.pop())
            for v in (p + DIRECTIONS) % 20:
                v = tuple(v)
                if v in unseen:
                    unseen.remove(v)
                    todo.append(v)
                    sites.append(v)
        sites_array = np.array(sites)
        result.update(dict.fromkeys(sites, sites_array))
    return result


def encoded_cost(v):
    return np.where(v == NO_SWAP, 128, v)


def raw_tests(raw):
    """Interpretable scalar tests of the raw 36 species. Same options for random and fitted trees."""
    tests = [raw]
    names = [f'species@{tuple(map(int, c))}' for c in CORE]
    # Individual species and differences/counts on opposing endcaps, flanks and halves.
    for sp in range(3):
        present = raw == sp
        tests.append(present)
        names += [f's{sp}@{tuple(map(int, c))}' for c in CORE]
        tests.append(present.sum(axis=1, keepdims=True))
        names.append(f'count s{sp} core')
        for k in range(3):
            planes = np.unique(CORE[:, k])
            for kind, (lo, hi) in [('face', (CORE[:, k] == planes.min(), CORE[:, k] == planes.max())),
                                   ('half', (CORE[:, k] < planes.mean(), CORE[:, k] > planes.mean()))]:
                if k and kind == 'half':  # same as outer face for a three-site width
                    continue
                a, b = present[:, lo].sum(1), present[:, hi].sum(1)
                tests.append(np.stack((a, b, b-a), axis=1))
                names.extend(f's{sp} {kind} {"uvw"[k]} {s}' for s in ('negative', 'positive', 'difference'))
    return np.concatenate(tests, axis=1).astype(np.int16), names


def simple_code(raw):
    # Species of the lower-species endpoint and occupancy excess in the outward cap.
    a = raw[:, np.all(CORE == [0, 0, 0], axis=1)].ravel()
    front = (raw[:, CORE[:, 0] == 2] != 0).sum(1)
    back = (raw[:, CORE[:, 0] == -1] != 0).sum(1)
    return ((a != 0).astype(np.int8)*2 + (front > back)).astype(np.int8)


def extract(state):
    """Extract one parent's motifs once, without consulting any future state or count."""
    if state.shape != (20, 20, 20) or state.dtype != np.uint8 or state.max() > 2:
        raise ValueError('expected 20^3 uint8 species')
    m, edges, minimum = costs(state)
    free = minimum <= 0
    clusters = components(free)
    distance = four_shells(free)
    density1 = neighbor_sum(free.astype(np.int16))
    edge_coord = np.argwhere(edges[..., 1::2] <= 0)  # each unordered bond once
    anchor, axis = edge_coord[:, :3].copy(), edge_coord[:, 3]
    near = (anchor + np.eye(3, dtype=np.int16)[axis]) % 20
    flip = state[tuple(anchor.T)] > state[tuple(near.T)]
    anchor[flip] = near[flip]
    sign = np.where(flip, -1, 1)
    n = len(anchor)
    basis = np.eye(3, dtype=np.int16)[(axis[:, None] + np.arange(3)) % 3]
    basis[:, 0] *= sign[:, None]

    def indices(off):
        xyz = (anchor[:, None] + (off @ basis)) % 20
        return np.ravel_multi_index(tuple(xyz.transpose(2, 0, 1)), state.shape)

    core_idx = indices(CORE)
    raw = state.ravel()[core_idx]
    target = np.zeros((n, 6, 12), dtype=np.int32)
    valid = np.zeros(target.shape, dtype=bool)
    for d, face in enumerate(FACES):
        ids = indices(face)
        target[:, d, :len(face)] = ids
        valid[:, d, :len(face)] = state.ravel()[ids] != 0
    count = valid.sum(axis=2)
    if n and np.any(~count.any(axis=1)):
        raise ValueError('motif has no eligible face')
    bond, face, slot = np.nonzero(valid)
    target_site = target[bond, face, slot]
    # Canonical order for every local directed edge, not merely its energy spectrum.
    dirs = DIRECTIONS @ basis
    world_dir = 2*np.abs(dirs).argmax(axis=-1) + (dirs.sum(axis=-1) == 1)
    flat_edges = encoded_cost(edges.reshape(-1, 6))
    seed_cost = flat_edges[indices(np.array([[0, 0, 0], [1, 0, 0]]))]
    seed_cost = np.take_along_axis(seed_cost, world_dir[:, None, :], axis=2)
    target_cost = np.take_along_axis(flat_edges[target_site], world_dir[bond], axis=1)
    cluster_feat = np.empty((n, 6, 9), np.float32)
    for i, (p, frame) in enumerate(zip(anchor, basis)):
        rel = (((clusters[tuple(p)] - p + 10) % 20) - 10) @ frame.T
        rel = rel.astype(float) - np.array([0.5, 0, 0])
        proj = rel @ DIRECTIONS.T
        cluster_feat[i] = np.column_stack((np.full(6, len(rel)), np.full(6, (rel**2).sum(1).mean()),
                                            proj.max(0), proj.min(0), proj.mean(0), proj.std(0),
                                            (proj > 0).mean(0), (proj == 0).mean(0), (proj > 1).mean(0)))
    # Direction-resolved core barrier and free-site counts + barrier spectrum at both endpoints.
    core_min = encoded_cost(minimum.ravel()[core_idx])
    core_blocks = []
    for d, direction in enumerate(DIRECTIONS):
        proj = (CORE - np.array([.5, 0, 0])) @ direction
        side = core_min[:, proj > 0]
        core_blocks.append(np.column_stack((side.min(1), (side <= 0).sum(1), (side <= 2).sum(1),
                                            (side <= 4).sum(1), (side == 128).sum(1))))
    core_blocks = np.stack(core_blocks, axis=1)
    endpoint = indices(np.array([[0, 0, 0], [1, 0, 0]]))
    seed = np.column_stack((state.ravel()[endpoint], m.ravel()[endpoint], seed_cost.reshape(n, 12),
                            np.sort(seed_cost, axis=2).reshape(n, 12), np.sort(core_min, axis=1)[:, [0, 2, 5, 12, 24, 32]]))
    # The target's own species, m, ALL directional unlike-bond costs, coordination and free-site context;
    # plus summaries over its occupied face. Distinguish geometrical slots via their canonical coordinates.
    cols = [seed[bond], cluster_feat[bond, face], core_blocks[bond, face], face,
            np.concatenate([FACES[d] for d in range(6)])[np.cumsum([0]+list(map(len, FACES)))[:-1][face]+slot],
            count[bond, face], state.ravel()[target_site], m.ravel()[target_site],
            minimum.ravel()[target_site].clip(max=128), distance.ravel()[target_site], density1.ravel()[target_site],
            target_cost, np.sort(target_cost, axis=1)]
    for grid in (state == 1, m, free, minimum <= 2, distance, density1):
        face_sum = (grid.ravel()[target] * valid).sum(axis=2)
        cols.append(face_sum[bond, face]/count[bond, face])
    x = np.column_stack(cols).astype(np.float32)
    return dict(raw=raw, axis=axis.astype(np.uint8), sign=sign.astype(np.int8), anchor=anchor.astype(np.int8),
                target_index=np.where(valid, target, -1), n_occupied=count.astype(np.uint8),
                site=target_site, bond=bond, face=face, x=x)
