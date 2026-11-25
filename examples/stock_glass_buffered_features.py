"""Gate 1 shell geometry and descriptors. No archive reads or physical targets.

Run as a script to regenerate the synthetic geometry/feature audit and diagram.
Inputs to features() are exclusively the 150 masked labels in mask() order.
"""
from functools import lru_cache
from itertools import permutations, product
from pathlib import Path
import json

import numpy as np


L = 20
RADII = (2, 4, 5)
FAMILIES = ((3, (1, 1, 1)), (7, (1, 2, 3)))
GROUPS = {
    'species_counts': [0, 3], 'face_species': [3, 21],
    'half_face_species': [21, 93], 'directed_bonds': [93, 201],
    'partial_coordination': [201, 417], 'plaquettes': [417, 513],
    'global_proxy_by_phase': [513, 717], 'local_proxy_best_phase': [717, 933],
}
LAYOUT = {
    'faces': 'normal z, y, x; sign -, + per normal; u,v are remaining axes in ascending (z,y,x) axis index order; u then v -2..2',
    'species_counts': 'raw shell counts V,A,B',
    'face_species': '(face, species V,A,B), fractions of 25',
    'half_face_species': '(face, tangent u,v, side negative,positive excluding zero, species V,A,B), fractions of 10',
    'directed_bonds': '(face, +u,+v, from V,A,B, to V,A,B); joint frequencies on observed unit bonds',
    'partial_coordination': '(face, observed degree d=2,3,4, center V,A,B, occupied neighbors k=0..d); joint fractions of degree-d sites; never the full coordination',
    'plaquettes': '(face, occupied-site bit code 0..15); bits 0,1,2,3 at (u,v),(u,v+1),(u+1,v),(u+1,v+1); real coplanar unit squares only',
    'global_proxy_by_phase': '(family 3,7; canonical normal lex; vacancy phase ascending; readout occupancy,A,B). Mask-projected balanced match = (fraction of template vacancy sites vacant + fraction of template nonvacancy sites occupied/label A/label B)/2.',
    'local_proxy_best_phase': '(family 3,7; face; canonical normal lex; occupancy,A,B). Same balanced match on each face; max over phase separately for each channel, no orientation pooling.',
    'detail': '450 features, index 3*i+k for masked site i=0..149, species k=0 vacancy,1 A,2 B. No products or full-lattice inputs.',
}


@lru_cache(maxsize=3)
def mask(r):
    """Relative integer z,y,x coordinates. Returned shared array is read-only."""
    if r not in RADII:
        raise ValueError(f'r must be one of {RADII}')
    out = np.empty((6, 25, 3), np.int16)
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        for side, sign in enumerate((-1, 1)):
            for j, (u, v) in enumerate(product(range(-2, 3), repeat=2)):
                p = out[2*axis+side, j]
                p[axis] = sign * (3 if r == 2 and max(abs(u), abs(v)) == 2 else r)
                p[a], p[b] = u, v
    out = out.reshape(150, 3)
    out.setflags(write=False)
    return out


def _normals(p, pattern):
    # Signed permutations give the full cubic orbit. Multiplication by a
    # nonzero residue only relabels phases, so make coefficient 0 equal to 1.
    vectors = set()
    for perm in set(permutations(pattern)):
        for signs in product((-1, 1), repeat=3):
            w = [a*b for a, b in zip(perm, signs)]
            inverse = pow(w[0] % p, -1, p)
            vectors.add(tuple((x*inverse) % p for x in w))
    return np.asarray(sorted(vectors), dtype=np.int16)


def _periodic_l1(a, b):
    d = np.abs(a[:, None, :] - b[None, :, :]) % L
    return np.minimum(d, L-d).sum(axis=-1)


@lru_cache(maxsize=3)
def _geometry(r):
    xyz = mask(r)
    adj = _periodic_l1(xyz, xyz) == 1
    # Folded contact patches remain disjoint even in L1 adjacency. All faces
    # have the same graph in local coordinates. Test both facts before reuse.
    graph = adj[:25, :25]
    if not np.array_equal(adj, np.kron(np.eye(6, dtype=bool), graph)):
        raise ValueError('unexpected cross-face or asymmetric patch bonds')
    degree = graph.sum(axis=1)
    grid = np.arange(25).reshape(5, 5)
    bonds = []
    for a, b in ((grid[:-1], grid[1:]), (grid[:, :-1], grid[:, 1:])):
        keep = graph[a, b]
        bonds.append((a[keep], b[keep]))
    corners = np.array([grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]]).reshape(4, -1)
    a, b, c, d = corners
    squares = corners[:, graph[a,b] & graph[a,c] & graph[b,d] & graph[c,d]]
    phases = []
    for p, pattern in FAMILIES:
        residue = xyz @ _normals(p, pattern).T % p
        phases.append((residue[:, :, None] == np.arange(p)).reshape(150, -1))
    projected = np.concatenate(phases, axis=1).reshape(6, 25, -1).astype('f')
    weight = projected.sum(axis=1)
    if not np.all((weight > 0) & (weight < 25)):
        raise ValueError('empty local template class')
    return graph.astype('f'), degree, bonds, squares, projected, weight


def _hist(code, bins):
    """Row/face joint frequency histogram, normalized by actual support."""
    n, f, m = code.shape
    offsets = np.arange(n*f, dtype=np.int64).reshape(n, f, 1) * bins
    return np.bincount((code + offsets).ravel(), minlength=n*f*bins).reshape(n, f, bins).astype('f') / m


def _score(on, total, vac_support, size):
    # on axes: example, [face,] channel, template. Exact integer counts in
    # float32. Nonvacancy positions may be compared to occupancy or A/B.
    off = total[..., :, None] - on
    vacancy_rate = on[..., 0, :] / vac_support
    occupied_rate = 1 - off[..., 0, :] / (size - vac_support)
    return np.stack((vacancy_rate + occupied_rate,
                     vacancy_rate + off[..., 1, :] / (size-vac_support),
                     vacancy_rate + off[..., 2, :] / (size-vac_support)), axis=-1) * np.float32(.5)


def _batch(s, r, base, detail):
    n = len(s)
    graph, degree, bonds, squares, projected, weight = _geometry(r)
    face = s.reshape(n, 6, 25)
    labels = detail.reshape(n, 6, 25, 3)
    np.equal(face[..., None], np.arange(3, dtype=np.uint8), out=labels)
    totals = labels.sum(axis=2)
    base[:, :3] = totals.sum(axis=1)
    base[:, 3:21] = totals.reshape(n, -1) / 25
    plane = labels.reshape(n, 6, 5, 5, 3)
    half = base[:, 21:93].reshape(n, 6, 4, 3)
    for h, sub in enumerate((plane[:, :, :2], plane[:, :, 3:], plane[:, :, :, :2], plane[:, :, :, 3:])):
        half[:, :, h, :] = sub.mean(axis=(2, 3))
    pairs = base[:, 93:201].reshape(n, 6, 2, 9)
    for k, (a, b) in enumerate(bonds):
        pairs[:, :, k, :] = _hist(3*face[:, :, a] + face[:, :, b], 9)
    oc = face != 0
    coordination = (oc.astype('f') @ graph).astype(np.uint8)
    start = 0
    motifs = base[:, 201:417].reshape(n, 6, 36)
    for d in (2, 3, 4):
        bins = 3*(d+1)
        cols = degree == d
        motifs[:, :, start:start+bins] = _hist(face[:, :, cols]*(d+1) + coordination[:, :, cols], bins)
        start += bins
    a,b,c,d = squares
    code = oc[:, :, a] + 2*oc[:, :, b] + 4*oc[:, :, c] + 8*oc[:, :, d]
    base[:, 417:513] = _hist(code, 16).reshape(n, -1)
    # Two scales: keep phase explicitly for whole-shell prediction; retain
    # orientation and face (separate possible domains) in the local scan.
    on = labels.swapaxes(-1, -2) @ projected
    global_scores = _score(on.sum(axis=1), totals.sum(axis=1), weight.sum(axis=0), 150)
    base[:, 513:717] = global_scores.reshape(n, -1)
    local = _score(on, totals, weight, 25)
    start = 0
    pos = 717
    for p, pattern in FAMILIES:
        nori = len(_normals_cached(p, pattern))
        end = start + nori*p
        width = 6*nori*3
        base[:, pos:pos+width] = local[:, :, start:end, :].reshape(n, 6, nori, p, 3).max(axis=3).reshape(n, width)
        pos += width
        start = end


_normals_cached = lru_cache(maxsize=2)(_normals)


def features(s, r):
    """Return conventional (N,933) and incremental (N,450) float32 arrays."""
    s = np.asarray(s)
    if s.ndim != 2 or s.shape[1] != 150 or not np.issubdtype(s.dtype, np.integer):
        raise ValueError('expected an integer (N,150) masked shell array')
    if r not in RADII or (s.size and (s.min() < 0 or s.max() > 2)):
        raise ValueError('invalid mask radius or species labels')
    base = np.empty((len(s), 933), np.float32)
    detail = np.empty((len(s), 450), np.float32)
    for lo in range(0, len(s), 512):
        cut = slice(lo, lo+512)
        _batch(s[cut], r, base[cut], detail[cut])
    return base, detail


PROVENANCE = {
    'finding': 'Neither named equilibrium crystal has a source-attested site-resolved vacancy/A/B basis here. Preferred coordination is not a unit cell, and pure ground-energy constructions need not be the mixed finite-temperature phases. No authenticity claim for either proxy.',
    'sources_checked': [
        ['https://doi.org/10.1103/PhysRevLett.125.065501', 'Nishikawa & Hukushima, PRL 2020; Hamiltonian and phase names.'],
        ['https://arxiv.org/html/2003.02872v2/full_pd_1.png', 'Fig. 1 examined directly: phase diagram in T/A and composition at density 0.75; no structure inset, reciprocal vectors, layer registry or species decoration. Its caption identifies 3 and 5 as preferred coordinations, not periods.'],
        ['https://arxiv.org/html/2003.02872v2', 'Author manuscript and supplemental items 1–3: low-T dynamical clusters, estimation of dynamical temperature, finite-size relaxation. S2 colors changed sites by dynamic cluster, not crystal species.'],
        ['https://doi.org/10.1103/PhysRevLett.132.067101', 'Nishikawa & Berthier follow-up on collective relaxation; the figures on mobile defects are not an ordered reference. https://arxiv.org/html/2307.08110v1'],
        ['https://yoshihikonishikawa.github.io/', 'Author publication index does not link a crystal dataset for the two lattice-glass papers.'],
        ['https://api.github.com/users/YoshihikoNishikawa/repos?per_page=100', 'Public repository names: HighFive, HukushimaNishikawa_MCMC_book_2025, partycls, SoftDisks, StationaryBootstrap, stresampling, personal site. No identified author-deposited lattice-crystal configurations. Targeted Zenodo/Figshare/Dryad/GitHub title/author searches did not identify a crystal dataset. No private data or author correspondence obtained; this is not proof none exists.'],
        ['https://arxiv.org/abs/cond-mat/0409234', 'Related finite-energy BM model penalizes excessive coordination; not the binary quadratic Hamiltonian. Not a source of authenticated Crystal3/5 decorations.'],
        ['https://arxiv.org/abs/1706.03589', 'Lee-code background only. A radius-one perfect vacancy packing does not identify the published equilibrium Crystal5.'],
    ],
    'proxy3': 'Vacancies on w·t=phase mod 3; w in cubic orbit of (1,1,1). All off-vacancy sites A would have exactly 3 occupied axial neighbors at density 2/3. Also score all-B and occupancy-only. No mixed basis specified by available evidence.',
    'proxy7': 'Vacancies on w·t=phase mod 7; w in cubic orbit of (1,2,3). The six axial steps have the six distinct nonzero residues, so each occupied site has one vacant axial neighbor. All-B realizes its preferred 5 at density 6/7. Also score all-A and occupancy-only.',
    'scan': 'All signed permutations (includes reflections). Normals differing by a nonzero scalar modulo p have identical translated vacancy sets: use unique representatives with first coefficient 1 and all p phases. Coordinates t are signed offsets from the anchor in z,y,x order; no absolute origin or aligned bulk lattice. Neither period tiles L20. Compare local fragments, not full periodic states or phase stability.',
    'decision_consequence': 'Even if these two trial packings are distinguishable on the mask, the authenticated-crystal control is unavailable and Gate 1 cannot pass.',
}


def _synthetic_invariance(r):
    """Change every hidden lattice value at each origin; never fit a head."""
    rng = np.random.default_rng(67000+r)
    original = rng.integers(3, size=(L, L, L), dtype=np.uint8)
    origins = [(0, 0, 0), (19, 0, 19), (10, 10, 10), (1, 19, 0)]
    pair = [[], []]
    hidden_changed = []
    for origin in origins:
        ids = tuple(((mask(r) + origin) % L).T)
        perturbed = (original + rng.integers(1, 3, size=original.shape, dtype=np.uint8)) % 3
        perturbed[ids] = original[ids]
        hidden_changed.append(int(np.count_nonzero(original != perturbed)))
        pair[0].append(original[ids])
        pair[1].append(perturbed[ids])
    (b0, d0), (b1, d1) = [features(np.stack(a), r) for a in pair]
    def probs(x, weights):
        y = x @ weights
        y = np.exp(y-y.max(axis=1, keepdims=True))
        return y/y.sum(axis=1, keepdims=True)
    probe_diffs = {}
    for name, x0, x1 in [('baseline', b0, b1), ('rich', np.hstack((b0,d0)), np.hstack((b1,d1)))]:
        w = rng.normal(0, .01, (x0.shape[1], 210)).astype('f')
        probe_diffs[name] = float(np.abs(probs(x0,w)-probs(x1,w)).max())
    differences = {'base': float(np.abs(b0-b1).max()), 'detail': float(np.abs(d0-d1).max())}
    return {'origins_zyx': origins, 'hidden_sites_changed': hidden_changed,
            'feature_max_abs_change': differences, 'synthetic_210_softmax_max_abs_change': probe_diffs,
            'prediction_scope': 'Fixed synthetic weights only. Bit-identical inputs imply identical outputs for any deterministic frozen heads taking only these features, including any deterministic compression.',
            'passed': all(v == L**3-150 for v in hidden_changed) and not any([*differences.values(), *probe_diffs.values()])}


def _arm_audit(r):
    coords = mask(r)
    star = np.vstack((np.zeros(3, dtype=int), np.eye(3, dtype=int), -np.eye(3, dtype=int)))
    graph, deg, bonds, squares, projected, weight = _geometry(r)
    variants = projected.reshape(150, -1).T.astype(bool)
    distance = (variants[:, None] != variants).sum(axis=-1)
    cross_min = int(distance[:12, 12:].min())
    families = {}
    for (p, pat), sl in zip(FAMILIES, (slice(0, 12), slice(12, 68))):
        support = weight[:, sl]
        templates = variants[sl]
        # Feed each perfect projection in both label assignments through the
        # real extraction path. The registered target phase / species, not
        # merely some independently rephased plane, must score perfectly.
        best = []
        for label in (1, 2):
            match, _ = features((~templates).astype(np.uint8)*label, r)
            match = match[:, 513:717].reshape(len(templates), 68, 3)
            best.append(float(match[np.arange(len(templates)), np.arange(68)[sl], label].min()))
        families[str(p)] = {
            'canonical_normals_zyx': _normals(p, pat).tolist(), 'phases_per_normal': p,
            'distinct_projected_vacancy_sets': len(np.unique(templates, axis=0)),
            'candidate_templates': len(templates),
            'shell_vacancy_support_min_max': [int(templates.sum(axis=1).min()), int(templates.sum(axis=1).max())],
            'face_vacancy_support_min_max': [int(support.min()), int(support.max())],
            'within_family_min_hamming_sites': int(distance[sl,sl][np.triu_indices(len(templates),1)].min()),
            'perfect_template_min_score_A_then_B': best,
        }
    shell_to_star = _periodic_l1(coords % L, star % L)
    # Exhaust all 8000 translations. Each relative Cartesian-axis distance
    # only depends on the translation component along that axis: exhausting
    # 20 offsets per axis certifies their full Cartesian product.
    for axis in range(3):
        ref = np.abs(coords[:,None,axis] - star[None,:,axis])
        for shift in range(L):
            a = (coords[:,None,axis]+shift) % L
            b = (star[None,:,axis]+shift) % L
            d = np.abs(a-b)
            if not np.array_equal(np.minimum(d,L-d), ref):
                raise ValueError('unexpected wrapped axis distance')
    min_dist = int(shell_to_star.min())
    if len(np.unique(coords % L, axis=0)) != 150 or min_dist != r-1:
        raise ValueError('invalid geometry: count or star gap')
    return {
        'mask_offsets_zyx': coords.tolist(), 'distinct_sites_periodic': 150, 'sites_per_face': [25]*6,
        'min_periodic_L1_star_shell': min_dist,
        'star_shell_pairs_by_distance': {int(v):int((shell_to_star==v).sum()) for v in np.unique(shell_to_star)},
        'direct_star_shell_unit_bonds': int((shell_to_star==1).sum()),
        'hamiltonian_term_max_L1_diameter': 2, 'disjoint_from_all_star_touching_terms': min_dist > 2,
        'wrap_check': 'All 20 shifts independently on all three axes; all 8000 origins have identical minimum-image separations.',
        'shell_unit_bonds': 6*int(graph.sum()/2), 'bonds_per_face_by_tangent_u_v': [len(a) for a,b in bonds],
        'degree_hist_per_face': {int(i):int((deg==i).sum()) for i in np.unique(deg)},
        'supported_unit_plaquettes': 6*squares.shape[1], 'fully_supported_six_neighbor_stars': int(6*(deg==6).sum()),
        'template_fragments': families, 'between_proxy_family_min_hamming_sites': cross_min,
        'formally_distinct_proxy_families_on_mask': cross_min > 0,
        'perturbation': _synthetic_invariance(r),
    }


def audit(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arms = {str(r): _arm_audit(r) for r in RADII}
    report = {
        'authenticated_crystals': False, 'masked_invariance': all(a['perturbation']['passed'] for a in arms.values()),
        'scope': 'Geometry, projected templates and synthetic labels only. No physical configuration, target, training split or fitted score accessed.',
        'crystal_provenance': PROVENANCE, 'L': L, 'columns_half_open': GROUPS, 'layout_within_groups': LAYOUT,
        'width_conventional': 933, 'width_detail': 450, 'width_rich': 1383, 'arms': arms,
    }
    (output/'mask-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    fig = plt.figure(figsize=(12, 4.5))
    colors = ['#1b7098', '#b57023', '#51916a']
    for i,r in enumerate(RADII):
        ax = fig.add_subplot(1,3,i+1, projection='3d')
        xyz = mask(r)[:,::-1].reshape(6,25,3)
        for face, points in enumerate(xyz):
            ax.scatter(*points.T, s=13, color=colors[face//2], alpha=.85)
        for idx in range(3):
            axis = np.eye(3)[idx]
            ax.plot(*np.array([-axis,axis]).T, color='black', lw=1.4)
            ax.scatter(*np.array([-axis,axis]).T, color='black', s=14)
        ax.scatter(0,0,0, color='black', s=14)
        label = ('Contact', 'Primary', 'Secondary')[i]
        gap = arms[str(r)]['min_periodic_L1_star_shell']
        ax.set(xlim=(-5,5), ylim=(-5,5), zlim=(-5,5), xticks=(-5,0,5), yticks=(-5,0,5), zticks=(-5,0,5),
               xlabel='x', ylabel='y', zlabel='z', title=f'{label}: min L1 = {gap}')
        ax.set_box_aspect((1,1,1))
    fig.suptitle('Observed sites only: 25 per face × 6 faces; central star in black')
    fig.subplots_adjust(left=.015,right=.97,wspace=.12,bottom=.07,top=.89)
    fig.savefig(output/'masks.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    audit(Path(__file__).resolve().parents[1]/'results'/'buffered-static-gate1')
