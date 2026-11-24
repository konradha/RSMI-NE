"""Archive-only Blackwell decision witnesses and a finite four-state face MI certificate."""
import argparse
import itertools
import math
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import t
from threadpoolctl import threadpool_limits

from face_optimizer import Channel, digest, load_map, NJ, NV, NY, K


NAMES = ('discrete', 'affine')
ACTIONS = np.array(list(itertools.product(range(K), repeat=K)))
UTILITY_FLOOR = 0.001
ALPHA = 0.05 / 30  # one-sided x two orderings x fifteen items


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inputs(a):
    meta = json.loads((a.bank / 'meta.json').read_text())
    path = a.bank / 'train.npy'
    if meta['index'] != 9 or digest(path) != meta['train_sha256']:
        raise ValueError('requires original cold training bank')
    counts = np.load(path, mmap_mode='r', allow_pickle=False)
    masses = counts.sum(axis=2, dtype=np.int64)
    if (counts.shape != (NJ, NV, NY) or not np.all(masses == masses[0]) or
            masses[0].sum() != meta['train_parents'] * meta['anchors_per_parent']):
        raise ValueError('invalid training counts')
    with np.load(a.maps, allow_pickle=False) as data:
        maps = [load_map(data, name) for name in NAMES]
    return meta, counts, maps


def joint(counts, maps):
    # encoder, direction, unknown word, observed symbol; sums to one within direction
    z = np.stack([np.stack([counts[:, h == k, :].sum(axis=1, dtype=np.int64)
                            for k in range(K)], axis=-1) for h in maps])
    return z / z.sum(axis=(-1, -2), keepdims=True)


def deficiency(src, dst):
    """Minimum prior-weighted TV to dst from src@G, plus [0,1] dual payoff.

    min sum slack subject to dst-src@G <= slack. Conditional prior identical,
    so positive-part mass is TV. HiGHS marginals of the covering constraints
    give U(y,action), separating the target identity decision from *all* source
    decisions. The dual gap below is checked against enumerated optimal actions.
    """
    y = len(src)
    op = sparse.kron(src, sparse.eye(K), format='csr')
    cons = sparse.hstack([-op, -sparse.eye(y*K)], format='csr')
    simplex = sparse.hstack([sparse.kron(sparse.eye(K), np.ones((1, K))),
                             sparse.csr_matrix((K, y*K))], format='csr')
    fit = linprog(np.r_[np.zeros(K*K), np.ones(y*K)], A_ub=cons, b_ub=-dst.ravel(),
                  A_eq=simplex, b_eq=np.ones(K), method='highs-ds',
                  options={'dual_feasibility_tolerance': 1e-10, 'primal_feasibility_tolerance': 1e-10})
    if not fit.success:
        raise ValueError(fit.message)
    g = fit.x[:K*K].reshape(K, K)
    u = np.clip(-fit.ineqlin.marginals.reshape(y, K), 0, 1)
    residual = float(np.abs(src @ g - dst).sum() / 2)
    source = src.T @ u
    target = dst.T @ u
    source_best = float(source.max(axis=1).sum())
    gain = float(np.trace(target) - source_best)
    enumeration = source[np.arange(K), ACTIONS].sum(axis=1).max()
    if (np.min(g) < -1e-9 or np.max(np.abs(g.sum(axis=1)-1)) > 1e-9 or
            max(abs(gain-residual), abs(enumeration-source_best), abs(residual-fit.fun)) > 1e-8):
        raise ArithmeticError('inconsistent primal LP, dual decision certificate or enumeration')
    return g, u, residual, gain


def witness(a):
    meta, counts, maps = inputs(a)
    z = joint(counts, maps)
    payoff = np.empty((2, NJ, NY, K))  # index: proposed superior (target)
    garble = np.empty((2, NJ, K, K))  # competitor -> target
    policies = np.empty((2, 2, NJ, K), dtype=np.uint8)  # target, evaluated encoder
    train_values = np.empty((2, 2, NJ))
    residual = np.empty((2, NJ))
    dual = np.empty_like(residual)
    for target in range(2):
        for j in range(NJ):
            g, u, residual[target, j], dual[target, j] = deficiency(z[1-target, j], z[target, j])
            garble[target, j], payoff[target, j] = g, u
            for e in range(2):
                score = z[e, j].T @ u
                policy = score.argmax(axis=1)
                policies[target, e, j] = policy
                train_values[target, e, j] = score[np.arange(K), policy].sum()
    # A relabel is lossless under a stochastic garbling. Independence of Y carries
    # no action information (all observations give the same prior). Both are executable.
    identity_null = deficiency(z[0, 0], z[0, 0, :, ::-1])[2]
    indep = z[0, 0].sum(axis=1)[:, None] * z[0, 0].sum(axis=0)
    u0 = payoff[0, 0]
    independent_null = float((indep.T @ u0).max(axis=1).sum() - (indep.sum(axis=1) @ u0).max())
    if max(abs(independent_null), identity_null) > 1e-8:
        raise ArithmeticError('control failure')
    freeze = a.out / '09-witness.npz'
    np.savez_compressed(freeze, payoff=payoff, policies=policies, garbling=garble,
                        maps=np.stack(maps), training_joint=z)
    words = counts.sum(axis=1, dtype=np.int64)
    report = dict(question='Do two bounded four-action decisions favor opposite face encoders?',
        precedent='https://arxiv.org/abs/1701.07602', distinction='Different operational utilities, not the previous face log score.',
        beta=meta['beta'], parents=meta['train_parents'], aggregate_only=True,
        anchors_per_parent=meta['anchors_per_parent'], words_per_direction=words.sum(axis=1).tolist(),
        supported_words=(words > 0).sum(axis=1).tolist(), smallest_word_count=words.min(axis=1).tolist(),
        train_sha256=meta['train_sha256'], encoder_file_sha256=digest(a.maps), witness_sha256=digest(freeze),
        axis_order=meta['axis_order'], encoder_order=NAMES,
        payoff_axes=['payoff naming target encoder', 'registered direction (observed)', 'unknown ternary word', 'action'],
        policy_axes=['payoff naming target encoder', 'evaluated encoder', 'registered direction', 'observed symbol'],
        garbling_axes=['target encoder (approximated from opposite encoder)', 'registered direction', 'source symbol', 'target symbol'],
        tv_by_target_and_direction=residual.tolist(), dual_identity_gain=dual.tolist(),
        train_optimal_utilities=train_values.tolist(), train_gains=[float((train_values[e,e]-train_values[e,1-e]).mean()) for e in range(2)],
        payoff_range=[float(payoff.min()), float(payoff.max())],
        nulls=dict(relabel_TV=identity_null, independent_information_value=independent_null,
                   max_primal_dual_error=float(np.abs(dual-residual).max())),
        evaluation_cohort='Existing b09 native validation face groups 0..15 ONLY (1024 nonreserved parents), if authorized after public checksum lock; exclude groups 16..31, raw trajectories, test, other cohorts and repeat.',
        denominator='Per opportunity, uniform six registered directions (known to the decision maker) and equal-size parents; utility in [0,1].',
        practical_margin=UTILITY_FLOOR, one_sided_alpha_each=ALPHA,
        contrast='For each target, use frozen target policy; upper-bound competing optimal value by allowing competitor to choose its optimal action per symbol and direction separately inside EACH validation group. Jensen: expected group maximum >= population maximum. Paired group difference estimates a lower bound, not a plug-in optimality gap.',
        inference='16 disjoint groups of 64 whole parents; within-group optimize competitor over all deterministic policies (also dominates randomization). One-sided paired Student-t lower limit with 15 degrees of freedom across independent groups, approximate; 0.05/(2*15). Both lower bounds > 0.001 required for practical reversal. Also report heldout aggregate maximized opponent, fixed-policy gains, and TV of frozen garblings (not proof of deficiency).',
        limitations='Cold training payoffs are data-adaptive; no population discovery from these fits. Parents are aggregated on disk, so training-cluster SE not identifiable. Opponent oracle is deliberately conservative; existing validation has historical use for OTHER outcomes.',
        decision='Freeze both directional payoff families and policies. Only separate authorized evaluation can test these train witnesses.')
    dump(a.out / '09-lock.json', report)
    print(json.dumps({k:report[k] for k in ('train_gains','witness_sha256','smallest_word_count')}), flush=True)


def evaluate(a):
    # NB: no validation.npy open, hash or probing on the witness/certificate paths.
    lock = a.out / '09-lock.json'
    if not a.lock_sha256 or digest(lock) != a.lock_sha256:
        raise ValueError('published decision-lock SHA256 required')
    cfg = json.loads(lock.read_text())
    freeze = a.out / '09-witness.npz'
    if digest(freeze) != cfg['witness_sha256']:
        raise ValueError('witness checksum mismatch')
    if (a.out / '09-evaluation.json').exists():
        raise FileExistsError('this contrast has already been issued')
    meta = json.loads((a.bank / 'meta.json').read_text())
    path = a.bank / 'validation.npy'
    if meta['index'] != 9 or meta['train_sha256'] != cfg['train_sha256']:
        raise ValueError('requires unchanged cold validation bank metadata')
    with np.load(freeze, allow_pickle=False) as f:
        payoff, policy, g, maps = (f[k] for k in ('payoff','policies','garbling','maps'))
    # Slice at the mmap boundary: no reading or hashing of reserved packet bytes.
    bank = np.load(path, mmap_mode='r', allow_pickle=False)[:16]
    groups = len(bank)
    if groups != 16 or meta['groups'] != 32 or meta['validation_parents'] != 2048:
        raise ValueError('not the prelocked parent groups')
    n = 64 * meta['anchors_per_parent']
    by_group = []
    for packet in bank:
        if not np.all(packet.sum(axis=(1, 2), dtype=np.int64) == n):
            raise ValueError('incorrect denominator')
        by_group.append(joint(packet, maps))
    z = np.stack(by_group)
    mean = z.mean(axis=0)
    value = np.einsum('nejyh,tjya->ntejha', z, payoff, optimize=True)
    fixed = np.take_along_axis(value, policy[None, ..., None], axis=-1)[..., 0].sum(axis=-1)
    oracle = value.max(axis=-1).sum(axis=-1)
    tv = np.empty((2, NJ))
    gains = []
    records = {}
    for e in range(2):
        tv[e] = np.abs(mean[1-e] @ g[e] - mean[e]).sum(axis=(1,2)) / 2
        candidate = fixed[:, e, e].mean(axis=1)
        opponent = oracle[:, e, 1-e].mean(axis=1)
        gap = candidate-opponent
        gains.append(gap)
        se = float(gap.std(ddof=1) / np.sqrt(groups))
        lcb = gap.mean() - t.ppf(1-ALPHA, groups-1)*se
        records[NAMES[e]] = dict(candidate_fixed=float(candidate.mean()), opponent_group_oracle=float(opponent.mean()),
            opponent_pooled_oracle=float(value.mean(axis=0)[e,1-e].max(axis=-1).sum(axis=-1).mean()),
            gain_over_frozen_opponent=float((candidate-fixed[:,e,1-e].mean(axis=1)).mean()),
            conservative_gain=float(gap.mean()), block_se=se, adjusted_lcb=float(lcb),
            MDE_80pct_normal_approx=float((t.ppf(1-ALPHA,groups-1)+0.841621)*se), exceeds_floor=bool(lcb>UTILITY_FLOOR))
    fp=a.out/'09-validation-groups.npz'
    np.savez_compressed(fp, group=np.arange(groups), parents_per_group=64, joint=z, direction_fixed=fixed,
                        direction_oracle=oracle, target_gain=np.array(gains))
    decision=('Opposite locked decisions both clear the practical floor against an upper bound on each opponent.'
              if all(x['exceeds_floor'] for x in records.values()) else 'Inconclusive for two-way practical reversal; no order inferred.')
    dump(a.out/'09-evaluation.json',dict(lock_sha256=a.lock_sha256, bank_sha256_from_metadata=meta['validation_sha256'],
        blocks_sha256=digest(fp), parents=1024, groups=groups, packet_indices=list(range(16)), parents_per_group=64,
        opportunities_per_direction=n*groups, registered_directions=NJ, results=records,
        heldout_frozen_garbling_tv=tv.tolist(), alpha=ALPHA, floor=UTILITY_FLOOR, decision=decision,
        limitations='Parent-group t inference, not an exact finite-sample guarantee. Prior-specific cold channel with direction supplied; not dominance for arbitrary priors, temperatures or observables. Bank contains per-group, not per-parent counts.'))
    print(json.dumps(records),flush=True)


# Absolute error per normalized binary conditional-entropy interval, bits.
# The cost uses three xlogx terms. IEEE binary64 round-to-nearest; no libm log
# needed. Relative error in p from int64 conversion/division <= 4u. On 0<p<=1,
# |p log2 p| < 1 and p|d(p log2 p)/dp| < 2. Frexp's power-of-two shift is
# exact. For z=(m-1)/(m+1), 1<=m<2, z<=1/3, the atanh-series omitted tail
# after z^31 is <=2*(1/3)^33/(33*(1-1/9)). Horner has 31 rounded operations
# on positive terms and relative error < 40u; normalizing by ln 2, adding the
# integer exponent and multiplying by p add < 10u absolute. Even 100u per
# xlogx plus three add/sub roundings is < 2e-13 (u=2^-53). All integer input,
# prefix sums and prefix differences are exact and positive, totals < 2^54.
INTERVAL_ERR = 2e-13
LINV2 = float.fromhex('0x1.71547652b82fep+0')


def xlogprob(x, total):
    p = np.asarray(x, dtype=np.float64) / float(total)
    m, exponent = np.frexp(p)
    m = np.where(p == 0, 1., 2*m)
    z = (m-1)/(m+1)
    zz = z*z
    poly = 1/31
    for odd in range(29, 0, -2):
        poly = 1/odd + zz*poly
    return p*((exponent-1) + (2*LINV2)*z*poly)


def cost(left, right, total):
    return xlogprob(left+right,total)-xlogprob(left,total)-xlogprob(right,total)


def sorted_atoms(left, right):
    # Sort by P(left | tile, reached node); exact adjacent-integer checks
    # protect nearly equal fractions and permit pooling exactly equal rows.
    w = left+right
    order = np.argsort(left.astype(np.longdouble)/w)
    cross = left[order].astype(object)
    mass = w[order].astype(object)
    l, r = cross[:-1]*mass[1:], cross[1:]*mass[:-1]
    if np.any(l > r):
        from fractions import Fraction
        order = np.array(sorted(range(len(w)), key=lambda i: Fraction(int(left[i]),int(w[i]))))
        cross, mass = left[order].astype(object), w[order].astype(object)
        l, r = cross[:-1]*mass[1:], cross[1:]*mass[:-1]
    starts = np.r_[0, np.flatnonzero(l != r)+1]
    return np.add.reduceat(left[order], starts), np.add.reduceat(right[order], starts)


def optimal_cost(left, right):
    """Minimize H(branch|bin), at most four bins, with roundoff-aware Monge DP.

    For sorted binary posteriors an optimal quantizer has contiguous bins.
    For each r and k, F[k,r]=min_s F[k-1,s]+C(s,r), s in [k-1,r).
    C is Monge (binary entropy impurity on a likelihood-ratio ordering);
    the leftmost exact minimizer is nondecreasing in r. At each midpoint use
    the FULL interval of possible minimizers, not a fragile floating argmin,
    to constrain both halves: a true minimizer is within 2*k*err of the
    computed minimum. Thus a minimum survives pruning at every row and the
    returned cost is within k*err of the exact minimum. See cited theorems.
    """
    left, right = sorted_atoms(left, right)
    n = len(left)
    a, b = (np.r_[0, x.cumsum(dtype=np.int64)] for x in (left, right))
    total = int(a[-1]+b[-1])
    prev = cost(a,b,total)
    candidates = n
    for k in range(2, min(K,n)+1):
        current = np.full(n+1, np.inf)
        lo, hi, a0, a1 = (np.array([v]) for v in (k,n,k-1,n-1))
        while lo.size:
            mid = (lo+hi)//2
            lengths = np.minimum(a1,mid-1)-a0+1
            starts = np.r_[0,lengths.cumsum()[:-1]]
            row = np.repeat(np.arange(len(mid)),lengths)
            s = np.arange(len(row)) + (a0-starts)[row]
            ends = mid[row]
            values = prev[s]+cost(a[ends]-a[s], b[ends]-b[s], total)
            candidates += len(values)
            best = np.minimum.reduceat(values,starts)
            current[mid] = best
            possible = values <= best[row]+(2*k+1)*INTERVAL_ERR
            lowarg = np.minimum.reduceat(np.where(possible,s,n),starts)
            higharg = np.maximum.reduceat(np.where(possible,s,-1),starts)
            left_half, right_half = lo < mid, mid < hi
            lo, hi, a0, a1 = (np.r_[lo[left_half],mid[right_half]+1],
                              np.r_[mid[left_half]-1,hi[right_half]],
                              np.r_[a0[left_half],lowarg[right_half]],
                              np.r_[higharg[left_half],a1[right_half]])
        prev = current
    return float(prev[-1]), n, candidates


def tree_nodes(lo, hi):
    if hi-lo > 1:
        mid = (lo+hi)//2
        yield lo, mid, hi
        yield from tree_nodes(lo,mid)
        yield from tree_nodes(mid,hi)


def control_dp():
    l = np.array([2,5,11,7,2,8,3], dtype=np.int64)
    r = np.array([2,3,1,7,6,2,5], dtype=np.int64)
    z = l+r
    # Enumerate *all* assignments, not just intervals; tiny finite channel.
    all_maps = np.array(list(itertools.product(range(K),repeat=len(l))))
    loss = np.zeros(len(all_maps))
    for k in range(K):
        hits = (all_maps == k)
        loss += cost(hits@l,hits@r,int(z.sum()))
    dp = optimal_cost(l,r)[0]
    null = optimal_cost(2*l,3*l)[0]-float(cost(2*l.sum(),3*l.sum(),5*l.sum()))
    if max(abs(dp-loss.min()),abs(null)) > 2*K*INTERVAL_ERR:
        raise ArithmeticError('binary quantizer controls failed')
    return dict(brute_assignments=len(loss), unrestricted_min_bits=float(loss.min()),
                interval_dp_min_bits=dp, identical_posteriors_information_bits=-null)


def certificate(a):
    meta, counts, maps = inputs(a)
    useful = .001 / np.log(2)  # 0.001 nat/opportunity, same named face scale
    controls = control_dp()
    smoothed = Channel(counts, True)
    direct_lower = smoothed.fit(maps[0])[2]
    observed_tiles = int(np.count_nonzero(counts[0].sum(axis=1)))
    # Smoothing in face_optimizer.Channel: +1 marginal draw per (direction,tile).
    # Prior draw for y is (2*c[j,:,y].sum()+1)/(2*D+81). This common
    # denominator makes all the integer prefix sums below exact, including the
    # optimizer's positive rows for unobserved tiles.
    words = counts.sum(axis=1,dtype=np.int64)
    D = int(words[0].sum())
    den = 2*D+NY
    per_dir = (D+NV)*den
    if per_dir >= 2**54 or np.iinfo(np.int64).max <= NJ*per_dir:
        raise ValueError('numerical error contract exceeded')
    table = counts.astype(np.int64)*den + (2*words[:,None,:]+1)
    prefix = np.concatenate([np.zeros((NJ,NV,1),dtype=np.int64), table.cumsum(axis=2)],axis=2)
    del table
    norm = NJ*per_dir
    if not np.all(prefix[:,:,-1].sum(axis=1)==per_dir):
        raise ArithmeticError('integer smoothing mass mismatch')
    nodes = []
    for j in range(NJ):
        for lo,mid,hi in tree_nodes(0,NY):
            mass = int((prefix[j,:,hi]-prefix[j,:,lo]).sum())
            nodes.append((mass,j,lo,mid,hi))
    nodes.sort(key=lambda v:(-v[0],*v[1:]))
    records, full, lower, tree = [], [], [], []
    for idx,(mass,j,lo,mid,hi) in enumerate(nodes):
        l = prefix[j,:,mid]-prefix[j,:,lo]
        r = prefix[j,:,hi]-prefix[j,:,mid]
        prior = float(cost(l.sum(),r.sum(),mass))
        full_node = prior-math.fsum(cost(l,r,mass))
        lbin,rbin = (np.array([x[maps[0]==k].sum() for k in range(K)]) for x in (l,r))
        feasible = prior-math.fsum(cost(lbin,rbin,mass))
        weight = mass/norm
        # Full tile/feasible sum exact up to errors on each term. All totals
        # finalized with fsum. Extra slack covers final multiplications/sums.
        full_up = weight*(full_node+(NV+2)*INTERVAL_ERR)+1e-15
        feasible_low = weight*(feasible-(K+2)*INTERVAL_ERR)-1e-15
        full.append(full_up)
        lower.append(feasible_low)
        row = dict(direction=j, lo=lo, mid=mid, hi=hi, probability=weight,
                   feasible_weighted_bits=weight*feasible, full_tile_upper_bits=full_up)
        up = full_up
        if idx < a.nodes:
            opt, atoms, cells = optimal_cost(l,r)
            relaxed = prior-opt
            up = min(up,weight*(relaxed+(K+2)*INTERVAL_ERR)+1e-15)
            row.update(distinct_posteriors=atoms, candidate_costs=cells,
                       optimal_four_bin_weighted_bits=weight*relaxed)
        if up < feasible_low:
            raise ArithmeticError('feasible information exceeds certified upper bound')
        row['upper_bits']=up
        tree.append(up)
        records.append(row)
    feasible = math.fsum(lower)-1e-14
    full_up = math.fsum(full)+1e-14
    upper = min(full_up, math.fsum(tree)+1e-14)
    controls['direct_optimizer_feasible_bits']=direct_lower
    controls['direct_optimizer_full_tile_bits']=smoothed.full
    controls['raw_chain_rule_feasible_minus_direct_bits']=math.fsum(v['feasible_weighted_bits'] for v in records)-direct_lower
    if (not feasible <= direct_lower <= upper or not 0 <= full_up-smoothed.full < 1e-6 or
            abs(controls['raw_chain_rule_feasible_minus_direct_bits']) > 1e-10):
        raise ArithmeticError('independent face_optimizer smoothing/chain-rule mismatch')
    levels = {}
    for n in sorted(set(min(x,a.nodes,len(nodes)) for x in (0,6,24,96,len(nodes)))):
        levels[str(n)] = math.fsum(tree[:n]+full[n:])+1e-14
    npz = a.out/'12-node-ledger.npz'
    keys = records[0].keys() | {'distinct_posteriors','candidate_costs','optimal_four_bin_weighted_bits'}
    np.savez_compressed(npz, **{k:np.array([row.get(k,np.nan) for row in records]) for k in sorted(keys)})
    solved = min(max(0,a.nodes),len(nodes))
    dump(a.out/'12-certificate.json',dict(question='How much train MI could ANY shared four-state face map still gain?',
        distinction='Bounds finite optimum, neither another optimizer run nor a predictive transfer score.',
        precedents=['https://arxiv.org/abs/1107.5637','https://arxiv.org/abs/1901.01659'],
        scope='Named cold finite smoothed channel; deterministic numerical bounds, not a population confidence interval. No validation required or used. A stochastic shared encoder cannot exceed the maximum over deterministic extremes by convexity of MI in the encoder.',
        parents=meta['train_parents'], anchors_per_parent=meta['anchors_per_parent'], directions=NJ,
        observed_per_direction=D, positive_tiles=observed_tiles, smoothed_tiles=NV,
        positive_face_words_per_direction=(words>0).sum(axis=1).tolist(),
        real_nonzero_tile_word_cells=int(np.count_nonzero(counts)), total_cells=int(counts.size),
        table_sha256=meta['train_sha256'], maps_sha256=digest(a.maps), nodes_sha256=digest(npz),
        smoothing='Exactly face_optimizer.Channel(True): empirical directional word marginal with 0.5 added to every word; normalized pseudocount of one per tile/direction. Integer numerator=(2D+81)c_jvy+2c_jy+1.',
        bound='Balanced recursive floor-midpoint splits of increasing little-endian ternary words [0,81). Chain rule on each known direction. Independently relax the shared four-state map at every node/direction. Sort exact binary posteriors, merge identical atoms, Monge interval DP. Full-tile branch MI upper-bounds unsolved nodes. Minimum with full-input bound. Marginal J is independent of tile by construction.',
        numerical='Integer pooled masses and prefix differences; exact rational ordering checks. Each interval entropy uses frexp and degree-31 atanh polynomial with conservative 2e-13 absolute bit allowance; interval-aware monotone candidate brackets preserve ties/roundoff, including ambiguity in previous DP rows. Weighted allowances carried into both endpoints. Assumes round-to-nearest IEEE binary64, no fast-math.',
        controls=controls, nodes=len(nodes), nodes_solved=solved, full_tile_remainder_nodes=len(nodes)-solved,
        weighted_full_tile_remainder_bits=math.fsum(full[solved:]),
        total_dp_candidates=sum(row.get('candidate_costs',0) for row in records),
        high_mass_first_upper_bits=levels, feasible_map='frozen discrete_lut', feasible_lower_bits=feasible,
        full_input_upper_bits=full_up, tree_upper_bits=upper, upper_gap_bits=upper-feasible,
        upper_gap_nats=(upper-feasible)*np.log(2), useful_gain_bits=useful, useful_gain_nats=.001,
        decision=('No useful gain possible on this finite table.' if upper-feasible < useful else
                  'Relaxation too wide to rule out a useful gain; upper gap is NOT an attainable improvement.'),
        limitations='Shared map constraint removed; especially loose across many low-level branches or different directions. No new physical states, no held-out physical optimality statement, no empirical sampling error needed for fixed-table arithmetic.'))
    print(json.dumps(dict(nodes=solved,lower=feasible,upper=upper,gap_bits=upper-feasible)),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('witness', 'evaluate', 'certificate'))
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--maps', type=Path, default=Path('results/l20-autoresearch/faces/optimized.npz'))
    ap.add_argument('--out', type=Path, default=Path('results/l20-autoresearch/fifteen'))
    ap.add_argument('--lock-sha256')
    ap.add_argument('--nodes', type=int, default=NJ*(NY-1), choices=range(NJ*(NY-1)+1), metavar='0..480')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        {'witness': witness, 'evaluate': evaluate, 'certificate': certificate}[args.mode](args)


if __name__ == '__main__':
    main()
