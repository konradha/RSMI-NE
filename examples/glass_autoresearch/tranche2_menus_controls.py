"""Independent planted two-bit decision controls; no archive access and no fitted policies."""
import itertools
import numpy as np
from tranche2_menus import OUT, enumerate_menus, exact_scores, report


def solve(count, payoff, labels):
    scores, scale = exact_scores(count, payoff)
    best, menu, lut, policies, _, _, _ = enumerate_menus(scores, labels)
    den = 2 * scale * int(count.sum())
    # Independent raw state-by-action evaluation, not a regrouping of menu optimizer scores.
    brute = max(sum(max(sum(count[v,y] * payoff[t,y,a] for v,h in enumerate(enc) if h==symbol for y in range(4))
                            for a in range(2)) for t in range(2) for symbol in range(labels))
                for enc in itertools.product(range(labels), repeat=4)) / (den / scale)
    realized = [sum(count[v,y] * payoff[t,y,policies[t,lut[v]]] for v in range(4) for y in range(4)) / count.sum() for t in range(2)]
    if best/den != brute or np.mean(realized) != brute:
        raise ArithmeticError('planted achievable, support and independent exhaustive bound disagree')
    return dict(support=best/den, brute=float(brute), achieved=list(map(float,realized)), menu=menu.tolist(), labels=lut.tolist())


if __name__ == '__main__':
    # The face realizes two independent fair bits, visible without noise in the word.
    # Pay 1 for recovering the first/second bit. Each separate one-bit task is perfect.
    pair = np.array([[[int(((y>>t)&1)==a) for a in range(2)] for y in range(4)] for t in range(2)])
    joint = np.eye(4, dtype=int)
    cases = {'fair_bits_one_label': (joint,pair,1,.5), 'fair_bits_two_labels': (joint,pair,2,.75),
             'fair_bits_four_labels': (joint,pair,4,1.), 'same_bit_two_labels': (joint,pair[[0,0]],2,1.),
             'independent_face_four_labels': (np.ones((4,4),dtype=int),pair,4,.5)}
    data = {}
    for name, (count,payoff,n,expected) in cases.items():
        result = solve(count,payoff,n)
        if result['support'] != expected:
            raise ArithmeticError(f'wrong analytic support for {name}')
        data[name] = {'analytic_support': expected, **result}
    report(OUT/'22-planted-controls.json', dict(control='Exact, synthetic fair-bit payoff recovery only; no archive data. Independent exhaustive encoding over all label assignments.', cases=data))
