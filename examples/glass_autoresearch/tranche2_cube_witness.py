"""Independent positive-coefficient check; consumes saved development counts only.

Unlike the production Fourier projection this uses independent binomial PMFs,
positive recurrence/convolution, and decimal arithmetic. Witness is the FIRST
presampled eligible opportunity of the first development parent of each stencil;
this is not a search for a striking example. Also reports a stable KL check on
all saved cube exteriors (a numerical check, not a changed endpoint).
"""
from decimal import Decimal as Q, localcontext
import json
from math import floor
from pathlib import Path
import time

import numpy as np
from tranche2_cube import PARITY, ROOT


def binomial(n,x):
    odds = x.exp(); p = odds/(1+odds)
    mode = floor((n+1)*p)
    # Build relative-to-mode weights outwards; each side decreases, so a
    # discarded coefficient below 1e-72 bounds each further coefficient.
    left, right = [], []
    v = Q(1)
    for k in range(mode,0,-1):
        v = v*k/((n-k+1)*odds)
        if v<Q('1e-72'):
            break
        left.append(v)
    v = Q(1)
    for k in range(mode,n):
        v = v*(n-k)*odds/(k+1)
        if v<Q('1e-72'):
            break
        right.append(v)
    mass = list(reversed(left))+[Q(1)]+right
    norm = sum(mass)
    return mode-len(left),[v/norm for v in mass]


def coefficient(h):
    beta = Q('3.05'); na = 2400
    h = list(map(int,h)); d = [beta*(4*k-16) for k in range(7)]
    # Tilt only for conditioning, not as a substitute for the coefficient.
    lo,hi = -40.,40.
    for _ in range(51):
        mid = (lo+hi)*.5
        if sum(n/(1+np.exp(float(dk)-mid)) for n,dk in zip(h,d))<na:
            lo = mid
        else:
            hi = mid
    u = Q(str((hi+lo)*.5)); zg = -u*na
    offset, poly = 0,[Q(1)]
    for k,n in enumerate(h):
        x = u-d[k]
        zg += n*((1+x.exp()).ln()-beta*(k-5)**2)
        loc,g = binomial(n,x); offset+=loc
        conv = [Q(0)]*(len(poly)+len(g)-1)
        for j,b in enumerate(g):
            for i,a in enumerate(poly):
                conv[i+j] += a*b
        poly = conv
    mass = poly[na-offset]
    # TV change from normalizing after truncating seven PMFs <= sum of tails.
    # Each has >=1 unit of relative mass at its mode; <=n+1 omitted entries.
    tail_bound = Q(sum(h)+len(h))*Q('1e-72')
    return zg+mass.ln(), tail_bound/(mass-tail_bound)


def dec_projection(z):
    p = [(v-max(z)).exp() for v in z]; norm = sum(p); p = [v/norm for v in p]
    s = list(map(int,PARITY))
    low = -min(v for v,b in zip(p,s) if b==1)
    hi = min(v for v,b in zip(p,s) if b==-1)
    for _ in range(400):
        mid = (hi+low)/2
        if mid in (low,hi):
            break
        q = [v+b*mid for v,b in zip(p,s)]
        if sum(b*v.ln() for v,b in zip(q,s))<0:
            low=mid
        else:
            hi=mid
    q = [v+b*(hi+low)/2 for v,b in zip(p,s)]
    kl = sum(v*(v/w).ln() for v,w in zip(p,q))
    return kl,p,q


def main():
    out = ROOT/'16-17-development-precision.json'
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic(); record = {}
    for geom,idx in [('cube',16),('far',17),('near',17)]:
        with np.load(ROOT/f'{idx}-{geom}-development-b09.npz') as f, localcontext() as ctx:
            ctx.prec=90
            h = f['base_h'][0].astype('i4')+f['delta_h'][0]
            z,tail = zip(*(coefficient(v) for v in h))
            want = f['logZ'][0]; error = max(abs(v-Q.from_float(w)) for v,w in zip(z,want))
            if error>Q('1e-9'):
                raise ArithmeticError('independent canonical coefficient disagreement')
            r = dict(native_parent=int(f['native_parent'][0]),anchor_zyx=list(map(int,np.unravel_index(f['anchor'][0],(20,)*3))),
                     observed_corner=int(f['observed'][0]),N_k_each_corner=h.tolist(),logZ_nats=[str(v) for v in z],
                     maximum_abs_logZ_difference=str(error),max_omitted_tail_error_logZ=str(max(tail)))
            if geom=='cube':
                cubic = sum(v*int(s) for v,s in zip(z,PARITY)); kl,p,q = dec_projection(z)
                y = r['observed_corner']; r.update(cubic=str(cubic),KL=str(kl),gain=str((p[y]/q[y]).ln()),
                    p=[str(v) for v in p],q=[str(v) for v in q],
                    evidence='For this fixed exterior the cubic log contrast excludes every microscopic pair Hamiltonian, of any range. Its magnitude is not a KL or a typicality statement.')
                lp,lq = f['logp'],f['logq']; u = lq-lp
                # Per-component generalized KL = p*(exp(u)-1-u), avoiding
                # tiny negative ordinary-KL residuals near equality.
                small = abs(u)<1e-4; phi = np.empty_like(u); us = u[small]
                phi[small] = us*us*(.5+us*(1/6+us*(1/24+us/120)))
                phi[~small] = np.expm1(u[~small])-u[~small]
                stable = (np.exp(lp)*phi).sum(1)
                r.update(development_all_stable_KL_mean=float(stable.mean()),raw_KL_mean=float(f['KL'].mean()),
                         maximum_per_exterior_KL_difference=float(abs(f['KL']-stable).max()))
            else:
                cross = z[0]+z[3]-z[1]-z[2]
                r.update(L=str(cross),reversed_L=str(z[1]+z[2]-z[0]-z[3]),
                         abs_L_difference=str(abs(cross-Q.from_float(f['L'][0]))),
                         susceptibility_prediction=float(f['predicted'][0]),semi_grand_L=float(f['semi_grand_L'][0]))
            record[geom] = r
    with out.open('x') as f:
        json.dump(dict(method='Independent Decimal 90-digit positive binomial convolution. Truncate beyond 1e-72 relative to each mode; each omitted tail entry no larger. Combined PMF total-variation <=6007e-72. No Fourier transform, negative terms, generated configurations, or searched cases.',
                       selected='First sampled eligible anchor of first development parent per stencil, index 9.',
                       elapsed_seconds=time.monotonic()-start,stencils=record),f,indent=2,allow_nan=False)
        f.write('\n')
    print(out,round(time.monotonic()-start,2),'s')


if __name__=='__main__':
    main()
