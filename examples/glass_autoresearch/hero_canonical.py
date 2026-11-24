"""Canonical beta-zero center/shell information from finite population counts, no samples."""
import json
from pathlib import Path
import numpy as np
from scipy.special import gammaln, logsumexp

PUBLIC=Path(__file__).resolve().parents[2]/'results/l20-autoresearch/hero-central'


def count_vectors(n):
    return np.array([(i,j,n-i-j) for i in range(n+1) for j in range(n-i+1)],dtype=int)


def run():
    pop=np.array([2000,2400,3600],int); volume=int(pop.sum()); n,m=27,386
    logfac=gammaln(1+np.arange(volume+1)); cv=count_vectors(n); ce=count_vectors(m)
    def logchoose(total,k):return logfac[total]-logfac[k]-logfac[total-k]
    def mass(total,k,vec): return sum(logchoose(total[j],vec[:,j]) for j in range(3))-logchoose(int(total.sum()),k)
    lc,le=mass(pop,n,cv),mass(pop,m,ce); lc-=logsumexp(lc); le-=logsumexp(le)
    kl=0.; largest_norm_error=0.
    for c,pc in zip(cv,np.exp(lc)):
        conditional=mass(pop-c,m,ce); z=logsumexp(conditional);largest_norm_error=max(largest_norm_error,abs(z)); conditional-=z
        kl+=float(pc*np.sum(np.exp(conditional)*(conditional-le)))
    r2=(n*m)/((volume-n)*(volume-m))
    assert 0<kl<.001 and 0<r2<.001 and largest_norm_error<1e-8
    row=dict(scope='Canonical uniform-composition ensemble, no draws, no physical archive rows. Not an empirical leakage control.',volume=volume,composition_vac_A_B=pop.tolist(),center_sites=n,shell_sites=m,
             central_inventory_cells=len(cv),shell_inventory_cells=len(ce),unconditional_ordered_V_E_MI_nat=kl,
             normalized_MSE_gain_for_optimal_linear_forecast_of_a_single_species_count=r2,maximum_unrenormalized_log_mass_error=largest_norm_error,
             proof='Given the three center counts C, the law of the shell sequence is independent of the ordered central configuration. Among shell assignments with given count vector D the likelihood ratio is constant. Thus I(V;E)=I(C;D), computed by summing P(C) KL[Hypergeom(N-C,386)||Hypergeom(N,386)]. R4 and sparse R5 have the same finite-inventory law. Squared linear count correlation is n*m/((N-n)*(N-m)); this is a benchmark for that count, not a universal forecast skill bound.',
             conditional='For any S(V), the chain rule gives I(V;E|S)=I(V;E)-I(S;E) <= I(V;E). For the exact radius-3 interior inventory I, the uniform conditional ensemble factorizes between interior and exterior at beta zero. For every central rule f(V), E and f are conditionally independent given I; the true conditional covariance with m1(E,I)-m0(E,I) is zero. Finite fitted forecast errors need not coincide even then. The planned empirical hot control remains a separate gate.')
    with (PUBLIC/'canonical-null.json').open('x') as f:json.dump(row,f,indent=2); f.write('\n')


if __name__=='__main__':run()
