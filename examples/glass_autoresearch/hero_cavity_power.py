"""Prospective cavity planning sensitivity; no archive access, no simulated data."""
import csv
import math
from pathlib import Path
from scipy.stats import nct, t

PUBLIC=Path(__file__).resolve().parents[2]/'results/l20-autoresearch/hero-central'


def prob(j,delta,variance):
    # P(lower endpoint of two-sided 95% t interval > margin).
    return float(nct.sf(t.ppf(.975,j-1),j-1,math.sqrt(j)*delta/math.sqrt(variance)))


def required(delta,variance):
    low,high=2,32
    while prob(high,delta,variance)<.90:high*=2
    while low<high:
        mid=(low+high)//2
        if prob(mid,delta,variance)>=.90:high=mid
        else:low=mid+1
    return max(32,low)


def run():
    with (PUBLIC/'cavity-power-sensitivity.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['endpoint','assumption_not_measurement','assumed_sd_between','assumed_sd_thermal','independent_replicas_per_boundary','required_independent_boundaries_or_pairs','individual_boundaries','total_replicas','power_at_available_issuance','within_256_boundaries_before_matching'])
        for endpoint,delta,mult,scenarios in [('normalized paired forecast gain',.01,1,[(.02,.10),(.04,.10),(.08,.10)]),('ordered high-low pair contrast',.10,2,[(.15,.70),(.30,.70),(.50,.70)])]:
            for sd_b,sd_w in scenarios:
                for r in (4,8,16,32,64):
                    variance=sd_b**2+sd_w**2/r
                    n=required(delta,variance)
                    assert prob(n,delta,variance)>=.9 and (n==32 or prob(n-1,delta,variance)<.9)
                    w.writerow([endpoint,'hypothetical normalized SD; replace by independent mixing pilot upper variance',sd_b,sd_w,r,n,mult*n,r*mult*n,round(prob(256//mult,delta,variance),7),mult*n<=256])


if __name__=='__main__':run()
