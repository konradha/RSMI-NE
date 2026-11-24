"""Exact finite-composition beta-zero null. No native parent, encoder or outcome bank."""
import math
import json
from pathlib import Path

import numpy as np
from cycle_bank import catalogue, PRIVATE
from spatial_cycle_bank import PUBLIC, array_sha


def run():
    _, _, ids, p, mass, overall = catalogue()
    # n ranges from 0 to 12; n_A < 13 so integer division recovers inventory exactly.
    occ = ids.astype(int)//(13*16*4)
    remain = 2000 - (12-occ)
    L, sites, shell = 20, 8000, 386
    q = remain/(sites-12)
    e_count = shell*q
    var_count = shell*q*(1-q)*(sites-12-shell)/(sites-13)
    mask = np.load(PRIVATE/'eligible.npy',allow_pickle=False)
    keep = np.isin(ids,mask)
    assert np.isclose(mass@p,overall) and np.isclose(mass@e_count,386*.25)
    m = mass[keep]; y = p[keep]; x = e_count[keep]; norm = m/m.sum()
    rows = [dict(stratum=int(s), n_occupied=int(n), eligible_mass=float(w),
                 cycle_probability=float(py), shell_expected_vacancies=float(ec),
                 shell_variance_vacancies=float(vc))
            for s,n,w,py,ec,vc in zip(ids[keep],occ[keep],m,y,x,var_count[keep])]
    result = dict(status='Analytic beta-zero reference only; no hot assessment, fitted model or new physical input.',
        volume=L**3, frozen_eligible_strata=len(mask), eligibility_i2_sha256=array_sha(mask.astype('<i2')),
        probability_all_cycles=overall, probability_eligible=float(m.sum()),
        probability_cycle_given_eligible=float(norm@y),
        proof='At beta=0 every species assignment with totals (2000 vacant,2400 A,3600 B) is equiprobable. Given any central word and species assignment in a single declared stratum s, inventory fixes n and n_A. For a specified 386-bit shell vacancy mask with k ones the probability is (2000-12+n)_k (6000-n)_(386-k)/(7988)_386 (falling factorials); it is the same for all central words in that stratum. Hence E and V, and therefore any deterministic frozen f(E) and cycle Y(V), are independent conditional on s. Mixing strata can reintroduce a tiny composition correlation. This is not a check of native archive sampling, neural training, calibration or software leakage.',
        conditional_mutual_information_nat=0, optimum_incremental_bernoulli_log_score_nat=0,
        covariance_cycle_with_shell_vacancy_fraction_unconditional=float(mass@(p*q)-(mass@p)*(mass@q)),
        covariance_cycle_with_shell_vacancy_fraction_mixed_eligible_strata=float((norm@(y*x)-(norm@y)*(norm@x))/shell),
        rows=rows)
    with (PUBLIC/'analytic-hot-null.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':run()
