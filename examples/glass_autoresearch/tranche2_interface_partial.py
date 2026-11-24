"""Exploratory partial identification after #23's complete-inventory support gate failed."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy
from scipy.stats import t

from tranche2_interface import ARMS, GAINS, IDEAL, INVENTORY, K, MARGIN, OUT, TILE, WORDS, intv, mappings, rows, save_json, sha, word_index

DESIGN = OUT/'23-partial-design.json'
V = np.array([[1., -1., 0.], [1., 0., -1.], [0., -1., 1.]])


def bound_cells(tables):
    """Max over interior pairs, relax exterior consistency across the eight corners."""
    o = (WORDS != 0).astype(int)
    adj = (np.abs(TILE[:, None]-TILE).sum(2) == 1).astype(int)
    if np.any(adj.sum(1) != 3): raise ArithmeticError('wrong interior graph')
    b = 2*o*(o@adj-IDEAL[WORDS]); states, types = np.unique(np.stack((o, b), axis=2).reshape(-1,2), axis=0, return_inverse=True)
    types = types.reshape(-1,8)
    corners = [(e*e+h, e) for e in range(4) for h in (-9*e, 5*e)]
    extrema = (states[:, None]-states[None, :]) @ np.array(corners).T
    span = np.ptp(extrema, axis=-1)
    bounds = np.zeros((3, K), int)
    for a, table in enumerate(tables):
        for s in np.unique(table):
            w = types[table == s]
            bounds[a,s] = span[w[:, None, :], w[None, :, :]].sum(2).max()
    if np.any(bounds > 768): raise ArithmeticError('exceeds independent owner bound')
    return bounds


def mass_counts(boxes, tables):
    counts = np.empty((len(boxes), 3, K), dtype=np.int32)
    for j, s in enumerate(boxes):
        words = np.bincount(word_index(s), minlength=len(WORDS))
        counts[j] = [np.bincount(tab, weights=words, minlength=K) for tab in tables]
    if not np.all(counts.sum(2) == 8000): raise ArithmeticError('mass counts')
    return counts.reshape(16, 32, 3, K)


def upper_kl(observed, n):
    if observed == 0: return float(-np.expm1(-np.log(3/.05)/n))
    if observed == 1: return 1.
    def residual(p):
        return xlogy(observed, observed/p) + xlogy(1-observed, (1-observed)/(1-p)) - np.log(3/.05)/n
    return brentq(residual, observed, 1-1e-14)


def direction(lo, hi):
    return 'positive' if lo > 0 else 'negative' if hi < 0 else 'unresolved'


def contrasts(center, radius, upper, v):
    c = center @ v.T; r = radius @ np.abs(v).T
    lo, hi = c-r, c+r
    tc = t.ppf(1-.05/6, len(lo)-1)/np.sqrt(len(lo))
    lo95 = lo.mean(0)-lo.std(0, ddof=1)*tc; hi95 = hi.mean(0)+hi.std(0, ddof=1)*tc
    strict = np.abs(v) @ upper; out = []
    for i, name in enumerate(GAINS):
        x = float(c[:,i].mean()); left = float(lo[:,i].mean()); right = float(hi[:,i].mean())
        l95, h95 = float(lo95[i]), float(hi95[i]); lstrict, hstrict = x-strict[i], x+strict[i]
        out.append(dict(contrast=name, coefficients=v[i].tolist(), supported_center=x, missing_only_interval=[left,right],
            worst_case_direction_at_empirical_mass=direction(left,right), block_interval=[l95,h95],
            direction_with_block_uncertainty=direction(l95,h95),
            strict_mass_sensitivity_at_estimated_supported_center=[lstrict,hstrict],
            direction_with_95pct_mass_upper_at_estimated_center=direction(lstrict,hstrict)))
    return out


def figure(report, dev, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9.4,3.8), layout='constrained')
    a,b=axs; q=np.arange(3)
    for j,(title,key) in enumerate([('Block intervals','block_interval'),('Conservative mass limits¹','strict_mass_sensitivity_at_estimated_supported_center')]):
        bounds=np.array([v[key] for v in report['independent_weights']['gain']]); center=bounds.mean(1)
        a.errorbar(center, q+(j-.5)*.15, xerr=[center-bounds[:,0],bounds[:,1]-center], label=title, fmt='o', capsize=3, ms=4)
    a.axvline(0, color='.6', ls=':'); a.set_yticks(q,['Inventory − opt','Inventory − affine','Affine − opt']); a.set_ylim(2.8,-.4)
    a.set_xlabel('Paired gain (energy per exchange)'); a.legend(fontsize=8); a.set_title('All inventories, exploratory development')
    cov = dev['cold']['min_mean_max_common_mass']; b.barh([1], [100*cov[1]], color='#9b6e51', label='Required: 90%')
    b.barh([0], [100*dev['hot']['min_mean_max_common_mass'][1]], color='#74a6ab'); b.axvline(90, color='black',ls='--')
    b.set_yticks([0,1],['β = 0','β = 3.05']); b.set_xlim(0,105); b.set_xlabel('Mean retained inventory mass (%)'); b.set_title('Registered primary: cold coverage failed'); b.legend(fontsize=8)
    fig.suptitle('Frozen labels and unrelaxed interface work • native TRAIN only', fontsize=11)
    fig.text(.02,-.035,'¹ Conditional on observed supports; uncertainty in the supported-work estimate is separate.',fontsize=8)
    fig.savefig(output,dpi=170,bbox_inches='tight'); plt.close(fig)


def main(base):
    dp = OUT/'23-dev.json'; fp = OUT/'23-support.json'; dev=json.loads(dp.read_text()); fit=json.loads(fp.read_text())['fit']['9']
    ep=OUT/dev['cold']['evidence_file']; out=OUT/'23-partial.json'; blockout=OUT/'23-partial-blocks.npz'
    if out.exists() or blockout.exists(): raise FileExistsError('supplement already calculated')
    if dev['cold']['evidence_sha256'] != sha(ep) or dev['support_sha256'] != sha(fp) or sha(dp) != '285820da76e722262f792c946d709b198826963836e8813a65c3fd4c6872c68e':
        raise ValueError('changed primary evidence')
    tables=mappings(); bounds=bound_cells(tables); possible=np.asarray(fit['possible_cells'], bool)
    with np.load(ep, allow_pickle=False) as data:
        denom=data['all_pair_counts']; num=data['work_numerators']; halves=data['one_parent_counts_by_half']; dev_parent=data['parent']; dev_ids=data['chain_id']; dev_seeds=data['chain_seed']
    missing=(denom == 0)&possible; mean=np.divide(num, denom, out=np.zeros(num.shape,float), where=denom>0)
    if np.any(np.abs(mean) > bounds+1e-8): raise ArithmeticError('observed cells lie outside bound')
    d,meta=rows(base, 9, 256,768)
    if np.intersect1d(d['parent'],dev_parent).size or np.unique(np.stack([np.r_[dev_ids.ravel(),d['chain_id']],np.r_[dev_seeds.ravel(),d['chain_seed']]],1),axis=0).shape[0]!=1024:
        raise ValueError('mass and work parent group overlap')
    count=mass_counts(d['species'], tables); freq=count.sum(1)/(32*8000); pooled=halves.sum(1)/(32*8000)
    possible_empirical_empty_both=(halves.sum(1)==0)&possible
    missing_mass=(freq*missing).sum(2); parent_missing=(count*missing[:,None,:,:]).sum(3)/8000
    parent_radii=(count*(bounds*missing)[:,None,:,:]).sum(3)/8000
    max_miss_B=(bounds*missing).max(axis=(0,2)); rad=parent_radii.mean((0,1)); upper=np.array([b*upper_kl(r/b,len(d['parent'])) if b else 0 for r,b in zip(rad,max_miss_B)])
    upper_mass=[upper_kl(m,len(d['parent'])) for m in missing_mass.mean(0)]
    vmargin=V.copy(); vmargin[:,0]-=MARGIN
    def estimate(p):
        center=(p*mean).sum(2); radius=(p*missing*bounds).sum(2)
        return dict(work_supported_center=center.mean(0).tolist(), work_missing_radius_mean=radius.mean(0).tolist(),
            work_empirical_identified_intervals=np.stack(((center-radius).mean(0),(center+radius).mean(0)),1).tolist(),
            gain=contrasts(center,radius,upper,V), gain_minus_material_margin=contrasts(center,radius,upper,vmargin))
    inv_mass=pooled[:,0].mean(0).reshape(81,4).sum(1); minima=np.asarray(fit['min_count_each_cell_over_16_halves']); excluded=[]
    for inv, mass in enumerate(inv_mass):
        if not mass or inv in fit['inventories']: continue
        ks=np.arange(4*inv,4*inv+4); excluded.append(dict(nA=inv//9,nB=inv%9,one_parent_mass=float(mass),
            failing_arm_label_and_minimum=[[ARMS[a],int(s),int(minima[a,4*inv+s])] for a,s in np.argwhere(possible[:,ks] & (minima[:,ks]<4))]))
    np.savez_compressed(blockout, fit_parent=d['parent'].reshape(16,32),fit_chain_id=d['chain_id'].reshape(16,32),fit_chain_seed=d['chain_seed'].reshape(16,32),
        fit_parent_cell_counts=count, missing_cells=missing, deterministic_work_bound=bounds, parent_missing_mass=parent_missing,parent_missing_radius=parent_radii)
    report=dict(status='Exploratory development partial identification only; original primary failed the >=90% coverage gate. No audit or revised primary.',
        design_sha256=sha(DESIGN),script_sha256=sha(Path(__file__)),primary_receipt_sha256=sha(dp),
        primary_evidence_sha256=sha(ep),supplement_evidence_sha256=sha(blockout),supplement_evidence_file=blockout.name,
        bank_sha256=meta['bank_sha256'],weight_parent_positions=[256,768],work_parent_positions=[4096,4608],
        n_blocks=16,work_parents_per_block=32,independent_weight_parents_per_block=32,arms=ARMS,
        primary_retained_mass_range=dev['cold']['min_mean_max_common_mass'],excluded_inventories_descending_mass=sorted(excluded,key=lambda x:-x['one_parent_mass']),
        matched_cross_pairs_over_all_cells=denom.sum((0,2)).tolist(),possible_cells_per_arm=possible.sum(1).tolist(),
        empty_in_either_half_cell_count_range=np.stack((missing.sum(2).min(0),missing.sum(2).max(0)),1).tolist(),
        empty_in_both_halves_cell_count_range=np.stack((possible_empirical_empty_both.sum(2).min(0),possible_empirical_empty_both.sum(2).max(0)),1).tolist(),
        largest_cell_bound=bounds.max(1).tolist(),largest_missing_cell_bound=max_miss_B.tolist(),
        missing_mass_mean=missing_mass.mean(0).tolist(),missing_mass_CI_approximate=intv(missing_mass,.05/3),
        distribution_free_conditional_missing_mass_upper95=upper_mass,
        distribution_free_conditional_missing_radius_upper95=upper.tolist(),
        independent_weights=estimate(freq),original_pooled_one_parent_weights_sensitivity=estimate(pooled),
        interpretation='No omitted cells are imputed. Block intervals combine the measured missing radius with work- and mass-parent sampling error; Student-t approximation, not a finite-sample guarantee. Conservative missing-radius-only sensitivity covers the conditional expected missing radius for these 16 supports via valid independent-fit-parent bounds. Its center is estimated, not exact. Missing-cell assignments can be correlated across arms; triangle limits deliberately need not be jointly attainable. No surface tension or phase inference.')
    save_json(out,report)
    figure(report,dev,OUT/'23-interface-partial.png')
    print(json.dumps({k:report[k] for k in ['missing_mass_mean','largest_missing_cell_bound','distribution_free_conditional_missing_radius_upper95','independent_weights']},indent=2),flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--bank',type=Path,required=True); main(p.parse_args().bank)
