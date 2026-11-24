"""Render the one released cold audit; check split integrity using saved IDs.

This consumes only completed receipts; never opens a box or recomputes an
endpoint. First panel uses cancellation-safe KL to render sub-epsilon values.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]/'results/l20-autoresearch/tranche2'


def generalized_kl(lp,lq):
    d = lq-lp
    small = abs(d)<1e-4
    v = np.empty_like(d); ds = d[small]
    v[small] = ds*ds*(.5+ds*(1/6+ds*(1/24+ds/120)))
    v[~small] = np.expm1(d[~small])-d[~small]
    return (np.exp(lp)*v).sum(1)


def main():
    record = json.loads((ROOT/'16-17-audit-b09.json').read_text())
    if not record['audit_release']:
        raise ValueError('no public release in the audit receipt')
    dev = np.load(ROOT/'16-cube-development-b09.npz')
    with np.load(ROOT/'16-cube-audit-b09.npz') as a:
        overlap = len(np.intersect1d(dev['native_parent'],a['native_parent']))
        lineage = len(set(zip(dev['chain_seed'],dev['chain_id']))&set(zip(a['chain_seed'],a['chain_id'])))
        if overlap or lineage:
            raise ValueError('development and audit parents are not disjoint')
        kl = generalized_kl(a['logp'],a['logq']); cubic = np.abs(a['cubic'])
        delta = float(np.max(np.abs(kl-a['KL'])))
    dev.close()
    with np.load(ROOT/'17-far-audit-b09.npz') as a:
        x = a['predicted']; y = a['L']
    with np.load(ROOT/'17-near-audit-b09.npz') as a:
        near = a['L']
    fig,axes = plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    positive = kl>0
    p = axes[0].hist2d(cubic[positive],np.log10(kl[positive]),bins=(60,65),norm=LogNorm(),cmin=1,cmap='Blues')
    fig.colorbar(p[3],ax=axes[0],label='Eligible opportunities')
    axes[0].axhline(-3,color='firebrick',ls='--',lw=1,label='0.001 nat')
    axes[0].set(xlabel='Absolute cubic log contrast (nats)',ylabel='log₁₀ pair-projection KL (nats)',title='A. Three registered bonds')
    axes[0].legend(frameon=False,loc='upper center')
    p = axes[1].hist2d(x,y,bins=70,norm=LogNorm(),cmin=1,cmap='Greens')
    fig.colorbar(p[3],ax=axes[1],label='Eligible opportunities')
    low,high = min(x.min(),y.min()),max(x.max(),y.max())
    axes[1].plot([low,high],[low,high],color='black',lw=.8,ls='--')
    axes[1].set(xlabel='−δa δb / χ (nats)',ylabel='Exact canonical L (nats)',title='B. Disjoint bonds')
    for values,label,color in [(y,'Disjoint','seagreen'),(near,'Nearby','darkorange')]:
        v = np.sort(np.abs(values)); keep = v>0
        axes[2].plot(v[keep],np.arange(1,len(v)+1)[keep]/len(v),label=label,color=color)
    axes[2].axvline(.01,color='firebrick',ls='--',lw=1)
    axes[2].set(xscale='log',xlabel='Absolute canonical L (nats)',ylabel='Cumulative opportunity fraction',title='C. Global budget vs contact',ylim=(0,1.02))
    axes[2].legend(frameon=False)
    fig.suptitle('Historical TRAIN audit · L=20 · β=3.05 · 512 whole parents')
    fig.savefig(ROOT/'16-17-opportunities.png',dpi=150);plt.close(fig)
    out = ROOT/'16-17-audit-integrity.json'
    with out.open('x') as f:
        json.dump(dict(development_parents=128,audit_parents=512,overlap_native_rows=overlap,
                       overlap_chain_seed_id=lineage,stable_KL_mean=float(kl.mean()),
                       stable_KL_max_abs_difference=delta,exactly_zero_stable_KL=int((kl==0).sum()),
                       original_points_and_intervals='16-17-audit-b09.json',
                       plot='16-17-opportunities.png',audit_release=record['audit_release']),f,indent=2)
        f.write('\n')


if __name__=='__main__':
    main()
