"""#18: archive entropy loss, preserving the full temperature-quadrature ambiguity."""
import argparse
import itertools
import json
from math import lgamma
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import xlogy
from scipy.stats import t
from mask_law import moments
from vacancy_free_energy import log_partition

N, OCC, A = 8000, 6000, 2400


def logchoose(n, k):
    return lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)


def check():
    # Direct Shannon sum, not a derivative or the same projection twice.
    neighbors = np.array([0, 2, 2, 3, 4, 6]); hist = np.bincount(neighbors, minlength=7)
    combos = list(itertools.combinations(range(6), 2)); energies = []
    for choice in combos:
        ell = np.full(6, 5); ell[list(choice)] = 3; energies.append(((neighbors-ell)**2).sum())
    errors = []
    for b in (0, .8, 3.05):
        w = np.exp(-b*(energies-np.min(energies))); q = w/w.sum()
        exact = -xlogy(q, q).sum(); reconstructed = log_partition(hist[None], b, 2)[0] + b*moments(hist, b, 2)[2]
        errors.append(abs(exact-reconstructed))
    if max(errors)>1e-11: raise ArithmeticError(errors)
    return max(errors)


def run(banks, out):
    tiny_error = check(); s0_label=logchoose(OCC,A)/N; s0_mask=logchoose(N,OCC)/N; s0=s0_mask+s0_label
    # Hypergeometric distribution of six distinct neighbors given the center occupied.
    p=(OCC-1)/(N-1); m0=6*p; v0=6*p*(1-p)*(N-7)/(N-2)
    e0=(A*(v0+(m0-3)**2)+(OCC-A)*(v0+(m0-5)**2))/N
    idx=[0,1,2,3,4,5,7,8,9]; n=256; crit=t.ppf(1-.05/(2*2*len(idx)),n-1)
    rows=[]; receipt={}; intlo=inthi=0.; prev=None
    for i in idx:
        with np.load(banks/f'b{i:02}.npz') as data:
            b=float(data['beta']); species=data['species'][:n]; archived=data['energy'][:n]; parent=data['parent'][:n]
        occ=species!=0; m=sum(np.roll(occ,s,axis=ax).astype(np.int8) for ax in (1,2,3) for s in (-1,1))
        h=np.stack([((m==k)&occ).sum((1,2,3)) for k in range(7)],axis=1)
        e=(occ*(m-np.where(species==1,3,5))**2).sum((1,2,3));
        if not np.array_equal(archived,e): raise ValueError(f'archive Hamiltonian mismatch at {i}')
        label=(log_partition(h,b,A)+b*np.array([moments(r,b,A)[2] for r in h]))/N if b else np.full(n,s0_label)
        raw=e/N; em=raw.mean() if b else e0; es=crit*raw.std(ddof=1)/np.sqrt(n) if b else 0.; sm=label.mean(); ss=crit*label.std(ddof=1)/np.sqrt(n) if b else 0.
        if prev is not None:
            width=b-prev[0]; intlo+=width*(em-es); inthi+=width*(prev[1]+prev[2])
        st=[max(0.,s0-inthi+b*(em-es)),min(s0,s0-intlo+b*(em+es))]; lab=[max(0.,sm-ss),min(s0_label,sm+ss)]; mask=[max(0.,st[0]-lab[1]),min(s0_mask,st[1]-lab[0])];loss=[s0_mask-mask[1],s0_mask-mask[0]];ll=[s0_label-lab[1],s0_label-lab[0]]
        if st[0]>st[1]+1e-12 or mask[0]>mask[1]+1e-12:raise ArithmeticError('entropy bound inconsistent')
        rows.append(dict(index=i,beta=b,energy_per_site=em,energy_simultaneous_halfwidth=es,label_entropy=sm,label_loss=ll,mask_loss=loss,total_entropy=st,integral_left_upper=inthi,integral_right_lower=intlo,ranking='decoration greater' if ll[0]>loss[1] else 'geometry greater' if loss[0]>ll[1] else 'unresolved'))
        receipt[f'parent{i}']=parent;receipt[f'energy{i}']=raw;receipt[f'label_entropy{i}']=label;prev=(b,em,es)
    np.savez_compressed(out/'18-parent-evidence.npz',**receipt)
    report=dict(units='nats/site',per_beta_parents=n,index6='excluded, monotone bracket from 5 to 7',energy_sampling='actual parent energies; exact hypergeometric mean at beta zero',simultaneous_approximate_confidence=.95,student_bonferroni_critical=crit,assumptions='Independent representative canonical parents and parent-mean t approximation. Deterministic left/right bounds for the population monotone curve; no trapezoid precision assumption.',beta_zero=dict(mask=s0_mask,labels=s0_label,total=s0,energy=e0),tiny_exact_max_abs_error=tiny_error,grid=rows)
    (out/'18-entropy.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(json.dumps(rows[-1]),flush=True)
    fig,ax=plt.subplots(figsize=(7.0,4.1),layout='constrained');x=[r['beta'] for r in rows];a=np.array([r['mask_loss'] for r in rows]);l=np.array([r['label_loss'] for r in rows]);ax.fill_between(x,*a.T,color='#6d94bd',alpha=.38,label='vacancy geometry: allowed bracket');ax.errorbar(x,l.mean(axis=1),yerr=np.maximum(0,(l[:,1]-l[:,0])/2),fmt='o-',color='#b94233',ms=4,label='A/B decoration: conditional entropy');ax.set(xlabel='Inverse temperature β',ylabel='Entropy loss from β=0 (nats/site)',title='Where does canonical entropy go?  L=20');ax.legend(frameon=False);fig.savefig(out/'18-entropy.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('banks',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.banks,a.out)
