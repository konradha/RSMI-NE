"""Cold fixed-pattern comparisons and independent-parent spatial correlations of frozen operators."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.special import softmax
from scipy.stats import t
from l20_pairs import L,partition


def connected(means,products):
    # Subtract |<p>|² using products of *different parents* rather than <|p|²>.
    n=len(means); s=means.sum(0); individual=(means*means).sum(1); ss=individual.sum()
    mu2=(s@s-ss)/(n*(n-1)); cov=products.mean(0)-mu2
    estimate=cov/cov[0]
    loo_mu2=(((s-means)**2).sum(1)-ss+individual)/((n-1)*(n-2))
    loo=(products.sum(0)-products)/(n-1)-loo_mu2[:,None]; loo/=loo[:,0:1].copy()
    se=np.sqrt((n-1)/n*np.sum((loo-loo.mean(0))**2,axis=0))
    return estimate,se


def correlations(states,weights):
    means=[]; products=[]
    for start in range(0,len(states),32):
        raw=states[start:start+32]; cube=raw.reshape(-1,10,2,10,2,10,2).transpose(0,1,3,5,2,4,6).reshape(-1,10,10,10,8)
        logits=np.stack([np.einsum('...i,i->...',cube==s,weights) for s in range(3)],axis=-1)
        p=softmax(logits,axis=-1); means.append(p.mean(axis=(1,2,3)))
        values=[]
        for shift in range(6):
            x=sum(np.einsum('nxyzc,nxyzc->n',p,np.roll(p,-shift,axis=d)) for d in (1,2,3))/(3*1000)
            values.append(x)
        products.append(np.stack(values,axis=-1))
    return connected(np.concatenate(means),np.concatenate(products))


def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0]); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    results={}
    for folder in ('fits','cold-controls'):
        for f in (a.out/folder).glob('beta-09_*.json'):
            r=json.loads(f.read_bytes());results[r['buffer'],r['method'],r['seed']]=r
    def scores(b,m,s):
        return np.array([v['paired'] for v in results[b,m,s]['test_batches']])
    fig,(score,spatial)=plt.subplots(1,2,figsize=(12.5,5.5));fig.subplots_adjust(left=.075,right=.98,top=.79,bottom=.22,wspace=.31)
    fig.text(.075,.93,'Can simple lattice patterns explain the cold signal?',weight='bold',fontsize=19)
    fig.text(.075,.865,'β = 3.05   •   The score comparisons and all spatial averages use held-out parent configurations.',fontsize=11,color='#485565')
    methods=('site','uniform','axis','face','staggered','learned'); colors=('#747b83','#dd8251','#c8a035','#648d5d','#8262ab','#096f7e')
    labels=('Single site','Uniform','Axis stripe','Face parity','Checkerboard','Learned'); rows=[]
    for mi,(m,color,label) in enumerate(zip(methods,colors,labels)):
        vals=[]; errors=[]
        for b in (0,1,2):
            learned=np.mean([scores(b,'learned',s) for s in (11,29,47)],axis=0)
            x=learned if m=='learned' else scores(b,m,11); d=learned-x
            ci=lambda x:t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
            rows.append(dict(buffer=b,method=m,paired_nats=x.mean(),ci95_halfwidth=ci(x),learned_minus_method=d.mean(),contrast_ci95_halfwidth=ci(d)))
            vals.append(x.mean()); errors.append(ci(x))
        score.errorbar(np.arange(3)+(mi-2.5)*.055,vals,yerr=errors,fmt='o',label=label,color=color,ms=4,capsize=2)
    table(a.out/'cold-controls.csv',rows);score.axhline(0,color='.5',lw=.8,ls=':');score.set(xticks=[0,1,2],xlabel='Buffer (lattice spacings)',ylabel='Held-out InfoNCE (nats)');score.legend(frameon=False,ncol=2,fontsize=8);score.set_title('a   Equal-budget ternary operators',loc='left')
    # A single seed specified before inspecting the fits, no best-of-seeds transfer.
    rows=[]; ci=1.96
    for index,style in ((0,'--'),(9,'-')):
        groups,_=partition(index);parents=groups[-1]
        with np.load(a.data/f'beta-{index:02d}.npz',allow_pickle=False) as z:
            states=z['species'][parents]; beta=float(z['beta']); keys=np.column_stack((z['chain_seed'],z['chain_id']))
        import hashlib
        source=json.loads((a.out/'fits'/f'beta-{index:02d}_b1_learned_11.json').read_bytes())
        if hashlib.sha256(keys[parents].astype('<u8').tobytes()).hexdigest()!=source['provenance']['parent_sha256'][-1]:
            raise ValueError('spatial split differs from InfoNCE test split')
        for m,color in [('learned',colors[-1]),('uniform',colors[1]),('staggered',colors[-2])]:
            w=np.array(results[1,m,11]['weights']);corr,err=correlations(states,w)
            for j,(x,s) in enumerate(zip(corr,err)):
                rows.append(dict(beta=beta,method=m,r_lattice_units=2*j,correlation=x,parent_jackknife_se=s))
            spatial.errorbar(np.arange(2,11,2),corr[1:],yerr=ci*err[1:],fmt='o'+style,color=color,ms=3,lw=1.2,capsize=2,label=f'{m}, β={beta:g}')
    table(a.out/'spatial.csv',rows);spatial.axhline(0,color='.5',lw=.8,ls=':');spatial.set(xticks=np.arange(2,11,2),xlabel='Block-center separation, r (sites)',ylabel='Connected category-probability correlation / C(0)');spatial.set_yscale('symlog',linthresh=.002);spatial.legend(frameon=False,ncol=2,fontsize=7.5);spatial.set_title('b   Apply the same cold filters at β=0 and 3.05',loc='left')
    for ax in (score,spatial): ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    fig.text(.075,.08,'a: pointwise 95% intervals over 32 independent parent batches; learned scores averaged over three fits. b: buffer-one, seed-11 kernels; 4,096 parents per β.',fontsize=9)
    fig.text(.075,.035,'Disjoint 2³ blocks; average over positions and three axes. Bars in b: ±1.96 parent jackknife SE. These are ordinary static correlations, not a point-to-set length.',fontsize=9)
    fig.savefig(a.out/'cold-operators.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
