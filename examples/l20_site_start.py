"""Exploratory nested-control check after the initial grid: beta=3.05, buffer=1 only.
The fixed and free models start at the same one-site logits. Both refit the critic for
96 epochs on equal budgets. The existing validation/test parents are reused, not a replication.
"""
import argparse
import json
from pathlib import Path


def summarize(root):
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import t
    scores={}; rules={}; rows=[]; seeds=(11,29,47)
    patterns=np.indices((3,)*8).reshape(8,-1).T  # exhaust all 6,561 distinct visible labelings
    fig,(ax,gain)=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    for method,color in [('site','#bb6632'),('site-start-learned','#126e88')]:
        histories=[]
        for s in seeds:
            r=json.loads((root/f'{method}-{s}.json').read_bytes())
            histories.append([v['val_nats'] for v in r['epochs']])
            scores[method,s]=np.array([b['paired'] for b in r['test_batches']])
            if method=='site-start-learned':
                logits=np.stack([(patterns==k)@np.array(r['weights']) for k in range(3)],axis=1)
                pr=np.exp(logits-logits.max(1,keepdims=True));pr/=pr.sum(1,keepdims=True)
                rules[s]=(int(np.count_nonzero(pr.argmax(1)!=patterns[:,0])),float(pr[np.arange(len(patterns)),patterns[:,0]].min()))
        histories=np.array(histories); x=np.arange(1,1+histories.shape[1]);ax.fill_between(x,histories.min(0),histories.max(0),alpha=.16,color=color)
        ax.plot(x,histories.mean(0),color=color,label=('One site' if method=='site' else 'Free eight-site filter'))
    for i,s in enumerate((*seeds,'mean')):
        get=lambda m: scores[m,s] if s!='mean' else np.mean([scores[m,k] for k in seeds],axis=0)
        free,site=get('site-start-learned'),get('site');d=free-site;ci=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        rows.append(dict(seed=s,free_nats=free.mean(),single_site_nats=site.mean(),gain_nats=d.mean(),low95=d.mean()-ci,high95=d.mean()+ci))
        checked=rules.values() if s=='mean' else [rules[s]]
        rows[-1].update(max_hard_disagreements_of_6561=max(q[0] for q in checked),min_probability_of_anchor_label=min(q[1] for q in checked))
        gain.errorbar(i,d.mean(),yerr=ci,fmt='o',capsize=3,color='#126e88' if s=='mean' else '#849398')
    with (root.parent/'site-start.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
    ax.set(title='Validation through training',xlabel='Epoch',ylabel='Validation InfoNCE (nats)');ax.legend(frameon=False);gain.axhline(0,color='.4',ls=':',lw=1)
    gain.set(title='Held-out gain over one site',xlabel='Initialization of the critic',ylabel='Free − single-site bound (nats)',xticks=range(4),xticklabels=['Seed 11','Seed 29','Seed 47','Mean'])
    fig.suptitle('Same single-site start, same training budget  |  β = 3.05, buffer = 1',fontsize=14)
    for x in (ax,gain): x.grid(axis='y',alpha=.18);x.spines[['top','right']].set_visible(False)
    fig.savefig(root.parent/'site-start.png',dpi=170);plt.close(fig)
    print(rows[-1])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path);p.add_argument('--out',required=True,type=Path)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--seed',type=int,choices=(11,29,47));mode.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:
        summarize(a.out);raise SystemExit(0)
    if not a.data:p.error('--seed requires --data')
    from l20_pairs import load_banks
    from l20_fit import fit
    meta,*banks=load_banks(a.data/'beta-09.npz',1,96);a.out.mkdir(exist_ok=True,parents=True)
    for method in ('site','site-start-learned'):
        r=fit(meta,*banks,method,a.seed);dest=a.out/f'{method}-{a.seed}.json'
        with dest.open('x') as f: json.dump(r,f,allow_nan=False,separators=(',',':')); f.write('\n')
        print(method,a.seed,r['test_nats'],r['best_epoch'],flush=True)
