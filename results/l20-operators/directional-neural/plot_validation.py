"""Replot neural model selection and proper scores from the two development CSVs, without refitting."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


out=Path(__file__).resolve().parent

def read(name):
    with (out/name).open() as f: return list(csv.DictReader(f))

scores=read('development.csv'); epochs=read('epochs.csv'); rows=[]
colors={'neural':'#ad4b24','neural-random':'#39739a','learned':'#597447'}
labels={'neural':'Neural code','neural-random':'Random nonlinear code','learned':'Primary hard tree'}
fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
ax=axes[0]
for var in colors:
    for seed in ('11','29','47'):
        r=[s for s in epochs if s['variant']==var and s['seed']==seed]; xx=np.array([int(s['epoch']) for s in r]); vv=np.array([float(s['validation_nce']) for s in r]); i=vv.argmax()
        ax.plot(xx,vv,color=colors[var],alpha=.65,lw=1,label=labels[var] if seed=='11' else None)
        ax.scatter(xx[i],vv[i],c=colors[var],s=13)
ax.set_xlabel('Training epoch'); ax.set_ylabel('Validation expected InfoNCE (nats)'); ax.axhline(0,color='black',lw=.7); ax.legend(fontsize=8)
contrast=[('direction','neural'),('neural-random','neural'),('learned','neural'),('barrier','barrier+neural'),('barrier+neural-random','barrier+neural'),('barrier+learned','barrier+neural')]
for ax,metric,title in zip(axes[1:],('logloss','direction_mse'),('Log-score gain (nats/event)','Within-bond direction MSE reduction')):
    for j,lag in enumerate((4096,16384)):
        ds=[];err=[]
        for base,mod in contrast:
            def select(name):return {r['parent_id']:float(r[metric]) for r in scores if r['split']=='validation' and int(r['lag'])==lag and r['model']==name and int(r['n_bonds'])}
            a,b=select(base),select(mod); d=np.array([v-b[p] for p,v in a.items()]); h=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)); ds.append(d.mean());err.append(h)
            rows.append(dict(lag=lag,metric=metric,base=base,model=mod,n_parents=len(d),gain=d.mean(),lo=d.mean()-h,hi=d.mean()+h))
        ax.errorbar(ds,np.arange(len(ds))+.12*(2*j-1),xerr=err,fmt='o',capsize=2,ms=4,label=f'{lag:,} sweeps')
    ax.axvline(0,color='black',lw=.75);ax.axhline(2.5,color='.8',lw=.8);ax.set_yticks(range(len(contrast)),['Direction','Random','Tree','Barrier','Barrier + random','Barrier + tree'] if metric=='logloss' else ['']*len(contrast));ax.invert_yaxis();ax.set_xlabel(title);ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2));ax.legend(fontsize=8)
fig.suptitle('Secondary neural bottleneck · archived model-selection data only\nLeft: three seeds per critic. Right: selected neural code versus comparator, 16 validation parents (descriptive paired 95% t intervals).',fontsize=12)
fig.savefig(out/'development.png',dpi=155);plt.close(fig)
with (out/'validation-gains.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
