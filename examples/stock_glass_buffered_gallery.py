"""Render held calibration and DEV depth, and package paired parent score data without copying physical configurations."""
import argparse
from pathlib import Path
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]/'results/buffered-static-gate1'

def publish(scores,out):
    for prefix in ('calibration','confidence','coverage'):
        files=sorted(scores.glob(f'{prefix}-b??.csv'));assert len(files)==12
        data=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);data.to_csv(out/f'{prefix}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    with zipfile.ZipFile(out/'parent-scores.zip','w',compression=zipfile.ZIP_STORED) as z:
        for f in sorted(scores.glob('*.npz')):z.write(f,f.name)
    dev=pd.read_csv(out/'dev.csv');fig,ax=plt.subplots(2,3,figsize=(12,6.8),constrained_layout=True)
    colors=dict(base='#82479A',rich='#037F79')
    for j,fr in enumerate((.25,.5,1)):
        sub=dev.query('label=="physical" and r==4 and fraction==@fr and model!="composition"')
        for i,(metric,title) in enumerate((('dev_logloss_bits','DEV'),('fit_logloss_bits','FIT'))):
            a=ax[i,j];v=sub[metric]
            for (name,arch),d in sub.groupby(['model','architecture']):
                a.plot(d.epoch,d[metric],marker='o',ms=3,color=colors[name],ls='-' if arch=='tanh' else '--',label=f'{name} · {arch}')
            a.set_xscale('symlog',linthresh=1);a.set_xticks([0,1,2,4,8,16],labels=['0','1','2','4','8','16']);a.grid(alpha=.2);a.set_title(f'{title} · {sub.parents.iloc[0]:,} fit parents');a.set_xlabel('Epoch');a.set_ylabel('Cross-entropy (bits/star)')
    ax[0,0].legend(fontsize=8);fig.suptitle('Depth and independent training data: R4 at β=3.05 · lower is better');fig.savefig(out/'dev-depth.png',dpi=155);plt.close(fig)
    cm=pd.read_csv(out/'calibration.csv.gz');co=pd.read_csv(out/'confidence.csv.gz');fig,ax=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for j,b in enumerate((9,8,11)):
        sub=cm.query('label=="physical" and r==4 and b==@b');a=ax[0,j];x=sub.exact_mass;y=sub.native_mass
        a.errorbar(x,y,yerr=1.96*sub.se_paired_parents,fmt='.',alpha=.55,c='#076F70',elinewidth=.65);hi=max(x.max(),y.max())*1.08;a.plot([0,hi],[0,hi],':',c='#555');a.set(xlabel='Exact exterior mean',ylabel='Native frequency',title=f'β={sub.beta.iloc[0]:.3f} · 210 star states',xlim=(-.0004,hi),ylim=(-.0004,hi));a.grid(alpha=.2)
        a=ax[1,j];v=co.query('label=="physical" and r==4 and b==@b');hi=1.08*max(.1,v.pred_confidence.max(),v.obs_accuracy.max(),v.exact_agreement.max());a.plot([0,hi],[0,hi],':',c='#555')
        for name,d in v.groupby('model'):
            a.plot(d.pred_confidence,d.obs_accuracy,color=colors[name],lw=.7);a.scatter(d.pred_confidence,d.obs_accuracy,s=10+70*d.parent_weight,color=colors[name],label=name);a.scatter(d.pred_confidence,d.exact_agreement,color=colors[name],marker='x',s=25)
        a.grid(alpha=.2);a.set(xlabel='Predictor confidence in top state',ylabel='Top state: native fraction / exact mass',xlim=(0,hi),ylim=(0,hi))
    ax[1,0].legend(fontsize=8);fig.suptitle('Held calibration · upper: 95% paired parent uncertainty on native − exact; lower: dots native, × exact');fig.savefig(out/'held-calibration.png',dpi=155);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();publish(a.scores,a.out)
