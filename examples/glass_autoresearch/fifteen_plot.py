"""Plot issued vacancy-placement evidence (no re-analysis of archived configurations)."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def draw(o):
    x=json.loads((o/'01-validation-b07.json').read_bytes());dec=json.loads((o/'01-decision.json').read_bytes());hot=json.loads((o/'01-development-b00.json').read_bytes());cold=json.loads((o/'01-development-b09.json').read_bytes());fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained',gridspec_kw=dict(width_ratios=[1.1,1.05,1.35]));ax,bx,cx=axes
    for i,(name,z,color) in enumerate([('β=0\ncontrol',hot,'#738086'),('β=2.37\nvalidation',x,'#007873'),('β=3.05\ndevelopment',cold,'#3a687a')]):
        v=z['gain_nat_per_eligible'];m=v['mean'];l,h=v['ci95'];ax.bar(i,m,color=color,width=.56);ax.errorbar(i,m,yerr=[[m-l],[h-m]],c='black',capsize=3,fmt='none');ax.text(i,m+.027,f'{m:.3f}',ha='center',fontsize=10)
    ax.axhline(np.log(2),ls=':',c='.4',lw=.8);ax.set(xticks=range(3),xticklabels=['0\ncontrol','2.37\nvalidation','3.05\ndevelopment'],xlabel='Inverse temperature β',ylim=(-.035,.79),ylabel='Gain over pair-law coin (nat/tie)',title='Which endpoint is occupied?');ax.text(-.35,.699,'ln 2',color='.3',fontsize=8)
    for z in x['reliability']:
        if not z['count'] or z['forecast']['mean'] is None:continue
        u=z['forecast']['mean'];m=z['observation']['mean'];l,h=z['observation']['ci95'];bx.errorbar(u,m,yerr=[[m-l],[h-m]],fmt='o',markersize=3.5+np.log10(z['count']),color='#007873',capsize=2.5)
    bx.plot([.48,1],[.48,1],ls=':',c='.55');bx.set(xlabel='Predicted favored-end probability',ylabel='Observed favored-end fraction',title='Calibration on 512 parents',ylim=(.44,1.025),xlim=(.47,1.025));bx.set_aspect('equal',adjustable='box')
    cx.set_xlim(0,1);cx.set_ylim(0,1);cx.axis('off');cx.set_title('What breaks the tie?');cx.text(.5,.91,'Exterior degrees with both ends empty',ha='center',fontsize=10);xs=[.13,.24,.35,.63,.74,.85];base=[4,4,4,3,3,5]
    for xx,v in zip(xs,base):cx.text(xx,.77,str(v),ha='center',va='center',bbox=dict(boxstyle='circle',fc='#e2e8e8',ec='none'))
    for yy,head,vals,col in [(.53,'Fill left',[5,5,5,3,3,5],'#cce8dc'),(.29,'Fill right',[4,4,4,4,4,6],'#f7dcca')]:
        cx.text(.5,yy+.104,head,ha='center',fontsize=9,color='.3');
        for xx,v in zip(xs,vals):cx.text(xx,yy,str(v),ha='center',va='center',bbox=dict(boxstyle='circle',fc=col,ec='none'))
    cx.text(.5,.10,'2,086 of 2,086 observed placements favored\nthe side restoring degrees 3 or 5.',ha='center',fontsize=9);fig.suptitle(f"{dec['eligible']:,} placements indistinguishable to every cubic vacancy pair law on the first three shells",fontsize=14)
    for a in (ax,bx):a.spines[['right','top']].set_visible(False)
    fig.savefig(o/'pair-tie.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);draw(p.parse_args().out)
