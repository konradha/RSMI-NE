"""Rebuild the equal-storage tradeoff from frozen cohort-level score receipts."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    root=Path(__file__).resolve().parents[2]/'results/l20-autoresearch';cd=json.loads((root/'codec/transfer.json').read_bytes());cohort=cd['temperatures'][0];lc=json.loads((root/'locality/beta-08.json').read_bytes());assert cohort['source']['lineage_sha256']['1']==lc['source']['lineage_sha256']['1']
    names={'rsmi':('RSMI','#176f79','o'),'species-A':('Two corner A bits','#ae652a','s'),'kmeans_A_dipole':('A dipole','#7c60a3','D'),'composition':('Composition','#6d8040','^')};fig,axes=plt.subplots(1,2,figsize=(9.2,3.65),layout='constrained');labels=[]
    for k,(label,c,mark) in names.items():
        lo=lc['models'][k]['near_lower'];hi=lc['models'][k]['near_upper'];x=(lo['estimate']+hi['estimate'])/2
        for ax,j in zip(axes,(0,2)):
            y,yl,yh=cohort['centered_R2'][k][j];ax.errorbar(x,y,xerr=[[x-lo['lo95']],[hi['hi95']-x]],yerr=[[y-yl],[yh-y]],fmt=mark,color=c,ms=7,capsize=3)
        labels.append(Line2D([],[],color=c,marker=mark,ls='none',label=label))
    for ax,title in zip(axes,('Period-two species-A fluctuations','Box-scale species-A fluctuations')):
        ax.axhline(0,color='.6',lw=.8);ax.set(xlabel='Information from six adjacent tiles (nats)',title=title,xlim=(-.003,.077));ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Variance recovered from 250 bytes (R²)');axes[1].legend(handles=labels,frameon=False,loc='center left',fontsize=8);fig.suptitle('Same archived parents, same two bits per tile · β = 2.7111',fontsize=12);fig.savefig(root/'application.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
