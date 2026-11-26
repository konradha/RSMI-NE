"""Show actual net-activity shapes at fixed counts in the first eligible source FIT parent; no shell labels or model scores."""
import json
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parents[1]/'results/history-overlap-field'

def render(bank,out):
    with np.load(bank) as z:maps=z['field'];ids=z['ids'];lag=int(z['lag'])
    xyz=np.indices((20,)*3).reshape(3,-1).T;offset=np.indices((3,)*3).reshape(3,-1).T-1;ix=tuple(np.moveaxis((xyz[:,None]+offset)%20,-1,0));chosen=None
    for a,p in zip(maps,ids):
        windows=a[ix].reshape(-1,3,3,3);allaxes=(1,2,3);sub=np.where(((windows!=0).sum(allaxes)==20)&((windows==2).sum(allaxes)==4))[0]
        if not len(sub):continue
        changed=windows[sub]==2;edges=np.stack([(np.take(changed,[0,1],axis=j)&np.take(changed,[1,2],axis=j)).sum(allaxes) for j in (1,2,3)],axis=1);num=edges.sum(1);k=int(num.argmax());nonoverlap=(np.abs((xyz[sub]-xyz[sub[k]]+10)%20-10)>2).any(1);jj=np.flatnonzero(nonoverlap)
        if not len(jj):continue
        m=jj[num[jj].argmin()]
        if num[m]==num[k]:continue
        chosen=dict(parent=int(p),matched_blocks=int(len(sub)),rows=[dict(anchor=xyz[sub[u]].tolist(),bonds_zyx=edges[u].tolist(),flat_index=int(sub[u]),ternary=windows[sub[u]].tolist()) for u in (k,m)]);break
    assert chosen is not None;chosen.update(source='beta=3.20 FIT only; 500M preparation, target Q=.80',lag=lag,n_initially_occupied=20,n_changed=4,selection='First rank-ordered eligible parent; maximum NN changed-changed edges, then minimum edges among nonoverlapping centers. Lexicographic tie breaks. No shell or fitted code used.',bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest());(out/'same-count-examples.json').write_text(json.dumps(chosen,indent=2)+'\n')
    fig,ax=plt.subplots(2,3,figsize=(7.3,4.4),constrained_layout=True);colors=['#eee9df','#427d9a','#da7a22'];cmap=ListedColormap(colors)
    for j,row in enumerate(chosen['rows']):
        a=np.array(row['ternary']);mode='More adjacent' if j==0 else 'Less adjacent'
        for k in range(3):
            ax[j,k].imshow(a[k],cmap=cmap,vmin=-.5,vmax=2.5,interpolation='nearest');ax[j,k].set_xticks([]);ax[j,k].set_yticks([])
            if j==0:ax[j,k].set_title(f'z = {k-1:+d}')
            if k==0:ax[j,k].set_ylabel(f'{mode}\n{sum(row["bonds_zyx"])} changed–changed bonds')
    from matplotlib.patches import Patch
    fig.suptitle('Actual FIT blocks: same parent, 20 initially occupied, 4 changed',fontsize=13)
    fig.legend([Patch(color=c) for c in colors],['Initial vacancy','Same species','Changed endpoint'],loc='upper center',bbox_to_anchor=(.52,-.015),ncol=3,frameon=False)
    fig.savefig(out/'same-count-examples.png',dpi=160,bbox_inches='tight');plt.close(fig);print(chosen['parent'],chosen['matched_blocks'],[(v['anchor'],v['bonds_zyx']) for v in chosen['rows']])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT);a=p.parse_args();render(a.bank,a.out)
