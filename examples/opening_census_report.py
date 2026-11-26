"""Reduce atomic census receipts, write parent-level intervals and plot actual coverage."""
import argparse
import csv
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from opening_census import NB, EDGES, MOMENTS
from opening_inputs import WIDTH, FILLED, N, RESULT, dump, require, sha


def report(scratch,out):
    manifest=json.loads((out/'inputs.json').read_text()); ledger=json.loads((out/'shard-digests.json').read_text());tmp=scratch/'shards'
    require(len(ledger['shards'])==10*N//WIDTH,'not a complete census')
    hj=np.zeros((10,4,NB,NB),dtype=np.int64);sums=np.zeros((10,4));squares=np.zeros_like(sums); seen=np.zeros((10,N),bool)
    retained=np.empty((10,3*N//4,4));native=np.empty((10,3*N//4),np.uint16);ret_group=np.empty((10,3*N//4),np.uint8);cnt=np.zeros(10,int)
    for rec in ledger['shards']:
        b,lo=rec['beta_index'],rec['lo'];p=tmp/f'b{b:02d}-{lo:05d}.npz'
        require(not seen[b,lo:lo+WIDTH].any() and rec['parents']==WIDTH and sha(p)==rec['sha256'] and rec['signature']==ledger['signature'],'overlap/checksum')
        seen[b,lo:lo+WIDTH]=True
        with np.load(p,allow_pickle=False) as z:
            hj[b]+=z['group_joint'];x=z['moments'];rows=z['retained'].astype(int)+lo;g=np.load(scratch/f'g{b:02d}.npy',mmap_mode='r')[rows]
            require(np.all(g<3) and len(rows)==len(x),'reserve leakage');sums[b]+=x.sum(0)+z['reserve_sum'];squares[b]+=np.square(x).sum(0)+z['reserve_square_sum'];i=cnt[b];native[b,i:i+len(x)]=rows;retained[b,i:i+len(x)]=x;ret_group[b,i:i+len(x)]=g;cnt[b]+=len(x)
    require(seen.all() and np.all(cnt==3*N//4),'did not visit all parents')
    require(np.all(hj.sum((2,3))==FILLED*np.array([6144,2048,4096,4096])),'lost sites after merge')
    np.savez_compressed(out/'census-distributions.npz',joint_counts=hj,positive_edges=EDGES,group_order=np.array(['fit','dev','assess','reserve:fixed tallies only']),native_rows=native,group=ret_group,parent_moments=retained,moment_names=np.asarray(MOMENTS))
    # Model-free sampling errors: each parent contributes its own all-site mean; reserve includes sums/squares only.
    means=sums/N;se=np.sqrt(np.maximum(0,squares-sums**2/N)/(N*(N-1)))
    betas=np.array([float(r['beta']) for r in manifest['rows']]); data=[]
    with (out/'census-temperature.csv').open('w') as f:
        wr=csv.writer(f);wr.writerow(['beta_index','beta','parents','occupied_sites','metric','mean','parent_se','ci95_low','ci95_high'])
        for i in range(10):
            for j,metric in enumerate(MOMENTS):
                mean=float(means[i,j]);err=float(se[i,j]);wr.writerow([i,betas[i],N,N*FILLED,metric,mean,err,mean-1.96*err,mean+1.96*err])
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout='constrained');titles=['Instantaneous change · Vh','Positive environmental opening · V²O','No incident change probability','No instantaneous opening contribution'];colors=['#28659a','#ad4c2e','#28659a','#ad4c2e']
    for j,(ax,title,c) in enumerate(zip(axes.flat,titles,colors)):
        ax.errorbar(betas,means[:,j],yerr=1.96*se[:,j],color=c,marker='o',markersize=4,capsize=3);ax.set_title(title,loc='left',fontweight='bold',fontsize=10);ax.set(xlabel='Inverse temperature β',ylabel='Fraction of occupied sites' if j>=2 else 'Mean across occupied sites');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        if j<2 and np.all(means[:,j]>0) and means[:,j].max()/means[:,j].min()>30:ax.set_yscale('log')
    fig.suptitle('Single-step mobility and opportunities to open it\n16,384 independent parents per temperature · 95% parent intervals',fontsize=14);fig.savefig(out/'census-temperature.png',dpi=170);plt.close(fig)
    # Fine zero + dyadic bins are in the NPZ. Coarse, pre-fixed visual projection of the same cells.
    coarse=np.array([-32,-16,-8,-4,-2,0,2]);lookup=np.zeros(NB,dtype=int);lookup[1]=1;lookup[2:]=1+np.searchsorted(coarse,np.arange(-128,8),side='right');nc=len(coarse)+2
    labels=['0','<−32','−32:−16','−16:−8','−8:−4','−4:−2','−2:0','0:2','≥2']; codes=(lookup[:,None]*nc+lookup).ravel();whole=hj.sum(axis=1);co=np.stack([np.bincount(codes,weights=v.ravel(),minlength=nc*nc).reshape(nc,nc) for v in whole]).astype(np.int64)
    with (out/'within-h.csv').open('w') as f:
        wr=csv.writer(f);wr.writerow(['beta','log2_Vh_bin','log2_V2O_bin','occupied_sites','of_h_bin','conditional_fraction'])
        for b in range(10):
            for h in range(nc):
                total=int(co[b,h].sum());wr.writerows((betas[b],labels[h],labels[o],co[b,h,o],total,co[b,h,o]/total if total else '') for o in range(nc))
    fig,axes=plt.subplots(2,5,figsize=(14,6.6),layout='constrained',sharex=True,sharey=True);cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee');
    for b,ax in enumerate(axes.flat):
        vals=co[b].T/(N*FILLED);im=ax.imshow(np.ma.masked_equal(vals,0),origin='lower',cmap=cmap,norm=LogNorm(1/(N*FILLED),1),interpolation='nearest',aspect='equal');ax.set(xticks=range(nc),yticks=range(nc),xticklabels=labels,yticklabels=labels,title=f'β = {betas[b]:.3f}');ax.tick_params(labelsize=7,axis='x',labelrotation=70);ax.tick_params(labelsize=7,axis='y');
    fig.supxlabel('Instantaneous Vh · exact zero or log₂ bin');fig.supylabel('Opening V²O · exact zero or log₂ bin');fig.suptitle('Joint mobility and opening across the complete L20 release',fontweight='bold');fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.70,label='Fraction of all occupied origins');fig.savefig(out/'census-joint.png',dpi=175);plt.close(fig)
    dump(out/'census-receipt.json',dict(complete=True,distinct_parents=10*N,occupied_centers=10*N*FILLED,parents_with_retained_individual_distributions=int(cnt.sum()),reserve_parents_fixed_aggregates_only=10*N//4,per_parent_sparse_shards='Regenerate from pinned native files; compressed non-reserve row distributions stored in the local shard cache; each shard has a SHA256 in shard-digests.json.',digest_manifest_sha256=sha(out/'shard-digests.json'),released_arrays_sha256=sha(out/'census-distributions.npz'),interval='Mean ±1.96 parent SE (large-N descriptive marginal); no site bootstrap.',within_h='Marginal empirical conditional on mobility bin; see history analysis for exact incident barriers. No causal claim.'))
    print(f'REDUCED {10*N*FILLED:,} occupied centers from {10*N:,} independent parents',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('scratch',type=Path);ap.add_argument('--out',type=Path,default=RESULT);a=ap.parse_args();report(a.scratch,a.out)
