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
    parentdir=out/'parent-distributions';parentdir.mkdir(exist_ok=True); exported=[]
    lengths=[];cells=[];counts=[]
    for rec in ledger['shards']:
        b,lo=rec['beta_index'],rec['lo'];p=tmp/f'b{b:02d}-{lo:05d}.npz'
        require(not seen[b,lo:lo+WIDTH].any() and rec['parents']==WIDTH and sha(p)==rec['sha256'] and rec['signature']==ledger['signature'],'overlap/checksum')
        seen[b,lo:lo+WIDTH]=True
        with np.load(p,allow_pickle=False) as z:
            hj[b]+=z['group_joint'];x=z['moments'];rows=z['retained'].astype(int)+lo;g=np.load(scratch/f'g{b:02d}.npy',mmap_mode='r')[rows]
            require(np.all(g<3) and len(rows)==len(x),'reserve leakage');sums[b]+=x.sum(0)+z['reserve_sum'];squares[b]+=np.square(x).sum(0)+z['reserve_square_sum'];i=cnt[b];native[b,i:i+len(x)]=rows;retained[b,i:i+len(x)]=x;ret_group[b,i:i+len(x)]=g;cnt[b]+=len(x)
            offsets=z['offsets'];cell=z['cell'];count=z['count']
            require(len(offsets)==len(rows)+1 and offsets[0]==0 and offsets[-1]==len(cell)==len(count) and np.all(offsets[1:]>offsets[:-1]),'invalid sparse parent offsets')
            require(np.all(np.add.reduceat(count,offsets[:-1].astype(np.intp),dtype=np.int64)==FILLED),'parent histogram lost occupied sites')
            lengths.append(np.diff(offsets));cells.append(cell);counts.append(count)
        if lo+WIDTH==N:
            require(cnt[b]==3*N//4,'incorrect retained population at beta boundary')
            dest=parentdir/f'beta-{b:02d}.npz';ptr=np.r_[np.uint32(0),np.concatenate(lengths).cumsum(dtype=np.uint32)]
            np.savez_compressed(dest,native_row=native[b],group=ret_group[b],offsets=ptr,cell=np.concatenate(cells),count=np.concatenate(counts))
            exported.append(dict(beta_index=b,parents=int(cnt[b]),file=str(dest.relative_to(out)),sha256=sha(dest)));lengths.clear();cells.clear();counts.clear()
    require(seen.all() and np.all(cnt==3*N//4),'did not visit all parents')
    require(np.all(hj.sum((2,3))==FILLED*np.array([6144,2048,4096,4096])),'lost sites after merge')
    np.savez_compressed(out/'census-distributions.npz',joint_counts=hj,positive_edges=EDGES,group_order=np.array(['fit','dev','assess','reserve:fixed tallies only']),native_rows=native,group=ret_group,parent_moments=retained,moment_names=np.asarray(MOMENTS))
    # Model-free sampling errors: each parent contributes its own all-site mean; reserve includes sums/squares only.
    means=sums/N;se=np.sqrt(np.maximum(0,squares-sums**2/N)/(N*(N-1)))
    betas=np.array([float(r['beta']) for r in manifest['rows']])
    # At beta zero, h counts unlike neighbors. O counts x-y-z nonbacktracking
    # paths with y matching the occupied center's species and z not matching.
    # This identity uses uniform fixed-inventory sampling, not independent sites.
    expected=np.array([sum(n/FILLED*2*(8000-n)/7999 for n in (2400,3600)),
                       sum(n/FILLED*(30/9)*(n-1)/7999*(8000-n)/7998 for n in (2400,3600))])
    dump(out/'beta-zero-check.json',dict(reference='uniform 2400 A, 3600 B, 2000 vacant; sampling without replacement',metrics=list(MOMENTS[:2]),expected=expected.tolist(),observed=means[0,:2].tolist(),parent_se=se[0,:2].tolist(),difference_in_parent_se=((means[0,:2]-expected)/se[0,:2]).tolist(),scope='descriptive external control, not used to change inclusion or fit'))
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
    # Distinguish rare cells from common ones and large opening from nonzero opening.
    support=co.sum(2); upper=co[:,:,lookup[int(np.where(EDGES==2**-8)[0][0])+2]:].sum(2)
    pct=np.divide(100*upper,support,out=np.full(support.shape,np.nan),where=support>0)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True,layout='constrained')
    charts=((np.ma.masked_equal(support.T,0),LogNorm(1,N*FILLED),'Occupied-site support','Observed sites'),(np.ma.masked_invalid(pct.T),None,'Opening within the same mobility bin','Percent with V²O ≥ 2⁻⁸'))
    for ax,(v,norm,title,label) in zip(axes,charts):
        im=ax.imshow(v,origin='lower',aspect='auto',interpolation='nearest',norm=norm,cmap=cmap,**({'vmin':0,'vmax':100} if norm is None else {}));ax.set(xticks=range(10),xticklabels=[f'{b:.2f}' for b in betas],yticks=range(nc),yticklabels=labels,xlabel='Inverse temperature β',title=title);ax.tick_params(axis='x',labelrotation=45);fig.colorbar(im,ax=ax,shrink=.85,label=label)
    axes[0].set_ylabel('Instantaneous Vh · exact zero or log₂ bin');fig.suptitle('How much opening remains at comparable instantaneous mobility?',fontweight='bold');fig.savefig(out/'census-within-h.png',dpi=175);plt.close(fig)
    dump(out/'census-receipt.json',dict(complete=True,distinct_parents=10*N,occupied_centers=10*N*FILLED,
         parents_with_retained_individual_distributions=int(cnt.sum()),reserve_parents_fixed_aggregates_only=10*N//4,
         nonreserve_joint_distributions=exported,format='For row j take offsets[j]:offsets[j+1]; cell = h_bin*138 + opening_bin, count is occupied centers. Bins: zero, positive < 2^-128, then dyadic, last >= 2^7. Map native_row with inputs.json.',
         digest_manifest_sha256=sha(out/'shard-digests.json'),released_arrays_sha256=sha(out/'census-distributions.npz'),
         interval='Mean ±1.96 parent SE (large-N descriptive marginal); no site bootstrap.',within_h='Marginal empirical conditional on mobility bin; see history analysis for exact incident barriers. No causal claim.'))
    live=json.loads((out/'status.json').read_text());physical=out/'history/transfer-decision.json';numeric=out/'numerics/decision.json'
    if physical.exists() and numeric.exists():
        phy=json.loads(physical.read_text())['pass_physical'];num=json.loads(numeric.read_text())['comparison_allowed']
        live.update(phase='scientific_complete' if not (phy and num) else 'conditional_comparison_review',
                    physical_gate='pass' if phy else 'no_go',numerical_gate='pass' if num else 'no_go',
                    downstream_comparison='not_run' if not (phy and num) else 'pending',terminal_decision='stop' if not (phy and num) else None,
                    publication='awaiting verified remote push',evidence='census-receipt.json')
        dump(out/'status.json',live)
    print(f'REDUCED {10*N*FILLED:,} occupied centers from {10*N:,} independent parents',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('scratch',type=Path);ap.add_argument('--out',type=Path,default=RESULT);a=ap.parse_args();report(a.scratch,a.out)
