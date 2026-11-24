"""Exact cardinality-constrained compression for an explicitly projected L20 channel.

V and Y are each an ordered pair of ternary species (9 letters). V is a body
 diagonal of a 3^3 block; each Y site is directly outward along the same cube
axis from one end, across b empty layers. 'Empty' denotes unobserved, not vacant.
Translation average within each parent; random frame assigned independently of
species. This is not the entire shell. No Monte Carlo, neural critic or test fold.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import xlogy
from scipy.stats import t
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from l20_pairs import partition

BUFFERS = (1, 2, 3, 4, 5)
AXIS = np.arange(9)
FIXED = (2*(AXIS//3==1)+(AXIS%3==1)).astype(int)
NFRAME = 24


def partitions():
    """Restricted-growth partitions; labels are equivalent up to permutation."""
    sets={k:[] for k in range(1,5)}
    def add(label, highest):
        if len(label)==9:
            sets[highest+1].append(label); return
        for x in range(min(3, highest+1)+1): add(label+(x,),max(highest,x))
    add((0,),0)
    lookup={k:np.asarray(values,dtype=np.int16) for k,values in sets.items()}
    assert [len(x) for x in lookup.values()]==[1,255,3025,7770]
    return lookup


def solutions(c, encodings):
    p=c.astype(float)/c.sum(); py=p.sum(0); subsets=(np.arange(512)[:,None]>>AXIS)&1
    q=subsets@p; joint_mi=xlogy(q,np.divide(q,q.sum(1)[:,None]*py,out=np.ones_like(q),where=q.sum(1)[:,None]*py>0)).sum(1)
    best={}; objectives={}
    for k,codes in encodings.items():
        masks=np.stack([((codes==i)*(1<<AXIS)).sum(1) for i in range(k)],1)
        values=joint_mi[masks].sum(1); w=values.argmax(); best[f'K={k}']=codes[w]; objectives[f'K={k}']=values[w]
    best['two A bits']=FIXED; best['both species']=AXIS
    return best,objectives


def tables():
    pos=np.indices((20,)*3).reshape(3,-1).T
    def addresses(dx): return np.ravel_multi_index(((pos+dx)%20).T,(20,)*3).astype(np.uint16)
    result=[]
    for signs in np.ndindex(2,2,2):
        v=np.array(signs)*2-1
        for axis in range(3):
            unit=np.zeros(3,int);unit[axis]=v[axis]
            result.append([np.stack([addresses(-v),addresses(v)])]+[np.stack([addresses(-v-(b+1)*unit),addresses(v+(b+1)*unit)]) for b in BUFFERS])
    return np.asarray(result)


def marginals(state, parents, ix, index, shuffled):
    """One joint table per 64 independent parents; negative partners stay in cluster."""
    groups=parents.reshape(-1,64); out=np.zeros((len(groups),len(BUFFERS),9,9),np.int64); null=np.zeros_like(out) if shuffled else None
    frames=np.random.default_rng([652827,index]).integers(NFRAME,size=len(state)); flat=state.reshape(-1,8000)
    for i,rows in enumerate(groups):
        # Group rows, never columns, for the shuffled control: same anchor, independent parents.
        for j in range(0,len(rows),16):
            ids=rows[j:j+16]; a=flat[ids]; tab=ix[frames[ids]]; row=np.arange(len(a))[:,None]
            pair=lambda two: 3*a[row,two[:,0]]+a[row,two[:,1]]
            v=pair(tab[:,0]);
            for n in range(len(BUFFERS)):
                y=pair(tab[:,n+1]); out[i,n]+=np.bincount((9*v+y).ravel(),minlength=81).reshape(9,9)
                if shuffled: null[i,n]+=np.bincount((9*v+np.roll(y,1,axis=0)).ravel(),minlength=81).reshape(9,9)
    assert np.all(out.sum((-2,-1))==64*8000)
    return out,null


def measurements(train, groups, code):
    k=int(code.max())+1; one=np.eye(k)[code]; train_h=one.T@train
    # Jeffreys smoothing for log score, not for selecting the empirical optimum.
    logq=np.log((train_h+.5)/(train_h.sum(1,keepdims=True)+4.5))
    ce=np.einsum('gvy,vy->g',groups,logq[code])/groups.sum((1,2)); m=groups.sum(1); m=m/m.sum(1,keepdims=True)
    # H_plugin within each independent cluster is downward biased in expectation;
    # H_plugin + E_test log q(Y|H) is conservative in expectation for a frozen q.
    return ce-xlogy(m,m).sum(1)


def ci(x):
    m=float(x.mean()); d=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)));return m,m-d,m+d


def draw(rows,maps,out):
    fig,axs=plt.subplots(1,3,figsize=(13,4.25),layout='constrained'); colors={'K=2':'#76729b','K=4':'#147b83','two A bits':'#c08427','both species':'#615867'}
    for code,col in colors.items():
        pts=[p for p in rows if p['index']==9 and p['code']==code]; x=np.array([p['buffer'] for p in pts]); y=np.array([p['score'] for p in pts]); d=np.array([(p['score']-p['lo'],p['hi']-p['score']) for p in pts]).T
        axs[0].errorbar(x,y,yerr=d,fmt='o-',lw=1.3,ms=4,label=code,color=col,capsize=2)
    axs[0].set(xlabel='Unobserved lattice layers',ylabel='Predictive information bound [nats]',xticks=BUFFERS,title='Cold glass · validation parents');axs[0].set_yscale('symlog',linthresh=1e-5);axs[0].axhline(0,lw=.7,color='0.6');axs[0].legend(frameon=False,fontsize=8)
    for b,col in [(1,'#805c8c'),(2,'#137e83'),(3,'#bf833e')]:
        pts=sorted([p for p in rows if p['buffer']==b and p['code']=='K=4'],key=lambda x:x['beta']); axs[1].plot([p['beta'] for p in pts],[p['score'] for p in pts],'o-',color=col,label=f'{b} layer'+('s' if b>1 else ''))
    axs[1].axhline(0,lw=.7,color='0.6');axs[1].set_yscale('symlog',linthresh=1e-5);axs[1].set(xlabel='Inverse temperature β',ylabel='Four-state bound [nats]',title='Compression versus temperature');axs[1].legend(frameon=False,fontsize=8)
    for ax in axs[:2]:
        ax.set_yticks([0,1e-5,1e-4,1e-3,1e-2],labels=['0','10⁻⁵','10⁻⁴','10⁻³','10⁻²'])
    code=np.array(maps['9:2']['K=4']).reshape(3,3);axs[2].imshow(code,interpolation='nearest',cmap='Pastel2',vmin=0,vmax=3)
    for (i,j),a in np.ndenumerate(code):axs[2].text(j,i,'H'+str(a),ha='center',va='center',fontsize=14)
    for ax in [axs[2]]:ax.set(xticks=[0,1,2],xticklabels=['vac.','A','B'],yticks=[0,1,2],yticklabels=['vac.','A','B'],xlabel='Second corner',ylabel='First corner',title='Exact training optimum · β=3.05, b=2')
    fig.suptitle('How much can two corner species tell us about two exterior species?',weight='bold');fig.savefig(out,dpi=175);plt.close(fig)


def run(data,out,indices):
    out.mkdir(parents=True,exist_ok=True); enc=partitions(); ix=tables(); rows=[]; maps={}; hashes={}
    assert abs(solutions(np.ones((9,9)),enc)[1]['K=4'])<1e-12
    planted=np.zeros((9,9));planted[AXIS,FIXED]=1; ph=np.bincount(FIXED)/9.; assert abs(solutions(planted,enc)[1]['K=4']+xlogy(ph,ph).sum())<1e-12
    for index in indices:
        path=data/f'beta-{index:02d}.npz'; meta=json.loads(path.with_suffix('.json').read_bytes())
        with path.open('rb') as f: hashes[str(index)]=hashlib.file_digest(f,'sha256').hexdigest()
        assert hashes[str(index)]==meta['endpoint_sha256']; (tr,va,_),_=partition(index)
        with np.load(path,allow_pickle=False) as f: state=f['species']; beta=float(f['beta'])
        train=marginals(state,tr,ix,index,False)[0].sum(0); valid,shuf=marginals(state,va,ix,index,True);del state
        for j,b in enumerate(BUFFERS):
            best,objective=solutions(train[j],enc);maps[f'{index}:{b}']={k:code.tolist() for k,code in best.items()}
            scores={k:measurements(train[j],valid[:,j],code) for k,code in best.items()}
            for k,code in best.items():
                mean,lo,hi=ci(scores[k]); delta,dlo,dhi=ci(scores[k]-scores['two A bits']); shuffled=measurements(train[j],shuf[:,j],code); xy,xlo,xhi=ci(scores[k]-shuffled)
                rows.append(dict(index=index,beta=beta,buffer=b,code=k,training_objective=objective.get(k,''),score=mean,lo=lo,hi=hi,shuffled=shuffled.mean(),paired_minus_shuffled=xy,contrast_lo=xlo,contrast_hi=xhi,over_A_bits=delta,over_A_lo=dlo,over_A_hi=dhi,train_parents=len(tr),validation_parents=len(va),independent_groups=len(valid)))
        print(f'beta={beta} validation done; 10,240 fit / 2,048 validation parents; final fold untouched',flush=True)
    with (out/'calibration.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'codes.json').write_text(json.dumps(dict(encodings=maps,source_sha256=hashes,partitions_per_k=[len(c) for c in enc.values()],seed=652827,split='original train,validation ONLY',interpretation='Maximizes empirical training I(H;Y) exactly for this 9x9 projection at fixed alphabet cardinality. Population predictive information is separately lower-bounded with frozen decoder and concave plug-in target entropy, 32 whole-parent clusters. Intervals are nominal validation diagnostics; each temperature fits its own encoder and decoder.'),indent=2)+'\n'); draw(rows,maps,out/'quotient.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(a.data,a.out,[0,5,7,8,9])
