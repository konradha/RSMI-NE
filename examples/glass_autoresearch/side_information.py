"""A negative-free rate–relevance curve, conditional on *all eight corner species*.

For a given frozen stochastic encoder pi and frozen predictors q0(h|c), r(h|e,c),
degrade the output channel by replacing it with q0 with probability 1-lambda.
No new configurations or sampled hidden labels. This is an achievable family,
not the rate-distortion frontier. Repeated translations are grouped by parent.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import xlogy
from scipy.stats import t
import information as info
from l20_pairs import partition
from l20_operators_static import dense, source

LAM = np.asarray([0., .125, .25, .50, .75, 1.])
GROUP = 128
LN2 = np.log(2.)


def interval(x):
    m=x.mean(0); d=x.std(0,ddof=1)*t.ppf(.975,len(x)-1)/np.sqrt(len(x));return np.stack([m,m-d,m+d],axis=-1)


def decode(fit, c, x):
    posterior=[]
    for i in range(2):
        weights=[fit[f'w{i}_0_{j}'] for j in range(4)]
        posterior.append(info.network(x,weights,fit['priors'][i,0,c])[2])
    return np.mean(posterior,axis=0,dtype=np.float64)


def measure(state,index,fit,project,geom,panels):
    # Independent orientations unlike the shared-stratum contrastive matcher: a
    # single case has the same framed law. No fitting, donor search or negatives.
    vt,et,vp,ep,*_=geom; ids=partition(index)[0][1]; levels=len(LAM); nb=len(ids)//GROUP
    prior=np.exp(fit['priors'][:,0].astype(np.float64)).mean(0);prior/=prior.sum(1,keepdims=True)
    names=('cond_rate_lower','cond_rate_upper','shell_bound','corner_only_null','total_rate_plugin','output_entropy','channel_entropy')
    values=np.empty((nb,levels,len(names)))
    for b,rows in enumerate(ids.reshape(nb,GROUP)):
        rng=np.random.default_rng([792241,index,b]); mixweight=LAM[:,None,None]
        groupcount=np.zeros(info.NC,np.int64); sum_hidden=np.zeros((info.NC,4)); h_pi=np.zeros(levels); logpred=np.zeros(levels); logprior=np.zeros(levels)
        for offset in range(0,panels,16):
            n=min(16,panels-offset); batch=np.tile(rows,n); a=rng.integers(8000,size=len(batch)); ori=rng.integers(48,size=len(batch))
            v=np.take_along_axis(state[batch[:,None],vt[a]],vp[ori],axis=1); e=np.take_along_axis(state[batch[:,None],et[a]],ep[ori],axis=1)
            c=v[:,info.CORNERS]@info.POW; y=project(v)[0].astype(float); y/=y.sum(1,keepdims=True)
            q0=prior[c]; pred=decode(fit,c,np.column_stack((dense(e),dense(v[:,info.CORNERS]))));pred/=pred.sum(1,keepdims=True)
            if np.min(pred)<=0:raise ArithmeticError('frozen posterior is not positive')
            pi=q0+mixweight*(y-q0); dec=q0+mixweight*(pred-q0)
            np.add.at(sum_hidden,c,y);groupcount+=np.bincount(c,minlength=info.NC)
            h_pi-=xlogy(pi,pi).sum((1,2));logpred+=xlogy(pi,dec).sum((1,2));logprior+=xlogy(pi,q0).sum((1,2))
        N=GROUP*panels; mask=groupcount>0; weights=groupcount[mask]/N; q0=prior[mask]; ec=sum_hidden[mask]/groupcount[mask,None]; qbar=q0+mixweight*(ec-q0)
        hcond=-np.einsum('lch,c->l',xlogy(qbar,qbar),weights); globalq=np.einsum('lch,c->lh',qbar,weights); hglobal=-xlogy(globalq,globalq).sum(1)
        h=h_pi/N; lr=hcond-h; ur=-logprior/N-h; relevance=hcond+logpred/N; null=hcond+logprior/N
        if (lr< -1e-8).any() or (lr>ur+1e-8).any() or (relevance>lr+1e-8).any() or (null>1e-8).any() or max(abs(lr[0]),abs(ur[0]),abs(relevance[0]))>1e-8:raise ArithmeticError('categorical information inequalities failed')
        values[b]=np.stack((lr,ur,relevance,null,hglobal-h,hglobal,h),1)/LN2
    return names,values


def draw(rows,out):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4),layout='constrained'); colors=['#8d9094','#c18a2a','#8772a8','#067b79']
    for i,idx in enumerate([0,6,8,9]):
        group=[r for r in rows if r['index']==idx]; lookup={(r['lambda'],r['metric']):r for r in group}; beta=group[0]['beta']
        get=lambda k,col:np.array([lookup[l,k][col] for l in LAM]);lo=get('cond_rate_lower','mean');hi=get('cond_rate_upper','mean');rel=get('shell_bound','mean');x=(lo+hi)/2
        ax.errorbar(x,rel,xerr=(hi-lo)/2+1e-12,yerr=np.stack((rel-get('shell_bound','lo'),get('shell_bound','hi')-rel)),fmt='o-',ms=3.5,capsize=2,color=colors[i],label=f'β = {beta:.3g}')
        bx.fill_between(LAM,lo,hi,alpha=.15,color=colors[i]);bx.plot(LAM,x,'o-',ms=3,color=colors[i])
    ax.axhline(0,color='.5',lw=.8); ax.set(xlabel='Residual input rate [bits / block]',ylabel='Achieved conditional shell bound [bits]',title='Relevance at fixed corner side information');ax.legend(frameon=False)
    bx.set(xlabel='Fraction of learned channel retained (λ)',ylabel='Residual input rate [bits / block]',title='Entropy brackets · not symbol counts');
    for a in (ax,bx):a.spines[['right','top']].set_visible(False)
    fig.savefig(out,dpi=170);plt.close(fig)


def run(data,out,panels):
    root=Path(__file__).resolve().parents[2]/'results/l20-autoresearch';lock=json.loads((root/'information/freeze.json').read_bytes());geo=info.geometry();project,_=info.encoders();rows=[];blocks={};hashes={}
    with ZipFile(root/'frozen-inference.zip') as z:
        content=z.read('information/critic-%04d.npz'%lock['selected_round']);assert hashlib.sha256(content).hexdigest()==lock['model_sha256']
        with np.load(io.BytesIO(content),allow_pickle=False) as f:fit=dict(f)
    for idx in [0,6,8,9]:
        s,m=source(data,idx);names,val=measure(s,idx,fit,project,geo,panels);del s; stats=interval(val);blocks[str(idx)]=val;hashes[str(idx)]=m['sha256']
        for i,lvl in enumerate(LAM):
            for j,name in enumerate(names):
                v,lo,hi=stats[i,j];rows.append({'index':idx,'beta':m['beta'],'lambda':lvl,'metric':name,'mean':v,'lo':lo,'hi':hi,'unit':'bit','parent_blocks':len(val),'parents_per_block':GROUP,'anchors_per_parent':panels})
        print('Conditional channel budgets:',m['beta'],'full-channel bit bounds',stats[-1,:3,0],flush=True)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'side-information.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'parent-blocks.npz',**blocks,metrics=names,lambdas=LAM)
    (out/'provenance.json').write_text(json.dumps(dict(sources=hashes,critic_sha256=lock['model_sha256'],seed=792241,fold='historical validation only; frozen retrospective analysis, not another test',panels=panels,unit='bits, not nats',channel='pi_l(h|v) = (1-l)*q0(h|c) + l*pi(h|v), q0 = arithmetic mean of the original two frozen cold training priors. Decoder is degraded by the same mixture. Distribution over uniform independent cubic frames. Neither label nor new glass configuration sampled.',bounds='H_emp(H|C) is concave in the joint measure, hence downward biased in expectation for each independent parent block; cross entropy against the fixed q0 is an upper bound on H(H|C). Subtract E H(pi_l|V) for the residual input-rate brackets. For relevance use H_emp(H|C) + E_pi_l log2 r_l(h|E,C): Barber–Agakov conditional lower bound in expectation. The corner-only decoder null is at most zero even on the empirical distribution (Gibbs inequality). Do not interpret differences of achieved bounds as differences of true MI. Pointwise t intervals over disjoint parent blocks conditional on frozen weights.'),indent=2)+'\n');draw(rows,out/'side-information.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--panels',type=int,default=512);a=p.parse_args();run(a.data,a.out,a.panels)
