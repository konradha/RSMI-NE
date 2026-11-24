"""Can the frozen four-state operator tell a distant shell anything beyond ALL eight corners?

Train conditional critics only. In each block of independent parents, the positive and
all eligible negative shells have exactly matching corner labels. No marginal-bound subtraction.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from l20_operators_static import banks,source,readfit,prob,forward,backward,init,dense,netfile,VREPS,TREPS,EPOCHS

BATCH=512
CORNERS=np.array([0,2,6,8,18,20,24,26])


def data(bank,model,kind):
    vs,es=bank;meta,enc,_=model
    for v,e in zip(vs,es):
        c=v[:,CORNERS]; side=(c==1).astype(np.float32) if kind=='eight-A-bits' else np.concatenate((c==1,c==2),axis=1).astype(np.float32)
        g=(c==1)@(2**np.arange(8)) if kind=='eight-A-bits' else c@(3**np.arange(8))
        yield prob(v,3,meta['spec'],enc),np.column_stack((dense(e),side)),g


def grouped(q,x,g,weights,update):
    _,cats=np.unique(g,return_inverse=True);cnt=np.bincount(cats); h,z=forward(x,weights);k=cnt[cats,None]; z=z.astype(np.float64);z-=z.max(0);ez=np.exp(z)
    def group_sum(value):return np.stack([np.bincount(cats,weights=col) for col in value.T],1)[cats]
    denom=group_sum(ez);log_ratio=z-np.log(denom/k)
    if update:
        dz=((ez/denom)*group_sum(q)-q)/len(x)
        return backward(x,h,weights,dz.astype(np.float32))
    within=np.divide(group_sum(log_ratio)-log_ratio,k-1,out=np.zeros_like(log_ratio),where=k>1)
    return [(q*log_ratio).sum()/len(q),(q*within).sum()/len(q),np.mean(k==1),k.mean()]


def evaluate(samples,weights):
    result=[]
    for q,x,g in samples:
        result.append([grouped(q[i:i+BATCH],x[i:i+BATCH],g[i:i+BATCH],weights,False) for i in range(0,len(g),BATCH)])
    return np.mean(result,axis=0)


def fit(kind,model,train,val):
    p=init(np.random.default_rng(47581),300+(8 if kind=='eight-A-bits' else 16));m=[np.zeros_like(w) for w in p];v=[np.zeros_like(w) for w in p]
    valid=list(data(val,model,kind));history=[];best=-np.inf;step=0;chosen=None
    for epoch,sample in enumerate(data(train,model,kind),1):
        q,x,g=sample
        for i in range(0,len(g),BATCH):
            grad=grouped(q[i:i+BATCH],x[i:i+BATCH],g[i:i+BATCH],p,True);step+=1
            for w,gg,mm,vv in zip(p,grad,m,v):
                mm*=.9;mm+=.1*gg;vv*=.999;vv+=.001*gg*gg;w-=.002*np.sqrt(1-.999**step)/(1-.9**step)*mm/(np.sqrt(vv)+1e-8)
        if epoch%3==0:
            value=evaluate(valid,p)[:,0].mean();history.append([epoch,value]);
            if value>best:best=value;chosen=[w.copy() for w in p]
    return chosen,history


def ci(values):
    m=values.mean();d=t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values));return float(m),float(m-d),float(m+d)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--static',type=Path,default=Path('results/l20-operators/static'));a=p.parse_args();o=a.out;o.mkdir(exist_ok=True,parents=True)
    model=readfit(netfile(a.static/'fits',3,1,'learned',11));s,_=source(a.data,9);tr=banks(s,9,3,1,0,EPOCHS);val=banks(s,9,3,1,1,VREPS); del s
    kinds=('eight-A-bits','eight-ternary-species');weights={};hist={}
    for k in kinds:weights[k],hist[k]=fit(k,model,tr,val)
    np.savez_compressed(o/'conditional-critics.npz',**{f'{k}-{j}':w for k,ws in weights.items() for j,w in enumerate(ws)});del tr,val
    receipt=dict(frozen_encoder='w3-b1-learned-s11',analysis='post-discovery; freeze encoder. Conditional denominator uses only shells from other independent parents with same entire corner pattern, plus the positive. K=1 gives 0. Weighted per positive, not per stratum. Diagnostic null uses all within-stratum mismatches.',corner_flat_zyx=CORNERS.tolist(),batch=BATCH,epochs=EPOCHS,validation_history=hist,critic_sha256=hashlib.sha256((o/'conditional-critics.npz').read_bytes()).hexdigest(),transfer=[7,8,9,0])
    (o/'freeze.json').write_text(json.dumps(receipt,indent=2)+'\n');rows=[];per={}
    for idx in receipt['transfer']:
        s,meta=source(a.data,idx);test=banks(s,idx,3,1,2,TREPS);del s
        for k in kinds:
            x=evaluate(data(test,model,k),weights[k]);per[f'i{idx}-{k}']=x;m,lo,hi=ci(x[:,0]);sh=x[:,1].mean();rows.append(dict(index=idx,beta=meta['beta'],condition=k,blocks=len(x),parents=len(x)*BATCH,raw=m,lo=lo,hi=hi,shuffled=sh,singleton_fraction=x[:,2].mean(),average_candidate_count=x[:,3].mean()));print(rows[-1],flush=True)
    with (o/'all-corners.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(o/'parent-blocks.npz',**per)

if __name__=='__main__':main()
