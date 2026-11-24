"""Evaluate a frozen stock code on independent conditional replicas; no critic queries or TEST fits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit,xlogy,roots_legendre
from stock_glass_cavity_memory import moments,solve,orient,evaluate,interval,compare,add,loadside
from stock_glass_cavity_exact import X,C
import stock_glass_modal as stock


def relaxed(z,tau,n):
    nodes,weight=roots_legendre(n);draw=expit((z[...,None]+logit(.5+.5*nodes))/tau);return np.stack((draw@(weight/2),draw**2@(weight/2)),axis=-1)


def entropy(p):return -np.sum(xlogy(p,p),axis=-1)


def distribution(z):
    hard=(2*(z[:,0]>=0)+(z[:,1]>=0)).astype(int);hot=expit(z);cold=expit(-z);q=np.stack([cold[:,0]*cold[:,1],cold[:,0]*hot[:,1],hot[:,0]*cold[:,1],hot[:,0]*hot[:,1]],axis=1)
    return {'hard':np.eye(4)[hard],'stochastic':q}


def contrast(p):
    m,c,s=moments(p);use=m>1e-14;use[m.argmax()]=False  # drop constant direction; all positive-mass categories otherwise available
    if not np.any(use):return None
    ev,v=solve(c,s,np.eye(4)[:,use]);return orient(v[:,:1],m)[:,0] if len(ev) else None


def readout(inputs,out,runs,reference=None):
    ref=reference or out
    meta=json.loads((inputs/'prepared.json').read_bytes());receipt=json.loads((ref/'exact-receipt.json').read_bytes());assert receipt['inputs']==stock.sha(inputs/'prepared.json') and receipt['eigenbasis']==stock.sha(ref/'frozen-modes.npz')
    frozen=np.load(ref/'frozen-modes.npz');model=[];coverage=[];traces=[];banks={};coords=C+1;max_quad=0;all_receipts=[]
    for b in range(10):
        tag=f'b{b:02}';beta=float(meta['coverage'][b]['beta']);tr,*_=loadside(inputs,b,'train',beta);te,*_=loadside(inputs,b,'test',beta);mt=tr.mean(0);mv=te.mean(0);test={k:evaluate(te,frozen[tag+'_'+k]) for k in ('oracle','best_simple','linear')};linear=frozen[tag+'_linear'];site=frozen[tag+'_best_simple'];B=(len(tr)*16+499)//500;fin=B*((10000+B-1)//B);depths=[850,1700,3400,6800,fin];assert len(set(depths))==5
        for seed in (6101,6102,6103):
            name=f'{tag}-s{seed}';p=runs/(name+'.npz');status='worker_failed' if (runs/(name+'-failed.json')).exists() else 'unrun';recs={};vals={};steps=0;pos=0
            if p.exists():
                with np.load(p,allow_pickle=False) as z:
                    r=json.loads(z['receipt'].item());assert r['index']==b and r['seed']==seed and r['design']==meta['design'] and r['batches']==B and r['final']==fin;recs=r['checkpoints'];status=r['fit_status'];steps=r['steps'];pos=r['positives'];assert steps<=fin
                    all_receipts.append(r);trace=z['train_trace'];traces.extend(dict(index=b,beta=beta,seed=seed,update=1+i*B,stock_train_bound=x,ceiling_violation=bool(x>np.log(500)+1e-5)) for i,x in enumerate(trace));vals={k:z[k] for k in z.files if k.startswith('logits') or k.startswith('w')}
            for dep in depths:
                ck=recs.get(str(dep),{});valid=ck.get('map_valid',False);row=dict(index=b,beta=beta,seed=seed,depth=dep,final=dep==fin,steps=steps,train_parents=len(tr),test_parents=len(te),bank_pairs=16*len(tr),processed_pairs=pos,status=status,checkpoint='finite' if valid else 'invalid' if ck else 'unreached',critic_finite=ck.get('critic_weights_finite'),hard_codes=0,eligible_exteriors_splits='fixed')
                if valid:
                    z=vals[f'logits{dep}'].astype('d');assert z.shape==(210,2) and np.isfinite(z).all();w=vals[f'w{dep}'];flat=np.stack([X==1,X==2],axis=-1).reshape(210,14).astype('d');w0=w[tuple(coords.T)].reshape(14,2);bits=w0.view(np.uint32);mask=np.uint32(0xffffe000)
                    # The H100 contracts rounded TF32 operands. Replay the arithmetic for integrity only;
                    # the signed partition and noisy channel always use the saved stock logits, never a reconstruction.
                    ordinary=flat@w0;replays={'float32':ordinary,'tf32_rn':flat@((bits+4095+((bits>>13)&1))&mask).view('f4'),'tf32_rna':flat@((bits+4096)&mask).view('f4')}
                    kind=min(replays,key=lambda k:np.max(abs(z-replays[k])));np.testing.assert_allclose(z,replays[kind],rtol=2e-5,atol=3e-5);row.update(contraction=kind,replay_max_error=float(np.max(abs(z-replays[kind]))),full_precision_sign_differences=int(np.sum((ordinary>=0)!=(z>=0))))
                    tau=ck['tau'];key=name+f'_d{dep}';banks[key+'_logits']=z;rel=relaxed(z,tau,256);diff=float(np.max(abs(rel-relaxed(z,tau,512))));max_quad=max(max_quad,diff);assert diff<1e-4;banks[key+'_relaxed_1st_2nd']=rel
                    for channel,q in distribution(z).items():
                        np.testing.assert_allclose(q.sum(1),1,rtol=0,atol=1e-14);y=tr@q;v=te@q;atr=contrast(y);pm=mv@q;count=int(np.count_nonzero(pm>1e-14));rs=dict(index=b,beta=beta,seed=seed,depth=dep,final=dep==fin,channel=channel,parents=len(te),output_support=count,entropy_nats=float(entropy(pm)),entropy_given_state_nats=mv@entropy(q),stock_tau=tau,fit=status,valid_contrast=atr is not None)
                        if channel=='hard':row['hard_codes']=count
                        g=evaluate(v,atr);rtrain=evaluate(y,atr)
                        if b==0 and atr is not None:
                            assert np.max(np.ptp(v,axis=0))<1e-14 and g[0]<1e-24;g=(0.,np.zeros(len(te)));rtrain=(0.,np.zeros(len(tr)))
                        add(rs,'T',interval(g));add(rs,'fraction',compare(g,test['oracle'],True));add(rs,'minus_linear',compare(g,test['linear']));add(rs,'minus_simple',compare(g,test['best_simple']));rs['train_T']=rtrain[0];rs['half_var_mass']=np.nan;rs['corr_site']=np.nan
                        banks[key+'_'+channel+'_Q']=q;banks[key+'_'+channel+'_a']=np.full(4,np.nan) if atr is None else atr;banks[key+'_'+channel+'_test_marginal']=pm
                        if atr is not None:
                            # Correlation includes intrinsic emission noise. Absolute value ignores arbitrary eigenvector sign.
                            sf=site-mv@site;a0=atr-pm@atr;rs['corr_site']=abs(float((mv*sf)@(q@a0)))/np.sqrt((mv@(sf**2))*(pm@(a0**2)));centered=(q@(a0**2));vshare=mv*centered;idx=np.argsort(-centered);k=np.searchsorted(vshare[idx].cumsum(),.5*vshare.sum());rs['half_var_mass']=mv[idx[:k+1]].sum();rs['between_test']=float(np.var(v@atr));rs['total_test']=float(pm@(a0**2))
                        else:rs['between_test']=0.;rs['total_test']=0.
                        model.append(rs)
                coverage.append(row)
            print('READOUT',name,status,steps,flush=True)
    assert len(coverage)==150
    pd.DataFrame(model).to_csv(out/'frozen-readout.csv',index=False,float_format='%.13g');pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False);pd.DataFrame(traces).to_csv(out/'train-trace.csv',index=False,float_format='%.9g')
    np.savez_compressed(out/'code-maps.npz',**banks);stock.write_json(out/'fit-receipts.json',dict(fits=all_receipts,max_quadrature_difference=max_quad,models=len(all_receipts),checkpoints_finite=sum(r['checkpoint']=='finite' for r in coverage)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','out','runs'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--reference',type=Path);a=p.parse_args();readout(a.inputs,a.out,a.runs,a.reference)
