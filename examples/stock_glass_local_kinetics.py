"""08: finite, reversible, six-bond Metropolis generator. No sampler, GPU or trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,expm
from scipy.sparse.csgraph import connected_components
from stock_glass_cavity_exact import X,LOOK,CODE,prob
from stock_glass_cubic import geometry,SEEDS
from stock_glass_modal import sha

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/experiment-08-local-kinetics';REF=ROOT/'results/experiment-03-cavity';N=len(X)
NAMES=('6101','6102','6103','center_vacancy','site_linear','common_slow','common_supported');LAGS=np.array([1.,10.,100.,1000.]);WIDTH=np.array([.0001,.001,.01]);INFO=('gap','resolution','unresolved_count','thermal_states','top_probability','min_eigenvalue_below_floor','stationary_error','balance_error','row_error','norm_error','slope_error','band_rank_0001','band_rank_001','band_rank_01')

def graph():
    # Directed target by bond for the independent construction and analytic checks.
    dest=[]
    for a in range(1,7):
        z=X.copy();z[:,[0,a]]=X[:,[a,0]];dest.append(LOOK[z@CODE].astype(int))
    dest=np.stack(dest,axis=1);i,k=np.nonzero(dest>np.arange(N)[:,None]);j=dest[i,k];assert np.unique(i*N+j).size==len(i)
    adj=np.zeros((N,N),bool);adj[i,j]=True;adj[j,i]=True;assert connected_components(adj,directed=False)[0]==1
    types=sorted(set(tuple(sorted((int(X[x,0]),int(X[y,0])))) for x,y in zip(i,j)));assert types==[(0,1),(0,2),(1,2)] and N==210
    return dest,i,j
DEST,I,J=graph()

def lap(weight):
    d=np.zeros((N,N));d[I,J]=d[J,I]=-weight;np.fill_diagonal(d,-d.sum(1));return d

def bank(b,side,root,meta):
    name=f'b{b:02}-{side}.npz';assert sha(root/name)==meta['inputs'][name]
    with np.load(root/name) as f:sel=f['selected'];ids=f['all_ids'][sel];en=f['relative_E'];assert len(ids)==len(np.unique(ids,axis=0))==meta['coverage'][b][side]['selected_parents'];return ids,sel,en

def common(p,cut):
    mean=p.mean(0);ess=p.sum(0)**2/np.sum(p*p,axis=0);keep=mean>=1e-10 if not cut else (mean>=.001)&(ess>=100)
    if not keep.any():
        assert cut
        return np.full(N,np.nan),dict(included=0,rank=0,mask=keep,w_cutoff=np.nan,train_rate=np.nan,train_within=np.nan,train_ess=np.nan,near_min=0,eigenvalues=np.array([]),train_mass=mean,state_parent_ess=ess)
    W=-(p.T@p)/len(p);np.fill_diagonal(W,0);np.fill_diagonal(W,-W.sum(1));D=lap(np.mean(np.minimum(p[:,I],p[:,J]),axis=0)/6)
    idx=np.where(keep)[0];wc=W[np.ix_(idx,idx)];l,v=eigh(wc,driver='evd');tol=max(1e-14,1e-12*l[-1]);ok=l>tol;wh=v[:,ok]/np.sqrt(l[ok]);dr=wh.T@D[np.ix_(idx,idx)]@wh;l,u=eigh((dr+dr.T)/2,driver='evd');assert len(l)>0 and l.min()>-1e-10
    f=np.zeros(N);f[idx]=wh@u[:,0];f-=mean@f;f*=1 if f[np.argmax(abs(f))]>=0 else -1
    dv=np.einsum('ij,ij->i',p,(f[None,:]-(p@f)[:,None])**2);di=np.mean(np.minimum(p[:,I],p[:,J]),axis=0)@((f[I]-f[J])**2)/6;slope=di/dv.mean();np.testing.assert_allclose([dv.mean(),slope],[1,l[0]],atol=2e-9,rtol=1e-8)
    return f,dict(included=idx.size,rank=ok.sum(),mask=keep,w_cutoff=tol,train_rate=slope,train_within=dv.mean(),train_ess=dv.sum()**2/(dv@dv),near_min=np.sum(l-l[0]<=1e-10),eigenvalues=l,train_mass=mean,state_parent_ess=ess)

def freeze(banks,out,meta,spec):
    assert sha(REF/'frozen-modes.npz')==spec['modes_sha'] and sha(REF/'corrected/code-maps.npz')==spec['maps_sha'];data=dict(names=NAMES,states=X,edges=np.stack([I,J],axis=1),lags=LAGS,widths=WIDTH);rows=[]
    with np.load(REF/'frozen-modes.npz') as modes,np.load(REF/'corrected/code-maps.npz') as maps:
        for b,rec in enumerate(meta['coverage']):
            tag=f'b{b:02}';beta=float(rec['beta']);ids,sel,en=bank(b,'train',banks,meta);p=prob(en,beta);cols=[]
            for seed in SEEDS:
                k=f'{tag}-s{seed}_d{10218 if b==9 else 10257}_hard';q=maps[k+'_Q'];a=maps[k+'_a'];assert np.isfinite(a).all() or (np.unique(q.argmax(1)).size==1 and np.isnan(a).all());cols.append(q@a)
            cols.extend([(X[:,0]==0).astype(float),modes[tag+'_linear']]);
            for restricted in (False,True):
                f,r=common(p,restricted);cols.append(f);k=tag+('_supported_' if restricted else '_common_');
                for v in ('eigenvalues','mask','train_mass','state_parent_ess'):data[k+v]=r.pop(v)
                rows.append(dict(index=b,beta=beta,ntrain=len(ids),reference='common_supported' if restricted else 'common_slow',**{k:float(v) for k,v in r.items()}))
            data[tag+'_functions']=np.stack(cols,axis=1);data[tag+'_train_ids']=ids
    np.savez_compressed(out/'frozen-functions.npz',**data);import pandas as pd;pd.DataFrame(rows).to_csv(out/'train-variational.csv',index=False);return sha(out/'frozen-functions.npz')

def kernel(e,beta):
    delta=beta*(e[J]-e[I]).astype(float);a=np.exp(-np.maximum(delta,0))/6.;b=np.exp(np.minimum(delta,0))/6.;degree=np.bincount(I,a,N)+np.bincount(J,b,N);h=np.diag(degree);h[I,J]=h[J,I]=-np.exp(-abs(delta)/2)/6;return h,a,b

def independent(e,beta):
    q=np.zeros((N,N));x=np.arange(N)
    for k in range(6):
        y=DEST[:,k];i=x[x!=y];j=y[i];q[i,j]+=np.exp(-beta*np.maximum(e[j]-e[i],0))/6
    np.fill_diagonal(q,-q.sum(1));return q

def verify_graph():
    q=independent(np.zeros(N),0.);arm=(X[:,1]==0).astype(float)-(X[:,2]==0);center=(X[:,0]==0).astype(float)-2/7
    np.testing.assert_allclose(q@arm,-arm/6,atol=2e-15);np.testing.assert_allclose(q@center,-7*center/6,atol=2e-15)
    m,sites,actions,_=geometry();rel=LOOK[np.where(X==0,1,np.where(X==1,0,2))@CODE];x=np.arange(N,dtype=float);e=((x*13+7)%19);base=independent(e,.719);mx=0.
    for perm in [*actions,rel]:mx=max(mx,float(np.max(abs(independent(e[perm],.719)-base[np.ix_(perm,perm)]))))
    assert mx<1e-15;return dict(states=N,undirected_edges=len(I),components=1,edge_types={f'{a}-{b}':int(np.sum((np.minimum(X[I,0],X[J,0])==a)&(np.maximum(X[I,0],X[J,0])==b))) for a,b in combinations(range(3),2)},beta0_arm_rate=1/6,beta0_center_rate=7/6,cubic_permutations=48,inventory_preserving_symbol_permutations_checked=1,permutation_max_error=mx)

def measure(e,beta,p,F,audit=False,qr_checks=None):
    h,a,b=kernel(e,beta);s=np.sqrt(p);w=s.copy();w[-1]+=1.;w/=np.linalg.norm(w);hw=h@w;v=w[:-1];d=h[:-1,:-1]-2*(np.outer(v,hw[:-1])+np.outer(hw[:-1],v))+4*(w@hw)*np.outer(v,v)
    try:
        rates,U=eigh(d,driver='evd',check_finite=False)
    except np.linalg.LinAlgError:
        # LAPACK divide-and-conquer can fail on a clustered spectrum. QR on the same unmodified matrix.
        rates,U=eigh(d,driver='ev',check_finite=True)
        residual=float(np.max(abs(d@U-U*rates)));assert residual<1e-11
        if qr_checks is not None:qr_checks.append(residual)
    floor=max(1e-12,100*N*np.finfo(float).eps*abs(h).sum(1).max());assert rates[0]>=-floor and rates[-1]<=2+floor
    g=F-F[p.argmax()];mu=g.T@p;g-=mu;mu+=F[p.argmax()];Z=s[:,None]*g;Y=Z[:-1]-2*v[:,None]*(w@Z);coef=U.T@Y;power=coef**2;var=(Z*Z).sum(0);dirich=(np.minimum(p[I],p[J])/6)@((F[I]-F[J])**2);rr=np.maximum(rates,0);ct=np.exp(-LAGS[:,None]*rr)@power;bands=(rates[None,:]<=rates[0]+WIDTH[:,None]);bp=bands@power;mass=p[I]*a;bal=np.max(abs(mass-p[J]*b));stat=np.max(abs(h@s));rowerr=abs(np.bincount(I,a,N)+np.bincount(J,b,N)-h.diagonal()).max();normerr=np.max(abs(power.sum(0)-var)/np.maximum(var,1e-300));slopeerr=np.max(abs(rates@power-dirich)/np.maximum(var,1e-300));assert normerr<1e-10 and slopeerr<1e-10 and bal<1e-12 and stat<1e-12 and np.all(ct>=0)
    gap=rates[0] if rates[0]>floor else np.nan;info=[gap,floor,sum(rates<=floor),1/(p@p),p.max(),max(0.,-rates[0]),stat,bal,rowerr,normerr,slopeerr,*bands.sum(1)];check={}
    if audit:
        q=independent(e,beta);db=np.max(abs(p[:,None]*q-(p[:,None]*q).T));assert np.min(q[~np.eye(N,dtype=bool)])>=0 and np.min(a)>0 and np.min(b)>0
        flux=np.max(abs(p@q));row=np.max(abs(q.sum(1)));vec=np.vstack([U,np.zeros(N-1)])-2*w[:,None]*(v@U);res=np.max(abs(h@vec-vec*rates));err=0.
        for j,t in enumerate(LAGS):
            cv=(g*p[:,None]*(expm(t*q)@g)).sum(0);err=max(err,np.max(abs(cv-ct[j])/np.maximum(var,1e-300)))
        assert max(err,row,db,flux,res)<1e-9;check=dict(direct_nonsymmetric_expm_max_error=float(err),dense_stationarity=float(flux),dense_row_sum=float(row),dense_detailed_balance=float(db),max_eigenpair_residual=float(res),smallest_legal_rate=float(min(a.min(),b.min())))
    return np.vstack((mu,var,dirich,ct,bp)),info,check

def evaluate(args):
    b,banks,out,meta,lock,resume=args;tag=f'b{b:02}';beta=float(meta['coverage'][b]['beta']);assert sha(out/'frozen-functions.npz')==lock
    with np.load(out/'frozen-functions.npz') as maps:F=maps[tag+'_functions'];train=maps[tag+'_train_ids']
    valid=np.isfinite(F).all(0)&(np.ptp(F,axis=0)>0);F[:,~valid]=0.;ids,sel,en=bank(b,'test',banks,meta);assert not {tuple(x) for x in train}.intersection(map(tuple,ids));uniq,inv=np.unique(en if beta else np.zeros_like(en),axis=0,return_inverse=True);p=prob(uniq,beta);stats=np.empty((len(uniq),3+len(LAGS)+len(WIDTH),len(NAMES)));info=np.empty((len(uniq),len(INFO)));check={}
    path=out/f'parents-{tag}.npz';qr=[]
    if resume and path.exists():
        j=inv[0];first,_,check=measure(uniq[j],beta,p[j],F,True)
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['ids'],ids);np.testing.assert_array_equal(saved['valid'],valid);np.testing.assert_array_equal(saved['roster_rows'],sel)
            old=saved['moments'];assert old.shape==(len(ids),stats.shape[1],len(NAMES)) and np.isfinite(old).all();np.testing.assert_allclose(old[0],first,rtol=2e-10,atol=1e-13)
        return dict(index=b,beta=beta,test=len(ids),distinct_generators_evaluated=len(uniq),dense_check_roster_row=int(sel[0]),retained_completed=True,parent_archive_sha=sha(path),**check)
    for i,e in enumerate(uniq):
        stats[i],info[i],ck=measure(e,beta,p[i],F,i==inv[0],qr)
        if ck:check=ck
    np.savez_compressed(path,ids=ids,roster_rows=sel,valid=valid,moments=stats[inv],fields=['parent_mean','within_variance','dirichlet',*[f'cov_{t:g}' for t in LAGS],*[f'band_{w:g}' for w in WIDTH]],spectral=info[inv],spectral_fields=INFO,names=NAMES);print('EXACT_KINETICS',b,len(ids),len(uniq),flush=True)
    return dict(index=b,beta=beta,test=len(ids),distinct_generators_evaluated=len(uniq),dense_check_roster_row=int(sel[0]),qr_recoveries=len(qr),max_qr_residual=max(qr,default=0.),parent_archive_sha=sha(path),**check)

def run(banks,out,workers,resume=None):
    spec=json.loads((OUT/'design.json').read_bytes());assert sha(banks/'prepared.json')==spec['manifest_sha'];meta=json.loads((banks/'prepared.json').read_bytes());out.mkdir(parents=True,exist_ok=True);graphcheck=verify_graph()
    lock=resume if resume else freeze(banks,out,meta,spec);assert sha(out/'frozen-functions.npz')==lock;print('FROZEN_TRAIN_FUNCTIONS',lock,flush=True)
    with ProcessPoolExecutor(workers) as pool:checks=list(pool.map(evaluate,[(b,banks,out,meta,lock,bool(resume)) for b in range(10)]))
    (out/'matrix-receipt.json').write_text(json.dumps(dict(graph=graphcheck,design=sha(OUT/'design.json'),frozen_hash=lock,bank_manifest=spec['manifest_sha'],audits=checks,new_trajectories=0,gpu=False,modal_usd=0,sequence_finished=True),indent=2)+'\n')
if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--banks',required=True,type=Path);pa.add_argument('--out',type=Path,default=OUT);pa.add_argument('--workers',type=int,default=5);pa.add_argument('--resume-frozen',metavar='SHA256');args=pa.parse_args();run(args.banks,args.out,args.workers,args.resume_frozen)
