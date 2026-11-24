"""06 frozen-function diagnostics; no optimization on held parents and no training."""
import argparse,json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from stock_glass_cubic import X,N,C,REF,OUT,SEEDS,SPACES,geometry,projectors,audit,selects,gram_basis,entropy
from stock_glass_cavity_memory import loadside
import stock_glass_modal as stock


def loggrad(p):return -np.log(p,where=p>0,out=np.zeros_like(p))


def held(m,x,y,per,fa,fb,sign):
    joint=np.bincount(4*x+y,weights=m,minlength=16).reshape(4,4);a=joint.sum(1);b=joint.sum(0);ha,hb,hj=entropy(a),entropy(b),entropy(joint.ravel());da=loggrad(a)[x];db=loggrad(b)[y];dj=loggrad(joint)[x,y];ok=np.count_nonzero(a)>1 and np.count_nonzero(b)>1;zeros=np.zeros(N);nan=(np.nan,zeros);out={}
    if ok:
        nmi=2*(ha+hb-hj)/(ha+hb);assert -1e-12<nmi<1+1e-12;out['NMI']=(nmi,(2*(da+db-dj)-nmi*(da+db))/(ha+hb))
    else:out['NMI']=nan
    good=(x==per[y]).astype('d');ac=m@good;pchance=a[per]@b;gchance=b[np.argsort(per)[x]]+a[per[y]];out.update(agreement=(ac,good),disagreement=(1-ac,-good),HA=(ha,da),HB=(hb,db),HA_given_B=(hj-hb,dj-db),HB_given_A=(hj-ha,dj-da));k=(ac-pchance)/(1-pchance) if ok and 1-pchance>1e-14 else np.nan;out['kappa']=(k,(good+(k-1)*gchance)/(1-pchance)) if np.isfinite(k) else nan
    if np.isfinite(fa).all() and np.isfinite(fb).all():
        l=fa-m@fa;r=fb-m@fb;sl=np.sqrt(m@(l*l));sr=np.sqrt(m@(r*r));
        if min(sl,sr)>1e-10:
            l/=sl;r*=sign/sr;rho=m@(l*r);out['scalar_r']=(rho,l*r-.5*rho*(l*l+r*r))
    out.setdefault('scalar_r',nan);return out


def diff(a,b):return a[0]-b[0],a[1]-b[1]


def write_metrics(row,metrics,V):
    keys=list(metrics);g=np.stack([metrics[k][1] for k in keys]);err=np.sqrt(np.maximum(0,np.sum((g@V)*g,axis=1)))
    for j,k in enumerate(keys):val=metrics[k][0];row[k]=val;row[k+'_lo']=val-err[j];row[k+'_hi']=val+err[j]


def actions(b,A,B,ta,tb,mt,mv,V,ix,draw):
    x,aa=ta;y,ab=tb;res=selects(mt,x,y,ix);rows={};metrics={}
    for space,alignment in res.items():
        g=alignment['g'];yy=y[ix[g]];fx=aa[x];fy=ab[yy];cov=mt@((fx-mt@fx)*(fy-mt@fy));sign=-1 if cov<0 else 1;p=alignment['perm'];metrics[space]=held(mv,x,yy,p,fx,fy,sign);row=dict(index=b,seed_A=A,seed_B=B,draw=draw,space=space,**{k:v for k,v in alignment.items() if k not in ('perm','g')},transform=g,improper=g>=24,train_scalar_sign=sign,label_B_to_A=''.join(map(str,p)),chance=(mv@np.eye(4)[x])[p]@(mv@np.eye(4)[yy]));rows[space]=row
        comp={'identity':'identity','proper':'identity','full':'proper'}[space];metrics[space]['gain_NMI']=diff(metrics[space]['NMI'],metrics[comp]['NMI']);write_metrics(row,metrics[space],V)
    return rows,metrics


def null_map(labels,seed,draw):return labels[np.random.default_rng([609006,seed,draw]).permutation(N)]


def reference_spaces(modes,b,m):
    tag=f'b{b:02}';eig=modes[tag+'_eigenvalues'];vectors=modes[tag+'_train_modes'];ref={'packing':gram_basis(np.concatenate([X==1,X==2],axis=1),m),'center_vacancy':gram_basis((X[:,0]==0),m)}
    for width in (.01,.005,.001):ref[f'band_{width}']=gram_basis(vectors[:,eig[0]-eig<=width],m)
    return ref


def decompose(z,ir,refs):
    rank=z.shape[1];pows={k:float(np.linalg.norm(Q.T@z)**2/rank) if rank else np.nan for k,Q in ir.items()};assert not rank or abs(sum(pows.values())-1)<2e-10
    over={k:float(np.linalg.norm(Q.T@z)**2/rank) if rank else np.nan for k,Q in refs.items()};assert not rank or all(-1e-12<=x<=1+1e-10 for x in over.values());return pows,over


def spectrum(b,beta,m,mt,ir,refs,actual):
    powers=[];overlaps=[];nullpowers=[];nulloverlaps=[]
    def record(name,seed,draw,z):
        ctx=dict(index=b,beta=beta,function=name,seed=seed,draw=draw,rank=z.shape[1]);p,ovs=decompose(z,ir,refs);which=nullpowers if draw else powers;which.append(dict(**ctx,**p,invariant=p['A1g'],even=sum(v for k,v in p.items() if k.endswith('g')),odd=sum(v for k,v in p.items() if k.endswith('u'))));dst=nulloverlaps if draw else overlaps
        for k,o in ovs.items():dst.append(dict(**ctx,reference=k,reference_rank=refs[k].shape[1],overlap=o,isotropic_chance=refs[k].shape[1]/(N-1)))
    for name,z in refs.items():record(name,0,0,z)
    for s,(x,a) in actual.items():
        for draw in range(65):
            y=null_map(x,s,draw) if draw else x;code=gram_basis(np.eye(4)[y],m);score=gram_basis(a[y],m) if np.isfinite(a[y]).all() else np.empty((N,0));record('labels',s,draw,code);record('scalar',s,draw,score)
    return powers,overlaps,nullpowers,nulloverlaps


def run(banks,ref,out):
    spec=json.loads((out/'design.json').read_bytes());assert all(stock.sha(ref/k)==v for k,v in spec['pins'].items());meta=json.loads((banks/'prepared.json').read_bytes());assert stock.sha(banks/'prepared.json')==spec['pins']['parent-census.json'];out.mkdir(parents=True,exist_ok=True);mat,site,ix,cls=geometry();ir,ranks=projectors(ix,cls);geom=audit(mat,site,ix,cls,ir,ranks)
    geompath=out/'geometry-audit.json'
    if geompath.exists():assert json.loads(geompath.read_bytes())==geom
    else:stock.write_json(geompath,geom)
    np.savez_compressed(out/'group-actions.npz',matrices=mat,site_permutations=site,state_permutations=ix,proper_classes=cls);real=[];null=[];power=[];ov=[];pn=[];on=[];weights={};exam=[];examples=[];max_var_diff=0.
    with np.load(ref/'corrected/code-maps.npz') as maps,np.load(ref/'frozen-modes.npz') as modes:
        for b,record in enumerate(meta['coverage']):
            beta=float(record['beta']);tag=f'b{b:02}';d=10218 if b==9 else 10257
            for side in ('train','test'):assert stock.sha(banks/f'{tag}-{side}.npz')==meta['inputs'][f'{tag}-{side}.npz']
            tr,*_=loadside(banks,b,'train',beta);te,*_=loadside(banks,b,'test',beta);mt=tr.mean(0);mt/=mt.sum();mv=te.mean(0);mv/=mv.sum();avg=mt[ix].mean(0);avg/=avg.sum();assert np.all(avg>0) and np.max(abs(avg[ix]-avg))<1e-13 and len(te)==record['test']['selected_parents'];weights[tag+'_invariant_train']=avg;weights[tag+'_raw_train']=mt;weights[tag+'_raw_test']=mv
            z=te-te[0];z-=z.mean(0);V=(z.T@z)*(t.ppf(.975,len(te)-1)**2/(len(te)*(len(te)-1)));assert np.isfinite(V).all();del tr
            truth={}
            for seed in SEEDS:
                k=f'{tag}-s{seed}_d{d}';q=maps[k+'_hard_Q'];xs=q.argmax(1);np.testing.assert_array_equal(q,np.eye(4)[xs]);logs=maps[k+'_logits'];np.testing.assert_array_equal(xs,2*(logs[:,0]>=0)+(logs[:,1]>=0));truth[seed]=xs,maps[k+'_hard_a']
            for A,B in combinations(SEEDS,2):
                raw,metrics=actions(b,A,B,truth[A],truth[B],mt,mv,V,ix,0);controls={k:[] for k in SPACES}
                for draw in range(1,65):
                    sa,sb=truth[A],truth[B];nrows,nmetrics=actions(b,A,B,(null_map(sa[0],A,draw),sa[1]),(null_map(sb[0],B,draw),sb[1]),mt,mv,V,ix,draw)
                    for k in SPACES:
                        nrows[k].update(beta=beta,train_parents=record['train']['selected_parents'],test_parents=len(te));null.append(nrows[k]);controls[k].append(nmetrics[k]['NMI'])
                for k in SPACES:
                    row=raw[k];stats=metrics[k];controls_k=controls[k];vals=np.array([v[0] for v in controls_k]);avgnull=(vals.mean(),np.mean([v[1] for v in controls_k],axis=0));write_metrics(row,{'null_mean_NMI':avgnull,'true_minus_null_NMI':diff(stats['NMI'],avgnull)},V);row.update(null_p025_NMI=np.quantile(vals,.025),null_p975_NMI=np.quantile(vals,.975),beta=beta,train_parents=record['train']['selected_parents'],test_parents=len(te));real.append(row)
                    # Independent verification: explicit parent influences agree with the fast state-covariance SE.
                    v,g=stats['NMI'];width=t.ppf(.975,len(te)-1)*np.std(te@g,ddof=1)/np.sqrt(len(te));delta=abs(width-(row['NMI_hi']-v)) if np.isfinite(v) else 0.;assert delta<1e-11;max_var_diff=max(max_var_diff,delta)
                if b==9 and (A,B)==(6101,6102):
                    row=raw['full'];g=row['transform'];p=np.array(list(map(int,row['label_B_to_A'])));x=truth[A][0];y=truth[B][0];order=np.argsort(-mt,kind='stable')
                    for j,k in enumerate(order):examples.append(dict(state=int(k),train_mass_order=j+1,pattern=''.join('.AB'[n] for n in X[k]),transformed_pattern=''.join('.AB'[n] for n in X[ix[g,k]]),raw_train_mass=mt[k],raw_test_mass=mv[k],A=x[k],B_native=y[k],B_transformed=y[ix[g,k]],B_matched=p[y[ix[g,k]]],equal=x[k]==p[y[ix[g,k]]],transform=g))
            refs=reference_spaces(modes,b,avg);pp,oo,pnn,onn=spectrum(b,beta,avg,mt,ir,refs,truth);power+=pp;ov+=oo;pn+=pnn;on+=onn;exam.append(dict(index=b,beta=beta,train=record['train']['selected_parents'],test=len(te),train_effective_states=np.exp(entropy(mt)),orbit_averaged_effective_states=np.exp(entropy(avg)),test_effective_states=np.exp(entropy(mv)),train_max_mass_change=float(abs(avg-mt).max()),top_eigenvalue=modes[tag+'_eigenvalues'][0],**{k:Q.shape[1] for k,Q in refs.items()}));print('SCORED_CUBIC',b,flush=True)
    for key,rows in [('seed-pairs',real),('random-pairs',null),('symmetry-power',power),('reference-overlap',ov),('random-power',pn),('random-overlap',on),('support',exam),('cold-example',examples)]:pd.DataFrame(rows).to_csv(out/(key+'.csv'),index=False,float_format='%.13g')
    np.savez_compressed(out/'state-masses.npz',**weights);stock.write_json(out/'receipt.json',dict(design=stock.sha(out/'design.json'),source_commit=spec['reference_commit'],parent_manifest=stock.sha(banks/'prepared.json'),pairs=len(real),null_rows=len(null),test_parent_counts=[r['test']['selected_parents'] for r in meta['coverage']],irrep_ranks=ranks,max_direct_vs_gram_parent_CI_error=max_var_diff,spend_usd=0,simulation=False,retrained=False,started_07=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--banks',required=True,type=Path);p.add_argument('--reference',default=REF,type=Path);p.add_argument('--out',default=OUT,type=Path);a=p.parse_args();run(a.banks,a.reference,a.out)
