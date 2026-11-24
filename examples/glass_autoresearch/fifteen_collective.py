"""Archive development observables 05, 07 and 08; no independent-site errors."""
import argparse,json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.stats import t
from fifteen_topology import at,interval,hot_diff
from scalar_rule import scalar_counts
from tile_codec import POW,TILE
from boundary_templates import dictionary,PERIOD
from rate import json_out,digest

R=Path(__file__).resolve().parents[2]/'results/l20-autoresearch';N=8000;SH=(20,)*3;C=np.array(list(product(range(3),repeat=3)));A=(-1.)**C[:,0];DEL=np.moveaxis((C[None,:,:]-C[:,None,:]+[0,0,5])%20,-1,0);D=np.indices(SH);DIST=np.minimum(D,20-D);DISP=np.max(DIST,axis=0);ORIENT=[np.ravel_multi_index(tuple(D[i]%PERIOD[j] for j,i in enumerate([k for k in range(3) if k!=axis]+[axis])),PERIOD).ravel() for axis in range(3)]

def corr(x,y):
    ax=tuple(range(-3,0));z=np.fft.rfftn(x,axes=ax);f=np.fft.rfftn(y,axes=ax);return np.fft.irfftn(z.conj()*f,s=x.shape[-3:],axes=ax).real/np.prod(x.shape[-3:])

def cross_contract(c,d):
    # Exclude precisely i=j and k=l, then add their intersection back once.
    return c.sum()*d.sum()-c.sum(0)@d.sum(0)-c.sum(1)@d.sum(1)+(c*d).sum()

def diffraction(s):
    v=(s==0).astype(float)-.25;am=sum(w*at(v,r) for w,r in zip(A,C));selfs=sum(at(v*v,r) for r in C);it=(am*am-selfs)/27;c=corr(v,v)[tuple(DEL)]*np.outer(A,A)
    return it.mean(),(it*at(it,(0,0,5))).mean(),c

def cumulant(x,y):
    a,ab,c=x;b,cd,d=y
    return np.r_[(ab+cd)/2-a*b,2*cross_contract(c,d)/729,(ab+cd)/2-a*b-2*cross_contract(c,d)/729]

def dense(s):return sum(at(s,v).astype('i4')*int(w) for v,w in zip(TILE,POW))

def packing(s,hit):
    v=(s.ravel()==0).astype(float);T=dictionary()[0];scores=[np.bincount(ix,weights=v,minlength=20)@T for ix in ORIENT];mx=max(z.max() for z in scores);eq=[z==mx for z in scores];nt=sum(j.sum() for j in eq);avg=sum((T[:,j].sum(1)[ix] if j.any() else 0) for j,ix in zip(eq,ORIENT))/nt;k=((v-.25)*(avg-.25)/(.25*.75)).reshape(SH);K=k.mean();assert abs(K-(mx/N-.0625)/.1875)<1e-9
    m=hit[dense(s)];M=int(m.sum());profile=np.bincount(DISP.ravel(),weights=corr(k,m).ravel(),minlength=11);assert abs(profile.sum()-K*M)<1e-8
    occ=s!=0;degree=sum(np.roll(occ,j,axis=i).astype('i1') for i,j in product(range(3),(-1,1)));E=int(np.sum(occ*(degree-np.where(s==1,3,5))**2));return K,M,E,profile,nt

def affinity(p):
    a,b=p[::2],p[1::2];k,m,e=p[:,:3].T;k0,k1=a[:,0],b[:,0];m0,m1=a[:,1],b[:,1];cross=.5*(k0*m1+k1*m0);freq=np.bincount(DISP.ravel())/N;decomp=.5*(a[:,3:14]+b[:,3:14])-cross[:,None]*freq;paired=.5*(k0-k1)*(m0-m1);assert np.allclose(decomp.sum(1),paired,atol=1e-8);n=len(p);j=np.arange(n);d=e-e.mean();proj=np.mean(d*(m-m.mean()))*np.mean(d*(k-k.mean()))/np.mean(d*d)*n/(n-1);return paired,decomp,float(proj)

def channels(s,labels=None):
    if labels is None:return np.stack([s==1,s==2]).astype(float)
    u=labels[dense(s)[::2,::2,::2]];return (u[None]==np.arange(4)[:,None,None,None]).astype(float)

def overlap(g):
    # A/B are the disjoint measured replica pair. Four separate control parents supply
    # unbiased centering constants at the two ends (necessary for a fluctuating code).
    a,b,c,d,e,f,h,i=g;w=lambda u:u.mean((1,2,3),keepdims=True);q=((a-w(c))*(b-w(d))).sum(0);z=((a-w(e))*(b-w(f))).sum(0);direct=corr(q,z);sh=q.shape;rs=np.arange(sh[0]);rgrid=sum((np.minimum(x,sh[0]-x))**2 for x in np.indices(sh));by=np.bincount(rgrid.ravel(),weights=direct.ravel());mat=np.zeros((len(a),len(a)));rhs=np.zeros_like(direct)
    # C is estimated without a noisy forced box mean: differences between two independent parents.
    u=(c-h)/np.sqrt(2.);v=(e-i)/np.sqrt(2.)
    for s in range(len(a)):
        for k in range(len(a)):
            z=corr(u[s],u[k])*corr(v[s],v[k]);rhs+=z;mat[s,k]=z.sum()
    return np.r_[direct.sum(),rhs.sum(),direct.sum()-rhs.sum()],by,np.bincount(rgrid.ravel(),weights=rhs.ravel()),mat

def run(a):
    o=a.out;o.mkdir(exist_ok=True,parents=True);spec=json.loads((R/'faces/scalar-rule.json').read_bytes());names,x=scalar_counts(np.arange(6561)[:,None]//POW%3);hit=x@np.array([spec['coefficients'][z] for z in names])>0;lut=np.load(R/'codec/comparators.npz')['rsmi_lut'];summaries={k:{} for k in ('05','07','08')};arrays={};cs={};masses=np.bincount(DISP.ravel())
    for idx,beta in ((9,3.05),(0,0.),(8,2.711111111111)):
        key=f'b{idx:02}'
        with np.load(a.bank/(key+'.npz')) as bank:states=bank['species'];ids=bank['parent']
        # fixed cap for the heavier crystal registry; same disjoint parent units for comparisons.
        n=len(ids);f=[];p=[];groups={k:[] for k in ('species','coarse')};dr={k:[] for k in groups};rr={k:[] for k in groups};sm={k:[] for k in groups}
        for start in range(0,n,8):
            block=states[start:start+8];
            for j in range(0,8,2):f.append(cumulant(diffraction(block[j]),diffraction(block[j+1])))
            for s in block:
                K,M,E,part,tie=packing(s,hit);p.append(np.r_[K,M,E,part,tie])
            for k,h in [('species',None),('coarse',lut)]:
                v,db,rb,ma=overlap([channels(s,h) for s in block]);groups[k].append(v);dr[k].append(db);rr[k].append(rb);sm[k].append(ma)
        f=np.asarray(f);p=np.asarray(p);cs[idx]=f[:,2];S=summaries['05'][key]=dict(independent_parents=n,disjoint_parent_pairs=n//2,self_removed_intensity_cov=interval(f[:,0]),both_pair_contractions=interval(f[:,1]),four_point_remainder=interval(f[:,2]),normalization='1/27 per intensity; q=(pi,0,0); disjoint windows +5 x');arrays[key+'_parent']=ids;arrays[key+'_f5']=f;arrays[key+'_f7']=p
        paired,dec,proj=affinity(p);S=summaries['07'][key]=dict(independent_parents=n,pairs=n//2,K=interval(p[:,0]),M=interval(p[:,1]),beta_cov=interval(beta*paired),contact_cheb_0_or_1=interval(beta*dec[:,:2].sum(1)),noncontact_cheb_ge2=interval(beta*dec[:,2:].sum(1)),displacement_slope=(beta*dec.mean(0)).tolist(),energy_linear_slope=beta*proj,fixed_density=True,energy_residual_slope=beta*(np.cov(p[:,0],p[:,1])[0,1]-proj),energy_share_note='Energy-linear projection over all development parents; not a state intervention. Registry depends on whole box.');
        T=summaries['08'][key]={}
        for k in groups:
            z=np.stack(groups[k]);rad=np.stack(rr[k]);md=np.stack(sm[k]);top=np.argsort(-abs(rad.mean(0)));T[k]=dict(independent_parents=n,disjoint_measured_replica_pairs=len(z),independent_groups=len(z),observed_chi=interval(z[:,0]),two_point_chi=interval(z[:,1]),paired_discrepancy=interval(z[:,2]),dominant_squared_displacements=top[:5].tolist(),dominant_contributions=rad.mean(0)[top[:5]].tolist(),r0_rhs=interval(rad[:,0]),species_channel_sums=md.mean(0).tolist(),reference='disjoint pairs of parent differences give independent connected correlations; two independent nuisance means per replica for coarse counts, not per-box fixed means');arrays[key+'_'+k+'_08']=z;arrays[key+'_'+k+'_radial_direct']=np.stack(dr[k]);arrays[key+'_'+k+'_radial_pair']=rad
    summaries['05']['cold_minus_beta0']=hot_diff(cs[9],cs[0]);summaries['08']['beta0_species_exact']=float((np.diag([.3,.45])-np.outer([.3,.45],[.3,.45])).__pow__(2).sum()*N/(N-1));np.savez_compressed(o/'collective-parents.npz',**arrays)
    for k,d in summaries.items():json_out(o/f'{k}-collective.json',dict(results=d,parent_evidence_sha256=digest(o/'collective-parents.npz'),scope='Discovery / calibration only; uncertainty is across disjoint whole-parent groups.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
