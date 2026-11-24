"""Experiment 03: exact fixed-inventory conditional enumeration. No new physical simulation."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.stats import t
import stock_glass_modal as stock

DESIGN=Path(__file__).resolve().parents[1]/'results/experiment-03-cavity/design.json'
DIR=np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
C=np.vstack([[0,0,0],DIR]); U=np.unique((C[:,None]+DIR).reshape(-1,3),axis=0); U=U[np.abs(U).sum(1)==2]
A=(np.abs(C[:,None]-C).sum(2)==1).astype('i2'); J=(np.abs(U[:,None]-C).sum(2)==1).astype('i2')
X=np.array([x for x in itertools.product(range(3),repeat=7) if x.count(0)==2 and x.count(1)==2],dtype='u1'); L=np.array([0,3,5],dtype='i2');O=(X!=0).astype('i2'); M=O@A.T;R=M-L[X];D=O@J.T
CODE=3**np.arange(6,-1,-1);LOOK=np.full(2187,255,'u1');LOOK[X@CODE]=np.arange(210)
assert X.shape==(210,7) and U.shape==(18,3) and np.all(A.sum(1)+J.sum(0)==6)


def energies(fc,su,fu):
    # Only x-dependent terms remain: interior penalties plus all touched occupied
    # exterior penalties. fu excludes cavity sites; the dropped squared term is a boundary constant.
    fc=fc.astype('i2');fu=fu.astype('i2');live=(su!=0).astype('i2')
    E=(O*R**2).sum(1)[None,:]+2*fc@(O*R).T+fc**2@O.T+2*(live*(fu-L[su]))@D.T+live@(D**2).T
    return (E-E.min(1)[:,None]).astype('i2')


def prob(e,beta):
    logp=-float(beta)*(e-e.min(1,keepdims=True)).astype('d');logp-=np.logaddexp.reduce(logp,axis=1,keepdims=True);p=np.exp(logp);p/=p.sum(1,keepdims=True);return p


def full(lat):
    n=lat!=0;m=sum(np.roll(n,j,i).astype('i2') for i in range(3) for j in (-1,1));return int((n*(m-L[lat])**2).sum())


def audit(lat,anchor,fc,su,fu,expected,beta):
    where=tuple(((anchor+C)%20).T);copy=lat.copy();obs=lat[where].copy();e=[]
    for x in X:copy[where]=x;e.append(full(copy))
    assert np.array_equal(expected,e-np.min(e));assert np.array_equal(obs,lat[where]);assert np.array_equal(np.sort(obs),[0,0,1,1,2,2,2]);p=prob(expected[None,:],beta)[0]
    if beta==0:np.testing.assert_allclose(p,1/210,atol=1e-15,rtol=0)
    # Active cubic transform: new geometry's entry at rotated coordinate equals old entry.
    for axes in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            ix=lambda pos:np.array([np.where((pos==z).all(1))[0][0] for z in pos[:,axes]*signs]);ic,iu=ix(C),ix(U);rc,ru=np.argsort(ic),np.argsort(iu)
            out=energies(fc[None,rc],su[None,ru],fu[None,ru])[0];perm=LOOK[X[:,rc]@CODE];assert np.array_equal(out[perm],expected)
    return dict(beta=beta,anchor=anchor.tolist(),periodic_wrap=bool(np.any((anchor+C<0)|(anchor+C>=20))),full_energies=210,cubic_transforms=48,max_integer_delta_error=0)


def prepare(a):
    release=a.archive/'release-index.json';assert (stock.sha(release),stock.sha(a.split),stock.sha(a.rosters))==(stock.INDEX,stock.SPLIT,stock.ROSTERS);a.out.mkdir(parents=True,exist_ok=False)
    split=json.loads(a.split.read_bytes()); geometry=np.indices((6,6,6)).reshape(3,-1).T*3;assert np.unique(((geometry[:,None]+C)%20).reshape(-1,3),axis=0).shape==(216*7,3)
    np.savez_compressed(a.out/'state-geometry.npz',states=X,star=C,touch=U,inner_edges=A,cavity_to_outer=J,anchors=geometry,ell=L);manifest={};counts=[];verify=[];seen=set()
    with np.load(a.rosters) as roster:
        for ent in json.loads(release.read_bytes())['endpoints']:
            b=ent['release_index'];tag=f'b{b:02}';src=a.archive/ent['npz'];assert stock.sha(src)==ent['npz_sha256'];reg=split['per_beta'][str(b)];bt=float(reg['beta']);rec=dict(index=b,beta=reg['beta'],archive=ent['npz'],archive_sha=ent['npz_sha256']);rr=[roster[tag+'_'+s] for s in ('train','test')];assert np.array_equal(np.sort(np.concatenate(rr)),np.arange(16384))
            with np.load(src) as native:
                ids=np.stack([native['chain_seed'],native['chain_id']],axis=1).astype('<u8');shift=np.empty((len(ids),3),'u1');order=np.empty((len(ids),216),'u1')
                for j,(seed,cid) in enumerate(ids):
                    rng=np.random.default_rng([309031,b,int(seed),int(cid)]);shift[j]=rng.integers(20,size=3);order[j]=rng.permutation(216)
                lat=native['species']
            for side,rows in zip(('train','test'),rr):
                keys=ids[rows];assert len(rows)==8192 and hashlib.sha256(keys.tobytes()).hexdigest()==reg[side+'_identity_sha256'];seen.update(map(tuple,keys.tolist()));allcounts=[];blocks=[]
                for start in range(0,len(rows),128):
                    ii=rows[start:start+128];vlat=lat[ii];j=np.arange(len(ii));anc=(shift[ii,None,:]+geometry[order[ii]])%20;pos=(anc[:,:,None,:]+C)%20;sp=vlat[j[:,None,None],*np.moveaxis(pos,-1,0)];ok=((sp==0).sum(2)==2)&((sp==1).sum(2)==2);num=ok.sum(1);allcounts.append(num.astype('u1'));sel=np.flatnonzero(num>0);sidx=ok[sel].argmax(1);c=anc[sel,sidx];state=sp[sel,sidx];obs=LOOK[state@CODE];assert (obs<210).all()
                    def gather(delta):
                        xyz=(c.reshape((-1,)+(1,)*(delta.ndim-1)+(3,))+delta)%20;ix=sel.reshape((-1,)+(1,)*(delta.ndim-1));return vlat[ix,*np.moveaxis(xyz,-1,0)]
                    su=gather(U);occ=(state!=0).astype('i2');fc=(gather(C[:,None]+DIR)!=0).sum(2)-occ@A.T;fu=(gather(U[:,None]+DIR)!=0).sum(2)-occ@J.T;assert np.min(fc)>=0 and np.min(fu)>=0
                    E=energies(fc,su,fu);assert np.max(E)<32767;blocks.append((start+sel,c,obs,fc.astype('u1'),su,fu.astype('u1'),E))
                j,c,obs,fc,su,fu,E=map(np.concatenate,zip(*blocks));num=np.concatenate(allcounts);assert np.array_equal(j,np.flatnonzero(num));boundary=np.flatnonzero(((c==0)|(c==19)).any(1));assert len(boundary)>0
                ix=int(boundary[0]);verify.append(dict(index=b,side=side,roster_row=int(j[ix]),**audit(lat[rows[j[ix]]],c[ix],fc[ix],su[ix],fu[ix],E[ix],bt)))
                f=a.out/f'{tag}-{side}.npz';np.savez_compressed(f,all_ids=keys,native_rows=rows,all_shifts=shift[rows],eligible_counts=num,selected=j.astype('<u2'),anchor=c.astype('u1'),observed=obs,fc=fc,su=su,fu=fu,relative_E=E);manifest[f.name]=stock.sha(f)
                rec[side]=dict(roster=8192,selected_parents=len(j),eligible_windows=int(num.sum()),candidates=8192*216,misses=int((num==0).sum()),min_eligible=int(num.min()),max_eligible=int(num.max()),identity=reg[side+'_identity_sha256'])
                if side=='train':
                    pr=prob(E,bt);cum=pr.cumsum(1);cum[:,-1]=1.;pair=np.empty((len(j),16,2),'u1')
                    for h,(seed,cid) in enumerate(keys[j]):
                        rng=np.random.default_rng([309032,b,int(seed),int(cid)]);pair[h]=np.searchsorted(cum[h],rng.random((16,2)),side='right')
                    f=a.out/f'{tag}-pairs.npz';np.savez_compressed(f,pairs=pair,selected=j.astype('<u2'),ids=keys[j],states=X);manifest[f.name]=stock.sha(f);rec['sample_equal_fraction']=float(np.mean(pair[:,:,0]==pair[:,:,1]));rec['exact_equal_fraction']=float((pr*pr).sum(1).mean())
            counts.append(rec);print('PREPARED',b,rec['train']['selected_parents'],rec['test']['selected_parents'],flush=True)
    assert len(seen)==163840;manifest['state-geometry.npz']=stock.sha(a.out/'state-geometry.npz');stock.write_json(a.out/'prepared.json',dict(base=stock.BASE,split=stock.SPLIT,design=stock.sha(DESIGN),unique=len(seen),inputs=manifest,coverage=counts,audit=verify))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('archive','split','rosters','out'):p.add_argument('--'+name,type=Path,required=True)
    prepare(p.parse_args())
