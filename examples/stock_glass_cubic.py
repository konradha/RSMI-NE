"""06 cubic actions on the fixed 223 state table. All transforms are active on states B(gx)."""
from itertools import permutations,product
from pathlib import Path
import numpy as np
from scipy.special import xlogy
from scipy.linalg import eigh
from stock_glass_cavity_exact import X,C,A,LOOK,CODE

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/experiment-06-cubic-families';REF=ROOT/'results/experiment-03-cavity';SEEDS=(6101,6102,6103);SPACES={'identity':1,'proper':24,'full':48};LABELS=np.array(list(permutations(range(4))));N=len(X)
# class order 1, eight C3, three face-axis C2, six C4, six edge-axis C2.
CHAR={'A1':[1,1,1,1,1],'A2':[1,1,1,-1,-1],'E':[2,-1,2,0,0],'T1':[3,0,-1,1,-1],'T2':[3,0,-1,-1,1]}


def geometry():
    eye=np.eye(3,dtype='i1');mats=[eye[list(k)]*np.array(s,'i1')[:,None] for k in permutations(range(3)) for s in product((-1,1),repeat=3)];proper=sorted((m for m in mats if np.linalg.det(m)>0),key=lambda m:(not np.array_equal(m,eye),tuple(m.ravel())));mat=np.stack(proper+[-m for m in proper]);site=[];ix=[];classes=[]
    for m in mat:
        d=[np.flatnonzero((C==c).all(1)).item() for c in C@m.T];site.append(d);ix.append(LOOK[X[:,np.argsort(d)]@CODE].astype(int))
        if len(classes)<24:
            tr=np.trace(m);classes.append(0 if tr==3 else 1 if tr==0 else 3 if tr==1 else 2 if np.array_equal(np.diag(np.diag(m)),m) else 4)
    return mat,np.array(site),np.array(ix),np.array(classes)


def projectors(ix,classes):
    ops=np.eye(N)[ix];basis={};dense={};ranks={}
    for ir,ch in CHAR.items():
        coeff=ch[0]*np.array(ch)[classes]/48.
        for par,sign in [('g',1),('u',-1)]:
            name=ir+par;P=np.tensordot(coeff,ops[:24]+sign*ops[24:],axes=1);np.testing.assert_allclose(P,P.T,atol=1e-14);np.testing.assert_allclose(P@P,P,atol=1e-14);ev,v=eigh(P);assert np.max(np.minimum(abs(ev),abs(ev-1)))<1e-12;basis[name]=v[:,ev>.5];dense[name]=P;ranks[name]=basis[name].shape[1]
    block=np.column_stack(list(basis.values()));np.testing.assert_allclose(block.T@block,np.eye(N),atol=2e-12);np.testing.assert_allclose(sum(dense.values()),np.eye(N),atol=1e-14);assert sum(ranks.values())==N
    return basis,ranks


def gram_basis(f,m):
    f=np.asarray(f,dtype='d').reshape(N,-1);z=np.sqrt(m[:,None])*(f-m@f);u,s,_=np.linalg.svd(z,full_matrices=False);use=s*s>max(1e-14,1e-12*s[0]**2);return u[:,use]


def entropy(q):return -np.sum(xlogy(q,q),axis=-1)


def joints(m,ab):
    # One 4x4 table per candidate group action; m always one side's marginal only.
    return np.bincount((ab+16*np.arange(len(ab))[:,None]).ravel(),weights=np.broadcast_to(m,ab.shape).ravel(),minlength=16*len(ab)).reshape(-1,4,4)


def selects(m,left,right,ix):
    ab=4*left[None,:]+right[ix];tab=joints(m,ab);na=entropy(tab.sum(2));nb=entropy(tab.sum(1));hj=entropy(tab.reshape(-1,16));good=(np.unique(left).size>1 and np.unique(right).size>1);nmi=2*(na+nb-hj)/(na+nb) if good else np.full(len(ix),np.nan)
    # A bijection exactly relates two partitions iff every occupied row/column has just one partner.
    struct=joints(np.ones(N),ab)>0;exact=(struct.sum(1).max(1)==1)&(struct.sum(2).max(1)==1);out={}
    for space,n in SPACES.items():
        if good:
            best=np.max(nmi[:n]);ties=np.flatnonzero(nmi[:n]>=best-1e-12);g=int(ties[0]);near=int(np.sum(nmi[:n]>=best-.01));nt=len(ties)
        else:g=0;near=nt=0
        opts=tab[g][LABELS,np.arange(4)].sum(1);per=LABELS[np.flatnonzero(opts>=opts.max()-1e-12)[0]];err=per[right[ix[g]]]!=left
        out[space]=dict(g=g,perm=per,train_NMI=nmi[g],train_disagree=float(m@err),ties=nt,near=near,exact_count=int(exact[:n].sum()),exact_chosen=not bool(err.any()),different_states=int(err.sum()),nonconstant=bool(good))
    return out


def audit(mat,site,ix,classes,basis,ranks):
    ids={tuple(m.ravel()):i for i,m in enumerate(mat)};assert len(ids)==48 and np.unique(ix,axis=0).shape[0]==48 and np.array_equal(ix[0],np.arange(N));np.testing.assert_array_equal(np.bincount(classes),[1,8,3,6,6]);ch=np.array(list(CHAR.values()));np.testing.assert_allclose((ch*np.bincount(classes))@ch.T/24,np.eye(5),atol=1e-14)
    for i,m in enumerate(mat):
        assert site[i,0]==0 and np.array_equal(A[np.ix_(site[i],site[i])],A) and np.array_equal(np.sort(ix[i]),np.arange(N));assert np.array_equal(X[ix[i]][:,site[i]],X);assert np.array_equal(ix[ids[tuple(m.T.ravel())]][ix[i]],np.arange(N));assert round(np.linalg.det(m))==(1 if i<24 else -1)
        for j,h in enumerate(mat):
            k=ids[tuple((m@h).ravel())];np.testing.assert_array_equal(ix[i,ix[j]],ix[k]);assert not (i<24 and j<24 and k>=24)
    rng=np.random.default_rng(609000);orig=rng.integers(4,size=N);train=np.arange(1.,N+1);train/=train.sum();test=train[::-1];plants=[]
    for j in (9,35):
        # right(gx) recovers original up to category names.
        right=np.array([2,3,1,0])[orig[ix[ids[tuple(mat[j].T.ravel())]]]];selected=selects(train,orig,right,ix)
        for space,r in selected.items():
            err=float(test@(orig!=r['perm'][right[ix[r['g']]]]));plants.append(dict(planted=j,space=space,chosen=r['g'],exact=r['exact_count'],held_disagree=err))
            if space=='full' or (j<24 and space=='proper'):assert r['g']==j and r['exact_count']==1 and err==0
            if j>=24 and space=='proper':assert r['exact_count']==0 and err>0
    # This vector representation is odd under inversion; the center vacancy is an even invariant.
    vec=(X!=0)@C;mass=np.ones(N)/N;vec=gram_basis(vec,mass);vac=gram_basis(X[:,0]==0,mass);assert vec.shape[1]==3 and np.isclose(np.linalg.norm(basis['T1u'].T@vec)**2,3);assert np.isclose(np.linalg.norm(basis['A1g'].T@vac)**2,1)
    return dict(proper=24,full=48,group_products=48**2,inverses=48,permuted_states=N,classes=np.bincount(classes).tolist(),character_ranks=ranks,fixture=plants,arm_vector='T1u, three dimensions',center_vacancy='A1g',projector_sum_error_limit=1e-14)
