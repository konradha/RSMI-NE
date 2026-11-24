"""Cold finite-channel operator spectroscopy: cubic symmetry, explicit species contributions,
and held-fixed transfer correlations. Maximal correlations are not information bounds."""
from itertools import permutations, product
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.stats import t
from face_bank import TILE,FACES,POW
from rate import digest,json_out

D8=np.arange(6561)[:,None]//POW%3
P4=3**np.arange(4);D4=np.arange(81)[:,None]//P4%3
REP=('A1g','A2g','Eg','T1g','T2g','A1u','A2u','Eu','T1u','T2u');DIMS=(1,1,2,3,3)*2


def rotations():
    from sectors import cubic
    _,weights,_,_=cubic(); sites=2*TILE-1;faces=2*np.vstack(FACES)-1;maps=[]
    lookup={tuple(v):i for i,v in enumerate(faces)}
    for order,sg in product(permutations(range(3)),product((-1,1),repeat=3)):
        g=np.eye(3,dtype=int)[list(order)]*np.array(sg)[:,None];new=(sites@g.T+1)//2;perm=np.ravel_multi_index(new.T,(2,2,2));vt=(D8@POW[perm]).astype(int)
        pos=np.array([lookup[tuple(v)] for v in faces@g.T]).reshape(6,4);y=[]
        for face in pos:
            if np.any(face//4!=face[0]//4):raise ValueError('invalid rotated face')
            y.extend(face[0]//4*81+D4@P4[face%4])
        if len(np.unique(vt))!=6561 or len(set(y))!=486:raise ValueError('invalid group action')
        maps.append((vt,np.array(y)))
    return maps,weights.astype(float)


def basis(d,degree):
    x=np.column_stack([np.ones(len(d)),d==1,d==2]).astype(float)
    if degree==1:return x
    cols=[x]
    for i in range(d.shape[1]):
        for j in range(i):
            cols += [((d[:,i]==s)&(d[:,j]==q))[:,None] for s,q in product((1,2), repeat=2)]
    return np.column_stack(cols)


def regression(x,w,f):
    gram=x.T@(w[:,None]*x);rhs=x.T@(w[:,None]*f);coef=np.linalg.pinv(gram,hermitian=True,rcond=1.e-11)@rhs;return x@coef


def census(file):
    a=np.load(file,mmap_mode='r',allow_pickle=False);return a.transpose(1,0,2).reshape(6561,486).astype(float)


def train(a):
    meta=json.loads((a.bank/'meta.json').read_bytes());cf=a.bank/'train.npy';seal=a.out/'mode-lock.json'
    if seal.exists() or meta['index']!=9 or meta['train_sha256']!=digest(cf):raise ValueError('expected unsealed cold training bank')
    raw=census(cf);maps,weights=rotations();p=np.zeros_like(raw);proj=np.zeros((10,486,486));idx=np.arange(486)
    for n,(vv,yy) in enumerate(maps):
        p+=raw[np.argsort(vv)][:,np.argsort(yy)];proj[:,yy,idx]+=weights[:,n,None]
    p/=p.sum();pv=p.sum(1);py=p.sum(0);z=np.sqrt(np.outer(pv,py));K=np.divide(p-pv[:,None]*py,z,out=np.zeros_like(p),where=z>0);G=K.T@K;space=basis(D8,1);space2=basis(D8,2);ar={};ds={};ref=[]
    for r,(rep,d) in enumerate(zip(REP,DIMS)):
        if not np.allclose(proj[r],proj[r].T,atol=1e-12):raise ValueError('asymmetric cube projector')
        eig,b=np.linalg.eigh(proj[r]);b=b[:,eig>.5];
        if not b.shape[1]:continue
        gram=b.T@G@b;v,q=np.linalg.eigh(gram);v=np.maximum(v[-d:],0.);sig=np.sqrt(v.mean());
        if v.max()-v.min()>1.e-9:raise ValueError('symmetry multiplet split')
        if sig<1e-8:continue
        psi=np.divide(b@q[:,-d:],np.sqrt(py)[:,None],out=np.zeros((486,d)),where=py[:,None]>0);f=np.divide(p@psi,(pv*sig)[:,None],out=np.zeros((6561,d)),where=pv[:,None]>0);lin=regression(space,pv,f);pair=regression(space2,pv,f);res=f-lin
        if abs((pv[:,None]*f*f).sum()/d-1)>1e-6 or not np.allclose(proj[r]@(np.sqrt(py)[:,None]*psi),np.sqrt(py)[:,None]*psi,atol=1e-7):raise ValueError('unphysical mode normalization')
        for key,field in [('visible',f),('exterior',psi),('linear',lin),('pair',pair),('residual',res)]:ar[rep+'_'+key]=field
        supported=pv>0;ds[rep]=dict(amplitude=float(sig),multiplet=d,dimension=int(b.shape[1]),linear_fraction=float((pv[:,None]*lin**2).sum()/d),pair_fraction=float((pv[:,None]*pair**2).sum()/d),center=float(np.max(np.abs(pv@f))),strongest_amplitude_is_not_MI=True)
        ref.append(rep)
    # All cubic harmonic sectors must reproduce the unconstrained largest singular value.
    winner=max(ref,key=lambda r:ds[r]['amplitude']);unconstrained=np.sqrt(np.linalg.eigvalsh(G)[-1]);
    if abs(unconstrained-ds[winner]['amplitude'])>1e-8:raise ValueError('leading sector failed to account for full channel mode')
    np.savez_compressed(a.out/'modes.npz',**ar,train_visible_mass=pv);metadata=dict(version=1,train_meta=meta,code=digest(Path(__file__)),weights=digest(a.out/'modes.npz'),winning_sector=winner,sectors=ds,
        calibration='Average the cold joint physical (tile, registered face) channel over all 48 cube operations, rotate both variables together. Decompose the normalized centered probability matrix by exact O_h projectors. Freeze the leading maximal-correlation singular subspace of each represented sector. Project source singular functions by weighted least squares onto all additive species indicators, then onto all species pair indicators. Operator on unseen tiles set to zero at fit; report support at transfer. No rotations or outcomes appended as independent parents.',
        prediction='At beta index 4, the fixed strongest cold sector retains positive cross-correlation, and at least half its oriented tile-face covariance is reproduced by its fixed cold one-site species projection. Test the paired linear contrast Cov(linear,exterior)-0.5*Cov(full,exterior) without ratios. Both 95% lower endpoints must be positive; opposite outcomes reported. This is a separate interpretable-operator diagnostic, not evidence for a thermodynamic transition or for defeating a full conventional model.',
        thermal='Only archived native validation packets. Score cold and beta8 descriptively and beta4 as predeclared warm primary; beta0 finite-composition control. Never optimize fields at transfer. The largest singular value/correlation does not equal mutual information. Uncertainty conditional on the cold trained fields. When training support falls below 0.95 at transfer report the numerical effect but do not promote the mechanism.')
    json_out(seal,metadata);print('MODE',winner,ds[winner],flush=True)


def parentstats(c,f,psi,lin,pair,support):
    v=c.sum(1);s=c.sum(0);n=v.sum();d=f.shape[1]; xp=c@psi/n
    return np.r_[1.,v@support/n,s@psi/n,*[q for field in (f,lin,pair) for q in (v@field/n,(xp*field).sum()/d)]]


def summary(v,d):
    # Per-group sum of centered-correlator sufficient moments; equal group sizes.
    def calc(r):
        ey=r[...,2:2+d];start=2+d;cov=[]
        for _ in range(3):
            ex=r[...,start:start+d];raw=r[...,start+d];cov.append(raw-(ey*ex).mean(-1));start+=d+1
        return np.stack([r[...,1],*cov,cov[0]-cov[1],cov[1]-.5*cov[0]],axis=-1)
    m=calc(v.mean(0));jack=calc((v.sum(0)-v)/(len(v)-1));se=np.sqrt((len(v)-1)*np.mean((jack-jack.mean(0))**2,axis=0));a=t.ppf(.975,len(v)-1)*se;return {k:dict(value=float(x),lo95=float(x-h),hi95=float(x+h)) for k,x,h in zip(('support','covariance','additive','pairs','beyond_additive','half_retention_margin'),m,a)}


def score(a):
    doc=json.loads((a.out/'mode-lock.json').read_bytes());meta=json.loads((a.bank/'meta.json').read_bytes());idx=meta['index'];out=a.out/f'modes-{idx:02d}.json'
    if out.exists() or idx not in (0,4,8,9) or doc['code']!=digest(Path(__file__)) or doc['weights']!=digest(a.out/'modes.npz') or meta['validation_sha256']!=digest(a.bank/'validation.npy'):raise ValueError('mode lock, validation bank or output mismatch')
    reps=list(doc['sectors']);vals={r:[] for r in reps};c=np.load(a.bank/'validation.npy',allow_pickle=False,mmap_mode='r')
    with np.load(a.out/'modes.npz') as z:fields={r:[z[r+'_'+k] for k in ('visible','exterior','linear','pair')] for r in reps};sup=(z['train_visible_mass']>0).astype(float)
    for packet in c:
        b=packet.transpose(1,0,2).reshape(6561,486).astype(float)
        for r in reps:vals[r].append(parentstats(b,*fields[r],sup))
    result={r:summary(np.array(v),doc['sectors'][r]['multiplet']) for r,v in vals.items()};winner=doc['winning_sector'];h=result[winner];decision='supports both covariance predictions' if h['support']['value']>=.95 and min(h['covariance']['lo95'],h['half_retention_margin']['lo95'])>0 else 'does not establish both predictions';json_out(out,dict(source=meta,lock_sha256=digest(a.out/'mode-lock.json'),primary=winner,decision=decision if idx==4 else 'descriptive',fields=result)); print('MODE_TRANSFER',idx,winner,decision,h,flush=True)


def plot(a):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    doc=json.loads((a.out/'mode-lock.json').read_bytes());rs=list(doc['sectors']);fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3.6),layout='constrained');y=np.arange(len(rs));ax.barh(y,[doc['sectors'][r]['amplitude'] for r in rs],color='#167b80');ax.set(yticks=y,yticklabels=rs,xlabel='Leading cold singular value',title='Cubic symmetry of the face channel');rows=[json.loads((a.out/f'modes-{i:02d}.json').read_bytes()) for i in (0,4,8,9)];x=[r['source']['beta'] for r in rows]
    for key,label in [('covariance','Full operator'),('additive','One-site projection'),('beyond_additive','Nonadditive remainder')]:
        v=np.array([[r['fields'][doc['winning_sector']][key][s] for s in ('value','lo95','hi95')] for r in rows]);m,l,h=v.T;bx.errorbar(x,m,yerr=[m-l,h-m],fmt='o-',label=label,capsize=3)
    bx.axhline(0,color='.6',lw=.7);bx.set(title='Frozen '+doc['winning_sector']+' operator',xlabel='Inverse temperature β',ylabel='Tile–exterior covariance');bx.legend(fontsize=8,frameon=False)
    for a0 in (ax,bx):a0.spines[['top','right']].set_visible(False)
    fig.savefig(a.out/'modes.png',dpi=165);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('train','score','plot'));p.add_argument('--bank',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);{'train':train,'score':score,'plot':plot}[a.mode](a)
