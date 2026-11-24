"""Archive-only cubic-irrep field tomography. Run one bounded validation stage.

The field is the posterior at fixed lab axes, not its critic, hard assignment,
sector norm or orbit average. See results/l20-autoresearch/sectors/design.json.
"""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import xlogy
from scipy.stats import t

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))
from l20_operators_static import OUT as STATIC, dense, netfile, offsets, readfit, softmax, source, table
from l20_pairs import L, partition

TARGET = (64, 128, 256, 512, 1024, 2048)
INDICES = (9, 7, 0)
PACK, REPS, SEED = 8, 4, 596121
NAMES = ('nonlinear', 'affine', 'two_A_bits', 'residual')
SECT = tuple(ir + par for par in 'gu' for ir in ('A1', 'A2', 'E', 'T1', 'T2')) + ('full',)
FAMILIES = ('axis', 'face', 'body')
WAVES = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1], [2,0,0], [10,10,10]])
RAYS = np.array([d for d in itertools.product((-1,0,1), repeat=3) if any(d) and next(x for x in d if x) == 1])
FAMILY_RAYS = [RAYS[np.count_nonzero(RAYS, axis=1) == i] for i in (1,2,3)]
LAGS = [(f, k) for f in range(3) for k in range(1,L//2) if (f+1)*k*k < (L//2)**2]
SHELL = [(r, k, p) for r,d in enumerate(RAYS) for k in (3,4,6) if (d@d)*k*k < (L//2)**2 for p in ('even','odd')]
CONTROL = ('A_contact', 'A_distal', 'A_plant', 'A_plant_gain', 'A_expected_gain', 'A_shuffle', 'A_k0_exact', 'B_k0_exact')
VIS = offsets(3,1)[0]
VT = table(VIS)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def save(path, values):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def cubic():
    mats = np.array([np.eye(3,dtype=int)[list(axes)] * np.array(signs)[:,None]
                     for axes in itertools.permutations(range(3)) for signs in itertools.product((-1,1),repeat=3)])
    # Class order: identity, transposition, disjoint transpositions, 3-cycle, 4-cycle.
    cycles = [(1,1,1,1),(1,1,2),(2,2),(1,3),(4,)]
    chars = np.array([[1,1,1,1,1],[1,-1,1,1,-1],[2,0,2,-1,0],[3,-1,-1,0,1],[3,1,-1,0,-1]])
    diag = FAMILY_RAYS[2]; cls=[]; parity=[]; perm=[]
    for g in mats:
        det = int(round(np.linalg.det(g))); parity.append(det)
        # g^-1 V sampled at a block coordinate x reads V(g x).
        perm.append(np.ravel_multi_index((VIS@g.T + 1).T, (3,)*3))
        rot = det*g
        s4 = np.argmax(np.abs((diag@rot.T)@diag.T),axis=1)
        if len(set(s4)) != 4: raise ValueError('body-diagonal permutation invalid')
        left=set(range(4)); lens=[]
        while left:
            j=next(iter(left)); n=0
            while j in left:
                left.remove(j); j=int(s4[j]); n+=1
            lens.append(n)
        cls.append(cycles.index(tuple(sorted(lens))))
    perm=np.array(perm); ch=chars[:,cls]; ch=np.concatenate((ch,ch*np.array(parity)))
    ident=int(np.flatnonzero(np.all(mats==np.eye(3),axis=(1,2)))[0]); dim=ch[:,ident]
    if dim@dim!=48 or not np.array_equal(ch@ch.T,48*np.eye(10)): raise ValueError('invalid character table')
    weights=(dim[:,None]*ch/48).astype('f')
    z,y,x=VIS.T
    for name, field in [('A1g',np.ones(27)),('T1u',x),('Eg',x*x-y*y),('T2g',x*z)]:
        goal=np.zeros((10,27)); goal[SECT.index(name)]=field
        if not np.allclose(weights.astype(float)@field[perm],goal,atol=1e-6): raise ValueError(f'wrong {name} action')
    lookup={tuple(g.flat):i for i,g in enumerate(mats)}
    mul=np.array([[lookup[tuple((g@h).flat)] for h in mats] for g in mats])
    return perm,weights,ident,mul


class Fields:
    def __init__(self, cold_train, work, cold_sha):
        _,enc,_=readfit(netfile(STATIC/'fits',3,1,'learned',11)); w,b,u,a,c=enc
        af=STATIC.parent/'affine'; receipt=json.loads((af/'freeze.json').read_text()); ap=af/f"seed-{receipt['affine']['seed']}.npz"
        if sha(ap)!=receipt['affine_weights_sha256']: raise ValueError('affine weights changed')
        with np.load(ap,allow_pickle=False) as f: aff=f['A']; bias=f['c']
        self.provenance={'static_archive_sha256':sha(STATIC/'fit-receipts.zip'),'nonlinear':'w3-b1-learned-s11','affine_sha256':sha(ap),'affine_seed':receipt['affine']['seed'],'train_source':cold_sha}
        self.perm,self.char,self.ident,self.mul=cubic()
        # Fixed additive projection onto all 54 one-site indicators plus intercept.
        # No validation score is used for fitting, selecting a penalty or centering.
        template=work/'additive.npz'; key=json.dumps(self.provenance,sort_keys=True)
        if not template.exists():
            rng=np.random.default_rng([SEED,1]); gram=np.zeros((55,55)); xy=np.zeros((55,4))
            for state in np.array_split(cold_train,16):
                anchors=rng.integers(L**3,size=(len(state),24)); v=state[np.arange(len(state))[:,None,None],VT[anchors]].reshape(-1,27)
                x=dense(v); scores=softmax(np.tanh(x@w+b)@u+x@a+c); design=np.column_stack((x,np.ones(len(x))))
                gram+=design.T@design; xy+=design.T@scores
            coef=np.linalg.solve(gram,xy).astype('f'); save(template,dict(coef=coef,key=key))
        with np.load(template,allow_pickle=False) as f:
            if str(f['key'])!=key: raise ValueError('incompatible template in --work')
            coef=f['coef']
        self.provenance.update(template_sha256=sha(template),template_training_parents=len(cold_train),template_anchors_per_parent=24)
        self.w=np.concatenate([w,a,aff,coef[:-1]],axis=1); self.c=np.concatenate([b,c,bias,coef[-1]]); self.u=u; self.h=len(b)

    def orbit(self, v):
        rotated=v[:,self.perm]; n=len(v); flat=rotated.reshape(-1,27)
        z=dense(flat)@self.w+self.c; h=self.h
        non=softmax(np.tanh(z[:,:h])@self.u+z[:,h:h+4]); af=softmax(z[:,h+4:h+8]); fix=np.eye(4,dtype='f')[2*(flat[:,0]==1)+(flat[:,-1]==1)]
        return np.stack((non,af,fix,non-z[:,h+8:]),axis=1).reshape(n,48,4,4)

    def check(self,v):
        q=self.orbit(v).astype(float); # [block, group, model, label]
        c=self.char.astype(float)
        # Evaluate every P_r f at every transformed block, not just the identity.
        p=np.einsum('sh,bghml->bgsml',c,q[:,self.mul],optimize=True)
        twice=np.einsum('tg,bgsml->btsml',c,p,optimize=True)
        target=np.eye(10)[None,:,:,None,None]*p[:,self.ident,None,:,:,:]
        err=float(np.max(np.abs(twice-target))); rec=float(np.max(np.abs(p.sum(2)-q)))
        # The orbit integration is the inner product for central-projector orthogonality.
        gram=np.einsum('bgsml,bgtml->bst',p,p)/48
        off=float(np.max(np.abs(gram*(1-np.eye(10)))))
        if max(err,rec,off)>3e-6: raise ValueError('observed orbit projector failure')
        return dict(observed_blocks=len(v),max_double_projection_error=err,max_orbit_gram_offdiag=off,max_abs_reconstruction=rec,dimensions_squared_sum=48,analytic_fields=['A1g count','T1u polar vector','Eg diagonal quadrupole','T2g off-diagonal quadrupole'])

    def project(self,v):
        out=np.empty((len(v),4,11,4),'f')
        for i in range(0,len(v),256):
            q=self.orbit(v[i:i+256]); n=len(q)
            p=(self.char@q.reshape(n,48,16)).reshape(n,10,4,4).transpose(0,2,1,3)
            out[i:i+n,:,:10]=p; out[i:i+n,:,10]=q[:,self.ident]
        return out


def with_trace(x):
    return np.concatenate([x,x.sum(-1,keepdims=True)],axis=-1)


def connected(x,y):
    return np.mean(x*(y-np.roll(y,1,axis=0)),axis=tuple(range(x.ndim-3)),dtype=float)


def address(x):
    return np.ravel_multi_index(np.moveaxis(x%L,-1,0),(L,)*3)


def measure(states, fields, index, packet):
    rng=np.random.default_rng([SEED,2,index,packet]); origins=rng.integers(L,size=(PACK,3,REPS,3)); direction=np.stack([ray[rng.integers(len(ray),size=(PACK,REPS))] for ray in FAMILY_RAYS],axis=1)
    pos=(origins[:,:,:,None,:]+np.arange(L)[None,None,None,:,None]*direction[:,:,:,None,:])%L
    addr=address(pos); parents=np.arange(PACK)
    v=states[parents[:,None,None,None,None],VT[addr]]; field=fields.project(v.reshape(-1,27)).reshape(PACK,3,REPS,L,4,11,4)
    posterior=field[...,:3,-1,:]; mass=posterior.mean((0,1,2,3),dtype=float); entropy=-xlogy(posterior,posterior).sum(-1).mean((0,1,2,3),dtype=float)
    contact=with_trace(connected(field,field)); lag=[]; null=[]
    for f,k in LAGS:
        x=field[:,f]; y=np.roll(x,-k,axis=2)
        lag.append(with_trace(connected(x,y))); null.append(connected(x,np.roll(y,1,axis=0)).sum(-1))
    # Fourier amplitudes on independent random-line panels. Distinct subsets, not
    # distinct physical parents, are needed to remove within-field sampling noise.
    amp=[]; species=np.array([1,2]); density=states[parents[:,None,None,None],addr][...,None]==species
    for sl in (slice(0,REPS//2),slice(REPS//2,None)):
        p=pos[:,:,sl].reshape(PACK,-1,3); x=field[:,:,sl].reshape(PACK,-1,4*11*4)
        x=np.concatenate((x,density[:,:,sl].reshape(PACK,-1,2)),axis=-1)
        phase=np.exp((-2j*np.pi/L)*(p@WAVES.T)).transpose(0,2,1).astype('F'); amp.append((phase@x.astype('F'))/x.shape[1])
    def spectrum(a,b):
        return L**3*(a*(b-np.roll(b,1,axis=0)).conj()).real.mean(0,dtype=float)
    sparse=spectrum(*amp); power=sparse[:,:-2].reshape(len(WAVES),4,11,4)
    full_density=np.fft.fftn((states[...,None]==species).reshape(PACK,L,L,L,2),axes=(1,2,3))/L**3
    full_density=full_density[:,WAVES[:,0],WAVES[:,1],WAVES[:,2]]
    exact=spectrum(full_density,full_density)
    density_power=np.stack((sparse[:,-2:],exact,sparse[:,-2:]-exact),axis=1)
    # Fixed laboratory ray labels (+/- parity), never average different rays of
    # an odd or vector channel before multiplying by the frozen field.
    center=pos[:,0].reshape(PACK,-1,3); x=field[:,0].reshape(PACK,-1,4*11*4)
    diffs=np.stack([RAYS[r]*k for r,k,p in SHELL[::2]])
    a=states[parents[:,None,None],address(center[:,:,None,:]+diffs)]; b=states[parents[:,None,None],address(center[:,:,None,:]-diffs)]
    species=np.array([1,2]); a=(a[...,None]==species).astype('f'); b=(b[...,None]==species).astype('f')
    y=np.stack(((a+b)*.5,(a-b)*.5),axis=-2).reshape(PACK,x.shape[1],len(SHELL)*2)
    angular=(x.reshape(-1,x.shape[-1]).T@(y-np.roll(y,1,axis=0)).reshape(-1,y.shape[-1])/(PACK*x.shape[1])).reshape(4,11,4,len(SHELL),2)
    site=(states[parents[:,None],address(center)]==1).astype('f').reshape(PACK,REPS,L)
    distal=np.roll(site,-6,axis=2); planted=np.where(rng.random(site.shape)<.1,site,distal)
    def cov(y): return float(np.mean(site*(y-np.roll(y,1,axis=0)),dtype=float))
    c0,cd,cp=cov(site),cov(distal),cov(planted); comp=[(states==s).mean(1) for s in (1,2)]
    control=[c0,cd,cp,cp-cd,.1*(c0-cd),cov(np.roll(distal,1,axis=0)),*[float(L**3*np.mean(a*(a-np.roll(a,1)))) for a in comp]]
    return dict(mass=mass,entropy=entropy,contact=contact,lag=np.stack(lag),null=np.stack(null),power=with_trace(power),density_power=density_power,angular=angular,control=np.array(control))


def interval(a,alpha=.05):
    a=np.asarray(a,dtype=float); n=len(a); mean=a.mean(0); hw=a.std(0,ddof=1)*t.ppf(1-alpha/2,n-1)/np.sqrt(n)
    return mean,mean-hw,mean+hw


def meta_rows():
    labels=('0','1','2','3','trace'); rows={}
    def ml(m,s,l): return dict(model=NAMES[m],sector=SECT[s],label=labels[l])
    msl=list(itertools.product(range(4),range(11),range(5)))
    rows['mass']=[dict(model=m,label=l) for m in NAMES[:3] for l in labels[:4]]
    rows['entropy']=[dict(model=m) for m in NAMES[:3]]
    rows['contact']=[dict(**ml(m,s,l),distance=0,gap='overlap') for m,s,l in msl]
    def lagmeta(f,k):return dict(direction=FAMILIES[f],step=k,distance=round(np.sqrt(f+1)*k,7),gap='overlap' if k<3 else k-3,stagger_sign=(-1)**((f+1)*k))
    rows['lag']=[dict(**ml(m,s,l),**lagmeta(f,k)) for f,k in LAGS for m,s,l in msl]
    rows['null']=[dict(**ml(m,s,4),**lagmeta(f,k)) for f,k in LAGS for m in range(4) for s in range(11)]
    def coord(v):return ','.join(map(str,v))
    rows['power']=[dict(**ml(m,s,l),wave=coord(w)) for w in WAVES for m,s,l in msl]
    rows['density_power']=[dict(wave=coord(w),model=m,target=species) for w in WAVES for m in ('sparse','dense_FFT','paired_sparse_minus_dense') for species in ('A','B')]
    rows['angular']=[dict(**ml(m,s,l),direction=coord(RAYS[r]),step=k,distance=round(np.linalg.norm(RAYS[r])*k,7),gap=k-2,ray_parity=p,target=species) for m,s,l in itertools.product(range(4),range(11),range(4)) for r,k,p in SHELL for species in ('A','B')]
    rows['control']=[dict(target=c) for c in CONTROL]
    return rows


def write_tables(out, data, betas):
    rows=meta_rows(); columns='index beta parents metric model sector label direction step distance gap stagger_sign wave ray_parity target mean lo95 hi95 staggered_mean staggered_lo95 staggered_hi95'.split()
    path=out/'sectors.csv.gz'; tmp=path.with_suffix('.tmp')
    with gzip.open(tmp,'wt',newline='',compresslevel=3) as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader()
        for index,stats in data.items():
            for metric,records in rows.items():
                x=stats[metric]; vals=np.stack(interval(x),-1).reshape(-1,3)
                for rec,(m,l,h) in zip(records,vals,strict=True):
                    sign=rec.get('stagger_sign'); extra={}
                    if sign is not None:extra=dict(staggered_mean=sign*m,staggered_lo95=min(sign*l,sign*h),staggered_hi95=max(sign*l,sign*h))
                    writer.writerow(dict(index=index,beta=betas[index],parents=len(x)*PACK,metric=metric,**rec,mean=f'{m:.9g}',lo95=f'{l:.9g}',hi95=f'{h:.9g}',**{k:f'{v:.9g}' for k,v in extra.items()}))
    tmp.replace(path)


def plots(out,data,betas):
    colors=('tab:blue','tab:orange','tab:green','tab:purple'); stats=data[9]; fig,ax=plt.subplots(1,2,figsize=(11,4.25),layout='constrained')
    contact=stats['contact'][...,:10,-1].mean(0); image=ax[0].imshow(contact,aspect='auto',cmap='viridis'); fig.colorbar(image,ax=ax[0],label='Connected trace [probability²]')
    ax[0].set(xticks=np.arange(10),xticklabels=SECT[:10],yticks=np.arange(4),yticklabels=NAMES,title='Represented sectors · contact'); ax[0].tick_params(axis='x',labelrotation=55)
    li=LAGS.index((0,6)); x=np.arange(10)
    for model in (0,1,2,3):
        m,l,h=interval(stats['lag'][:,li,model,:10,-1]); ax[1].errorbar(x+(model-1.5)*.15,m,yerr=[m-l,h-m],fmt='.',capsize=2,color=colors[model],label=NAMES[model])
    ax[1].axhline(0,color='.5',lw=.8); ax[1].set(xticks=x,xticklabels=SECT[:10],ylabel='Signed trace covariance',title='Separated fields · six axial steps');ax[1].tick_params(axis='x',labelrotation=55);ax[1].legend(frameon=False,fontsize=8)
    fig.suptitle(f'Frozen fields at β={betas[9]:g} · {len(stats["lag"])*PACK} validation parents',weight='bold'); fig.savefig(out/'sectors.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained');
    for ax,index in zip(axes,INDICES):
        stats=data[index]
        for model in (0,1,2,3):
            ix=[i for i,(f,k) in enumerate(LAGS) if f==0 and k>=3]; steps=np.array([LAGS[i][1] for i in ix]); m,l,h=interval(stats['lag'][:,ix,model,-1,-1]);ax.plot(steps,m,'o-',ms=3,color=colors[model],label=NAMES[model]);ax.fill_between(steps,l,h,color=colors[model],alpha=.13)
        ax.set(xlabel='Axial separation [sites]',title=f'β={betas[index]:g}');ax.axhline(0,color='.5',lw=.8);ax.set_yscale('symlog',linthresh=.0004)
    axes[0].set_ylabel('Full field · signed trace covariance');axes[-1].legend(frameon=False,fontsize=8); fig.suptitle('Disjoint visible blocks · bands use whole-parent packets',weight='bold');fig.savefig(out/'separation.png',dpi=145);plt.close(fig)


def analyze(out, round_id, data, betas, provenance, checks, design_hash, incomplete):
    li=LAGS.index((0,6)); idx=INDICES[0]; x=data[idx]; count=len(x['lag'])*PACK
    convert=lambda a:[float(v) for v in interval(a,.05/len(TARGET))]
    band=convert(x['lag'][:,li,3,-1,-1]); null=convert(x['null'][:,li,3,-1]); plant=convert(x['control'][:,CONTROL.index('A_plant_gain')]); hw=(band[2]-band[1])/2
    sufficient=count>=256 and hw<=.0005 and plant[1]>0 and null[1]<=0<=null[2]; stop=bool(sufficient or count==TARGET[-1]); reason='precision and calibration reached' if sufficient else ('validation exhausted' if stop else ('deadline checkpoint' if incomplete else 'continue: precision or calibration not yet met'))
    ent={i:{name:{'posterior_mass':s['mass'][:,j].mean(0).tolist(),'marginal_entropy_nats':float(-xlogy(s['mass'][:,j].mean(0),s['mass'][:,j].mean(0)).sum()),'conditional_entropy_nats':float(s['entropy'][:,j].mean())} for j,name in enumerate(NAMES[:3])} for i,s in data.items()}
    power={i:{'residual_at_six':convert(s['lag'][:,li,3,-1,-1]),'shuffled_residual_at_six':convert(s['null'][:,li,3,-1]),'feature_plant_gain':convert(s['control'][:,3]),'matched_plant_minus_prediction':convert(s['control'][:,3]-s['control'][:,4]),'exact_composition_zero_k':[float(np.max(np.abs(s['control'][:,j]))) for j in (6,7)]} for i,s in data.items()}
    summary=dict(round=round_id,status='partial' if incomplete else 'complete',stop=stop,reason=reason,parents_per_temperature=count,independent_packets=count//PACK,new_stage_parent_range=[0 if round_id==0 else TARGET[round_id-1],count],design_sha256=design_hash,sources=provenance,projector_checks=checks,entropy=ent,calibration=power,prespecified_precision_target=dict(estimate_lo_hi=band,half_width=hw,excludes_effects_of_either_sign_ge_0_001=bool(band[1]>-.001 and band[2]<.001),resolved_nonzero=bool(band[1]>0 or band[2]<0)),interpretation='Validation-only fixed posterior fields. Contact is representation, not transfer. Signed separated and angular covariances are observables, not information bounds. The noisy unbiased two-panel Fourier cross-power has no nonnegative guarantee. A resolved nonzero finite separation or mode does not identify a thermodynamic long-range field; unresolved intervals are not evidence of absence.',outputs=['sectors.csv.gz','sectors.png','separation.png'])
    summary['cold_sector_atlas']={
        model:{s:{'contact_trace':convert(x['contact'][:,mi,si,-1]),
                  'axial_six_trace':convert(x['lag'][:,li,mi,si,-1]),
                  'fundamental_axis_z_trace_power':convert(x['power'][:,1,mi,si,-1])}
               for si,s in enumerate(SECT)}
        for mi,model in enumerate(NAMES)}
    summary['atlas_intervals']='estimate, lower, upper; group-t adjusted for looks ONLY, exploratory across labels/sectors/radii'
    write_tables(out,data,betas);plots(out,data,betas);dump(out/f'round-{round_id:04d}.json',summary)
    print(json.dumps({'round':round_id,'parents_per_temperature':count,'residual_at_six':band,'stop':stop,'reason':reason}),flush=True)


def run(args):
    out=args.out; work=args.work; out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    design=EXAMPLES.parent/'results/l20-autoresearch/sectors/design.json'; design_hash=sha(design); n=args.round
    if n:
        prev=json.loads((out/f'round-{n-1:04d}.json').read_text())
        if prev['status']!='complete' or prev['stop'] or prev['design_sha256']!=design_hash: raise ValueError('previous stage not completed or stopping condition already reached')
    if time.time()>=args.end_epoch: raise SystemExit('deadline reached before stage')
    pools={}; betas={}; provenance={}
    for index in INDICES:
        raw,meta=source(args.data,index); (tr,va,_),_=partition(index)
        pools[index]=raw[va];betas[index]=meta['beta'];provenance[index]=meta
        if index==9:
            fields=Fields(raw[tr[:2048]],work,meta['sha256']); checks=fields.check(raw[tr[:8],:][:,VT[157]])
        del raw
    provenance['models']=fields.provenance
    # Restart-safe packet artifacts bound to the actual frozen data/model/design.
    key=hashlib.sha256(json.dumps([list(provenance.values()),design_hash,sha(Path(__file__))],sort_keys=True).encode()).hexdigest()[:16]
    def packet_path(p):return work/f'{key}-packet-{p:04d}.npz'
    target=TARGET[n]//PACK; done=0
    for p in range(target):
        path=packet_path(p)
        if path.exists():done=p+1;continue
        if p<(TARGET[n-1]//PACK if n else 0): raise ValueError('previous packet cache missing; reuse its --work directory')
        if time.time()+12>=args.end_epoch:break
        packet={}
        for index in INDICES:
            stats=measure(pools[index][p*PACK:(p+1)*PACK],fields,index,p)
            packet.update({f'{index}_{m}':v for m,v in stats.items()})
        save(path,packet);done=p+1
    if done<2: raise SystemExit(f'Only {done} completed independent packets; cannot estimate uncertainty. Resume the same stage before the deadline.')
    data={i:{} for i in INDICES}
    for p in range(done):
        with np.load(packet_path(p),allow_pickle=False) as f:
            for name in f.files:
                i,m=name.split('_',1);data[int(i)].setdefault(m,[]).append(f[name])
    data={i:{m:np.stack(a) for m,a in s.items()} for i,s in data.items()}
    analyze(out,n,data,betas,provenance,checks,design_hash,done<target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--round',type=int,choices=range(len(TARGET)),required=True);parser.add_argument('--end-epoch',type=float,required=True);run(parser.parse_args())
