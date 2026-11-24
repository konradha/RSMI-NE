"""05 archive adapter: same immutable 03 anchors and trained stock coefficients, new observed sectors."""
import argparse
import hashlib
import io
import itertools
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import stock_glass_modal as stock
from stock_glass_cavity_exact import C,U,DIR,A,J,L,CODE,full,prob

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/experiment-05-inventory-transfer';REF=ROOT/'results/experiment-03-cavity';DESIGN=OUT/'design.json';SEEDS=(6101,6102,6103);SECTORS=tuple(json.loads(DESIGN.read_bytes())['sector_order']);SOURCE='223'


class Sector:
    def __init__(self,label):
        nv,na,nb=map(int,label);self.label=label;self.x=np.array([s for s in itertools.product(range(3),repeat=7) if s.count(0)==nv and s.count(1)==na],dtype='u1');self.occ=(self.x!=0).astype('i4');self.r=self.occ@A.T-L[self.x];self.outer=self.occ@J.T
        self.lookup=np.full(3**7,255,'u1');self.lookup[self.x@CODE]=np.arange(len(self.x));self.raw=np.stack((self.x==1,self.x==2),axis=-1).reshape(len(self.x),14).astype('d');assert len(self.x)<255 and nv+na+nb==7
    def energies(self,fc,su,fu):
        fc=fc.astype('i4');fu=fu.astype('i4');live=(su!=0).astype('i4');E=(self.occ*self.r**2).sum(1)+2*fc@(self.occ*self.r).T+(fc**2)@self.occ.T+2*(live*(fu-L[su]))@self.outer.T+live@(self.outer**2).T;return (E-E.min(1,keepdims=True)).astype('i2')


def pinned(ref):
    for n,h in json.loads(DESIGN.read_bytes())['pins'].items():assert stock.sha(ref/n)==h,n
    return json.loads((ref/'parent-census.json').read_bytes())


def replay(ref,out):
    pinned(ref);sec={s:Sector(s) for s in SECTORS};rec=[];output={};frozen={};pins=json.loads((ref/'corrected/first-updates.json').read_bytes())['checkpoints'];xyz=tuple((C+1).T)
    with np.load(ref/'corrected/code-maps.npz') as maps,np.load(ref/'frozen-modes.npz') as modes,zipfile.ZipFile(ref/'corrected/stock-checkpoints.zip') as pack:
        for b in range(10):
            depth=10218 if b==9 else 10257
            for seed in SEEDS:
                stem=f'b{b:02}-s{seed}';k=stem+f'_d{depth}';buf=pack.read(stem+'.npz');assert hashlib.sha256(buf).hexdigest()==pins[stem+'.npz']
                with np.load(io.BytesIO(buf)) as fit:
                    r=json.loads(fit['receipt'].item());assert r['steps']==r['final']==depth and r['seed']==seed and r['index']==b and r['fit_status']=='complete';w=fit[f'w{depth}'][xyz].reshape(14,2);native=fit[f'logits{depth}'];np.testing.assert_array_equal(native,maps[k+'_logits'])
                bits=w.view('u4');qw=((bits+4095+((bits>>13)&1))&np.uint32(0xffffe000)).view('f4');z=sec[SOURCE].raw@qw;q=maps[k+'_hard_Q'];a=maps[k+'_hard_a'];pred=lambda t:(2*(t[:,0]>=0)+(t[:,1]>=0)).astype('u1')
                np.testing.assert_array_equal(pred(z),q.argmax(1));np.testing.assert_allclose(z,native,rtol=2e-5,atol=3e-5);sm=modes[f'b{b:02}_mean']@q;keep=sm>1e-14;nonref=keep.copy();nonref[sm.argmax()]=False;S=(np.diag(sm)-np.outer(sm,sm))[np.ix_(nonref,nonref)];v=np.linalg.eigvalsh(S) if S.size else np.array([]);rank=int(sum(v>max(1e-14,1e-12*v[-1]))) if len(v) else 0;identified=keep&np.isfinite(a)&(rank==nonref.sum())
                frozen[stem]=(qw,a,sm,identified);output[stem+'_a']=a;output[stem+'_source_mass']=sm;output[stem+'_identified']=identified;rec.append(dict(index=b,seed=seed,depth=depth,source_states=len(z),source_label_mismatches=0,source_max_logit_error=float(np.max(abs(z-native))),source_cov_rank=rank,source_labels=int(sum(keep)),source_scalar_finite=bool(np.isfinite(a).all()),kernel_sha256=pins[stem+'.npz'],scalar_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    # All 30 source sign gates passed before extrapolating. A TF32 weight is a dyadic rational.
    # For a near-zero sum, certify exact float32 addition in ANY order from integer-grid bounds.
    signrows=[]
    for stem,(qw,a,sm,identified) in frozen.items():
        for s,g in sec.items():
            z=g.raw@qw;near=abs(z)<=16*np.finfo('f4').eps*(g.raw@abs(qw));unverified=0
            for i,h in zip(*np.where(near)):
                ws=qw[g.raw[i].astype(bool),h];rat=[float(v).as_integer_ratio() for v in ws];den=max(v for _,v in rat);bound=sum(abs(n)*(den//d) for n,d in rat);unverified+=int(bound.bit_length()>23)
            assert unverified==0,(stem,s,unverified);output[stem+'_'+s]=pred(z);signrows.append(dict(fit=stem,sector=s,states=len(z),near_zero=int(near.sum()),unresolved_signs=unverified,min_abs_logit=float(abs(z).min())))
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'frozen-transfer.npz',**output);pd.DataFrame(rec).to_csv(out/'source-replay.csv',index=False);pd.DataFrame(signrows).to_csv(out/'sign-audit.csv',index=False);return sec


def prepare(a):
    meta=pinned(a.reference);geoms=replay(a.reference,a.out);manifest=a.source_banks/'prepared.json';assert stock.sha(manifest)==stock.sha(a.reference/'parent-census.json');a.banks.mkdir(parents=True,exist_ok=False);idx=a.archive/'release-index.json';assert stock.sha(idx)==stock.INDEX;ents=json.loads(idx.read_bytes())['endpoints'];coverage=[];audits=[];hashes={};unique=set();shared=0
    with np.load(a.source_banks/'state-geometry.npz') as z:geom=z['anchors'];np.testing.assert_array_equal(geom,np.indices((6,6,6)).reshape(3,-1).T*3);np.testing.assert_array_equal(geoms[SOURCE].x,z['states'])
    for ent,r in zip(ents,meta['coverage'],strict=True):
        b=r['index'];tag=f'b{b:02}';assert b==ent['release_index'];path=a.archive/ent['npz'];assert stock.sha(path)==r['archive_sha']==ent['npz_sha256']
        with np.load(path) as snap:lat=snap['species'];ids=np.stack((snap['chain_seed'],snap['chain_id']),axis=1).astype('<u8')
        for side in ('train','test'):
            name=f'{tag}-{side}.npz';assert stock.sha(a.source_banks/name)==meta['inputs'][name]
            with np.load(a.source_banks/name) as base:
                ii=base['native_rows'];keys=ids[ii];np.testing.assert_array_equal(keys,base['all_ids']);unique.update(map(tuple,keys.tolist()));N=len(ii);assert N==8192;shifts=np.empty((N,3),'i2');order=np.empty((N,216),'u1')
                for j,(sd,cid) in enumerate(keys):
                    rnd=np.random.default_rng([309031,b,int(sd),int(cid)]);shifts[j]=rnd.integers(20,size=3);order[j]=rnd.permutation(216)
                np.testing.assert_array_equal(shifts,base['all_shifts']);arrays={s:[] for s in SECTORS};counts={s:[] for s in SECTORS}
                for j0 in range(0,N,128):
                    slab=lat[ii[j0:j0+128]];anc=(shifts[j0:j0+128,None,:]+geom[order[j0:j0+128]])%20;pos=(anc[:,:,None,:]+C)%20;sp=slab[np.arange(len(slab))[:,None,None],*np.moveaxis(pos,-1,0)];nv=(sp==0).sum(2);na=(sp==1).sum(2)
                    for s,g in geoms.items():
                        match=(nv==int(s[0]))&(na==int(s[1]));num=match.sum(1);counts[s].append(num.astype('u1'));sel=np.flatnonzero(num);k=match[sel].argmax(1) if len(sel) else np.empty(0,'i4');c=anc[sel,k];state=sp[sel,k];observed=g.lookup[state@CODE];assert np.all(observed<255)
                        def gather(off):
                            p=(c.reshape((-1,)+(1,)*(off.ndim-1)+(3,))+off)%20;return slab[sel.reshape((-1,)+(1,)*(off.ndim-1)),*np.moveaxis(p,-1,0)]
                        oc=(state!=0).astype('i2');fc=(gather(C[:,None]+DIR)!=0).sum(2)-oc@A.T;su=gather(U);fu=(gather(U[:,None]+DIR)!=0).sum(2)-oc@J.T;assert np.all(fc>=0) and np.all(fu>=0);arrays[s].append((j0+sel,c,observed,fc,su,fu))
                for s,g in geoms.items():
                    sel,c,obs,fc,su,fu=map(np.concatenate,zip(*arrays[s]));nums=np.concatenate(counts[s]);np.testing.assert_array_equal(sel,np.flatnonzero(nums));local=dict(eligible_counts=nums,selected=sel.astype('<u2'),anchor=c.astype('u1'),observed=obs,fc=fc.astype('u1'),su=su.astype('u1'),fu=fu.astype('u1'))
                    if s==SOURCE:
                        for k,v in local.items():np.testing.assert_array_equal(v,base[k]);np.testing.assert_array_equal(sel,base['selected'])
                        np.testing.assert_array_equal(g.energies(fc,su,fu),base['relative_E']);shared+=len(sel)
                    else:
                        f=a.banks/f'{tag}-{s}-{side}.npz';np.savez_compressed(f,**local);hashes[f.name]=stock.sha(f)
                    overlap=np.intersect1d(sel,base['selected'],assume_unique=True);coverage.append(dict(index=b,beta=float(r['beta']),sector=s,side=side,roster=N,parents=len(sel),missing=N-len(sel),eligible_windows=int(nums.sum()),candidate_windows=N*216,shared_source_parents=len(overlap),states=len(g.x),min_eligible=int(nums.min()),median_eligible=float(np.median(nums)),max_eligible=int(nums.max())))
                    if len(sel):
                        wrap=np.flatnonzero(((c==0)|(c==19)).any(1));i=int(wrap[0]) if len(wrap) else 0;v=lat[ii[sel[i]]].copy();pos=tuple(((c[i]+C)%20).T);np.testing.assert_array_equal(v[pos],g.x[obs[i]]);E=[]
                        for x in g.x:v[pos]=x;E.append(full(v))
                        expect=np.asarray(E)-min(E);got=g.energies(fc[i:i+1],su[i:i+1],fu[i:i+1])[0];np.testing.assert_array_equal(got,expect);assert np.max(abs(prob(got[None,:],0)-1/len(g.x)))<1e-14;audits.append(dict(index=b,sector=s,side=side,roster_row=int(sel[i]),periodic=bool(len(wrap)),anchor=c[i].tolist(),integer_checks=len(E),max_error=0))
        print('CENSUS',tag,flush=True)
    assert len(unique)==163840 and shared==163822;pd.DataFrame(coverage).to_csv(a.out/'coverage.csv',index=False);stock.write_json(a.out/'bank-receipt.json',dict(design=stock.sha(DESIGN),parents=len(unique),source_matched=shared,source_manifest=stock.sha(manifest),target_banks=hashes,full_energy_audits=audits,transferred=stock.sha(a.out/'frozen-transfer.npz')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--source-banks',type=Path,required=True);ap.add_argument('--banks',type=Path,required=True);ap.add_argument('--reference',type=Path,default=REF);ap.add_argument('--out',type=Path,default=OUT);prepare(ap.parse_args())
