"""Stock RSMI-NE experiment 02: a disjoint-pair calibration, not the radius-four shell channel."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import stock_glass_modal as stock

STUDY='stock-glass-coordination-02'
DESIGN=Path(__file__).resolve().parents[1]/'results/experiment-02-coordination/design.json'
SEEDS=(6101,6102,6103)
EARLY=(850,1700,3400,6800)
N=8192


def prepare(a):
    import numpy as np
    release=a.archive/'release-index.json';assert (stock.sha(release),stock.sha(a.rosters),stock.sha(a.split))==(stock.INDEX,stock.ROSTERS,stock.SPLIT)
    a.out.mkdir(parents=True,exist_ok=False); split=json.loads(a.split.read_bytes());files={};counts=[];seen=set()
    dirs=np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
    left=np.array([(4*z,4*y,x) for z in range(5) for y in range(5) for x in (0,4)]);right=left+[0,0,8];base=np.stack([left,right],axis=1)
    stars=(base[:,:,None,:]+np.vstack([[0,0,0],dirs]))%20;assert np.unique(stars.reshape(-1,3),axis=0).shape==(700,3)
    with np.load(a.rosters) as rosters:
        for ent in json.loads(release.read_bytes())['endpoints']:
            b=ent['release_index'];tag=f'b{b:02}';f=a.archive/ent['npz'];assert stock.sha(f)==ent['npz_sha256'];reg=split['per_beta'][str(b)]
            with np.load(f) as native:
                ids=np.stack([native['chain_seed'],native['chain_id']],axis=1).astype('<u8')
                # Candidate positions AND fair orientations are determined before opening species.
                shifts=np.empty((len(ids),3),'u1');flips=np.empty((len(ids),50),bool)
                for j,(seed,cid) in enumerate(ids):
                    rng=np.random.default_rng([209157,b,int(seed),int(cid)]);shifts[j]=rng.integers(0,20,3);flips[j]=rng.integers(0,2,50).astype(bool)
                lattice=native['species']
            regrows=[rosters[f'{tag}_{side}'] for side in ('train','test')];assert np.array_equal(np.sort(np.concatenate(regrows)),np.arange(16384)); rec=dict(index=b,beta=reg['beta'])
            for side,rows in zip(('train','test'),regrows):
                keys=ids[rows]; asserts=hashlib.sha256(keys.tobytes()).hexdigest();assert len(rows)==N and asserts==reg[side+'_identity_sha256'];seen.update(map(tuple,keys.tolist())); Vs=[];Ys=[];Ps=[];Cs=[];cens=np.zeros((3,3,13),dtype='i8');invs=[]
                for lo in range(0,N,128):
                    rows0=rows[lo:lo+128];p=np.arange(len(rows0));coords=(base[None,:,:,:]+shifts[rows0,None,None,:])%20
                    coords=np.where(flips[rows0,:,None,None],coords[:,:,::-1,:],coords);lat=lattice[rows0];sp=lat[p[:,None,None],*np.moveaxis(coords,-1,0)]; near=(coords[:,:,:,None,:]+dirs)%20
                    V=lat[p[:,None,None,None],*np.moveaxis(near,-1,0)]!=0; m=V.sum(-1);d=(m[:,:,0]-m[:,:,1]).astype('i');cens+=np.bincount((13*(3*sp[:,:,0]+sp[:,:,1]).astype('i')+d+6).ravel(),minlength=117).reshape(3,3,13)
                    good=(sp[:,:,0]*sp[:,:,1]==2); parent,slot=np.where(good);Vs.append(V[parent,slot].transpose(0,2,1).astype('u1'));Ys.append((sp[parent,slot,0]==1).astype('u1'));Ps.append((lo+parent).astype('<u2'));Cs.append(slot.astype('u1'));invs.append(np.stack([(lat==i).sum(axis=(1,2,3)) for i in range(3)],axis=1))
                V,Y,P,C=map(np.concatenate,(Vs,Ys,Ps,Cs));inv=np.concatenate(invs);assert len(Y)==cens[1,2].sum()+cens[2,1].sum() and cens.sum()==50*N and len(np.unique(P.astype('i')*50+C))==len(Y);f=a.out/f'{tag}-{side}.npz';np.savez_compressed(f,V=V,Y=Y,parent=P,slot=C,ids=keys,shifts=shifts[rows],flips=np.packbits(flips[rows],axis=1),inventory=inv.astype('<u2'),census=cens);files[f.name]=stock.sha(f)
                rec[side]=dict(identity=asserts,parents=N,candidates=int(cens.sum()),both_occupied=int(cens[1:,1:].sum()),unlike=len(Y),usable_parents=len(np.unique(P)),inventory_min=inv.min(0).tolist(),inventory_max=inv.max(0).tolist(),center_species_by_signed_d=cens.tolist())
            counts.append(rec);print('PREPARED',b,rec['train']['unlike'],rec['test']['unlike'],flush=True)
    assert len(seen)==163840
    stock.write_json(a.out/'prepared.json',dict(base=stock.BASE,split=stock.SPLIT,design=stock.sha(DESIGN),unique=len(seen),inputs=files,coverage=counts))


def groups(parent,b):
    import numpy as np
    for block in range(32):
        ix=np.flatnonzero(parent//256==block); r=np.random.default_rng([209158,b,block]);ix=r.permutation(ix)
        for j,g in enumerate(np.array_split(ix,math.ceil(len(ix)/500))):
            assert len(g)>=2; rng=np.random.default_rng([209159,b,block,j]);p=rng.permutation(len(g));fixed=np.arange(len(g))
            while np.any(p==fixed):p=rng.permutation(len(g))
            yield block,g,p


def raw(x):
    return x.astype('float32')[:,None,:,:,None]  # six directions, two sites; no m, d or species


def code(h):
    return 2*(h[:,0]>=.5).astype('uint8')+(h[:,1]>=.5)


def fit(a):
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp
    import tqdm
    from contextlib import redirect_stderr
    from unittest.mock import patch
    from rsmine.coarsegrainer import cg_optimisers as core
    b,s=a.index,a.seed;cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[a.slot%len(cpus)]})
    gpu,=tf.config.list_physical_devices('GPU');tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    uid=stock.cuda_uuid();device=tf.config.experimental.get_device_details(gpu)['device_name']
    with tf.device('/GPU:0'): place=tf.matmul(tf.ones((2,2)),tf.ones((2,2))).device
    assert 'GPU:0' in place and 'H100' in device
    tf.keras.utils.set_random_seed(s);stem=f'b{b:02}-s{s}'; meta=json.loads(Path('prepared.json').read_bytes());cov=meta['coverage'][b]
    def load(side):
        f=Path(f'b{b:02}-{side}.npz');assert stock.sha(f)==meta['inputs'][f.name]
        with np.load(f) as z:
            assert hashlib.sha256(z['ids'].tobytes()).hexdigest()==cov[side]['identity'];return z['V'],z['Y'],z['parent']
    v,y,p=load('train');m=v.sum(1,dtype='int16');d=m[:,0]-m[:,1]; n=len(y);B=math.ceil(n/500);epochs=math.ceil(10000/B);final=B*epochs
    cg=dict(ll=(1,6),visible_dim=2,nonlinearCG=None,init_temperature=.75,min_temperature=.1,relaxation_rate=.0001,Nq=None,conv_activation=None,hidden_dim=2,h_embed=True,use_probs=False,use_logits=True)
    crit=dict(layers=3,embed_dim=8,hidden_dim=32,activation='relu',use_dropout=True,dropout_rate=.2)
    opt=dict(batch_size=500,iterations=epochs,shuffle=100000,learning_rate=.009);adam=tf.keras.optimizers.Adam(.009)
    refs={};saved={};recs={};ticks=0;start=time.monotonic()
    def capture(key,orig):
        def create(*args,**kwargs):
            assert key not in refs;refs[key]=orig(*args,**kwargs);return refs[key]
        return create
    orig_update=tqdm.tqdm.update
    def observe(bar,n=1):
        nonlocal ticks
        ticks+=n
        if ticks in (*EARLY,final,1):
            assert int(adam.iterations.numpy())==ticks
            if ticks==1:print('FIRST_CUDA_STEP',stem,uid,ticks,flush=True)
            else:
                recs[ticks]=dict(step=int(refs['cg'].global_step),elapsed=time.monotonic()-start)
                saved[f'w{ticks}']=refs['cg']._Λ.get_weights()[0]
                for j,w in enumerate(refs['critic'].get_weights()):saved[f'c{ticks}_{j}']=w
        return orig_update(bar,n)
    eye=np.eye(2,dtype='f')
    with patch.object(core,'CoarseGrainer',capture('cg',core.CoarseGrainer)),patch.object(core,'SeparableCritic',capture('critic',core.SeparableCritic)),patch.object(tqdm.tqdm,'update',observe),open(os.devnull,'w') as quiet,redirect_stderr(quiet):
        trace,h,filters,encoder=core.train_RSMI_optimiser(cg,crit,opt,dict(N_samples=n,dimension=2),V=raw(v),E=eye[y],use_GPU=True,optimizer=adam)
    steps=int(adam.iterations.numpy());wall=time.monotonic()-start
    rec=dict(index=b,beta=cov['beta'],seed=s,steps=steps,final=final,epochs=epochs,batches=B,last_batch=n-(B-1)*500,examples=(steps//B)*n+min((steps%B)*500,n),train_usable=cov['train']['usable_parents'] if steps>=B else None,fit_status='complete' if steps==final and ticks==steps else 'nonfinite' if ticks+1==steps else 'partial',uuid=uid,device=device,base=stock.BASE,design=meta['design'],train_s=wall,checkpoints=recs)
    saved.update(trace=trace,filters=filters,train_logged_h=h,returned_filter=encoder.get_weights()[0])
    if recs:
        tv,ty,tp=load('test');groups0=list(groups(tp,b));critic=refs['critic'];nc=len(critic.weights);eye_tf=tf.constant(eye)
        for dep,r in recs.items():
            cw=[saved[f'c{dep}_{i}'] for i in range(nc)];w=saved[f'w{dep}']
            if not all(np.isfinite(x).all() for x in [w,*cw]):r.update(valid=False,status='nonfinite weights');continue
            encoder.set_weights([w]);critic.set_weights(cw);refs['cg'].global_step=r['step']
            def draw(vv):
                return np.concatenate([encoder(raw(vv[i:i+4096]),training=False).numpy() for i in range(0,len(vv),4096)])
            tf.keras.utils.set_random_seed(209160+b);ch=code(draw(v)); minus=np.bincount(ch[d<0],minlength=4);plus=np.bincount(ch[d>0],minlength=4)
            cmp=minus*plus.sum()-plus*minus.sum(); rule=(1+np.sign(cmp))/2  # P(choose i lower); ties split
            r.update(rule=rule.tolist(),train_code_dneg=minus.tolist(),train_code_dpos=plus.tolist())
            tf.keras.utils.set_random_seed(209161+b);th=draw(tv);score=np.concatenate([critic(eye_tf,th[i:i+4096],training=False).numpy() for i in range(0,len(th),4096)])
            saved[f'h{dep}']=th;saved[f's{dep}']=score
            if not np.isfinite(th).all() or not np.isfinite(score).all():r.update(valid=False,status='nonfinite scores');continue
            summary=[];ok=True
            for block,g,perm in groups0:
                y0=ty[g].astype('i4');u=score[g];ng=len(g);ix=np.arange(ng);counts=np.bincount(y0,minlength=2)
                # Exactly the two repeated E columns; same H and same critic for both masks.
                matrix=tf.gather(tf.constant(u),y0,axis=1);bound=tfp.vi.mutual_information.lower_bound_info_nce
                re=float(bound(logu=matrix,joint_sample_mask=tf.eye(ng,dtype=tf.bool)));nu=float(bound(logu=matrix,joint_sample_mask=tf.equal(ix[None,:],perm[:,None])))
                z=u.astype('d');z-=z.max(1)[:,None]
                den=np.logaddexp(z[:,0]+np.log(counts[0]),z[:,1]+np.log(counts[1]))-np.log(ng) if counts.min() else z[:,int(counts[1]>0)]
                r64=float(np.mean(z[ix,y0]-den));n64=float(np.mean(z[ix,y0[perm]]-den))
                good=np.isfinite([re,nu,r64,n64]).all() and max(re,nu)<=np.log(ng)+1e-5 and max(abs(re-r64),abs(nu-n64))<=.001
                ok &= bool(good);summary.append([block,ng,re,nu,r64,n64,int(good)])
            saved[f'g{dep}']=np.array(summary);r.update(valid=ok,status='complete' if ok else 'invalid frozen score',test_usable=len(np.unique(tp)),test_pairs=len(tp))
    assert steps==int(adam.iterations.numpy());np.savez_compressed(stem+'.npz',receipt=json.dumps(rec,allow_nan=False),**saved)
    print('COORD_RESULT',stem,steps,final,rec['fit_status'],flush=True)


def launch(a):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime,timezone
    from decimal import Decimal as D
    import queue,subprocess,tempfile
    import modal
    from modal.stream_type import StreamType
    r=modal.Workspace.from_context().billing.rates();life,cpu,gib=2100,10,34;cost=D(life)/3600*(r['gpu_hour_cost_h100']+cpu*r['cpu_hour_cost_sandbox']+gib*r['mem_gib_hour_cost_sandbox'])+8*r['egress_gib_cost']
    prior=D('4.85172072');assert cost<=D(str(a.budget))<=min(D('3.95'),25-prior)
    meta=json.loads((a.inputs/'prepared.json').read_bytes());assert meta['design']==stock.sha(DESIGN) and meta['unique']==163840
    repo=Path(__file__).resolve().parents[1];pins=json.loads((repo/'results/stock-glass-all-beta-critic/launch.json').read_bytes())['wheel_sha256']
    assert {p.name:stock.sha(p) for p in a.wheels.glob('*.whl')}==pins and all(stock.sha(a.inputs/k)==v for k,v in meta['inputs'].items())
    stamp=lambda:datetime.now(timezone.utc).isoformat();a.out.mkdir(parents=True,exist_ok=False)
    stock.write_json(a.out/'reservation.json',dict(prior=str(prior),cap=str(cost),life=life,cpu=cpu,gib=gib,design=meta['design'],launcher=stock.sha(Path(__file__)),image=stock.IMAGE))
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp);subprocess.run(['git','archive','--format=tar','-o',str(tmp/'fork.tar'),stock.BASE,'rsmine','examples/tutorial.ipynb'],cwd=repo,check=True)
        subprocess.run(['bash','-ec','set -o pipefail; tar -C "$1" -cf - . | split -b 384M - "$2/wheels-"','pack',str(a.wheels.resolve()),str(tmp)],check=True)
        app=modal.App.lookup(STUDY,create_if_missing=True);sb=modal.Sandbox.create('/bin/sleep',str(life),name=STUDY,app=app,image=modal.Image.from_id(stock.IMAGE),gpu='H100!',cpu=(cpu,cpu),memory=(gib*1024,gib*1024),block_network=True,timeout=life,tags={'study':STUDY})
        try:
            stock.write_json(a.out/'provider.json',dict(sandbox=sb.object_id,app=app.app_id,utc=stamp()));print('PROVIDER_ID',sb.object_id,app.app_id,'max_usd',cost,flush=True)
            fs=sb.filesystem;remote=stock.REMOTE;fs.make_directory(remote)
            for f in [*sorted(tmp.glob('wheels-*')),tmp/'fork.tar',Path(__file__),Path(stock.__file__),a.inputs/'prepared.json',*(a.inputs/n for n in meta['inputs'])]:fs.copy_from_local(f,remote+'/'+f.name)
            def run(cmd,secs):
                pr=sb.exec('/bin/bash','-ec',cmd,timeout=secs,workdir=remote,stderr=StreamType.STDOUT,bufsize=1);tail=[]
                for line in pr.stdout:
                    line=line.rstrip();tail=(tail+[line])[-16:]
                    if any(x in line for x in ('FIRST_CUDA_STEP','COORD_RESULT','Error','Traceback')):print(line,flush=True)
                if pr.wait():raise RuntimeError(str(tail))
            run('set -o pipefail; mkdir wheels; cat wheels-* | tar -xf - -C wheels; rm wheels-*; tar -xf fork.tar; /opt/conda/bin/python -m venv env; env/bin/pip install --no-index --find-links=wheels --disable-pip-version-check -q '+' '.join(repr(x) for x in stock.REQUIREMENTS),1200)
            env='export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 TF_USE_LEGACY_KERAS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 WANDB_MODE=disabled; export LD_LIBRARY_PATH="$(find "$PWD/env" -path "*/nvidia/*/lib" -type d | paste -sd: -):${LD_LIBRARY_PATH:-}"; '
            slots=queue.Queue()
            for j in range(8):slots.put(j)
            def execute(cell):
                b,s=cell;slot=slots.get();stem=f'b{b:02}-s{s}';name=stem+'.npz'
                try:
                    run(env+f'env/bin/python -u stock_glass_coordination.py fit --index {b} --seed {s} --slot {slot}',1000)
                    assert 0<fs.stat(remote+'/'+name).size<32*1024**2
                    with (a.out/name).open('xb') as f:f.write(fs.read_bytes(remote+'/'+name))
                    print('SAVED',stem,stock.sha(a.out/name),flush=True)
                except Exception as e:
                    stock.write_json(a.out/(stem+'-failed.json'),dict(error=str(e)));print('FIT_ERROR',stem,str(e),flush=True)
                finally:slots.put(slot)
            with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(execute,[(b,s) for s in SEEDS for b in range(10)]))
        finally:
            rc=sb.terminate(wait=True);stock.write_json(a.out/'terminated.json',dict(sandbox=sb.object_id,utc=stamp(),code=rc));print('TERMINATED',sb.object_id,rc,flush=True)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='cmd',required=True);p=sub.add_parser('prepare')
    for name in ('archive','rosters','split','out'):p.add_argument('--'+name,type=Path,required=True)
    f=sub.add_parser('fit');f.add_argument('--index',required=True,type=int,choices=range(10));f.add_argument('--seed',required=True,type=int,choices=SEEDS);f.add_argument('--slot',required=True,type=int,choices=range(8))
    l=sub.add_parser('launch')
    for name in ('inputs','wheels','out'):l.add_argument('--'+name,type=Path,required=True)
    l.add_argument('--budget',type=float,required=True)
    a=parser.parse_args();{'prepare':prepare,'fit':fit,'launch':launch}[a.cmd](a)

if __name__=='__main__':main()
