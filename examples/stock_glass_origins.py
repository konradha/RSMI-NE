"""Experiment 01 only. Import the existing stock launcher; never replace stock training math."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import stock_glass_modal as stock

STUDY = 'stock-glass-origins-01'
DESIGN = Path(__file__).resolve().parents[1]/'results/experiment-01-origins/design.json'
DEPTHS = (858,1716,3432,6732,10032)
SEEDS = (6101,6102,6103)
LEVELS = (1,4,8)
ROWS, PARENTS, BATCHES = 65536,8192,132


def prepare(a):
    import numpy as np
    release=a.archive/'release-index.json'
    if (stock.sha(release),stock.sha(a.rosters),stock.sha(a.split))!=(stock.INDEX,stock.ROSTERS,stock.SPLIT):
        raise ValueError('Native release or parent assignment changed')
    a.out.mkdir(parents=True,exist_ok=False); split=json.loads(a.split.read_bytes()); old=json.loads((a.fixed/'prepared.json').read_bytes()); seen=set(); files={}; coverage=[]
    offsets=np.indices((9,9,9)).reshape(3,-1).T-4; shell=offsets[np.abs(offsets).max(1)==4]; center=np.indices((3,3,3)).reshape(3,-1).T-1
    original=np.ravel_multi_index((10,10,10),(20,20,20)); assert len(shell)==386
    with np.load(a.rosters) as roster:
        for entry in json.loads(release.read_bytes())['endpoints']:
            b=entry['release_index']; tag=f'b{b:02}'; path=a.archive/entry['npz']; receipt=split['per_beta'][str(b)]
            if stock.sha(path)!=entry['npz_sha256']: raise ValueError('Native archive digest')
            train,test=[roster[f'{tag}_{s}'] for s in ('train','test')]; assert np.array_equal(np.sort(np.r_[train,test]),np.arange(16384)) and len(train)==len(test)==PARENTS
            with np.load(path,allow_pickle=False) as native:
                ids=np.column_stack((native['chain_seed'],native['chain_id'])).astype('<u8'); lattice=native['species'][train]
            for side,rows in (('train',train),('test',test)):
                keys=ids[rows]; assert hashlib.sha256(keys.tobytes()).hexdigest()==receipt[f'{side}_identity_sha256']; seen.update(map(tuple,keys.tolist()))
                f=a.fixed/f'{tag}-{side}.npz'; assert stock.sha(f)==old['inputs'][f.name]
                with np.load(f) as fixed:
                    assert np.array_equal(keys,fixed['ids'])
                if side=='test': shutil.copyfile(f,a.out/f.name); files[f.name]=old['inputs'][f.name]
            origins=np.empty((PARENTS,8),dtype='<u2'); order=np.empty((PARENTS,8),dtype='u1'); origins[:,0]=original
            for j,(seed,cid) in enumerate(ids[train]):
                rng=np.random.default_rng([103071,b,int(seed),int(cid)]); x=rng.choice(7999,7,replace=False); origins[j,1:]=x+(x>=original); order[j]=rng.permutation(8)
            assert (np.diff(np.sort(origins,axis=1),axis=1)>0).all()
            v=np.empty((PARENTS,8,3,3,3),'u1'); e=np.empty((PARENTS,8,386),'u1')
            for lo in range(0,PARENTS,256):
                hi=lo+256; base=np.stack(np.unravel_index(origins[lo:hi],(20,20,20)),axis=-1); ix=np.arange(lo,hi)[:,None,None]
                for d,dest in ((center,v),(shell,e)):
                    pos=(base[:,:,None,:]+d)%20; dest[lo:hi]=lattice[ix,pos[...,0],pos[...,1],pos[...,2]].reshape(dest[lo:hi].shape)
            with np.load(a.fixed/f'{tag}-train.npz') as z: assert np.array_equal(z['V'],v[:,0]) and np.array_equal(z['E'],e[:,0])
            f=a.out/f'{tag}-bank.npz'; np.savez_compressed(f,V=v,E=e,ids=ids[train],origins=origins,order=order);files[f.name]=stock.sha(f)
            coverage.append(dict(index=b,beta=receipt['beta'],train_parents=PARENTS,test_parents=PARENTS,train_identity=receipt['train_identity_sha256'],test_identity=receipt['test_identity_sha256'],rows_per_arm=ROWS,distinct_parent_origins=[PARENTS*k for k in LEVELS],origin_sha256=hashlib.sha256(origins.tobytes()).hexdigest(),slot_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()));print('PREPARED',b,flush=True)
    assert len(seen)==163840 and len(coverage)==10
    stock.write_json(a.out/'prepared.json',dict(base=stock.BASE,design=stock.sha(DESIGN),split=stock.SPLIT,inputs=files,coverage=coverage,unique=163840))


def encode(x):
    import numpy as np
    return np.stack((x==1,x==2),axis=-1).astype('f')


def assessment(b,test,encoder,critic):
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp
    from scipy.special import logsumexp
    tf.keras.utils.set_random_seed(440311+b)
    rounds=[]; latent=[]; valid=True
    for j,lo in enumerate(range(0,PARENTS,500)):
        sl=slice(lo,lo+500); h=encoder(encode(test['V'][sl])[...,None],training=False)
        n=len(h); s=critic(encode(test['E'][sl]).reshape(n,-1),h,training=False)
        ix=np.arange(n); rng=np.random.default_rng([771904,b,j]); perm=rng.permutation(n)
        while np.any(perm==ix): perm=rng.permutation(n)
        psha=hashlib.sha256(perm.astype('<i4').tobytes()).hexdigest()
        x=s.numpy(); h=h.numpy(); latent.append(h)
        row=dict(n=n,perm=psha,valid=False)
        if x.shape==(n,n) and np.isfinite(x).all() and np.isfinite(h).all():
            bound=tfp.vi.mutual_information.lower_bound_info_nce
            real=float(bound(logu=s,joint_sample_mask=tf.eye(n,dtype=tf.bool)))
            null=float(bound(logu=s,joint_sample_mask=tf.equal(ix[None,:],perm[:,None])))
            y=x.astype('d'); y-=y.max(1)[:,None]; den=logsumexp(y,axis=1)-np.log(n)
            r64=float(np.mean(y[ix,ix]-den)); n64=float(np.mean(y[ix,perm]-den))
            if np.isfinite([real,null,r64,n64]).all():
                ok=max(real,null)<=np.log(n)+1e-5 and max(abs(real-r64),abs(null-n64))<=.001
                row.update(real=real,shuffle=null,gain=real-null,real64=r64,shuffle64=n64,gain64=r64-n64,valid=bool(ok))
        rounds.append(row);valid &= row['valid']
    latent=np.concatenate(latent)
    counts=np.bincount((latent[:,0]>=.5).astype('i')*2+(latent[:,1]>=.5),minlength=4)
    p=counts[counts>0]/PARENTS
    return dict(valid=bool(valid),test_parents=len(latent),batches=rounds,counts=counts.tolist(),entropy_bits=float(-np.sum(p*np.log2(p)))),latent


def run_fit(a):
    import numpy as np
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[a.slot%len(cpus)]})
    from contextlib import redirect_stderr
    from unittest.mock import patch
    import tensorflow as tf
    import tqdm
    from rsmine.coarsegrainer import cg_optimisers as core
    b,k,seed=a.index,a.origins,a.seed; tag=f'b{b:02}'; stem=f'{tag}-k{k}-s{seed}'
    gpu,=tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    device=tf.config.experimental.get_device_details(gpu)['device_name'];uid=stock.cuda_uuid()
    with tf.device('/GPU:0'): placement=tf.matmul(tf.ones((2,2)),tf.ones((2,2))).device
    assert 'H100' in device and 'GPU:0' in placement
    tf.keras.utils.set_random_seed(seed)
    meta=json.loads(Path('prepared.json').read_bytes()); cov=meta['coverage'][b]
    def load(s):
        f=Path(f'{tag}-{s}.npz');assert stock.sha(f)==meta['inputs'][f.name];return np.load(f,allow_pickle=False)
    with load('bank') as z:
        ix=np.arange(PARENTS)[:,None]; order=z['order']%k
        assert hashlib.sha256(z['ids'].tobytes()).hexdigest()==cov['train_identity']
        v=encode(z['V'][ix,order].reshape(ROWS,3,3,3))[...,None]
        e=encode(z['E'][ix,order].reshape(ROWS,386)).reshape(ROWS,-1)
    cg=dict(ll=(3,3,3),visible_dim=2,nonlinearCG=None,init_temperature=.75,min_temperature=.1,relaxation_rate=.0001,Nq=None,conv_activation=None,hidden_dim=2,h_embed=True,use_probs=False,use_logits=True)
    critic_opts=dict(layers=3,embed_dim=8,hidden_dim=32,activation='relu',use_dropout=True,dropout_rate=.2)
    opt=dict(batch_size=500,iterations=76,shuffle=100000,learning_rate=.009); adam=tf.keras.optimizers.Adam(.009)
    refs={}; weights={}; checkpoints={}; ticks=0; start=time.monotonic()
    def capture(key,constructor):
        def make(*args,**kwargs):
            assert key not in refs
            refs[key]=constructor(*args,**kwargs); return refs[key]
        return make
    # tqdm.update is reached after stock apply_gradients and its NaN test. Merely copy
    # weights here; defer ALL stochastic evaluation until the single fit has returned.
    original=tqdm.tqdm.update
    def observe(bar,n=1):
        nonlocal ticks
        ticks+=n
        if ticks in DEPTHS or ticks==1:
            actual=int(adam.iterations.numpy());assert ticks==actual
            if ticks==1: print('FIRST_CUDA_STEP',stem,uid,actual,flush=True)
            if ticks in DEPTHS:
                enc=refs['cg']._Λ; critic=refs['critic']
                weights[f'enc_{ticks}']=enc.get_weights()[0]
                for i,w in enumerate(critic.get_weights()): weights[f'c{ticks}_{i:02}']=w
                checkpoints[ticks]=dict(global_step=int(refs['cg'].global_step),elapsed_s=time.monotonic()-start)
        return original(bar,n)
    with patch.object(core,'CoarseGrainer',capture('cg',core.CoarseGrainer)),patch.object(core,'SeparableCritic',capture('critic',core.SeparableCritic)),patch.object(tqdm.tqdm,'update',observe),open(os.devnull,'w') as quiet,redirect_stderr(quiet):
        trace,h,filters,encoder=core.train_RSMI_optimiser(cg,critic_opts,opt,dict(N_samples=ROWS,dimension=3),V=v,E=e,use_GPU=True,optimizer=adam)
    steps=int(adam.iterations.numpy());elapsed=time.monotonic()-start;del v,e
    status='complete' if steps==DEPTHS[-1] and ticks==steps and np.isfinite(trace).all() else 'nonfinite' if ticks==steps-1 else 'partial'
    n_seen=(steps//BATCHES)*ROWS+min(ROWS,(steps%BATCHES)*500)
    rec=dict(index=b,beta=cov['beta'],origins=k,seed=seed,steps=steps,examples=n_seen,full_epochs=steps//BATCHES,train_parents=PARENTS if steps>=BATCHES else None,train_identity=cov['train_identity'],test_identity=cov['test_identity'],bank_rows=ROWS,device=device,uuid=uid,pid=os.getpid(),train_s=elapsed,status=status,checkpoints=checkpoints,last_batch=36,trace_ceiling_violations=int(np.sum(trace>np.log(500)+1e-5)),base=stock.BASE,design=meta['design'])
    weights.update(train=trace,train_draws=h,train_filters=filters,last_kernel=encoder.get_weights()[0])
    if checkpoints:
        with load('test') as z: test={key:z[key] for key in ('V','E','ids')}
        assert hashlib.sha256(test['ids'].tobytes()).hexdigest()==cov['test_identity']
        nc=len(refs['critic'].weights)
        for d,record in checkpoints.items():
            ew=weights[f'enc_{d}'];cw=[weights[f'c{d}_{i:02}'] for i in range(nc)]
            if not all(np.isfinite(w).all() for w in [ew,*cw]):
                record.update(valid=False,status='nonfinite weights',test_parents=0);continue
            encoder.set_weights([ew]);refs['critic'].set_weights(cw);refs['cg'].global_step=record['global_step']
            testrec,draws=assessment(b,test,encoder,refs['critic']);record.update(testrec);weights[f'test_h_{d}']=draws
            record['status']='complete' if testrec['valid'] else 'invalid stock readout'
        if DEPTHS[-1] in checkpoints and not checkpoints[DEPTHS[-1]]['valid']: rec['status']=checkpoints[DEPTHS[-1]]['status']
    assert int(adam.iterations.numpy())==steps
    np.savez_compressed(stem+'.npz',receipt=json.dumps(rec,allow_nan=False),**weights)
    print('ORIGINS_RESULT',stem,steps,n_seen,rec['status'],flush=True)


def launch(a):
    import modal
    import queue
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime,timezone
    from decimal import Decimal
    from modal.stream_type import StreamType
    life,cpu,gib=4800,12,56; prior=Decimal('3.26270440'); rates=modal.Workspace.from_context().billing.rates()
    cost=Decimal(life)/3600*(rates['gpu_hour_cost_h100']+cpu*rates['cpu_hour_cost_sandbox']+gib*rates['mem_gib_hour_cost_sandbox'])+8*rates['egress_gib_cost']
    if not (cost<=Decimal(str(a.budget))<=min(Decimal('9.66'),25-prior)): raise ValueError(f'Budget: prior {prior}, reservation {cost}, invocation {a.budget}')
    meta=json.loads((a.inputs/'prepared.json').read_bytes());assert meta['design']==stock.sha(DESIGN) and meta['unique']==163840 and len(meta['inputs'])==20
    for name,digest in meta['inputs'].items(): assert stock.sha(a.inputs/name)==digest
    repo=Path(__file__).resolve().parents[1]; pinned=json.loads((repo/'results/stock-glass-all-beta-critic/launch.json').read_bytes())['wheel_sha256']
    if {p.name:stock.sha(p) for p in a.wheels.glob('*.whl')}!=pinned: raise ValueError('Offline runtime changed')
    a.out.mkdir(parents=True,exist_ok=False); stamp=lambda:datetime.now(timezone.utc).isoformat()
    stock.write_json(a.out/'reservation.json',dict(prior_usd=str(prior),max_usd=str(cost),cpu=cpu,gib=gib,seconds=life,source=stock.sha(Path(__file__)),design=meta['design'],image=stock.IMAGE,plans=90))
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp);subprocess.run(['git','archive','--format=tar','-o',str(tmp/'fork.tar'),stock.BASE,'rsmine','examples/tutorial.ipynb'],cwd=repo,check=True)
        subprocess.run(['bash','-ec','set -o pipefail; tar -C "$1" -cf - . | split -b 384M - "$2/wheels-"','pack',str(a.wheels.resolve()),str(tmp)],check=True)
        app=modal.App.lookup(STUDY,create_if_missing=True)
        sb=modal.Sandbox.create('/bin/sleep',str(life),name=STUDY,app=app,image=modal.Image.from_id(stock.IMAGE),gpu='H100!',cpu=(cpu,cpu),memory=(gib*1024,gib*1024),block_network=True,timeout=life,tags={'study':STUDY})
        try:
            stock.write_json(a.out/'provider.json',dict(app=app.app_id,sandbox=sb.object_id,utc=stamp()));print('PROVIDER_ID',sb.object_id,app.app_id,'cap',cost,flush=True)
            remote=stock.REMOTE;fs=sb.filesystem;fs.make_directory(remote)
            for f in [*sorted(tmp.glob('wheels-*')),tmp/'fork.tar',Path(__file__),Path(stock.__file__),a.inputs/'prepared.json',*(a.inputs/n for n in meta['inputs'])]:
                fs.copy_from_local(f,remote+'/'+f.name)
            def run(cmd,secs):
                p=sb.exec('/bin/bash','-ec',cmd,workdir=remote,timeout=secs,stderr=StreamType.STDOUT,bufsize=1);tail=[]
                for line in p.stdout:
                    line=line.rstrip();tail=(tail+[line])[-12:]
                    if any(t in line for t in ('FIRST_CUDA_STEP','ORIGINS_RESULT','Error','Killed','Traceback')): print(line,flush=True)
                rc=p.wait()
                if rc: raise RuntimeError(f'worker rc={rc}: '+str(tail))
            run('set -o pipefail; mkdir wheels; cat wheels-* | tar -xf - -C wheels; rm wheels-*; tar -xf fork.tar; /opt/conda/bin/python -m venv env; env/bin/pip install --no-index --find-links=wheels --disable-pip-version-check -q '+' '.join(repr(x) for x in stock.REQUIREMENTS),1200)
            env='export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 TF_USE_LEGACY_KERAS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 WANDB_MODE=disabled; export LD_LIBRARY_PATH="$(find "$PWD/env" -path "*/nvidia/*/lib" -type d | paste -sd: -):${LD_LIBRARY_PATH:-}"; '
            slots=queue.Queue()
            for i in range(8): slots.put(i)
            def job(cell):
                b,k,s=cell;slot=slots.get();stem=f'b{b:02}-k{k}-s{s}';name=stem+'.npz'
                try:
                    run(env+f'env/bin/python -u stock_glass_origins.py fit --index {b} --origins {k} --seed {s} --slot {slot}',1200)
                    if not 0<fs.stat(remote+'/'+name).size<12*1024**2: raise ValueError('Invalid result size')
                    with (a.out/name).open('xb') as f: f.write(fs.read_bytes(remote+'/'+name))
                    print('SAVED_FIT',stem,stock.sha(a.out/name),flush=True)
                except Exception as exc:
                    stock.write_json(a.out/(stem+'-failure.json'),dict(cell=cell,error=str(exc),utc=stamp()));print('FIT_ERROR',stem,exc,flush=True)
                finally: slots.put(slot)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(job,[(b,k,s) for s in SEEDS for b in range(10) for k in LEVELS]))
        finally:
            rc=sb.terminate(wait=True);stock.write_json(a.out/'terminated.json',dict(sandbox=sb.object_id,utc=stamp(),code=rc));print('TERMINATED',sb.object_id,rc,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); cmds=parser.add_subparsers(dest='cmd',required=True)
    sub=cmds.add_parser('prepare')
    for key in ('archive','rosters','split','fixed','out'): sub.add_argument('--'+key,type=Path,required=True)
    fit=cmds.add_parser('fit');fit.add_argument('--index',type=int,choices=range(10),required=True);fit.add_argument('--origins',type=int,choices=LEVELS,required=True);fit.add_argument('--seed',type=int,choices=SEEDS,required=True)
    fit.add_argument('--slot',type=int,choices=range(8),required=True)
    l=cmds.add_parser('launch')
    for key in ('inputs','out','wheels'):l.add_argument('--'+key,type=Path,required=True)
    l.add_argument('--budget',type=float,required=True)
    args=parser.parse_args(); {'prepare':prepare,'fit':run_fit,'launch':launch}[args.cmd](args)
