"""Three waves of ten stock RSMI temperature fits sharing one H100.

  python examples/stock_glass_modal.py prepare --archive DATA --rosters rosters.npz --split split.json --out RUN
  python examples/stock_glass_modal.py launch --inputs RUN --out FRESH_RESULTS --wheels WHEELHOUSE --budget 12.61

Use Python 3.11 and binary wheels for REQUIREMENTS (pip download --only-binary=:all:).
No model/critic/gradient code; the imported fork is archived from BASE, not the working tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BASE = 'f898730809c1244b6ef749fdaeea2e8796949ab2'
IMAGE = 'im-1aUX0dAUmmZQ583Ohg7Icu'
SPLIT = 'a0532740cc0d8ebf95b654e2834d5a52faeaac47262ea5291727258b4991db01'
INDEX = 'd4c591bab7a75e785fb749fa42cc78353dfc6d093c6742ae8d621ca2113040ff'
ROSTERS = '6fa06a3c3151c0622af86a134f5d3bc42decd7caf2093ba019010523d7480db9'
REQUIREMENTS = 'tensorflow[and-cuda]==2.18.1 tf-keras==2.18.0 tensorflow-probability==0.25.0 keras==3.15.1 numpy==1.26.4 pandas==2.2.3 wandb==0.19.11 networkx==3.4.2 tqdm==4.67.1 scipy==1.14.1 matplotlib==3.9.4 seaborn==0.13.2 scikit-learn==1.6.1'.split()
REMOTE = '/tmp/stock-glass'
SEEDS = (6101, 6102, 6103)
EPOCHS, BATCHES, GPU_MIB = 589, 17, 4096
STUDY = 'stock-glass-h100-concurrent-v2'


def cuda_uuid():
    """Query the CUDA driver in the worker's visible device namespace."""
    import ctypes as ct
    lib=ct.CDLL('libcuda.so.1'); device=ct.c_int(); raw=(ct.c_ubyte*16)()
    if lib.cuInit(0) or lib.cuDeviceGet(ct.byref(device),0) or lib.cuDeviceGetUuid(ct.byref(raw),device) or not any(raw):
        raise RuntimeError('Could not identify physical CUDA device')
    import uuid
    return str(uuid.UUID(bytes=bytes(raw)))


def barrier(seed, label, index, timeout=300):
    """Gates the first step and the release of the second step on all ten workers."""
    mark=Path(f'wave-{seed}-{label}-{index:02}')
    mark.touch(exist_ok=False); start=time.monotonic()
    expected=[Path(f'wave-{seed}-{label}-{b:02}') for b in range(10)]
    while not all(p.exists() for p in expected):
        if time.monotonic()-start>timeout: raise TimeoutError(f'not all ten workers reached {label}')
        time.sleep(.05)


class Progress:
    """Observe stock tqdm after apply_gradients; never change an optimizer or a training batch."""
    def __init__(self, adam, seed, index, uid, synchronize=True):
        self.adam,self.seed,self.index,self.uid,self.synchronize=adam,seed,index,uid,synchronize
        self.original=sys.stderr; self.samples=[]; self.next=0.

    def write(self,text):
        if 'Gumbel-softmax temperature' not in text:
            if text.strip() and '\r' not in text: self.original.write(text)
            return len(text)
        now=time.monotonic()
        if now>=self.next:
            step=int(self.adam.iterations.numpy())
            self.samples.append([now,step]);self.next=now+5.
            if len(self.samples)==1:
                if step!=1: raise RuntimeError(f'first observed Adam iteration was {step}')
                print('FIRST_CUDA_STEP',self.seed,self.index,os.getpid(),self.uid,step,text.strip(),flush=True)
                if self.synchronize: barrier(self.seed,'step1',self.index)
        return len(text)

    def flush(self): self.original.flush()


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def prepare(a):
    import numpy as np
    index = a.archive/'release-index.json'
    if (sha(index),sha(a.split),sha(a.rosters)) != (INDEX,SPLIT,ROSTERS):
        raise ValueError('Archive or exact parent split differs')
    split = json.loads(a.split.read_bytes()); a.out.mkdir(parents=True,exist_ok=False)
    manifest = {}; seen = set()
    grid = np.indices((9,9,9)).reshape(3,-1).T-4
    shell = (10+grid[np.abs(grid).max(1)==4]).T
    with np.load(a.rosters,allow_pickle=False) as rosters:
        for meta in json.loads(index.read_bytes())['endpoints']:
            b = meta['release_index']; tag = f'b{b:02}'; path = a.archive/meta['npz']
            if sha(path)!=meta['npz_sha256']: raise ValueError('Changed native snapshot')
            with np.load(path,allow_pickle=False) as data:
                ids = np.column_stack((data['chain_seed'],data['chain_id'])).astype('<u8'); species=data['species']
                if species.shape!=(16384,20,20,20) or species.dtype!=np.uint8 or species.max()>2: raise ValueError('Native geometry')
                t,r = (rosters[tag+'_'+side] for side in ('train','test'))
                if len(t)!=8192 or len(r)!=8192 or not np.array_equal(np.sort(np.r_[t,r]),np.arange(16384)): raise ValueError('Split membership')
                # One fixed translation per native parent, no overlapping rows across folds.
                for side,rows in [('train',t),('test',r)]:
                    keys=ids[rows]; observed=hashlib.sha256(keys.tobytes()).hexdigest()
                    if observed!=split['per_beta'][str(b)][side+'_identity_sha256']: raise ValueError('Parent identity mismatch')
                    seen.update(map(tuple,keys.tolist())); filename=tag+'-'+side+'.npz'
                    np.savez_compressed(a.out/filename, V=species[rows,9:12,9:12,9:12], E=species[rows[:,None],*shell], ids=keys)
                    manifest[filename]=sha(a.out/filename)
    if len(seen)!=163840 or len(manifest)!=20: raise ValueError('Incomplete or repeated identities')
    write_json(a.out/'prepared.json',dict(base=BASE,split=SPLIT,inputs=manifest,betas=[split['per_beta'][str(i)]['beta'] for i in range(10)],train=81920,test=81920))


def endpoint_test(index, tensor, encoder, critic, encode):
    """Frozen stock H (rows) vs E (columns), one draw/parent and one score matrix/batch."""
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp
    from scipy.stats import t
    tf.keras.utils.set_random_seed(440311+index)
    bounds=tfp.vi.mutual_information.lower_bound_info_nce
    draws=[]; batches=[]
    for j,start in enumerate(range(0,len(tensor['ids']),500)):
        sel=slice(start,start+500); v=encode(tensor['V'][sel])[...,None]; n=len(v)
        h=encoder(v,training=False); e=encode(tensor['E'][sel]).reshape(n,-1)
        scores=critic(e,h,training=False)
        if scores.shape!=(n,n) or not bool(tf.reduce_all(tf.math.is_finite(scores))):
            raise ValueError('Unscorable frozen stock critic matrix')
        draws.append(h.numpy()); rng=np.random.default_rng([771904,index,j]); ids=np.arange(n)
        perm=rng.permutation(n)
        while np.any(ids==perm): perm=rng.permutation(n)
        true=float(bounds(logu=scores,joint_sample_mask=tf.eye(n,dtype=tf.bool)))
        null=float(bounds(logu=scores,joint_sample_mask=tf.equal(ids[None,:],perm[:,None])))
        batches.append(dict(n=n,paired=true,shuffled=null,difference=true-null,permutation_sha256=hashlib.sha256(perm.astype('<i4').tobytes()).hexdigest()))
    n=np.array([r['n'] for r in batches]); assert n.tolist()==[500]*16+[192]
    w=n/n.sum(); result={}
    for key in ('paired','shuffled','difference'):
        x=np.array([r[key] for r in batches]);mu=float(w@x);se=float(np.sqrt(len(w)/(len(w)-1)*np.sum((w*(x-mu))**2)))
        half=t.ppf(.975,len(w)-1)*se
        if not np.isfinite([mu,half]).all(): raise ValueError('Nonfinite held-out score')
        result[key]=dict(mean_nats=mu,batch_se=se,ci95=[mu-half,mu+half])
    return np.concatenate(draws),dict(batches=batches,summary=result,parents=int(n.sum()),independent_batches=len(n),df=len(n)-1)


def worker(a):
    # Set per-process VRAM ceiling after physical discovery but before any TF logical device is initialized.
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)>=10:
        os.sched_setaffinity(0,{cpus[a.index]})
    import numpy as np
    import tensorflow as tf
    from importlib.metadata import version
    b=a.index; tag=f'b{b:02}'; g=tf.config.list_physical_devices('GPU')
    if len(g)!=1 or not tf.test.is_built_with_cuda(): raise RuntimeError('CUDA GPU required; no CPU fallback')
    tf.config.set_logical_device_configuration(g[0],[tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MIB)])
    details=tf.config.experimental.get_device_details(g[0]); uid=cuda_uuid()
    from rsmine.coarsegrainer.cg_optimisers import train_RSMI_optimiser, RSMI_estimate
    with tf.device('/GPU:0'):
        placement=tf.linalg.matmul(tf.ones((2,2)),tf.ones((2,2))).device
    if 'GPU:0' not in placement or 'H100' not in details['device_name']: raise RuntimeError((placement,details))
    tf.keras.utils.set_random_seed(a.seed)
    cg=dict(ll=(3,3,3),visible_dim=2,nonlinearCG=None,init_temperature=.75,min_temperature=.1,relaxation_rate=.0001,Nq=None,conv_activation=None,hidden_dim=2,h_embed=True,use_probs=False,use_logits=True)
    critic=dict(layers=3,embed_dim=8,hidden_dim=32,activation='relu',use_dropout=True,dropout_rate=.2)
    opt=dict(batch_size=500,iterations=EPOCHS,shuffle=100000,learning_rate=.009)
    def encode(v): return np.stack([v==1,v==2],axis=-1).astype('f')  # A,B indicators; vacancy=(0,0)
    meta=json.loads(Path('prepared.json').read_bytes()); beta=meta['betas'][b]
    def path(side):
        p=Path(tag+'-'+side+'.npz')
        if sha(p)!=meta['inputs'][p.name]: raise ValueError('Transferred snapshot mismatch')
        return p
    with np.load(path('train'),allow_pickle=False) as d: v=encode(d['V'])[...,None];e=encode(d['E']).reshape(8192,-1)
    adam=tf.keras.optimizers.Adam(opt['learning_rate'])  # identical to the stock default; retain its actual iterations
    endpoint=a.endpoint
    if endpoint and (a.seed!=6101 or b not in (0,9)): raise ValueError('Unregistered endpoint replay')
    progress=Progress(adam,a.seed,b,uid,synchronize=not endpoint)
    from rsmine.coarsegrainer import cg_optimisers
    from contextlib import redirect_stderr, nullcontext
    from unittest.mock import patch
    captured=[]; original=cg_optimisers.SeparableCritic
    def capture(*args,**kwargs):
        if captured: raise RuntimeError('Stock trainer constructed multiple critics')
        obj=original(*args,**kwargs); captured.append(obj); return obj
    if not endpoint: barrier(a.seed,'ready',b)
    start=time.monotonic()
    with (patch.object(cg_optimisers,'SeparableCritic',capture) if endpoint else nullcontext()), redirect_stderr(progress):
        estimates,h,filters,encoder=train_RSMI_optimiser(cg,critic,opt,dict(N_samples=8192,dimension=3),V=v,E=e,use_GPU=True,optimizer=adam)
    end=time.monotonic(); steps=int(adam.iterations.numpy())
    if endpoint:
        if len(captured)!=1 or cg_optimisers.SeparableCritic is not original: raise RuntimeError('Stock critic capture failed')
        stored={f'critic_{j:02}':v for j,v in enumerate(captured[0].get_weights())}
        stored.update(estimates=estimates,coarse_vars=h,filters=filters,final_kernel=encoder.get_weights()[0])
        complete=steps==EPOCHS*BATCHES and len(estimates)==EPOCHS and all(np.isfinite(x).all() for x in stored.values())
        rec=dict(beta=beta,index=b,seed=a.seed,base=BASE,split=SPLIT,gpu_uuid=uid,actual_steps=steps,complete=bool(complete),
                 train_wall_s=end-start,optimizer_samples=progress.samples,train_parents=8192,test_parents_scored=0,
                 critic_class=type(captured[0]).__name__,critic_weight_names=[x.name for x in captured[0].weights],
                 test_score=None,status='stock training incomplete or nonfinite; no TEST score')
        if complete:
            with np.load(path('test'),allow_pickle=False) as z:
                test={k:z[k] for k in ('V','E','ids')}
            try:
                relaxed,score=endpoint_test(b,test,encoder,captured[0],encode)
            except ValueError as exc: rec['status']='frozen evaluation failed: '+str(exc)
            else:
                stored.update(test_ids=test['ids'],test_relaxed=relaxed)
                rec.update(status='complete',test_score=score,test_parents_scored=8192)
        if steps!=int(adam.iterations.numpy()): raise RuntimeError('Optimizer changed during evaluation')
        np.savez_compressed(f'{tag}-s{a.seed}.npz',receipt=json.dumps(rec),**stored)
        print('ENDPOINT_RESULT',b,a.seed,steps,rec['status'],flush=True);return
    if steps!=EPOCHS*BATCHES or len(estimates)!=EPOCHS or not np.isfinite(estimates).all() or len(progress.samples)<2:
        raise RuntimeError(f'Incomplete stock training ({steps} optimizer steps)')
    ema=float(RSMI_estimate(estimates,ema_span=20))
    # Public return includes the encoder, not the fitted critic. Do not fit a replacement on TEST.
    with np.load(path('test'),allow_pickle=False) as d:
        testv=encode(d['V'])[...,None]; ids=d['ids']; relaxed=np.concatenate([encoder(testv[i:i+500],training=False).numpy() for i in range(0,8192,500)])
    if relaxed.shape!=(8192,2) or not np.isfinite(relaxed).all(): raise ValueError('Returned encoder output')
    data=dict(beta=beta,index=b,base=BASE,split=SPLIT,seed=a.seed,gpu=details['device_name'],gpu_uuid=uid,pid=os.getpid(),tf_gpu=placement,tf=tf.__version__,tfp=version('tensorflow-probability'),tfkeras=version('tf-keras'),gpu_limit_mib=GPU_MIB,peak_tf_bytes=tf.config.experimental.get_memory_info('GPU:0')['peak'],cg=cg,critic=critic,opt=opt,train_parents=8192,test_parents=8192,actual_adam_iterations=steps,last_batch_each_epoch=192,training_relaxed_InfoNCE_EMA20_nats=ema,train_start=start,train_end=end,wall_s=end-start,optimizer_samples=progress.samples,steps_per_s=steps/(end-start),short_filter='filters[100]: actual update 1701, first stock filter after 100 complete epochs',test_score=None)
    stem=f'{tag}-s{a.seed}'; np.savez_compressed(stem+'.npz',receipt=json.dumps(data),estimates=estimates,coarse_vars=h,filters=filters,final_kernel=encoder.get_weights()[0],test_ids=ids,test_relaxed=relaxed,test_thresholded=(relaxed>=.5))
    write_json(Path(stem+'.json'),data)
    print('STOCK_DONE',a.seed,b,steps,round(data['steps_per_s'],2),flush=True)


def wave(a):
    """Ten subprocesses, one device; failure terminates the wave, never retries serially."""
    start=time.monotonic(); procs=[]
    try:
        for b in range(10):
            procs.append(subprocess.Popen([sys.executable,'-u',__file__,'worker','--index',str(b),'--seed',str(a.seed)]))
        while any(p.poll() is None for p in procs):
            if any(p.returncode not in (None,0) for p in procs): raise RuntimeError(f'worker exit codes {[p.returncode for p in procs]}')
            time.sleep(.25)
        if any(p.returncode!=0 for p in procs): raise RuntimeError('incomplete wave')
    finally:
        for p in procs:
            if p.poll() is None: p.kill()
        for p in procs: p.wait()
    rows=[json.loads(Path(f'b{b:02}-s{a.seed}.json').read_bytes()) for b in range(10)]
    common_start=max(r['optimizer_samples'][0][0] for r in rows); common_end=min(r['train_end'] for r in rows)
    samples=[[s for s in r['optimizer_samples'] if common_start<=s[0]<=common_end] for r in rows]
    if not (len({r['gpu_uuid'] for r in rows})==1 and len({r['pid'] for r in rows})==10 and
            all(len(x)>=2 and x[-1][1]>x[0][1] for x in samples)): raise RuntimeError('ten-way active optimizer overlap not demonstrated')
    end=max(r['train_end'] for r in rows); begin=min(r['train_start'] for r in rows)
    proof=dict(seed=a.seed,uuid=rows[0]['gpu_uuid'],common_active_s=common_end-common_start,wall_s=time.monotonic()-start,
               train_span_s=end-begin,optimizer_steps=sum(r['actual_adam_iterations'] for r in rows),
               updates_per_second=sum(r['actual_adam_iterations'] for r in rows)/(end-begin),
               workers=[dict(index=r['index'],pid=r['pid'],first_inside=x[0],last_inside=x[-1]) for r,x in zip(rows,samples)])
    write_json(Path(f'wave-{a.seed}.json'),proof); print('VERIFIED_WAVE',a.seed,proof['uuid'],round(proof['common_active_s'],2),round(proof['updates_per_second'],2),flush=True)


def launch(a):
    import modal
    from decimal import Decimal
    from modal.stream_type import StreamType
    from datetime import datetime, timezone
    ep=a.endpoint_replay; rates=modal.Workspace.from_context().billing.rates()
    life,cpu,gib=(1800,4,16) if ep else (6000,16,48)
    carry=Decimal('11.47')+Decimal('.46928229')+(Decimal('.47796996') if ep else 0)
    cost=Decimal(life)/3600*(rates['gpu_hour_cost_h100']+cpu*rates['cpu_hour_cost_sandbox']+gib*rates['mem_gib_hour_cost_sandbox'])+8*rates['egress_gib_cost']
    if not (cost<=Decimal(str(a.budget))<=Decimal('25')-carry): raise ValueError('Original cumulative budget exceeded')
    meta=json.loads((a.inputs/'prepared.json').read_bytes()); assert meta['base']==BASE and meta['split']==SPLIT and len(meta['inputs'])==20
    for file,digest in meta['inputs'].items():
        if sha(a.inputs/file)!=digest: raise ValueError(file)
    repo=Path(__file__).resolve().parents[1]; stage='stock-glass-endpoint-critic' if ep else STUDY
    a.out.mkdir(parents=True,exist_ok=False)
    app=modal.App.lookup(stage,create_if_missing=True)
    stamp=lambda:datetime.now(timezone.utc).isoformat(); sb=None
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp); subprocess.run(['git','archive','--format=tar','-o',str(tmp/'fork.tar'),BASE,'rsmine','examples/tutorial.ipynb'],cwd=repo,check=True)
        subprocess.run(['bash','-ec','set -o pipefail; tar -C "$1" -cf - . | split -b 384M - "$2/wheels-"','pack',str(a.wheels.resolve()),str(tmp)],check=True)
        parts=sorted(tmp.glob('wheels-*'))
        if not parts or not 0<sum(p.stat().st_size for p in parts)<7*1024**3: raise ValueError('Offline wheel payload outside bounds')
        write_json(a.out/'launch.json',dict(utc=stamp(),total_max_usd=float(cost),prior_metered_usd=2.53855756 if ep else 2.06058760,prior_carry_usd=float(carry),life_s=life,cpu=cpu,memory_gib=gib,one_h100=True,workers=2 if ep else 10,waves=[6101] if ep else SEEDS,epochs=EPOCHS,per_worker_gpu_mib=GPU_MIB,inputs=meta,short='stock snapshot 1701, EMA20 of first 100 logged estimates; same trajectory',readout_plan_sha256=sha(repo/'results/stock-glass-endpoint-critic/readout-plan.json') if ep else None,launcher_sha256=sha(Path(__file__)),wheel_sha256={p.name:sha(p) for p in sorted(a.wheels.glob('*.whl'))}))
        try:
            sb=modal.Sandbox.create('/bin/sleep',str(life),name=stage,app=app,image=modal.Image.from_id(IMAGE),gpu='H100!',cpu=(cpu,cpu),memory=(gib*1024,gib*1024),block_network=True,timeout=life,tags={'study':stage})
            print('PROVIDER_ID',app.app_id,sb.object_id,stamp(),'max_usd',cost,flush=True); write_json(a.out/'provider.json',dict(app=app.app_id,sandbox=sb.object_id,utc=stamp(),image=IMAGE))
            fs=sb.filesystem; fs.make_directory(REMOTE)
            for p in [*parts,tmp/'fork.tar',a.inputs/'prepared.json',Path(__file__).resolve()]: fs.copy_from_local(p,REMOTE+'/'+p.name)
            def run(command,limit):
                p=sb.exec('/bin/bash','-ec',command,workdir=REMOTE,timeout=limit,stderr=StreamType.STDOUT,bufsize=1)
                for line in p.stdout: print(line.rstrip(),flush=True)
                rc=p.wait()
                if rc!=0: raise RuntimeError(f'Stock sandbox command failed: {rc}')
            run('set -o pipefail; mkdir wheels; cat wheels-* | tar -xf - -C wheels; rm wheels-*; tar -xf fork.tar; /opt/conda/bin/python -m venv env; env/bin/pip install --no-index --find-links=wheels --disable-pip-version-check -q '+ ' '.join(repr(x) for x in REQUIREMENTS)+'; echo OFFLINE_RUNTIME_READY',1200)
            env='export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 TF_USE_LEGACY_KERAS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 WANDB_MODE=disabled; export LD_LIBRARY_PATH="$(find "$PWD/env" -path "*/nvidia/*/lib" -type d | paste -sd: -):${LD_LIBRARY_PATH:-}"; '
            for name in meta['inputs']:
                if not ep or name.startswith(('b00-','b09-')): fs.copy_from_local(a.inputs/name,REMOTE+'/'+name)
            if ep:
                from concurrent.futures import ThreadPoolExecutor
                def replay(b):
                    run(env+f'env/bin/python -u stock_glass_modal.py worker --index {b} --seed 6101 --endpoint',1200)
                    name=f'b{b:02}-s6101.npz'
                    if not 0<fs.stat(REMOTE+'/'+name).size<8*1024**2: raise ValueError('Unexpected endpoint result size')
                    with (a.out/name).open('xb') as f: f.write(fs.read_bytes(REMOTE+'/'+name))
                    print('SAVED_ENDPOINT',b,sha(a.out/name),flush=True)
                with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(replay,(0,9)))
                return
            for seed in SEEDS:
                run(env+f'env/bin/python -u stock_glass_modal.py wave --seed {seed}',life)
                for name in [f'wave-{seed}.json',*[f'b{b:02}-s{seed}.npz' for b in range(10)]]:
                    if not 0<fs.stat(REMOTE+'/'+name).size<8*1024**2: raise ValueError('Invalid stock result size')
                    with (a.out/name).open('xb') as f: f.write(fs.read_bytes(REMOTE+'/'+name))
                print('SAVED_WAVE',seed,flush=True)
        finally:
            if sb is not None:
                rc=sb.terminate(wait=True); write_json(a.out/'terminated.json',dict(sandbox=sb.object_id,utc=stamp(),code=rc)); print('TERMINATED',sb.object_id,rc,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--archive',type=Path,required=True); p.add_argument('--split',type=Path,required=True); p.add_argument('--rosters',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('launch'); p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--wheels',type=Path,required=True);p.add_argument('--budget',type=float,required=True)
    p.add_argument('--endpoint-replay',action='store_true')
    p=sub.add_parser('worker');p.add_argument('--index',type=int,required=True,choices=range(10));p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    p.add_argument('--endpoint',action='store_true')
    p=sub.add_parser('wave');p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    a=parser.parse_args(); {'worker':worker,'prepare':prepare,'launch':launch,'wave':wave}[a.cmd](a)
