"""03: train the pinned, unmodified stock optimiser on independent conditional replicas."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import stock_glass_modal as stock

STUDY='stock-glass-cavity-03'; SEEDS=(6101,6102,6103); EARLY=(850,1700,3400,6800)
ROOT=Path(__file__).resolve().parents[1]; DESIGN=ROOT/'results/experiment-03-cavity/design.json'


def encode_states(states,coords):
    import numpy as np
    raw=np.stack([states==1,states==2],axis=-1).astype('f')
    cube=np.zeros((len(states),3,3,3,2),'f');cube[:,coords[:,0],coords[:,1],coords[:,2],:]=raw
    return raw,cube[...,None]


def fit(a):
    from contextlib import redirect_stderr
    from unittest.mock import patch
    import numpy as np
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[a.slot%len(cpus)]})
    import tensorflow as tf
    from rsmine.coarsegrainer import cg_optimisers as core
    import tqdm
    b,s=a.index,a.seed;stem=f'b{b:02}-s{s}';gpu,=tf.config.list_physical_devices('GPU');tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    device=tf.config.experimental.get_device_details(gpu)['device_name'];uid=stock.cuda_uuid()
    with tf.device('/GPU:0'):placement=tf.matmul(tf.ones((2,2)),tf.ones((2,2))).device
    assert 'GPU:0' in placement and 'H100' in device;tf.keras.utils.set_random_seed(s)
    meta=json.loads(Path('prepared.json').read_bytes());cov=meta['coverage'][b];name=f'b{b:02}-pairs.npz';assert stock.sha(Path(name))==meta['inputs'][name]
    with np.load(name) as z:pairs=z['pairs'];X=z['states'];assert len(pairs)==cov['train']['selected_parents'] and pairs.shape[1:]==(16,2)
    with np.load('state-geometry.npz') as z:coords=z['star']+1;assert np.array_equal(X,z['states'])
    raw,visible=encode_states(X,coords)
    pairs=pairs.reshape(-1,2);v=visible[pairs[:,0]];e=raw[pairs[:,1]].reshape(len(pairs),14);n=len(e);B=math.ceil(n/500);epochs=math.ceil(10000/B);final=B*epochs
    cg=dict(ll=(3,3,3),visible_dim=2,nonlinearCG=None,init_temperature=.75,min_temperature=.1,relaxation_rate=.0001,Nq=None,conv_activation=None,hidden_dim=2,h_embed=True,use_probs=False,use_logits=True)
    critic=dict(layers=3,embed_dim=8,hidden_dim=32,activation='relu',use_dropout=True,dropout_rate=.2);opt=dict(batch_size=500,iterations=epochs,shuffle=100000,learning_rate=.009);adam=tf.keras.optimizers.Adam(.009)
    refs={};stored={};checkpoints={};ticks=0;start=time.monotonic()
    def capture(k,orig):
        def make(*args,**kw):
            assert k not in refs;refs[k]=orig(*args,**kw);return refs[k]
        return make
    orig=tqdm.tqdm.update
    def observe(bar,n=1):
        nonlocal ticks
        ticks+=n
        if ticks==1 or ticks in (*EARLY,final):
            assert int(adam.iterations.numpy())==ticks
            if ticks==1:print('FIRST_CUDA_STEP',stem,uid,ticks,flush=True)
            else:
                stored[f'w{ticks}']=refs['cg']._Λ.get_weights()[0];cw=refs['critic'].get_weights()
                for j,w in enumerate(cw):stored[f'c{ticks}_{j:02}']=w
                checkpoints[ticks]=dict(global_step=int(refs['cg'].global_step),tau=float(refs['cg'].tau),critic_weights_finite=all(bool(np.isfinite(w).all()) for w in cw),seconds=time.monotonic()-start)
        return orig(bar,n)
    with patch.object(core,'CoarseGrainer',capture('cg',core.CoarseGrainer)),patch.object(core,'SeparableCritic',capture('critic',core.SeparableCritic)),patch.object(tqdm.tqdm,'update',observe),open(os.devnull,'w') as quiet,redirect_stderr(quiet):
        trace,h,filters,enc=core.train_RSMI_optimiser(cg,critic,opt,dict(N_samples=n,dimension=3),V=v,E=e,use_GPU=True,optimizer=adam)
    steps=int(adam.iterations.numpy());wall=time.monotonic()-start;stored.update(train_trace=trace,train_h=h,train_filters=filters,returned_filter=enc.get_weights()[0]);del v,e
    for dep,r in checkpoints.items():
        w=stored[f'w{dep}'];r['map_valid']=bool(np.isfinite(w).all())
        if not r['map_valid']:continue
        enc.set_weights([w]);refs['cg'].global_step=r['global_step'];logit=refs['cg'].coarse_grainer(visible).numpy().reshape(210,2);stored[f'logits{dep}']=logit;cpu=raw.reshape(210,14)@w[coords[:,0],coords[:,1],coords[:,2]].reshape(14,2);r['max_contraction_difference']=float(np.max(abs(cpu-logit)));r['map_valid']=bool(np.isfinite(logit).all());tf.keras.utils.set_random_seed(309034+b);stored[f'relaxed_draw{dep}']=enc(visible,training=False).numpy()
    rec=dict(index=b,beta=cov['beta'],seed=s,base=stock.BASE,design=meta['design'],uuid=uid,device=device,epochs=epochs,batches=B,last_batch=n-(B-1)*500,final=final,steps=steps,positives=(steps//B)*n+min(steps%B*500,n),bank_positives=n,train_parents=cov['train']['selected_parents'] if steps>=B else None,seconds=wall,fit_status='complete' if steps==final and ticks==steps else 'nonfinite' if ticks+1==steps else 'partial',trace_above_ceiling=int(np.sum(trace>np.log(500)+1e-5)),checkpoints=checkpoints)
    assert int(adam.iterations.numpy())==steps;np.savez_compressed(stem+'.npz',receipt=json.dumps(rec,allow_nan=False),**stored);print('CAVITY_RESULT',stem,steps,rec['fit_status'],flush=True)


def launch(a):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime,timezone
    from decimal import Decimal as D
    import queue,subprocess,tempfile,modal
    from modal.stream_type import StreamType
    prior=D('5.67073697');r=modal.Workspace.from_context().billing.rates();life,cpu,gib=2100,10,34;cost=D(life)/3600*(r['gpu_hour_cost_h100']+cpu*r['cpu_hour_cost_sandbox']+gib*r['mem_gib_hour_cost_sandbox'])+8*r['egress_gib_cost'];assert cost<=D(str(a.budget))<=min(D('3.95'),25-prior)
    meta=json.loads((a.inputs/'prepared.json').read_bytes());bench=json.loads((DESIGN.parent/'exact-receipt.json').read_bytes());assert bench['unique']==meta['unique']==163840 and bench['design']==meta['design']==stock.sha(DESIGN) and bench['inputs']==stock.sha(a.inputs/'prepared.json') and len(bench['by_beta'])==10 and bench['by_beta'][0]['train_top']==0
    assert bench['eigenbasis']==stock.sha(DESIGN.parent/'frozen-modes.npz') and len(bench['energy_audits'])==20
    names=[f'b{b:02}-pairs.npz' for b in range(10)]+['state-geometry.npz'];assert all(stock.sha(a.inputs/n)==meta['inputs'][n] for n in names)
    pins=json.loads((ROOT/'results/stock-glass-all-beta-critic/launch.json').read_bytes())['wheel_sha256'];assert {p.name:stock.sha(p) for p in a.wheels.glob('*.whl')}==pins
    stamp=lambda:datetime.now(timezone.utc).isoformat();a.out.mkdir(parents=True,exist_ok=False);stock.write_json(a.out/'reservation.json',dict(prior=str(prior),cap=str(cost),life=life,cpu=cpu,gib=gib,design=meta['design'],benchmark=stock.sha(DESIGN.parent/'exact-receipt.json'),launcher=stock.sha(Path(__file__)),image=stock.IMAGE))
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp);subprocess.run(['git','archive','--format=tar','-o',str(tmp/'fork.tar'),stock.BASE,'rsmine','examples/tutorial.ipynb'],cwd=ROOT,check=True)
        subprocess.run(['bash','-ec','set -o pipefail; tar -C "$1" -cf - . | split -b 384M - "$2/wheels-"','pack',str(a.wheels.resolve()),str(tmp)],check=True)
        app=modal.App.lookup(STUDY,create_if_missing=True);sb=modal.Sandbox.create('/bin/sleep',str(life),name=STUDY,app=app,image=modal.Image.from_id(stock.IMAGE),gpu='H100!',cpu=(cpu,cpu),memory=(gib*1024,gib*1024),block_network=True,timeout=life,tags={'study':STUDY})
        try:
            stock.write_json(a.out/'provider.json',dict(sandbox=sb.object_id,app=app.app_id,utc=stamp()));print('PROVIDER_ID',sb.object_id,app.app_id,'max_usd',cost,flush=True);fs=sb.filesystem;remote=stock.REMOTE;fs.make_directory(remote)
            for f in [*sorted(tmp.glob('wheels-*')),tmp/'fork.tar',Path(__file__),Path(stock.__file__),a.inputs/'prepared.json',*(a.inputs/n for n in names)]:fs.copy_from_local(f,remote+'/'+f.name)
            def run(cmd,secs):
                pr=sb.exec('/bin/bash','-ec',cmd,timeout=secs,workdir=remote,stderr=StreamType.STDOUT,bufsize=1);tail=[]
                for line in pr.stdout:
                    line=line.rstrip();tail=(tail+[line])[-16:]
                    if any(x in line for x in ('FIRST_CUDA_STEP','CAVITY_RESULT','Traceback','Error')):print(line,flush=True)
                rc=pr.wait()
                if rc:raise RuntimeError(f'exec exit {rc}: {tail}')
            run('set -o pipefail; mkdir wheels; cat wheels-* | tar -xf - -C wheels; rm wheels-*; tar -xf fork.tar; /opt/conda/bin/python -m venv env; env/bin/pip install --no-index --find-links=wheels --disable-pip-version-check -q '+' '.join(repr(x) for x in stock.REQUIREMENTS),1200)
            env='export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 TF_USE_LEGACY_KERAS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 WANDB_MODE=disabled; export LD_LIBRARY_PATH="$(find "$PWD/env" -path "*/nvidia/*/lib" -type d | paste -sd: -):${LD_LIBRARY_PATH:-}"; '
            slots=queue.Queue()
            for j in range(8):slots.put(j)
            def execute(cell):
                b,s=cell;slot=slots.get();stem=f'b{b:02}-s{s}';name=stem+'.npz'
                try:
                    run(env+f'env/bin/python -u stock_glass_cavity_gpu.py fit --index {b} --seed {s} --slot {slot}',1100)
                    assert 0<fs.stat(remote+'/'+name).size<16*1024**2
                    with (a.out/name).open('xb') as f:f.write(fs.read_bytes(remote+'/'+name))
                    print('SAVED',stem,stock.sha(a.out/name),flush=True)
                except Exception as exc:stock.write_json(a.out/(stem+'-failed.json'),dict(error=str(exc)));print('FIT_ERROR',stem,exc,flush=True)
                finally:slots.put(slot)
            with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(execute,[(b,s) for s in SEEDS for b in range(10)]))
        finally:
            rc=sb.terminate(wait=True);stock.write_json(a.out/'terminated.json',dict(sandbox=sb.object_id,utc=stamp(),code=rc));print('TERMINATED',sb.object_id,rc,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True);f=sub.add_parser('fit');f.add_argument('--index',required=True,type=int,choices=range(10));f.add_argument('--seed',required=True,type=int,choices=SEEDS);f.add_argument('--slot',required=True,type=int,choices=range(8));l=sub.add_parser('launch')
    for name in ('inputs','wheels','out'):l.add_argument('--'+name,required=True,type=Path)
    l.add_argument('--budget',required=True,type=float);a=p.parse_args();{'fit':fit,'launch':launch}[a.cmd](a)
