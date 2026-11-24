"""One stock RSMI fit per temperature. Prepare locally, then launch one bounded GPU.

  python examples/stock_glass_modal.py prepare --archive DATA --rosters rosters.npz --split split.json --out RUN
  python examples/stock_glass_modal.py launch --out RUN --wheels WHEELHOUSE --budget 4.85

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


def worker(a):
    import numpy as np
    import tensorflow as tf
    from rsmine.coarsegrainer.cg_optimisers import train_RSMI_optimiser, RSMI_estimate
    from importlib.metadata import version
    b=a.index; tag=f'b{b:02}'; g=tf.config.list_physical_devices('GPU')
    if len(g)!=1 or not tf.test.is_built_with_cuda(): raise RuntimeError('CUDA GPU required; no CPU fallback')
    details=tf.config.experimental.get_device_details(g[0]); tf.config.experimental.set_memory_growth(g[0],True)
    with tf.device('/GPU:0'):
        placement=tf.linalg.matmul(tf.ones((2,2)),tf.ones((2,2))).device
    if 'GPU:0' not in placement or 'H100' not in details['device_name']: raise RuntimeError((placement,details))
    tf.keras.utils.set_random_seed(81791+b)
    cg=dict(ll=(3,3,3),visible_dim=2,nonlinearCG=None,init_temperature=.75,min_temperature=.1,relaxation_rate=.0001,Nq=None,conv_activation=None,hidden_dim=2,h_embed=True,use_probs=False,use_logits=True)
    critic=dict(layers=3,embed_dim=8,hidden_dim=32,activation='relu',use_dropout=True,dropout_rate=.2)
    opt=dict(batch_size=500,iterations=100,shuffle=100000,learning_rate=.009)
    def encode(v): return np.stack([v==1,v==2],axis=-1).astype('f')  # A,B indicators; vacancy=(0,0)
    meta=json.loads(Path('prepared.json').read_bytes()); beta=meta['betas'][b]
    def path(side):
        p=Path(tag+'-'+side+'.npz')
        if sha(p)!=meta['inputs'][p.name]: raise ValueError('Transferred snapshot mismatch')
        return p
    with np.load(path('train'),allow_pickle=False) as d: v=encode(d['V'])[...,None];e=encode(d['E']).reshape(8192,-1)
    print('GPU_READY',beta,details,placement,'tensorflow',tf.__version__,flush=True)
    start=time.monotonic()
    estimates,h,filters,encoder=train_RSMI_optimiser(cg,critic,opt,dict(N_samples=8192,dimension=3),V=v,E=e,use_GPU=True)
    if len(estimates)!=100 or not np.isfinite(estimates).all(): raise RuntimeError('Incomplete/nonfinite stock training')
    ema=float(RSMI_estimate(estimates,ema_span=20))
    # Public return includes the encoder, not the fitted critic. Do not fit a replacement on TEST.
    with np.load(path('test'),allow_pickle=False) as d:
        testv=encode(d['V'])[...,None]; ids=d['ids']; relaxed=np.concatenate([encoder(testv[i:i+500],training=False).numpy() for i in range(0,8192,500)])
    if relaxed.shape!=(8192,2) or not np.isfinite(relaxed).all(): raise ValueError('Returned encoder output')
    data=dict(beta=beta,base=BASE,split=SPLIT,seed=81791+b,gpu=details['device_name'],tf_gpu=placement,tf=tf.__version__,tfp=version('tensorflow-probability'),tfkeras=version('tf-keras'),cg=cg,critic=critic,opt=opt,train_parents=8192,test_parents=8192,stock_batches=1700,last_batch_each_epoch=192,training_relaxed_InfoNCE_EMA20_nats=ema,wall_s=time.monotonic()-start,test_score=None,test_score_reason='Stock return contains encoder but no trained critic; RSMI_estimate is a training-curve EMA, not held-out MI.')
    np.savez_compressed(tag+'-result.npz',receipt=json.dumps(data),estimates=estimates,coarse_vars=h,filters=filters,final_kernel=encoder.get_weights()[0],test_ids=ids,test_relaxed=relaxed,test_thresholded=(relaxed>=.5))
    print('STOCK_RESULT',json.dumps(data),flush=True)


def launch(a):
    import modal
    from decimal import Decimal
    from modal.stream_type import StreamType
    from datetime import datetime, timezone
    rates=modal.Workspace.from_context().billing.rates(); life=3600
    cost=Decimal(life)/3600*(rates['gpu_hour_cost_h100']+2*rates['cpu_hour_cost_sandbox']+12*rates['mem_gib_hour_cost_sandbox'])+Decimal('.32')
    if not (cost<=Decimal(str(a.budget))<=Decimal('25')-Decimal('11.47')): raise ValueError('Budget must cover all ten fits, including retained prior ceilings')
    meta=json.loads((a.out/'prepared.json').read_bytes()); assert meta['base']==BASE and meta['split']==SPLIT and len(meta['inputs'])==20
    for file,digest in meta['inputs'].items():
        if sha(a.out/file)!=digest: raise ValueError(file)
    repo=Path(__file__).resolve().parents[1]; stage='stock-glass-gpu-v1'; app=modal.App.lookup(stage,create_if_missing=True)
    stamp=lambda:datetime.now(timezone.utc).isoformat(); sb=None
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp); subprocess.run(['git','archive','--format=tar','-o',str(tmp/'fork.tar'),BASE,'rsmine','examples/tutorial.ipynb'],cwd=repo,check=True)
        subprocess.run(['bash','-ec','set -o pipefail; tar -C "$1" -cf - . | split -b 384M - "$2/wheels-"','pack',str(a.wheels.resolve()),str(tmp)],check=True)
        parts=sorted(tmp.glob('wheels-*'))
        if not parts or not 0<sum(p.stat().st_size for p in parts)<7*1024**3: raise ValueError('Offline wheel payload outside bounds')
        write_json(a.out/'launch.json',dict(utc=stamp(),total_max_usd=float(cost),prior_verified_usd=1.59130531,prior_conservative_carry_usd=11.47,ten_fits=True,life_s=life,launcher_sha256=sha(Path(__file__)),wheel_sha256={p.name:sha(p) for p in sorted(a.wheels.glob('*.whl'))}))
        try:
            sb=modal.Sandbox.create('/bin/sleep',str(life),name=stage,app=app,image=modal.Image.from_id(IMAGE),gpu='H100!',cpu=(2,2),memory=(12288,12288),block_network=True,timeout=life,tags={'study':stage})
            print('PROVIDER_ID',app.app_id,sb.object_id,stamp(),'max_usd',cost,flush=True); write_json(a.out/'provider.json',dict(app=app.app_id,sandbox=sb.object_id,utc=stamp(),image=IMAGE))
            fs=sb.filesystem; fs.make_directory(REMOTE)
            for p in [*parts,tmp/'fork.tar',a.out/'prepared.json',Path(__file__).resolve()]: fs.copy_from_local(p,REMOTE+'/'+p.name)
            def run(command,limit):
                p=sb.exec('/bin/bash','-ec',command,workdir=REMOTE,timeout=limit,stderr=StreamType.STDOUT,bufsize=1)
                for line in p.stdout: print(line.rstrip(),flush=True)
                rc=p.wait()
                if rc!=0: raise RuntimeError(f'Stock sandbox command failed: {rc}')
            run('set -o pipefail; mkdir wheels; cat wheels-* | tar -xf - -C wheels; rm wheels-*; tar -xf fork.tar; /opt/conda/bin/python -m venv env; env/bin/pip install --no-index --find-links=wheels --disable-pip-version-check -q '+ ' '.join(repr(x) for x in REQUIREMENTS)+'; echo OFFLINE_RUNTIME_READY',1200)
            env='export PYTHONPATH="$PWD" TF_USE_LEGACY_KERAS=1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 WANDB_MODE=disabled; export LD_LIBRARY_PATH="$(find "$PWD/env" -path "*/nvidia/*/lib" -type d | paste -sd: -):${LD_LIBRARY_PATH:-}"; '
            for b in range(10):
                tag=f'b{b:02}'
                for side in ('train','test'): fs.copy_from_local(a.out/f'{tag}-{side}.npz',f'{REMOTE}/{tag}-{side}.npz')
                # Display the stock optimiser's first completed CUDA training step; omit thousands of redundant tqdm redraws.
                run(env+"set -o pipefail; env/bin/python -u stock_glass_modal.py worker --index "+str(b)+" 2>&1 | tr '\\r' '\\n' | awk '/Gumbel-softmax temperature/ {if (!seen++) {print \"FIRST_CUDA_TRAINING_OUTPUT beta_index="+str(b)+"\",$0; fflush()} next} {print; fflush()}'",900)
                name=tag+'-result.npz'
                if not 0<fs.stat(REMOTE+'/'+name).size<8*1024**2: raise ValueError('Invalid stock result size')
                blob=fs.read_bytes(REMOTE+'/'+name)
                with (a.out/name).open('xb') as f: f.write(blob)
                print('SAVED',name,sha(a.out/name),flush=True)
        finally:
            if sb is not None:
                rc=sb.terminate(wait=True); write_json(a.out/'terminated.json',dict(sandbox=sb.object_id,utc=stamp(),code=rc)); print('TERMINATED',sb.object_id,rc,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--archive',type=Path,required=True); p.add_argument('--split',type=Path,required=True); p.add_argument('--rosters',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('launch'); p.add_argument('--out',type=Path,required=True); p.add_argument('--wheels',type=Path,required=True);p.add_argument('--budget',type=float,required=True)
    p=sub.add_parser('worker');p.add_argument('--index',type=int,required=True,choices=range(10)); a=parser.parse_args(); {'worker':worker,'prepare':prepare,'launch':launch}[a.cmd](a)
