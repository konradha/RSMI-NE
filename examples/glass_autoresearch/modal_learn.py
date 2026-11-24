"""One explicitly funded PyTorch fit on a read-only archived release. No simulator is mounted.

The decoder's separate, frozen confirm command runs locally. This runner only searches.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
IMAGE = 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755'
VOLUME = 'rsmi-archived-compression-20261124'
LEDGER = Path.home()/'.local/state/rsmi/archive-learning-1795520725'
END = 1795556725
QUEUE, STARTUP, TEARDOWN = 155, 130, 45
RATE = {'L4': .000222, 'L40S': .000542}


def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def put(path, rec):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');temp.replace(path)


def worker(req, queue, vol_id, latest):
    import modal
    started=time.monotonic()
    if time.time()>latest: raise RuntimeError('queue lease expired before start')
    modal.Queue.from_id(queue).put(True)
    volume=modal.Volume.from_id(vol_id); dest=Path('/history')/req['name'];dest.mkdir(exist_ok=False)
    stat={'status':'failed','name':req['name'],'kind':'existing-data training only'}
    try:
        for k,sha in req['inputs'].items():
            if digest(Path('/inputs')/k)!=sha: raise ValueError(f'change after admission: {k}')
        stat['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True,timeout=12).strip()
        exe='/opt/conda/bin/python'; stat['packages']=subprocess.check_output([exe,'-c','import torch,numpy,scipy; print(torch.__version__,numpy.__version__,scipy.__version__); assert torch.cuda.is_available()'],text=True,timeout=25).strip()
        argv=[exe,'-u','/inputs/examples/glass_autoresearch/decoder.py','--mode','search','--device','cuda','--data','/inputs/data','--indices','9','--out',str(dest),'--end-epoch',str(min(END,int(time.time()+req['seconds']-(time.monotonic()-started)-18))),*req['fit_args']]
        stat['args']=argv
        log=Path('/tmp')/(req['name']+'.log')
        with log.open('w') as f:
            proc=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'},timeout=max(3,req['seconds']-(time.monotonic()-started)-8))
        stat['exit']=proc.returncode
        (dest/'fit.log').write_bytes(log.read_bytes())
        if proc.returncode!=0 or not (dest/'validation.json').is_file(): raise RuntimeError('decoder did not finish: '+(dest/'fit.log').read_text()[-1200:])
        stat['status']='ok'
    except Exception as e: stat['error']=repr(e)
    stat['worker_s']=time.monotonic()-started;stat['compute_usd_at_allocated_rate_not_invoice']=stat['worker_s']*req['rate'];put(dest/'modal-receipt.json',stat);volume.commit()
    return {p.name:p.read_bytes() for p in dest.iterdir() if p.is_file() and p.stat().st_size<=12*1024**2}


def run(a):
    import modal
    if sys.version_info[:2]!=(3,11): raise ValueError('use a Python 3.11 Modal client: the pinned PyTorch worker runs /opt/conda/bin/python 3.11')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,49}',a.name): raise ValueError('invalid cell name')
    if not (0<a.budget_usd<=25 and math.isfinite(a.budget_usd) and 45<=a.seconds<=5400 and 0<a.steps<=131072 and 0<a.batch<=8192 and a.buffer in (1,2,3) and a.seed>=0): raise ValueError('invalid paid-work limits')
    duration=QUEUE+STARTUP+TEARDOWN+a.seconds
    if time.time()+duration>END: raise ValueError('allocation could cross campaign deadline')
    rate=RATE[a.gpu]+2*.0000131+6*.00000222
    # Full queue, startup, capped function and teardown charged even if unsuccessful;
    # flat cushion for CPU image preparation and transferring small artifacts.
    reserve=duration*rate+.20
    if reserve>a.budget_usd:raise ValueError(f'needs ${reserve:.3f} conservative reservation, above requested ${a.budget_usd:.2f}')
    files={'examples/glass_autoresearch/decoder.py':HERE/'decoder.py','examples/l20_pairs.py':HERE.parent/'l20_pairs.py','data/beta-09.npz':a.data/'beta-09.npz','data/beta-09.json':a.data/'beta-09.json'}
    sha={k:digest(v) for k,v in files.items()}; assert sha['data/beta-09.npz']==json.loads(files['data/beta-09.json'].read_bytes())['endpoint_sha256']
    req=dict(name=a.name,gpu=a.gpu,inputs=sha,image=IMAGE,volume=VOLUME,seconds=a.seconds,rate=rate,reservation_usd=reserve,requested_budget_usd=a.budget_usd,fit_args=['--seed',str(a.seed),'--buffer',str(a.buffer),'--steps',str(a.steps),'--batch',str(a.batch),'--lr','.002','--save-every',str(max(8,min(256,a.steps//2))),'--eval-every',str(max(8,min(512,a.steps//2))),'--curve-anchors',str(a.curve_anchors),'--eval-anchors',str(a.eval_anchors),'--calibrate-steps',str(a.calibrate_steps)])
    LEDGER.mkdir(exist_ok=True,parents=True); dest=a.out/a.name
    with (LEDGER/'lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX); prev=sum(json.loads(p.read_bytes())['reservation_usd'] for p in LEDGER.glob('*.json'))
        if prev+reserve>25 or (LEDGER/(a.name+'.json')).exists():raise ValueError('budget exhausted or cell already admitted')
        req['cumulative_reserved_usd']=prev+reserve;dest.mkdir(exist_ok=False,parents=True);put(LEDGER/(a.name+'.json'),req);put(dest/'admission.json',req)
    print(f"Funding {a.name}: ${reserve:.3f} reserved, ${prev+reserve:.3f}/25 cumulative. No sampling.",flush=True)
    image=modal.Image.from_registry(IMAGE,add_python='3.12').run_commands('/opt/conda/bin/python -m pip install --disable-pip-version-check numpy==1.26.4 scipy==1.14.1')
    for k,v in files.items():image=image.add_local_file(v,'/inputs/'+k,copy=False)
    volume=modal.Volume.from_name(VOLUME,create_if_missing=True);app=modal.App('rsmi-archive-learning',include_source=False)
    fn=app.function(image=image,gpu=a.gpu,cpu=(2,2),memory=(6144,6144),retries=0,min_containers=0,buffer_containers=0,max_containers=1,scaledown_window=2,timeout=a.seconds,startup_timeout=STARTUP,volumes={'/history':volume},serialized=True)(worker)
    with modal.enable_output(),app.run(),modal.Queue.ephemeral() as ack:
        end=time.monotonic()+QUEUE; call=fn.spawn(req,ack.object_id,volume.object_id,time.time()+QUEUE)
        put(dest/'allocation.json',dict(app_id=app.app_id,call_id=call.object_id))
        try:
            ack.get(timeout=max(.01,end-time.monotonic())); output=call.get(timeout=a.seconds+5)
        except BaseException:
            call.cancel(terminate_containers=True);raise
    for name,b in output.items():
        if Path(name).name!=name or (dest/name).exists():raise ValueError('invalid output name')
        (dest/name).write_bytes(b)
    receipt=json.loads((dest/'modal-receipt.json').read_bytes()); print(json.dumps(receipt),flush=True)
    if receipt['status']!='ok':raise RuntimeError(receipt.get('error'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['data','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--budget-usd',type=float,required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--gpu',choices=RATE,default='L4')
    for k,v in [('seed',68239),('buffer',2),('steps',512),('batch',2048),('calibrate-steps',256),('curve-anchors',8),('eval-anchors',64)]:p.add_argument('--'+k,type=int,default=v)
    run(p.parse_args())
