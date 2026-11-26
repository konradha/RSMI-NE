"""One authorized submission on 2026-11-26. No retry, new workspace cap or TEST access.

The supervisor authorized $165 as a workspace *reference*, not an independently verified
account limit. Subtract live gross metered usage and take the lesser with remaining project
authorization and the explicit $2.22 all-in ceiling. A provider refusal ends the attempt.
Only the primary source FIT and DEV files are transferred.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal as D
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import stock_glass_modal as stock

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT/'results/history-overlap-field'
WORKER = Path(__file__).with_name('stock_glass_history_gpu.py')
STUDY = 'stock-glass-history-overlap-field'
CPU, GIB = 8, 40
MIN_S, EXT_S = 974, 1297
BANKS = tuple(f'b0-a500-q80-{side}.npz' for side in ('fit','dev'))


def get(path): return json.loads(path.read_bytes())

def stamp(): return datetime.now(timezone.utc).isoformat()

def require(ok, msg):
    if not ok: raise ValueError(msg)


def verify_inputs(a):
    import numpy as np
    gate = get(RESULT/'gate.json'); primary = gate['primary']; require(primary['pass'] is True and gate['held_parents_opened']==0 and primary['parents']==16,'CPU gate did not pass')
    require(stock.sha(RESULT/'design.md')==gate['design_sha256'], 'Unbound gate design')
    roster=list(csv.DictReader((RESULT/'split.csv').read_text().splitlines())); rows=list(csv.DictReader((RESULT/'map-banks-train.csv').read_text().splitlines())); seen=set(); hashes={}
    for name,n,side in zip(BANKS,(48,16),('fit','dev')):
        p=a.inputs/name; row,=[r for r in rows if Path(r['path']).name==name]; digest=stock.sha(p); hashes[name]=digest
        require(row['sha256']==digest and (side!='dev' or gate['map_sha256'][name]==digest),'Input digest mismatch')
        ids=[int(r['parent_id']) for r in sorted([r for r in roster if r['beta']=='3.20' and r['side']==side],key=lambda r:int(r['rank']))]
        with np.load(p,allow_pickle=False) as z:
            f=z['field']; require(len(ids)==len(set(ids))==n and not (set(ids)&seen) and np.array_equal(ids,z['ids']), 'Parent roster mismatch')
            require(f.shape==(n,20,20,20) and f.dtype==np.uint8 and int(f.max())<=2 and int(z['lag'])==primary['selected_saved_lag'], 'Invalid primary maps')
        seen.update(ids)
    pins=get(ROOT/'results/stock-glass-all-beta-critic/launch.json')['wheel_sha256']; require({p.name:stock.sha(p) for p in a.wheels.glob('*.whl')}==pins,'Pinned offline wheels changed')
    return hashes


def authorization(a):
    import modal
    require(a.submit_one and datetime.now(timezone.utc).date().isoformat()=='2026-11-26','Explicit one-time release required')
    pub=get(RESULT/'paid-release.json')
    require((pub['workspace_reference_usd'],pub['workspace_limit_independently_verified'],pub['project_envelope_usd'],D(str(pub['max_all_in_usd'])),pub['max_h100_sandboxes'],pub['max_concurrent_workers_on_that_gpu'],pub['retries'])==(165,False,25,D('2.22'),1,6,0),'No matching published paid release')
    w=modal.Workspace.from_context(); w.hydrate(); env=modal.Environment.from_context(); env.hydrate(); require(w.name=='konradha' and env.name=='main','Workspace/environment changed')
    led=get(RESULT/'admission.json')['project_ledger']; prior=D(led['prior_gross_usd']); summary=w.billing.summary(); require(summary.start.month==datetime.now(timezone.utc).month and summary.metered_cost_breakdown['Deployed Apps']==prior,'Project ledger requires reconciliation')
    now=datetime.now(timezone.utc); today=now.replace(hour=0,minute=0,second=0,microsecond=0)
    rows=list(w.billing.report(start=summary.start,end=today,resolution='d')) if summary.start<today else []
    if now.hour: rows+=list(w.billing.report(start=today,end=now,resolution='h'))
    charges={name:sum((r.cost for r in rows if r.description==name),D(0)) for name in led['app_charges_usd']}
    require(charges=={k:D(v) for k,v in led['app_charges_usd'].items()} and not any(r.description==STUDY for r in rows), 'Provider ledger or one-shot study changed')
    workspace_room=D('165')-summary.metered_cost
    r=w.billing.rates(); rate=r['gpu_hour_cost_h100']+CPU*r['cpu_hour_cost_sandbox']+GIB*r['mem_gib_hour_cost_sandbox']; egress=r['egress_gib_cost']
    ceiling=min(D('2.22'),workspace_room,D('25')-prior)
    life=min(EXT_S,math.floor((ceiling-egress)*3600/rate)); cost=life*rate/3600+egress
    require(life>=MIN_S and cost<=ceiling, f'Minimum not affordable: {life}s within {ceiling}')
    return dict(utc=stamp(),workspace=w.name,environment=env.name,prior_usd=str(prior),metered_usd=str(summary.metered_cost),workspace_reference_usd='165',reference_independently_verified=False,remaining_workspace_planning_usd=str(workspace_room),
                available_usd=str(ceiling),max_usd=str(cost),hourly_usd=str(rate),egress_per_GiB_usd=str(egress),cpu=CPU,gib=GIB,seconds=life,
                extension_authorized=life>MIN_S,paid_release_sha256=stock.sha(RESULT/'paid-release.json'),
                gate_sha256=stock.sha(RESULT/'gate.json'),worker_sha256=stock.sha(WORKER),dispatch_sha256=stock.sha(Path(__file__)),base=stock.BASE,image=stock.IMAGE)


def update_status(**vals):
    path=RESULT/'status.json'; new=dict(get(path),**vals); tmp=path.with_suffix('.json.part'); tmp.write_text(json.dumps(new,indent=2)+'\n'); tmp.replace(path)


def collect(sb, directory, remote, seconds):
    # Network blocked; SDK output is a fixed small allowlist packed in one transfer, not an unrestricted sync.
    import tarfile
    import io
    from modal.stream_type import StreamType
    pr=sb.exec('/bin/bash','-ec','test -d checkpoints; tar -czf output.tgz --exclude="*.part" checkpoints',workdir=remote,timeout=max(1,min(20,seconds)),stdout=StreamType.DEVNULL,stderr=StreamType.DEVNULL)
    require(pr.wait()==0,'Cannot pack training outputs'); fs=sb.filesystem; path=remote+'/output.tgz'; size=fs.stat(path).size; require(0<size<=64*1024**2,'Output exceeds 64 MiB cap')
    raw=fs.read_bytes(path); require(len(raw)==size,'Incomplete output'); total=0
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
        for entry in tar:
            if entry.isdir(): continue
            name=Path(entry.name); total+=entry.size
            require(entry.isfile() and len(name.parts)==2 and name.parts[0]=='checkpoints' and name.name not in ('.','..') and total<=256*1024**2,'Unexpected artifact')
            with (directory/name.name).open('xb') as f: f.write(tar.extractfile(entry).read())
    return size


def launch(a):
    import modal
    from modal.stream_type import StreamType
    # The inexpensive first guard runs before packing wheels. Re-check immediately before allocation.
    require(get(RESULT/'gate.json')['primary']['pass'] is True,'CPU gate failed'); require(not (RESULT/'provider.json').exists() and not (RESULT/'reservation.json').exists(), 'One-shot launch already attempted')
    authorization(a)
    try: existing=modal.Sandbox.from_name(STUDY,STUDY,environment_name='main',include_terminated=True)
    except modal.exception.NotFoundError: pass
    else: raise ValueError('Named sandbox already exists: '+existing.object_id)
    inputs=verify_inputs(a)
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td); subprocess.run(['git','archive','--format=tar','-o',str(tmp/'fork.tar'),stock.BASE,'rsmine'],cwd=ROOT,check=True)
        subprocess.run(['bash','-ec','set -o pipefail; tar -C "$1" -cf - . | split -b 384M - "$2/wheels-"','pack',str(a.wheels.resolve()),str(tmp)],check=True)
        r=authorization(a); life=r['seconds']; ck=RESULT/'checkpoints'; ck.mkdir(exist_ok=False); stock.write_json(RESULT/'reservation.json',dict(r,inputs=inputs))
        try:
            app=modal.App.lookup(STUDY,environment_name='main',create_if_missing=True); start=time.monotonic(); end=start+life
            sb=modal.Sandbox.create('/bin/sleep',str(life),name=STUDY,app=app,image=modal.Image.from_id(stock.IMAGE),gpu='H100!',cpu=(CPU,CPU),memory=(GIB*1024,GIB*1024),block_network=True,timeout=life,tags={'study':STUDY})
        except Exception as exc:
            stock.write_json(RESULT/'provider-refusal.json',dict(utc=stamp(),error=repr(exc),sandbox_id=None,retry=False)); update_status(state='failed',reason='Provider did not return an allocation ID. No retry.',gpu_id=None); raise
        remote=stock.REMOTE; fs=sb.filesystem; wave_started=False; error=None; collected=None
        try:
            provider=dict(utc=stamp(),sandbox=sb.object_id,app=app.app_id,portal='https://modal.com/id/'+sb.object_id); stock.write_json(RESULT/'provider.json',provider)
            update_status(state='gpu_running',gpu_id=sb.object_id,portal=provider['portal'],paid_admission='released',gross_reservation_usd=r['max_usd']); print('ACTUAL_PROVIDER_ID',sb.object_id,app.app_id,flush=True)
            fs.make_directory(remote)
            for f in [*sorted(tmp.glob('wheels-*')),tmp/'fork.tar',WORKER,Path(stock.__file__),*(a.inputs/n for n in BANKS)]: fs.copy_from_local(f,remote+'/'+f.name)
            def run(cmd,remaining):
                require(remaining>0,'Hard wall insufficient'); pr=sb.exec('/bin/bash','-ec',cmd,workdir=remote,timeout=int(remaining),stdout=StreamType.PIPE,stderr=StreamType.STDOUT)
                tail=[]
                for line in pr.stdout:
                    tail=(tail+[line[-2000:]])[-6:]
                    if line.startswith(('FIRST_CUDA','CHECKPOINT','DEPTH_DECISION')): print(line.rstrip(),flush=True)
                if pr.wait()!=0: raise RuntimeError('Remote execution failed: '+''.join(tail))
            run('set -o pipefail; mkdir wheels; cat wheels-* | tar -xf - -C wheels; rm wheels-*; tar -xf fork.tar; /opt/conda/bin/python -m venv env; env/bin/pip install --no-index --find-links=wheels --disable-pip-version-check -q '+' '.join(repr(x) for x in stock.REQUIREMENTS),min(480,end-time.monotonic()-80))
            env='export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 TF_USE_LEGACY_KERAS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 WANDB_MODE=disabled; export LD_LIBRARY_PATH="$(find "$PWD/env" -path "*/nvidia/*/lib" -type d | paste -sd: -):${LD_LIBRARY_PATH:-}"; '
            remaining=math.floor(end-time.monotonic()-35); require(remaining>=324+45,'Insufficient actual wall after setup for minimum; do not substitute shorter fits')
            wave_started=True; run(env+f'env/bin/python -u stock_glass_history_gpu.py wave --seconds {remaining}'+(' --extension' if r['extension_authorized'] else ''),remaining+8)
        except Exception as exc: error=repr(exc)
        finally:
            try:
                if wave_started: collected=collect(sb,ck,remote,int(end-time.monotonic()))
            except Exception as exc: error=(error or '')+' collection: '+repr(exc)
            finally:
                code=sb.terminate(wait=True); stock.write_json(RESULT/'terminated.json',dict(sandbox=sb.object_id,termination_utc=stamp(),returncode=code,elapsed_local_s=time.monotonic()-start,download_bytes=collected))
        complete=(error is None and (ck/'wave.json').is_file() and get(ck/'wave.json')['complete']); updates=sum(get(p)['adam_iterations'] for p in ck.glob('*-final.json'))
        if not complete:
            stock.write_json(RESULT/'gpu-failure.json',dict(error=error or 'incomplete wave',app=app.app_id,retained_final_updates=updates)); update_status(state='failed',gpu_id=sb.object_id,optimizer_updates=updates,reason=error or 'incomplete wave'); raise RuntimeError(error or 'incomplete wave')
        decision=get(ck/'depth-decision.json'); require(updates==6*decision['selected_depth'],'Incomplete matched wave'); stock.write_json(RESULT/'gpu-completion.json',dict(selected_depth=decision['selected_depth'],actual_updates=updates,retained_checkpoint_files=len(list(ck.glob('*-u*.npz'))),underconverged_due_to_cap=decision['underconverged_due_to_cap'],awaiting='post-freeze readout and settled billing'))


def billing():
    """Read only, after provider termination and a completed billing interval. No launch."""
    import modal
    provider=get(RESULT/'provider.json'); t=get(RESULT/'terminated.json'); start=datetime.fromisoformat(provider['utc']); end=datetime.fromisoformat(t['termination_utc']); now=datetime.now(timezone.utc)
    require(end.replace(minute=0,second=0,microsecond=0)<now.replace(minute=0,second=0,microsecond=0),'Wait for completed provider interval'); rows=modal.Workspace.from_context().billing.report(start=start.replace(minute=0,second=0,microsecond=0),end=now,resolution='d' if (now-start).days>=6 else 'h'); project=[r for r in rows if r.object_id==provider['app']]; require(project,'No settled attributable rows yet'); by={}
    for row in project:
        for k,v in row.cost_by_resource.items(): by[k]=by.get(k,D(0))+v
    amount=sum((r.cost for r in project),D(0)); require(amount<=D(get(RESULT/'reservation.json')['max_usd']), 'Provider charge exceeds reserved cap; investigate')
    stock.write_json(RESULT/'billing.json',dict(provider_app=provider['app'],sandbox=provider['sandbox'],checked_utc=stamp(),gross_usd=str(amount),resources_usd={k:str(v) for k,v in by.items()},prior_usd='7.00677575',cumulative_usd=str(D('7.00677575')+amount),excluded='Workspace-level network/storage adjustments not attributed to app; reservation includes bounded egress.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter); sub=ap.add_subparsers(dest='cmd',required=True); l=sub.add_parser('launch')
    for key in ('inputs','wheels'): l.add_argument('--'+key,type=Path,required=True)
    l.add_argument('--submit-one',action='store_true',required=True)
    sub.add_parser('billing'); args=ap.parse_args()
    if args.cmd=='launch': launch(args)
    else: billing()
