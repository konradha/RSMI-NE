"""HERO v3 single-ticket main/refit sender. Only Main may run a numbered release.

--pins REV and --controls are local, no provider calls. No rerun after uncertainty.
--meter reads existing object status/attributable closed-hour billing only.
"""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
from decimal import Decimal
import fcntl
import hashlib
import hmac
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time
import zipfile

import modal
from modal.exception import SandboxFilesystemNotFoundError
from modal.stream_type import StreamType
from modal.types import FileType
from full_hero_sandbox import APP_NAME,IMAGE_ID,RATE_KEYS
from full_hero_sandbox_run import CUTOFF,HERE,LEDGER,PUB,PYTHON,RUN,WORK,digest,put,read_regular,route_identity,sync_dir,utc
from full_hero_sandbox_job import IN_LIMIT,OUT_LIMIT,MAX_RAW,MAX_CHUNKS,MIB,SOURCES,Exporter,file_sha,shard_rows,valid_output

STAGES={'main':(4800,'all','6.15','1.05'),'refit':(3300,'refit','4.27','7.20')}
REMOTE='/tmp/hero-v3-run'
PUBLIC=('design.json','split.json','training-release-1.json','stream-release-1.json','stream-summary.json',
        'offline-transport.json','sandbox-probe.json','sandbox-admission.json','sandbox-release-1.json','h100-main-allocation.json')
GUARDS=('full_hero_sandbox.py','full_hero_sandbox_run.py','full_hero_sandbox_main.py','full_hero_sandbox_job.py')
TRANSPORT_PIN='b5943fac475d8d11321bf45f12566674a74ebcf64b6478fe6666a3e8955f44b0'
MANIFEST_PIN='0c5cc6f84560821a468dc4f92c2b3047fc2dfa55f77ff498243b177b0bf64428'
GATE='''import hashlib,hmac,json,os,sys,time
os.umask(0o077); sys.dont_write_bytecode=True
proof,until,cfg,hashes=sys.argv[1:]; until=int(until)
if time.time()>=until: sys.exit(40)
nonce=sys.stdin.buffer.read(33)
if len(nonce)!=32 or not hmac.compare_digest(hashlib.sha256(nonce).hexdigest(),proof): sys.exit(41)
del nonce
with open('one-time-claim','xb'): pass
for name,want in json.loads(hashes).items():
    with open(name,'rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=want: sys.exit(42)
if time.time()>=until: sys.exit(43)
sys.path.insert(0,os.getcwd())
from full_hero_sandbox_job import run
sys.exit(run(json.loads(cfg),until))
'''


def stage_dir(stage): return WORK/f'h100-{stage}-1'


def small(path): return json.loads(read_regular(path,2*MIB))


def inputs(stage,revision):
    if not re.fullmatch(r'[a-f0-9]{40}',revision): raise ValueError('full source revision required')
    public={n:read_regular(PUB/n,2*MIB) for n in (*PUBLIC,f'sandbox-{stage}-plan.json')}
    guards={n:read_regular(HERE/n,512*1024) for n in GUARDS}
    payload={n:public[n] for n in ('design.json','split.json','training-release-1.json','offline-transport.json')}
    payload['control-gate.json']=read_regular(PUB.parent/'hero-full-v2/control-gate.json',2*MIB)
    if stage=='refit':
        for n in ('nomination.json','refit-release.json'): payload[n]=read_regular(PUB/n,2*MIB)
    for n in SOURCES:
        blob=subprocess.check_output(['git','show',f'{revision}:examples/glass_autoresearch/{n}'],cwd=HERE,timeout=5)
        if not 0<len(blob)<=512*1024: raise ValueError('invalid pinned source size')
        payload[n]=blob
    payload['full_hero_sandbox_job.py']=guards['full_hero_sandbox_job.py']
    pins={k:digest(d) for k,d in {**public,**payload,**guards}.items()}
    train=json.loads(payload['training-release-1.json']); md=json.loads(payload['offline-transport.json']); plan=json.loads(public[f'sandbox-{stage}-plan.json'])
    if not (train['schema']=='hero-full-v3-training-release-1' and train['indices']==list(range(10)) and train['allows_test'] is False
        and train['split_sha256']==pins['split.json'] and train['design_sha256']==pins['design.json'] and train['control_gate_sha256']==pins['control-gate.json']
        and pins['offline-transport.json']==TRANSPORT_PIN and md['release_sha256']==pins['stream-release-1.json']
        and md['source_summary_sha256']==pins['stream-summary.json'] and md['input_manifest_sha256']==MANIFEST_PIN and md['shards']==320
        and md['test_species_rows_returned']==0 and md['stored_bytes']==432842329 and md['restored_npy_bytes']==650448640
        and plan['allocation_sha256']==pins['h100-main-allocation.json'] and plan['panel']==STAGES[stage][1]
        and plan['max_depth'] in (8192,16384,32768,65536) and plan['matched_controls'] is False):
        raise PermissionError('train-only transport/cost allocation mismatch')
    if stage=='main' and plan['max_depth']!=65536: raise PermissionError('wrong main depth cap')
    if stage=='refit':
        nom=json.loads(payload['nomination.json']); rel=json.loads(payload['refit-release.json'])
        if not (rel['schema']=='hero-full-v3-refit-release-1' and rel['all_train'] is True and rel['allows_test'] is False
            and rel['nomination_sha256']==pins['nomination.json'] and rel['split_sha256']==nom['split_sha256']==pins['split.json']
            and rel['rosters_sha256']==train['rosters_sha256']==nom['rosters_sha256'] and rel['depth']==nom['depth']==plan['max_depth']
            and nom['status']=='qualified' and digest(read_regular(stage_dir('main')/'result/nomination.json',2*MIB))==pins['nomination.json']):
            raise PermissionError('registered all-TRAIN nomination freeze missing')
    return pins,payload,md,plan


def author(stage,revision):
    life,panel,cap,prior=STAGES[stage]
    return {'schema':'hero-full-v3-h100-single-stage-release-1','issuer':'Main','action':f'launch-one-train-only-{stage}',
            'source_revision':revision,'maximum_sandboxes':1,'exclusive_envelope_lock_usd':25,'normal_total_limit_usd':15,
            'prior_gross_carry_usd':prior,'stage_max_gross_usd':cap,'lifetime_s':life,'panel':panel,'pins':inputs(stage,revision)[0]}


def frozen_release(stage,expected):
    doc=read_regular(PUB/f'sandbox-{stage}-release.json',65536)
    if not hmac.compare_digest(digest(doc),expected): raise PermissionError('wrong explicit Main release digest')
    r=json.loads(doc); locked=author(stage,r['source_revision'])
    if any(r.get(k)!=v for k,v in locked.items()): raise PermissionError('signed fields/guards differ')
    expires=datetime.fromisoformat(r['expires_utc'].replace('Z','+00:00'))
    if expires.utcoffset()!=timedelta(0) or not time.time()+21*60+STAGES[stage][0]+30<expires.timestamp()<=CUTOFF:
        raise PermissionError('release or discovery cutoff lacks capacity wait + object lifespan + teardown')
    return r,expires.timestamp()


def bundle(path,md):
    root=WORK/'offline-transport/shards'; rows=shard_rows(md)
    if set(p.name for p in root.iterdir())!=set(rows): raise ValueError('shard directory roster mismatch')
    with zipfile.ZipFile(path,'x',zipfile.ZIP_STORED) as z:
        for name,r in rows.items():
            data=read_regular(root/name,8*MIB)
            if len(data)!=r['bytes'] or digest(data)!=r['sha256']: raise ValueError('local input shard digest mismatch')
            z.writestr(zipfile.ZipInfo(name),data)
    return file_sha(path)


def prior(stage):
    expected={'pilot-1'} if stage=='main' else {'pilot-1','main-1'}
    if {p.name for p in LEDGER.iterdir()}!=expected: raise PermissionError('other ticket/missing predecessor; uncertainty hold')
    pilot=small(LEDGER/'pilot-1/reservation.json'); alloc=small(PUB/'h100-main-allocation.json')
    if pilot['pilot_conservative_model_ceiling_usd']!='1.05' or pilot['release_sha256']!=digest(read_regular(PUB/'sandbox-release-1.json',65536)):
        raise PermissionError('unrecognized pilot carry')
    app=modal.App.lookup(APP_NAME,environment_name='main',create_if_missing=False)
    for cell in sorted(expected):
        path=RUN if cell=='pilot-1' else stage_dir('main')
        ident=small(path/'provider-id.json'); terminal=small(path/'termination.json'); sid=ident['sandbox_id']
        if ident['app_id']!=app.app_id or terminal['sandbox_id']!=sid or type(terminal['terminated_exit_code']) is not int:
            raise PermissionError('wrong or missing terminal ID')
        if cell=='pilot-1' and (alloc['pilot']['sandbox_id']!=sid or not (RUN/'transport.json').is_file()): raise PermissionError('wrong pilot identity')
        if cell=='main-1' and (not small(path/'status.json')['complete'] or small(path/'verified.json')['sandbox_id']!=sid
                               or small(LEDGER/cell/'reservation.json')['stage_max_gross_usd']!='6.15'):
            raise PermissionError('main partial or unattributed; no refit approval')
        sb=modal.Sandbox.from_id(sid)
        if sb.get_tags()!={'study':'hero-full-v3','cell':cell} or sb.poll() is None: raise PermissionError('prior Sandbox not terminal/unknown')
    return app


class Receiver:
    def __init__(self,path):
        self.path=path;self.count=self.raw=self.wire=self.files=0;self.pairs=0
        (path/'result').mkdir()

    def extract(self,data):
        if len(data)+self.wire>OUT_LIMIT-8192 or self.count>=MAX_CHUNKS: raise ValueError('network result bound')
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos=z.infolist(); n=sum(i.file_size for i in infos); names=z.namelist()
            if not infos or len(set(names))!=len(infos) or self.raw+n>MAX_RAW or self.files+len(infos)>2000:
                raise ValueError('result aggregate bound')
            for i in infos:
                if not valid_output(i.filename) or i.compress_type!=zipfile.ZIP_DEFLATED or not 0<i.file_size<=2*MIB:
                    raise ValueError('wrong output file')
            for i in infos:
                p=self.path/'result'/i.filename; p.parent.mkdir(exist_ok=True)
                with p.open('xb') as f: f.write(z.read(i));f.flush();os.fsync(f.fileno())
            self.pairs+=sum(s.startswith('learning/') for s in names)
        self.raw+=n;self.wire+=len(data); self.files+=len(infos); self.count+=1

    def pull(self,fs):
        while self.count<MAX_CHUNKS:
            remote=f'{REMOTE}/export/chunk-{self.count:03d}.zip'
            try: info=fs.stat(remote)
            except SandboxFilesystemNotFoundError: return
            if info.type!=FileType.FILE or not 0<info.size<=min(12*MIB,OUT_LIMIT-8192-self.wire): raise ValueError('remote delta bound')
            data=fs.read_bytes(remote)
            if len(data)!=info.size: raise ValueError('immutable fragment size mismatch')
            self.extract(data)
            print(f'CHECKPOINTS {utc()} committed_curves={self.pairs} fragments={self.count} result_wire_bytes={self.wire}',flush=True)


def launch(stage,expected):
    if version('modal')!='1.6.4': raise PermissionError('unexpected Modal client')
    os.umask(0o077); life,panel,cap,carry=STAGES[stage];r,expires=frozen_release(stage,expected)
    pins,payload,md,plan=inputs(stage,r['source_revision'])
    if pins!=r['pins']: raise PermissionError('inputs changed since release')
    probe=small(PUB/'sandbox-probe.json')
    if asyncio.run(route_identity())!=probe['workspace_sha256']: raise PermissionError('other workspace or image')
    rates=modal.Workspace.from_context().billing.rates()
    if any(rates[k]!=Decimal(probe['authenticated_tariff_usd'][k]) for k in RATE_KEYS): raise PermissionError('tariff changed')
    path=stage_dir(stage);cell=stage+'-1';ticket=LEDGER/cell
    with tempfile.TemporaryDirectory(prefix='stage-',dir=WORK) as tmp:
        packed=Path(tmp)/'shards.zip'; whole=bundle(packed,md); nbytes=packed.stat().st_size+sum(map(len,payload.values()))
        if nbytes>IN_LIMIT: raise ValueError('input cap including headers/source')
        with DirectoryLock():
            app=prior(stage)
            if time.time()+21*60+life+30>=expires: raise PermissionError('insufficient capacity and teardown clock')
            ticket.mkdir();sync_dir(LEDGER);path.mkdir();sync_dir(WORK)
            put(ticket/'reservation.json',{'at_utc':utc(),'release_sha256':expected,'stage_max_gross_usd':cap,'prior_gross_carry_usd':carry,
                 'gross_envelope_locked_usd':25,'normal_limit_usd':15,'charged_or_invoiced_usd':None,'no_reissue':True})
            put(path/'create-attempt.json',{'at_utc':utc(),'allow_resubmit':False,'requested_gpu':'H100!','app_id':app.app_id,'name':'v3-h100-'+cell})
            sb=None;start=time.monotonic();wall=time.time();rx=Receiver(path)
            try:
                sb=modal.Sandbox.create('/bin/sleep',str(life-1),app=app,name='v3-h100-'+cell,tags={'study':'hero-full-v3','cell':cell},
                        image=modal.Image.from_id(IMAGE_ID),gpu='H100!',cpu=(2,2),memory=(12288,12288),runtime='gvisor',timeout=life,
                        block_network=True,volumes={},secrets=[],include_oidc_identity_token=False)
                scheduled=time.monotonic();put(path/'provider-id.json',{'at_utc':utc(),'app_id':app.app_id,'sandbox_id':sb.object_id,'image_id':IMAGE_ID})
                print(f'PROVIDER_ID {utc()} stage={stage} app={app.app_id} sandbox={sb.object_id} cap_usd={cap} prior_carry_usd={carry}',flush=True)
                fs=sb.filesystem;fs.make_directory(REMOTE);fs.copy_from_local(packed,REMOTE+'/shards.zip')
                for n,data in payload.items(): fs.write_bytes(data,REMOTE+'/'+n)
                uploaded=time.monotonic(); deadline=int(min(wall+life-75,expires-75,CUTOFF-75)); remaining=int(deadline-time.time())
                if remaining<60: raise TimeoutError('too little protected runtime; never resubmit')
                nonce=secrets.token_bytes(32); hs={n:pins[n] for n in payload};hs['shards.zip']=whole
                cfg={'pins':{n:pins[n] for n in payload},'panel':panel,'max_depth':plan['max_depth']}
                put(path/'exec-attempt.json',{'at_utc':utc(),'sandbox_id':sb.object_id,'attempts_authorized':1,'input_bytes':nbytes,
                    'packed_sha256':whole,'child_deadline_unix':deadline,'sdk_exec_timeout_s':remaining+45,
                    'create_to_handle_s':scheduled-start,'upload_s':uploaded-scheduled})
                env={k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_CACHE_DISABLE')}
                p=sb.exec(PYTHON,'-I','-c',GATE,digest(nonce),str(deadline),json.dumps(cfg),json.dumps(hs),workdir=REMOTE,env=env,
                          stdout=StreamType.DEVNULL,stderr=StreamType.DEVNULL,text=False,timeout=remaining+45)
                p.stdin.write(nonce);p.stdin.write_eof();p.stdin.drain();del nonce
                print(f'EXEC_ARMED {utc()} stage={stage} sandbox={sb.object_id} bytes={nbytes} upload_s={uploaded-scheduled:.3f} child_deadline_unix={deadline}',flush=True)
                while True:
                    rc=p.poll();rx.pull(fs)
                    if rc is not None: break
                    if time.time()>deadline+40: raise TimeoutError('local collection timeout; preserve all complete deltas')
                    time.sleep(8)
                info=fs.stat(REMOTE+'/export/status.json')
                if info.type!=FileType.FILE or not 0<info.size<=8192: raise ValueError('diagnostic cap')
                status=json.loads(fs.read_bytes(REMOTE+'/export/status.json'))
                if (status['schema']!='hero-full-v3-gpu-export-1' or [status[k] for k in ('chunks','files','raw_bytes','wire_bytes')]!=
                    [rx.count,rx.files,rx.raw,rx.wire]): raise ValueError('incomplete remote output transfer')
                put(path/'status.json',dict(status,at_utc=utc(),provider_exec_rc=rc,sandbox_id=sb.object_id,committed_curves=rx.pairs))
                if rc!=0 or not status['complete']: raise RuntimeError(f'partial stage; preserve, no retry: {status}')
                summary=small(path/'result'/('completion.json' if stage=='main' else 'refit.json')); dev=summary['device']; gpu=dev['gpu']
                if (dev['effective_device']!='cuda' or gpu['visible_device_count']!=1 or 'H100' not in gpu['name'] or gpu['capability']!=[9,0]
                    or not 78*1024**3<=gpu['total_memory_bytes']<=81*1024**3 or summary['allows_test'] is not False or
                    summary['release_sha256']!=pins['training-release-1.json'] or summary['split_sha256']!=pins['split.json']):
                    raise ValueError('result not a verified single H100 TRAIN stage')
                put(path/'verified.json',{'at_utc':utc(),'sandbox_id':sb.object_id,'release_sha256':expected,'source':'archived-physical-train-only',
                    'input_bytes':nbytes,'wire_result_bytes':rx.wire+info.size,'committed_curves':rx.pairs,'device':gpu,'elapsed_wall_s':time.monotonic()-start})
                print(f'CUDA_RESULT {utc()} sandbox={sb.object_id} stage={stage} curves={rx.pairs} gpu={gpu["name"]} elapsed_s={time.monotonic()-start:.3f}',flush=True)
            except BaseException as exc:
                put(path/'failure.json',{'at_utc':utc(),'class':type(exc).__name__,'preserved_curves':rx.pairs,'resubmit':False});raise
            finally:
                if sb is not None:
                    rc=sb.terminate(wait=True)
                    put(path/'termination.json',{'at_utc':utc(),'sandbox_id':sb.object_id,'terminated_exit_code':rc,'whole_wall_s':time.monotonic()-start})


class DirectoryLock:
    def __enter__(self):
        self.fd=os.open(LEDGER,os.O_DIRECTORY|os.O_RDONLY)
        try: fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException: os.close(self.fd);raise
    def __exit__(self,*_): os.close(self.fd)


def meter(stage):
    path=stage_dir(stage); ident=small(path/'provider-id.json');sid=ident['sandbox_id'];sb=modal.Sandbox.from_id(sid)
    if sb.get_tags()!={'study':'hero-full-v3','cell':stage+'-1'}: raise ValueError('wrong prior object')
    start=datetime.fromisoformat(ident['at_utc']).replace(minute=0,second=0,microsecond=0)
    end=datetime.now(timezone.utc).replace(minute=0,second=0,microsecond=0);rows=[]
    if end>start: rows=[asdict(r) for r in modal.Workspace.from_context().billing.report(start=start,end=end,resolution='h') if r.object_id in (sid,ident['app_id'])]
    return {'source':'authenticated provider; no preemption inference or invoice assumption','utc':utc(),'sandbox_id':sid,
            'app_id':ident['app_id'],'terminal_code':sb.poll(),'hourly_app_and_object_rows_do_not_sum_twice':rows,'start':start,'end':end,'carry_release':False}


def controls():
    with tempfile.TemporaryDirectory() as name:
        root=Path(name);p=root/'sweep';f=root/'frags';d=root/'download';f.mkdir();d.mkdir();(p/'learning').mkdir(parents=True);(p/'checkpoints').mkdir()
        stamp='affine-uniform-d1-s6101-u1024';w=p/'checkpoints'/(stamp+'.npz');w.write_bytes(b'synthetic-checkpoint-only');ref=p/'learning'/(stamp+'.json')
        x=Exporter(p,f);x.collect();assert x.count==0
        ref.write_text(json.dumps({'checkpoint':w.name,'checkpoint_sha256':file_sha(w)}));x.collect();x.collect(); rx=Receiver(d);rx.extract((f/'chunk-000.zip').read_bytes())
        assert x.count==rx.count==1 and rx.pairs==1 and file_sha(d/'result/checkpoints'/w.name)==file_sha(w)
        for wrong in ('../test.npy','raw/train.npy','optimizer/model.npz','learning/../../other.json'):
            assert not valid_output(wrong)
        gate=root/'gate';gate.mkdir();nonce=secrets.token_bytes(32);j=gate/'full_hero_sandbox_job.py'
        j.write_text('def run(cfg,until):\n    with open("starts","a") as f: f.write("one\\n")\n    return 0\n')
        cmd=[sys.executable,'-I','-c',GATE,digest(nonce),str(int(time.time()+30)),'{}',json.dumps({j.name:file_sha(j)})]
        codes=[subprocess.run(cmd,input=k,cwd=gate,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5).returncode for k in (b'',nonce,nonce)]
        assert codes[:2]==[41,0] and codes[2]!=0 and (gate/'starts').read_text()=='one\n'
        return {'source':'synthetic local only','utc':utc(),'paid_calls':0,'physical_parents':0,'wire_bytes':rx.wire,'checkpoint_pairs':rx.pairs,'transport_gate_exit_codes':codes,
                'main_gross_ceiling_usd':'6.15','refit_gross_ceiling_usd':'4.27','cumulative_with_pilot_usd':'11.47'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=STAGES,default='main')
    g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--pins',metavar='SOURCE_REVISION');g.add_argument('--release-sha256');g.add_argument('--meter',action='store_true');g.add_argument('--controls',action='store_true')
    a=parser.parse_args()
    if a.release_sha256: launch(a.stage,a.release_sha256)
    else: print(json.dumps(author(a.stage,a.pins) if a.pins else meter(a.stage) if a.meter else controls(),indent=2,default=str))
