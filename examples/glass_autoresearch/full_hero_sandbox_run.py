"""HERO v3 single Modal H100 pilot. --pins REV is local metadata only.

Only Main may issue the numbered release and run --release-sha256. Never retry
an indeterminate create or exec. --reconcile reads billing; it cannot rearm.
"""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import hmac
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time

import modal
from google.protobuf.empty_pb2 import Empty
from modal.client import _Client
from modal.exception import NotFoundError
from modal.stream_type import StreamType
from modal.types import FileType
from modal_proto import api_pb2 as api

from full_hero_sandbox import APP_NAME, IMAGE_ID, RATE_KEYS, REQUIRED_PACKAGES

HERE = Path(__file__).resolve().parent
PUB = HERE.parents[1] / 'results/l20-autoresearch/hero-full-v3'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/hero-full-v3'
LEDGER = Path.home() / '.local/state/rsmi/hero-spatial-incremental-25-1795520725'
RUN = WORK / 'h100-pilot-1'
SB_NAME = 'v3-h100-pilot-1'
RELEASE = PUB / 'sandbox-release-1.json'
REMOTE = '/tmp/hero-v3/payload'
PYTHON = '/opt/conda/bin/python'
LIFETIME = 300
EXEC_LIMIT = 180
CUTOFF = datetime(2026, 11, 24, 21, 0, 25, tzinfo=timezone.utc).timestamp()
MIB = 1024**2
OUTPUT_LIMIT = 128*1024
SOURCE_NAMES = ('full_hero_v3_pilot.py','full_hero_train.py','hero_train.py')
INPUT_CAPS = {'packet.npz': 64*MIB, 'packet.json': 65536, **dict.fromkeys(SOURCE_NAMES,262144)}
PUBLIC_FILES = ('design.json','split.json','stream-release-1.json','pilot-packet.json',
                'training-release-1.json','sandbox-admission.json','sandbox-probe.json')

# No token in argv, environment, image, file or logs. One explicit exec tied
# to one Modal task/process ID receives it via stdin; only its digest is in
# argv. A new boot/exec sees neither a fresh token nor a resubmitted input.
GATE = '''import hashlib,hmac,json,os,runpy,signal,sys,time,traceback
os.umask(0o077)
challenge, until, expected, stream_sha, split_sha, packet_sha = sys.argv[1:]
if time.time() >= float(until): sys.exit(40)
signal.alarm(max(1,int(float(until)-time.time())))
nonce=sys.stdin.buffer.read(33)
if len(nonce)!=32 or not hmac.compare_digest(hashlib.sha256(nonce).hexdigest(),challenge): sys.exit(41)
del nonce
with open('claimed','xb'): pass
for name, want in json.loads(expected).items():
    with open(name,'rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=want: sys.exit(42)
if time.time() >= float(until): sys.exit(43)
sys.dont_write_bytecode=True
sys.path.insert(0,os.getcwd())
sys.argv=['full_hero_v3_pilot.py','--packet','packet.npz','--receipt','packet.json','--release-sha256',stream_sha,'--split-sha256',split_sha,'--packet-sha256',packet_sha,'--device','cuda','--out','/tmp/pilot']
try:
    runpy.run_path(sys.argv[0],run_name='__main__')
except Exception as exc:
    # Keep trusted frame locations, not arbitrary exception text, locals,
    # inputs or a traceback with lines copied from the data being processed.
    frames=[(os.path.basename(f.filename),f.lineno) for f in traceback.extract_tb(exc.__traceback__) if os.path.basename(f.filename) in expected]
    cls=type(exc).__name__[:80]
    labels=('out of memory','cuda','h100','support','provenance','shape','gradient','digest','schema','dtype','module')
    detail={'class':cls,'labels':[x for x in labels if x in str(exc).lower()],'frames':frames[-8:]}
    with open('pilot-error.json','x') as f: json.dump(detail,f)
    sys.exit(44)
'''


def digest(data):
    return sha256(data).hexdigest()


def read_regular(path, cap):
    with open(path, 'rb', opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW)) as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= cap:
            raise ValueError(f'wrong file type or size: {path.name}')
        data = f.read(cap + 1)
        if len(data) != info.st_size:
            raise ValueError(f'changed size: {path.name}')
        return data


def put(path, document):
    # Append-only even after process loss. An incomplete write is a hold.
    with path.open('x') as f:
        json.dump(document, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    sync_dir(path.parent)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def source_blob(name, revision):
    if name not in SOURCE_NAMES or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('expected pilot source and full git commit')
    data = subprocess.check_output(['git','show',f'{revision}:examples/glass_autoresearch/{name}'], cwd=HERE, timeout=5)
    if not 0<len(data)<=INPUT_CAPS[name]: raise ValueError('unexpected frozen source size')
    return data


def pins(revision):
    files = {p: PUB/p for p in PUBLIC_FILES}
    files.update({p: HERE/p for p in ('full_hero_sandbox_run.py','full_hero_sandbox.py')})
    files['packet.json'] = WORK/'streams/packet.json'
    result = {key: digest(read_regular(path, 2*MIB)) for key,path in files.items()}
    result.update({k:digest(source_blob(k,revision)) for k in SOURCE_NAMES})
    manifest=json.loads((PUB/'pilot-packet.json').read_bytes())
    result['packet.npz']=manifest['packet_sha256']
    partial={str(i):digest(read_regular(WORK/'streams'/f'b{i:02d}'/'pilot-subset.json',65536)) for i in range(10)}
    if result['pilot-packet.json']!=result['packet.json'] or manifest['partial_receipts_sha256']!=partial:
        raise PermissionError('public/private manifests or per-beta pilot receipts differ')
    split=json.loads((PUB/'split.json').read_bytes())
    train=json.loads((PUB/'training-release-1.json').read_bytes())
    stream=json.loads((PUB/'stream-release-1.json').read_bytes())
    if not (split['schema']=='hero-full-v3-exact-half-split-1' and split['cross_side_intersections']==0
            and split['global_unique_parents']==163840 and set(split['per_beta'])=={str(i) for i in range(10)}
            and all(row['train']==row['test']==8192 for row in split['per_beta'].values())
            and train['schema']=='hero-full-v3-training-release-1' and train['stage']=='train-only'
            and train['allows_test'] is False and train['indices']==list(range(10)) and train['candidate_folds']==list(range(2,8))
            and train['split_sha256']==stream['split_sha256']==result['split.json'] and stream['stage']=='train'
            and train['design_sha256']==stream['design_sha256']==result['design.json']
            and train['rosters_sha256']==stream['rosters_sha256']):
        raise PermissionError('V3 immutable split/TRAIN-only approvals do not agree')
    return result


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


async def route_identity():
    stub = (await _Client.from_env())._stub
    i = await stub.WorkspaceNameLookup(Empty())
    image = await stub.ImageFromId(api.ImageFromIdRequest(image_id=IMAGE_ID))
    pkg = {k: image.metadata.python_packages.get(k) for k in REQUIRED_PACKAGES}
    if image.image_id != IMAGE_ID or pkg != REQUIRED_PACKAGES:
        raise PermissionError('cached image or package metadata changed')
    return digest((i.workspace_name or i.username).encode())


def checked_release(expected):
    doc = read_regular(RELEASE,65536)
    if not hmac.compare_digest(digest(doc),expected):
        raise PermissionError('wrong numbered Main release digest')
    r = json.loads(doc)
    locked = {'schema':'hero-full-v3-h100-pilot-release-1','issuer':'Main','action':'launch-one-train-only-h100-pilot',
              'maximum_sandboxes':1,'incremental_used_or_reserved_before_usd':0,'exclusive_envelope_lock_usd':25,
              'pins':pins(r['source_revision'])}
    if any(r.get(k)!=v for k,v in locked.items()):
        raise PermissionError('release does not match single-shot bound/pins')
    end = datetime.fromisoformat(r['expires_utc'].replace('Z','+00:00'))
    if end.utcoffset()!=timedelta(0) or not time.time()+LIFETIME < end.timestamp() <= CUTOFF:
        raise PermissionError('insufficient UTC release lifetime')
    if time.time()+21*60+LIFETIME+30 > CUTOFF:
        raise PermissionError('insufficient discovery clock for capacity wait and sandbox')
    return r,end.timestamp()


def validate_inputs(r):
    files = {name:source_blob(name,r['source_revision']) if name in SOURCE_NAMES else read_regular(WORK/'streams'/name,cap)
             for name,cap in INPUT_CAPS.items()}
    if any(digest(data)!=r['pins'][name] for name,data in files.items()):
        raise PermissionError('payload hash mismatch')
    receipt = json.loads(files['packet.json'])
    if (receipt['schema']!='hero-full-v3-fit-pilot-1'
            or receipt['source']!='archived-physical' or receipt['role']!='fit' or receipt['fit_only'] is not True
            or receipt['all_ten'] is not True or receipt['beta_indices']!=list(range(10))
            or receipt['train_only'] is not True
            or receipt['split_sha256']!=r['pins']['split.json']
            or receipt['test_species_rows_returned']!=0 or receipt['train_folds']!=list(range(2,8))
            or receipt['parents_per_beta']!={str(i):64 for i in range(10)}
            or receipt['release_sha256']!=r['pins']['stream-release-1.json'] or receipt['raw_bytes']>64*MIB
            or receipt['stored_bytes']!=len(files['packet.npz'])):
        raise PermissionError('FIT release/provenance mismatch')
    return files


def run(expected):
    if version('modal')!='1.6.4': raise PermissionError('unaudited SDK')
    os.umask(0o077)
    r,expires = checked_release(expected)
    probe = json.loads((PUB/'sandbox-probe.json').read_bytes())
    if asyncio.run(route_identity())!=probe['workspace_sha256']:
        raise PermissionError('unexpected authenticated workspace')
    rates = modal.Workspace.from_context().billing.rates()
    if any(rates[k]!=Decimal(probe['authenticated_tariff_usd'][k]) for k in RATE_KEYS):
        raise PermissionError('tariff changed')
    files = validate_inputs(r)
    # O_EXCL-equivalent mkdir is the cross-study, non-expiring USD25 envelope
    # lock. Nobody else may be admitted while this funded/uncertain ticket is
    # open; this launcher has exactly ONE fixed ticket and no renewal path.
    LEDGER.mkdir(parents=True,exist_ok=True)
    if any(LEDGER.iterdir()): raise PermissionError('nonempty shared incremental ledger; no retry or second admission')
    ticket = LEDGER/'pilot-1'; ticket.mkdir(); sync_dir(LEDGER)
    RUN.mkdir(exist_ok=False)
    put(ticket/'reservation.json',{'at_utc':utc(),'release_sha256':expected,'gross_envelope_locked_usd':25,
                                  'pilot_conservative_model_ceiling_usd':'1.05','scope':'all incremental studies',
                                  'charged_or_invoiced_usd':None,'next_admission_allowed':False})
    try:
        existing = modal.App.lookup(APP_NAME, environment_name='main', create_if_missing=False)
    except NotFoundError:
        existing = None
    if existing is not None: raise PermissionError('study App already exists: no reuse or retry')
    app = modal.App.lookup(APP_NAME,environment_name='main',create_if_missing=True)
    put(RUN/'app.json',{'at_utc':utc(),'app_id':app.app_id,'study':APP_NAME})
    t = time.monotonic(); times = {}; started = utc(); sb = None
    try:
        if time.time()+LIFETIME >= expires: raise PermissionError('expired before create')
        put(RUN/'create-attempt.json',{'at_utc':utc(),'name':SB_NAME,'allow_resubmit':False})
        sb = modal.Sandbox.create('/bin/sleep','295',app=app,name=SB_NAME,tags={'study':'hero-full-v3','cell':'pilot-1'},
                    image=modal.Image.from_id(IMAGE_ID),gpu='H100!',cpu=(1,1),memory=(4096,4096),runtime='gvisor',
                    timeout=LIFETIME,block_network=True,volumes={},secrets=[],include_oidc_identity_token=False)
        times['create_to_scheduled_handle_s'] = time.monotonic()-t
        put(RUN/'provider-id.json',{'at_utc':utc(),'app_id':app.app_id,'sandbox_id':sb.object_id,'image_id':IMAGE_ID})
        print(f'PROVIDER_ID {utc()} app={app.app_id} sandbox={sb.object_id} gross_lock_usd=25 pilot_cap_usd=1.05',flush=True)
        now = time.monotonic(); sb.filesystem.make_directory(REMOTE)
        for name,data in files.items(): sb.filesystem.write_bytes(data,REMOTE+'/'+name)
        times['payload_upload_s'] = time.monotonic()-now
        deadline = min(expires,CUTOFF,time.time()+EXEC_LIMIT)
        if deadline-time.time()<5: raise PermissionError('insufficient time to arm timed command')
        nonce = secrets.token_bytes(32); proof = digest(nonce)
        args = [PYTHON,'-I','-c',GATE,proof,str(int(deadline)),json.dumps({k:r['pins'][k] for k in files}),
                r['pins']['stream-release-1.json'],r['pins']['split.json'],r['pins']['packet.npz']]
        env = {k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_CACHE_DISABLE')}
        put(RUN/'exec-attempt.json',{'at_utc':utc(),'sandbox_id':sb.object_id,'attempts_authorized':1,
                                     'payload_bytes':sum(map(len,files.values())),'deadline_unix':int(deadline)})
        now=time.monotonic(); p=sb.exec(*args,workdir=REMOTE,env=env,stdout=StreamType.DEVNULL,stderr=StreamType.DEVNULL,
                                      text=False,timeout=EXEC_LIMIT)
        p.stdin.write(nonce); p.stdin.write_eof(); p.stdin.drain(); del nonce
        rc=p.wait(); times['arm_through_task_exit_s']=time.monotonic()-now
        if rc!=0:
            reason={'rc':rc,'utc':utc(),'sanitized_exception':None}
            try:
                path=REMOTE+'/pilot-error.json'; info=sb.filesystem.stat(path)
                if info.type==FileType.FILE and 0<info.size<=8192:
                    reason['sanitized_exception']=json.loads(sb.filesystem.read_bytes(path))
            except Exception as exc:
                reason['diagnostic_unavailable']=type(exc).__name__
            put(RUN/'failure.json',reason)
            raise RuntimeError(f'one-shot failed without retry: {reason}')
        now=time.monotonic(); sizes={}; output={}
        for name in ('model.npz','summary.json'):
            info=sb.filesystem.stat('/tmp/pilot/'+name)
            if info.type!=FileType.FILE or not 0<info.size<=OUTPUT_LIMIT-sum(sizes.values()):
                raise ValueError('unexpected remote output type/aggregate size')
            sizes[name]=info.size
        for name,n in sizes.items():
            output[name]=sb.filesystem.read_bytes('/tmp/pilot/'+name)
            if len(output[name])!=n: raise ValueError('output changed since stat')
        summary=json.loads(output['summary.json']); dev=summary['device']; gpu=dev['gpu']
        if (summary['packet_sha256']!=r['pins']['packet.npz'] or summary['model_sha256']!=digest(output['model.npz']) or
            summary['split_sha256']!=r['pins']['split.json'] or summary['actual_training_updates']!=256 or
            summary['release_sha256']!=r['pins']['stream-release-1.json'] or summary['source']!='archived-physical-train-only' or
            summary['schema']!='hero-full-v3-h100-pilot-1' or
            dev['effective_device']!='cuda' or 'H100' not in gpu['name'] or gpu['visible_device_count']!=1 or
            not 78*1024**3<=gpu['total_memory_bytes']<=81*1024**3 or gpu['capability']!=[9,0]):
            raise ValueError('result/device/provenance does not match released pilot')
        print(f"CUDA_RESULT {utc()} sandbox={sb.object_id} gpu={gpu['name']} actual_cuda_updates=256 elapsed_since_create_s={time.monotonic()-t:.3f}",flush=True)
        for name,data in output.items():
            with (RUN/name).open('xb') as f: f.write(data)
        times['result_transfer_s']=time.monotonic()-now
        put(RUN/'transport.json',{'source':'archived-physical-train-only','started_utc':started,'finished_fetch_utc':utc(),
             'sandbox_id':sb.object_id,'app_id':app.app_id,'requested_gpu':'H100!','result_bytes':sizes,'device':gpu,
             'payload_bytes':{k:len(v) for k,v in files.items()},'timing':times,'elapsed_to_fetch_s':time.monotonic()-t})
    finally:
        if sb is not None:
            now=time.monotonic(); rc=sb.terminate(wait=True)
            put(RUN/'termination.json',{'at_utc':utc(),'sandbox_id':sb.object_id,'terminated_exit_code':rc,
                                       'terminate_wait_s':time.monotonic()-now,'create_through_termination_wall_s':time.monotonic()-t})


def gate_control():
    # Nonphysical control of the actual wrapper, Python dispatch and EOF. A harmless
    # child records one line; no Modal endpoint or study input is accessed.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'full_hero_v3_pilot.py'
        path.write_text("with open('test-count','a') as f: f.write('one\\n')\n")
        nonce=secrets.token_bytes(32)
        cmd=[sys.executable,'-I','-c',GATE,digest(nonce),str(int(time.time()+30)),
             json.dumps({path.name:digest(path.read_bytes())}),'nonphysical','nonphysical','nonphysical']
        attempts=[subprocess.run(cmd, input=token, cwd=tmp, capture_output=True,timeout=5).returncode
                  for token in (b'',nonce,nonce)]
        if attempts[:2]!=[41,0] or attempts[2]==0 or (Path(tmp)/'test-count').read_text()!='one\n':
            raise RuntimeError('synthetic transport gate failed')
        # Separate fresh synthetic box for the exception path, no rearm of
        # the once-consumed directory. No free-form exception text may leave.
        fresh=Path(tmp)/'failure'; fresh.mkdir()
        other=fresh/path.name; other.write_text("raise ValueError('cuda SENTINEL_DO_NOT_EXPORT')\n")
        cmd[6]=json.dumps({other.name:digest(other.read_bytes())})
        failure=subprocess.run(cmd,input=nonce,cwd=fresh,capture_output=True,timeout=5)
        detail=read_regular(fresh/'pilot-error.json',8192)
        if failure.returncode!=44 or b'SENTINEL' in detail or json.loads(detail)['labels']!=['cuda']:
            raise RuntimeError('synthetic sanitized failure receipt failed')
        return {'source':'local synthetic transport only, no CUDA or Modal calls','utc':utc(),
                'missing_token_exit':attempts[0],'valid_token_exit':attempts[1],'replay_exit':attempts[2],
                'exception_control_exit':failure.returncode,'diagnostic_bytes':len(detail),
                'actual_harmless_child_starts':1,'physical_rows':0,'paid_calls':0}


async def metered_gpu(sb_id):
    stub=(await _Client.from_env())._stub
    v=await stub.SandboxGetResourceUsage(api.SandboxGetResourceUsageRequest(sandbox_id=sb_id))
    return {'gpu_type':v.gpu_type,'gpu_seconds':v.gpu_nanosecs/1e9,'cpu_core_seconds':v.cpu_core_nanosecs/1e9,
            'memory_gib_seconds':v.mem_gib_nanosecs/1e9}


def reconcile():
    # Can be called later, with no training release, including after local
    # failure. Never releases the reservation. Assessment of late/partial
    # metering across *all* relevant resources belongs to Main.
    app=json.loads((RUN/'app.json').read_bytes())
    if app['study']!=APP_NAME: raise ValueError('unrecognized study app')
    if (RUN/'provider-id.json').exists():
        identity=json.loads((RUN/'provider-id.json').read_bytes())
        if identity['app_id']!=app['app_id']: raise ValueError('unrecognized study job')
        sb=modal.Sandbox.from_id(identity['sandbox_id'])
    else:
        # Uncertain create: read-only lookup of the named one-shot. Not found
        # does not rearm the ticket. Never create or submit from reconciliation.
        sb=modal.Sandbox.from_name(APP_NAME,SB_NAME,environment_name='main',include_terminated=True)
    if sb.get_tags()!={'study':'hero-full-v3','cell':'pilot-1'}: raise ValueError('unexpected Sandbox tags')
    out={'source':'authenticated provider accounting, no physical/synthetic measurement','utc':utc(),
         'app_id':app['app_id'],'sandbox_id':sb.object_id,'poll_exit_code':sb.poll(),
         'possible_delayed_or_noncompute_rows':True,'reservation_released':False}
    for key, fetch in [('sandbox_stats',lambda:asdict(sb._experimental_stats())),
                       ('provider_resource_seconds',lambda:asyncio.run(metered_gpu(sb.object_id)))]:
        try: out[key]=fetch()
        except Exception as exc: out[key]=None; out[key+'_error']=type(exc).__name__
    start=datetime.fromisoformat(app['at_utc']).replace(minute=0,second=0,microsecond=0)
    end=datetime.now(timezone.utc).replace(minute=0,second=0,microsecond=0)
    rows=[]
    if end>start:
        for row in modal.Workspace.from_context().billing.report(start=start,end=end,resolution='h'):
            if row.object_id in (sb.object_id,app['app_id']): rows.append(asdict(row))
    out.update(billing_report_full_hours_since=start,billing_report_full_hours_until=end,
               attributable_gross_hourly_rows=rows,missing_row_is_not_zero=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pins',metavar='SOURCE_COMMIT');group.add_argument('--release-sha256');group.add_argument('--reconcile',action='store_true')
    group.add_argument('--gate-control',action='store_true')
    cmd=parser.parse_args()
    if cmd.pins: print(json.dumps(pins(cmd.pins),indent=2))
    elif cmd.gate_control: print(json.dumps(gate_control(),indent=2))
    elif cmd.reconcile: print(json.dumps(reconcile(),indent=2,default=str))
    else: run(cmd.release_sha256)
