"""HERO v3 offline H100: one stdin-gated child; immutable delta files, no network."""
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import sys
import time
import traceback
import zipfile

MIB = 1024**2
IN_LIMIT = 2048*MIB
OUT_LIMIT = 64*MIB
MAX_RAW = 256*MIB
CHUNK_RAW = 6*MIB
MAX_CHUNKS = 512
CONTROL = {'device-numeric.json','support-d1.json','support-d4.json','support-d16.json',
           'nomination.json','completion.json','refit.json'}
STAMP = r'(?:refit-)?(?:affine|tanh|site|composition|coordination|crystal|random_affine|random_tanh)-(?:uniform|matched)-(?:d(?:1|4|16)-)?s610[1-5]-u(?:1024|2048|4096|8192|16384|32768|65536)'
SOURCES = ('full_hero_v3_sweep.py','full_hero_v3_selection.py','full_hero_v3_math.py','full_hero_v3_transport.py',
           'full_hero_transport.py','full_hero_train.py','hero_train.py','hero_rule.py')


def valid_output(name):
    return name in CONTROL or re.fullmatch('learning/'+STAMP+r'\.json|checkpoints/'+STAMP+r'\.npz',name) is not None


def file_sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def atomic(path,obj):
    tmp=path.with_suffix('.tmp')
    with tmp.open('x') as f:
        json.dump(obj,f,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.link(tmp,path); tmp.unlink()


def sanitized(exc):
    frames=traceback.extract_tb(exc.__traceback__)
    labels=('out of memory','cuda','h100','support','provenance','shape','gradient','digest','schema','dtype','module','mismatch','bound','timeout','denominator','rank','continuation')
    return {'class':type(exc).__name__[:80], 'labels':[x for x in labels if x in str(exc).lower()],
            'frames':[(Path(x.filename).name,x.lineno) for x in frames if Path(x.filename).name in (*SOURCES,Path(__file__).name)][-12:]}


def shard_rows(md):
    entries={}
    for b in range(10):
        rows=md['per_beta'][str(b)]['shards']
        if len(rows)!=32: raise ValueError('shard count')
        for j,r in enumerate(rows):
            name=f'b{b:02d}-train-{j:03d}.z'
            if r['name']!=name or r['start']!=j*256 or r['parents']!=256 or not 0<r['bytes']<=8*MIB:
                raise ValueError('shard shape')
            entries[name]=r
    if sum(r['bytes'] for r in entries.values())!=md['stored_bytes'] or md['stored_bytes']>=IN_LIMIT:
        raise ValueError('shard bundle bound')
    return entries


def scientific(cfg):
    from full_hero_v3_transport import restore
    pins=cfg['pins']; md=json.loads(Path('offline-transport.json').read_bytes())
    rows=shard_rows(md); zpath=Path('shards.zip'); stage=Path('shards'); stage.mkdir()
    with zipfile.ZipFile(zpath) as z:
        if len(z.infolist())!=len(rows) or set(z.namelist())!=set(rows): raise ValueError('shard roster mismatch')
        for i in z.infolist():
            r=rows[i.filename]
            if i.compress_type!=zipfile.ZIP_STORED or i.file_size!=r['bytes']: raise ValueError('shard size mismatch')
            data=z.read(i)
            if hashlib.sha256(data).hexdigest()!=r['sha256']: raise ValueError('shard digest mismatch')
            with (stage/i.filename).open('xb') as f: f.write(data)
    zpath.unlink()
    restored=Path('/tmp/hero-input')
    manifest=restore(Path('offline-transport.json'),pins['offline-transport.json'],stage,restored,drop=True)
    if manifest!=md['input_manifest_sha256']: raise ValueError('input manifest mismatch')
    cmd=['full_hero_v3_sweep.py','--release','training-release-1.json','--release-sha256',pins['training-release-1.json'],
         '--control-gate','control-gate.json','--device','cuda','--offline-root',str(restored),'--manifest-sha256',manifest,
         '--panel',cfg['panel'],'--max-depth',str(cfg['max_depth']),'--out','result']
    if cfg['panel']=='refit': cmd+=['--refit-release','refit-release.json','--refit-release-sha256',pins['refit-release.json'],'--nomination-file','nomination.json']
    sys.argv=cmd; runpy.run_path(cmd[0],run_name='__main__')


class Exporter:
    """Only committed learning receipts plus the weights whose hashes they sign."""
    def __init__(self,root,fragments):
        self.root=root; self.fragments=fragments; self.sent=set(); self.raw=self.wire=self.count=0

    def collect(self):
        pending=[]; raw=0
        for p in sorted(self.root.glob('*.json'))+sorted((self.root/'learning').glob('*.json')):
            name=p.relative_to(self.root).as_posix()
            if name in self.sent or not valid_output(name): continue
            size=p.stat().st_size
            if not 0<size<=2*MIB or p.is_symlink(): raise ValueError('receipt size bound')
            group=[p]
            if name.startswith('learning/'):
                r=json.loads(p.read_bytes()); c=r['checkpoint']
                fname,h=(c['path'],c['sha256']) if isinstance(c,dict) else (c,r['checkpoint_sha256'])
                if fname!=p.stem+'.npz': raise ValueError('checkpoint name mismatch')
                cp=self.root/'checkpoints'/fname
                if cp.is_symlink() or not 0<cp.stat().st_size<=2*MIB or file_sha(cp)!=h: raise ValueError('checkpoint digest mismatch')
                group.append(cp)
            pending.extend(group);raw+=sum(x.stat().st_size for x in group)
            if raw>=CHUNK_RAW:
                self.emit(pending); pending=[]; raw=0
        if pending: self.emit(pending)

    def emit(self,files):
        raw=sum(f.stat().st_size for f in files)
        if self.raw+raw>MAX_RAW or self.count>=MAX_CHUNKS: raise ValueError('raw output bound')
        name=f'chunk-{self.count:03d}.zip'; tmp=self.fragments/'current.tmp'
        with zipfile.ZipFile(tmp,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for f in files: z.write(f,f.relative_to(self.root).as_posix())
        n=tmp.stat().st_size
        if self.wire+n>OUT_LIMIT-8192: raise ValueError('wire output bound')
        os.link(tmp,self.fragments/name);tmp.unlink()
        self.count+=1; self.raw+=raw;self.wire+=n
        self.sent.update(f.relative_to(self.root).as_posix() for f in files)


def run(cfg,deadline):
    start=time.monotonic(); fragments=Path('export'); fragments.mkdir(exist_ok=False)
    exp=Exporter(Path('result'),fragments); pid=None;rc=None;failure=None
    try:
        pid=os.fork()
        if pid==0:
            try:
                scientific(cfg)
            except BaseException as exc:
                atomic(Path('child-error.json'),sanitized(exc));os._exit(45)
            os._exit(0)
        while True:
            ended,status=os.waitpid(pid,os.WNOHANG)
            if ended:
                pid=None;rc=os.waitstatus_to_exitcode(status);break
            if time.time()>=deadline: raise TimeoutError('training timeout')
            exp.collect();time.sleep(min(15,max(0,deadline-time.time())))
    except BaseException as exc:
        failure=sanitized(exc)
    finally:
        if pid:
            os.kill(pid,signal.SIGKILL);_,status=os.waitpid(pid,0);rc=os.waitstatus_to_exitcode(status)
        try: exp.collect()
        except BaseException as exc: failure=sanitized(exc)
        ep=Path('child-error.json')
        if ep.exists(): failure=json.loads(ep.read_bytes())
        required={'completion.json','nomination.json'} if cfg['panel']=='all' else {'refit.json'}
        ok=rc==0 and failure is None and required<=exp.sent
        atomic(fragments/'status.json',{'schema':'hero-full-v3-gpu-export-1','complete':ok,'child_rc':rc,'failure':failure,
               'chunks':exp.count,'raw_bytes':exp.raw,'wire_bytes':exp.wire,'files':len(exp.sent),'wall_s':time.monotonic()-start})
    return 0 if ok else 46
