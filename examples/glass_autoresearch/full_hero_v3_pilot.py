"""V3 bounded H100 pilot: 640 archived TRAIN parents, every beta, no TEST reader."""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from hero_train import LOG_C, dump_new, indicators
from full_hero_train import MIB, Temperature, Trainer, device_setup, loss_fn, pilot_math, sha


def run(a):
    wall=time.monotonic()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError('Fresh output required')
    device=device_setup(a.device)
    st=time.monotonic();check=pilot_math(a.device);math_sec=time.monotonic()-st
    st=time.monotonic()
    if a.packet.stat().st_size>64*MIB or a.receipt.stat().st_size>65536:
        raise ValueError('Oversize train-only timing payload')
    r=json.loads(a.receipt.read_text())
    if not (r['schema']=='hero-full-v3-fit-pilot-1' and r['role']=='fit'
            and r['source']=='archived-physical' and r['train_only'] is True and r['fit_only'] is True and r['all_ten'] is True
            and r['train_folds']==[2,3,4,5,6,7] and r['beta_indices']==list(range(10))
            and r['split_sha256']==a.split_sha256 and r['release_sha256']==a.release_sha256
            and r['packet_sha256']==a.packet_sha256==sha(a.packet)):
        raise PermissionError('Wrong v3 training-only provenance')
    with zipfile.ZipFile(a.packet) as z:
        if sorted(z.namelist())!=['E.npy','V.npy','beta.npy','parent.npy'] or sum(x.file_size for x in z.infolist())>64*MIB:
            raise ValueError('Unexpected packet arrays/size')
    with np.load(a.packet,allow_pickle=False) as z:
        v,e,p,beta=(z[k] for k in ('V','E','parent','beta'))
    if (v.shape!=(640,64,27) or e.shape!=(640,64,386) or v.dtype!='uint8' or e.dtype!='uint8'
            or p.dtype!='int32' or p.shape!=(640,) or beta.dtype!='int16' or beta.shape!=(640,)
            or v.max()>2 or e.max()>2 or not np.array_equal(np.sort(beta),np.repeat(np.arange(10),64))):
        raise ValueError('Expected exactly 64 registered TRAIN parents, 64 anchors, each of ten betas')
    ds={};hold={}
    for b in range(10):
        ix=np.flatnonzero(beta==b)
        if len(np.unique(p[ix]))!=64:
            raise ValueError('Parent duplication')
        fit=ix[:-8];hold[b]=ix[-8:]
        ds[b]=Temperature(b,v[fit],e[fit],p[fit])
    prep_sec=time.monotonic()-st
    z=Trainer(ds,'affine','uniform',6101,a.device);timing=z.advance(256)
    st=time.monotonic(); scores={};rng=np.random.default_rng(84203)
    with torch.no_grad():
        for b,parents in hold.items():
            ix=np.tile(parents,(64,1));anchor=rng.integers(64,size=ix.shape)
            x,y=(torch.as_tensor(indicators(arr[ix,anchor]).reshape(-1,2*arr.shape[-1]),device=a.device) for arr in (v,e))
            scores[b]=dict(n_distinct_fit_validation_parents=8,groups=64,expected_nats=LOG_C-float(loss_fn(z.m,x,y).cpu()))
    valid_sec=time.monotonic()-st
    device['peak_allocated_bytes']=torch.cuda.max_memory_allocated();device['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'model.npz').open('xb') as f:
        np.savez(f,**z.weights())
    dump_new(a.out/'summary.json',dict(schema='hero-full-v3-h100-pilot-1',source='archived-physical-train-only',
             device=device,arch='affine',law='uniform',seed=6101,split_sha256=a.split_sha256,release_sha256=a.release_sha256,
             packet_sha256=a.packet_sha256,packet_bytes=a.packet.stat().st_size,model_sha256=sha(a.out/'model.npz'),
             beta_indices=list(range(10)),train_folds=r['train_folds'],n_distinct_packet_parents=640,n_training_parents=560,
             n_reserved_validation_parents=80,held_fit_validation=scores,held_fit_validation_wall_sec=valid_sec,
             validation='Last eight TRAIN packet parents each beta unused by this pilot optimizer. No fold0/1/TEST.',
             synthetic_oracle_check=check,synthetic_oracle_sec=math_sec,actual_numeric_oracle_updates=2,
             actual_training_updates=256,host_packet_decode_support_sec=prep_sec,no_dev_or_eval=True,
             runner_wall_before_final_json_sec=time.monotonic()-wall,**timing))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('packet','receipt','out'):
        ap.add_argument('--'+k,type=Path,required=True)
    for k in ('release-sha256','split-sha256','packet-sha256'):
        ap.add_argument('--'+k,required=True)
    ap.add_argument('--device',choices=('cuda',),required=True)
    run(ap.parse_args())
