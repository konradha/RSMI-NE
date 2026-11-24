"""Synthetic V3 ten-beta numeric / checkpoint integrity and frozen critic checks."""
import argparse
import resource
import tempfile
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from full_hero_train import Evaluation, Temperature, Trainer, device_setup, loss_fn, pilot_math
from full_hero_v3_sweep import ShellRows, restore, save_optimizer, state_hash, write_weights
from hero_train import ARMS, LAWS, dump_new, indicators, init_model, objective_grad


def main():
    info=device_setup('cpu');t=time.monotonic()
    rng=np.random.default_rng(923642)
    data={b:Temperature(b,rng.integers(3,size=(42+b,64,27),dtype=np.uint8),
                        rng.integers(3,size=(42+b,64,386),dtype=np.uint8),np.arange(42+b,dtype='<i4')) for b in range(10)}
    equality=pilot_math('cpu')
    m=init_model('affine',6101);v=rng.integers(3,size=(64,27),dtype=np.uint8)
    e=rng.integers(3,size=(64,386),dtype=np.uint8);h=v[:,13].astype('i8')
    ref,g=objective_grad(m,v,e,frozen=h)
    params={k:torch.tensor(a,requires_grad=True) for k,a in m.items() if k.startswith('e')}
    loss=loss_fn(params,None,torch.from_numpy(indicators(e)),torch.from_numpy(h));loss.backward()
    grad_err=max(float(abs(g[k]-p.grad.numpy()).max()) for k,p in params.items())
    loss_err=abs(float(loss.detach())-ref)
    if max(grad_err,loss_err)>2e-6: raise ValueError('Frozen hard-state loss or gradient mismatch')
    held=Evaluation(data,1901,groups=32);curves={};matched_hashes={}
    with tempfile.TemporaryDirectory() as root:
        root=Path(root);(root/'optimizer').mkdir();(root/'checkpoints').mkdir()
        for arch,law in product(ARMS,LAWS):
            z=Trainer(data,arch,law,6101,'cpu')
            a=z.advance(256);scores=held.score(z.weights())
            if not scores['all_beta_evaluable'] or not np.isfinite(scores['macro_common_hard_nats']):
                raise ValueError('Missing synthetic ten-temperature evaluation')
            key=f'{arch}-{law}'
            hashes=[v['within_beta_ordered_draws_sha256'] for v in a['beta_positive_exposures'].values()]
            if law in matched_hashes and hashes!=matched_hashes[law]:raise ValueError('Different draws for matched seeds/law')
            matched_hashes[law]=hashes
            curves[key]=dict(train_wall_sec=a['train_wall_sec'],train_cpu_sec=a['train_cpu_sec'],groups_per_wall_second=a['groups_per_wall_second'],
                             total_optimizer_steps=a['total_optimizer_steps'],beta_draw_hashes=hashes,macro_hard_nats=scores['macro_common_hard_nats'])
            if key=='tanh-matched':
                filename='math-only.npz';file_sha=write_weights(root/'checkpoints'/filename,z.weights())
                os_sha=save_optimizer(root/'optimizer'/filename,z)
                trial=Trainer(data,arch,law,6101,'cpu')
                restore(trial,root,dict(checkpoint=filename,checkpoint_sha256=file_sha,optimizer_file_sha256=os_sha,
                        optimizer_state_sha256=state_hash(z),updates=256,beta_positive_exposures={str(b):v for b,v in a['beta_positive_exposures'].items()}))
                z.advance(512);trial.advance(512)
                if state_hash(z)!=state_hash(trial):raise ValueError('Adam/model continuation diverged')
                continued_hash=state_hash(z)
    pub=Path(__file__).resolve().parents[2]/'results/l20-autoresearch/hero-full-v3'
    dump_new(pub/'optimizer-ten-beta-numerical.json',dict(source='synthetic-only generated ternary tensors',physical_parents=0,
             fake_beta_count=10,device=info,torch_numpy_math=equality,frozen_critic_loss_abs_error=loss_err,
             frozen_critic_grad_max_abs_error=grad_err,continuation_exact_at_512=True,continued_state_sha256=continued_hash,
             posterior_expectation='after each hard-symbol candidate log normalizer, never log of mixture',
             same_seed_law_exact_draws_across_arch=True,curves=curves,wall_sec=time.monotonic()-t,
             previous_planted_control_gate_sha256='dc9ed901a39d1e1ad84e24fc999bf35e58f891ec99d8592cb4cb1523fdfe784c'))


def full_shape():
    info=device_setup('cpu');start=time.monotonic()
    rng=np.random.default_rng(682217)
    # 3^4 equiprobable valid bytes with the same low-two-bits-first E2 packing.
    val=np.arange(81);table=sum((val//(3**j)%3)<<(2*j) for j in range(4)).astype('u1')
    fit={};dev={};generation_sec=0.;support_sec=0.
    for b in range(10):
        for n,data in ((6144,fit),(1024,dev)):
            t=time.monotonic()
            v=rng.integers(3,size=(n,64,27),dtype='u1');p=np.arange(n,dtype='<i4')
            packed=np.empty((n,64,97),dtype='u1')
            for lo in range(0,n,128):
                a=packed[lo:lo+128]
                a[:]=table[rng.integers(81,size=a.shape,dtype='u1')];a[...,-1]&=15
            generation_sec+=time.monotonic()-t;t=time.monotonic()
            src=ShellRows(packed,p,True)
            data[b]=Temperature(b,v,src,p);support_sec+=time.monotonic()-t
    t=time.monotonic();ev=Evaluation(dev,1901);ev_init=time.monotonic()-t
    measurements={}
    for arch,law in product(ARMS,LAWS):
        z=Trainer(fit,arch,law,6101,'cpu')
        m=z.advance(2048)
        t=time.monotonic();s=ev.score(z.weights());vt=time.monotonic()-t
        if not s['all_beta_evaluable'] or any(not v['full_roster_exposed'] for v in m['beta_positive_exposures'].values()):
            raise ValueError('Representative full shape did not exercise all ten pseudo-beta banks')
        measurements[arch+'-'+law]=dict(wall_sec=m['train_wall_sec'],cpu_sec=m['train_cpu_sec'],
            host_batch_sec=m['host_sampling_encoding_transfer_submit_sec'],groups_per_second=m['groups_per_wall_second'],
            updates=m['total_optimizer_steps'],dev_evaluation_wall_sec=vt,
            distinct_positive_parent_counts=[r['distinct_positive_parents'] for r in m['beta_positive_exposures'].values()])
    pub=Path(__file__).resolve().parents[2]/'results/l20-autoresearch/hero-full-v3'
    dump_new(pub/'full-shape-synthetic-timing.json',dict(source='independent generated synthetic ternary arrays only',
             registered_shape='ten betas, 6144/1024 pseudo-parents each fit/dev, 64 anchors, 27/386 ternary sites',
             packed_input='lossless E2 in RAM; random batch decode; not local native filesystem or H100 IO',
             physical_parents=0,device=info,generation_sec=generation_sec,support_construction_sec=support_sec,
             evaluation_group_construction_sec=ev_init,peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             actual_optimizer_measurements=measurements,wall_sec=time.monotonic()-start))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--full-shape-timing',action='store_true')
    full_shape() if ap.parse_args().full_shape_timing else main()
