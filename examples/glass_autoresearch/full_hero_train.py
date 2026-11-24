"""Forward HERO v2. Same-beta coverage sweeps; no temperature in either network."""
import argparse
import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from hero_train import (ARMS, C, G, K, LAWS, LOG_C, SEEDS, DrawLaw, center_logits,
                        dump_new, hbits, indicators, init_model, logsoftmax, objective_grad, shell_logits,
                        softmax, strata)

PUBLIC = Path(__file__).resolve().parents[2] / 'results/l20-autoresearch/hero-full-v2'
WORK = Path.home() / 'rsmi-work/l20-autoresearch-1795520725/hero-full-v2'
DEPTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
MIB = 1024**2


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(MIB), b''):
            h.update(chunk)
    return h.hexdigest()


def device_setup(dev):
    torch.set_num_threads(1)
    if dev == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; not available. No fallback for sandbox pilot.')
    properties = None
    if dev == 'cuda':
        p = torch.cuda.get_device_properties(0)
        if (torch.cuda.device_count()!=1 or 'H100' not in p.name or 'MIG' in p.name or
                (p.major,p.minor)!=(9,0) or not 78*1024**3 <= p.total_memory <= 81*1024**3):
            raise RuntimeError('Exactly one full H100 80GB is required')
        properties = dict(name=p.name,capability=[p.major,p.minor],total_memory_bytes=p.total_memory,
                          visible_device_count=torch.cuda.device_count())
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision('highest')
    return dict(torch_version=torch.__version__, numpy_version=np.__version__, cuda_build=torch.version.cuda,
                cuda_is_available=torch.cuda.is_available(), effective_device=dev, gpu=properties,
                cpu_threads=torch.get_num_threads(), cuda_event_train_ms=None)


def choose_true(rng, mask):
    """Uniform column among true columns of each row; requires nonempty rows."""
    counts = mask.sum(axis=1)
    return (mask.cumsum(axis=1) > rng.integers(counts)[:, None]).argmax(axis=1)


class Temperature:
    def __init__(self, beta, v, e, parent, evaluation_only=False):
        n, q, d = v.shape
        if d != 27 or e.shape != (n,q,386) or parent.shape != (n,) or n < (1 if evaluation_only else C) or q < 1:
            raise ValueError(f'Bad beta {beta} geometry or inadequate training parents')
        if len(np.unique(parent)) != n or v.dtype != 'uint8' or e.dtype != 'uint8':
            raise ValueError('Duplicate parents or invalid ternary array type')
        self.beta, self.v, self.e, self.parent, self.q = beta, v, e, parent, q
        self.s = strata(v)
        ordered = np.sort(self.s, axis=1)
        unique = np.ones_like(ordered, dtype=bool)
        unique[:,1:] = ordered[:,1:] != ordered[:,:-1]
        counts = np.bincount(ordered[unique])
        self.keep = counts >= C
        self.eligible = np.flatnonzero(self.keep[self.s].any(axis=1))
        if not evaluation_only and not len(self.eligible):
            raise ValueError(f'No supported matched groups at beta index {beta}')
        mass = np.bincount(self.s.ravel(), minlength=len(counts))
        self.support = dict(n_parents=n, anchors_per_parent=q, n_matched_parents=len(self.eligible),
                            unsupported_parent_ids=np.asarray(parent[~self.keep[self.s].any(axis=1)]).tolist(),
                            retained_anchor_fraction=float(mass[self.keep].sum()/mass.sum()),
                            strata_s_nparents_nanchors_kept=[[int(k),int(counts[k]),int(mass[k]),bool(self.keep[k])]
                                                        for k in np.flatnonzero(counts)])
        self.by_s = {}

    def supported_parents(self, s):
        if s not in self.by_s:
            self.by_s[s] = np.flatnonzero((self.s == s).any(axis=1))
        return self.by_s[s]


class Sweep:
    """Every supported parent is an anchor once per sweep, plus explicit padding.

    Each parent chooses a uniform supported translation; matched bins are padded
    only with other parents of that same S and a uniform eligible translation.
    All eight positions are positives and candidates symmetrically. This coverage
    measure, including its actual stratum frequencies, is not the iid dev law.
    """
    def __init__(self, data, law, seed):
        self.d, self.law, self.seed = data, law, seed
        self.digest = hashlib.sha256()
        self.groups = np.empty((0,C), dtype=np.int64)
        self.pos = self.completed = 0
        self.seen = np.zeros(len(data.parent), np.int64)
        self.gs = np.zeros(len(data.keep), np.int64)

    def cycle(self):
        d = self.d
        rng = np.random.default_rng([self.seed,917,d.beta,self.completed])
        parents = rng.permutation(len(d.parent))
        uniform = rng.random(len(parents))
        if self.law == 'uniform':
            rows = parents*d.q + (uniform*d.q).astype(np.int64)
            bins = [(None, rows)]
        else:
            good = np.isin(parents,d.eligible)
            p = parents[good]
            u = uniform[good]
            a = np.empty(len(p), np.int64)
            for lo in range(0,len(p),4096):
                sel = p[lo:lo+4096]
                mask = d.keep[d.s[sel]]
                rank = (u[lo:lo+len(sel)] * mask.sum(axis=1)).astype(np.int64)
                a[lo:lo+len(sel)] = (mask.cumsum(axis=1)>rank[:,None]).argmax(axis=1)
            rows = p*d.q+a
            ss = d.s[p,a]
            order = np.argsort(ss,kind='stable')
            bins = [(int(ss[order[i[0]]]), rows[order[i]]) for i in
                    np.split(np.arange(len(p)),np.flatnonzero(np.diff(ss[order]))+1)]
        groups = []
        for s, rows in bins:
            rem = len(rows)%C
            if rem:
                eligible = np.arange(len(d.parent)) if s is None else d.supported_parents(s)
                rest = np.setdiff1d(eligible,rows[-rem:]//d.q,assume_unique=True)
                pad = rng.choice(rest,C-rem,replace=False)
                a = rng.integers(d.q,size=len(pad)) if s is None else choose_true(rng,d.s[pad]==s)
                rows = np.r_[rows,pad*d.q+a]
            groups.append(rows.reshape(-1,C))
        groups = np.concatenate(groups)
        self.groups = groups[rng.permutation(len(groups))]
        self.pos = 0

    def draw(self,n):
        out = np.empty((n,C),dtype=np.int64)
        i = 0
        while i < n:
            if self.pos == len(self.groups):
                self.cycle()
            k = min(n-i,len(self.groups)-self.pos)
            out[i:i+k] = self.groups[self.pos:self.pos+k]
            self.pos += k
            self.completed += int(self.pos == len(self.groups))
            i += k
        return out

    def record(self, rows):
        # Called only when the entire batch of optimizer steps actually completed.
        np.add.at(self.seen, rows.ravel()//self.d.q,1)
        self.digest.update(rows.astype('<i8',copy=False).tobytes())
        if self.law == 'matched':
            np.add.at(self.gs,self.d.s.ravel()[rows[:,0]],1)

    def receipt(self):
        x = self.seen
        return dict(n_positive_slots=int(x.sum()),distinct_positive_parents=int(np.count_nonzero(x)),
                    min_exposure=int(x.min()),median_exposure=float(np.median(x)),max_exposure=int(x.max()),
                    within_beta_ordered_draws_sha256=self.digest.hexdigest(),
                    full_roster_exposed=bool(x.min()>0),completed_coverage_sweeps=self.completed,
                    matched_groups_by_s={int(i):int(self.gs[i]) for i in np.flatnonzero(self.gs)})


class Pool:
    def __init__(self,data,law,seed,frozen=None):
        self.data, self.frozen = data, frozen
        self.betas = sorted(data)
        self.rng = np.random.default_rng([seed,917])
        self.sweeps = {b:Sweep(d,law,seed) for b,d in data.items()}
        self.order = np.array([],dtype=np.int64)

    def batch(self, steps):
        # A shuffled complete beta cycle (not iid temperature draws); imbalance <=1 group.
        while len(self.order)<steps*G:
            self.order = np.r_[self.order,self.rng.permutation(self.betas)]
        beta,self.order = self.order[:steps*G], self.order[steps*G:]
        width = 27 if self.frozen is None else 1
        v = np.empty((steps*G,C,width),np.uint8)
        e = np.empty((steps*G,C,386),np.uint8)
        used = {}
        for b in self.betas:
            pos = np.flatnonzero(beta == b)
            if not len(pos):
                continue
            d = self.data[b]
            idx = self.sweeps[b].draw(len(pos))
            used[b] = idx
            v[pos] = (d.v.reshape(-1,27) if self.frozen is None else self.frozen[b].reshape(-1,1))[idx]
            e[pos] = d.e.reshape(-1,386)[idx]
        return v.reshape(steps,G*C,width), e.reshape(steps,G*C,386), used


def branch(m,x,key):
    if key+'1W' in m:
        x = torch.tanh(x @ m[key+'1W']+m[key+'1b'])
    return x @ m[key+'W'] + m[key+'b']


def loss_fn(m,x,y,hard=None):
    lp = branch(m,y,'e').reshape(-1,C,K).log_softmax(1)
    if hard is not None:
        return -lp.gather(-1,hard.reshape(-1,C,1)).mean()
    q = branch(m,x,'v').reshape(-1,C,K).softmax(-1)
    return -(q*lp).sum(-1).mean()


class Trainer:
    def __init__(self, data, arch, law, seed, device, frozen=None):
        self.pool, self.it, self.dev = Pool(data,law,seed,frozen), 0, device
        self.frozen = frozen
        m = init_model(arch,seed)
        self.m = {k:torch.tensor(v,device=device,requires_grad=True) for k,v in m.items()
                  if frozen is None or k.startswith('e')}
        self.opt = torch.optim.Adam(self.m.values(),lr=.0008,foreach=False,fused=(device=='cuda'))

    def weights(self):
        return {k:p.detach().cpu().numpy().copy() for k,p in self.m.items()}

    def advance(self,end):
        t, cpu = time.monotonic(), time.process_time()
        events = None
        if self.dev == 'cuda':
            events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            events[0].record()
        squares = torch.zeros(2,device=self.dev)
        logs = 0
        start = self.it
        host_batch = 0.
        cuda_windows = []
        while self.it < end:
            n = min(end-self.it,32)
            bt = time.monotonic()
            v,e,used = self.pool.batch(n)
            x = torch.as_tensor(indicators(v),device=self.dev) if self.frozen is None else None
            y = torch.as_tensor(indicators(e),device=self.dev)
            hard = torch.as_tensor(v,device=self.dev,dtype=torch.long) if self.frozen is not None else None
            host_batch += time.monotonic()-bt
            if events:
                w=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                w[0].record()
            for j in range(n):
                self.opt.param_groups[0]['lr'] = .0008/math.sqrt(1+(self.it+1)/1024)
                self.opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.m,x[j] if x is not None else None,y[j],hard[j] if hard is not None else None)
                loss.backward()
                if (self.it+1)%32 == 0:
                    for key,p in self.m.items():
                        squares[int(key[0]=='e')] += p.grad.square().sum()
                    logs += 1
                self.opt.step()
                self.it += 1
            if events:
                w[1].record()
                cuda_windows.append(w)
            for b,idx in used.items():
                self.pool.sweeps[b].record(idx)
        if events:
            events[1].record()
            events[1].synchronize()
        elapsed = time.monotonic()-t
        return dict(start_step=start,optimizer_steps=self.it-start,total_optimizer_steps=self.it,
                    last_minibatch_loss=float(loss.detach().cpu()),train_wall_sec=elapsed,
                    groups_per_wall_second=G*(self.it-start)/elapsed, host_sampling_encoding_transfer_submit_sec=host_batch,
                    cuda_forward_backward_adam_ms=sum(w[0].elapsed_time(w[1]) for w in cuda_windows) if events else None,
                    train_cpu_sec=time.process_time()-cpu,
                    cuda_event_train_ms=events[0].elapsed_time(events[1]) if events else None,
                    gradient_rms_norm_encoder_critic=(squares/max(logs,1)).sqrt().cpu().tolist(),
                    beta_positive_exposures={b:s.receipt() for b,s in self.pool.sweeps.items()})


class Evaluation:
    """Presampled dev measure: iid stratum by retained anchor mass, uniform distinct
    parents within stratum, uniform retained anchor per sampled parent. No training.
    """
    def __init__(self, data, seed, groups=512):
        self.data, self.rows, self.locs, self.blocks, self.hashes = data, {}, {}, {}, {}
        for b,d in data.items():
            if len(d.parent)<C or not len(d.eligible):
                self.hashes[b] = {l:None for l in LAWS}
                continue
            dl = DrawLaw(d.v)
            ix = np.array([dl.sample(np.random.default_rng([seed,b,j]),groups,l) for j,l in enumerate(LAWS)])
            self.hashes[b] = {l:hashlib.sha256(ix[j].astype('<i8',copy=False).tobytes()).hexdigest() for j,l in enumerate(LAWS)}
            self.rows[b], inv = np.unique(ix,return_inverse=True)
            self.locs[b] = inv.reshape(ix.shape)
            # Balanced randomly assigned blocks of whole parents. The same group is deleted
            # if ANY candidate is in the removed block (not just the focal candidate).
            block = np.random.default_rng([seed,b,323]).permutation(len(d.parent))%16
            self.blocks[b] = block[ix//d.q]

    def score(self,m,frozen=None):
        per = {}
        for b,d in self.data.items():
            if b not in self.rows:
                v = d.v.reshape(-1,27)
                q = softmax(center_logits(m,v)).astype('d') if frozen is None else np.eye(K)[frozen[b].ravel()]
                hf = np.bincount(q.argmax(-1),minlength=K)/len(q)
                ent,cond = float(hbits(q.mean(0))),float(hbits(q).mean())
                per[b] = {l:dict(hard_nats=None,expected_nats=None,eligible_parents=len(d.parent),
                                 reason='fewer than 8 distinct parents or no eligible matched stratum') for l in LAWS}
                per[b]['center_only_all_anchors'] = dict(H_bits=ent,H_given_V_bits=cond,I_V_H_bits=ent-cond,
                    hard_H_bits=float(hbits(hf)),hard_frequency=hf.tolist(),stochastic_frequency=q.mean(0).tolist())
                continue
            rows,loc = self.rows[b],self.locs[b]
            v = d.v.reshape(-1,27)[rows]
            q = softmax(center_logits(m,v)).astype('d') if frozen is None else np.eye(K)[frozen[b].ravel()[rows]]
            h = q.argmax(-1)
            e = shell_logits(m,d.e.reshape(-1,386)[rows]).astype('d')
            scores = {}
            for j,law in enumerate(LAWS):
                ii = loc[j]
                lp = LOG_C+logsoftmax(e[ii],1)
                hard = np.take_along_axis(lp,h[ii,None],axis=-1)[...,0].mean(1)
                soft = (lp*q[ii]).sum(-1).mean(1)
                # Uniform-law rows also give unbiased sampled code frequencies/entropies.
                freq = q[ii].mean(axis=(0,1)); hf = np.bincount(h[ii].ravel(),minlength=K)/ii.size
                cond = float(hbits(q[ii]).mean()); ent = float(hbits(freq))
                keep = [(self.blocks[b][j]!=a).all(1) for a in range(16)]
                scores[law] = dict(hard_nats=float(hard.mean()),expected_nats=float(soft.mean()),
                                   H_bits=ent,H_given_V_bits=cond,I_V_H_bits=ent-cond,
                                   hard_H_bits=float(hbits(hf)),hard_frequency=hf.tolist(),stochastic_frequency=freq.tolist(),
                                   delete_parent_block_hard_nats=[float(hard[k].mean()) if k.any() else None for k in keep])
            per[b] = scores
        complete = len(self.rows)==len(self.data)
        def macro(key):
            return float(np.mean([x[l][key] for x in per.values() for l in LAWS])) if complete else None
        return dict(per_beta=per,all_beta_evaluable=complete,macro_common_hard_nats=macro('hard_nats'),
                    macro_common_expected_nats=macro('expected_nats'))


def pilot_math(device):
    rng=np.random.default_rng(100332)
    v=rng.integers(3,size=(64,27),dtype=np.uint8)
    e=rng.integers(3,size=(64,386),dtype=np.uint8)
    rows={}
    for arch in ARMS:
        m=init_model(arch,6101)
        p={k:torch.tensor(a,device=device,requires_grad=True) for k,a in m.items()}
        ref,g=objective_grad(m,v,e)
        x,y=(torch.as_tensor(indicators(a),device=device) for a in (v,e))
        loss=loss_fn(p,x,y)
        loss.backward()
        grads={k:a.grad.detach().cpu().numpy() for k,a in p.items()}
        err=max(float(abs(grads[k]-g[k]).max()) for k in m)
        lr=.0008/math.sqrt(1+1/1024)
        torch.optim.Adam(p.values(),lr=lr,foreach=False,fused=device=='cuda').step()
        # Adam amplifies roundoff at zero (e.g. unidentifiable candidate-constant
        # logits); record raw differences as well as testing active directions.
        manual={k:m[k]-lr*grads[k]/(abs(grads[k])+1e-8) for k in m}
        expected={k:m[k]-lr*g[k]/(abs(g[k])+1e-8) for k in m}
        got={k:a.detach().cpu().numpy() for k,a in p.items()}
        update=max(float(abs(got[k]-manual[k]).max()) for k in m)
        active=max(float(abs(got[k]-expected[k])[abs(g[k])>1e-5].max(initial=0)) for k in m)
        raw=max(float(abs(got[k]-expected[k]).max()) for k in m)
        dl=abs(ref-loss.detach().item())
        rows[arch]=dict(loss_error=dl,gradient_max_error=err,adam_update_error_vs_analytic_same_grad=update,
                        raw_update_error_vs_numpy_grad=raw,active_update_error_vs_numpy_grad=active)
        if max(dl,err)>2e-6 or update>3e-7 or active>1e-6:
            raise ArithmeticError('Synthetic CUDA/CPU objective/gradient/update mismatch')
    return rows


def pilot(a):
    wall = time.monotonic()
    device = device_setup(a.device)
    mt=time.monotonic()
    math_check=pilot_math(a.device)
    math_sec=time.monotonic()-mt
    pt=time.monotonic()
    if a.packet.stat().st_size > 64*MIB or a.receipt.stat().st_size > 65536:
        raise ValueError('Oversize packet')
    with zipfile.ZipFile(a.packet) as z:
        if sorted(z.namelist())!=['E.npy','V.npy','beta.npy','parent.npy'] or sum(x.file_size for x in z.infolist())>64*MIB:
            raise ValueError('Unexpected array payload')
    r = json.loads(a.receipt.read_text())
    if not (r['schema']=='hero-full-v2-fit-pilot-1' and r['role']=='fit' and r['source']=='archived-physical'
            and r['index6_excluded'] is True and r['fit_only'] is True and r['release_sha256']==a.release_sha256
            and r['packet_sha256']==a.packet_sha256==sha(a.packet)):
        raise PermissionError('Packet provenance does not match registered input')
    with np.load(a.packet,allow_pickle=False) as z:
        v,e,p,beta = (z[k] for k in ('V','E','parent','beta'))
    betas = np.unique(beta)
    if (p.dtype!='int32' or beta.dtype!='int16' or beta.shape!=p.shape or beta.shape!=(len(v),)
            or set(betas)!=set(r['beta_indices']) or 6 in betas or v.shape[1]!=64
            or v.max()>2 or e.max()>2 or not (16<=min((beta==b).sum() for b in betas)
                                                   <=max((beta==b).sum() for b in betas)<=64)):
        raise ValueError('Unregistered fit-only pilot shape/support')
    ds,validation={},{}
    for b in betas:
        ix=np.flatnonzero(beta==b)
        if len(np.unique(p[ix]))!=len(ix):
            raise ValueError('Pilot parents repeated across optimization/validation roles')
        fit,hold=ix[:-C],ix[-C:]
        ds[int(b)]=Temperature(int(b),v[fit],e[fit],p[fit])
        validation[int(b)]=hold
    prep_sec=time.monotonic()-pt
    a.out.mkdir(parents=True,exist_ok=True)
    if any(a.out.iterdir()):
        raise FileExistsError('Pilot output must be empty')
    z = Trainer(ds,'affine','uniform',6101,a.device)
    m = z.advance(256)
    vt=time.monotonic()
    scores={}
    rng=np.random.default_rng(84203)
    with torch.no_grad():
        for b,hold in validation.items():
            ix=np.tile(hold,(64,1))
            anchor=rng.integers(64,size=ix.shape)
            x,y=(torch.as_tensor(indicators(a[ix,anchor]).reshape(-1,2*a.shape[-1]),device=z.dev) for a in (v,e))
            scores[b]=dict(n_distinct_fit_validation_parents=len(hold),groups=len(ix),
                           expected_nats=LOG_C-float(loss_fn(z.m,x,y).cpu()))
    validate_sec=time.monotonic()-vt
    device['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    device['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    with (a.out/'model.npz').open('xb') as f:
        np.savez(f,**z.weights())
    dump_new(a.out/'summary.json',dict(source='archived-physical-fit-only',device=device,arch='affine',law='uniform',
             seed=6101,packet_sha256=a.packet_sha256,release_sha256=a.release_sha256,model_sha256=sha(a.out/'model.npz'),
             synthetic_oracle_check=math_check,synthetic_oracle_sec=math_sec,
             actual_training_updates=256,actual_numeric_oracle_updates=2,
             packet_bytes=a.packet.stat().st_size,host_packet_decode_support_sec=prep_sec,
             validation='Last eight FIT packet parents each beta held from this pilot only; not dev/exam or physical endpoint.',
             held_fit_validation=scores,held_fit_validation_wall_sec=validate_sec,
             runner_wall_before_final_json_sec=time.monotonic()-wall,
             no_dev_or_eval=True,**m))


def cli():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd',required=True)
    p = sub.add_parser('pilot')
    p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--release-sha256',required=True)
    p.add_argument('--packet-sha256',required=True)
    p.add_argument('--device',choices=('cuda',),required=True)
    p.add_argument('--out',type=Path,required=True)
    a = ap.parse_args()
    pilot(a)


if __name__ == '__main__':
    cli()
