"""HERO v3 released TRAIN fold selection and final refit. No TEST reader.

One selected device; no cloud calls, retries, spending or implicit fallback.
"""
import argparse
import hashlib
import json
import math
import time
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import torch
from hero_train import ARMS, BASELINES, LAWS, SEEDS, center_logits, hbits, indicators, init_model, objective_grad
from hero_rule import _control_value, _mapped
from full_hero_train import Evaluation, Temperature, Trainer, device_setup, loss_fn, pilot_math, sha
from full_hero_v3_selection import Published, depth_status

BETAS=tuple(range(10))
DEPTHS=(1024,2048,4096,8192)
CUTOFF=datetime.fromisoformat('2026-11-24T21:00:25+00:00').timestamp()
WORK=Path.home()/'rsmi-work/l20-autoresearch-1795520725/hero-full-v3'


def utc():
    return datetime.now(timezone.utc).isoformat()


def cutoff():
    if time.time()>=CUTOFF:
        raise TimeoutError('Archive discovery cutoff reached; no new work')


def checked_json(path,digest):
    if sha(path)!=digest:
        raise PermissionError(f'Wrong pinned digest: {path.name}')
    return json.loads(path.read_text())


def rawsha(a):
    h=hashlib.sha256()
    for i in range(0,len(a),128):
        h.update(np.ascontiguousarray(a[i:i+128]).view(np.uint8))
    return h.hexdigest()


def dump_new(path,obj):
    """A complete receipt, never a partially written JSON, is the export barrier."""
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name('.'+path.name+'.partial')
    with partial.open('x') as f:
        json.dump(obj,f,separators=(',',':'),allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(partial,path)
    partial.unlink()


def hard_oracle(dev):
    rng=np.random.default_rng(8198)
    v=rng.integers(3,size=(64,27),dtype='u1');e=rng.integers(3,size=(64,386),dtype='u1')
    hard=v[:,13].astype('i8');m=init_model('affine',6101)
    expected,g=objective_grad(m,v,e,frozen=hard)
    p={k:torch.tensor(a,device=dev,requires_grad=True) for k,a in m.items() if k.startswith('e')}
    loss=loss_fn(p,None,torch.as_tensor(indicators(e),device=dev),torch.as_tensor(hard,device=dev))
    loss.backward()
    errs={k:float(abs(a.grad.detach().cpu().numpy()-g[k]).max()) for k,a in p.items()}
    diff=abs(expected-float(loss.detach().cpu()))
    if max(diff,*errs.values())>2e-6:
        raise ArithmeticError('Explicit hard-symbol oracle/gradient mismatch')
    return dict(abs_loss_error=diff,per_parameter_max_abs_gradient_error=errs)


class ShellRows:
    """Raw or packed shell rows: decode only requested minibatch/eval indices."""
    dtype=np.dtype('u1')

    def __init__(self,source,parents,packed,flat=False):
        self.source,self.parents,self.packed,self.flat=source,parents,packed,flat
        self.q=source.shape[1]
        self.shape=(len(parents)*self.q,386) if flat else (len(parents),self.q,386)

    def reshape(self,*shape):
        if shape!=(-1,386):
            raise ValueError('Shell bank supports only flat opportunity indexing')
        return ShellRows(self.source,self.parents,self.packed,True)

    def __getitem__(self,ix):
        if not self.flat:
            return ShellRows(self.source,self.parents[ix],self.packed)
        ids=np.asarray(ix)
        a=self.source[self.parents[ids//self.q],ids%self.q]
        if self.packed:
            from full_hero_v3_transport import unpack_e
            return unpack_e(a)
        return a


def load_arrays(args):
    cutoff()
    auth=checked_json(args.release,args.release_sha256)
    if (auth['schema']!='hero-full-v3-training-release-1' or auth['stage']!='train-only'
            or auth['indices']!=list(BETAS) or auth['candidate_folds']!=list(range(2,8))
            or auth['selection_fold']!=0 or auth['rule_fold']!=1 or auth['allows_test'] is not False
            or auth['permanent_train']!=81920 or auth['primary_seed']!=6101 or auth['seed_stability']!=list(SEEDS)):
        raise PermissionError('Not the v3 TRAIN-only release')
    pub=args.release.parent
    checked_json(pub/'design.json',auth['design_sha256'])
    split=checked_json(pub/'split.json',auth['split_sha256'])
    args.split_sha256=auth['split_sha256'];args.rosters_sha256=auth['rosters_sha256']
    if split['train']!=81920 or split['cross_side_intersections']!=0:
        raise ValueError('Invalid split identity')
    if not checked_json(args.control_gate,auth['control_gate_sha256'])['passed']:
        raise PermissionError('Failed numerical gate')
    args.nomination=None
    all_train=args.panel=='refit'
    if all_train:
        ref=checked_json(args.refit_release,args.refit_release_sha256)
        n=checked_json(args.nomination_file,ref['nomination_sha256'])
        if (ref['schema']!='hero-full-v3-refit-release-1' or ref['all_train'] is not True or ref['allows_test'] is not False
                or ref['split_sha256']!=args.split_sha256 or ref['rosters_sha256']!=args.rosters_sha256
                or ref['design_sha256']!=auth['design_sha256'] or ref['seeds']!=list(SEEDS)
                or ref['primary_seed']!=n['primary_seed'] or n['primary_seed']!=6101 or n['scope']!='training-fold0-selection'
                or n['schema']!='hero-full-v3-nomination-1' or n['status']!='qualified'
                or any(ref[k]!=n[k] for k in ('split_sha256','rosters_sha256','depth','arch','law'))):
            raise PermissionError('Final all-TRAIN refit not frozen or not released')
        args.nomination=n;args.nomination_sha256=ref['nomination_sha256']
    manifest=None
    if args.offline_root:
        manifest=checked_json(args.offline_root/'manifest.json',args.manifest_sha256)
        expected={f'b{b:02d}/train/{k}.npy' for b,k in product(BETAS,('V','E2','parent'))}
        if (manifest['schema']!='hero-full-v3-offline-training-input-1' or manifest['rosters_sha256']!=args.rosters_sha256
                or manifest['split_sha256']!=args.split_sha256 or set(manifest['files'])!=expected
                or manifest['train_only'] is not True or manifest['beta_indices']!=list(BETAS)
                or manifest['encoding']!='E2-le2bit-v1' or set(manifest['stream_receipts'])!={str(b) for b in BETAS}):
            raise PermissionError('Offline content outside v3 registration')
    elif args.device=='cuda':
        raise PermissionError('H100 main must run from verified bounded offline payload')
    positions=np.arange(8192)
    role_ids={'fit':positions} if all_train else {'fit':positions[positions%8>=2],'dev':positions[::8]}
    arrays={r:{} for r in role_ids};receipt_hashes={}
    for b in BETAS:
        name=f'b{b:02d}'
        if manifest is None:
            receipt=checked_json(pub/f'stream-{name}.json',args.source_pins[str(b)])
            if (receipt['index']!=b or receipt['source']!='archived-physical' or receipt['stage']!='train'
                    or receipt['rosters_sha256']!=args.rosters_sha256 or receipt['split_sha256']!=args.split_sha256):
                raise PermissionError('Wrong TRAIN roster/source')
            receipt_hashes[str(b)]=args.source_pins[str(b)]
        else:
            receipt_hashes[str(b)]=manifest['stream_receipts'][str(b)]
        fields={}
        for k in ('V','E','parent'):
            member='E2' if k=='E' and manifest is not None else k
            key=f'{name}/train/{member}.npy'
            path=(args.offline_root if manifest is not None else WORK/'streams')/key
            digest=manifest['files'][key] if manifest is not None else receipt['roles']['train']['files'][k+'.npy']['sha256']
            if sha(path)!=digest:
                raise PermissionError('Changed TRAIN-only stream')
            fields[k]=np.load(path,mmap_mode='r',allow_pickle=False)
        if (fields['parent'].dtype!='int32' or len(np.unique(fields['parent']))!=8192
                or fields['parent'].shape!=(8192,) or fields['V'].shape!=(8192,64,27)
                or fields['E'].shape!=(8192,64,97 if manifest is not None else 386)):
            raise ValueError('Incorrect TRAIN geometry or parent identity duplication')
        for r,ix in role_ids.items():
            views={k:(a if all_train else a[ix]) for k,a in fields.items() if k!='E'}
            views['E']=ShellRows(fields['E'],ix,manifest is not None)
            arrays[r][b]=views
    return arrays,receipt_hashes


def make_data(fields,denominator,eval_only=False):
    data={}
    for b,a in fields.items():
        n=len(a['V'])
        count=n if eval_only else min(n,max(32,math.ceil(n/denominator)))
        data[b]=Temperature(b,a['V'][:count],a['E'][:count],a['parent'][:count],evaluation_only=eval_only)
    return data


def reference_rate(train,entry,roots):
    paths=[r/'checkpoints'/entry['path'] for r in roots if (r/'checkpoints'/entry['path']).is_file()]
    if len(paths)!=1 or sha(paths[0])!=entry['sha256']:
        raise PermissionError('Missing/ambiguous frozen primary-seed reference weights')
    with np.load(paths[0],allow_pickle=False) as z:
        m={k:z[k] for k in z.files if k.startswith('v')}
    masses={}
    for b,d in train.items():
        v=d.v.reshape(-1,27);counts=np.zeros(4,np.int64)
        for lo in range(0,len(v),8192):
            h=center_logits(m,v[lo:lo+8192]).argmax(-1)
            counts+=np.bincount(h,minlength=4)
        masses[b]=(counts/len(v)).tolist()
    mass=np.mean(list(masses.values()),axis=0);target=np.sort(mass)[::-1]
    return dict(primary_seed=6101,weights_sha256=entry['sha256'],hard_mass_by_beta=masses,
                hard_mass=mass.tolist(),descending_target_mass=target.tolist(),hard_entropy_bits=float(hbits(mass)))


def frozen_maps(train,dev,control,reference=None):
    """Exact equal-beta empirical percentile; species ordering and ties follow pilot."""
    values={b:_mapped(d.v,lambda w:_control_value(w,control)) for b,d in train.items()}
    target=np.full(4,.25)
    if control.startswith('random_'):
        if reference is None:
            raise ValueError('Random controls need the frozen primary-seed empirical hard mass')
        target=np.asarray(reference['descending_target_mass'])
    if control=='site':
        cuts=[]
    else:
        # V3 is exactly balanced at every beta; pooled ranks equal the beta-macro CDF.
        if len({x.size for x in values.values()})!=1:
            raise ValueError('Nonuniform parent/opportunity weights in a V3 bin')
        vals=np.concatenate([x.ravel() for x in values.values()])
        cuts=np.quantile(vals,np.clip(target.cumsum()[:-1],0,1),method='inverted_cdf',overwrite_input=True).tolist()
    def label(x):
        return (x if control=='site' else np.searchsorted(cuts,x,side='left')).astype('u1')
    ff={b:label(x) for b,x in values.items()}
    fd={b:label(_mapped(d.v,lambda w:_control_value(w,control))) for b,d in dev.items()}
    spec=dict(control=control,cuts=cuts,fit_beta_weight='equal',use='fit-only thresholds frozen for all folds',
              fit_mass_by_beta={b:(np.bincount(h.ravel(),minlength=4)/h.size).tolist() for b,h in ff.items()})
    achieved=np.mean(list(spec['fit_mass_by_beta'].values()),axis=0)
    spec.update(target_mass=None if control=='site' else target.tolist(),achieved_equal_beta_mass=achieved.tolist(),
                achieved_hard_entropy_bits=float(hbits(achieved)),tie_policy='<= threshold belongs to lower bin; empty bins allowed',
                random_rate_reference=reference if control.startswith('random_') else None)
    return ff,fd,spec


def state_hash(z):
    h=hashlib.sha256()
    for name,p in sorted(z.m.items()):
        h.update(name.encode()+b'\0')
        h.update(p.detach().cpu().numpy().tobytes())
        for k in ('step','exp_avg','exp_avg_sq'):
            h.update(z.opt.state[p][k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def write_weights(path,m):
    with path.open('xb') as f:
        np.savez(f,**m)
    return sha(path)


def save_optimizer(path,z):
    a={}
    for k,p in z.m.items():
        for key in ('step','exp_avg','exp_avg_sq'):
            a[f'{k}_{key}']=z.opt.state[p][key].detach().cpu().numpy()
    return write_weights(path,a)


def restore(z,root,r):
    key=r['checkpoint'][:-4]
    ap=root/'optimizer'/(key+'.npz')
    if sha(ap)!=r['optimizer_file_sha256']:
        raise ValueError('Optimizer bytes differ from released checkpoint')
    with np.load(root/'checkpoints'/r['checkpoint'],allow_pickle=False) as w,np.load(ap,allow_pickle=False) as a:
        with torch.no_grad():
            for k,p in z.m.items():
                p.copy_(torch.as_tensor(w[k],device=z.dev))
                z.opt.state[p]={x:torch.as_tensor(a[k+'_'+x].copy(),device=z.dev) for x in ('step','exp_avg','exp_avg_sq')}
    if state_hash(z)!=r['optimizer_state_sha256']:
        raise ValueError('Restored Adam moments/weights mismatch')
    # Replay *indices only*, not physical arrays, gradients or optimizer steps.
    # Each call uses the same complete beta permutation schedule as Pool.batch.
    for _ in range(0,r['updates'],32):
        while len(z.pool.order)<32*8:
            z.pool.order=np.r_[z.pool.order,z.pool.rng.permutation(z.pool.betas)]
        beta,z.pool.order=z.pool.order[:256],z.pool.order[256:]
        for b,s in z.pool.sweeps.items():
            ix=s.draw(int((beta==b).sum()));s.record(ix)
    for b,s in z.pool.sweeps.items():
        old=r['beta_positive_exposures'][str(b)]
        if s.receipt()['within_beta_ordered_draws_sha256']!=old['within_beta_ordered_draws_sha256']:
            raise ValueError('Continuation sampler differs from completed physical optimizer exposure')
    z.it=r['updates']


def one_run(args,info,fit,dev,fe,de,den,arch,law,seed,ff=None,fd=None,spec=None,
            depths=DEPTHS,model=None,prior=None):
    start=time.monotonic()
    z=model if model is not None else Trainer(fit,arch if spec is None else 'affine',law,seed,args.device,ff)
    if model is None:
        initial=z.weights()
        z.init_hash=hashlib.sha256(b''.join(initial[k].tobytes() for k in sorted(initial))).hexdigest()
    if prior:
        restore(z,*prior)
        if z.init_hash!=prior[1]['initial_weights_sha256']:
            raise ValueError('Continuation seed initialization mismatch')
    for depth in depths:
        cutoff(); t=time.monotonic()
        timing=z.advance(depth)
        et=time.monotonic(); w=z.weights()
        fs,ds=fe.score(w,ff),de.score(w,fd)
        if not fs['all_beta_evaluable'] or not ds['all_beta_evaluable']:
            raise ValueError('Cannot rank a candidate on fewer than ten betas')
        es=time.monotonic()-et
        stamp=f'{arch}-{law}-d{den}-s{seed}-u{depth}'
        cp=args.out/'checkpoints'/(stamp+'.npz')
        digest=write_weights(cp,w)
        opt_sha=None
        if den==1 and depth>=8192 and spec is None:
            opt_sha=save_optimizer(args.out/'optimizer'/(stamp+'.npz'),z)
        full=all(r['full_roster_exposed'] for r in timing['beta_positive_exposures'].values())
        r=dict(schema='hero-full-v3-learning-1',source=args.source,arch=arch,law=law,seed=seed,denominator=den,
               updates=depth,checkpoint=cp.name,checkpoint_sha256=digest,optimizer_state_sha256=state_hash(z),
               initial_weights_sha256=z.init_hash,optimizer_file_sha256=opt_sha,split_sha256=args.split_sha256,
               release_sha256=args.release_sha256,manifest_sha256=args.manifest_sha256,
               fit=fs,dev=ds,all_enrolled_fit_parents_exposed=full,
               fit_support=f'support-d{den}.json',device=info,frozen_spec=spec,utc=utc(),
               local_cpu_measured_or_cuda='effective device in device field; no substitution',
               evaluation_cpu_numpy_wall_sec=es,stage_wall_sec=time.monotonic()-t,run_wall_sec=time.monotonic()-start,**timing)
        if args.device=='cuda':
            r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        dump_new(args.out/'learning'/(stamp+'.json'),r)
    return z,full


def finalists(args,info,fit,ds,fe,de):
    source=Published([args.out,*args.prior_root],args.release_sha256,args.split_sha256)
    choices,screen=source.primary_choice()
    models={};changes={l:[] for l in LAWS}
    for l,a in choices.items():
        for s in SEEDS[3:]:
            models[l,s],_=one_run(args,info,fit,ds,fe,de,1,a,l,s)
    for end in (8192,16384,32768,65536):
        if end>8192:
            for l,a in choices.items():
                for s in SEEDS:
                    m=models.get((l,s))
                    prev=source.read(a,l,s,end//2) if m is None else None
                    models[l,s],_=one_run(args,info,fit,ds,fe,de,1,a,l,s,depths=(end,),model=m,prior=prev)
        for l,a in choices.items():
            if end==8192:
                changes[l].append(source.change(a,l,2048,4096))
            changes[l].append(source.change(a,l,end//2,end))
        status={l:depth_status(changes[l],end,args.max_depth) for l in LAWS}
        # A significant fall is distinguished from a plateau. The same terminal
        # rung and five seeds are always compared across the two finalists.
        if all(status.values()) or any(v and v.startswith('overfit') for v in status.values()):
            break
    final={l:source.stats(a,l,SEEDS,end).tolist() for l,a in choices.items()}
    winner=max(LAWS,key=lambda l:(np.mean(final[l]),choices[l]=='affine',l=='uniform'))
    arch=choices[winner]
    weights={str(s):dict(path=source.read(arch,winner,s,end)[1]['checkpoint'],
                         sha256=source.read(arch,winner,s,end)[1]['checkpoint_sha256']) for s in SEEDS}
    nomination=dict(schema='hero-full-v3-nomination-1',scope='training-fold0-selection',status='qualified',
                    indices=list(BETAS),split_sha256=args.split_sha256,rosters_sha256=args.rosters_sha256,release_sha256=args.release_sha256,
                    depth=end,arch=arch,law=winner,primary_seed=6101,denominator=1,updates=end,
                    checkpoints=weights,paired_screen_scores=screen,selected_by_law=choices,
                    five_seed_final_scores=final,paired_doubling_changes=changes,stop_by_law=status,
                    hard_depth_cap=args.max_depth,allows_test=False)
    dump_new(args.out/'nomination.json',nomination)
    return end


def final_refit(args,info,fields,source_pins):
    """Fresh initialization on every permanent TRAIN parent; no selection scores."""
    started=time.monotonic();n=args.nomination
    if args.seed is not None:
        raise PermissionError('Final frozen path needs all five fixed seeds')
    fit=make_data(fields,1)
    receipt=dict(schema='hero-full-v3-refit-1',status='complete',source=args.source,
                 primary_seed=6101,indices=list(BETAS),arch=n['arch'],law=n['law'],depth=n['depth'],updates=n['depth'],
                 seeds=list(SEEDS),split_sha256=args.split_sha256,rosters_sha256=args.rosters_sha256,
                 release_sha256=args.release_sha256,nomination_sha256=args.nomination_sha256,
                 refit_release_sha256=args.refit_release_sha256,manifest_sha256=args.manifest_sha256,
                 stream_receipts=source_pins,checkpoints={},controls={},device=info,allows_test=False,
                 source_roles=['permanent-TRAIN-all-folds'],n_train_parents={b:len(d.parent) for b,d in fit.items()})
    def execute(name,seed,ff,spec):
        cutoff()
        normalized=name if spec is None else 'affine'
        z=Trainer(fit,normalized,n['law'],seed,args.device,ff)
        traj=[]
        for p in range(10,int(math.log2(n['depth']))+1):
            cutoff();traj.append(z.advance(2**p))
        full=all(x['full_roster_exposed'] for x in traj[-1]['beta_positive_exposures'].values())
        if not full:
            raise ArithmeticError('Refit did not expose every permanent TRAIN parent')
        cp=args.out/'checkpoints'/f'refit-{name}-{n["law"]}-s{seed}-u{n["depth"]}.npz'
        entry=dict(path=cp.name,sha256=write_weights(cp,z.weights()),optimizer_state_sha256=state_hash(z))
        dump_new(args.out/'learning'/(cp.stem+'.json'),dict(schema='hero-full-v3-final-refit-curve-1',
                 source=args.source,name=name,law=n['law'],seed=seed,split_sha256=args.split_sha256,
                 refit_release_sha256=args.refit_release_sha256,full_positive_exposure=full,frozen_spec=spec,
                 timing=traj,checkpoint=entry,device=info))
        return entry
    for seed in SEEDS:
        receipt['checkpoints'][str(seed)]=execute(n['arch'],seed,None,None)
    # Only the v3 registered six fixed controls. Refit their cuts on all TRAIN
    # and replay the primary seed's exact physical opportunity stream.
    reference=reference_rate(fit,receipt['checkpoints']['6101'],[args.out])
    for name in BASELINES:
        ff,_,spec=frozen_maps(fit,{},name,reference)
        receipt['controls'][name]=dict(frozen_spec=spec,seed=6101,**execute(name,6101,ff,spec))
    receipt['wall_sec']=time.monotonic()-started
    dump_new(args.out/'refit.json',receipt)


def main(args):
    wall=time.monotonic();cpu=time.process_time(); started=utc()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Output root must be fresh (no silent retry/replay)')
    if bool(args.offline_root)!=bool(args.manifest_sha256):
        raise ValueError('Offline directory and manifest pin must both be supplied')
    if args.panel in ('all','finalists') and args.seed is not None:
        raise ValueError('Finalist selection requires the complete paired seed set')
    if args.panel in ('full-primary','ladder') and args.seed not in (None,*SEEDS[:3]):
        raise ValueError('Screening seeds restricted to first three starts')
    info=device_setup(args.device)
    args.source='archived-physical'
    numeric=dict(source='synthetic-only generated ternary tensors',physical_parents=0,
                 frozen_hard_symbol_oracle=hard_oracle(args.device),post_log_categorical_oracle=pilot_math(args.device),
                 effective_device=info,actual_numeric_optimizer_steps=2)
    if args.device=='cpu' and not args.offline_root:
        summary=checked_json(args.source_summary,args.source_summary_sha256)
        args.source_pins=summary['stream_receipts_sha256']
    arrays,source_pins=load_arrays(args)
    data_check_sec=time.monotonic()-wall
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'checkpoints').mkdir();(args.out/'learning').mkdir()
    (args.out/'optimizer').mkdir()
    dump_new(args.out/'device-numeric.json',numeric)
    if args.panel=='refit':
        return final_refit(args,info,arrays['fit'],source_pins)
    ds=make_data(arrays['dev'],1,True); de=Evaluation(ds,1901)
    denoms={'all':(1,16,4),'full-primary':(1,),'ladder':(16,4),'frozen':(1,),'finalists':(1,)}[args.panel]
    seeds=SEEDS[:3] if args.seed is None else (args.seed,)
    for den in denoms:
        cutoff(); fit=make_data(arrays['fit'],den);fe=Evaluation(fit,1701)
        def support(d,e):
            return {b:dict(**x.support,roster_sha256=rawsha(x.parent),eval_draw_sha256=e.hashes[b]) for b,x in d.items()}
        sp=args.out/f'support-d{den}.json'
        dump_new(sp,dict(source='archived-physical',denominator=den,release_sha256=args.release_sha256,
                          fit=support(fit,fe),development_fold0=support(ds,de)))
        if args.panel not in ('frozen','finalists'):
            for arch,law,seed in product(ARMS,LAWS,seeds):
                one_run(args,info,fit,ds,fe,de,den,arch,law,seed)
    final_depth=None
    if args.panel in ('all','finalists','frozen') and den!=1:
        fit=make_data(arrays['fit'],1);fe=Evaluation(fit,1701)
    if args.panel in ('all','finalists'):
        final_depth=finalists(args,info,fit,ds,fe,de)
    if args.panel in ('all','frozen'):
        control_seeds=seeds
        control_depths=DEPTHS
        roots=[args.out,*args.prior_root]
        files=[r/'nomination.json' for r in roots if (r/'nomination.json').exists()]
        if len(files)!=1:
            raise ValueError('Random controls require a unique frozen finalist record')
        nomination=json.loads(files[0].read_text())
        if nomination['split_sha256']!=args.split_sha256 or nomination['release_sha256']!=args.release_sha256:
            raise PermissionError('Wrong TRAIN-only finalist record')
        final_depth=nomination['updates']
        reference=reference_rate(fit,nomination['checkpoints']['6101'],roots)
        if args.matched_controls:
            control_seeds=SEEDS if args.seed is None else (args.seed,)
            control_depths=tuple(2**n for n in range(10,int(math.log2(final_depth))+1))
        for control in BASELINES:
            cutoff();ff,fd,spec=frozen_maps(fit,ds,control,reference)
            for law,seed in product(LAWS,control_seeds):
                one_run(args,info,fit,ds,fe,de,1,control,law,seed,ff,fd,spec,depths=control_depths)
    curves=[json.loads(p.read_text()) for p in (args.out/'learning').glob('*.json')]
    unique={(r['denominator'],r['arch'],r['law'],r['seed']) for r in curves}
    dump_new(args.out/'completion.json',dict(schema='hero-full-v3-sweep-completion-1',source='archived-physical',
             release_sha256=args.release_sha256,stream_receipts=source_pins,device=info,
             started=started,ended=utc(),wall_sec=time.monotonic()-wall,cpu_sec=time.process_time()-cpu,
             source_validation_and_setup_wall_sec=data_check_sec,panel=args.panel,n_runs=len(unique),n_checkpoints=len(curves),
             optimizer_steps=sum(c['optimizer_steps'] for c in curves),
             complete_parent_exposures=all(r['all_enrolled_fit_parents_exposed'] for r in curves if r['updates']>=8192),
             split_sha256=args.split_sha256,finalist_depth=final_depth,matched_controls=args.matched_controls,
             completed_runs=sorted(unique),allows_test=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release',type=Path,required=True);ap.add_argument('--release-sha256',required=True)
    ap.add_argument('--offline-root',type=Path);ap.add_argument('--manifest-sha256')
    ap.add_argument('--device',choices=('cpu','cuda'),required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--panel',choices=('all','full-primary','ladder','frozen','finalists','refit'),required=True)
    ap.add_argument('--seed',type=int,choices=SEEDS)
    ap.add_argument('--control-gate',type=Path,required=True);ap.add_argument('--prior-root',type=Path,action='append',default=[])
    ap.add_argument('--source-summary',type=Path);ap.add_argument('--source-summary-sha256')
    ap.add_argument('--refit-release',type=Path);ap.add_argument('--refit-release-sha256');ap.add_argument('--nomination-file',type=Path)
    ap.add_argument('--max-depth',type=int,choices=(8192,16384,32768,65536),default=65536)
    ap.add_argument('--matched-controls',action='store_true')
    main(ap.parse_args())
