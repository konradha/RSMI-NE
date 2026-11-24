"""#21 diagnostic: deterministic replay, same optimizer through 20480. Never opens audit rows."""
import datetime
import time
import tranche2_estimator as e  # sets single-thread environment before NumPy import
import numpy as np


ARMS = ('direct', 'contrastive')
STEPS = (5120, 10240, 20480)
STRIDE = 32
ORIGINAL_SHA = 'ca10153185829e4a13c3abac48157855f7e395b69b3b1d9d363c68b1d2b066ba'
FREEZE = '21af85d1cc73181b14c69839e70d7993eb677cb5'


def hard_score(h, fit, dev, dv, dy):
    f = fit.hard(h); d = dev.hard(h); decoder = e.decoder(f)
    par = e.parent_scores(h, decoder, dv, dy)
    train = float(np.sum(f * decoder) / f.sum()); held = float(np.sum(d * decoder) / d.sum())
    if not np.isclose(par.mean(),held,rtol=0.,atol=1e-12): raise ValueError('parent weighting mismatch')
    return dict(fit=train, development=held, fit_H=e.hard_stats(f)['hard_entropy_nats'], dev_H=e.hard_stats(d)['hard_entropy_nats']), par, decoder


def train(seed, arm, fit, vf, yf, frozen, dev, dv, dy, deadline):
    key=f'{arm}_{seed}'; p=e.init(seed); ap=e.Adam(p,.005); rng=np.random.default_rng(seed+10000)
    critic=np.zeros((e.J*e.Y,e.H),'f4'); ac=e.Adam([critic],.01)
    path=np.empty((1+STEPS[-1]//STRIDE,e.V),'u1'); path[0]=e.forward(p,e.X)[1].argmax(1)
    clocks=np.zeros((len(path),2)); state={}; replay=None
    for batch in range(1,len(path)):
        if time.monotonic()>deadline: raise TimeoutError('registered ceiling reached')
        wall=time.perf_counter(); cpu=time.process_time()
        for _ in range(STRIDE):
            if arm=='contrastive':
                _,g,cg=e.nce(p,critic,e.nce_sample(rng,vf,yf)); ac.update([cg])
            else: _,g=e.full_obj(p,fit,'direct',None)
            ap.update(g)
        clocks[batch]=clocks[batch-1]+(time.process_time()-cpu,time.perf_counter()-wall)
        h=path[batch];h[:]=e.forward(p,e.X)[1].argmax(1); step=batch*STRIDE
        if step==STEPS[0]:
            score,par,decoder=hard_score(h,fit,dev,dv,dy)
            checks={f'encoder_p{i}':a.dtype==frozen[f'{key}_p{i}'].dtype and a.tobytes()==frozen[f'{key}_p{i}'].tobytes() for i,a in enumerate(p)}
            checks.update(lut=np.array_equal(h,frozen[key+'_h']),fit_decoder=np.array_equal(decoder,frozen[key+'_fitdec']),dev_scores=np.array_equal(par,frozen[key+'_5120_dev_parents']))
            if not all(checks.values()):raise RuntimeError(f'5120 replay mismatch: {key} {checks}')
            replay=dict(exact_checks=checks,encoder_adam_updates=ap.n,critic_adam_updates=ac.n,process_cpu_s=clocks[batch,0],training_wall_s=clocks[batch,1]);print('VALIDATED',key,flush=True)
        if step in STEPS:
            # Portable state sufficient for continuation without another replay. Never overwrites original.
            dest=e.OUT/f'21-convergence-state-{key}-{step}.npz'; values={'lut':h.copy(),'critic':critic,'rng_state':e.json.dumps(rng.bit_generator.state),'encoder_step':ap.n,'critic_step':ac.n}
            for name,arrays in [('p',p),('m',ap.m),('v',ap.v),('cm',ac.m),('cv',ac.v)]:
                values.update({f'{name}{i}':value for i,value in enumerate(arrays)})
            np.savez_compressed(dest,**values);state[step]=e.digest(dest)
    return path,clocks,replay,state


def run():
    start=time.monotonic(); original=e.OUT/'21-frozen-4096.npz'; prior=e.json.loads((e.OUT/'21-fit-4096.json').read_text())['elapsed_since_ladder_start_seconds']
    deadline=start+min(3*3600-prior,datetime.datetime(2026,11,24,16,40,tzinfo=datetime.timezone.utc).timestamp()-time.time())
    if e.digest(original)!=ORIGINAL_SHA:raise ValueError('original frozen checkpoint mismatch')
    if (e.OUT/'21-convergence-seed-2101.json').exists():raise FileExistsError('refusing overwrite of existing diagnostic')
    v,y=e.extract(0,4608); fit=e.Channel(e.counts(v[:4096],y[:4096])); dv,dy=v[4096:],y[4096:];dev=e.Channel(e.counts(dv,dy));vf=v[:4096].ravel();yf=y[:4096].reshape(-1,6)
    with np.load(original) as frozen:
        for seed in e.SEEDS:
            traces={arm:train(seed,arm,fit,vf,yf,frozen,dev,dv,dy,deadline) for arm in ARMS}
            evidence={};rows=[]
            for end in STEPS:
                target=end//STRIDE; budget=min(traces[a][1][target,0] for a in ARMS)
                for basis in ('updates','cpu'):
                    for arm in ARMS:
                        path,clocks,_,_=traces[arm];index=target if basis=='updates' else int(np.searchsorted(clocks[:target+1,0],budget,side='right')-1)
                        metrics,par,_=hard_score(path[index],fit,dev,dv,dy);evidence[f'{basis}_{end}_{arm}']=par
                        rows.append(dict(endpoint=end,basis=basis,arm=arm,seed=seed,actual_step=int(index*STRIDE),cpu_s=float(clocks[index,0]),training_wall_s=float(clocks[index,1]),matched_budget_s=float(budget) if basis=='cpu' else None,unused_cpu_s=float(budget-clocks[index,0]) if basis=='cpu' else None,**metrics))
            ev=e.OUT/f'21-convergence-parents-{seed}.npz';np.savez_compressed(ev,**evidence)
            e.jout(e.OUT/f'21-convergence-seed-{seed}.json',dict(freeze_commit=FREEZE,source_sha256=e.digest(original),implementation_sha256=e.digest(e.Path(__file__)),bank_positions_read=[0,4608],fit_parents=4096,development_parents=512,anchors_per_parent=1000,faces_per_anchor=6,seed=seed,replay={a:t[2] for a,t in traces.items()},model_state_sha256={a:t[3] for a,t in traces.items()},scores=rows,parents_sha256=e.digest(ev),earlier_ladder_wall_s=prior,elapsed_wall_s=time.monotonic()-start,clock='single-thread process time for gradient, sampling, critic and Adam only; excludes checkpointing, hard LUT extraction and evaluation. 5120 replay time included in matched training clock, not counted twice.'))
            print('COMPLETE',seed,'elapsed',round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':run()
