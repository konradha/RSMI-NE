"""Fit/development only. Run with the standard single-thread RSMI venv. Store final audit inputs without reading audit species."""
import tranche2_estimator as e

import itertools
import json
import time

import numpy as np


def hist_agree(a, b, freq):
    joint = np.bincount(a.astype(int)*4+b, weights=freq, minlength=16).reshape(4,4)
    return float(max(joint[np.arange(4), cols].sum() for cols in itertools.permutations(range(4))))


def run():
    start = time.monotonic(); e.OUT.mkdir(parents=True, exist_ok=True)
    obs = []
    for n in (*e.RUNGS, 'development', 'audit'):
        sl = slice(0, n) if isinstance(n, int) else slice(4096,4608) if n == 'development' else slice(4608,5120)
        with np.load(e.BANK) as archive:
            lineage = {k: archive[k][sl] for k in ('parent','chain_id','chain_seed')}
            assert len(np.unique(lineage['parent'])) == len(lineage['parent'])
            obs.append(dict(fold=n, n=len(lineage['parent']), lineage_sha256={k:e.hashlib.sha256(a.tobytes()).hexdigest() for k,a in lineage.items()}))
    with np.load(e.BANK) as arch:
        assert len(np.unique(arch['parent'])) == 5120
        assert len(np.unique(np.stack((arch['chain_seed'],arch['chain_id']),axis=1), axis=0)) == 5120
    v,y = e.extract(0,4608); dv,dy = v[4096:],y[4096:]; dev = e.Channel(e.counts(dv,dy))
    hist_path = e.ROOT/'results/l20-autoresearch/faces/optimized.npz'
    with np.load(hist_path) as arch: prior = arch['discrete_lut']
    e.jout(e.OUT/'21-lineage.json',dict(bank_meta=json.loads(e.BANK.with_suffix('.json').read_text()), folds=obs, all_disjoint_except_declared_fit_nesting=True, physical_species_read="positions [0,4608) only before audit release", historical_teacher_only_postfit_fit_weighted_agreement=e.digest(hist_path), extract_wall_seconds=time.monotonic()-start))
    for n in e.RUNGS:
        fv,fy = v[:n],y[:n]; vf,yf = fv.ravel(),fy.reshape(-1,6)
        fit = e.Channel(e.counts(fv,fy)); now=time.monotonic(); gt,steps=e.teacher(fit)
        tr = fit.hard(gt); dc=dev.hard(gt); lr=e.decoder(tr); devp=e.parent_scores(gt,lr,dv,dy)
        arrays = dict(teacher=gt, teacher_fit_decoder=lr, teacher_dev_decoder=e.decoder(dc), teacher_dev_parents=devp)
        rec = dict(n=n, teacher=dict(sweeps_changed_tiles=steps, fit=e.hard_stats(tr), dev=e.hard_stats(dc), proper_fit=float((tr*lr).sum()/tr.sum()), proper_dev=float(devp.mean()), fit_weighted_permutation_agreement_with_pooled_reference=hist_agree(gt,prior,fit.p), seconds=time.monotonic()-now), arms={})
        for seed in e.SEEDS:
            for arm in e.ARMS:
                key=f'{arm}_{seed}'; rng=np.random.default_rng(seed+10000)
                p=e.init(seed); critic=np.zeros((e.J*e.Y,e.H),'f4'); ap=e.Adam(p,.005); ac=e.Adam([critic],.01)
                trace=[]; checkpoints={}; objbuf=[]; gradbuf=[]; stim=time.monotonic(); compute=0.; cpu=0.
                for step in range(e.CHECK[-1]+1):
                    if step:
                        ti=time.perf_counter(); ct=time.process_time()
                        if arm=='contrastive':
                            obj,g,cg=e.nce(p,critic,e.nce_sample(rng,vf,yf)); ac.update([cg])
                        else:
                            obj,g=e.full_obj(p,fit,arm,gt)
                        ap.update(g); objbuf.append(obj); gradbuf.append(float(np.linalg.norm(e.gradvec(g))))
                        compute+=time.perf_counter()-ti; cpu+=time.process_time()-ct
                        if step%64==0:
                            trace.append([step,float(np.mean(objbuf)),float(np.mean(gradbuf)),compute,cpu]); objbuf=[]; gradbuf=[]
                    if step not in e.CHECK: continue
                    score,h,lr,par,devdec=e.measure(p,gt,fit,dev,dv,dy)
                    score.update(training_clock_s=compute,training_cpu_s=cpu,wall_including_evaluation_s=time.monotonic()-stim,gradient_noise=e.noise(p,critic,vf,yf,seed+step+4500) if arm=='contrastive' else dict(encoder_batch_gradient_rms_noise=0.,reason='exact full-batch, not population uncertainty'))
                    score['fit']['best_permutation_agreement_with_published_teacher_on_fit_only']=hist_agree(h,prior,fit.p)
                    checkpoints[step]=score; arrays[f'{key}_{step}_dev_parents']=par
                    if step==e.CHECK[-1]:
                        arrays[key+'_h']=h; arrays[key+'_fitdec']=lr; arrays[key+'_devdec']=devdec
                        arrays[key+'_final_fit_parents']=e.parent_scores(h,lr,fv,fy)
                        for l,coeff in enumerate(p): arrays[f'{key}_p{l}']=coeff
                rec['arms'][key]=dict(checkpoints=checkpoints,trace_columns=['step','mean_training_objective_prior64','mean_encoder_gradient_norm_prior64','training_wall_seconds','training_cpu_seconds'],trace=trace,effective_positive_cases_per_update=(1024 if arm=='contrastive' else n*1000*6 if arm=='direct' else n*1000),unique_word_forward_rows_per_update=(1024 if arm=='contrastive' else 6561),negative_faces_per_positive=(15 if arm=='contrastive' else 0))
                rec['elapsed_since_ladder_start_seconds']=time.monotonic()-start
                # Task-level checkpoint after each independent trajectory; physical audit never read here.
                e.jout(e.OUT/f'21-fit-{n}.json',rec); np.savez_compressed(e.OUT/f'21-frozen-{n}.npz',**arrays)
                print('FINISHED',n,seed,arm,'updates',step,'training_wall',round(compute,1),'total_elapsed',round(time.monotonic()-start,1),flush=True)
        if time.monotonic()-start > 3*3600: raise RuntimeError('registered training wall cap reached')


if __name__=='__main__':
    run()
