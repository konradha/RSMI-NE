"""Summarize every checkpoint; lock all final maps and both decoder calibrations before one explicitly released audit."""
import tranche2_estimator as e

import argparse
import csv
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def summ(a):
    a=np.asarray(a); return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),ci95=e.ci(a),n=len(a))


def contrast(seed_x_parent):
    a=np.asarray(seed_x_parent)
    return dict(paired_seed_means=a.mean(1).tolist(),seed_average_parent_interval=summ(a.mean(0)),parent_average_seed_interval=summ(a.mean(1)),each_seed_parent_intervals=[summ(z) for z in a])


def plot_dev(rows, path):
    fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    colors=dict(direct='#0072B2',contrastive='#D55E00',distilled='#009E73')
    for k,n in enumerate(e.RUNGS):
        top,bot=axs[:,k]; rr=[r for r in rows if r['fit_n']==n]
        for arm in e.ARMS:
            for si,s in enumerate(e.SEEDS):
                line=[r for r in rr if r['seed']==s and r['arm']==arm]
                top.plot([r['step'] for r in line],[r['dev_hard_proper'] for r in line],color=colors[arm],alpha=.65,marker='.',label=arm if si==0 else None)
        top.set(title=f'{n:,} fitting parents',xlabel='Adam updates',ylabel='Dev proper gain (nat/face)'); top.grid(alpha=.2)
        if k==0: top.legend(fontsize=8)
        for si,s in enumerate(e.SEEDS):
            def get(arm): return np.array([r['dev_hard_proper'] for r in rr if r['seed']==s and r['arm']==arm])
            bot.plot(e.CHECK,get('direct')-get('contrastive'),marker='o',ms=3,label=f'{s}')
        bot.axhline(.001,color='black',ls='--',lw=.8);bot.axhline(0,color='grey',lw=.5);bot.set(xlabel='Adam updates',ylabel='Direct − contrastive (nat/face)');bot.grid(alpha=.2)
        if k==0:bot.legend(fontsize=8,title='Paired seed')
    fig.savefig(path,dpi=150);plt.close(fig)


def development():
    rows=[]; rungs={}; hashes={}
    for n in e.RUNGS:
        fp=e.OUT/f'21-frozen-{n}.npz'; hashes[fp.name]=e.digest(fp)
        rec=json.loads((e.OUT/f'21-fit-{n}.json').read_text()); rr={}; rungs[n]=rr
        with np.load(fp) as a:
            for step in e.CHECK:
                grids={arm:np.stack([a[f'{arm}_{s}_{step}_dev_parents'] for s in e.SEEDS]) for arm in e.ARMS}
                rr[step]=dict(direct_minus_contrastive=contrast(grids['direct']-grids['contrastive']),distilled_minus_fit_only_teacher=contrast(grids['distilled']-a['teacher_dev_parents']), distill_minus_direct=contrast(grids['distilled']-grids['direct']),scores={arm:summ(grid.mean(0)) for arm,grid in grids.items()})
                for s in e.SEEDS:
                    for arm in e.ARMS:
                        c=rec['arms'][f'{arm}_{s}']['checkpoints'][str(step)]; train,dev=c['fit'],c['dev']
                        parents=a[f'{arm}_{s}_{step}_dev_parents']; op=summ(train['hard_fit_decoder_nats']-parents)
                        rows.append(dict(fit_n=n,seed=s,arm=arm,step=step,fit_hard_proper=train['hard_fit_decoder_nats'],dev_hard_proper=dev['hard_fit_decoder_nats'],train_dev_optimism=op['mean'],optimism_lo_conditional_fit=op['ci95'][0],optimism_hi_conditional_fit=op['ci95'][1],fit_hard_plugin=train['hard_plugin_nats'],dev_hard_plugin=dev['hard_plugin_nats'],fit_soft_MI=train['soft_exact_MI_nats'],dev_soft_MI=dev['soft_exact_MI_nats'],fit_hard_entropy=train['hard_entropy_nats'],dev_hard_entropy=dev['hard_entropy_nats'],fit_teacher_agreement=train['teacher_agreement'],dev_teacher_agreement=dev['teacher_agreement'],train_seconds=c['training_clock_s'],train_cpu_seconds=c['training_cpu_s'],grad_noise_rms=c['gradient_noise']['encoder_batch_gradient_rms_noise']))
            rr['last_extension']={}
            for arm in e.ARMS:
                d=np.stack([a[f'{arm}_{s}_5120_dev_parents']-a[f'{arm}_{s}_2560_dev_parents'] for s in e.SEEDS]); rr['last_extension'][arm]=dict(change=contrast(d),mean_abs_change_below_plateau=bool(abs(d.mean())<.0001))
            rr['teacher_dev_score']=summ(a['teacher_dev_parents']); rr['teacher_dev_score']['fit_score']=rec['teacher']['proper_fit']
            # Secondary sensitivity only. Choose checkpoints by elapsed time, never by scores.
            same_time=[]; policies=[]
            for s in e.SEEDS:
                clock={arm:{k:rec['arms'][f'{arm}_{s}']['checkpoints'][str(k)]['training_clock_s'] for k in e.CHECK} for arm in ('direct','contrastive')}
                budget=min(clock[arm][5120] for arm in clock)
                chosen={arm:max(k for k,v in clock[arm].items() if v<=budget) for arm in clock}
                same_time.append(a[f"direct_{s}_{chosen['direct']}_dev_parents"]-a[f"contrastive_{s}_{chosen['contrastive']}_dev_parents"])
                policies.append(dict(seed=s,wall_budget_s=budget,steps=chosen,used_wall_s={arm:clock[arm][k] for arm,k in chosen.items()}))
            rr['wall_budget_sensitivity']=dict(rule='Largest available prespecified checkpoint not exceeding minimum final training wall time of the paired arms; coarse checkpoint spacing leaves some budget unused. Development only.',policies=policies,direct_minus_contrastive=contrast(same_time))
    with (e.OUT/'21-learning.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plot_dev(rows,e.OUT/'21-learning.png')
    e.jout(e.OUT/'21-development.json',dict(rungs=rungs,statistical_units='Intervals condition on fixed fitted maps. Parent interval on seed-averaged within-parent effects (512 disjoint parents); paired-seed t interval across 3 seed means conditional on parents, not a combined interval. Nested rungs share parents. Fit-minus-dev interval holds observed adaptive training score fixed and measures development-sampling uncertainty, not selection correction. Hard plugin is descriptive only.'))
    d=rungs[4096][5120]['direct_minus_contrastive']; effect=d['seed_average_parent_interval']['mean']
    prediction='above +0.001' if effect>.001 else 'below -0.001' if effect<-.001 else 'within [-0.001,+0.001]'
    lock=dict(phase='before audit species or scores; historical TRAIN, no claim of pristine population',public_design_commit='1036785',hashes=hashes,development_summary_sha=e.digest(e.OUT/'21-development.json'),latest_common_update=5120,seeds=e.SEEDS,primary_fit_parents=4096,primary_decoder='fitted to the 512 development parents; no audit fitting; direction-marginal baseline on same dev parents',predicted_audit_mean_direct_minus_contrastive_category=prediction,development_forecast_center_fit_calibrated=effect,meaningful_margin_nat_per_face=.001,primary_rule='Largest-rung seed-averaged hard log predictive score of direct minus contrastive, report 95% paired parent interval and separate seed CI; parent lower bound > +0.001 required for positive materiality. Report at all three rungs, fit calibration as prespecified sensitivity, and frozen fit-only teacher. One audit pass, no reselection.',references=['https://arxiv.org/abs/1905.06922','https://doi.org/10.1103/PhysRevLett.127.240603'])
    e.jout(e.OUT/'21-prediction.json',lock)
    print('PRIMARY DEVELOPMENT',d,flush=True)


def audit(public_lock):
    started=time.monotonic(); lock=json.loads((e.OUT/'21-prediction.json').read_text()); dest=e.OUT/'21-audit.json'
    if dest.exists() or not public_lock or len(public_lock)<7: raise RuntimeError('requires a new, explicitly released public lock')
    for name,sha in lock['hashes'].items(): assert e.digest(e.OUT/name)==sha
    v,y=e.extract(4608,5120); parents={}; rungs={}
    for n in e.RUNGS:
        nr={}; rungs[n]=nr
        with np.load(e.OUT/f'21-frozen-{n}.npz') as a:
            for cal,tkey,key in [('development','teacher_dev_decoder','devdec'),('fit','teacher_fit_decoder','fitdec')]:
                teacher=e.parent_scores(a['teacher'],a[tkey],v,y); parents[f'{n}_{cal}_teacher']=teacher; grids={}
                for arm in e.ARMS:
                    grids[arm]=np.stack([e.parent_scores(a[f'{arm}_{s}_h'],a[f'{arm}_{s}_{key}'],v,y) for s in e.SEEDS]);parents[f'{n}_{cal}_{arm}']=grids[arm]
                nr[cal]=dict(teacher=summ(teacher),scores={arm:contrast(g) for arm,g in grids.items()},direct_minus_contrastive=contrast(grids['direct']-grids['contrastive']),distilled_minus_teacher=contrast(grids['distilled']-teacher),teacher_minus_direct=contrast(teacher-grids['direct']),teacher_minus_contrastive=contrast(teacher-grids['contrastive']),audit_dev_calibration_fit_sensitivity_note='Separate marginal and conditional probabilities calibrated on the same named fold')
    primary=rungs[4096]['development']['direct_minus_contrastive']; cint=primary['seed_average_parent_interval']['ci95']; seed_ci=primary['parent_average_seed_interval']['ci95']
    if cint[0]>.001: decision='Material advantage of the direct estimator on this fixed recipe, conditional on the three trained seeds.'
    elif cint[1]<-.001: decision='Material advantage of the contrastive recipe on this fixed comparison, conditional on the three trained seeds.'
    elif max(abs(x) for x in cint)<.001: decision='Seed-average effect lies inside the ±0.001 practical equivalence band conditional on these three trained seeds.'
    elif cint[1]<.001: decision='No +0.001 direct advantage for these three trained seeds; interval is not a two-sided equivalence verdict.'
    else: decision='Inconclusive at the +0.001 margin for these three trained seeds.'
    obs=dict(public_lock=public_lock,prediction_sha256=e.digest(e.OUT/'21-prediction.json'),cohort='Historical b09 native TRAIN audit positions [4608:5120]; maps/calibration frozen, never used for tuning',face_units_per_parent=6000,n_parents=512,seconds=time.monotonic()-started,rungs=rungs,decision=decision,primary_seed_interval=seed_ci,limitations=['CIs over parents condition on three paired fits; n=3 paired training-seed t interval separately conditions on these parents and is fragile. Neither estimates joint over bank and seed.', 'Neither estimator is an encoder-independent bound; one architecture and critic, fixed learning rate, optimizer, update cap, varying statistical work per step. No inference about other shell targets or global optimality.', 'The decoder log-score gain is a proper operational measure of relevance, not an unbiased MI point estimate. Prior historical TRAIN reuse makes this exploratory.'])
    np.savez_compressed(e.OUT/'21-audit-parents.npz',**parents);e.jout(dest,obs)
    fig,(left,right)=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for i,n in enumerate(e.RUNGS):
        row=rungs[n]['development']
        for s,(c,d) in enumerate(zip(row['scores']['contrastive']['paired_seed_means'],row['scores']['direct']['paired_seed_means'])):
            left.plot([i-.2,i+.2],[c,d],'-o',color=('#0072B2','#D55E00','#009E73')[s],alpha=.8,label=str(e.SEEDS[s]) if i==0 else None)
        gain=row['direct_minus_contrastive']
        for j,key in enumerate(('seed_average_parent_interval','parent_average_seed_interval')):
            s=gain[key]; right.errorbar(i+(j-.5)*.16,s['mean'],yerr=[[s['mean']-s['ci95'][0]],[s['ci95'][1]-s['mean']]],fmt='o',capsize=3,color=('black','#CC79A7')[j],label=('parent interval','seed interval')[j] if i==0 else None)
    left.set(xticks=range(3),xticklabels=e.RUNGS,xlabel='Fit parents (contrastive → direct at each rung)',ylabel='Audit proper score (nat/face)');left.legend(title='Seed')
    right.axhline(0,color='grey',lw=.7);right.axhline(.001,color='grey',ls='--');right.set(xticks=range(3),xticklabels=e.RUNGS,xlabel='Fitting parents',ylabel='Audit direct − contrastive (nat/face)');right.legend()
    fig.savefig(e.OUT/'21-audit-scatter.png',dpi=150);plt.close(fig)
    print('PRIMARY AUDIT',primary,decision,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('development','audit')); p.add_argument('--released-public-lock'); args=p.parse_args()
    if args.phase=='development': development()
    else: audit(args.released_public_lock)
