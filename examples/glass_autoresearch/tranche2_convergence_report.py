"""Summarize the frozen training-only diagnostic; no configuration input or audit path."""
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tranche2_estimator as e
from tranche2_convergence import ARMS,STEPS,FREEZE


def summary(x):
    parent=x.mean(0);seed=x.mean(1)
    return dict(mean=float(parent.mean()),seed_means=seed.tolist(),seed_minmax=[float(seed.min()),float(seed.max())],parent_ci95=e.ci(parent),seed_ci95=e.ci(seed),parent_n=len(parent),seed_n=len(seed))


def diagnose(x):
    d=summary(x);lo,hi=d['parent_ci95'];seeds=np.array(d['seed_means'])
    d['classification']='narrowing' if hi<0 and np.all(seeds<0) else 'level within 0.0001' if lo>=-.0001 and hi<=.0001 and np.max(abs(seeds))<=.0001 else 'unresolved'
    return d


def run():
    reports=[];frames=[];evidence={};wall=0.
    for seed in e.SEEDS:
        rec=json.loads((e.OUT/f'21-convergence-seed-{seed}.json').read_bytes());assert rec['freeze_commit']==FREEZE and rec['bank_positions_read']==[0,4608] and all(all(a['exact_checks'].values()) for a in rec['replay'].values());reports.append(rec);frames.extend(rec['scores']);fp=e.OUT/f'21-convergence-parents-{seed}.npz';assert e.digest(fp)==rec['parents_sha256']
        with np.load(fp) as a:
            for k in a.files:evidence.setdefault(k,[]).append(a[k])
        wall=max(wall,rec['elapsed_wall_s'])
    evidence={k:np.stack(v) for k,v in evidence.items()};by={}; changes={}
    for basis in ('updates','cpu'):
        gaps=[];steps={}
        for end in STEPS:
            dx,cx=(evidence[f'{basis}_{end}_{arm}'] for arm in ARMS);gaps.append(dx-cx);rows=[r for r in frames if r['basis']==basis and r['endpoint']==end]; scores={}
            for arm,arr in zip(ARMS,(dx,cx)):
                arm_rows=[r for r in rows if r['arm']==arm]
                scores[arm]=dict(fit_proper_gain=np.mean([r['fit'] for r in arm_rows]),dev_proper_gain=summary(arr),realized_steps=[r['actual_step'] for r in arm_rows],cpu_s=[r['cpu_s'] for r in arm_rows],hard_dev_entropy=np.mean([r['dev_H'] for r in arm_rows]))
            slack=[r['unused_cpu_s']/r['matched_budget_s'] for r in rows] if basis=='cpu' else []
            steps[end]=dict(arms=scores,gap=summary(gaps[-1]),matched_cpu_max_unused_fraction=max(slack) if slack else None)
        changes[basis]={'initial_to_final':diagnose(gaps[2]-gaps[0]),'first_doubling':diagnose(gaps[1]-gaps[0]),'last_doubling':diagnose(gaps[2]-gaps[1])};by[basis]=steps
    dest=e.OUT/'21-convergence.csv'
    with dest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(frames[0]));writer.writeheader();writer.writerows(frames)
    endpoint_rows=[r for r in frames if r['basis']=='updates' and r['endpoint']==STEPS[-1]];total_cpu=sum(r['cpu_s'] for r in endpoint_rows);replay_cpu=sum(t['process_cpu_s'] for r in reports for t in r['replay'].values());obj=dict(freeze_commit=FREEZE,diagnostic_not_audit=True,primary_curve='same updates; equal-process-CPU sensitivity uses last 32-update grid point within each seed-specific budget',endpoints=STEPS,training_seeds=e.SEEDS,results=by,changes=changes,training_process_cpu_s_including_replay=total_cpu,replay_cpu_s=replay_cpu,continuation_cpu_s=total_cpu-replay_cpu,total_diagnostic_wall_s=wall,wall_s_including_prior_original_ladder=wall+reports[-1]['earlier_ladder_wall_s'],cost_USD=0,interpretation='All uncertainty conditional on a repeatedly viewed development set and fixed fit bank. Paired parent intervals condition on these three trained starts; separate small-n seed intervals condition on these parents. Descriptive only; no new audit, extrapolation, or winner. No #25 models run. True scores between checkpoints need not be linear.');e.jout(e.OUT/'21-convergence.json',obj)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.6,4.1),layout='constrained');x=np.arange(3);colors=('#2377a0','#b44a21')
    for arm,col in zip(ARMS,colors):
        rr=by['updates'];y=np.array([rr[z]['arms'][arm]['dev_proper_gain']['seed_means'] for z in STEPS]);ax.plot(x,y,lw=.8,alpha=.4,color=col);ax.plot(x,y.mean(1),'-o',label=arm,color=col);ax.plot(x,[rr[z]['arms'][arm]['fit_proper_gain'] for z in STEPS],':',lw=1.7,color=col)
    ax.set(title='Hard predictive gain',ylabel='nats / microscopic face');ax.legend(title='solid: dev • dotted: fit',frameon=False)
    for j,(basis,name,col) in enumerate([('updates','Same updates','#285f8a'),('cpu','Same CPU budget','#8b4d7a')]):
        xx=x+(j-.5)*.08;g=[by[basis][s]['gap'] for s in STEPS];m=np.array([v['mean'] for v in g]);limits=np.array([v['parent_ci95'] for v in g]);bx.errorbar(xx,m,yerr=np.abs(limits.T-m),fmt='o-',capsize=3,label=name,color=col);bx.scatter(np.repeat(xx,3),np.array([v['seed_means'] for v in g]).ravel(),s=11,alpha=.55,color=col)
    bx.axhline(0,color='0.6',linewidth=.8);bx.set(title='Direct minus contrastive',ylabel='paired dev gain (nat / face)');bx.legend(frameon=False)
    for a in (ax,bx):a.set_xticks(x,['5,120','10,240','20,480']);a.set_xlabel('Registered update endpoint');a.grid(axis='y',alpha=.18)
    fig.suptitle('Training diagnostic only • 4,096 fit / 512 dev parents • three paired starts');fig.savefig(e.OUT/'21-convergence.png',dpi=160);plt.close(fig);print(json.dumps(changes),flush=True)

if __name__=='__main__':run()
