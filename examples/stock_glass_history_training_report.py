"""Report actual receipts only: last attested Adam step is a lower bound after external termination."""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'results/history-overlap-field'

def read(p):return json.loads(p.read_bytes()) if p.is_file() else {}

def csvfile(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def report(out):
    ck=out/'checkpoints';prov=read(out/'provider.json');res=read(out/'reservation.json');term=read(out/'terminated.json');bill=read(out/'billing.json');depth=read(ck/'depth-decision.json');error=read(out/'gpu-failure.json');assert prov and term and term['sandbox']==prov['sandbox']
    rows=[];curves=[];uu=set();device=set();first=[];terminal=[]
    for seed in (6101,6102,6103):
        for arm in ('true','displaced'):
            stem=f'{arm}-s{seed}';comp=read(ck/(stem+'-final.json'));fail=read(ck/(stem+'-failure.json'));log=read(ck/(stem+'-progress.json'));fst=read(ck/(stem+'-first.json'));p=comp or fail or log or fst
            steps=p.get('adam_iterations');elapsed=p.get('elapsed_s');eval_s=p.get('checkpoint_evaluation_and_wait_s');runtime=(elapsed-eval_s) if elapsed is not None and eval_s is not None else None
            if p.get('uuid'):uu.add(p['uuid']);device.add(p['device'])
            if fst:first.append(fst['first_cuda_monotonic'])
            if (comp or fail) and p.get('now_monotonic'):terminal.append(p['now_monotonic'])
            rows.append(dict(pairing=arm,seed=seed,pid=p.get('pid'),first_cuda_update=fst.get('adam_iterations'),last_attested_adam_update=steps,complete=bool(comp.get('complete')),minimum_reached=bool(steps is not None and steps>=32768),kernel_and_input_pipeline_side_s=runtime,actual_updates_per_worker_s=steps/runtime if runtime and steps else None,checkpoint_and_wait_s=eval_s,fit_elapsed_s=elapsed,fit_parents=p.get('parents') if steps else None,positive_anchor_draws=p.get('positives'),distinct_origins_per_fit_parent=p.get('unique_origins_per_parent'),parent_stream_sha256=p.get('stream_sha256'),gpu_uuid=p.get('uuid'),failure=fail.get('error')))
            rows[-1].update(exact_terminal_count=bool(comp or fail),post_first_step_rate_s=(steps-1)/(p['now_monotonic']-fst['first_cuda_monotonic']) if fst and steps and steps>1 and p['now_monotonic']>fst['first_cuda_monotonic'] else None)
            for step in (8192,16384,32768,65536):
                z=read(ck/(stem+f'-u{step}.json'))
                if not z:continue
                for evalarm in ('true','displaced'):
                    v=z.get('shell',{}).get(evalarm,{});m=v.get('means_bits') or [None]*6
                    curves.append(dict(pairing=arm,seed=seed,update=step,evaluated_shell=evalarm,valid=z.get('valid'),weights_adam_finite=z.get('weights_and_adam_finite'),stock_real_bits=m[0],stock_shuffle_bits=m[1],stable_real_bits=m[2],stable_shuffle_bits=m[3],true_minus_crossparent_bits=v.get('stable_gain_bits'),hard_entropy_bits=z.get('entropy_bits'),code_counts=z.get('code_counts'),dev_parents=z.get('parents'),panels=z.get('panels')))
    csvfile(out/'training-by-seed.csv',rows)
    if curves:csvfile(out/'dev-critic-checkpoints.csv',curves)
    observed=sum(r['last_attested_adam_update'] or 0 for r in rows);nonfinal=any(not r['complete'] for r in rows);wall=term['elapsed_local_s'];usd=Decimal(str(wall))*Decimal(res['hourly_usd'])/3600;eg=Decimal(str(term.get('download_bytes') or 0))/2**30*Decimal(res['egress_per_GiB_usd']);tele=read(ck/'telemetry.json');util=[];mem=[]
    for t in tele:
        parts=t['gpu'].split(',');util.append(float(parts[2].strip()));mem.append(float(parts[3].strip()))
    # A non-final progress receipt is a lower bound, not proof that the process exited then.
    wave=read(ck/'wave.json');early_wave_stop='worker failure' in wave.get('error','')
    overlapping=bool(len(first)==6 and terminal and max(first)<min(terminal) and (early_wave_stop or len(terminal)==6))
    reported=bill.get('gross_usd');gross=reported if reported is not None else str(round(usd+eg,6))
    receipt=dict(app=prov['app'],sandbox=prov['sandbox'],portal=prov['portal'],device=sorted(device),device_uuid=sorted(uu),
                 allocation_utc=prov['utc'],termination_utc=term['termination_utc'],elapsed_local_s=wall,hard_ttl_s=res['seconds'],max_all_in_usd=res['max_usd'],
                 cost_usd=gross,cost_kind='Provider billing report for app; workspace egress separate' if reported is not None else 'Estimated only: conservative local allocation wall × reserved resources and tariff, plus downloaded outputs at egress tariff; not an invoice.',
                 provider_app_charge_usd=reported,fallback_resource_estimate_usd=str(round(usd,6)),payload_egress_estimate_usd=str(round(eg,8)),
                 aggregate_attested_adam_updates=observed,all_terminal_counts_observed=all(r['exact_terminal_count'] for r in rows),first_cuda_workers=len(first),
                 overlap_of_all_six_observed=overlapping,min_update_criterion_met=all(r['minimum_reached'] for r in rows),
                 max_gpu_util_pct=max(util) if util else None,mean_sampled_gpu_util_pct=sum(util)/len(util) if util else None,max_sampled_used_MiB=max(mem) if mem else None,
                 telemetry_samples=len(util),depth_decision=depth,provider_error=error,per_worker=rows)
    (out/'gpu-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if curves:
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for se,co in zip((6101,6102,6103),('#18798a','#a0522d','#795ab1')):
            for ar,ls in (('true','-'),('displaced','--')):
                sub=[v for v in curves if v['seed']==se and v['pairing']==ar and v['evaluated_shell']==ar];sub=sorted(sub,key=lambda v:v['update'])
                if not sub:continue
                for ax,key in zip(axes,('true_minus_crossparent_bits','hard_entropy_bits')):ax.plot([v['update'] for v in sub],[v[key] for v in sub],ls=ls,color=co,marker='o',label=f'{se} {ar}')
        for ax in axes:ax.set(xlabel='Actual Adam update',xscale='log',xticks=[8192,16384,32768,65536],xticklabels=['8,192','16,384','32,768','65,536']);ax.grid(alpha=.18)
        axes[0].set(ylabel='Real − cross-parent shuffled retrieval (bits)',title='Frozen relaxed critic · own shell pairing');axes[1].set(ylabel='Entropy (bits, maximum 2)',title='Hard four-label distribution',ylim=(-.07,2.07));axes[0].legend(fontsize=7);fig.suptitle('Source DEV diagnostics · not the held conditional physical gain');fig.savefig(out/'dev-training.png',dpi=155);plt.close(fig)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('per_worker','depth_decision')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();report(a.out)
