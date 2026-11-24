"""Summarize archived scores after release; never open the audit again."""
import json
from tranche2_hierarchy import OUT, WORK, SEEDS, Kernels, load, dump, intervals
import numpy as np


def data(filename):return json.loads((OUT/filename).read_text())


def main():
    lock=data('25-lock.json');audit=data('25-audit.json');aux=data('25-rawlinear-development.json');auxaudit=data('25-rawlinear-audit.json');step=lock['checkpoint'];r=4096
    base=np.load(OUT/f'25-model-frozen-251-{step}.npz')['base'];counts=base*(r*1000+40.5)-.5
    assert np.allclose(counts,counts.round(),atol=1e-8,rtol=0)
    # Only development face frequencies are needed; audit is not read by this program.
    _,yd,*_=load(Kernels(WORK),4096,4608)
    fitbase=float(counts@np.log(base)/counts.sum());devbase=float(np.log(base[yd]).mean());arms={};rungs={}
    for n in (256,1024,4096):
        ck=step if n==r else 4;c=data(f'25-curves-{n}-{ck}.json');v=np.load(OUT/f'25-dev-parents-{n}-{ck}.npz');per={}
        for arm in ('raw','frozen','joint'):
            last=[c['seeds'][str(s)]['arms'][arm][ck] for s in SEEDS];fit=float(np.mean([a['fit_score'] for a in last]));d=intervals(np.mean([v[f'{arm}_{s}_{ck}'] for s in SEEDS],0))
            per[arm]={'fit_gain':fit,'development_gain':d,'timed_post_init_seconds_per_seed':float(np.mean([a['elapsed_s'] for a in last]))}
            if n==r:
                arms[arm]={'fit_absolute_log_score':fitbase+fit,'dev_absolute_log_score':devbase+d['mean'],'fit_gain':fit,'development_gain':d,'audit_gain':audit[arm],
                           'audit_geometric_mean_true_face_probability_ratio_vs_marginal':float(np.exp(audit[arm]['mean']))}
        rungs[n]={'full_batch_faces':n*1000,'fit_development_wall_seconds':c['elapsed_seconds'],'raw_suffstat_seconds_shared_across_seeds':c['raw_suffstat_seconds'],
                  'paired_seed_mean_difference':c['paired_seed_mean'][str(ck)],'arms':per}
    last=[aux[str(r)]['seeds'][str(s)][-1] for s in SEEDS];fit=float(np.mean([a['fit_score'] for a in last]));d=lock['diagnostic']['dev'];a=auxaudit['scores']['mean']
    arms['rawlinear']={'fit_absolute_log_score':fitbase+fit,'dev_absolute_log_score':devbase+d['mean'],'fit_gain':fit,'development_gain':d,'audit_gain':a,
                       'audit_geometric_mean_true_face_probability_ratio_vs_marginal':float(np.exp(a['mean']))}
    c=data(f'25-curves-4096-{step}.json');hard={}
    for s in SEEDS:
        start=c['seeds'][str(s)];f=start['arms']['joint'][-1]
        hard[s]={'greedy_child_full_fit_iterations':len(start['greedy_child_training']),'greedy_child_final_own_face_fit_nats':start['greedy_child_training'][-1]['hard_fit'],
                 'initial_final_training_joint_equals_frozen':bool(start['arms']['joint'][0]['fit_score']==start['arms']['frozen'][0]['fit_score']),
                 'realized_joint_child_mass_changed':f['child_fit_mass_changed_since_greedy'],'final_child_freq_fit':f['child_frequencies'],'final_H_dev_bits':f['dev_entropy_bits'],
                 'actual_final_child_gradient_rms':f['finite_child_grad_rms'],'paired_audit':audit['paired_seeds'][str(s)]}
    result={'question':'#25 joint credit assignment across two hard categorical maps','design':'1036785','audit_release':audit['release'],
            'source':json.loads((WORK/'b09.json').read_text()),'margin_nats_per_face':.001,'fit_marginal_log_score':fitbase,'dev_marginal_log_score':devbase,'largest_rung':arms,
            'rungs':rungs,'largest_rung_seed_diagnostics':hard,'final_joint_minus_frozen':audit['joint_minus_frozen'],'extension_rule_observed':lock['extension_trigger_dev_change_2_to_4'],
            'stopping':'Endpoint locked at four full-data sweeps for all arms/seeds. The preregistered extension required >=0.0001 nat/face dev increase (2->4) in either hierarchy arm; neither qualified. Shared first code stabilized in all three own-face fits. No convergence theorem for joint discrete optimization.',
            'disposition':'Primary raw reference and lower-variance raw diagnostic both have strictly positive audit lower bounds. Joint training changed the actual shared child code and raised fit score, but did not reproduce a useful held-apart improvement; 95% upper endpoint below +0.001. All three realized paired audit estimates negative but zero is inside the prespecified seed-mean interval.',
            'uncertainty':'Paired per-parent averaging across 1,000 correlated anchors and across three fixed seeds; normal 95% interval over 512 disjoint (chain_seed,chain_id) parents, MDE≈2.8 SE (power 80% at two-sided 5%). Seed scatter descriptive; no independent-anchor or independent-seed inflation. Fit/development/audit split of historically reused TRAIN, not native unseen population.',
            'limits':'No optimality or universal hierarchy verdict, no phase or scale-invariance claim. Conditional on fixed geometry, seeds, empirical-Bayes smoothing and hard-coordinate training. Top updates optimize fit with shared decoder; joint extra child work costs more time than frozen, so the comparison is update/anchor matched, not equal CPU seconds. Log-score gain over fit marginal is predictive and is not exact mutual information.',
            'citation':'Lenggenhager, Goekmen et al., Phys. Rev. X 10, 011037 (2020), https://doi.org/10.1103/PhysRevX.10.011037'}
    dump('25-disposition.json',result)
    print(json.dumps({'fit_marginal':fitbase,'dev_marginal':devbase,'largest_rung':{a:[p['fit_absolute_log_score'],p['dev_absolute_log_score'],p['fit_gain'],p['development_gain']['mean'],p['audit_gain']['mean']] for a,p in arms.items()}},indent=2))


if __name__=='__main__':main()
