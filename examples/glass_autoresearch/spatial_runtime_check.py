"""Fail closed on packing inference across sklearn serializers; never read cycle outcomes."""
import argparse
import json
import warnings

from spatial_raw import WORK, PUBLIC, packing_offsets, load, np, cm
import sklearn


def check(which):
    path=WORK/'calibration.npz'; expected=json.loads((PUBLIC/'bank-calibration.json').read_bytes())['bank_sha256']
    if cm.digest(path)!=expected: raise ValueError('changed bank')
    with np.load(path,allow_pickle=False) as z:
        s=z['stratum']; m=np.isin(s,np.load(WORK/'eligible.npy',allow_pickle=False)); sh=z['shell'][m]; s=s[m]
    result={}; records=json.loads((PUBLIC/'packing-fit.json').read_bytes())['fits']
    for st in (512,2048):
        model=WORK/records[str(st)]['model']; assert cm.digest(model)==records[str(st)]['model_sha256']; o=load(model)
        result[str(st)]=np.concatenate([np.column_stack(packing_offsets(o,sh[j:j+2048],s[j:j+2048])) for j in range(0,len(s),2048)])
    old=WORK/'calibration-sklearn-1.8-logits.npz'
    if which=='original-calibration-runtime':
        assert sklearn.__version__=='1.8.0'
        with old.open('xb') as f:np.savez(f,**result)
    else:
        assert sklearn.__version__=='1.9.1'; diff={}
        with np.load(old,allow_pickle=False) as z:
            for key,v in result.items(): diff[key]=float(np.max(np.abs(z[key]-v)))
        passed=max(diff.values())<=1e-10
        info=dict(calibration_runtime='1.8.0',fit_and_dev_runtime=sklearn.__version__,eligible_calibration_opportunities=len(s),parents=m.shape[0],diagnostic='Compare the stratum and packing logits for all eligible calibration inputs at both fit sizes. No cycle outcomes, no probabilities rescored, no alternate fit. Earlier cross-version loading emitted a warning.',largest_abs_logit_discrepancy=diff,required_maximum=1e-10,passed=passed)
        with (PUBLIC/'packing-runtime-check.json').open('x') as f:json.dump(info,f,indent=2);f.write('\n')
        if not passed:raise SystemExit('Cross-version numerical ambiguity; development blocked.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('which',choices=('original-calibration-runtime','fit-runtime'));check(p.parse_args().which)
