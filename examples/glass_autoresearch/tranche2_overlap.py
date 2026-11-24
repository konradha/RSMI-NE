"""Reference-to-reference overlap variance from two independent baths (archive only)."""
import argparse, itertools, json
from pathlib import Path
import numpy as np
from scipy.stats import t
from face_bank import code
from tile_codec import TILE, POW
from rate import digest


def spectra(s):
    # Cross spectra / N. An inner product using Hermitian multiplicities yields
    # translation-averaged squared *extensive* centered overlap, without listing translations.
    axes=tuple(range(1,s.ndim)); n=np.prod(s.shape[1:]); f=[np.fft.rfftn(s==i,axes=axes)/np.sqrt(n) for i in range(3)]
    for x in f:x[(slice(None),)+(0,)*len(axes)]=0
    mult=np.full(f[0].shape[-1],2.);mult[0]=1
    if s.shape[-1]%2==0:mult[-1]=1
    features=[]
    for i in range(3):
        for j in range(i,3):
            xy=f[i].conj()*f[j]*np.sqrt(mult*(1 if i==j else 2));features.append(xy.real)
            if i!=j:features.append(xy.imag)
    return np.stack(features,axis=1).reshape(len(s),-1)


def control():
    states=np.array(sorted(set(itertools.permutations([0,1,1,2,2,2]))));r=states[3:10];x=spectra(states);v=spectra(r)@x.mean(0);brute=np.array([(states==a).sum(1).var() for a in r]);p=np.array([1,2,3])/6;cov=np.diag(p)-np.outer(p,p);exact=np.square(cov).sum()*36/5
    # Explicit arbitrary independent bath parent, averaged over its translations.
    actual=spectra(states[11:12])@spectra(states[14:15]).T;expected=np.var([(states[11]==np.roll(states[14],j)).sum() for j in range(6)]);err=max(abs(v-brute).max(),abs(v-exact).max(),abs(actual.item()-expected));
    if err>1e-12:raise ArithmeticError(err)
    return {'exhaustive_parents':len(states),'maximum_absolute_error':err,'tiny_null_variance':exact}


def physical(s,lut):
    occ=s!=0; m=sum(np.roll(occ,a,axis=k).astype('i1') for k in (1,2,3) for a in (-1,1)); energy=(occ*(m-np.where(s==1,3,5))**2).mean((1,2,3));coord=[((m==k)&occ).mean((1,2,3)) for k in range(7)];grid=np.indices(s.shape[1:]);pi=[]
    for b in itertools.product((0,1),repeat=3):
        if any(b):pi.append((np.einsum('ijk,nijk->n',(-1.)**np.einsum('i,ijkl->jkl',b,grid),~occ)/8000)**2)
    h=lut[code(s,TILE,POW)];return np.stack([energy,*coord,np.max(pi,axis=0),*[(h==k).mean((1,2,3)) for k in range(3)]],axis=1)


def features(data,n_bath,n_ref,groups,lut):
    width=2*n_bath+n_ref;out=[]; xs=[];pids=[]
    for i in range(groups):
        block=data[i*width:(i+1)*width]; refs=block[-n_ref:];ref=spectra(refs);bath=[]
        for row in (block[:n_bath],block[n_bath:2*n_bath]):
            # 8-parent chunks bound memory at the cold rung.
            means=sum(spectra(row[j:j+8]).sum(0) for j in range(0,n_bath,8))/n_bath;bath.append(ref@means)
        out.append(np.stack(bath,axis=-1));xs.append(physical(refs,lut));pids.append(np.arange(i*width+2*n_bath,(i+1)*width))
    return np.stack(out),np.stack(xs),np.stack(pids)


def summarize(v):
    a,b=v[:,:,0],v[:,:,1];c=((a-a.mean(1)[:,None])*(b-b.mean(1)[:,None])).sum(1)/(a.shape[1]-1);m=v.mean((1,2));g=len(c);mc=c.mean();mm=m.mean();r=mc/mm**2;influence=(c-mc)/mm**2-2*r*(m-mm)/mm;crit=t.ppf(.975,g-1);hw=crit*influence.std(ddof=1)/np.sqrt(g);target=.0001;score=c-target*mm**2-2*target*mm*(m-mm);lo_test=score.mean()-crit*score.std(ddof=1)/np.sqrt(g)
    return dict(variance_mean=mm,reference_variance_signed=mc,relative_variance_signed=r,ci95_relative_variance=[r-hw,r+hw],threshold_percent_rms=1.,threshold_contrast_lower=lo_test,gate=bool(lo_test>0),blocks=g,bath_parents_each=v.shape[1]*0,reference_parents=g*v.shape[1],relative_rms_estimate=np.sqrt(r) if r>=0 else None)


def reg_fit(x,y):
    x=x.reshape(-1,x.shape[-1]);y=y.ravel();mu=x.mean(0);sig=x.std(0);sig[sig<1e-12]=1.;xx=(x-mu)/sig;coef=np.linalg.solve(xx.T@xx+.01*len(x)*np.eye(len(mu)),xx.T@(y-y.mean()));return dict(mean=mu.tolist(),scale=sig.tolist(),coef=coef.tolist(),intercept=y.mean())


def score(x,v,weights,scale):
    preds=[]
    for w in weights:
        dim=len(w['coef']);preds.append((x[:,:,:dim]-w['mean'])/w['scale']@w['coef']+w['intercept'])
    def loss(p):return (p-v[:,:,0])*(p-v[:,:,1])
    block=(loss(preds[0])-loss(preds[1])).mean(1)/scale;hw=t.ppf(.975,len(block)-1)*block.std(ddof=1)/np.sqrt(len(block));return dict(gain_fraction_of_fit_variance=block.mean(),ci95=[block.mean()-hw,block.mean()+hw],group_fractions=block.tolist(),scale='signed fit cross-bath variance; fixed pre-audit')


def run(base,out,mode,lock):
    from time import monotonic
    start=monotonic();tiny=control();repo=Path(__file__).resolve().parents[2];f=repo/'results/l20-autoresearch/fifteen/09-witness.npz';lut=np.load(f)['maps'][0]; fd=out/'24-predictor.json';report={};evidence={}
    if mode=='audit' and (not lock or digest(fd)!=lock):raise ValueError('requires published predictor digest')
    for idx,stage,lo,nb,nr,g in ([(9,'fit',0,48,32,32),(0,'hot',0,12,8,8)] if mode=='fit' else [(9,'dev' if mode=='dev' else 'audit',4096 if mode=='dev' else 4608,24,16,8)]):
        with np.load(base/f'b{idx:02}.npz') as z:
            end=lo+(2*nb+nr)*g;s=z['species'][lo:end];pid=z['parent'][lo:end]
        v,x,sel=features(s,nb,nr,g,lut);res=summarize(v);res['bath_parents_each']=nb*g;res['parents_total']=len(pid);res['fit_train_positions']=[lo,end];report[stage]=res;evidence.update({stage+'_v':v,stage+'_x':x,stage+'_reference':pid[sel],stage+'_all_parents':pid});
        if stage=='fit' and res['gate']:
            weights=[reg_fit(x[:,:,:k],v.mean(2)) for k in (9,12)];(out/'24-fit-weights.json').write_text(json.dumps(dict(models=weights,scale=res['reference_variance_signed']),indent=2)+'\n')
        elif stage in ('dev','audit'):
            spec=json.loads((out/('24-fit-weights.json' if stage=='dev' else '24-predictor.json')).read_text());res['prediction']=score(x,v,spec['models'],spec['scale']);
            if stage=='dev':spec.update(pre_audit_prediction='Additional three code frequencies improve mean squared error by at least 1% of the measured fit reference variance.',dev_measurement=res['prediction'],freeze_conventional_dim=9,fit_only=True);fd.write_text(json.dumps(spec,indent=2)+'\n')
    p=np.array([.25,.30,.45]);S=np.diag(p)-np.outer(p,p);full=dict(results=report,numerical_control=tiny,hot_exact_variance=np.sum(S*S)*8000**2/7999,hot_exact_reference_variation=0,units='Extensive number of matching species including vacancies; variance in match-count squared',map_sha256=digest(f),design_sha256=digest(out/'18-24-design.json'),elapsed_seconds=monotonic()-start,audit_predictor_sha256=lock);np.savez_compressed(out/f'24-{mode}-parents.npz',**evidence);(out/f'24-{mode}.json').write_text(json.dumps(full,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('banks',type=Path);p.add_argument('out',type=Path);p.add_argument('mode',choices=['fit','dev','audit']);p.add_argument('--predictor-sha256');a=p.parse_args();run(a.banks,a.out,a.mode,a.predictor_sha256)
