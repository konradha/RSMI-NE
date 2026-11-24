"""Localize the already-measured nonlinear tripod score using only archived parent counts.
The prospective cell contrast is secondary: the aggregate validation results are known.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr
from scipy.stats import t

import tripod_channel as channel
from rate import digest, json_out


def source(path):
    return {r['source']['index']:r for r in json.loads((path/'results.json').read_bytes())['temperatures']}


def tables(path, src, index):
    if index not in (9, 7, 0):raise ValueError('only previously scored diagnostic temperatures allowed')
    f=path/f'parents-{index:02d}.npz'
    if digest(f)!=src[index]['counts_sha256']:raise ValueError('parent counts changed')
    return np.load(f,allow_pickle=False)


def fingerprint():
    return {f.name:digest(f) for f in [Path(__file__),Path(channel.__file__)]}


def freeze(inp,out):
    f=out/'cell-lock.json'
    if f.exists():raise FileExistsError('cell already locked')
    src=source(inp)
    with tables(inp,src,9) as bank:train=bank['train']  # never request validation or warm cells here
    total=train.sum(0,dtype='i8');n=total[...,0];model=channel.fit(total);q=model['registered_pairs'];p=model['unrestricted'];support=(train[...,0]>0).sum(0)
    merit=(rel_entr(p,q)+rel_entr(1-p,1-q))*n/n.sum();i,k=map(int,np.unravel_index(np.argmax(np.where(support>=512,merit,-np.inf)),n.shape));diff=float(p[i,k]-q[i,k]);
    if not diff or support[i,k]<512:raise ValueError('no measurable cell')
    plan=dict(study='One secondary arrangement-specific prediction; not a rescue of the original failed transfer. Validation aggregates known; cell chosen on cold train only.',code=fingerprint(),source_digest=digest(inp/'results.json'),parent_digests={str(k):r['counts_sha256'] for k,r in src.items()},index=9,stratum=i,cell=k,species=['vacancy','A','B'][i//7],occupied_neighbors=i%7,parallel_vacant=bool(k//3),transverse_vacancies=k%3,train_cell_fraction=float(n[i,k]/n.sum()),support_parents=int(support[i,k]),selection='largest cold-training frequency-weighted Bernoulli KL(unrestricted || registered pairs), among cells seen in at least 512 distinct training parents. No ranking on validation. Ties in species/coordination/cell index order.',training_pair_A=float(q[i,k]),training_unrestricted_A=float(p[i,k]),sign=int(np.sign(diff)),threshold=.5*abs(diff),primary_temperature_index=7,prediction='At beta=2.3722, in this exact cell, observed outer A minus its temperature-specific registered-pair forecast retains the cold sign and at least half its cold training magnitude. Freeze the rule and model family, not a temperature-independent coupling constant.',estimate='Binomial observed A rate in selected cell minus its pair-logit forecast at each temperature. Fixed cold-selected cell and margin. Cold validation and beta=0 descriptive. Validation parents and 32 deleted training-parent groups for joint uncertainty; 95% primary interval. No use of index 6, test folds or dynamics.')
    json_out(f,plan);print(json.dumps({k:plan[k] for k in ('species','occupied_neighbors','parallel_vacant','transverse_vacancies','sign','threshold','train_cell_fraction')}),flush=True)


def interval(value,val_deletions,train_deletions):
    def jack(d):return (len(d)-1)*np.mean((d-d.mean())**2)
    a,b=jack(val_deletions),jack(train_deletions);s=(a+b)**.5
    df=(a+b)**2/(a*a/(len(val_deletions)-1)+b*b/(len(train_deletions)-1)) if s else len(val_deletions)-1
    half=s*t.ppf(.975,df);return dict(value=float(value),lo95=float(value-half),hi95=float(value+half),se=float(s),fit_se=float(b**.5),validation_se=float(a**.5))


def effects(count,q,u,i,k):
    n,a=count[...,0],count[...,1];nc,ac=n[...,i,k],a[...,i,k];tot=n.sum(axis=(-2,-1))
    if np.any(nc<=0):raise ValueError('cell not supported; do not replace with another')
    gain=a*np.log(u/q)+(n-a)*np.log((1-u)/(1-q));cell=gain[...,i,k]/tot;full=gain.sum(axis=(-2,-1))/tot
    return np.stack((ac/nc-q[i,k],cell,full-cell,full),axis=-1)


def score(inp,out):
    fn=out/'cell-results.json';lock=out/'cell-lock.json'
    if fn.exists():raise FileExistsError('refusing to score the locked geometry twice')
    p=json.loads(lock.read_bytes());src=source(inp);i,k=p['stratum'],p['cell']
    if p['code']!=fingerprint() or p['source_digest']!=digest(inp/'results.json'):raise ValueError('locked implementation or counts changed')
    rr=[]
    for idx in (9,7,0):
        with tables(inp,src,idx) as z:train,val,tids,vids=(z[j] for j in ('train','validation','train_rows','validation_rows'))
        if len(train)!=10240 or len(val)!=2048 or np.intersect1d(tids,vids).size:raise ValueError('invalid parent split')
        total=train.sum(0,dtype='i8');v=val.sum(0,dtype='i8');m=channel.fit(total);q,u=m['registered_pairs'],m['unrestricted'];support=int((val[:,i,k,0]>0).sum())
        if support<64:rr.append(dict(index=idx,status='insufficient parent support',parents_with_cell=support));continue
        point=effects(v,q,u,i,k);vd=effects(v-val,q,u,i,k);td=[]
        for group in np.array_split(np.arange(len(train)),32):
            m=channel.fit(total-train[group].sum(0,dtype='i8'));td.append(effects(v,m['registered_pairs'],m['unrestricted'],i,k))
        if abs(point[3]-src[idx]['contrasts']['unrestricted_over_registered_pairs']['nats'])>2e-12:raise ValueError('cell attribution failed to reproduce total published score')
        sts={key:interval(point[j],vd[:,j],np.array(td)[:,j]) for j,key in enumerate(('residual_A','cell_gain','other_gain','total_gain'))};orient=interval(p['sign']*point[0],p['sign']*vd[:,0],p['sign']*np.array(td)[:,0]);n,a=v[i,k];rr.append(dict(index=idx,beta=src[idx]['source']['beta'],status='scored',parents_with_cell=support,validation_parents=len(val),cell_fraction=float(n/v[...,0].sum()),actual_A=float(a/n),pair_A=float(q[i,k]),signed=orient,statistics=sts,decision=('supports locked magnitude' if orient['lo95']>p['threshold'] else 'excludes locked magnitude' if orient['hi95']<p['threshold'] else 'unresolved at locked magnitude') if idx==7 else 'descriptive'))
    json_out(fn,dict(plan_sha256=digest(lock),temperatures=rr,interpretation='Gain decomposition is additive in nats per exterior site, not a decomposition of true MI. This cell-specific test compares to a pair logit given center species and coordination, not to all conventional models or to the barrier-inclusive model. Selection was frozen before this cell-specific comparison, not before the aggregate data existed.'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,(left,right)=plt.subplots(1,2,figsize=(9.4,3.5),layout='constrained');rows=sorted([r for r in rr if r['status']=='scored'],key=lambda r:r['beta']);xx=[r['beta'] for r in rows]
    for ax,keys,scale in [(left,[('signed','Measured minus pair, cold sign')],100),(right,[('cell_gain','Selected arrangement'),('other_gain','All other arrangements')],1e6)]:
        for j,(key,label) in enumerate(keys):
            vals=[r['signed'] if key=='signed' else r['statistics'][key] for r in rows];y=np.array([v['value'] for v in vals]);lo=np.array([v['lo95'] for v in vals]);hi=np.array([v['hi95'] for v in vals]);ax.errorbar(np.array(xx)+(.035*(2*j-1) if len(keys)>1 else 0),scale*y,yerr=scale*np.stack((y-lo,hi-y)),fmt=('o' if j==0 else 's'),capsize=3,label=label)
        ax.axhline(0,color='.6',lw=.8);ax.set(xticks=xx,xticklabels=[f'{v:.2f}' for v in xx],xlabel='Inverse temperature β');ax.spines[['right','top']].set_visible(False)
    left.axhline(100*p['threshold'],color='#a9503d',ls='--',label='Locked warm prediction');left.set(ylabel='Percentage points',title='Directional outer-A miscalibration');right.set(ylabel='Gain (µnats / exterior site)',title='Where the nonlinear gain resides')
    for ax in (left,right):ax.legend(fontsize=7,frameon=False)
    fig.savefig(out/'cell.png',dpi=160);plt.close(fig);print([(r['index'],r.get('decision'),r.get('signed')) for r in rr],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('freeze','score'));parser.add_argument('--counts',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True);(freeze if a.mode=='freeze' else score)(a.counts,a.out)
