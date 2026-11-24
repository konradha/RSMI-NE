"""04: intervene on Hamiltonian terms, never on empirical boundary weights or frozen fits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from stock_glass_cavity_exact import C,U,X,O,M,R,D,J,L,A,prob
from stock_glass_cavity_memory import evaluate,interval,add
import stock_glass_modal as stock

ROOT=Path(__file__).resolve().parents[1]; DESIGN=ROOT/'results/experiment-04-boundary-couplings/design.json'
SEEDS=(6101,6102,6103); NAMES=('vacancy','linear',*(f'hard_{s}' for s in SEEDS))
# Field switch, interior-centered pair strength, exterior-centered pair strength.
LAWS={'intrinsic':(0,0,0),'fields':(1,0,0),'joint25':(1,.25,.25),'joint50':(1,.5,.5),'joint75':(1,.75,.75),'full':(1,1,1),'in_only':(1,1,0),'out_only':(1,0,1)}
H0=(O*R**2).sum(1,dtype='i4'); OCC=O.astype('i4'); ADJ=D.astype('i4'); PAIRS=ADJ*(ADJ-1)


def terms(z):
    d=z['fc'].astype('i4');su=z['su'];a=z['fu'].astype('i4');n=(su!=0).astype('i4');assert not d[:,0].any()
    field=(d*d)@OCC.T-2*d@(OCC*L[X]).T+(n*(2*(a-L[su])+1))@ADJ.T
    inside=2*d@(OCC*M).T;outside=n@PAIRS.T
    return field,inside,outside


def law_p(parts,law,beta):
    field,jin,jout=parts
    if law=='intrinsic':return np.broadcast_to(prob(H0[None,:],beta),(len(field),len(H0)))
    _,i,o=LAWS[law];energy=H0+field
    if i:energy=energy+i*jin
    if o:energy=energy+o*jout
    return prob(energy,beta)


def mean_if(x):
    d=x-x[0];return float(x[0]+d.mean()),d-d.mean()


def diff(x,y):return x[0]-y[0],x[1]-y[1]


def scalar(p,f):
    """Influences of population variances and their ratio on the same parent roster."""
    n=len(p)
    if f is None or np.ptp(f)==0:
        zero=(0.,np.zeros(n));return {'between':zero,'total':zero,'T':(np.nan,np.full(n,np.nan))},np.zeros((n,2))
    u=p@f;centered=u-u[0];centered-=centered.mean();mu=u[0]+np.mean(u-u[0]);v=p@((f-mu)**2)
    between=mean_if(centered**2);total=mean_if(v);assert total[0]>0;T=between[0]/total[0];assert 0<=T<=1+1e-12
    return {'between':between,'total':total,'T':(T,(between[1]-T*total[1])/total[0])},np.stack([u,v],axis=1)


def js(p,q):
    # rel_entr handles zero masses without arbitrary pseudocounts.
    mix=(p+q)*.5;j=(rel_entr(p,mix)+rel_entr(q,mix)).sum(axis=1)*(.5/np.log(2.));assert np.min(j)>=-2e-12 and np.max(j)<=1+2e-12
    return np.clip(j,0.,1.)


def inputs(a):
    design=json.loads(DESIGN.read_bytes());pins=design['reference_files_sha256'];assert all(stock.sha(a.reference/k)==v for k,v in pins.items());meta=json.loads((a.banks/'prepared.json').read_bytes());assert stock.sha(a.banks/'prepared.json')==pins['parent-census.json']
    for name,v in meta['inputs'].items():
        if name.endswith(('-train.npz','-test.npz')) or name=='state-geometry.npz':assert stock.sha(a.banks/name)==v
    with np.load(a.banks/'state-geometry.npz') as g:
        for k,v in [('states',X),('star',C),('touch',U),('inner_edges',A),('cavity_to_outer',J),('ell',L)]:np.testing.assert_array_equal(g[k],v)
    assert sorted(J.sum(1))==[1]*6+[2]*12
    for ids in [np.flatnonzero(row) for row in J if row.sum()==2]:assert 0 not in ids and C[ids[0]]@C[ids[1]]==0
    a.out.mkdir(parents=True,exist_ok=True);return meta


def operators(b,mode,maps):
    tag=f'b{b:02}';depth=10218 if b==9 else 10257;fs={'vacancy':(X[:,0]==0).astype('d'),'linear':mode[tag+'_linear']};projections={}
    for seed in SEEDS:
        k=tag+f'-s{seed}_d{depth}_hard';q=maps[k+'_Q'];a=maps[k+'_a'];np.testing.assert_array_equal(q,np.eye(4)[q.argmax(1)]);name=f'hard_{seed}';projections[name]=q
        if np.isnan(a).all():assert np.unique(q.argmax(1)).size==1;fs[name]=None
        else:assert np.isfinite(a).all();fs[name]=q@a
    return fs,projections


def validate_values(rows,b,what,estimated,expected):
    x=np.asarray(estimated);y=np.asarray(expected);np.testing.assert_array_equal(np.isnan(x),np.isnan(y));np.testing.assert_allclose(x,y,rtol=0,atol=1e-9,equal_nan=True);rows.append(dict(index=b,observable=what,defined=bool(np.isfinite(y[0])),T=x[0],published=y[0],lower=x[1],published_lower=y[1],upper=x[2],published_upper=y[2],max_error=float(np.max(abs(x-y))) if np.isfinite(y[0]) else None))


def audit(a,meta,mode,maps):
    energ=[];lineage=set();full=[];null=[];static=pd.read_csv(a.reference/'benchmark.csv');rsmi=pd.read_csv(a.reference/'corrected/frozen-readout.csv');rsmi=rsmi[rsmi.final];max_p=0.
    # Gate the entire corpus before computing any response statistic.
    for r in meta['coverage']:
        b=r['index'];tag=f'b{b:02}'
        for side in ('train','test'):
            with np.load(a.banks/f'{tag}-{side}.npz') as z:
                ids=z['all_ids'];lineage.update(map(tuple,ids.tolist()));sel=z['selected'];assert len(ids)==8192 and len(sel)==r[side]['selected_parents'] and np.array_equal(sel,np.flatnonzero(z['eligible_counts']))
                parts=terms(z);absolute=H0+sum(parts);er=absolute-absolute.min(1,keepdims=True)-z['relative_E'];bad=np.count_nonzero(er);assert bad==0,(b,side,bad)
                energ.append(dict(index=b,beta=float(r['beta']),side=side,roster=len(ids),eligible=len(sel),comparisons=er.size,errors=int(bad),max_integer_error=int(abs(er).max())))
    assert len(lineage)==meta['unique']==163840
    for r in meta['coverage']:
        b=r['index'];tag=f'b{b:02}';beta=float(r['beta']);fs,_=operators(b,mode,maps)
        with np.load(a.banks/f'{tag}-test.npz') as z:
            parts=terms(z);p=law_p(parts,'full',beta);original=prob(z['relative_E'],beta);err=float(np.max(np.abs(p-original)));assert err<2e-14;max_p=max(max_p,err)
            for row in static[static['index']==b].itertuples(index=False):
                k=tag+'_'+row.observable;f=mode[k] if k in mode else None;validate_values(full,b,row.observable,interval(scalar(p,f)[0]['T']),(row.T,row.T_lo,row.T_hi))
            for row in rsmi[rsmi['index']==b].itertuples(index=False):
                name=f'{tag}-s{row.seed}_d{row.depth}_{row.channel}';a0=maps[name+'_a'];q=maps[name+'_Q'];f=fs[f'hard_{row.seed}'];val=scalar(p,f)[0]['T'] if row.channel=='hard' else evaluate(p@q,None if not np.isfinite(a0).any() else a0)
                validate_values(full,b,f'{row.channel}_{row.seed}',interval(val),(row.T,row.T_lo,row.T_hi))
            for law in (LAWS if b==0 else ('intrinsic',)):
                px=law_p(parts,law,beta);spread=float(np.max(abs(px-px[0])));assert spread==0
                if b==0:assert np.max(abs(px-1/210))<=1e-14
                for name,f in fs.items():
                    metrics,_=scalar(px,f);T=metrics['T'][0];assert np.isnan(T) if f is None else abs(T)<1e-12;null.append(dict(index=b,beta=beta,law=law,observable=name,parents=len(px),T=T,total=metrics['total'][0],max_boundary_probability_shift=spread))
    pd.DataFrame(energ).to_csv(a.out/'energy-audit.csv',index=False);pd.DataFrame(full).to_csv(a.out/'03-equality.csv',index=False);pd.DataFrame(null).to_csv(a.out/'nulls.csv',index=False)
    receipt=dict(design=stock.sha(DESIGN),bank_manifest=stock.sha(a.banks/'prepared.json'),parents_scanned=len(lineage),eligible=sum(r['eligible'] for r in energ),integer_checks=sum(r['comparisons'] for r in energ),total_mismatches=0,max_full_probability_error=max_p,endpoint_rows=len(full),undefined_equalities=sum(not x['defined'] for x in full),max_finite_03_error=max(x['max_error'] or 0 for x in full),null_rows=len(null));stock.write_json(a.out/'audit.json',receipt);print('AUDIT',receipt,flush=True)


def measure(a,meta,mode,maps):
    audit_rec=json.loads((a.out/'audit.json').read_bytes());assert audit_rec['design']==stock.sha(DESIGN) and audit_rec['bank_manifest']==stock.sha(a.banks/'prepared.json') and audit_rec['total_mismatches']==0
    rows=[];dist=[];archive={};dpi=[]
    for r in meta['coverage']:
        b=r['index'];tag=f'b{b:02}';beta=float(r['beta']);fs,proj=operators(b,mode,maps);base={};measurements={}
        with np.load(a.banks/f'{tag}-test.npz') as z:
            parts=terms(z);ids=z['all_ids'][z['selected']];archive[tag+'_ids']=ids;archive[tag+'_anchors']=z['anchor']
        pfull=law_p(parts,'full',beta);n=len(ids);qfull={k:pfull@q for k,q in proj.items()}
        for law,(u,i,o) in LAWS.items():
            p=pfull if law=='full' else law_p(parts,law,beta);assert p.shape==(n,210) and np.isfinite(p).all() and np.max(abs(p.sum(1)-1))<1e-14
            state_js=js(p,pfull);base[law]=mean_if((p*p).sum(1));metrics={};ctx=dict(index=b,beta=beta,law=law,field=u,jin=i,jout=o,parents=n)
            for name,f in fs.items():
                metrics[name],moments=scalar(p,f)
                if law in ('fields','full'):archive[f'{tag}_{law}_{name}_moments']=moments
            measurements[law]=metrics
            for name in ('state',*proj):
                if name=='state':j=state_js
                else:
                    j=js(p@proj[name],qfull[name]);excess=float((j-state_js).max());assert excess<=2e-12,(b,law,name,excess);dpi.append((b,law,name,excess))
                if law=='fields':archive[tag+'_endpoint_js_'+name]=j
                if b==0 or law=='full':assert np.max(abs(j))<2e-12
                d=dict(**ctx,space=name);add(d,'JS_bits',interval(mean_if(j)));add(d,'state_minus_projected_bits',interval(mean_if(state_js-j)));dist.append(d)
            if law in ('fields','full'):archive[tag+'_'+law+'_coincidence']=(p*p).sum(1)
        for law,(u,i,o) in LAWS.items():
            for name,m in measurements[law].items():
                row=dict(index=b,beta=beta,parents=n,law=law,field=u,jin=i,jout=o,observable=name);add(row,'state_coincidence',interval(base[law]));add(row,'full_minus_coincidence',interval(diff(base['full'],base[law])))
                for key,x in m.items():
                    add(row,key,interval(x));add(row,'full_minus_'+key,interval(diff(measurements['full'][name][key],x)));add(row,'minus_fields_'+key,interval(diff(x,measurements['fields'][name][key])))
                rows.append(row)
        print('MEASURED',b,n,flush=True)
    pd.DataFrame(rows).to_csv(a.out/'memory.csv',index=False,float_format='%.13g');pd.DataFrame(dist).to_csv(a.out/'distortion.csv',index=False,float_format='%.13g');np.savez_compressed(a.out/'parent-endpoints.npz',**archive)
    pd.DataFrame(dpi,columns=['index','law','code','max_parent_projected_minus_state_bits']).to_csv(a.out/'projection-audit.csv',index=False,float_format='%.13g');stock.write_json(a.out/'measurement.json',dict(audit=stock.sha(a.out/'audit.json'),test_parents=sum(r['test']['selected_parents'] for r in meta['coverage']),laws=len(LAWS),observables=list(NAMES),memory_rows=len(rows),distortion_rows=len(dist),projection_parent_comparisons=sum(r['test']['selected_parents'] for r in meta['coverage'])*len(LAWS)*len(SEEDS),max_projection_excess_bits=max(v[-1] for v in dpi)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['audit','measure']);ap.add_argument('--banks',type=Path,required=True);ap.add_argument('--reference',type=Path,default=ROOT/'results/experiment-03-cavity');ap.add_argument('--out',type=Path,default=DESIGN.parent);args=ap.parse_args();meta=inputs(args)
    with np.load(args.reference/'frozen-modes.npz') as mode,np.load(args.reference/'corrected/code-maps.npz') as maps:{'audit':audit,'measure':measure}[args.command](args,meta,mode,maps)
