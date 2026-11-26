"""History-field CPU gate. Only source DEV is read; patch counts are not independent observations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]/'results/history-overlap-field'
WORK=Path('/home/fritz/rsmi-work/history-overlap-field')

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def geometry():
    grid=np.indices((9,9,9)).reshape(3,-1).T-4
    core=grid[np.abs(grid).max(1)<=1];shell=grid[np.abs(grid).max(1)==4]
    def distance(a,b):
        d=abs((a[:,None]-b+10)%20-10);return int(d.sum(2).min())
    def linear(x):return np.ravel_multi_index((x%20).T,(20,)*3)
    s0=linear(grid);s1=linear(grid+10);assert len(np.unique(s0))==729 and not len(np.intersect1d(s0,s1));assert distance(core,shell)==3 and (len(core),len(shell))==(27,386)
    return core,shell,dict(core=len(core),shell=len(shell),true_min_periodic_L1=distance(core,shell),displaced_min_periodic_L1=distance(core,shell+10),true_displaced_full_cube_intersection=0,displacement=[10,10,10],origins_per_parent=8000)

def box(a,r):
    y=a.astype('i4');
    for i in range(3):y=sum(np.roll(y,n,axis=i) for n in range(-r,r+1))
    return y

def fields(a):return box(a,1).astype('d')/27,(box(a,4)-box(a,3)).astype('d')/386

def centered(x):return x-x.mean()

def residual(y,x):
    y=centered(y);x=centered(x);vx=np.mean(x*x);return y-x*(np.mean(x*y)/vx if vx>0 else 0.)

def association(x,y):
    x=centered(x);y=centered(y);vs=(np.mean(x*x),np.mean(y*y));den=np.sqrt(vs[0]*vs[1]);
    if den<1e-13:raise ValueError('Unsupported constant DEV parent, no valid gate correlation')
    return np.mean(x*y)/den,np.mean(x*np.roll(y,(10,10,10),axis=(0,1,2)))/den,*vs

def summary(z,weights):
    v=weights@z;return dict(mean=float(z.mean()),lo=float(np.quantile(v,.025)),hi=float(np.quantile(v,.975)))

def writecsv(path,rows):
    with path.open('w') as h:
        w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def execute(args):
    out=args.out;work=args.work;core,shell,geom=geometry();rows=[];rpts=[];hashes={};comparison={};primary=None
    for age in [500,300]:
        for level in [80,75,85]:
            f=work/f'b0-a{age}-q{level}-dev.npz';hashes[f.name]=sha(f)
            with np.load(f,allow_pickle=False) as d:maps=d['field'];ids=d['ids'];qs=d['Q'];lag=int(d['lag'])
            assert maps.shape==(16,20,20,20) and maps.dtype==np.uint8 and np.isin(maps,[0,1,2]).all() and len(np.unique(ids))==16
            assert (np.sum(maps!=0,axis=(1,2,3))==6000).all();np.testing.assert_allclose(qs,((maps==1).sum(axis=(1,2,3))/6000-.39)/.61,atol=2e-15,rtol=0)
            vals=[]
            for i,(a,pid,q) in enumerate(zip(maps,ids,qs)):
                x,y=fields(a==2);vx,vy=fields(a!=0);real,far,_,_=association(x,y);rx,ry=residual(x,vx),residual(y,vy);rr,rf,xvar,yvar=association(rx,ry)
                for o in (np.array([0,0,0]),np.array([19,19,19])):
                    j=tuple(o);assert np.isclose(x[j],(a[tuple(((o+core)%20).T)]==2).mean()) and np.isclose(y[j],(a[tuple(((o+shell)%20).T)]==2).mean())
                row=dict(age_M=age,target_Q=level/100,lag=lag,dev_rank=48+i,parent_id=int(pid),Q=float(q),real=float(rr),far=float(rf),adjusted_delta=float(rr-rf),raw_real=float(real),raw_far=float(far),raw_delta=float(real-far),center_resid_variance=float(xvar),shell_resid_variance=float(yvar));rows.append(row);vals.append([rr,rf,rr-rf,real,far,real-far])
            vals=np.asarray(vals);ids=ids.tolist()
            if comparison:assert comparison['ids']==ids
            else:comparison['ids']=ids
            w=np.random.default_rng(641201).multinomial(16,[1/16]*16,size=4096)/16
            sums={k:summary(z,w) for k,z in zip(['real','displaced','delta','raw_real','raw_displaced','raw_delta'],vals.T)};delta=sums['delta'];pos=int(np.count_nonzero(vals[:,2]>0));halves=[float(np.mean(v[:,2])) for v in np.split(vals,2)];passed=delta['lo']>0 and pos>=12 and min(halves)>0
            rec=dict(age_M=age,target_Q=level/100,selected_saved_lag=lag,measured_dev_mean_Q=float(qs.mean()),parents=16,origins=128000,raw_and_adjusted=sums,positive_parents=pos,half_means=halves)
            rpts.append(rec)
            if (age,level)==(500,80):primary={**rec,'pass':bool(passed)}
    writecsv(out/'gate-parents.csv',rows);receipt=dict(design_sha256=sha(out/'design.md'),map_sha256=hashes,scope='beta=3.20 source DEV only, 16 independent parents, all spatial translations, identical within-parent displacement',geometry=geom,primary=primary,descriptive=[x for x in rpts if (x['age_M'],x['target_Q'])!=(500,.8)],held_parents_opened=0,modal_spend=0);(out/'gate.json').write_text(json.dumps(receipt,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,(ax,ay)=plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True);prim=[v for v in rows if (v['age_M'],v['target_Q'])==(500,.8)];xs=np.arange(16)
    for i,v in enumerate(prim):ax.plot([i,i],[v['far'],v['real']],color='#acb4b8',lw=1)
    ax.scatter(xs,[v['real'] for v in prim],c='#147858',s=20,label='True shell');ax.scatter(xs,[v['far'] for v in prim],c='#aa552d',s=20,label='Displaced shell');ax.axhline(0,c='gray',lw=.8);ax.set(xlabel='Independent parent (development rank order)',ylabel='Spatial correlation after occupancy adjustment',title='Primary: 500M preparation, matched Q≈0.80');ax.legend(fontsize=8)
    for age,m,c in [(300,'o','#7557a0'),(500,'s','#147858')]:
        sub=sorted((d for d in rpts if d['age_M']==age),key=lambda d:d['target_Q']);y=np.array([d['raw_and_adjusted']['delta']['mean'] for d in sub]);lo=[d['raw_and_adjusted']['delta']['lo'] for d in sub];hi=[d['raw_and_adjusted']['delta']['hi'] for d in sub];ay.errorbar([d['measured_dev_mean_Q'] for d in sub],y,yerr=[y-lo,hi-y],fmt=m+'-',c=c,label=f'{age}M prep',capsize=3)
    ay.axhline(0,c='gray',ls=':');ay.set(xlabel='Measured mean normalized overlap on DEV',ylabel='True − displaced correlation',title='Same parents at both ages · whole-parent 95% CI');ay.legend(fontsize=8);fig.savefig(out/'gate.png',dpi=160);plt.close(fig);print(json.dumps(primary,sort_keys=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=WORK);p.add_argument('--out',type=Path,default=OUT);execute(p.parse_args())
