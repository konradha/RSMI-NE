"""Equal-rate, whole-configuration application of the archived two-bit tile RG. No simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

import rg
import tile_codec as codec
from rate import source_rows, partition, digest, json_out

TARGETS=('A_X_star','energy_per_site','A_first_wave_star')
BLOCKS=np.arange(6561)[:,None]//codec.POW%3
RG=Path(__file__).resolve().parents[2]/'results/l20-autoresearch/rg'
SEED=509321


def maps():
    result=json.loads((RG/'confirmation.json').read_bytes());r=json.loads((RG/f"round-{result['selected_round']:04d}.json").read_bytes());weights=RG/'frozen-operator.npz';models,_=rg.restore(weights,r);arm=tuple(result['arm'])
    if arm!=(1,'tile','vector',1) or result['selected']!='learned0':raise ValueError('unexpected frozen operator')
    out={'rsmi':rg.netcode(BLOCKS,arm,models[0])};out.update(rg.fixed(BLOCKS,r['calibration']));axes=1-2*codec.TILE; tetra=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    for label,s in [('A',1),('B',2),('vacancy',0)]:
        dip=(BLOCKS==s).astype(int)@axes;out[f'tetra_{label}']=np.argmax(dip@tetra.T,axis=1)
    return {k:v.astype(np.uint8) for k,v in out.items()},digest(weights)


def kmeans_dipole(freq):
    dip=(BLOCKS==1).astype(int)@(1-2*codec.TILE);z,inverse=np.unique(dip,axis=0,return_inverse=True);inverse=inverse.ravel();mass=np.bincount(inverse,weights=freq,minlength=len(z));rng=np.random.default_rng(SEED);best=np.inf
    for _ in range(16):
        c=z[rng.choice(len(z),4,False,p=mass/mass.sum())].astype(float)
        for _ in range(80):
            ds=((z[:,None]-c)**2).sum(2);g=ds.argmin(1);w=np.bincount(g,weights=mass,minlength=4)
            if w.min()==0:break
            cc=np.stack([np.bincount(g,weights=mass*z[:,j],minlength=4)/w for j in range(3)],axis=1)
            if np.max(np.abs(c-cc))<1e-12:break
            c=cc
        val=mass@(((z[:,None]-c)**2).sum(2).min(1))
        if val<best:best=val;chosen=((z[:,None]-c)**2).sum(2).argmin(1)
    return chosen[inverse].astype(np.uint8)


def physical(s):
    s=s.reshape(-1,20,20,20);occ=(s!=0).astype(np.int16);m=sum(np.roll(occ,k,axis) for axis in (1,2,3) for k in (-1,1));en=(occ*(m-np.array([0,3,5],dtype=np.int16)[s])**2).sum((1,2,3));a=s==1;parity=1-2*(np.arange(20)%2);wave=np.exp((-2j*np.pi/20)*np.arange(20));x=np.zeros(len(s));w=np.zeros(len(s))
    for axis in (1,2,3):
        proj=a.sum(tuple(k for k in (1,2,3) if k!=axis));x+=(proj@parity)**2;w+=np.abs(proj@wave)**2
    return np.column_stack((x/24000,en/8000,w/24000))


def bank(data,idx,folds,tabs,out,train_dip=False):
    (lookup,states),meta=source_rows(data,idx,folds);ids=partition(idx)[0];ordered=np.concatenate([ids[f] for f in folds]);tiles=np.empty((len(ordered),10,10,10),dtype=np.uint16);y=np.empty((len(ordered),3));freq=np.zeros(6561,'i8');ntr=len(ids[0]) if 0 in folds else 0
    for start in range(0,len(ordered),32):
        sl=slice(start,start+32);s=states[lookup[ordered[sl]]].reshape(-1,20,20,20);tiles[sl]=codec.tile_index(s);y[sl]=physical(s)
        if start<ntr:freq+=np.bincount(tiles[start:min(start+32,ntr)].ravel(),minlength=6561)
    del states
    with np.load(data/f'beta-{idx:02d}.npz',allow_pickle=False) as f:energy=f['energy'][ordered]
    if not np.allclose(8000*y[:,1],energy,atol=1e-8,rtol=0):raise ValueError('microscopic total energy does not match source')
    if train_dip:tabs['kmeans_A_dipole']=kmeans_dipole(freq)
    xs={};keys=list(tabs)
    for name in keys:
        parts=[];packed=np.empty((len(ordered),250),dtype=np.uint8) if name=='rsmi' else None
        for start in range(0,len(ordered),32):
            sl=slice(start,start+32);h=tabs[name][tiles[sl]];b=codec.pack(h);u=codec.unpack(b)
            if not np.array_equal(h,u):raise ValueError('2-bit packing changed the tile field')
            parts.append(codec.features(u));
            if packed is not None:packed[sl]=b
        xs[name]=np.concatenate(parts)
        if packed is not None:
            subset=packed[ntr:];np.save(out/f'packed-{idx:02d}.npy',subset,allow_pickle=False);np.save(out/f'rows-{idx:02d}.npy',ordered[ntr:],allow_pickle=False)
    return xs,y,ntr,meta


def fit(x,y):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1;z=(x-mu)/sd;y0=y.mean(0);w=np.linalg.solve(z.T@z/len(z)+1e-3*np.eye(z.shape[1]),z.T@(y-y0)/len(z));return dict(mean=mu,scale=sd,coef=w,intercept=y0)


def pred(x,m):return (x-m['mean'])/m['scale']@m['coef']+m['intercept']


def r2(y,p):
    err=y-p
    def var_del(x):
        z=x-x.mean(0);return (z*z).sum(0)-len(x)/(len(x)-1)*z*z
    return 1-err.var(0)/y.var(0),1-var_del(err)/var_del(y),1-(err*err).mean(0)/y.var(0)


def summary(m,j):
    se=np.sqrt((len(j)-1)*np.mean((j-j.mean(0))**2,axis=0));return np.stack((m,m-t.ppf(.975,len(j)-1)*se,m+t.ppf(.975,len(j)-1)*se),axis=-1).tolist()


def inspect(y,ps,comp):
    results={k:r2(y,p) for k,p in ps.items()};s0,d0,_=results['rsmi'];mean=np.array([results[k][0][i] for i,k in enumerate(comp)]);leave=np.stack([results[k][1][:,i] for i,k in enumerate(comp)],axis=1)
    return dict(parents=len(y),targets=list(TARGETS),target_mean=y.mean(0).tolist(),target_sd=y.std(0).tolist(),comparators=comp,centered_R2={k:summary(m,j) for k,(m,j,_) in results.items()},raw_R2={k:v.tolist() for k,(_,_,v) in results.items()},paired_vs_best=summary(s0-mean,d0-leave),parent_derangement_centered_R2=r2(y,np.roll(ps['rsmi'],1,axis=0))[0].tolist())


def cold(a):
    lock=a.out/'prediction.json'
    if lock.exists():raise FileExistsError('codec already frozen')
    tabs,rgsha=maps();xs,y,n,meta=bank(a.data,9,(0,1),tabs,a.out,True);models={k:fit(x[:n],y[:n]) for k,x in xs.items()};ps={k:pred(x[n:],models[k]) for k,x in xs.items()};names=[k for k in models if k!='rsmi'];comp=[max(names,key=lambda k:r2(y[n:],ps[k])[0][i]) for i in range(3)];val=inspect(y[n:],ps,comp)
    np.savez_compressed(a.out/'codec.npz',lut=tabs['rsmi'],targets=np.array(TARGETS),**models['rsmi']);allmodels={k+'_'+s:v for k,params in models.items() for s,v in dict(**params,lut=tabs[k]).items()};np.savez_compressed(a.out/'comparators.npz',**allmodels)
    with np.load(a.out/'codec.npz') as model:
        if not np.allclose(codec.predict(np.load(a.out/'packed-09.npy')[:64],model),ps['rsmi'][:64],atol=1e-10):raise ValueError('standalone packed inference mismatch')
    json_out(a.out/'cold.json',dict(source=meta,**val));json_out(lock,dict(weights_sha256=rgsha,codec_sha256=digest(a.out/'codec.npz'),comparators_sha256=digest(a.out/'comparators.npz'),script_sha256=digest(Path(__file__)),codec_implementation_sha256=digest(Path(codec.__file__)),methods=list(tabs),comparators=comp,primary='At beta index 8 the frozen 250-byte RSMI representation retains at least half of within-temperature variance of A-species X-star intensity, and separately beats the strongest equal-size cold-selected conventional encoder for it.',threshold=.5,utility='R2_centered = 1 - Var(error)/Var(Y) within a temperature. Raw R2 separately retains mean error; no model refitting or new microscopic observations at transfer. Paired whole-parent jackknife normal/t intervals conditional on frozen probes. Beta=0 is a control for RSMI specificity, not an expectation of zero ability to encode occupancies.',targets=['mean of S_A(pi,0,0) over three axes','exact H/L^3','mean of S_A(2pi/L,0,0) over three axes'],readout='Same fixed ridge on three symbol frequencies, their six quadratic moments, all 48 directed nearest-tile pair frequencies and six reduced first-wave power bilinears. The X target alone can be estimated from counts, so this is not a lower limit on bytes needed for that single quantity.',format='Nonoverlapping stride-2 tiles at origin 0. Flatten 10^3 labels z,y,x and pack four labels in each byte, least significant label first. 250 bytes per parent, excluding amortized table/probe and row identifiers. Lossy: never replace microscopic archives. 8x smaller than packing 8000 microscopic species into two bits each.'))
    print('cold',val['centered_R2']['rsmi'],comp,flush=True)


def transfer(a):
    fn=a.out/'transfer.json'
    if fn.exists():raise FileExistsError('already evaluated')
    lock=json.loads((a.out/'prediction.json').read_bytes());assert lock['codec_sha256']==digest(a.out/'codec.npz') and lock['comparators_sha256']==digest(a.out/'comparators.npz') and lock['script_sha256']==digest(Path(__file__)) and lock['codec_implementation_sha256']==digest(Path(codec.__file__))
    with np.load(a.out/'comparators.npz') as f:mods={k:{s:f[k+'_'+s] for s in ('lut','mean','scale','coef','intercept')} for k in lock['methods']}
    tabs={k:m['lut'] for k,m in mods.items()};rows=[]
    for idx in (8,0):
        xs,y,_,meta=bank(a.data,idx,(1,),tabs,a.out);p={k:pred(x,mods[k]) for k,x in xs.items()};result=inspect(y,p,lock['comparators']);rows.append(dict(source=meta,**result))
    json_out(fn,dict(lock_sha256=digest(a.out/'prediction.json'),temperatures=rows));plot(a.out,[json.loads((a.out/'cold.json').read_bytes()),*rows]);print('transfer',[(r['source']['beta'],r['centered_R2']['rsmi'],r['paired_vs_best']) for r in rows],flush=True)


def plot(out,rows):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained');xx=np.arange(len(rows));labs=[f"{r['source']['beta']:.2f}" for r in rows];
    for i,(ax,title) in enumerate(zip(axes,['A: period-two fluctuations','Energy per site','A: box-scale fluctuations'])):
        for j,(name,fmt) in enumerate([('RSMI','o'),('Conventional','s')]):
            vals=np.array([r['centered_R2']['rsmi' if j==0 else r['comparators'][i]][i] for r in rows]);c,l,h=vals.T;ax.errorbar(xx+(2*j-1)*.075,c,yerr=[c-l,h-c],fmt=fmt,capsize=3,label=name)
        ax.axhline(0,c='.65',lw=.7);ax.set(xticks=xx,xticklabels=labs,title=title,xlabel='Inverse temperature β');ax.spines[['top','right']].set_visible(False)
    axes[0].axhline(.5,c='.55',ls='--',lw=.8);axes[0].set_ylabel('Within-temperature variance recovered (R²)');axes[0].legend(frameon=False,fontsize=8);fig.savefig(out/'utility.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('cold','transfer'));p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);(cold if a.mode=='cold' else transfer)(a)
