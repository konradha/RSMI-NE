"""Items 14–15: one locked physical calibration, exact allowlisted child rows only."""
import argparse
import csv
import hashlib
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import t

from fifteen_dynamics_lineage import OUT, ROOT, LOCK, LOCK_COMMIT, readj, store, sha
from l20_pairs import partition
from l20_propensity_predict import coordination_barrier, NO_SWAP
from l20_propensity_proximity import four_shells
from tile_codec import tile_index


def frame(path, cohort, lag):
    """Inflate/seek past uninterpreted bytes; only copy explicitly authorized state rows.

    The multi-temperature checkpoint has one deflated member, so getting a chosen
    slice requires the ZIP reader to traverse earlier compressed bytes. Never call
    np.load()['states'], extract a whole member, or decode any skipped configurations.
    """
    if sha(path)!=path.stem.split('-')[1]:raise ValueError('committed snapshot fingerprint mismatch')
    children=np.array([[int(c) for c in p['child_stream_ids']] for p in cohort],np.uint64)
    with np.load(path,allow_pickle=False) as z:
        ids=z['ids']; positions={int(x):i for i,x in enumerate(ids)}; slots=np.array([[positions[int(x)] for x in cc] for cc in children]); pp=z['parent_ids'][slots]; age=z['parent_age'][slots]; beta=z['betas'][slots]; h=z['parent_hash'][slots]
        assert int(z['sweep'])==lag and len(np.unique(slots))==len(cohort)*2 and np.all(age==378368) and np.all(beta.astype(float)==2.372222222222)
        assert np.array_equal(pp,np.broadcast_to([[int(p['parent_id'])] for p in cohort],pp.shape)) and np.all(h==np.array([p['packed_parent_sha256'] for p in cohort])[:,None])
    cells=np.empty((len(cohort)*2,8000),np.uint8); flat=slots.ravel()
    with zipfile.ZipFile(path) as z,z.open('states.npy') as f:
        assert np.lib.format.read_magic(f)==(1,0);shape,fortran,dtype=np.lib.format.read_array_header_1_0(f);assert shape==(len(ids),8000) and not fortran and dtype==np.uint8
        start=f.tell()
        for j in np.argsort(flat):
            f.seek(start+8000*int(flat[j]));assert f.readinto(memoryview(cells[j]))==8000
    if lag==0:
        for p,cc in zip(cohort,cells.reshape(-1,2,8000)):
            assert np.array_equal(cc[0],cc[1]) and hashlib.sha256(cc[0]).hexdigest()==p['packed_parent_sha256']
    cells &= 3
    assert np.all((cells==0).sum(1)==2000) and np.all((cells==1).sum(1)==2400) and np.all((cells==2).sum(1)==3600)
    return cells.reshape(-1,2,20,20,20)


def ci(v):
    v=np.asarray(v,float);n=len(v);mean=v.mean(0);se=v.std(0,ddof=1)/np.sqrt(n);a=t.ppf(.975,n-1);b=t.ppf(1-.05/30,n-1)
    return dict(mean=mean.tolist(),parent95=[(mean-a*se).tolist(),(mean+a*se).tolist()],family15=[(mean-b*se).tolist(),(mean+b*se).tolist()],MDE80_family_approx=((b+.841621)*se).tolist(),n=n)


def features(origin,lut):
    barrier=np.stack([coordination_barrier(s)[1] for s in origin]); h=lut[tile_index(origin)].repeat(2,1).repeat(2,2).repeat(2,3)
    # Every no-unlike-neighbor site has its own barrier category, not a zero cost.
    bm=np.where(barrier==NO_SWAP,127,barrier);assert bm.min()>-128
    base=origin.astype(np.int32)*256+bm.astype(np.int32)+128
    return barrier,h,base


def forecast(origin,h,base,event,nfit):
    ok=origin[:nfit]!=0;y=event[:nfit].mean(1)[ok];pred=np.full(origin[nfit:].shape,y.mean());eligible=origin[nfit:]!=0
    star=[];unknown=[]
    for all_keys in [origin,base,4*base+h]:
        a=all_keys[:nfit][ok].astype(int);b=all_keys[nfit:];n=np.bincount(a,minlength=int(all_keys.max())+1);s=np.bincount(a,weights=y,minlength=len(n));tab=s/np.maximum(n,1)
        unknown.append(int(((n[b]==0)&eligible).sum()));pred=np.where(n[b]>0,tab[b],pred)
        if all_keys is not origin:star.append(np.where(eligible,pred-y.mean(),0))
    return float(y.mean()),star,unknown


def fft_stars(field):
    # Real axial projections, retaining signed complex phase shared by siblings.
    f=np.fft.fftn(field,axes=(-3,-2,-1),norm='ortho')
    return np.stack([np.stack([f[...,k,0,0],f[...,0,k,0],f[...,0,0,k]],axis=-1) for k in (1,10)],axis=-2)


def spectra(y,fore):
    z=fft_stars(y);self_power=(abs(z)**2).mean(axis=(1,3));cross=(z[:,0]*z[:,1].conj()).real.mean(-1);res=[]
    for field in fore:
        f=fft_stars(field);res.append((2*(f*z.mean(1).conj()).real-abs(f)**2).mean(-1))
    assert np.allclose(self_power-cross, .5*(abs(z[:,0]-z[:,1])**2).mean(-1),rtol=1e-12,atol=1e-14)
    return self_power,cross,res


def ratios(cross,auto):
    # Family-resolved denominator test *before* taking a ratio; no sign clipping.
    n=len(auto);resolved=np.array(ci(auto)['family15'][0])>0
    if not resolved.any():return dict(resolved=resolved.tolist(),by_star=[None,None],low_minus_high=None)
    rat=cross.mean(0)/auto.mean(0);jk=((cross.sum(0)-cross)/(auto.sum(0)-auto));pseudo=n*rat-(n-1)*jk
    def report(x,est):
        # The ratio estimator, not the jackknife bias-corrected point estimate.
        return ci(x-x.mean()+est)
    stars=[report(pseudo[:,i],rat[i]) if resolved[i] else None for i in range(2)]
    return dict(resolved=resolved.tolist(),by_star=stars,low_minus_high=report(pseudo[:,0]-pseudo[:,1],rat[0]-rat[1]) if resolved.all() else None)


def run14(out,audit,parent,origin,event,code,base):
    fit=11;mu,fore,unseen=forecast(origin,code,base,event,fit);elig=origin[fit:]!=0
    y=(event[fit:]-mu)*elig[:,None];auto,cross,(f0,f1)=spectra(y,fore);delta=f1-f0;frac=ratios(cross,auto)
    rows=out/'14-parent-power.npz';np.savez_compressed(rows,parent_id=parent[fit:],stars=np.array(['axial_1','axial_10']),eligible=elig.sum((1,2,3)),changed_counts=event[fit:].sum((2,3,4)),auto=auto,cross=cross,barrier_predictive=f0,barrier_code_predictive=f1,delta=delta)
    lo,hi=np.array(ci(delta[:,0])['family15']).ravel();floor=.01
    verdict=('reaches' if lo>=floor else 'excludes' if hi<floor else 'does not resolve')+' the prespecified +0.01 low-star incremental forecast threshold on this pilot.'
    obj=dict(item=14,question='At which wavelength is endpoint mobility predictable from the origin?',historical_difference='Unlike the historical spatially averaged sitewise sibling-half covariance and Brier fit, two fixed nonzero axial wavevectors retain collective Fourier cross-power and the code ablation is scored at k.',primary='Ineligible at beta=3.05. The 64x16 and directional 512x8 are native TEST; original 32x1 has no distinct siblings. Beta=2.7111 directional also TEST and original/doubled are single-child.',primary_eligible_parents=0,primary_eligible_disjoint_pairs=0,primary_MDE='Unbounded/unidentifiable: zero eligible replicated parents.',
             lineage_receipt='14-15-lineage.json',published_lock_commit=LOCK_COMMIT,immutable_lock_sha256=LOCK,population='Secondary calibration only; beta=2.372222222222, candidate age 378368, exact 4096 local sweeps; not matched fractional relaxation time.',normalized_lag_in_tau=4096/70067.75933204957,
             parents_fit=11,parents_score=10,replicas=2,disjoint_pairs_score=10,forks_scored=20,occupied_origin_sites_per_parent=6000,eligible_child_site_trials=120000,total_sites_per_fft=8000,distinct_original_trials_changed=int(event[fit:].sum()),fit_occupied_change_mean=mu,fit_unseen_score_sites_parent_fallback_by_model=dict(zip(['species','species+exactBarrier','species+exactBarrier+code'],unseen)),
             definition='Y_child(k)=unitary FFT[1(initial occupied)*(1(final species != initial species) - fit occupied mean)]. Shared global centering fixed outside score parents. Low: +1 axial star (negative conjugates redundant); high: 10 axial Nyquist star,3 distinct modes each. Average modes before inference; cross=Re(Y0 conj(Y1)); self=(|Y0|²+|Y1|²)/2. Null/negative powers retained. Ratio=mean cross/mean auto, computed only if family-adjusted parent interval on mean auto excludes zero; ratio errors by delete-parent linearization.',
             star_order=['fundamental','Nyquist'],self_power=ci(auto),structural_cross_power=ci(cross),incoherent_power=ci(auto-cross),structural_fraction=frac,low_minus_high_fraction_useful_floor=.05,low_minus_high_structural_power=ci(cross[:,0]-cross[:,1]),barrier_predictive_power=ci(f0),barrier_code_predictive_power=ci(f1),delta_predictive_power=ci(delta),fixed_forecast='Training cell means for species x exact minimum unlike-neighbor swap cost, then split by historical stride-two tile code containing the site. Unseen cells fall back to parent class; no fit on evaluation parents. Predictive power averages 2 Re(F conj(Y))-|F|²; a poor forecast can have negative utility.',useful_gain=.01,decision='No cold/warm estimate. Secondary '+verdict,
             checks='Paired sibling identity self-cross = |Y0-Y1|²/2 verified in each parent/star; synthetically opposed and identical amplitudes retain negative and unit structural fractions respectively.',
             inference='Intervals apply only to 10 independent native-training pilot parents and fixed forecasts (conditional on fit); axes, sites, children are not independent samples. A static conditional two-child decomposition at L20 is not a structural length measurement.',parent_evidence=rows.name,parent_evidence_sha256=sha(rows),citations=['https://arxiv.org/abs/0706.1044','https://arxiv.org/abs/2611.27231'])
    store(out/'14-mobility-wavelength.json',obj)


def joint(early,late):
    same=(early*late).sum(0,dtype=np.int16);other=(early*late[::-1]).sum(0,dtype=np.int16)
    assert np.array_equal(same-other,(early[0].astype(np.int16)-early[1])*(late[0].astype(np.int16)-late[1]))
    return same,other


def run15(out,parent,origin,code,barrier,s16,s256,s4096):
    records=[];cells=[]
    for p,s,h,b,first,middle,last in zip(parent,origin,code,barrier,s16,s256,s4096):
        valid=s!=0;early=(s!=first)&valid;late=(middle!=last)&valid
        same,cross=joint(early,late);n=int(valid.sum());assert n==6000
        proximity=np.digitize(four_shells(b<=0),[1,3,5]);key=np.stack([s[valid],b[valid],proximity[valid]],axis=1)
        strata,inv=np.unique(key,axis=0,return_inverse=True);inv=np.ravel(inv);sh=(len(strata),2);binary=(h[valid]!=0).astype(int);id=inv*2+binary
        def group(v):return np.bincount(id,weights=v,minlength=2*len(strata)).astype(int).reshape(sh)
        count=group(np.ones(n));S=group(same[valid]);C=group(cross[valid]);E=group(early.sum(0)[valid]);L=group(late.sum(0)[valid]);both=count.min(1)>0
        # One shared weighting distribution, the pooled eligible-site composition on common support.
        used=count[both].sum();weights=count[both].sum(1)/used;matched=weights@((S[both]-C[both])/(2*count[both]));
        for j,st in enumerate(strata):
            for g in range(2):
                cells.append(dict(parent_id=int(p),origin_species=int(st[0]),minimum_dE='none' if st[1]==NO_SWAP else int(st[1]),distance_bin=['0','1-2','3-4','>=5'][st[2]],code_group=['0','other'][g],sites=int(count[j,g]),early_events=int(E[j,g]),late_events=int(L[j,g]),same_joint_events=int(S[j,g]),other_joint_events=int(C[j,g]),common_support=bool(both[j])))
        z=dict(parent_id=int(p),sites=n,early=int(early.sum()),late=int(late.sum()),same_joint=int(S.sum()),sibling_joint=int(C.sum()),same_joint_rate=S.sum()/(2*n),sibling_joint_rate=C.sum()/(2*n),difference=(S.sum()-C.sum())/(2*n),code0_sites=int(count[:,0].sum()),common_sites=int(used),coverage=used/n,matched_code0=float(matched[0]),matched_other=float(matched[1]),moderation=float(matched[0]-matched[1]),late_change_vs_origin_not_used=int(((s!=last)&valid).sum()))
        records.append(z)
    def write_csv(name,rows):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_csv('15-parent-joints.csv',records);write_csv('15-strata.csv',cells)
    def estimates(k):return ci([r[k] for r in records])
    effect=estimates('difference');lo,hi=effect['family15']; verdict=('reaches' if lo>=.001 else 'excludes' if hi<.001 else 'does not resolve')+' the prespecified +.001 excess threshold.'
    store(out/'15-sibling-history.json',dict(item=15,question='Do early and late endpoint changes in the same child co-occur beyond sharing the origin?', contrast_to_history='Paired, disjoint endpoint intervals and child identity vs ordered different sibling at the identical site; historical fit predicts single-interval propensity or static proximity, neither is this joint contrast.',primary_eligible_parents=0,primary_eligible_disjoint_pairs=0,primary_MDE='Unbounded/unidentifiable at cold beta; 64x16 cold forks are all static TEST. Directional cold 512x8 has no 16/256 frames and is static TEST. Warm directional also lacks those exact two early frames and is TEST; original cold/warm are single child.',lineage_receipt='14-15-lineage.json',published_lock_commit=LOCK_COMMIT,immutable_lock_sha256=LOCK,population='Secondary calibration only: beta=2.372222222222, candidate age 378368, same 10 locked calibration parents as #14, two independent physical siblings per parent. No historical test rows of native static split.',calibration_parents=10,score_children=20,disjoint_child_pairs=10,ordered_child_comparisons_per_site=2,total_eligible_sites=sum(r['sites'] for r in records),same_opportunities=120000,ordered_sibling_opportunities=120000,event_counts={k:sum(r[k] for r in records) for k in ['early','late','same_joint','sibling_joint']},
        definition='E_a=1(s16_a != s0), L_a=1(s4096_a != s256_a) on the same initially occupied site. D=mean_sites[(E0-E1)(L0-L1)/2]. Each observed early and late marginal appears equally on both sides. Vacancy at the late interval start is allowed. Neither event requires the site to stay changed and returns are counted only by its endpoints.',same_joint_rate=estimates('same_joint_rate'),sibling_joint_rate=estimates('sibling_joint_rate'),signed_excess=effect,floor_positive_excess=.001,decision='No cold or warm estimate. Secondary calibration '+verdict,
        matching='Same site in both terms gives exact initial species, barrier and free-edge proximity match without dropping any main-contrast site. All exact strata shown in 15-strata.csv. Moderator weights the two frozen symbol groups (code0 minus all others) to their shared pooled parent-specific distribution over those strata, excludes unmatched cells in the moderator only.',common_strata_coverage=estimates('coverage'),common_strata_sites=sum(r['common_sites'] for r in records),matched_code0=estimates('matched_code0'),matched_other=estimates('matched_other'),code_moderator=estimates('moderation'),
        controls='Within every parent and site tabulated same-minus-cross equals the signed two-sibling product. Toy constant early (or late) events give zero; exchanging late child identities reverses the sign. Age/hot contrasts outside the published lock not analyzed; older origin from same prep stream not an independent parent. This is a signed association, not causal transmission or a lifetime.',
        parent_evidence='15-parent-joints.csv',strata_evidence='15-strata.csv',parent_evidence_sha256=sha(out/'15-parent-joints.csv'),strata_evidence_sha256=sha(out/'15-strata.csv'),citations=['https://arxiv.org/abs/1211.0778','https://arxiv.org/abs/2609.34022']))


def main(runs,out):
    audit=readj(out/'14-15-lineage.json');assert audit['lock_sha256']==sha(out/'physical-lock.json')==LOCK and audit['published_lock_commit']==LOCK_COMMIT
    p=audit['eligible_child_streams']; assert len(p)==21 and [v['role'] for v in p]==['fit']*11+['calibration']*10 and np.isin([x['release_row'] for x in p],partition(7)[0][0]).all()
    parent=np.array([int(v['parent_id']) for v in p]);assert (np.diff(parent)>0).all()
    # Run all algebra controls without any trajectory.
    x=np.array([[0,1],[1,0]],bool);s,c=joint(x,x);u,v=joint(x,x[::-1]);assert np.array_equal(s-c,v-u) and np.all((joint(x,np.ones_like(x))[0]-joint(x,np.ones_like(x))[1])==0)
    wave=np.cos(np.arange(20)*2*np.pi/20)[None,None,None,:]*np.ones((1,20,20,1));fixture=np.stack([wave,-wave],axis=1);aa,cc,ss=spectra(fixture,[np.zeros_like(wave)]);assert aa[0,0]>0 and np.isclose(cc[0,0],-aa[0,0]) and ss[0][0,0]==0
    aa,cc,ss=spectra(np.stack([wave,wave],axis=1),[wave]);assert np.isclose(cc[0,0],aa[0,0]) and np.isclose(ss[0][0,0],aa[0,0]);
    paths={int(k):runs/v for k,v in audit['frame_paths'].items()}
    origin=frame(paths[0],p,0)[:,0];last=frame(paths[4096],p,4096);event=(last!=origin[:,None])&(origin[:,None]!=0)
    with np.load(ROOT/'results/l20-autoresearch/codec/codec.npz') as f:lut=f['lut']
    b,h,base=features(origin,lut);run14(out,audit,parent,origin,event,h,base)
    # No need to open 16/256 for the eleven fit parents; #15 has no outcome-fitted parameters.
    s16=frame(paths[16],p[11:],16);s256=frame(paths[256],p[11:],256)
    run15(out,parent[11:],origin[11:],h[11:],b[11:],s16,s256,last[11:])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',type=Path);p.add_argument('--out',type=Path,default=OUT);args=p.parse_args();main(args.runs,args.out)
