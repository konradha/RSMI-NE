"""Secondary warm native forks: filter whole reserve families BEFORE row access."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import zipfile

import numpy as np

import opening_history_match as m
from opening_history_stage import ARCHIVE, LOCK, OUT, ROOT, SCRATCH, native, save_json
from opening_history_report import finite, metrics, write_csv

STUDIES = (('l20-b305-production-v1',tuple(i for i in range(10) if i!=8)),
           ('l20-b2711-double-v1',(8,)))
ROSTER = ROOT/'results/buffered-static-gate1/roster.npz'
WORK = SCRATCH/'warm'


def plan(stage):
    p=stage/'plan.json'; digest=native.sha(p); batch=stage/'batch-00000000'; head=json.loads((batch/'head.json').read_bytes())
    if digest!=head['plan_sha256'] or head['batch']!=0:raise ValueError('plan provenance')
    files={int(s[:20]):batch/s for s in head['snapshots']}
    if list(files)!=sorted(files) or min(files)!=0:raise ValueError('warm schedule')
    return json.loads(p.read_bytes()),digest,files


def checked_meta(path,lag,digest):
    if path.name!=f'{lag:020d}-{native.sha(path)}.npz':raise ValueError('snapshot digest')
    nm=native.native_meta(path,lag,digest)
    if len(np.unique(nm['ids']))!=len(nm['ids']):raise ValueError('duplicate native child IDs')
    return nm


def rows_only(path,ix,total):
    """Mixed deflated members: seek over unrequested rows; never read/return a reserve row."""
    ix=np.asarray(ix,dtype=int); order=np.argsort(ix); ix=ix[order]
    if len(ix)==0 or ix[0]<0 or ix[-1]>=total or np.any(np.diff(ix)<=0):raise ValueError('invalid row selection')
    arr=np.empty((len(ix),8000),np.uint8); breaks=np.r_[0,np.flatnonzero(np.diff(ix)>1)+1,len(ix)]
    with zipfile.ZipFile(path) as archive,archive.open('states.npy') as f:
        if np.lib.format.read_magic(f)!=(1,0):raise ValueError('states.npy version')
        shape,fort,dtype=np.lib.format.read_array_header_1_0(f); start=f.tell()
        if shape!=(total,8000) or fort or dtype!=np.uint8:raise ValueError('non-native warm schema')
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            f.seek(start+int(ix[lo])*8000)
            # Read only contiguous runs of already routed IDs.
            raw=f.read((hi-lo)*8000); arr[order[lo:hi]]=np.frombuffer(raw,np.uint8).reshape(-1,8000)
    return arr


def stage():
    if native.sha(OUT/'protocol.json')!=LOCK:raise ValueError('protocol mismatch')
    WORK.mkdir(parents=True,exist_ok=True); receipts=[]; totals=[]; output=[]
    with np.load(ROSTER) as z:
        ix=z['beta_index']; groups=z['group']; ids=z['ids']
    for study,targets in STUDIES:
        root=ARCHIVE/'runs'/study; src,sd,sp=plan(root/'probe'); local,ld,lp=plan(root/'physical')
        if src['move']!='nonlocal' or local['move']!='local' or src['L']!=20 or local['source_plan_sha256']!=sd:
            raise ValueError('invalid prep/fork link')
        # Independently reconstruct the source save-point corpus hash used by native local plan.
        import hashlib
        hh=hashlib.sha256()
        target_ages=set(src['sweeps']) | {t for t in src['save_at'] if t<=max(src['sweeps'])}
        for age in sorted(target_ages):
            hh.update((json.dumps([0,sp[age].name],separators=(',',':'))+'\n').encode())
        if hh.hexdigest()!=local['source_corpus_sha256']:raise ValueError('warm prep corpus hash')
        meta=checked_meta(lp[0],0,ld); by_age=defaultdict(list); records=[]
        for slot,b in enumerate(range(10) if len(targets)>1 else [8]):
            if b not in targets:continue
            beta=local['betas'][slot]; mask=(ix==b)&(ids[:,0]==src['seed']); by_id={int(p):int(g) for p,g in zip(ids[mask,1],groups[mask])}
            rows=np.flatnonzero(meta['betas']==beta); parents=np.unique(meta['parent_ids'][rows]);
            if set(map(int,parents))!=set(by_id):raise ValueError('not a one-to-one link to release probe IDs')
            ages=set(local['parent_age_pairs'][slot]); expected=local['replicas_per_beta'][slot]
            allowed=0
            for pid in parents:
                idx=rows[meta['parent_ids'][rows]==pid]; counts={int(a):int((meta['parent_age'][idx]==a).sum()) for a in ages}
                if set(meta['parent_age'][idx])!=ages or any(v!=expected for v in counts.values()):raise ValueError('unpaired warm ages/siblings')
                if by_id[int(pid)]==3:continue
                allowed+=1
                for ii in idx:
                    r=dict(index=b,beta=beta,source_seed=src['seed'],parent_id=int(pid),age=int(meta['parent_age'][ii]),
                           child_id=int(meta['ids'][ii]),row=int(ii),parent_hash=str(meta['parent_hash'][ii]),
                           roster_group=by_id[int(pid)],study=study)
                    records.append(r);by_age[r['age']].append(r)
            powers=[t for t in lp if t and t<=min(10**6,local['sweeps'][slot]) and t&(t-1)==0]
            if powers!=[1<<i for i in range(len(powers))]:raise ValueError('non-common powers grid')
            totals.append(dict(index=b,beta=beta,study=study,all_parents=len(parents),eligible_parents=allowed,
                               reserve_excluded=len(parents)-allowed,all_forks=len(rows),eligible_forks=allowed*len(ages)*expected,
                               reserve_forks_excluded=(len(parents)-allowed)*len(ages)*expected,ages=sorted(ages),siblings_per_age=expected,
                               powers=powers,terminal_physical_sweeps=local['sweeps'][slot]))
        # No seal-breaking path: construct both state row selections only from non-reserve records.
        eligible=rows_only(lp[0],[r['row'] for r in records],len(meta['ids']))
        origin_by_child={r['child_id']:state for r,state in zip(records,eligible)}
        for age,rec in by_age.items():
            nm=checked_meta(sp[age],age,sd); lookup={int(pid):i for i,pid in enumerate(nm['ids'])}
            unique={r['parent_id']:r for r in rec}; rows=[lookup[pid] for pid in unique]
            prep=rows_only(sp[age],rows,len(nm['ids'])); src_state={}
            for (pid,r),pos,state in zip(unique.items(),rows,prep):
                if nm['betas'][pos]!=r['beta'] or pid!=nm['parent_ids'][pos] or native.hash_bytes(state)!=r['parent_hash']:
                    raise ValueError('warm preparation identity mismatch')
                src_state[pid]=state
            for r in rec:
                if not np.array_equal(src_state[r['parent_id']],origin_by_child[r['child_id']]):raise ValueError('warm origin mismatch')
            # Store one physical configuration per independent preparation parent × age, never per replica.
            for pid,r in unique.items():
                state=src_state[pid]&3
                if (state>2).any() or np.count_nonzero(state)!=6000:raise ValueError('warm composition')
                np.save(WORK/f"origin-{r['index']}-{pid}-{age}.npy",state)
        receipts.append(dict(study=study,source_plan=sd,local_plan=ld,prep_corpus=hh.hexdigest(),origin=lp[0].name))
        output.extend(records)
    write_csv(OUT/'warm-eligible-forks.csv',output)
    save_json(OUT/'warm-staging.json',dict(protocol=LOCK,roster_sha256=native.sha(ROSTER),sources=receipts,
        cohorts=sorted(totals,key=lambda r:r['index']),eligible_forks=len(output),reserve_group=3,
        route='Mandatory identity join to the static release by probe-plan seed, beta index and parent ID before selecting any fork or prep state; excludes all group-3 siblings and ages. Deflated member seeks skip nonselected rows; indexing never returns their contents. Outcome rows read: zero.'))


def run(lib):
    lock=json.loads((OUT/'fit-lock.json').read_bytes()); stages=json.loads((OUT/'warm-staging.json').read_bytes()); oracle=m.Oracle(lib)
    if native.sha(lib)!=lock['library_sha256'] or native.sha(OUT/'fit-cuts.npz')!=lock['cuts_sha256'] or native.sha(ROSTER)!=stages['roster_sha256']:
        raise ValueError('changed locked input')
    cuts=dict(np.load(OUT/'fit-cuts.npz')); summary=[]; parents=[]; curves=[]
    with (OUT/'warm-eligible-forks.csv').open() as f: rows=list(csv.DictReader(f))
    for study,targets in STUDIES:
        root=ARCHIVE/'runs'/study; _,ld,lp=plan(root/'physical'); data=[]
        for co in stages['cohorts']:
            if co['index'] not in targets:continue
            b=co['index']; rs=[r for r in rows if int(r['index'])==b]; origin={}; members=defaultdict(list)
            for j,r in enumerate(rs):
                key=(int(r['parent_id']),int(r['age'])); members[key].append(j)
                if key not in origin:origin[key]=np.load(WORK/f'origin-{b}-{key[0]}-{key[1]}.npy')
            # Physical siblings share a fork origin but each contributes its OWN first observation.
            t=np.array([0,*co['powers']]); first=np.full((len(rs),6000),len(t),np.uint8)
            data.append((co,rs,origin,members,t,first))
        for lag in sorted({int(lag) for co,rs,orig,mem,t,first in data for lag in t[1:]}):
            nm=checked_meta(lp[lag],lag,ld); lookup={int(c):i for i,c in enumerate(nm['ids'])}; req=[]
            for co,rs,orig,mem,t,first in data:
                tt=np.flatnonzero(t==lag)
                if not len(tt):continue
                for key,js in mem.items():
                    state=orig[key]; occ=state!=0; ref=state[occ];
                    for j in js:
                        r=rs[j]; row=lookup[int(r['child_id'])]
                        if (int(nm['parent_ids'][row]),int(nm['parent_age'][row]),nm['betas'][row],nm['parent_hash'][row]) != (*key,r['beta'],r['parent_hash']):raise ValueError('warm outcome lineage mismatch')
                        req.append((row,occ,ref,first[j],tt[0],len(t)))
            values=rows_only(lp[lag],[r[0] for r in req],len(nm['ids']))
            for (_,occ,ref,first,i,sentinel),packed in zip(req,values):
                changed=(packed[occ]&3)!=ref; first[changed&(first==sentinel)]=i
        for co,rs,origin,members,t,first in data:
            b=co['index']; beta=float(co['beta']); n=len(t); ages=defaultdict(dict); coverage_raw=defaultdict(dict)
            for (pid,age),js in members.items():
                # Compute features once per distinct origin, not per stochastic sibling.
                f=oracle(origin[pid,age],beta) if beta else None
                groups=m.groups(f,cuts,beta) if beta else {}
                for j in js:
                    if beta:
                        block=m.fork_risks(first[j],groups,n)
                    else:
                        block={'all_sites':({'overlap':0,'matched_low':0,'matched_high':0}, {'all_sites':m.cdf(first[j],n)})}
                    for mode,(support,c) in block.items():
                        curveset=ages[pid,age].setdefault(mode,defaultdict(list))
                        for measure,a in c.items():curveset[measure].append(a)
                        coverage_raw[pid,age][mode]=(support['matched_low']+support['matched_high'])/6000
            for pid in sorted({p for p,age in ages}):
                ag=[ages[pid,a] for a in co['ages']]
                for mode in ag[0]:
                    arr={name: np.array([np.mean(g[mode][name],0) for g in ag]).mean(0) for name in ag[0][mode]}
                    for name,val in arr.items():
                        curves.extend(dict(index=b,beta=beta,parent_id=pid,mode=mode,measure=name,lag=int(lag),risk=finite(v)) for lag,v in zip(t,val))
                    dy=arr['matched_high_risk'][-1]-arr['matched_low_risk'][-1] if beta else np.nan
                    cover=np.mean([coverage_raw[pid,age][mode] for age in co['ages']]) if np.isfinite(dy) else 0.
                    parents.append(dict(index=b,beta=beta,parent_id=pid,mode=mode,matched_coverage=cover,terminal_lag=int(t[-1]),
                                        all_site_terminal_risk=float(arr['all_sites'][-1]),paired_risk_difference=finite(dy)))
            for mode in m.MODES if beta else ['all_sites']:
                r=[p for p in parents if p['index']==b and p['mode']==mode]; effect=np.array([np.nan if p['paired_risk_difference'] is None else p['paired_risk_difference'] for p in r]);
                summary.append(dict(index=b,beta=beta,mode=mode,common_terminal=int(t[-1]),all_site_terminal_risk=float(np.mean([p['all_site_terminal_risk'] for p in r])),
                           diagnostic=(metrics(effect,len(r),float(np.mean([p['matched_coverage'] for p in r]))) if beta else None)))
    write_csv(OUT/'warm-parents.csv',parents);write_csv(OUT/'warm-parent-curves.csv',curves)
    save_json(OUT/'warm-secondary.json',dict(cuts=lock['cuts_sha256'],results=summary,
         note='Probe/preparation parents linked to native local forks only, neither pseudo-histories for the release nor part of cold gate. Means aggregate siblings, paired ages, then independent parents. Beta zero has no defined normalized opening score. Horizons differ by beta.'))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(14,6),sharey=True,constrained_layout=True)
    for ax,co in zip(axes.flat,stages['cohorts']):
        b=co['index']; r=[p for p in parents if p['index']==b and p['mode']==('opening+packing' if b else 'all_sites')]
        ax.scatter(range(len(r)),[p['all_site_terminal_risk'] for p in r],s=7,label='all-site risk');
        ax.scatter(range(len(r)),[np.nan if p['paired_risk_difference'] is None else p['paired_risk_difference'] for p in r],s=7,label='matched high−low')
        ax.axhline(0,color='gray',lw=.5); ax.set(title=f"β={float(co['beta']):.3g}; t={co['powers'][-1]:,}",xlabel='Non-reserve parent')
    axes.flat[0].legend(fontsize=7);fig.suptitle('Separate linked warm forks: matched-grid terminal, siblings and ages clustered')
    fig.savefig(OUT/'warm-secondary.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage','score']);p.add_argument('--qualified-library',type=Path);a=p.parse_args()
    if a.action=='stage':stage()
    else:run(a.qualified_library)
