"""Two registered secondary comparisons, including independent pair-law training uncertainty."""
import argparse,json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy.stats import t
from rate import partition,json_out,digest
from vacancy_dipole import selected_member
from fifteen_topology import pairfit,topology_table,at,PATCH,MOM,N
from fifteen_codec import spectral,dense,R


def counts(states):
    rows=[]
    for s in states:
        mask=sum((at(s,p)!=0).astype('i2')*(1<<i) for i,p in enumerate(PATCH));rows.append(np.bincount(mask.ravel(),minlength=4096))
    return np.array(rows)


def main(a):
    o=a.out;target=o/'04-10-transfer.json';lock=o/'04-10-lock.json';reg=json.loads((o/'01-validation-lock.json').read_bytes());inputbank=a.bank/'b07.npz';src=a.data/'beta-07.npz';doc=json.loads((o/'01-decision.json').read_bytes());train=partition(7)[0][0][:256];val=partition(7)[0][1][:512]
    if target.exists() or digest(lock)!=a.lock_sha256 or digest(src)!=reg['source_release_sha256'] or digest(inputbank)!=doc['issued_bank_sha256'] or np.intersect1d(train,val).size:raise ValueError('changed roster, lock, or existing evaluation')
    with ZipFile(src) as z:x=counts(selected_member(z,'species',train,(20,20,20)))
    h=(topology_table()>0).astype(float);p,err=pairfit(x.sum(0));q=p@h;mu=p@MOM;dev=MOM-mu;fisher=dev.T@(p[:,None]*dev);coef=np.linalg.solve(fisher,dev.T@(p*(h-q)));influence=(x@MOM/N-mu)@coef;trainse=influence.std(ddof=1)/np.sqrt(len(train));crit=t.ppf(1-.05/(2*15),255)
    with np.load(inputbank) as z:
        if not np.array_equal(z['parent'],val):raise ValueError('wrong validation bank order')
        states=z['species'];outcomes=counts(states)@h/N
    vs=outcomes.std(ddof=1)/np.sqrt(len(val));se=np.hypot(trainse,vs);gap=outcomes.mean()-q;top=dict(predicted_hole_probability=q,observed_hole_probability=outcomes.mean(),gain=gap,training_parents=len(train),validation_parents=len(val),train_se=trainse,validation_se=vs,family15=[gap-crit*se,gap+crit*se],conditional95=[gap-t.ppf(.975,511)*vs,gap+t.ppf(.975,511)*vs],gradient_error=err,fisher_cond=float(np.linalg.cond(fisher)),mde80_family_approx=(crit+.842)*se,pass_margin=bool(gap-crit*se>=.001));tables=np.load(R/'codec/comparators.npz');pw=[]
    for s in states:
        w=dense(s);pw.append([spectral(tables[k][w])[0] for k in ('rsmi_lut','composition_lut')])
    pw=np.array(pw);A=pw[:,:,1:].sum(2);D=pw.sum(2);r=A.sum(0)/D.sum(0);z=((A-r*D)/D.mean(0));d=r[0]-r[1];s=(z[:,0]-z[:,1]).std(ddof=1)/np.sqrt(len(z));hcrit=t.ppf(1-.05/(2*15),511);codec=dict(historical_alias_fraction=r[0],composition_alias_fraction=r[1],difference=d,paired_se=s,family15=[d-hcrit*s,d+hcrit*s],pass_margin=bool(d-hcrit*s>=.05),mde80_family_approx=(hcrit+.842)*s);rows=o/'04-10-parents.npz';np.savez_compressed(rows,train_ids=train,train_influence=influence,val_ids=val,hole_rates=outcomes,alias_power=pw);json_out(target,dict(lock_sha256=a.lock_sha256,implementation_sha256=digest(Path(__file__)),parent_evidence_sha256=digest(rows),beta=2.372222222222,topology=top,aliasing=codec,scope='Prelocked 04/10 secondary contrasts, parent-based Bonferroni 15; beta7 validation, reused archive. Diagnostics at one L, not bulk phases.'));print(top,codec,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--lock-sha256',required=True);main(p.parse_args())
