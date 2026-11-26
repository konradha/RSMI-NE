"""Route immutable parent IDs before indexing species. No off-sector assignments."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from stock_glass_cavity_exact import CODE, C, LOOK, X
from stock_glass_buffered_banks import cold_source
from stock_glass_cold_banks import _snapshot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/frozen-field-distance'
ROSTER = ROOT/'results/buffered-static-gate1/roster.npz'
ROSTER_SHA = '2c82ec7a7d830fad9f55af5872634e6fdc3d6fbac8e32f2c5da5bc4ac6f04b15'
INDEX = ROOT/'results/opening-mechanism/inputs.json'
COLD = 'l20-b320-b340-128x2-500m-aging-v1/prep'
COHORTS={'source-train':(9,(0,1),8192),'source-test':(9,(2,),4096),'beta-zero':(0,(2,),4096),'transfer-320':(10,(2,),128),'transfer-340':(11,(2,),128)}
DISTS=(3,4,5,6,7,8,9)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save_json(path,rec):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def need(x,msg):
    if not x:raise ValueError(msg)

def star_states(lat):
    # exact.C traverses each storage axis (-,+), in z,y,x order.
    code=lat.astype(np.int16)*int(CODE[0]);
    for i,c in enumerate(C[1:]):
        ax=int(np.flatnonzero(c)[0]);code+=int(CODE[i+1])*np.roll(lat,-int(c[ax]),axis=ax).astype(np.int16)
    si=LOOK[code].astype(np.int16);si[si==255]=-1
    return si

def route(tag):
    b,groups,count=COHORTS[tag]; need(sha(ROSTER)==ROSTER_SHA,'roster digest');
    with np.load(ROSTER) as z:
        ids=z['ids']; ix=np.flatnonzero((z['beta_index']==b)&np.isin(z['group'],groups));ix=ix[np.argsort(z['native_rows'][ix])]; rows=z['native_rows'][ix];keys=np.ascontiguousarray(ids[ix]);
        need(len(np.unique(ids,axis=0))==len(ids)==164096,'archive/cold identities not disjoint')
        need(len(ix)==len(np.unique(rows))==count and np.all(z['group'][ix]!=3),'forbidden or missing parent')
    return rows,keys

def make_bank(tag,runs,staged,scratch):
    b,_,n=COHORTS[tag]; rows,keys=route(tag); scratch.mkdir(parents=True,exist_ok=True)
    if tag!='source-train':
        gate=json.loads((OUT/'admission.json').read_text())
        need(gate['held_spatial_scoring_permitted'] and all(sha(OUT/f)==h for f,h in gate['pins'].items()),'held access not qualified')
    if b<10:
        m=json.loads(INDEX.read_text())['rows'][b];f=staged/f'b{b:02d}.npy';source=runs/m['source'];need(sha(f)==m['staged_sha256'] and sha(source)==m['archive_sha256'],'archive/staging digest');
        with np.load(source) as z:np.testing.assert_array_equal(np.column_stack([z['chain_seed'][rows],z['chain_id'][rows]]),keys)
        # mmap, index only admitted rows; file digest is an integrity check not a field evaluation.
        native=np.load(f,mmap_mode='r');pin=m['archive_sha256']
    else:
        prep=runs/COLD;file=cold_source(prep); p=json.loads((prep/'plan.json').read_text());cid,native,_,_=_snapshot(file,500,sha(prep/'plan.json'));np.testing.assert_array_equal(keys[:,1],cid[rows]);need(np.all(keys[:,0]==p['seed']),'cold lineage');pin=sha(file)
    tmp=scratch/(tag+'.partial.npy');out=np.lib.format.open_memmap(tmp,mode='w+',shape=(n,20,20,20),dtype=np.int16)
    hm=np.zeros(210);nel=np.zeros(n,np.uint16);audit=False
    for i,k in enumerate(rows):
        x=native[k];s=star_states(x);out[i]=s;v=s[s>=0];nel[i]=len(v)
        if tag=='source-train' and len(v):hm+=np.bincount(v,minlength=210)/len(v)
        if not audit:
            # Independent lookup for every center in this first admitted parent, including wrapped/ineligible stars.
            xyz=np.indices((20,20,20)).reshape(3,-1).T;loc=(xyz[:,None]+C)%20;states=x[tuple(loc.transpose(2,0,1))];ok=((states==0).sum(1)==2)&((states==1).sum(1)==2);np.testing.assert_array_equal(s.ravel()>=0,ok);np.testing.assert_array_equal(X[s.ravel()[ok]],states[ok]);audit=True
    out.flush();del out;dest=scratch/(tag+'.npy');tmp.replace(dest)
    np.savez(scratch/(tag+'-parents.npz'),ids=keys,native_rows=rows,eligible=nel)
    rec=dict(cohort=tag,parents=n,unique=len(np.unique(keys,axis=0)),parents_with_star=int((nel>0).sum()),sites=n*8000,eligible_stars=int(nel.sum()),eligible_per_parent_range=[int(nel.min()),int(nel.max())],ids_sha256=hashlib.sha256(keys.astype('<u8',copy=False).tobytes()).hexdigest(),ordered_rows_sha256=hashlib.sha256(rows.astype('<u2',copy=False).tobytes()).hexdigest(),species_archive_sha256=pin,star_bank_sha256=sha(dest),metadata_sha256=sha(scratch/(tag+'-parents.npz')),reserve_outcomes_accessed=0,native_axis_order='z,y,x')
    if tag=='source-train':
        need((nel>0).all(),'source parent without eligible star');np.save(OUT/'train-mass.npy',hm/n);rec['train_mass_sha256']=sha(OUT/'train-mass.npy')
    save_json(OUT/(tag+'-input.json'),rec); print(f'STAGED {tag}: {n} parents, {nel.sum()} stars',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('tag',choices=COHORTS);a.add_argument('runs',type=Path);a.add_argument('staged',type=Path);a.add_argument('scratch',type=Path);p=a.parse_args();make_bank(p.tag,p.runs,p.staged,p.scratch)
