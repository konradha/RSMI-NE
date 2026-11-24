"""Freeze one stable-ID half split per temperature. Reads no physical outcome members."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
PUB=ROOT/'results/l20-autoresearch/hero-full-v3'
WORK=Path.home()/'rsmi-work/l20-autoresearch-1795520725/hero-full-v3'
DATA=Path.home()/'datasets/lattice-glass/l20-16384-v1'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def split_keys(pairs, seed):
    domain=b'hero-full-v3/split\0'+struct.pack('<Q',seed)
    return sorted(range(len(pairs)),key=lambda r:(hashlib.sha256(domain+struct.pack('<QQ',*pairs[r])).digest(),pairs[r]))


def main():
    design=json.loads((PUB/'design.json').read_bytes()); src=DATA/'release-index.json'
    if sha(src)!=design['source_manifest_sha256']:
        raise ValueError('archive manifest hash mismatch')
    parents={}; registered={}; roster={}; tables={}; splitsets=[set(),set()]; raw_rows=[]
    entries=sorted(json.loads(src.read_bytes())['endpoints'],key=lambda v:v['release_index'])
    if [v['release_index'] for v in entries]!=list(range(10)):
        raise ValueError('expected all ten native beta indices')
    for v in entries:
        b=v['release_index']; path=DATA/v['npz']
        if sha(path)!=v['npz_sha256']:
            raise ValueError(f'changed native archive at index {b}')
        with np.load(path,allow_pickle=False) as z:
            s,i=z['chain_seed'],z['chain_id']
        if s.shape!=(16384,) or i.shape!=(16384,) or s.dtype.kind!='u' or s.dtype.itemsize!=8 or i.dtype.kind not in 'ui' or i.min()<0:
            raise ValueError(f'bad uint64 seed/ID metadata at index {b}')
        pairs=list(zip(map(int,s),map(int,i)))
        for r,k in enumerate(pairs):
            if k in parents:
                raise ValueError(f'duplicate physical lineage {k} at {parents[k]} and {(b,r)}; refusing naive independent assignments')
            parents[k]=(b,r)
        order=split_keys(pairs,int(design['seed'])); tab={}
        for side,label in enumerate(['train','test']):
            a=np.array(order[side*8192:(side+1)*8192],dtype='<i4'); key=f'b{b:02}_{label}'
            matrix=np.asarray([pairs[r] for r in a],dtype='<u8'); splitsets[side].update(pairs[r] for r in a)
            tab[label]=len(a); tab[label+'_identity_sha256']=hashlib.sha256(matrix.tobytes()).hexdigest(); roster[key]=a
            registered[key]=hashlib.sha256(a.tobytes()).hexdigest()
            for j,r in enumerate(a):
                raw_rows.append((b,v['beta'],int(r),*pairs[r],label,j,j%8 if side==0 else ''))
        tables[str(b)]={'beta':v['beta'],'native_parents':len(pairs),'native_identity_sha256':hashlib.sha256(np.asarray(pairs,dtype='<u8').tobytes()).hexdigest(),**tab}
        print(b,v['beta'],tab['train'],tab['test'],flush=True)
    if len(parents)!=163840 or any(len(a)!=81920 for a in splitsets) or splitsets[0]&splitsets[1]:
        raise ValueError('archive cardinality or cross-side global identity failure')
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'rosters.npz').open('xb') as f:
        np.savez(f,**roster)
    with (PUB/'split-rows.csv.gz').open('xb') as f, gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as gz, io.TextIOWrapper(gz,encoding='ascii',newline='') as txt:
        w=csv.writer(txt,lineterminator='\n'); w.writerow(['index','beta','native_row','chain_seed','chain_id','side','rank_in_side','inner_training_fold']); w.writerows(raw_rows)
    result={'schema':'hero-full-v3-exact-half-split-1','seed':design['seed'],'design_sha256':sha(PUB/'design.json'),'source_manifest_sha256':sha(src),'identity':['chain_seed','chain_id'],'physical_member_reads':0,'global_unique_parents':len(parents),'cross_side_intersections':len(splitsets[0]&splitsets[1]),'train':len(splitsets[0]),'test':len(splitsets[1]),'per_beta':tables,'rosters_sha256':sha(WORK/'rosters.npz'),'rows_sha256':sha(PUB/'split-rows.csv.gz'),'ordered_native_int32le_roster_sha256':registered,'source':'Direct read of chain_seed and chain_id from all ten manifest-pinned native NPZs, with exact unsigned 64-bit serialization. No old roster/scores used; no TEST species read.'}
    with (PUB/'split.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print('TRAIN',result['train'],'TEST',result['test'],'shared identities',result['cross_side_intersections'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.parse_args(); main()
