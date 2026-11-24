"""Synthetic-only offline V3 fold integrity, exact continuation, fixed controls and refit qualification."""
import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path
import tempfile
import time

import numpy as np
from full_hero_train import Evaluation, Trainer, device_setup, sha
from full_hero_v3_sweep import (BASELINES, BETAS, SEEDS, dump_new, final_refit, frozen_maps, hard_oracle, load_arrays,
                                make_data, one_run, reference_rate, restore, save_optimizer, state_hash)
from full_hero_v3_transport import unpack_e


def main():
    t=time.monotonic();info=device_setup('cpu');rng=np.random.default_rng(141220);proof={}
    pub=Path(__file__).resolve().parents[2]/'results/l20-autoresearch/hero-full-v3'
    gate=pub.parent/'hero-full-v2'/'control-gate.json'
    with tempfile.TemporaryDirectory(prefix='hero-v3-nonphysical-') as tmp:
        root=Path(tmp);src=root/'input';src.mkdir();out=root/'screen';out.mkdir()
        for q in ('checkpoints','optimizer','learning'):(out/q).mkdir()
        def document(name,fields):
            path=root/name;dump_new(path,dict(source='synthetic-only software qualification fixture',**fields));return sha(path)
        design=document('design.json',{'purpose':'no physical stage authority, no physical data'})
        split=document('split.json',{'train':81920,'cross_side_intersections':0,'design_sha256':design})
        roster=hashlib.sha256(b'synthetic-only').hexdigest()
        release=document('training-release-1.json',dict(schema='hero-full-v3-training-release-1',stage='train-only',indices=list(BETAS),
                        candidate_folds=list(range(2,8)),selection_fold=0,rule_fold=1,allows_test=False,permanent_train=81920,
                        primary_seed=6101,seed_stability=list(SEEDS),design_sha256=design,split_sha256=split,
                        rosters_sha256=roster,control_gate_sha256=sha(gate)))
        folder=src/'b00/train';folder.mkdir(parents=True)
        v=rng.integers(3,size=(8192,64,27),dtype='u1');p=np.arange(8192,dtype='<i4')
        vals=np.arange(81);table=sum((vals//3**j%3)<<(2*j) for j in range(4)).astype('u1')
        packed=table[rng.integers(81,size=(8192,64,97),dtype='u1')];packed[...,-1]&=15
        for k,a in [('V',v),('E2',packed),('parent',p)]:np.save(folder/(k+'.npy'),a)
        files={};pins={str(b):roster for b in BETAS}
        for b in BETAS:
            d=src/f'b{b:02d}/train';d.mkdir(exist_ok=True,parents=True)
            for f in ('V.npy','E2.npy','parent.npy'):
                if b:os.link(folder/f,d/f)
                files[str((d/f).relative_to(src))]=sha(d/f)
        md=dict(schema='hero-full-v3-offline-training-input-1',rosters_sha256=roster,split_sha256=split,files=files,
                stream_receipts=pins,train_only=True,beta_indices=list(BETAS),encoding='E2-le2bit-v1',source='synthetic-only')
        dump_new(src/'manifest.json',md)
        args=Namespace(release=root/'training-release-1.json',release_sha256=release,control_gate=gate,panel='full-primary',
                       source='synthetic-only',offline_root=src,manifest_sha256=sha(src/'manifest.json'),out=out,device='cpu',seed=None)
        data,_=load_arrays(args)
        for b in BETAS:
            assert np.array_equal(data['fit'][b]['parent'],p[p%8>=2]) and np.array_equal(data['dev'][b]['parent'],p[::8])
            ix=np.array([[0,12,78],[320,1024,120]])
            got=data['fit'][b]['E'].reshape(-1,386)[ix]
            assert np.array_equal(got,unpack_e(packed[p[p%8>=2][ix//64],ix%64]))
        # Downsized only AFTER exercising genuine full-shape byte verification and fold filtering.
        def subset(role,n):return {b:{k:a[:n] for k,a in d.items()} for b,d in role.items()}
        fit=make_data(subset(data['fit'],96),1);dev=make_data(subset(data['dev'],64),1,True)
        fe=Evaluation(fit,1701,groups=64);de=Evaluation(dev,1901,groups=64)
        for arch,law in [('affine','uniform'),('tanh','matched')]:
            z,_=one_run(args,info,fit,dev,fe,de,1,arch,law,6101,depths=(128,))
            name=f'{arch}-{law}-d1-s6101-u128';r=json.loads((out/'learning'/(name+'.json')).read_text())
            r['optimizer_file_sha256']=save_optimizer(out/'optimizer'/(name+'.npz'),z)
            other=Trainer(fit,arch,law,6101,'cpu');restore(other,out,r)
            orig=z.advance(256);resumed=other.advance(256)
            assert state_hash(z)==state_hash(other)
            assert orig['beta_positive_exposures']==resumed['beta_positive_exposures']
            proof[arch+'-'+law]=dict(continuous_and_resumed_state_sha256=state_hash(z),paired_draw_hashes_equal=True,
                                     actual_continuous_updates=256,actual_resumed_updates=128)
        cp=out/'checkpoints'/(name+'.npz');ref=reference_rate(fit,{'path':cp.name,'sha256':sha(cp)},[out])
        specs={}
        for ctr in BASELINES:
            ff,fd,sp=frozen_maps(fit,dev,ctr,ref)
            if ctr.startswith('random_'):assert sp['target_mass']==ref['descending_target_mass']
            z,_=one_run(args,info,fit,dev,fe,de,1,ctr,'uniform',6101,ff,fd,sp,depths=(32,))
            specs[ctr]=dict(achieved=sp['achieved_equal_beta_mass'],target=sp['target_mass'])
        allone={**ref,'descending_target_mass':[1.,0.,0.,0.]}
        _,_,collapsed=frozen_maps(fit,dev,'random_tanh',allone)
        assert collapsed['achieved_equal_beta_mass']==[1.,0.,0.,0.]
        nominee=dict(schema='hero-full-v3-nomination-1',scope='training-fold0-selection',status='qualified',
                     split_sha256=split,rosters_sha256=roster,primary_seed=6101,arch='affine',law='uniform',depth=1024)
        nd=document('nomination.json',nominee)
        rd=document('refit-release.json',dict(schema='hero-full-v3-refit-release-1',all_train=True,allows_test=False,
                    split_sha256=split,rosters_sha256=roster,design_sha256=design,seeds=list(SEEDS),primary_seed=6101,
                    nomination_sha256=nd,arch='affine',law='uniform',depth=1024))
        args.panel='refit';args.refit_release=root/'refit-release.json';args.refit_release_sha256=rd;args.nomination_file=root/'nomination.json'
        alltrain,pins=load_arrays(args)
        assert list(alltrain)==['fit'] and all(np.array_equal(d['parent'],p) for d in alltrain['fit'].values())
        args.out=root/'refit';args.out.mkdir();(args.out/'checkpoints').mkdir()
        final_refit(args,info,subset(alltrain['fit'],96),pins)
        rr=json.loads((args.out/'refit.json').read_text());assert not rr['allows_test'] and len(rr['checkpoints'])==5 and len(rr['controls'])==6
    dump_new(pub/'synthetic-offline-lifecycle.json',dict(source='synthetic-only',physical_inputs=0,device=info,
             n_beta=10,full_layout_before_subsampling=8192,verified_fit_fold_counts=[6144,1024],
             reference_rate=ref,matched_controls=specs,collapse_reference_handled=True,hard_direct_oracle=hard_oracle('cpu'),
             resume=proof,refit_fixture_uses_96_per_beta=96,actual_refit_updates=11*1024,refit_weights=11,
             no_test_reader=True,full_input_shape_validated=True,elapsed_wall_sec=time.monotonic()-t))


if __name__=='__main__':main()
