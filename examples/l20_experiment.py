"""Run one CPU shard of the frozen L20 temperature / buffer / operator grid."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--shard',type=int,default=0); ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args()
    if not 0 <= args.shard < args.shards:
        ap.error('invalid shard')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',WANDB_MODE='disabled',MPLBACKEND='Agg')
    from l20_pairs import load_banks
    from l20_fit import fit
    cells=[(i,b) for i in (9,0,8,1,7,2,6,3,5,4) for b in (1,0,2)]
    for i,b in cells[args.shard::args.shards]:
        meta,train,val,test=load_banks(args.data/f'beta-{i:02d}.npz',b,24)
        fits=[('learned',s) for s in (11,29,47)] + [('uniform',11),('staggered',11)]
        if i==9:
            fits.extend((m,11) for m in ('site','axis','face'))
        for method,seed in fits:
            folder=args.out/('fits' if method in ('learned','uniform','staggered') else 'cold-controls')
            folder.mkdir(parents=True,exist_ok=True); dest=folder/f'beta-{i:02d}_b{b}_{method}_{seed}.json'
            if dest.exists():
                raise FileExistsError(f'refusing to overwrite an existing fit: {dest}')
            start=time.monotonic(); row=fit(meta,train,val,test,method,seed)
            tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');tmp.replace(dest)
            print(f'{dest.name} epoch={row["best_epoch"]} seconds={time.monotonic()-start:.1f}',flush=True)
        del train,val,test


if __name__=='__main__':
    main()
