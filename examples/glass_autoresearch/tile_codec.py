"""Lossy L20 glass codec. NumPy only. Symbols are 2-bit labels, not microscopic spins.
Example: python tile_codec.py --model codec.npz --input lattice.npy --output compressed.npy
Use --predict to read packed arrays and print predictions from the frozen cold probes.
"""
import argparse
from pathlib import Path
import numpy as np

TILE=np.array([(z,y,x) for z in range(2) for y in range(2) for x in range(2)])
POW=(3**np.arange(8)).astype(np.uint16)
UP=np.triu_indices(3)


def tile_index(s):
    s=np.asarray(s)
    if s.ndim==3:s=s[None]
    if s.ndim!=4 or s.shape[1:]!=(20,20,20) or s.dtype.kind not in 'iu' or s.min()<0 or s.max()>2:raise ValueError('expected L20 z,y,x arrays with species 0,1,2')
    idx=np.zeros((len(s),10,10,10),dtype=np.uint16)
    for (z,y,x),w in zip(TILE,POW):idx+=s[:,z::2,y::2,x::2].astype(np.uint16)*w
    return idx


def pack(labels):
    h=np.asarray(labels).reshape(-1,250,4)
    if h.dtype!=np.uint8 or h.max()>3:raise ValueError('bad coarse symbols')
    return h[:,:,0]|(h[:,:,1]<<2)|(h[:,:,2]<<4)|(h[:,:,3]<<6)


def unpack(b):
    b=np.asarray(b)
    if b.ndim!=2 or b.shape[1]!=250 or b.dtype!=np.uint8:raise ValueError('expected uint8 [parents,250]')
    return ((b[:,:,None]>>(2*np.arange(4,dtype=np.uint8)))&3).reshape(-1,10,10,10)


def features(h):
    """Only coarse-symbol counts, their pair counts and coarse low-wave modes."""
    h=np.asarray(h);n=len(h);off=16*np.arange(n).reshape(-1,1,1,1);x=(h[:,:,:,:,None]==np.arange(3));mass=x.mean((1,2,3));cols=[mass,(mass[:,:,None]*mass[:,None,:])[:,UP[0],UP[1]]];low=np.zeros((n,3,3));wave=np.exp((-2j*np.pi/10)*np.arange(10))
    for axis in range(1,4):
        key=off+4*h+np.roll(h,-1,axis);cols.append(np.bincount(key.ravel(),minlength=16*n).reshape(n,16)/1000.)
        profile=x.sum(tuple(d for d in (1,2,3) if d!=axis));amp=np.einsum('pqc,q->pc',profile,wave);low+=(amp[:,:,None]*amp[:,None,:].conj()).real/3000.
    cols.append(low[:,UP[0],UP[1]]);return np.column_stack(cols)


def predict(packed,model):
    return (features(unpack(packed))-model['mean'])/model['scale']@model['coef']+model['intercept']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--predict',action='store_true');a=p.parse_args();x=np.load(a.input,mmap_mode='r',allow_pickle=False)
    with np.load(a.model,allow_pickle=False) as m:
        if a.predict:
            if a.output is None:print(' '.join(m['targets']));print(predict(x,m));return
            out=np.lib.format.open_memmap(a.output,mode='w+',dtype='d',shape=(len(x),len(m['targets'])))
            for i in range(0,len(x),32):out[i:i+32]=predict(x[i:i+32],m)
        else:
            if a.output is None:p.error('--output is required when encoding')
            if x.ndim==3:x=x[None]
            out=np.lib.format.open_memmap(a.output,mode='w+',dtype='u1',shape=(len(x),250))
            for i in range(0,len(x),32):out[i:i+32]=pack(m['lut'][tile_index(x[i:i+32])]);
        out.flush()


if __name__=='__main__':main()
