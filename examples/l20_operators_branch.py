"""Exact, label-free two-move diagnostic for a free bond (six oriented faces).

An available continuation is a new unlike non-uphill exchange incident on an
endpoint of the first move, other than reversal. A continuation need not occur.
"""
import itertools
import numpy as np

IDEAL=np.array([0,3,5],np.int16)
STEPS=np.array([np.eye(3,dtype=int)[a]*s for a in range(3) for s in (-1,1)])


def branching(origin, parent, anchor, axis, sign):
    L=origin.shape[1]
    m=sum(np.roll((origin!=0).astype(np.int16),d,ax) for ax in (1,2,3) for d in (-1,1))
    n=len(parent); released=np.zeros((n,6),np.int8); lost=np.zeros_like(released)
    central=np.empty(n,np.int16)
    for ax,sgn in itertools.product(range(3),(-1,1)):
        ix=np.flatnonzero((axis==ax)&(sign==sgn))
        if not len(ix):continue
        p=parent[ix]; pos=anchor[ix]; basis=np.eye(3,dtype=int)[[ax,(ax+1)%3,(ax+2)%3]];basis[0]*=sgn
        u=np.array([1,0,0]);a=np.zeros(3,dtype=int);b=u
        def gather(rel):
            xyz=(pos[:,None,:]+np.asarray(rel)@basis)%L
            idx=(p[:,None],xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]);return origin[idx],m[idx]
        def changes(ends, outer, before, after):
            s0,m0=gather(outer);amap=(np.abs(outer[:,None,:]-ends[None,:,:]).sum(-1)==1).astype(np.int16)
            same=(outer[:,None,:]==ends[None,:,:]).all(-1);sites,labels=np.nonzero(same)
            def energy(V):
                s=s0.copy();s[:,sites]=V[:,labels]
                counts=m0+((V!=0).astype(np.int16)-(before!=0)).astype(np.int16)@amap.T
                return ((s!=0)*(counts-IDEAL[s])**2).sum(1,dtype=np.int32)
            return energy(after)-energy(before)
        for origin_end, e in [(a,0),(b,1)]:
            for direction,step in enumerate(STEPS):
                c=origin_end+step
                if np.array_equal(c,b if e==0 else a):continue
                triplet=np.array([a,b,c]);s,_=gather(triplet); first=s[:,[1,0,2]]
                if np.any(s[:,0]>=s[:,1]): raise ValueError('not a lower-to-higher unlike bond')
                other=[0,1,2];other[e],other[2]=2,e
                outer=np.unique(np.concatenate(([origin_end,c],origin_end+STEPS,c+STEPS)),axis=0)
                d0=changes(triplet,outer,s,s[:,other]);d1=changes(triplet,outer,s,first[:,other])-changes(triplet,outer,s,first)
                old=(s[:,e]!=s[:,2])&(d0<=0);new=(first[:,e]!=first[:,2])&(d1<=0)
                released[ix,direction]+=(~old&new);lost[ix,direction]+=(old&~new)
        ends=np.array([a,b]);s,_=gather(ends);affected=np.unique(np.concatenate((ends,a+STEPS,b+STEPS)),axis=0)
        central[ix]=changes(ends,affected,s,s[:,::-1]);
    if np.any(central>0): raise ValueError('non-free central bond')
    return released,lost,central


def whole_h(s):
    occ=(s!=0).astype(np.int16);m=sum(np.roll(occ,k,d) for k in (-1,1) for d in range(3))
    return np.sum(occ*(m-IDEAL[s])**2,dtype=np.int64)


def literal(s,a,ax,sgn):
    """Whole-lattice two-successive-swap check; independent from local energy formula."""
    base=whole_h(s); a=np.array(a);n=s.shape[0];u=np.eye(3,dtype=int)[ax]*sgn;b=(a+u)%n
    sc=s.copy();ta,tb=tuple(a),tuple(b);sc[ta],sc[tb]=s[tb],s[ta];now=whole_h(sc)
    released=np.zeros(6,int);lost=np.zeros(6,int)
    for i,origin in enumerate((a,b)):
        for d,step in enumerate(STEPS):
            axis=int(np.flatnonzero(step)[0]);gaxis=(axis+ax)%3;sign=int(step[axis])*(sgn if axis==0 else 1)
            c=origin.copy();c[gaxis]=(c[gaxis]+sign)%n;x,y=tuple(origin),tuple(c)
            if y==(ta if i else tb):continue
            def active(state,h):
                if state[x]==state[y]:return False
                v=state.copy();v[x],v[y]=state[y],state[x];return whole_h(v)<=h
            old,new=active(s,base),active(sc,now);released[d]+=new and not old;lost[d]+=old and not new
    return released,lost,now-base
