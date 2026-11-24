"""Analytic packing templates at exactly the binary glass density and composition.

Not equilibrated configurations: this is a finite candidate dictionary of periodic
crystal patterns used to score the observed boundary. No archive or MCMC is used.
"""
from functools import lru_cache
from itertools import combinations,product
import numpy as np

SHELL=np.array([r for r in product(range(-4,5),repeat=3) if max(map(abs,r))==4]); PERIOD=(2,2,5)


@lru_cache(maxsize=1)
def dictionary():
    # Every five-vacancy decoration of this 20-site cell, including every phase.
    # For each geometry the best exact six A labels are the six lowest energy
    # differences d_k=(k-3)^2-(k-5)^2=4k-16. Others are B. Count BOTH bonds
    # along a period-two axis: they are distinct neighbors in the infinite crystal.
    ids=np.array(list(combinations(range(20),5)));vac=np.zeros((len(ids),20),np.int8);vac[np.arange(len(ids))[:,None],ids]=1
    occ=(1-vac).reshape(-1,*PERIOD);m=sum(np.roll(occ,sign,axis) for sign,axis in product((-1,1),(1,2,3))).reshape(-1,20);db=np.where(vac!=0,100,4*m.astype('i4')-16)
    E=((1-vac)*(m-5)**2).sum(1)+np.partition(db,5,axis=1)[:,:6].sum(1);templates=np.ascontiguousarray(vac[E==E.min()].T,dtype='f');projections=[]
    for axis in range(3):
        xyz=SHELL[:,[k for k in range(3) if k!=axis]+[axis]]%PERIOD;idx=np.ravel_multi_index(xyz.T,PERIOD);projection=np.eye(20,dtype='f')[idx];mass=projection.sum(0)@templates/len(SHELL);inv=np.divide(1,np.sqrt(mass*(1-mass)),where=(mass>0)&(mass<1),out=np.zeros_like(mass));projections.append((projection,mass,inv))
    return templates,projections,int(E.min())


def packing20(x):
    """Four cubic-invariant similarities of observed vacancy shell to registered crystals."""
    if x.ndim!=2 or x.shape[1]!=len(SHELL):raise ValueError('packing20 requires radius-four shell')
    templates,p,_=dictionary();out=np.empty((len(x),4),dtype='f');n=len(SHELL)
    for a in range(0,len(x),2048):
        b=np.asarray(x[a:a+2048],dtype='f');rate=b.mean(1)[:,None];inv=np.divide(1,np.sqrt(rate*(1-rate)),out=np.zeros_like(rate),where=(rate>0)&(rate<1));best=[];variation=[]
        for align,base,norm in p:
            cov=(b@align)@templates/n-rate*base;cor=cov*inv*norm;best.append(cor.max(1));variation.append(cor.std(1))
        best=np.column_stack(best);peak=best.max(1);out[a:a+len(b)]=np.column_stack((peak,best.mean(1),np.max(variation,axis=0),peak-np.median(best,axis=1)))
    return out
