"""Measure the hard child adjoint against exhaustive finite child edits on abstract category tuples; no lattice states."""
from tranche2_hierarchy import WORK, Kernels, dump
import numpy as np


def main():
    k=Kernels(WORK)
    # Every category occurs once per row; changing a shared LUT entry changes no more
    # than one coordinate of a row. Thus its full finite score difference must equal
    # the sum of its single-occurrence derivatives, not just a first-order approximation.
    w=np.stack([np.roll(np.arange(8,dtype=np.uint16),j%8) for j in range(81)])
    y=np.arange(81,dtype=np.uint8);child=(np.arange(6561)%4).astype(np.uint8)
    t=np.arange(65536,dtype=np.uint32);top=((t%7+t//19+(t//213)%11)%4).astype(np.uint8)
    value=np.sin(np.arange(324).reshape(4,81)*.071);tup=k.codes(w,child);g=np.zeros((6561,4))
    k.run('adjoint',len(y),w,tup,y,top,value,g);actual=float(value[top[tup],y].sum());err=[];rise=[]
    for word in range(8):
        for label in range(4):
            new=child.copy();new[word]=label;observed=float(value[top[k.codes(w,new)],y].sum())-actual
            expected=g[word,label]-g[word,child[word]];err.append(abs(observed-expected));rise.append(abs(observed))
    assert max(err)<1.e-11 and max(rise)>1.
    dump('25-gradient-control.json',{'nonphysical_categorical_rows':len(y),'finite_differences':len(err),'max_abs_score_change':max(rise),'max_abs_error':max(err),'interpretation':'One occurrence per row: adjoint agrees with exact shared-word finite replacements. Actual TRAIN repeats can interact; all bulk updates are accepted only after exact recomputation of the hard forward score.'})


if __name__=='__main__':main()
