"""Exact polyphase Fourier decomposition and locally erasable equal-energy label pairs."""
import argparse,json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.stats import t
from fifteen_collective import dense,R
from fifteen_topology import at,interval
from tile_codec import TILE,POW
from rate import digest,json_out

N=8000;L=20;XYZ=np.indices((L,)*3).reshape(3,-1).T;OFF=np.array(list(product((0,1),repeat=3)));WALSH=(-1)**(OFF@OFF.T);AX=np.eye(3,dtype=int);DIR=np.r_[AX,-AX]

def spectral(labels):
    # All coefficients normalized by number of real-space cells in their own lattice.
    # Use all four one-hot indicators (sum of color powers), not arbitrary numeric label IDs.
    u=(labels[...,None]==np.arange(4)).astype(float);u-=u.mean((0,1,2),keepdims=True);power=np.zeros(8);maxerr=0.
    for axis in range(3):
        k=AX[axis];terms=[]
        for r in OFF:
            sub=u[r[0]::2,r[1]::2,r[2]::2];axes=tuple(i for i in (0,1,2) if i!=axis)
            terms.append((np.exp(-2j*np.pi*np.arange(10)/10)@sub.mean(axes))/10*np.exp(-2j*np.pi*(k@r)/20))
        terms=np.stack(terms);true=WALSH@terms/8
        f=np.fft.fftn(u,axes=(0,1,2))/N;v=f[tuple((k+10*OFF).T)];error=np.max(abs(true-v));maxerr=max(maxerr,error)
        if error>1e-11 or np.max(abs(WALSH@true-terms))>1e-11:raise ArithmeticError('staggered Fourier decomposition mismatch')
        pw=(abs(true)**2).sum(1);power+=pw
        if abs(pw.sum()-np.mean((abs(terms)**2).sum(1)))>1e-12:raise ArithmeticError('alias-power sum rule failed')
    return power,maxerr


def prep_edges():
    tables=[]
    for e in AX:
        # Tile origins incident to either endpoint; both altered digits appear together for shared tiles.
        origins=sorted(set(tuple(-v) for v in TILE)|set(tuple(e-v) for v in TILE));orig=np.asarray(origins)
        sites=np.ravel_multi_index(((XYZ[:,None,:]+orig)%20).transpose(2,0,1),(20,)*3)
        b=np.ravel_multi_index(((XYZ+e)%20).T,(20,)*3);diff=[]
        for o in orig:
            i=np.flatnonzero(np.all(TILE==-o,axis=1));j=np.flatnonzero(np.all(TILE==e-o,axis=1));diff.append((int(POW[i[0]]) if len(i) else 0)-(int(POW[j[0]]) if len(j) else 0))
        if len(origins)!=12 or np.sum(np.array(diff)==0):raise ValueError('bad translated tile footprint')
        tables.append((b,sites,np.array(diff)))
    return tables


def erase(s,words,q,luts,tables):
    # Nothing is written to the physical state. Apply digit differences to LUT indices only.
    u=s.ravel().astype('i2');w=words.ravel();q=q.ravel();candidates=[];seen={k:0 for k in luts};available=0
    for ax,(b,sites,digit) in enumerate(tables):
        j=np.flatnonzero((u!=0)&(u[b]!=0)&(u!=u[b])&(q==q[b]));available+=len(j);patch=sites[j];w0=w[patch];w1=w0+(u[b[j]]-u[j])[:,None]*digit
        if w1.size and (w1.min()<0 or w1.max()>6560):raise ValueError('invalid ternary LUT digit replacement')
        d={k:j[np.all(table[w0]==table[w1],axis=1)] for k,table in luts.items()};candidates.append(d)
    results={}
    for k in luts:
        used=np.zeros(N,bool);used_sites=np.zeros(N,bool);witness=[];count=0
        for ax,g in enumerate(candidates):
            b,sites,_=tables[ax];count+=len(g[k])
            for j in g[k]:
                if used_sites[j] or used_sites[b[j]] or used[sites[j]].any():continue
                used[sites[j]]=True;used_sites[[j,b[j]]]=True;witness.append((int(j),ax))
        results[k]=(available,count,len(witness),witness)
    return results

def prop(x):
    D=x.sum(1);A=x[:,1:].sum(1);g=A.sum()/D.sum();n=len(x);se=np.sqrt(n/(n-1)*sum((A-g*D)**2))/sum(D);h=t.ppf(.975,n-1)*se;return dict(estimate=float(g),lo=float(g-h),hi=float(g+h),se=float(se),mde80_approx=float(2.8*se))

def run(a):
    o=a.out;control=np.load(R/'codec/comparators.npz');opt=np.load(R/'faces/optimized.npz');luts={'historical':control['rsmi_lut'],'affine':opt['affine_lut'],'composition':control['composition_lut']};bits={'historical':luts['historical'],'optimized':opt['discrete_lut']};table=prep_edges();out10={};out11={};rows={};examples={}
    for idx in (9,0,8):
        key=f'b{idx:02}';ss={k:[] for k in luts};es={k:[] for k in bits};err=0.
        with np.load(a.bank/(key+'.npz')) as z:states=z['species'];ids=z['parent']
        for i,s in zip(ids,states):
            v=dense(s);q=sum(at(s!=0,r).astype('i1') for r in DIR)
            for k,lut in luts.items():p,e=spectral(lut[v]);ss[k].append(p);err=max(err,e)
            for k,row in erase(s,v,q,bits,table).items():
                es[k].append(row[:3]);
                if row[2] and key+k not in examples:examples[key+k]=dict(parent=int(i),edge_start_and_positive_zyx_axis=row[3])
        A={};B={};rows[key+'_parent']=ids
        for k,mat in ss.items():
            mat=np.array(mat);rows[key+'_powers_'+k]=mat;A[k]=dict(low_power=interval(mat[:,0]),aliased_power=interval(mat[:,1:].sum(1)),alias_fraction_of_mean_power=prop(mat),alias_subbands_000_to_111=mat.mean(0).tolist(),parents=len(ids),max_complex_reconstruction_error=err)
        for k,mat in es.items():
            mat=np.array(mat);rows[key+'_erasures_'+k]=mat;B[k]=dict(eligible_swaps=int(mat[:,0].sum()),invisible_swaps=int(mat[:,1].sum()),witness_boxes=int((mat[:,2]>0).sum()),certified_bits_per_box=interval(mat[:,2]),largest_disjoint_family=int(mat[:,2].max()),parents=len(ids))
        B['optimized_minus_historical_bits']=interval(np.array(es['optimized'])[:,2]-np.array(es['historical'])[:,2]);out10[key]=A;out11[key]=B
    dest=o/'codec-parent-evidence.npz';np.savez_compressed(dest,**rows)
    json_out(o/'10-alias.json',dict(results=out10,script=digest(Path(__file__)),parent_evidence=digest(dest),estimand='Numerator = sum of powers of modes k+pi*b with b nonzero, summed over three first axial k and four centered labels; denominator additionally includes microscopic k. Equal to mean coarse power across eight origins; phases are not extra parents.'))
    json_out(o/'11-erasure.json',dict(results=out11,examples=examples,script=digest(Path(__file__)),parent_evidence=digest(dest),proof='An eligible exchange preserves composition and occupied mask; equal degree makes its energy change zero. Erasability checks the actual two replaced digits at each of 12 affected dense tiles. Disjoint footprints and vertices imply all 2^m substitutions preserve every translated symbol and Gibbs weight. Hence P(X=x|C)<=2^-m(x) and H(X|C)>=E[m(X)] under the canonical ensemble. Greedy packing is a lower bound; absence would not prove injectivity.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
