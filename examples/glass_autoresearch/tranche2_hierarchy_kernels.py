"""Single-thread kernels; no JIT dependencies beyond the host C++ compiler."""
import ctypes as ct
import hashlib
import subprocess
from pathlib import Path

import numpy as np


SRC = r'''
#include <cstdint>
using U=uint8_t; using W=uint16_t; using I=uint32_t;
extern "C" {
void extract(int n,const U *a,W *ws,U *y,U *yc){
  for(int p=0;p<n;p++,a+=8000){
    W child[1000];
    auto s=[&](int i,int j,int k){return a[(i%20)*400+(j%20)*20+k%20];};
    for(int i=0;i<10;i++)for(int j=0;j<10;j++)for(int k=0;k<10;k++){
      int w=0,f=0,g=0;
      for(int x=0;x<2;x++)for(int b=0;b<2;b++)for(int c=0;c<2;c++)w=3*w+s(2*i+x,2*j+b,2*k+c);
      for(int b=0;b<2;b++)for(int c=0;c<2;c++){
        f=3*f+s(2*i+6,2*j+1+b,2*k+1+c); g=3*g+s(2*i+3,2*j+b,2*k+c);
      }
      int t=100*i+10*j+k; child[t]=w; *y++=f; *yc++=g;
    }
    for(int i=0;i<10;i++)for(int j=0;j<10;j++)for(int k=0;k<10;k++)
      for(int x=0;x<2;x++)for(int b=0;b<2;b++)for(int c=0;c<2;c++)
        *ws++=child[100*((i+x)%10)+10*((j+b)%10)+(k+c)%10];
  }
}
void rawhist(int n,const W *w,const U *y,I *counts){
  for(int i=0;i<n;i++,w+=8)for(int j=0;j<8;j++)++counts[(j*6561+w[j])*81+y[i]];
}
void rawapply(int n,const W *w,const float *a,U *h){
  for(int i=0;i<n;i++,w+=8){float s[4]={};for(int j=0;j<8;j++){
    const float *v=a+(j*6561+w[j])*4;for(int k=0;k<4;k++)s[k]+=v[k];}
    int b=0;for(int k=1;k<4;k++)if(s[k]>s[b])b=k;h[i]=b;}
}
void encode(int n,const W *w,const U *child,W *out){
  for(int i=0;i<n;i++){unsigned t=0;for(int j=0;j<8;j++)t|=unsigned(child[*w++])<<(2*j);out[i]=t;}
}
void tophist(int n,const W *v,const U *y,I *counts){for(int i=0;i<n;i++)++counts[v[i]*81+y[i]];}
void tally(int n,const U *a,const U *b,I *counts){for(int i=0;i<n;i++)++counts[a[i]*81+b[i]];}
void parentmean(int n,const U *a,const U *b,const double *value,double *out){
  for(int p=0;p<n;p++){double v=0; for(int t=0;t<1000;t++)v+=value[81*(*a++)+(*b++)];out[p]=v/1000;}
}
void adjoint(int n,const W *w,const W *v,const U *y,const U *top,const double *value,double *g){
  for(int i=0;i<n;i++,w+=8){unsigned t=v[i];int f=y[i];for(int j=0;j<8;j++){
    int base=t&~(3u<<(2*j)); double *r=g+w[j]*4;
    for(int k=0;k<4;k++)r[k]+=value[81*top[base|(k<<(2*j))]+f];
  }}
}
}
'''


class Kernels:
    def __init__(self, work):
        so = Path(work) / ('25-' + hashlib.sha256(SRC.encode()).hexdigest()[:16] + '.so')
        if not so.is_file():
            subprocess.run(['g++', '-std=c++11', '-O3', '-march=native', '-fPIC', '-shared', '-x', 'c++', '-', '-o', str(so)], input=SRC.encode(), check=True)
        self.dll = ct.CDLL(str(so))

    def run(self, name, n, *arrays):
        assert all(a.flags.c_contiguous for a in arrays)
        getattr(self.dll, name)(ct.c_int(n), *(ct.c_void_p(a.ctypes.data) for a in arrays))

    def extract(self, species):
        w = np.empty((len(species)*1000, 8), np.uint16)
        y = np.empty(len(w), np.uint8)
        yc = np.empty_like(y)
        self.run('extract', len(species), species, w, y, yc)
        return w, y, yc

    def codes(self, w, child):
        out = np.empty(len(w), np.uint16)
        self.run('encode', len(w), w, child, out)
        return out

    def raw(self, w, rule):
        out = np.empty(len(w), np.uint8)
        self.run('rawapply', len(w), w, rule, out)
        return out

    def hist(self, name, shape, *inputs):
        out = np.zeros(shape, np.uint32)
        self.run(name, len(inputs[0]), *inputs, out)
        return out
