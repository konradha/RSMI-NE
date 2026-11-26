// Qualification only. Every queried H is a full periodic-lattice sum from species bytes;
// no locality/stencil assumptions, potential cache or production barrier implementation.
#include <array>
#include <algorithm>
#include <cstdint>
#include "efma.hpp" // Original native reference, not used by the full-H oracle.
#include <vector>
namespace {
struct Reference {
    int L,V;std::vector<uint8_t> s;std::vector<std::array<int,6>> neighbors;
    Reference(int l,const uint8_t* state):L(l),V(l*l*l),s(state,state+V),neighbors(V){
        for(int i=0;i<V;i++)for(int dir=0;dir<6;dir++){
            int step=1;for(int axis=0;axis<dir/2;axis++)step*=L;
            int c=i/step%L,sign=1-2*(dir%2); neighbors[i][dir]=i+step*((c+sign+L)%L-c);
        }
    }
    int H() const {
        int total=0;
        for(int i=0;i<V;i++){
            int n=0;for(int j:neighbors[i])n+=s[j]!=0;
            int residual=n-(s[i]==2?5:3);total+=(s[i]!=0)*residual*residual;
        }
        return total;
    }
    int trial(int a,int b,int h){
        if(s[a]==s[b])return -128;
        std::swap(s[a],s[b]);int delta=H()-h;std::swap(s[a],s[b]);return delta;
    }
};
}
// Report interventional raw barriers for Python integer arithmetic. `after` is
// [3V,probes,6], only defined for unlike environmental swaps nonincident to probe.
extern "C" int opening_replay(int L,const uint8_t* state,const int* probes,int n,int16_t* original,int16_t* after){
    Reference r(L,state);int base=r.H();
    for(int a=0;a<r.V;a++)for(int axis=0;axis<3;axis++){
        int d=2*axis,b=r.neighbors[a][d],e=3*a+axis;
        int de=r.trial(a,b,base);original[6*a+d]=original[6*b+d+1]=int16_t(de);
        if(de==-128)continue;
        std::swap(r.s[a],r.s[b]);
        for(int p=0;p<n;p++){
            int x=probes[p];if(x==a||x==b)continue;
            for(int f=0;f<6;f++)after[(e*n+p)*6+f]=int16_t(r.trial(x,r.neighbors[x][f],base+de));
        }
        std::swap(r.s[a],r.s[b]);
    }
    return base;
}

// Native table outputs for every positive integer barrier through 127; the
// empty beta-zero table denotes 2^64, not UINT64_MAX.
extern "C" int opening_native_thresholds(const char* text,uint64_t* out){
    auto t=solid::makeThresholds(text);
    if(t.empty())return 1;
    for(int d=1;d<=127;d++)out[d]=(uint64_t(t[d].hi)<<32)|t[d].lo;
    return 0;
}
