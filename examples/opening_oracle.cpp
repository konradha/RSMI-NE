// Build: g++ -std=c++17 -O3 -march=native -fPIC -shared opening_oracle.cpp -o /tmp/opening_oracle.so
// Linux/glibc (binary128 expf128); no global mutable state or internal parallelism.
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
constexpr int same=-128;
constexpr int dirs[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
struct Weight { double hi,lo; }; // base 2^32 limbs, exact in all six-term signed sums
inline Weight operator+(Weight a,Weight b){return {a.hi+b.hi,a.lo+b.lo};}
inline Weight operator-(Weight a,Weight b){return {a.hi-b.hi,a.lo-b.lo};}
inline double probability(Weight a){return (a.hi*0x1p32+a.lo)*0x1p-64;}
struct Law {
    double key=-1; Weight weights[256]{}; double p[256]{};
    void prepare(double beta){
        if(key==beta)return;
        char text[64]; auto end=std::to_chars(text,text+63,beta); *end.ptr=0;
        // All released decimal strings (including the amended point) are recovered
        // by shortest round trip; the validation script checks all threshold integers.
        _Float128 b=strtof128(text,nullptr);
        for(int d=-128;d<128;++d){
            Weight w{};
            if(d!=same){
                if(d<=0 || beta==0)w={0x1p32,0};
                else {
                    _Float128 y=expf128(-b*d)*(_Float128)0x1p64;
                    uint64_t t=y>=(_Float128)0x1p64 ? UINT64_MAX : uint64_t(y);
                    w={double(t>>32),double(uint32_t(t))};
                }
            }
            weights[d+128]=w; p[d+128]=probability(w);
        }
        key=beta;
    }
};
struct Field { int8_t t,f[3]; };
inline int ideal(int t){return t==1?3:5;}
inline int resid(int t,int n){return t ? n-ideal(t) : 0;}
inline Field field(int t,int n,int m){
    const int k=n*(n+1)/2-(t ? (ideal(t)+1)*n : 0)+m;
    Field c{int8_t(t),{}};
    if(!t){ c.f[1]=int8_t(k-4*n+6);c.f[2]=int8_t(k-6*n+10); }
    else {c.f[0]=int8_t(-k-n);c.f[3-t]=int8_t((t==1?-2:2)*n);}
    return c;
}
inline int energy(Field a,Field b){return a.t==b.t ? same : 2*(a.f[b.t]+b.f[a.t]);}
struct Edge {uint8_t a,b,d;};
struct Node {uint8_t pos,nfrom,from[6];int8_t sign;};
struct Patch {
    int first,changed; // first-shell and changed-field prefixes, endpoints first
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<uint16_t> map; // actual lattice index for each origin and relative node
};
struct Geometry {
    int side=0,V=0;
    std::vector<std::array<int,6>> nn;
    Patch patch[3][2];
    int index(int x,int y,int z)const {return ((z+side)%side*side+(y+side)%side)*side+(x+side)%side;}
    void build(int L){
        if(side==L)return;
        side=L;V=L*L*L;nn.resize(V);
        for(int s=0;s<V;s++)for(int d=0;d<6;d++)nn[s][d]=index(s%L+dirs[d][0],s/L%L+dirs[d][1],s/(L*L)+dirs[d][2]);
        for(int axis=0;axis<3;axis++)for(int vac=0;vac<=1;vac++){
            Patch &p=patch[axis][vac]; p=Patch{};
            // BFS on the actual torus; works also on small periodic fixtures.
            int b=nn[0][2*axis]; std::vector<int> ids{0,b},rank(V,-1);rank[0]=0;rank[b]=1;
            auto expand=[&](){int n=int(ids.size());for(int i=0;i<n;i++)for(int s:nn[ids[i]])if(rank[s]<0){rank[s]=int(ids.size());ids.push_back(s);}};
            expand();p.first=int(ids.size()); if(vac)expand();p.changed=int(ids.size());expand();
            p.nodes.resize(ids.size());
            for(size_t i=0;i<ids.size();i++){
                int s=ids[i]; Node &nd=p.nodes[i]; nd.pos=uint8_t(i);
                for(int d=0;d<6;d++){
                    int r=rank[nn[s][d]];
                    if(r>=0 && r<(vac?p.first:2))nd.from[nd.nfrom++]=uint8_t(r);
                    if(r==0)++nd.sign;
                    if(r==1)--nd.sign;
                    if(int(i)<p.changed && r>int(i) && !(i==0 && r==1))p.edges.push_back({uint8_t(i),uint8_t(r),uint8_t(d)});
                }
            }
            p.map.resize(size_t(V)*ids.size());
            for(int a=0;a<V;a++)for(size_t i=0;i<ids.size();i++){
                int s=ids[i];p.map[size_t(a)*ids.size()+i]=uint16_t(index(a%L+s%L,a/L%L+s/L%L,a/(L*L)+s/(L*L)));
            }
        }
    }
};
struct Scratch { Geometry geo;Law law;std::vector<Field> base;std::vector<int8_t> r,m;std::vector<uint8_t> unlike;};
thread_local Scratch scratch;

template<bool Vac>
void event(const Patch& p,const uint16_t* map,const uint8_t* n,const int8_t* barrier,double* opening,double pe){
    auto& sc=scratch;const int count=int(p.nodes.size());
    Field cells[128];Weight sum[128]{};int8_t dr[16];
    const int ta=sc.base[map[0]].t,tb=sc.base[map[1]].t,inc=(tb!=0)-(ta!=0);
    for(int i=0;i<(Vac?p.first:2);i++){
        int s=map[i],t=i<2?(i==0?tb:ta):sc.base[s].t;
        dr[i]=int8_t(resid(t,n[s]+(Vac?p.nodes[i].sign*inc:0))-sc.r[s]);
    }
    for(int i=0;i<p.changed;i++){
        const Node& nd=p.nodes[i];int s=map[i],t=i<2?(i==0?tb:ta):sc.base[s].t;
        int m=sc.m[s];for(int k=0;k<nd.nfrom;k++)m+=dr[nd.from[k]];
        cells[i]=field(t,n[s]+(Vac?nd.sign*inc:0),m);
    }
    for(int i=p.changed;i<count;i++)cells[i]=sc.base[map[i]];
    const Weight* w=sc.law.weights+128;
    for(Edge edge:p.edges){
        const int a=edge.a,b=edge.b,old=barrier[6*map[a]+edge.d],fresh=energy(cells[a],cells[b]);
        if(old!=fresh){ Weight diff=w[fresh]-w[old];sum[a]=sum[a]+diff;sum[b]=sum[b]+diff; }
    }
    // x at either endpoint is excluded (nonincident), not merely required to
    // retain species. Each other candidate is included once per environmental e.
    for(int i=2;i<count;i++)if(cells[i].t){double delta=probability(sum[i]);if(delta>0)opening[map[i]]+=pe*delta;}
}
} // namespace

extern "C" int opening_threshold(double beta,int d,uint64_t* high,uint64_t* low){
    if(!(beta>=0)||!std::isfinite(beta)||d<same||d>127||!high||!low)return 1;
    scratch.law.prepare(beta);Weight w=scratch.law.weights[d+128];*high=uint64_t(w.hi);*low=uint64_t(w.lo);return 0;
}
extern "C" int opening_oracle(int L,double beta,const uint8_t* state,double* vh,double* v2o,int8_t* de,uint8_t* n){
    if(L<4||L>32||!(beta>=0)||!std::isfinite(beta)||!state||!vh||!v2o||!de||!n)return 1;
    auto& sc=scratch;sc.geo.build(L);sc.law.prepare(beta);int V=sc.geo.V;auto& nn=sc.geo.nn;
    sc.r.resize(V);sc.m.resize(V);sc.base.resize(V);sc.unlike.resize(V);
    for(int i=0;i<V;i++){
        if(state[i]>2)return 2;
        int count=0;for(int j:nn[i])count+=state[j]!=0;
        n[i]=uint8_t(count);sc.r[i]=int8_t(resid(state[i],count));
    }
    for(int i=0;i<V;i++){
        int m=0;for(int j:nn[i])m+=sc.r[j];sc.m[i]=int8_t(m);sc.base[i]=field(state[i],n[i],m);
    }
    for(int i=0;i<V;i++)for(int d=0;d<6;d+=2){int j=nn[i][d];int8_t v=int8_t(energy(sc.base[i],sc.base[j]));de[6*i+d]=v;de[6*j+(d^1)]=v;}
    for(int i=0;i<V;i++){
        Weight h{};int count=0;for(int d=0;d<6;d++){int e=de[6*i+d];h=h+sc.law.weights[e+128];count+=e!=same;}
        sc.unlike[i]=uint8_t(count);vh[i]=state[i]?probability(h)/3:0;v2o[i]=0;
    }
    if(beta==0){
        for(int i=0;i<V;i++)if(state[i]){int total=0;for(int y:nn[i])if(state[y]==state[i])total+=sc.unlike[y];v2o[i]=total/9.;}
        return 0;
    }
    for(int a=0;a<V;a++)for(int axis=0;axis<3;axis++){
        int d=2*axis;double pe=sc.law.p[de[6*a+d]+128];if(!pe)continue;
        int b=nn[a][d];bool vac=!(state[a]&&state[b]);const Patch& p=sc.geo.patch[axis][vac];
        auto map=p.map.data()+a*p.nodes.size();
        if(vac)event<true>(p,map,n,de,v2o,pe);else event<false>(p,map,n,de,v2o,pe);
    }
    for(int i=0;i<V;i++)v2o[i]/=9.;
    return 0;
}

// Qualification-only view of the same positive-part kernel, before multiplying
// by the environmental acceptance. Recompute the baseline; never assume a
// preceding call on this thread supplied the same configuration.
extern "C" int opening_oracle_gains(int L,double beta,const uint8_t* state,const int* probe,int np,double* gains){
    if(L<4 || L>32 || np<1 || !state || !probe || !gains)return 1;
    int V=L*L*L;
    std::vector<double> h(V),o(V);
    std::vector<int8_t> de(6*V);
    std::vector<uint8_t> n(V);
    if(opening_oracle(L,beta,state,h.data(),o.data(),de.data(),n.data()))return 1;
    for(int k=0;k<np;k++)if(probe[k]<0 || probe[k]>=V)return 1;
    for(int a=0;a<V;a++)for(int axis=0;axis<3;axis++){
        int b=scratch.geo.nn[a][2*axis];
        for(int k=0;k<np;k++)o[probe[k]]=0;
        if(state[a]!=state[b]){
            bool vac=!(state[a]&&state[b]);const Patch& p=scratch.geo.patch[axis][vac];
            const auto map=p.map.data()+a*p.nodes.size();
            if(vac)event<true>(p,map,n.data(),de.data(),o.data(),1.);
            else event<false>(p,map,n.data(),de.data(),o.data(),1.);
        }
        for(int k=0;k<np;k++)gains[(3*a+axis)*np+k]=o[probe[k]];
    }
    return 0;
}
