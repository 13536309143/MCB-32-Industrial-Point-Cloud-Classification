// MCB_POINTCLOUD_V1: portable native sampling kernels.
// C++17, strict float64 arithmetic; no fast-math or fused contraction.
#if defined(_WIN32)
#define MCB_EXPORT extern "C" __declspec(dllexport)
#else
#define MCB_EXPORT extern "C"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include <stdexcept>
using namespace std;
struct IndexedMaxHeap {
 vector<int> ids,pos; vector<double>& values;
 IndexedMaxHeap(vector<double>&v,int excluded=-1):pos(v.size(),-1),values(v){
  for(int i=0;i<(int)v.size();i++)if(i!=excluded){pos[i]=ids.size();ids.push_back(i);}
  for(int i=(int)ids.size()/2-1;i>=0;i--)down(i);
 }
 bool better(int a,int b)const{return values[a]>values[b]||(values[a]==values[b]&&a<b);}
 void swap_at(int a,int b){swap(ids[a],ids[b]);pos[ids[a]]=a;pos[ids[b]]=b;}
 void down(int a){for(;;){int b=2*a+1;if(b>=(int)ids.size())return;int c=b+1;if(c<(int)ids.size()&&better(ids[c],ids[b]))b=c;if(!better(ids[b],ids[a]))return;swap_at(a,b);a=b;}}
 int pop(){int id=ids[0];pos[id]=-1;int last=ids.back();ids.pop_back();if(!ids.empty()){ids[0]=last;pos[last]=0;down(0);}return id;}
 void decreased(int id){if(pos[id]>=0)down(pos[id]);}
};
struct Node{double lo[3],hi[3],upper=0;int start,end,left=-1,right=-1;};
struct KDTree {
 const double* p; vector<int> order; vector<Node> nodes;
 KDTree(const double*points,int n):p(points),order(n){iota(order.begin(),order.end(),0);nodes.reserve(n/4);build(0,n);}
 double dist(int a,int b)const{double x=p[3*a]-p[3*b],y=p[3*a+1]-p[3*b+1],z=p[3*a+2]-p[3*b+2];return x*x+y*y+z*z;}
 double min_box(int node,int query)const{double sum=0;const Node&t=nodes[node];for(int j=0;j<3;j++){double x=p[3*query+j],d=x<t.lo[j]?t.lo[j]-x:(x>t.hi[j]?x-t.hi[j]:0);sum+=d*d;}return sum;}
 int build(int start,int end){Node t;t.start=start;t.end=end;for(int j=0;j<3;j++){t.lo[j]=INFINITY;t.hi[j]=-INFINITY;}
  for(int i=start;i<end;i++)for(int j=0;j<3;j++){double x=p[3*order[i]+j];t.lo[j]=min(t.lo[j],x);t.hi[j]=max(t.hi[j],x);}
  int index=nodes.size();nodes.push_back(t);if(end-start>16){int axis=0;for(int j=1;j<3;j++)if(t.hi[j]-t.lo[j]>t.hi[axis]-t.lo[axis])axis=j;int mid=(start+end)/2;
   nth_element(order.begin()+start,order.begin()+mid,order.begin()+end,[&](int a,int b){return p[3*a+axis]<p[3*b+axis]||(p[3*a+axis]==p[3*b+axis]&&a<b);});
   int left=build(start,mid),right=build(mid,end);nodes[index].left=left;nodes[index].right=right;}
  return index;
 }
 double initialize_upper(int node,const vector<double>& d){Node&t=nodes[node];if(t.left<0){double mx=0;for(int i=t.start;i<t.end;i++)mx=max(mx,d[order[i]]);return t.upper=mx;}return t.upper=max(initialize_upper(t.left,d),initialize_upper(t.right,d));}
 void update_nearest(int node,int query,vector<double>&d,IndexedMaxHeap&heap){Node&t=nodes[node];if(min_box(node,query)>=t.upper)return;
  if(t.left<0){double mx=0;for(int k=t.start;k<t.end;k++){int id=order[k];if(heap.pos[id]<0)continue;double fresh=dist(id,query);if(fresh<d[id]){d[id]=fresh;heap.decreased(id);}mx=max(mx,d[id]);}t.upper=mx;return;}
  int a=t.left,b=t.right;if(min_box(a,query)>min_box(b,query))swap(a,b);update_nearest(a,query,d,heap);update_nearest(b,query,d,heap);t.upper=max(nodes[t.left].upper,nodes[t.right].upper);
 }
 template<class Callback>void radius_query(int node,int query,double radius2,Callback&cb)const{
  const Node&t=nodes[node];if(min_box(node,query)>radius2)return;if(t.left<0){for(int k=t.start;k<t.end;k++){int id=order[k];double d2=dist(id,query);if(d2<radius2)cb(id,d2);}return;}
  radius_query(t.left,query,radius2,cb);radius_query(t.right,query,radius2,cb);
 }
};
MCB_EXPORT int exact_fps(const double*points,int n,int k,int32_t*selected){try{
 if(k<1||k>n)return 1;KDTree tree(points,n);vector<double>nearest(n);for(int i=0;i<n;i++)nearest[i]=tree.dist(i,0);IndexedMaxHeap heap(nearest,0);tree.initialize_upper(0,nearest);selected[0]=0;
 for(int step=1;step<k;step++){int id=heap.pop();selected[step]=id;nearest[id]=0;tree.update_nearest(0,id,nearest,heap);}return 0;
 }catch(...){return 2;}}
MCB_EXPORT int poisson_elimination(const double*points,int n,int k,double surface_area,int32_t*selected){try{
 if(k<1||k>n||surface_area<=0)return 1;KDTree tree(points,n);vector<double>weights(n,0);
 double maximum=2*sqrt(surface_area/(k*2*sqrt(3.0)));double minimum=maximum*0.65*(1-pow(double(k)/n,1.5));double radius2=maximum*maximum;
 auto weight=[&](double d2){double d=max(sqrt(d2),minimum);double x=max(0.0,1-d/maximum);double q=x*x;q*=q;return q*q;};
 for(int id=0;id<n;id++){auto sum=[&](int nb,double d2){if(nb!=id)weights[id]+=weight(d2);};tree.radius_query(0,id,radius2,sum);}
 IndexedMaxHeap heap(weights);for(int remaining=n;remaining>k;remaining--){int id=heap.pop();auto decrement=[&](int nb,double d2){if(heap.pos[nb]>=0){weights[nb]=max(0.0,weights[nb]-weight(d2));heap.decreased(nb);}};tree.radius_query(0,id,radius2,decrement);}
 int count=0;for(int id=0;id<n;id++)if(heap.pos[id]>=0)selected[count++]=id;return count==k?0:3;
 }catch(...){return 2;}}
