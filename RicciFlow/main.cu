#include "common.h"
#include "graph.h"
#include "ATD.h"
__global__ d_update_weights(GPUGraph* g, float* edge_RC){
    float sumw = 0;
    for(ui e=GLTHID; e<M; e+=N_THREADS){
        g->d_weights[e]-=EPSILON*edge_RC[e]*g->d_weights[e];
        sumw+=g->d_weights[e];
    } 
    warp_sum(sumw);
    if (LANEID==0)
    atomicAdd(gsum, sumw);
}
__global__ d_normalize_weights(GPUGraph* g, float* edge_RC){
    for(ui e=GLTHID; e<M; e+=N_THREADS){
        g->d_weights[e]*=(m/gsum[0]);
    }
}

class RicciFlow{
    GPUGraph* g;
    float* gsum;
    float* apsp;
    float* edge_RC;
    float* node_RC;
    string method="ATD";
public:
    RicciFlow(GPUGraph *_g){
        g=_g;
        cudaMallocManaged(&apsp, sizeof(float)*N*N);
        cudaMallocManaged(&edge_RC, sizeof(float)*M);
        cudaMallocManaged(&node_RC, sizeof(float)*N);
        cudaMallocManaged(&gsum, sizeof(float));
    }
    void compute_edge_RC(){
        if (method=="ATD"){
            compute_RC_ATD(g, apsp, edge_RC);
        }
        else if(method=="OTD"){
            compute_RC_OTD(g, apsp, edge_RC);
        }
        else if(method=="Sinkhorn"){
            compute_RC_Sinkhorn(g, apsp, edge_RC);
        }
        else{
            cout<<"Not a valid method"<<endl;
            return;
        }
    }
    void update_weights(){
        *gsum = 0;
        d_update_weights<<<BLK_NUMS, BLK_DIM>>>(g, edge_RC);
        d_normalize_weights<<<BLK_NUMS, BLK_DIM>>>(g);
        cudaDeviceSynchronize();
    }
    void ricci_flow(){

        for(ui i=0;i<N_ITER; i++){
            compute_edge_RC(g, apsp, edge_RC);
            // compute_node_RC(g, edge_RC, node_RC);
            update_weights(g, edge_RC); // step 3, 4 of algo
            // todo check the condition cruvatuer values do not change a lot
        }
    }
};
    
int main(int argc, char* argv[]) { 
    GPUGraph *g;
    cudaMallocManaged(&g, sizeof(GPUGraph));
    g->readBinaryFile(argv[1]);
    g->copyToGPU();
    RicciFlow rf(g);
    rf.ricci_flow();
    return 0;
}