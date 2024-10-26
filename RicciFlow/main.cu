#include "common.h"
#include "graph.h"
#include "ATD.h"
__managed__ float gsum;

__global__ void d_update_weights(GPUGraph* g, float* edge_RC){
    float sumw = 0;
    for(ui e=GTHID; e<M; e+=N_THREADS){
        g->d_weights[e]-=EPSILON*edge_RC[e]*g->d_weights[e];
        sumw+=g->d_weights[e];
    } 
    warp_sum(sumw);
    if (LANEID==0)
    atomicAdd(&gsum, sumw);
}
__global__ void d_normalize_weights(GPUGraph* g){
    for(ui e=GTHID; e<M; e+=N_THREADS){
        g->d_weights[e]*=(M/gsum);
    }
}

class RicciFlow{
    GPUGraph* g;
    float* apsp;
    float* edge_RC;
    float* node_RC;
    string method="ATD";
public:
    RicciFlow(GPUGraph *_g){
        g=_g;
        cudaMalloc(&apsp, sizeof(float)*N*N);
        cudaMalloc(&edge_RC, sizeof(float)*M);
        cudaMalloc(&node_RC, sizeof(float)*N);
        // cudaMallocManaged(&gsum, sizeof(float));
    }
    void compute_edge_RC(){
        if (method=="ATD"){
            compute_RC_ATD(g, apsp, edge_RC);
        }
        // else if(method=="OTD"){
        //     compute_RC_OTD(g, apsp, edge_RC);
        // }
        // else if(method=="Sinkhorn"){
        //     compute_RC_Sinkhorn(g, apsp, edge_RC);
        // }
        else{
            cout<<"Not a valid method"<<endl;
            return;
        }
    }
    void update_weights(){
        gsum = 0;
        // cudaMemset(gsum, sizeof(float), 0);
        d_update_weights<<<BLK_NUMS, BLK_DIM>>>(g, edge_RC);
        d_normalize_weights<<<BLK_NUMS, BLK_DIM>>>(g);
        cudaDeviceSynchronize();
    }
    void ricci_flow(){

        for(ui i=0;i<N_ITER; i++){
            // compute_edge_RC();
            // compute_node_RC(g, edge_RC, node_RC);
            // update_weights(); // step 3, 4 of algo
            // todo check the condition curvature values do not change a lot
        }
        cudaDeviceSynchronize();
        float *weights=new float[M];
        chkerr(cudaMemcpy(weights, g->d_weights, M * sizeof(float), cudaMemcpyDeviceToHost));
        // chkerr(cudaMemcpy(weights, g->d_degree, 10 * sizeof(ui), cudaMemcpyDeviceToHost));

        for(ui i=0;i<M;i++){
            cout<<weights[i]<<" ";
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