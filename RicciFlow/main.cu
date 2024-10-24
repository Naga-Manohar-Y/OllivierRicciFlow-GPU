#include "graph.h"
#include "ATD.h"

int main(int argc, char* argv[]) { 
    GPUGraph *g;
    cudaMallocManaged(&g, sizeof(GPUGraph));
    g->readBinaryFile(argv[1]);
    for(ui i=0;i<g->n;i++){
        cout<<g->offset[i]<<" ";
    }
    cout<<endl;
    floyd_warshall_kernel<<<BLK_NUMS, BLK_DIM>>>(g);
    cudaDeviceSynchronize();

    // chkerr(cudaMemcpy(g->offset, g->d_offset, g->n * sizeof(ui), cudaMemcpyDeviceToHost));
    for(ui i=0;i<g->n;i++){
        cout<<g->offset[i]<<" ";
    }
    cout<<endl;
    return 0;
}