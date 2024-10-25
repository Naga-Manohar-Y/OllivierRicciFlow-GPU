#ifndef _ATD_H_
#define _ATD_H_
#include "common.h"
#include "graph.h"

__global__ void fw_pass(GPUGraph* g, ui w, float* apsp) {
    ui msize=N*N;
    for(ui uv=GTHID; uv<msize; uv+=N_THREADS){
        ui u = GTHID/N;
        ui v = GTHID%N;

        float x = apsp[w*N+u];
        float y = apsp[w*N+v];
        if(x==INF||y==INF) continue;
        x+=y;
        if(x<apsp[uv]) apsp[uv] = x;
    }
}


float* compute_apsp(GPUGraph* g){
    float* apsp;
    cudaMallocManaged(&apsp, N*N*sizeof(float));
    fill(apsp, apsp+N, INF);
    for(ui i=0;i<N;i++){
        for(ui j=0;j<N;j++)
            cout<<apsp[i*N+j]<<" ";
        cout<<endl;
    }
    for(ui u=0;u<N;u++)
        for(ui j=g->offset[u];j<g->offset[u+1];j++)
            apsp[u*N+g->neighbors[j]]= 1;
    
    for(ui w=0;w<N;w++)
        fw_pass<<<BLK_NUMS, BLK_DIM>>>(g, w, apsp);
    cudaDeviceSynchronize();
    return apsp;
}

#endif