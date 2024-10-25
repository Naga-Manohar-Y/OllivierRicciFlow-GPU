#ifndef _ATD_H_
#define _ATD_H_
#include "common.h"
#include "graph.h"

__global__ void fw_pass(GPUGraph* g, ui w, float* apsp) {
    ui msize=N*N;
    for(ui uv=GTHID; uv<msize; uv+=N_THREADS){
        ui u = GTHID/N;
        ui v = GTHID%N;

        float x = apsp[m_ind(w, u)]; // still need to address memory coalescing issue. and may be we need to calculate only upper/lower diagonal of matrix
        float y = apsp[m_ind(w, v)];
        if(x==INF||y==INF) continue;
        x+=y;
        if(x<apsp[uv]) apsp[uv] = x;
    }
}


void compute_apsp(GPUGraph* g, float* apsp){
    fill(apsp, apsp+N*N, INF);
    for(ui u=0;u<N;u++)
        for(ui j=g->offset[u];j<g->offset[u+1];j++)
            apsp[u*N+g->neighbors[j]]= g->d_weights[j];
    
    for(ui w=0;w<N;w++)
        fw_pass<<<BLK_NUMS, BLK_DIM>>>(g, w, apsp);
    cudaDeviceSynchronize();
}

__global__ d_compute_RC_ATD(GPUGraph* g, float* apsp, float* edge_RC){
    for(ui e=GLWARPID; e<M; e+=N_WARPS){
        ui src, dst;
        src=g->edges[e*2];
        dst=g->edges[e*2+1];

        ui  en=g->d_offset[src+1];
        float cost=0;
        for(ui i=g->d_offset[src]; i<en; i++){
            ui u=g->d_neighbors[i];
            ui st_dst=g->d_offset[dst], en_dst=g->d_offset[dst+1];
            for(ui j=st_dst+LANEID; j<en_dst; j+=32){
                ui v = g->d_neighbors[j];
                if(u>v) continue; // because edges are undirected, we want to sum each edge once i.e. u<v
                cost += apsp[m_ind(u, v)];
            }
        }
        float share=(1.0 - ALPHA)/(g->d_degree[src] * g->d_degree[dst]);
        cost = warp_sum(cost)*share;
        cost += ALPHA*apsp[m_ind(src, dst)]; // adding self cost
        if(LANEID==0)
            edge_RC[e] = 1.0-cost/g->d_weight[e]; // todo should it be apsp[e]??
    }   
}

void compute_RC_ATD(GPUGraph* g, float* apsp, float* edge_RC){
    compute_apsp(g, apsp);
    d_compute_RC_ATD<<<BLK_NUMS, BLK_DIM>>>(g, apsp, edge_RC);
    cudaDeviceSynchronize();
}

#endif