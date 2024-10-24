#ifndef _ATD_H_
#define _ATD_H_
#include "common.h"
#include "graph.h"

__global__ void floyd_warshall_kernel(GPUGraph* g) {
    if(THID<g->n) g->d_offset[i]*=2;
}

#endif