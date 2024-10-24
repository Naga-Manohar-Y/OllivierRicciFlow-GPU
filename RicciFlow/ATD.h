#ifndef _ATD_H_
#define _ATD_H_
#include "common.h"
#include "graph.h"

__global__ void floyd_warshall_kernel(GPUGraph* g) {
    if(THID<10) printf("%d ", g->d_offset[THID]);
}

#endif