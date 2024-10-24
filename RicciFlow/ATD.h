#ifndef _ATD_H_
#define _ATD_H_
#include "common.h"
#include "graph.h"

__global__ void floyd_warshall_kernel(GPUGraph* g) {
    g->d_offset[5]++;
}

#endif