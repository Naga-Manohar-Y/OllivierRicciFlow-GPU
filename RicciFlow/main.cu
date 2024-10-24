#include "graph.h"
#include "ATD.h"

int main(int argc, char* argv[]) { 
    GPUGraph *g;
    cudaMallocManaged(&g, sizeof(GPUGraph));
    g->readBinaryFile(argv[1]);
    floyd_warshall_kernel<<<BLK_NUMS, BLK_DIM>>>(g);
    printf("\n");
    return 0;
}