#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>
#include <cuda_runtime_api.h>
#include <cuda.h>
using namespace std;
typedef unsigned int ui;
typedef ui ept;


#define BLK_NUMS 56
#define BLK_DIM 1024
#define WARPS_EACH_BLK (BLK_DIM >> 5)
#define N_THREADS (BLK_DIM * BLK_NUMS)
#define N_WARPS (BLK_NUMS * WARPS_EACH_BLK)
#define THID threadIdx.x
#define BLKID blockIdx.x
#define WARP_SIZE 32
#define WARPID (THID >> 5)
#define LANEID (THID & 31)
#define FULL 0xFFFFFFFF
#define GLWARPID (BLKID * WARPS_EACH_BLK + WARPID)
#define GTHID (BLKID*N_THREADS+THID)

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        cout<<"Error occured: ";
        std::cout << cudaGetErrorString(code) << std::endl;
        exit(-1);
    }
}

#endif