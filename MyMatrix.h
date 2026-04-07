#include <vector>
#include <cuda_fp16.h>

#ifndef MYMATRIX_H
#define MYMATRIX_H

//Helper macros for CUDA
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Helper macro for cuBLAS errors
#define CUBLAS_CHECK(call) \
{ \
    cublasStatus_t err = call; \
    if(err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

std::vector<float> cudaMultiply(std::vector<half>&, std::vector<half>&, int, int, int);
void cudaSigmoid(std::vector<float>&);
void cudaAdd(std::vector<float>&, std::vector<float>&, std::vector<float>&, int, int);
__device__ void getIndexJump(int&, int&);
__global__ void weightInitializeKernel(float*, int, int unsigned long);
void getThreadsBlocks(int&, int&, int);

#endif