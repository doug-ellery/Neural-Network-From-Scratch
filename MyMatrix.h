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

void cudaMultiply(float* d_A, float* d_B, float* d_C, int M, int N, int K);
__device__ void getIndexJump(int& index, int& jump);
void getThreadsBlocks(int& threadsPerBlock, int& numBlocks, int size);
void cudaAdd(float* A, float* B, int M, int N);
__global__ void weightInitializeKernel(float* weights, int size, int n_in, unsigned long seed);
void cudaReLUActivation(float* A, int size);

#endif