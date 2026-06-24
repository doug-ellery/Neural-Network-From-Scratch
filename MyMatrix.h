#include <vector>
#include <cublas_v2.h>

#ifndef MYMATRIX_H
#define MYMATRIX_H

//Helper macros for CUDA
#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA ERROR\n"                                           \
                  << "  File: " << __FILE__ << "\n"                           \
                  << "  Line: " << __LINE__ << "\n"                           \
                  << "  Call: " << #call << "\n"                              \
                  << "  Code: " << err << "\n"                                \
                  << "  Message: " << cudaGetErrorString(err) << std::endl;   \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while(0)

// Helper macro for cuBLAS errors
#define CUBLAS_CHECK(call) \
{ \
    cublasStatus_t err = call; \
    if(err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void cudaMultiply(float* A, float* B, float* d_C, int M, int N, int K, cublasHandle_t handle, cublasOperation_t opA = CUBLAS_OP_N, cublasOperation_t opB = CUBLAS_OP_N);
__device__ void getIndexJump(int& index, int& jump);
void getThreadsBlocks(int& threadsPerBlock, int& numBlocks, int size);
void cudaAdd(float* A, float* B, int M, int N);
__global__ void weightHeInitializeKernel(float* weights, int size, int n_in, unsigned long seed);
__global__ void weightXavierInitializeKernel(float* weights, int size, int n_in, unsigned long seed);
__global__ void reluActivationKernel(float* layer, int size);
void cudaReLUActivation(float* A, int size);
__global__ void costKernel(float* predictions, float* correctOnes, float* cost, int numSamples, int outputSize);
__global__ void startingDeltaKernel(float* predictions, float* correctOnes, float* startingDeltas, int size);
__global__ void weightedSumKernel(float* deltaLPlusOne, float* weights, int n_out, int n_in, float* weightedSum, int batchNum, int j);
__global__ void deltaKernel(float* delta, float* deltaLPlusOne, float* z, float* weights, int n_out, int n_in, int numSamples);
__global__ void reluPrimeKernel(float* z, float* deltaL, int n_in, int numSamples);
void printVec(std::vector<float> vec, int M, int N);
__global__ void columnSumKernel(float * input, float * output, int rows, int cols);
__global__ void scalarMultiplyKernel(float * mat, float scalar, int size);
__global__ void updateParameterKernel(float * params, float * gradient, float learning_rate, int size, float * momentum, float * variance, float beta_1, float beta_2, float epsilon, int t);
__global__ void tanhActivationKernel(float* layer, int size);
void cudaTanhActivation(float* A, int size);
__global__ void tanhPrimeKernel(float * a, float* delta_l, int n_in, int samples);


#endif