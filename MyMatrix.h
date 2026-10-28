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

//enum for activation function options
enum Activation_funcs{
    ReLU,
    Tanh
};

void cudaMultiply(const float*, const float*, float*, int, int, int, cublasHandle_t, cublasOperation_t, cublasOperation_t);
__device__ void getIndexJump(int&, int&);
void getThreadsBlocks(int&, int&, int);
void cudaAdd(float*, const float*, int, int);
__global__ void addKernel(float*, const float*, int, int)
__global__ void weightHeInitializeKernel(float*, int, int, unsigned long);
__global__ void weightXavierInitializeKernel(float*, int, int, unsigned long);
__global__ void reluActivationKernel(float*, int);
void cudaReLUActivation(float*, int);
__global__ void costKernel(const float*, const float*, float*, int, int);
__global__ void startingDeltaKernel(const float*, const float*, float*, int, int);
__global__ void reluPrimeKernel(const float*, float*, int, int);
__global__ void columnSumKernel(const float*, float*, int, int);
__global__ void scalarMultiplyKernel(float*, float, int);
__global__ void updateParameterKernel(float*, const float*, float, int, float*, float*, float, float, float, int);
__global__ void tanhActivationKernel(float*, int);
void cudaTanhActivation(float*, int);
__global__ void tanhPrimeKernel(const float*, float*, int, int);
__global__ void getSoftmaxMaxKernel(const float*, float*, int, int);
__global__ void getSoftmaxSumKernel(const float*, float*, const float*, int, int);
__global__ void softmaxActivationKernel(float*, const float*, const float*, int, int)
__global__ void transposeKernel(const float*, float*, int, int);
void transposeMatrix(const float*, float*, int, int);


#endif