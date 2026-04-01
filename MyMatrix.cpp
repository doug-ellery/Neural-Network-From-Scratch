#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "MyMatrix.h"

//Helper macros for CUDA
#define CUDA_CHECK(err) \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Helper macro for cuBLAS errors
#define CUBLAS_CHECK(err) \
    if(err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

//multiply two matrices A and B, where A = M x K, B = K x N
static std::vector<float> MyMatrix::cudaMultiply(std::vector<half>& A, std::vector<half>& B, int M, int N, int K){
    //output array C = M x N
    std::vector<float> C(M*N,0.0f);

    //device (GPU) matrices, C will be output
    half* d_A,d_B;
    float* d_C;

    //allocate space for matrices on GPU, M = Rows of A and C, K = columns of A, 
    //rows of B, N = columns of B and C
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(half)));

    //Copy A, B, & C to device (GPU)
    CUDA_CHECK(cudaMempy(d_A, A.data(), M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMempy(d_B, B.data(), K*N*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMempy(d_C, 0, M*N*sizeof(float)));

    //CUBLAS handle initialize
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    //Enable tensor cores to be used
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH))

    //scalars which cublas uses in the equation C = alpha*A*B + beta*C
    //we just want C = A*B, so set alpha/beta accordingly
    float alpha = 1.0f;
    float beta = 0.0f;

    //GPU uses column major indexing, so we will compute B^t * A^t which equals C^t
    //which is the column major version of C, so when we read it back into the program,
    //C^t will become C because C^t is the column major version of C.
    //Because CUBLAS will automatically interpret A and B as A^t and B^t respectivley,
    //we just need to feed two the matrices backwards (B then A), making sure to 
    //also feed the dimensions backwards as well, as opposed to M,N,K
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16F, N,    //feeding B first with leading dimension (cols)
        d_A, CUDA_R_16F, K,    //feeding A second with leading dimension (cols)
        &beta,
        d_C, CUDA_R_32F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP. //use tensor cores for math
    ));

    //copy result back into host (C array)
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    //clean up memory
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));\

    //return result
    return C;
}