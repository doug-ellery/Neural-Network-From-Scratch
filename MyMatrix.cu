#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "MyMatrix.h"

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

//multiply two matrices A and B, where A = M x K, B = K x N
std::vector<float> cudaMultiply(std::vector<half>& A, std::vector<half>& B, int M, int N, int K){
    //output array C = M x N
    std::vector<float> C(M*N,0.0f);

    //device (GPU) matrices, C will be output
    half* d_A;
    half* d_B;
    float* d_C;
    std::cout<<"Starting parallel multiplication...\n\n";
    auto start = std::chrono::high_resolution_clock::now();
    //allocate space for matrices on GPU, M = Rows of A and C, K = columns of A, 
    //rows of B, N = columns of B and C
    CUDA_CHECK(cudaMalloc((void**)&d_A, M*K*sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K*N*sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M*N*sizeof(float)));

    //Copy A, B, & C to device (GPU)
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K*N*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M*N*sizeof(float)));

    //CUBLAS handle initialize
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    //Enable tensor cores to be used
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

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
        (const void*) &alpha,
        (const void*) d_B, CUDA_R_16F, N,    //feeding B first with leading dimension (cols)
        (const void*) d_A, CUDA_R_16F, K,    //feeding A second with leading dimension (cols)
        (const void*) &beta,
        d_C, CUDA_R_32F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP   //use tensor cores for math
    ));

    //copy result back into host (C array)
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    //clean up memory
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSecondsParallel = end - start;
    std::cout<<"\n\nTime for multiplication with gpu: "<<elapsedSecondsParallel.count()<<"\n\n";

    //return result
    return C;
}

//GPU function to get the index and jump value for a particular thread 
//to tell it what items it needs to handle
static __device__ void getIndexJump(int& index, int& jump){
    //get a unique index based on thread ID so that each thread handles its own index.
    index = blockIdx.x * blockDim.x + threadIdx.x;

    //total number of threads
    jump = blockDim.x * gridDim.x;
}

//function to get threads per block and number of blocks
static void getThreadsBlocks(int& threadsPerBlock, int& numBlocks, int size){
    //if we have 1024 elements or less, we can have enough threads in 1 block to handle 
    //everything
    if(size <= 1024){
        //calculate next multiple of 32 (to keep warps full)
        threadsPerBlock = ((size + 31) / 32) * 32;
        numBlocks = 1;
    }
    //otherwise we can evenly distribute threads among an appropriate number of blocks
    else{
        //I might play around with how I distribute the threads in this case and see what 
        //is most efficient.
        //highest multiple of 32 for each block
        threadsPerBlock = 1024;
        numBlocks = static_cast<int>(ceil((double)size / 1024));
    }
}


//Sigmoid stuff, starting with the sigmoid kernel which defines the 
//operation of a single thread for applying the sigmoid function.
//__global__ tells compiler this code is meant for the GPU.
//n is size of matrix

static __global__ void sigmoidKernel(float* A, int n){
    int index, jump;
    getIndexJump(index, jump);

    //its possible for index to be out of valid range, because not all threads will necessarily
    //be used, so make sure i < n, but also if we have more matrix items than threads, one 
    //thread may have to handle multiple matrix items, so we should jump our index by the total
    //number of threads to figure out the next one we need to handle
    for(int i = index; i < n; i += jump){
        A[i] = 1.0f / (1.0f + expf(-1 * A[i]));
    }
}

void cudaSigmoid(std::vector<float>& A){
    int size = A.size();
    float* d_A;
    std::cout<<"Starting parallel sigmoid...\n\n";
    auto start = std::chrono::high_resolution_clock::now();
    //Allocate memory for our matrix and copy it onto the gpu memory
    CUDA_CHECK(cudaMalloc((void **) &d_A, size*sizeof(float)));
    CUDA_CHECK(cudaMemcpy((void *) d_A, A.data(), size*sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, size);

    //Call our kernel using threads based on values of threadsPerBlock and numBlocks
    sigmoidKernel<<<numBlocks, threadsPerBlock>>>(d_A, size);

    //Wait for GPU to finish
    cudaDeviceSynchronize();

    //copy d_A back in to A and get rid of d_A on GPU
    CUDA_CHECK(cudaMemcpy((void *) A.data(), d_A, size*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSecondsParallel = end - start;
    std::cout<<"\n\nTime for sigmoid with gpu: "<<elapsedSecondsParallel.count()<<"\n\n";
    return;
}

//Addition stuff, starting with kernel

static __global__ void addKernel(float* A, float* B, float* C, int n){
    int index, jump;
    getIndexJump(index, jump);

    for(int i = index; i < n; i += jump){
        C[i] = A[i] + B[i];
    }
}

void cudaAdd(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int M, int N){
    float* d_A, *d_B, *d_C;
    std::cout<<"Starting parallel addition...\n\n";
    auto start = std::chrono::high_resolution_clock::now();
    //allocate GPU memory for device arrays
    CUDA_CHECK(cudaMalloc((void **) &d_A, M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_B, M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_C, M*N*sizeof(float)));

    //copy host arrays in to device
    CUDA_CHECK(cudaMemcpy((void *) d_A, A.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void *) d_B, B.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset((void *) d_C, 0, M*N*sizeof(float)));

    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, M*N);

    //call adding kernel
    addKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M*N);

    //copy result in to C
    CUDA_CHECK(cudaMemcpy((void *) C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    //free up memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSecondsParallel = end - start;
    std::cout<<"\n\nTime for addition with gpu: "<<elapsedSecondsParallel.count()<<"\n\n";
    
    return;
}


