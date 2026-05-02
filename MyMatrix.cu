#include <iostream>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "MyMatrix.h"
#include <curand_kernel.h>



//multiply two matrices A and B, where A = M x K, B = K x N
void cudaMultiply(float* A, float* B, float* d_C, int M, int N, int K){
    //output array d_C = M x N
    CUDA_CHECK(cudaMemset(d_C, 0, M*N*sizeof(float)));
    
    //we need to use the __half versions of A and B
    __half* d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(__half)));
    floatToHalfCast(A, d_A, M*K);
    floatToHalfCast(B, d_B, K*N);

    //CUBLAS handle initialize
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    //Enable tensor cores to be used
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    //scalars which cublas uses in the equation C = alpha*A*B + beta*C
    //we just want C = A*B, so set alpha/beta accordingly
    const float alpha = 1.0f;
    const float beta = 0.0f;

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
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP   //use tensor cores for math
    ));
    CUDA_CHECK(cudaDeviceSynchronize()); 
    
    //Get rid of the half arrays
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

}

//GPU function to get the index and jump value for a particular thread 
//to tell it what items it needs to handle
__device__ void getIndexJump(int& index, int& jump){
    //get a unique index based on thread ID so that each thread handles its own index.
    index = blockIdx.x * blockDim.x + threadIdx.x;

    //total number of threads
    jump = blockDim.x * gridDim.x;
}

//function to get threads per block and number of blocks
void getThreadsBlocks(int& threadsPerBlock, int& numBlocks, int size){
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

//Addition stuff, starting with kernel
//total size of A in A + B = size, A = m x n, but I only need n to extract the row index from
//it 

static __global__ void addKernel(float* A, float* B, int size, int n){
    int index, jump;
    getIndexJump(index, jump);
    int row = -1;
    for(int i = index; i < size; i += jump){
        //figure out what row in A we are in first, this tells us what element to add to it 
        //from B
        row = i / n;
        A[i] = A[i] + B[row];
    }
}

//Add an M x N and an M x 1 matrix
//Effectivley I am adding an M x N matrix to an M x N matrix, but the second matrix is N 
//identical columns, because it will be N columns of the biases for a given layer, which I
//need for broadcasting in order to pull off the vectorization
//Result is stored in A, so that no extra memory is used
void cudaAdd(float* A, float* B, int M, int N){

    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, M*N);

    //call adding kernel
    addKernel<<<numBlocks, threadsPerBlock>>>(A, B, M*N, N);
}

//Kernel to help initialize the weights for a layer, using He initialization
__global__ void weightInitializeKernel(float* weights, int size, int n_in, unsigned long seed){
    int index, jump;
    getIndexJump(index,jump);
    if(index < size){
        for(int i = index; i < size; i += jump){
            //create RNG state
            curandState state;
            curand_init(seed, i, 0, &state);

            //He Normal Scaling, which works by using the property that for a standard
            //gaussian RV, X = N(0,1), if we multiply by a we get a*X = N(0,a^2). For He normal,
            //we want the random variable: N(0,2/n_in), so we can multiply the standard gaussian 
            //by sqrt(2/n_in) to get this. 
            float sigma = sqrtf(2.0f / n_in);
            float rand_num = curand_normal(&state);
            weights[i] = rand_num * sigma;

        }
    }
}

//ReLU = Rectified Linear Unit => f(x) = max(0, x)
static __global__ void reluActivationKernel(float* layer, int size){
    int index, jump;
    getIndexJump(index, jump);

    for(int i = index; i < size; i += jump){
        if(layer[i] < 0.0f){
            layer[i] = 0.0f;
        }
    }
}

void cudaReLUActivation(float* A, int size){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, size);

    //call ReLU kernel
    reluActivationKernel<<<numBlocks, threadsPerBlock>>>(A, size);
}

static __global__ void floatToHalfKernel(float* in, __half* out, int size){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < size; i += jump){
        out[i] = __float2half(in[i]);
    }
}

void floatToHalfCast(float* in, __half* out, int size){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, size);

    //call floatToHalfKernel
    floatToHalfKernel<<<numBlocks, threadsPerBlock>>>(in, out, size);
}

/*Kernel for calculating the cost function, using log loss, where I sum 
up actual*log(predicted) for all data points, and then multiply by -1/number of samples
*/
__global__ void costKernel(float* predictions, float* correctOnes, float* cost, int numSamples, int outputSize){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < numSamples*outputSize; i += jump){
        //Key: use atomic addition because multiple threads may be trying to add 
        //to cost at the same time, so this avoids race conditions that can lead to wrong answers
        //atomicAdd(cost, (-1.0f)*correctOnes[index]*__logf(predictions[index])/(float)numSamples);
        atomicAdd(cost, (1.0f)/numSamples*(predictions[index] - correctOnes[index])*(predictions[index] - correctOnes[index]));
    }
}




