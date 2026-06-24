#include <iostream>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "MyMatrix.h"
#include <curand_kernel.h>



//multiply two matrices A and B, where A = M x K, B = K x N
void cudaMultiply(float* A, float* B, float* d_C, int M, int N, int K, cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB){

    //output array d_C = M x N
    CUDA_CHECK(cudaMemset(d_C, 0, M*N*sizeof(float)));
    

    //get leading dimensions for gemm ex
    int lda = (opB == CUBLAS_OP_N) ? N : K;
    int ldb = (opA == CUBLAS_OP_N) ? K : M;
    

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
    CUBLAS_CHECK(cublasSgemm(handle, opB, opA, N, M, K, &alpha, B, lda, A, ldb, &beta, d_C, N));
    
    CUDA_CHECK(cudaDeviceSynchronize()); 

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
__global__ void weightHeInitializeKernel(float* weights, int size, int n_in, unsigned long seed){
    int index, jump;
    getIndexJump(index, jump);
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

//very similar to the above kernel, but use sqrt(1.0/n_in), not 2.0
__global__ void weightXavierInitializeKernel(float* weights, int size, int n_in, unsigned long seed){
    int index, jump;
    getIndexJump(index, jump);
    if(index < size){
        for(int i = index; i < size; i += jump){
            //create RNG state
            curandState state;
            curand_init(seed, i, 0, &state);

            //Xavier Normal Scaling, which works by using the property that for a standard
            //gaussian RV, X = N(0,1), if we multiply by a we get a*X = N(0,a^2). For He normal,
            //we want the random variable: N(0,1.0/n_in), so we can multiply the standard gaussian 
            //by sqrt(1.0/n_in) to get this. 
            float sigma = sqrtf(1.0f / n_in);
            float rand_num = curand_normal(&state);
            weights[i] = rand_num * sigma;

        }
    }
}

//ReLU = Rectified Linear Unit => f(x) = max(0, x)
__global__ void reluActivationKernel(float* layer, int size){
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


//Kernel for calculating the cost function, using MSE

__global__ void costKernel(float* predictions, float* correctOnes, float* cost, int numSamples, int outputSize){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < numSamples*outputSize; i += jump){
        //Key: use atomic addition because multiple threads may be trying to add 
        //to cost at the same time, so this avoids race conditions that can lead to wrong answers
        //added a 2 on the bottom to make the derivative cleaner later on, so that the 2 on the 
        // bottom cancels with the 2 that comes on top when we do power rule for the delta calculation
        atomicAdd(cost, (1.0f)/(2.0f*numSamples)*(predictions[i] - correctOnes[i])*(predictions[i] - correctOnes[i]));
    }
}

//kernel to get my deltas for the ouput layer, because I am doing batching, this array is 
// numNodes x numSamples, as opposed to just numNodes
__global__ void startingDeltaKernel(float* predictions, float* correctOnes, float* startingDeltas, int size){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < size; i+= jump){
        //delta = a_i - y_i for this layer, because of the derivative of MSE
        startingDeltas[i] = predictions[i] - correctOnes[i];
    }
}

//kernel to apply relu derivative to the matrix produced from calculating W^t * delta^l+1,
//to give us our delta^1
__global__ void reluPrimeKernel(float* z, float* deltaL, int n_in, int numSamples){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < n_in*numSamples; i+= jump){
        //if z_i = negative, relu'(z_i) = 0
        if(z[i] < 0){
            deltaL[i] = 0;
        }
    }
}

//helper function to print matrices that are stored as 1D arrays but are of dimension M x N
void printVec(std::vector<float> vec, int M, int N){
    for(int r = 0; r < M; r++){
        for(int c = 0; c < N; c++){
            std::cout<<vec[r*N + c]<<" ";
        }
        std::cout<<"\n";
    }
}
//input = rows x cols, output = rows x 1
//column sum sums up all items in the same ROW, visiting each column.
//Assuming output was zero intiialized 
__global__ void columnSumKernel(float * input, float * output, int rows, int cols){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < rows*cols; i += jump){
        int row = i / cols;
        atomicAdd(&output[row], input[i]);
    }
}

//for multiplying a matrix by a scalar value, which mutliplies each individual item
__global__ void scalarMultiplyKernel(float * mat, float scalar, int size){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < size; i += jump){
        mat[i] *= scalar;
    }
}

//for updating either the weights or biases after running back prop
__global__ void updateParameterKernel(float * params, float * gradient, float learning_rate, int size){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < size; i += jump){
        //follow formula: W = W - lr*dW or B = B - lr*dB
        params[i] -= learning_rate * gradient[i];
    }
}

//adding a new activation function, tanh, which is better at learning curved functions with smaller networks
__global__ void tanhActivationKernel(float* layer, int size){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < size; i += jump){
        layer[i] = tanhf(layer[i]);
    }
}

void cudaTanhActivation(float* A, int size){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, size);
    tanhActivationKernel<<<numBlocks, threadsPerBlock>>>(A, size);
}

__global__ void tanhPrimeKernel(float * a, float* delta_l, int n_in, int samples){
    int index, jump;
    getIndexJump(index, jump);
    for(int i = index; i < n_in*samples; i += jump){
        delta_l[i] *= 1 - a[i]*a[i];
    }
}