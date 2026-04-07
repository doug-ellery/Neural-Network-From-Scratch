#include <vector>
#include "Layer.h"
#include <iostream>
#include "MyMatrix.h"
#include <ctime>

Layer::Layer(int nodesThisLayer, int nodesNextLayer){
    //n_in = how many nodes are in our layer, n_out = how many nodes are in the next layer,
    //ie. how many weights each node needs to have
    n_in = nodesThisLayer;
    n_out = nodesNextLayer;
    CUDA_CHECK(cudaMalloc(((void **)&weights, n_in * n_out *sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&biases, n_in * sizeof(float)));

    //biases can be initialized to 0
    CUDA_CHECK(cudaMemset((void *)biases, 0, n_in * sizeof(float)));

    //weights cannot, use He normal initialization because we are using ReLU for now as the activation function
    CUDA_CHECK(cudaMemset((void *)weights, 0, n_in * n_out * sizeof(float)));
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_in * n_out);
    unsigned long seed = (unsigned long)time(NULL);
    
    //call our He normal kernel
    weightInitializeKernel<<<numBlocks, threadsPerBlock>>>(weights, n_in * n_out, n_in, seed);
}

Layer::~Layer(){
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(biases));
}



