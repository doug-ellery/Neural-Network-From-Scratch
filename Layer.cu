#include <vector>
#include "Layer.h"
#include <iostream>
#include "MyMatrix.h"
#include <ctime>
#include <cuda_runtime.h>

Layer::Layer(int nodesThisLayer, int nodesNextLayer, int numSamples){
    //n_in = how many nodes are in our layer, n_out = how many nodes are in the next layer,
    //ie. how many weights each node needs to have. n_out = how many nodes are in the next layer,
    //and in this design, biases is going to be the biases for the next layer, just how the 
    //weights are the weights that take us to the next layer, thus there are n_out biases for 
    //the next layer as each node has one bias
    //In multiplication, we well treat weights as an n_out x n_in matrix
    n_in = nodesThisLayer;
    n_out = nodesNextLayer;
    samples = numSamples;
    CUDA_CHECK(cudaMalloc((void **)&weights, n_in * n_out *sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&biases, n_out * sizeof(float)));

    //biases can be initialized to 0
    CUDA_CHECK(cudaMemset((void *)biases, 0, n_out * sizeof(float)));

    //weights cannot, use He normal initialization because we are using ReLU for now as the activation function
    CUDA_CHECK(cudaMemset((void *)weights, 0, n_in * n_out * sizeof(float)));
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_in * n_out);
    unsigned long seed = (unsigned long)time(NULL);

    //call our He normal kernel
    weightInitializeKernel<<<numBlocks, threadsPerBlock>>>(weights, n_in * n_out, n_in, seed);
}

//getNextLayer will use the weights, biases, and activation to get us the next layer and 
//return the values for this layer


float* Layer::getNextLayer(float* prevLayer){
    float* nextLayer;
    CUDA_CHECK(cudaMalloc((void **) &nextLayer, n_out*samples*sizeof(float)));
    //multiply weights x prevLayer first
    cudaMultiply(weights, prevLayer, nextLayer, n_out, samples, n_in);
    CUDA_CHECK(cudaFree(prevLayer));

    //Add this matrix and biases
    cudaAdd(nextLayer, biases, n_out, samples);
    cudaReLUActivation(nextLayer, n_out * samples);

    return nextLayer;
}


Layer::~Layer(){
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(biases));
}



