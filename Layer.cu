#include <vector>
#include "Layer.h"
#include <iostream>
#include "MyMatrix.h"
#include <ctime>
#include <cuda_runtime.h>

Layer::Layer(int nodesThisLayer, int nodesNextLayer, int numSamples, bool lastLayer){
    //n_in = how many nodes are in our layer, n_out = how many nodes are in the next layer,
    //ie. how many weights each node needs to have. n_out = how many nodes are in the next layer,
    //and in this design, biases is going to be the biases for the next layer, just how the 
    //weights are the weights that take us to the next layer, thus there are n_out biases for 
    //the next layer as each node has one bias
    //In multiplication, we well treat weights as an n_out x n_in matrix
    n_in = nodesThisLayer;
    n_out = nodesNextLayer;
    samples = numSamples;
    this->lastLayer = lastLayer;
    //allocate memory for all of our device arrays
    CUDA_CHECK(cudaMalloc((void **)&weights, n_in * n_out *sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&biases, n_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a, n_out*samples*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&z, n_out*samples*sizeof(float)));
    
    //biases can be initialized to 0, same with a and z
    CUDA_CHECK(cudaMemset((void *)biases, 0.0, n_out * sizeof(float)));

    CUDA_CHECK(cudaMemset((void *)a, 0.0, n_out * samples * sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)z, 0.0, n_out * samples * sizeof(float)));

    //weights cannot, use He normal initialization because we are using ReLU for now as the activation function
    CUDA_CHECK(cudaMemset((void *)weights, 0, n_in * n_out * sizeof(float)));
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_in * n_out);
    unsigned long seed = (unsigned long)time(NULL);

    //call our He normal kernel
    weightInitializeKernel<<<numBlocks, threadsPerBlock>>>(weights, n_in * n_out, n_in, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//Move constructor so that std::vector uses std::move when I call push_back, instead of a normal
//copy operation
Layer::Layer(Layer&& other) noexcept {
    weights = other.weights;
    biases = other.biases;
    z = other.z;
    a = other.a;

    n_in = other.n_in;
    n_out = other.n_out;
    samples = other.samples;
    other.n_in = 0;
    other.n_out = 0;
    other.samples = 0;

    other.weights = nullptr;
    other.biases = nullptr;
    other.z = nullptr;
    other.a = nullptr;
}

//getNextLayer will use the weights, biases, and activation to get us the next layer and 
//return the values for this layer
//note that prevLayer should be the attribute "a" for the previous layer

float* Layer::getNextLayer(float* prevLayer){
    //multiply weights x prevLayer first, store it in preactivation (z)
    cudaMultiply(weights, prevLayer, z, n_out, samples, n_in);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Add this matrix and biases
    cudaAdd(z, biases, n_out, samples);

    //copy z contents into a, and if there is an activation function, a will change accordingly
    CUDA_CHECK(cudaMemcpy(a, z, n_out*samples*sizeof(float), cudaMemcpyDeviceToDevice));

    if(!lastLayer){
        //apply the activation to a now
        cudaReLUActivation(a, n_out * samples);
    }
    

    return a;
}

//Helper function for debugging, weights is 
void Layer::printWeights(){
    std::vector<float> printer(n_out*n_in, 0);
    std::cout<<"Weights: \n";
    CUDA_CHECK(cudaMemcpy(printer.data(), weights, n_out*n_in*sizeof(float), cudaMemcpyDeviceToHost));
    for(int r = 0; r < n_out; r++){
        for(int c = 0; c < n_in; c++){
            std::cout<<printer[n_in*r + c]<<"  ";
        }
        std::cout<<"\n";
    }
}

void Layer::printActivation(){
    std::vector<float> nodes(n_out*samples, 0);
    CUDA_CHECK(cudaMemcpy(nodes.data(), a, n_out*samples*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < nodes.size(); i++){
        std::cout<<nodes[i]<<"\n";
    }
}

void Layer::printPreActivation(){
    std::vector<float> nodes(n_out*samples, 0);
    CUDA_CHECK(cudaMemcpy(nodes.data(), z, n_out*samples*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < nodes.size(); i++){
        std::cout<<nodes[i]<<"\n";
    }
}

void Layer::setBiases(std::vector<float> hardcodedBiases){
    CUDA_CHECK(cudaMemcpy((void *)biases, hardcodedBiases.data(), n_out*sizeof(float), cudaMemcpyHostToDevice));
}

void Layer::setWeights(std::vector<float> hardcodedWeights){
    CUDA_CHECK(cudaMemcpy((void *)weights, hardcodedWeights.data(), n_out*n_in*sizeof(float), cudaMemcpyHostToDevice));
}

Layer::~Layer(){
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(biases));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(z));
}



