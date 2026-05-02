#include "NeuralNet.h"
#include "Layer.h"
#include "MyMatrix.h"
#include <cuda_runtime.h>
#include <iostream>


NeuralNet::NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>& data, std::vector<float>& outputs){
    this->numSamples = numSamples;
    this->outputSize = outputSize;
    //create input layer weights
    layers.reserve(numHiddenLayers + 1);
    layers.push_back(Layer(inputSize, nodesPerHiddenLayer, numSamples));
    int i = 0;
    //create hidden layer weights
    while(i < numHiddenLayers - 1){
        layers.push_back(Layer(nodesPerHiddenLayer, nodesPerHiddenLayer, numSamples));
        i++;
    }
    //create weights that take us to output layer
    layers.push_back(Layer(nodesPerHiddenLayer, outputSize, numSamples));
    //get values of first layer onto GPU
    CUDA_CHECK(cudaMalloc(&inputs, inputSize * numSamples * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(inputs, data.data(), inputSize * numSamples * sizeof(float), cudaMemcpyHostToDevice));
    //get predictions onto gpu (itll start as empty because we haven't done a forward pass
    CUDA_CHECK(cudaMalloc(&predictions, outputSize*numSamples*sizeof(float)));
    CUDA_CHECK(cudaMemset(predictions, 0, outputSize*numSamples*sizeof(float)));
    //get outputs onto GPU
    CUDA_CHECK(cudaMalloc(&correctOutputs, outputSize*numSamples*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(correctOutputs, outputs.data(), outputSize*numSamples*sizeof(float), cudaMemcpyHostToDevice));

}


std::vector<float> NeuralNet::forwardPass(){
    float* prev = layers[0].getNextLayer(inputs);
    float* curr = nullptr;
    for(int i = 1; i < layers.size(); i++){
        std::cout<<"Loop "<<i<<"starting...\n";
        curr = layers[i].getNextLayer(prev);
        prev = curr;
    }
    predictions = prev;
    std::vector<float> outVec(numSamples * outputSize);
    CUDA_CHECK(cudaMemcpy(outVec.data(), predictions, numSamples * outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    return outVec;
}

void NeuralNet::getCost(){
    //Get a cost variable onto the GPU
    float* cost;
    CUDA_CHECK(cudaMalloc(&cost, sizeof(float)));
    CUDA_CHECK(cudaMemset(cost, 0, sizeof(float)));
    
    //get threads per block and number of blocks
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, numSamples*outputSize);

    //call cost kernel
    costKernel<<<numBlocks, threadsPerBlock>>>(predictions, correctOutputs, cost, numSamples, outputSize);
    float result;
    CUDA_CHECK(cudaMemcpy(&result, cost, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cost));
    std::cout<<"Cost: "<<result<<"\n\n";
    
}

NeuralNet::~NeuralNet() {}

