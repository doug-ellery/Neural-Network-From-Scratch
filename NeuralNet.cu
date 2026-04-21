#include "NeuralNet.h"
#include "Layer.h"
#include "MyMatrix.h"
#include <cuda_runtime.h>
#include <iostream>


NeuralNet::NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>& data){
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

}


std::vector<float> NeuralNet::forwardPass(){
    float* prev = layers[0].getNextLayer(inputs);
    float* curr = nullptr;
    for(int i = 1; i < layers.size(); i++){
        std::cout<<"Loop "<<i<<"starting...\n";
        curr = layers[i].getNextLayer(prev);
        prev = curr;
    }
    float* output = prev;
    std::vector<float> outVec(numSamples * outputSize);
    CUDA_CHECK(cudaMemcpy(outVec.data(), output, numSamples * outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(output));
    return outVec;
}

NeuralNet::~NeuralNet() {}

