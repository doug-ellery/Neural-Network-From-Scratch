#include "NeuralNet.h"
#include "Layer.h"
#include "MyMatrix.h"
#include <cuda_runtime.h>


NeuralNet::NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>& data){
    //create input layer weights
    layers.push_back(Layer(inputSize, nodesPerHiddenLayer, numSamples));
    int i = 0;
    //create hidden layer weights
    while(i < numHiddenLayers - 1){
        layers.push_back(Layer(nodesPerHiddenLayer, nodesPerHiddenLayer, numSamples));
    }
    //create weights that take us to output layer
    layers.push_back(Layer(nodesPerHiddenLayer, outputSize, numSamples));
    //get values of first layer onto GPU
    CUDA_CHECK(cudaMalloc(&inputs, inputSize * numSamples * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(inputs, data.data(), inputSize * numSamples * sizeof(float), cudaMemcpyHostToDevice));

}

std::vector<float> NeuralNet::forwardPass(){
    float* curr, prev;
    prev = layers[0].getNextLayer(inputs);
    for(int i = 1; i < layers.size(); i++){
        curr = layers[i].getNextLayer(prev);
        CUDA_CHECK(cudaMemfree(prev));
        prev = curr;
    }
    vector<int> outVec;
    CUDA_CHECK(cudaMemcpy(outVec.data(), curr, numSamples * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(curr));
    return outVec;
}
