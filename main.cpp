#include <iostream>
#include "MyMatrix.h"
#include "NeuralNet.h"
#include <chrono>
#include <vector>

int main(){
    int numSamples = 10;
    int inputSize = 100;
    int numHiddenLayers = 20;
    int nodesPerLayer = 1200;
    int outputSize = 4;
    std::vector<float> inputs;
    for(int i = 0; i < inputSize * numSamples; i++){
        inputs.push_back(i);
    }
    NeuralNet testNet(numHiddenLayers, nodesPerLayer, inputSize, outputSize, numSamples, inputs);
    std::vector<float> outputs = testNet.forwardPass();
    for(int r = 0; r < outputSize; r++){
        for(int c = 0; c < numSamples; c++){
            std::cout<<outputs[r*numSamples + c]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n\n";
    
    return 0;

}