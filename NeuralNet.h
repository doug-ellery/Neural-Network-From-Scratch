#include "Layer.h"
#include <vector>


#ifndef NEURALNET_H
#define NEURALNET_H

class NeuralNet{
    std::vector<Layer> layers;
    float* inputs;
    public:
        NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>);
        std::vector<float> forwardPass();
        ~NeuralNet();
};

#endif