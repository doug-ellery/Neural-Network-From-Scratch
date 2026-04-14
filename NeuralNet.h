#include "Layer.h"
#include <vector>

#ifndef NEURALNET_H
#define NEURALNET_H

class NeuralNet{
    std::vector<Layer> layers;
    public:
        NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples);
        ~NeuralNet();
};

#endif