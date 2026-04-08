#include "Layer.h"
#include <vector>

#ifndef NEURALNET_H
#define NEURALNET_H

class NeuralNet{
    std::vector<Layer> layers;
    public:
        NeuralNet();
        ~NeuralNet();
};

#endif