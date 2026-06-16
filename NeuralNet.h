#include "Layer.h"
#include <vector>


#ifndef NEURALNET_H
#define NEURALNET_H

class NeuralNet{
    
    float* inputs, *predictions, *correctOutputs, *startingDelta;
    int numSamples, outputSize, inputSize;
    public:
        std::vector<Layer> layers;
        NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>&, std::vector<float>&);
        std::vector<float> forwardPass();
        void NeuralNet::getStartingDelta();
        void getCost();
        void getAllDeltas();
        void showAllDeltas();
        ~NeuralNet();

        //Delete copying operators for this class
        NeuralNet(const NeuralNet&) = delete;
        NeuralNet& operator=(const NeuralNet&) = delete;
};

#endif