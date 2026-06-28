#include "Layer.h"
#include <vector>
#include <string>
#include <cublas_v2.h>
#include <random>


#ifndef NEURALNET_H
#define NEURALNET_H

class NeuralNet{
    
    float* inputs, *predictions, *correctOutputs, *startingDelta, *transposed_batch, learning_rate, curr_cost;
    int numSamples, outputSize, inputSize, batch_size;
    std::string activation_func;
    cublasHandle_t handle;
    //host side arrays for the inputs and outputs
    std::vector<float> h_inputs, h_outputs, shuffled_inputs, shuffled_outputs;
    //members needed for shuffling the inputs and outputs
    std::vector<int> h_inputs_indices;
    std::mt19937 rng;
    public:
        std::vector<Layer> layers;
        NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>&, std::vector<float>&, std::string);
        std::vector<float> forwardPass(float * batch_input);
        void NeuralNet::getStartingDelta(float * batch_output);
        void getCost(float* batch_output);
        void getAllDeltas();
        void showAllDeltas();
        void backProp(int t, float* batch_input, float* batch_output, bool shouldLog);
        void train();
        std::vector<float> predict(std::vector<float> prediction_inputs, int num_predictions);
        ~NeuralNet();

        //Delete copying operators for this class
        NeuralNet(const NeuralNet&) = delete;
        NeuralNet& operator=(const NeuralNet&) = delete;
};

#endif