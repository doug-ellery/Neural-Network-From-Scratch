#include "Layer.h"
#include <vector>
#include <string>
#include <cublas_v2.h>
#include <random>


#ifndef NEURALNET_H
#define NEURALNET_H

class NeuralNet{
    
    rmm::device_uvector<float> inputs, predictions, correctOutputs, startingDelta, transposed_batch;
    float learning_rate, curr_cost;
    int numSamples, outputSize, inputSize, batch_size;
    Activation_funcs activation_func;
    cublasHandle_t handle;
    //host side arrays for the inputs and outputs
    std::vector<float> h_inputs, h_outputs, shuffled_inputs, shuffled_outputs;
    //members needed for shuffling the inputs and outputs
    std::vector<int> h_inputs_indices;
    std::mt19937 rng;
    std::vector<Layer> layers;

    public:
        NeuralNet(int, int, int, int, int, std::vector<float>&, std::vector<float>&, Activation_funcs);
        void forwardPass(const float*);
        void getStartingDelta(const float*);
        void getCost(const float*);
        void getAllDeltas();
        void backProp(int, const float*, const float*);
        void train();
        std::vector<float> predict(const std::vector<float>&, int);
        ~NeuralNet();

        //Delete copying operators for this class
        NeuralNet(const NeuralNet&) = delete;
        NeuralNet& operator=(const NeuralNet&) = delete;
};

#endif