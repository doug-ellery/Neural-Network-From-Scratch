#include <vector>
#include <string>

#ifndef LAYER_H
#define LAYER_H

class Layer{
    float* weights, *biases;
    int n_in, n_out, samples;
    //activation and preactivation arrays that we'll get from computing the next layer
    float* a, *z, *prediction_a, *prediction_z;
    //gradients for the weights and biases
    float* weight_gradients, *bias_gradients;
    //delta array
    float* delta;
    //For my forward pass logic, so I know whether or not to apply activation
    bool lastLayer;

    std::string activation_func;

    public:
        Layer(int, int, int, bool, std::string, int);
        Layer(Layer&& other) noexcept;
        float* getNextLayer(float*);
        float* getNextLayerPrediction(float* prevLayer);
        void getWeightGradients(float* delta_l, float* a_l_minus_one);
        void getBiasGradients(float* delta_l);
        void setWeights(std::vector<float> hardcodedWeights);
        void setBiases(std::vector<float> hardcodedBiases);
        void printActivation();
        void printWeights();
        void printBiases();
        void printWeightGradients();
        void printBiasGradients();
        void printPreActivation();
        void printNodes();
        void printDelta();
        float * getActivation();
        float* getDelta(float* deltaLPlusOne, float* z_l);
        float * returnDelta();
        float* getZ();
        void updateWeights(float learning_rate);
        void updateBiases(float learning_rate);
        ~Layer();

        //Don't allow layers to be copied, this causes wierd stuff to happen because we 
        //have raw pointers that we use to free cuda memory, we want these pointers to ALWAYS be unique
        Layer(const Layer&) = delete;
        Layer& operator=(const Layer&) = delete;
};

#endif