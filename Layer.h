#include <vector>

#ifndef LAYER_H
#define LAYER_H

class Layer{
    float* weights, *biases;
    int n_in, n_out, samples;
    //activation and preactivation arrays that we'll get from computing the next layer
    float* a, *z;
    //delta array
    float* delta;
    //For my forward pass logic, so I know whether or not to apply activation
    bool lastLayer;
    public:
        Layer(int, int, int, bool);
        Layer(Layer&& other) noexcept;
        float* getNextLayer(float*);
        void setWeights(std::vector<float> hardcodedWeights);
        void setBiases(std::vector<float> hardcodedBiases);
        void printActivation();
        void printWeights();
        void printPreActivation();
        void printNodes();
        void printDelta();
        float* getDelta(float* deltaLPlusOne, float* z_l);
        float* Layer:: getZ();
        ~Layer();

        //Don't allow layers to be copied, this causes wierd stuff to happen because we 
        //have raw pointers that we use to free cuda memory, we want these pointers to ALWAYS be unique
        Layer(const Layer&) = delete;
        Layer& operator=(const Layer&) = delete;
};

#endif