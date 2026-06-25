#include <vector>
#include <string>
#include <cublas_v2.h>


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
    //Adam optimization terms (momentum & variance for weights and biases)
    float * m_weights, *v_weights, *m_biases, *v_biases;
    float beta_1, beta_2, epsilon;

    std::string activation_func;

    public:
        static cublasHandle_t handle;
        Layer(int, int, int, bool, std::string, int);
        Layer(Layer&& other) noexcept;
        float* getNextLayer(float* prevLayer, int batch_size);
        float* getNextLayerPrediction(float* prevLayer);
        void getWeightGradients(float* delta_l, float* a_l_minus_one, int batch_size);
        void getBiasGradients(float* delta_l, int batch_size);
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
        float* getDelta(float* deltaLPlusOne, float* z_l, int batch_size);
        float * returnDelta();
        float* getZ();
        void updateWeights(float learning_rate, int t);
        void updateBiases(float learning_rate, int t);
        ~Layer();

        //Don't allow layers to be copied, this causes wierd stuff to happen because we 
        //have raw pointers that we use to free cuda memory, we want these pointers to ALWAYS be unique
        Layer(const Layer&) = delete;
        Layer& operator=(const Layer&) = delete;
};

#endif