#include <vector>
#include <string>
#include <cublas_v2.h>


#ifndef LAYER_H
#define LAYER_H

class Layer{
    rmm::device_uvector<float> weights, biases, a, z, weight_gradients, bias_gradients, delta, m_weights, v_weights, m_biases, v_biases;
    int n_in, n_out, samples;
    //For my forward pass logic, so I know whether or not to apply activation
    bool lastLayer;
    float beta_1, beta_2, epsilon;

    Activation_funcs activation_func;

    public:
        static cublasHandle_t handle;
        Layer(int, int, int, bool, Activation_funcs, int);
        //move constructor
        Layer(Layer&& other) noexcept = default;
        const float* getNextLayer(const float* const prevLayer, int batch_size);
        void getWeightGradients(const float* const delta_l, const float* const  a_l_minus_one, int batch_size);
        void getBiasGradients(const float* const delta_l, int batch_size);
        const float * getActivation();
        const float* getDelta(const float* const deltaLPlusOne, const float* const z_l, int batch_size);
        const float * returnDelta();
        const float* getZ();
        void updateWeights(float learning_rate, int t);
        void updateBiases(float learning_rate, int t);
        ~Layer();

        //Don't allow layers to be copied, this causes wierd stuff to happen because we 
        //have raw pointers that we use to free cuda memory, we want these pointers to ALWAYS be unique
        Layer(const Layer&) = delete;
        Layer& operator=(const Layer&) = delete;
};

#endif