#include <vector>
#include <string>
#include <cublas_v2.h>
#include <rmm/device_uvector.hpp>
#include "MyMatrix.h"


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
        const float* getNextLayer(const float*, int);
        void getWeightGradients(const float*, const float*, int);
        void getBiasGradients(const float*, int);
        const float * getActivation() const;
        const float* calculateDelta(const float*, const float*, int);
        const float * getDelta() const;
        const float* getZ() const;
        void updateWeights(float, int);
        void updateBiases(float, int);

        //Don't allow layers to be copied, use std::move
        Layer(const Layer&) = delete;
        Layer& operator=(const Layer&) = delete;
};

#endif