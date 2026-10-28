#include <vector>
#include "Layer.h"
#include <iostream>
#include "MyMatrix.h"
#include <ctime>
#include <cuda_runtime.h>
#include <string>
#include <rmm/device_uvector.hpp>
#include <rmm/cuda_stream_view.hpp>

cublasHandle_t Layer::handle;

Layer::Layer(int nodesThisLayer, int nodesNextLayer, int numSamples, bool lastLayer, Activation_funcs activation_func, int index){
    //n_in = how many nodes are in our layer, n_out = how many nodes are in the next layer,
    //ie. how many weights each node needs to have. n_out = how many nodes are in the next layer,
    //and in this design, biases is going to be the biases for the next layer, just how the 
    //weights are the weights that take us to the next layer, thus there are n_out biases for 
    //the next layer as each node has one bias
    //In multiplication, we well treat weights as an n_out x n_in matrix
    n_in = nodesThisLayer;
    n_out = nodesNextLayer;
    samples = numSamples;
    this->lastLayer = lastLayer;
    this->activation_func = activation_func;
    beta_1 = 0.9f;
    beta_2 = 0.999f;
    epsilon = 1e-8f;
    //allocate memory for all of our device arrays
    weights = rmm::device_uvector<float>(n_in*n_out, rmm::cuda_stream_default);
    biases = rmm::device_uvector<float>(n_out, rmm::cuda_stream_default);
    a = rmm::device_uvector<float>(n_out*samples, rmm::cuda_stream_default);
    z = rmm::device_uvector<float>(n_out*samples, rmm::cuda_stream_default);
    delta = rmm::device_uvector<float>(n_in*samples, rmm::cuda_stream_default);
    weight_gradients = rmm::device_uvector<float>(n_in*n_out, rmm::cuda_stream_default);
    bias_gradients = rmm::device_uvector<float>(n_out, rmm::cuda_stream_default);
    m_weights = rmm::device_uvector<float>(n_in*n_out, rmm::cuda_stream_default);
    v_weights = rmm::device_uvector<float>(n_in*n_out, rmm::cuda_stream_default);
    m_biases = rmm::device_uvector<float>(n_out, rmm::cuda_stream_default);
    v_biases = rmm::device_uvector<float>(n_out, rmm::cuda_stream_default);
    

    
    //zero out this memory (weights aren't zeroed out because they are initialized below)
    CUDA_CHECK(cudaMemset(biases.data(), 0.0, n_out * sizeof(float)));
    CUDA_CHECK(cudaMemset(m_weights.data(), 0.0, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset(v_weights.data(), 0.0, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset(m_biases.data(), 0.0, n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset(v_biases.data(), 0.0, n_out*sizeof(float)));

    //weights cannot, use He normal initialization because we are using ReLU for now as the activation function
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_in * n_out);
    //time has second-level precision, so use the index passed in to make sure layers constructed in the same second don't have the same seed
    unsigned long seed = (unsigned long)time(NULL) + index;

    //call our He normal kernel
    if(activation_func == Activation_funcs::ReLU){
        weightHeInitializeKernel<<<numBlocks, threadsPerBlock>>>(weights.data(), n_in * n_out, n_in, seed);
    }
    else if(activation_func == Activation_funcs::Tanh){
        weightXavierInitializeKernel<<<numBlocks, threadsPerBlock>>>(weights.data(), n_in * n_out, n_in, seed);
    }
    else{
        std::cerr<<"Bad activation func"<<std::endl;
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


//getNextLayer will use the weights, biases, and activation to get us the next layer and 
//return the values for this layer
//note that prevLayer should be the attribute "a" for the previous layer

const float* Layer::getNextLayer(const float* prevLayer, int batch_size){
    //multiply weights x prevLayer first, store it in preactivation (z)
    cudaMultiply(weights.data(), prevLayer, z.data(), n_out, batch_size, n_in, handle);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Add this matrix and biases
    cudaAdd(z.data(), biases.data(), n_out, batch_size);

    //copy z contents into a, and if there is an activation function, a will change accordingly
    CUDA_CHECK(cudaMemcpy(a.data(), z.data(), n_out*batch_size*sizeof(float), cudaMemcpyDeviceToDevice));

    if(!lastLayer){
        //apply the activation to a now
        if(activation_func == Activation_funcs::ReLU){
            cudaReLUActivation(a.data(), n_out * batch_size);
        }
        else if(activation_func == Activation_funcs::Tanh){
            cudaTanhActivation(a.data(), n_out*batch_size);
        }
        else{
            std::cerr<<"Bad activation func"<<std::endl;
            exit(1);
        }
    }
    else{

        //if its the last layer, we need to apply softmax, so get the softmax sum and then apply softmax
        int threadsPerBlock, numBlocks;
        getThreadsBlocks(threadsPerBlock, numBlocks, n_out*batch_size);
        rmm::device_uvector<float> sums(batch_size, rmm::cuda_stream_default);
        rmm::device_uvector<float> maxes(batch_size, rmm::cuda_stream_default);
        CUDA_CHECK(cudaMemset(sums.data(), 0.0, batch_size * sizeof(float)));
        getSoftmaxMaxKernel<<<numBlocks, threadsPerBlock>>>(z.data(), maxes.data(), n_out, batch_size);
        getSoftmaxSumKernel<<<numBlocks, threadsPerBlock>>>(z.data(), sums.data(), maxes.data(), n_out, batch_size);
        softmaxActivationKernel<<<numBlocks, threadsPerBlock>>>(a.data(), sums.data(), maxes.data(), n_out, batch_size);
    }
    return a.data();
}

//get the previous delta given the delta of the layer after this one
//using RELU -> second ptr is z_l, using tanh -> second ptr is a_l
const float* Layer::calculateDelta(const float* deltaLPlusOne, const float* z_l_or_a_l, int batch_size){
    //compute weights^T*delta^l+1 first, where M = n_in, K = n_out, N = samples
    cudaMultiply(weights.data(), deltaLPlusOne, delta.data(), n_in, batch_size, n_out, handle, CUBLAS_OP_T, CUBLAS_OP_N);
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_in*batch_size);

    if(activation_func == Activation_funcs::ReLU){
        reluPrimeKernel<<<numBlocks, threadsPerBlock>>>(z_l_or_a_l, delta.data(), n_in, batch_size);
    }
    else if(activation_func == Activation_funcs::Tanh){
        tanhPrimeKernel<<<numBlocks, threadsPerBlock>>>(z_l_or_a_l, delta.data(), n_in, batch_size); 
    }
    else{
        std::cerr<<"Bad activation func"<<std::endl;
        exit(1);
    }
    return delta.data();
}


//gradient functions, important note: I needed to get passed delta_l because technically 
//the delta stored in this layer is actually delta_l-1, when using standard layer conventions
void Layer::getWeightGradients(const float* delta_l, const float* a_l_minus_one, int batch_size){
    //follow formula: dC/dW_l = delta_l * (a_l-1)^T
    //delta_l = n_out x samples, a_l-1 = n_in x samples
    //thus, M = n_out, K = samples, M = n_in
    cudaMultiply(delta_l, a_l_minus_one, weight_gradients.data(), n_out, n_in, batch_size, handle, CUBLAS_OP_N, CUBLAS_OP_T);
    //average out by multiplying by 1/samples
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out*n_in);
    scalarMultiplyKernel<<<numBlocks, threadsPerBlock>>>(weight_gradients.data(), 1.0f/batch_size, n_out*n_in);
}

void Layer::getBiasGradients(const float* delta_l, int batch_size){
    //get rid of previous bias gradient so that the column sum doesn't add to this
    CUDA_CHECK(cudaMemset(bias_gradients.data(), 0.0, n_out*sizeof(float)));
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out*batch_size);
    //column sum for each delta from each sample
    columnSumKernel<<<numBlocks, threadsPerBlock>>>(delta_l, bias_gradients.data(), n_out, batch_size);
    //average out
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out);
    scalarMultiplyKernel<<<numBlocks, threadsPerBlock>>>(bias_gradients.data(), 1.0f/batch_size, n_out);
}
//getter for the preactivation, which is needed so the neural network can give z from one 
//layer to another for the delta calculation
const float* Layer:: getZ() const{
    return z.data();
}

const float* Layer::getActivation() const{
    return a.data();
}

const float* Layer::getDelta() const{
    return delta.data();
}

//updating weights after backprop
void Layer::updateWeights(float learning_rate, int t){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out*n_in);
    //call the update kernel with the weight array and weight gradient
    updateParameterKernel<<<numBlocks, threadsPerBlock>>>(weights.data(), weight_gradients.data(), learning_rate, n_out*n_in, m_weights.data(), v_weights.data(), beta_1, beta_2, epsilon, t);
}

//updating biases after backprop
void Layer::updateBiases(float learning_rate, int t){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out);
    //call the update kernel with the bias array and bias gradient
    updateParameterKernel<<<numBlocks, threadsPerBlock>>>(biases.data(), bias_gradients.data(), learning_rate, n_out, m_biases.data(), v_biases.data(), beta_1, beta_2, epsilon, t);
}




