#include <vector>
#include "Layer.h"
#include <iostream>
#include "MyMatrix.h"
#include <ctime>
#include <cuda_runtime.h>
#include <string>

Layer::Layer(int nodesThisLayer, int nodesNextLayer, int numSamples, bool lastLayer, std::string activation_func, int index){
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
    //allocate memory for all of our device arrays
    CUDA_CHECK(cudaMalloc((void **)&weights, n_in * n_out *sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&biases, n_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a, n_out*samples*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&z, n_out*samples*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&delta, n_in*samples*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&weight_gradients, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&bias_gradients, n_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&prediction_a, n_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&prediction_z, n_out*sizeof(float)));
    

    
    //biases can be initialized to 0, same with a, z, delta, and gradients
    CUDA_CHECK(cudaMemset((void *)biases, 0.0, n_out * sizeof(float)));

    CUDA_CHECK(cudaMemset((void *)a, 0.0, n_out * samples * sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)z, 0.0, n_out * samples * sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)delta, 0.0, n_in * samples * sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)weight_gradients, 0.0, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)bias_gradients, 0.0, n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset((void *) prediction_a, 0.0, n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset((void *) prediction_z, 0.0, n_out*sizeof(float)));

    //weights cannot, use He normal initialization because we are using ReLU for now as the activation function
    CUDA_CHECK(cudaMemset((void *)weights, 0, n_in * n_out * sizeof(float)));
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_in * n_out);
    //time has second-level precision, so use the index passed in to make sure layers constructed in the same second don't have the same seed
    unsigned long seed = (unsigned long)time(NULL) + index;

    //call our He normal kernel
    if(activation_func == "RELU"){
        weightHeInitializeKernel<<<numBlocks, threadsPerBlock>>>(weights, n_in * n_out, n_in, seed);
    }
    else{
        weightXavierInitializeKernel<<<numBlocks, threadsPerBlock>>>(weights, n_in * n_out, n_in, seed);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//Move constructor so that std::vector uses std::move when I call push_back, instead of a normal
//copy operation
Layer::Layer(Layer&& other) noexcept {
    weights = other.weights;
    biases = other.biases;
    z = other.z;
    a = other.a;
    delta = other.delta;
    weight_gradients = other.weight_gradients;
    bias_gradients = other.bias_gradients;
    prediction_a = other.prediction_a;
    prediction_z = other.prediction_z;
    activation_func = other.activation_func;

    n_in = other.n_in;
    n_out = other.n_out;
    samples = other.samples;
    lastLayer = other.lastLayer;
    other.n_in = 0;
    other.n_out = 0;
    other.samples = 0;
    other.activation_func = "";

    other.weights = nullptr;
    other.biases = nullptr;
    other.z = nullptr;
    other.a = nullptr;
    other.delta = nullptr;
    other.weight_gradients = nullptr;
    other.bias_gradients = nullptr;
    other.prediction_a = nullptr;
    other.prediction_z = nullptr;
}

//getNextLayer will use the weights, biases, and activation to get us the next layer and 
//return the values for this layer
//note that prevLayer should be the attribute "a" for the previous layer

float* Layer::getNextLayer(float* prevLayer){
    //multiply weights x prevLayer first, store it in preactivation (z)
    cudaMultiply(weights, prevLayer, z, n_out, samples, n_in);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Add this matrix and biases
    cudaAdd(z, biases, n_out, samples);

    //copy z contents into a, and if there is an activation function, a will change accordingly
    CUDA_CHECK(cudaMemcpy(a, z, n_out*samples*sizeof(float), cudaMemcpyDeviceToDevice));

    if(!lastLayer){
        //apply the activation to a now
        if(activation_func == "RELU"){
            cudaReLUActivation(a, n_out * samples);
        }
        else{
            cudaTanhActivation(a, n_out*samples);
        }
    }
    float* out;
    CUDA_CHECK(cudaMalloc((void**)&out, n_out*samples*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(out, a, n_out*samples*sizeof(float), cudaMemcpyDeviceToDevice));
    return out;
}

//A version of getNextLayer designed for when we are running a prediction, and not training using a batch,
//essentially just getNextLayer but using samples = 1;

float* Layer::getNextLayerPrediction(float* prevLayer){
    //multiply weights x prevLayer first, store it in preactivation (prediction_z)
    cudaMultiply(weights, prevLayer, prediction_z, n_out, 1, n_in);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Add this matrix and biases
    cudaAdd(prediction_z, biases, n_out, 1);

    //copy z contents into prediction_a, and if there is an activation function, a will change accordingly
    CUDA_CHECK(cudaMemcpy(prediction_a, prediction_z, n_out*sizeof(float), cudaMemcpyDeviceToDevice));

    if(!lastLayer){
        //apply the activation to prediction_a now
        if(activation_func == "RELU"){
            cudaReLUActivation(prediction_a, n_out * 1);
        }
        else{
            cudaTanhActivation(prediction_a, n_out* 1);
        }
    }
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&out, n_out*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(out, prediction_a, n_out*sizeof(float), cudaMemcpyDeviceToDevice));
    return out;
}

//Helper function for debugging the weights
void Layer::printWeights(){
    std::vector<float> printer(n_out*n_in, 0);
    std::cout<<"Weights: \n";
    CUDA_CHECK(cudaMemcpy(printer.data(), weights, n_out*n_in*sizeof(float), cudaMemcpyDeviceToHost));
    printVec(printer, n_out, n_in);
}

void Layer::printBiases(){
    std::vector<float> printer(n_out, 0);
    std::cout<<"Biases: \n";
    CUDA_CHECK(cudaMemcpy(printer.data(), biases, n_out*sizeof(float), cudaMemcpyDeviceToHost));
    printVec(printer, n_out, 1);
}

//for debugging
void Layer::printActivation(){
    std::vector<float> nodes(n_out*samples, 0);
    CUDA_CHECK(cudaMemcpy(nodes.data(), a, n_out*samples*sizeof(float), cudaMemcpyDeviceToHost));
    printVec(nodes, n_out, samples);
}
//for debugging
void Layer::printPreActivation(){
    std::vector<float> nodes(n_out*samples, 0);
    CUDA_CHECK(cudaMemcpy(nodes.data(), z, n_out*samples*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < nodes.size(); i++){
        std::cout<<nodes[i]<<"\n";
    }
}

void Layer::setBiases(std::vector<float> hardcodedBiases){
    CUDA_CHECK(cudaMemcpy((void *)biases, hardcodedBiases.data(), n_out*sizeof(float), cudaMemcpyHostToDevice));
}

void Layer::setWeights(std::vector<float> hardcodedWeights){
    CUDA_CHECK(cudaMemcpy((void *)weights, hardcodedWeights.data(), n_out*n_in*sizeof(float), cudaMemcpyHostToDevice));
}

//get the previous delta given the delta of the layer after this one
//using RELU -> second ptr is z_l, using tanh -> second ptr is a_l
float* Layer::getDelta(float* deltaLPlusOne, float* z_l_or_a_l){
    //compute weights^T*delta^l+1 first, where M = n_in, K = n_out, N = samples
    cudaMultiply(weights, deltaLPlusOne, delta, n_in, samples, n_out, CUBLAS_OP_T, CUBLAS_OP_N);
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_in*samples);

    if(activation_func == "RELU"){
        reluPrimeKernel<<<numBlocks, threadsPerBlock>>>(z_l_or_a_l, delta, n_in, samples);
    }
    else{
        tanhPrimeKernel<<<numBlocks, threadsPerBlock>>>(z_l_or_a_l, delta, n_in, samples); 
    }
    float* out;
    CUDA_CHECK(cudaMalloc((void**)&out, n_in*samples*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(out, delta, n_in*samples*sizeof(float), cudaMemcpyDeviceToDevice));
    return out;
}

//getter for the preactivation, which is needed so the neural network can give z from one 
//layer to another for the delta calculation
float* Layer:: getZ(){
    return z;
}


void Layer::printDelta(){
    std::vector<float> printer(n_in*samples);
    CUDA_CHECK(cudaMemcpy(printer.data(), delta, n_in*samples*sizeof(float), cudaMemcpyDeviceToHost));
    printVec(printer, n_in, samples);
}

//gradient functions, important note: I needed to get passed delta_l because technically 
//the delta stored in this layer is actually delta_l-1, when using standard layer conventions
void Layer::getWeightGradients(float* delta_l, float* a_l_minus_one){
    //follow formula: dC/dW_l = delta_l * (a_l-1)^T
    //delta_l = n_out x samples, a_l-1 = n_in x samples
    //thus, M = n_out, K = samples, M = n_in
    cudaMultiply(delta_l, a_l_minus_one, weight_gradients, n_out, n_in, samples, CUBLAS_OP_N, CUBLAS_OP_T);
    //average out by multiplying by 1/samples
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out*n_in);
    scalarMultiplyKernel<<<numBlocks, threadsPerBlock>>>(weight_gradients, 1.0f/samples, n_out*n_in);
}

void Layer::getBiasGradients(float* delta_l){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out*samples);
    //column sum for each delta from each sample
    columnSumKernel<<<numBlocks, threadsPerBlock>>>(delta_l, bias_gradients, n_out, samples);
    //back to back kernels that are changing bias_gradients, so we should sync up before 
    //moving to the second kernel
    cudaDeviceSynchronize();
    //average out
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out);
    scalarMultiplyKernel<<<numBlocks, threadsPerBlock>>>(bias_gradients, 1.0f/samples, n_out);
}

void Layer::printWeightGradients(){
    std::vector<float> printer(n_out*n_in);
    CUDA_CHECK(cudaMemcpy(printer.data(), weight_gradients, n_out*n_in*sizeof(float), cudaMemcpyDeviceToHost));
    printVec(printer, n_out, n_in);
}

void Layer::printBiasGradients(){
    std::vector<float> printer(n_out);
    CUDA_CHECK(cudaMemcpy(printer.data(), bias_gradients, n_out*sizeof(float), cudaMemcpyDeviceToHost));
    printVec(printer, n_out, 1);
}

float * Layer::getActivation(){
    return a;
}

float * Layer::returnDelta(){
    return delta;
}

//updating weights after backprop
void Layer::updateWeights(float learning_rate){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out*n_in);
    //call the update kernel with the weight array and weight gradient
    updateParameterKernel<<<numBlocks, threadsPerBlock>>>(weights, weight_gradients, learning_rate, n_out*n_in);
}

//updating biases after backprop
void Layer::updateBiases(float learning_rate){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out);
    //call the update kernel with the bias array and bias gradient
    updateParameterKernel<<<numBlocks, threadsPerBlock>>>(biases, bias_gradients, learning_rate, n_out);
}

Layer::~Layer(){
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(biases));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(z));
    CUDA_CHECK(cudaFree(delta));
    CUDA_CHECK(cudaFree(weight_gradients));
    CUDA_CHECK(cudaFree(bias_gradients));
    CUDA_CHECK(cudaFree(prediction_a));
    CUDA_CHECK(cudaFree(prediction_z));
}



