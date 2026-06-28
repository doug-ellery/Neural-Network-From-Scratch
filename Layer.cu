#include <vector>
#include "Layer.h"
#include <iostream>
#include "MyMatrix.h"
#include <ctime>
#include <cuda_runtime.h>
#include <string>

cublasHandle_t Layer::handle;

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
    beta_1 = 0.9f;
    beta_2 = 0.999f;
    epsilon = 1e-8f;
    //allocate memory for all of our device arrays
    CUDA_CHECK(cudaMalloc((void **)&weights, n_in * n_out *sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&biases, n_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a, n_out*samples*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&z, n_out*samples*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&delta, n_in*samples*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&weight_gradients, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&bias_gradients, n_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&m_weights, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&v_weights, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&m_biases, n_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&v_biases, n_out*sizeof(float)));
    

    
    //zero out this memory (weights aren't zeroed out because they are initialized belowcan)
    CUDA_CHECK(cudaMemset((void *)biases, 0.0, n_out * sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)a, 0.0, n_out * samples * sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)z, 0.0, n_out * samples * sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)delta, 0.0, n_in * samples * sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)weight_gradients, 0.0, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset((void *)bias_gradients, 0.0, n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset((void**)m_weights, 0.0, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset((void**)v_weights, 0.0, n_in*n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset((void**)m_biases, 0.0, n_out*sizeof(float)));
    CUDA_CHECK(cudaMemset((void**)v_biases, 0.0, n_out*sizeof(float)));

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
    activation_func = other.activation_func;
    m_weights = other.m_weights;
    v_weights = other.v_weights;
    m_biases = other.m_biases;
    v_biases = other.v_biases;

    n_in = other.n_in;
    n_out = other.n_out;
    samples = other.samples;
    lastLayer = other.lastLayer;
    beta_1 = other.beta_1;
    beta_2 = other.beta_2;
    epsilon = other.epsilon;
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
    other.m_weights = nullptr;
    other.v_weights = nullptr;
    other.m_biases = nullptr;
    other.v_biases = nullptr;
}

//getNextLayer will use the weights, biases, and activation to get us the next layer and 
//return the values for this layer
//note that prevLayer should be the attribute "a" for the previous layer

float* Layer::getNextLayer(float* prevLayer, int batch_size){
    //multiply weights x prevLayer first, store it in preactivation (z)
    cudaMultiply(weights, prevLayer, z, n_out, batch_size, n_in, handle);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Add this matrix and biases
    cudaAdd(z, biases, n_out, batch_size);

    //copy z contents into a, and if there is an activation function, a will change accordingly
    CUDA_CHECK(cudaMemcpy(a, z, n_out*batch_size*sizeof(float), cudaMemcpyDeviceToDevice));

    if(!lastLayer){
        //apply the activation to a now
        if(activation_func == "RELU"){
            cudaReLUActivation(a, n_out * batch_size);
        }
        else{
            cudaTanhActivation(a, n_out*batch_size);
        }
    }
    else{

        //if its the last layer, we need to apply softmax, so get the softmax sum and then apply softmax
        int threadsPerBlock, numBlocks;
        getThreadsBlocks(threadsPerBlock, numBlocks, n_out*batch_size);
        float * sums;
        float * maxes;
        CUDA_CHECK(cudaMalloc(&sums, batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&maxes, batch_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(sums, 0.0, batch_size * sizeof(float)));
        getSoftmaxMaxKernel<<<numBlocks, threadsPerBlock>>>(z, maxes, n_out, batch_size);
        cudaDeviceSynchronize();
        getSoftmaxSumKernel<<<numBlocks, threadsPerBlock>>>(z, sums, maxes, n_out, batch_size);
        cudaDeviceSynchronize();
        softmaxActivationKernel<<<numBlocks, threadsPerBlock>>>(a, sums, maxes, n_out, batch_size);
        CUDA_CHECK(cudaFree(sums));
        CUDA_CHECK(cudaFree(maxes));
    }
    float* out;
    CUDA_CHECK(cudaMalloc((void**)&out, n_out*batch_size*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(out, a, n_out*batch_size*sizeof(float), cudaMemcpyDeviceToDevice));
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
float* Layer::getDelta(float* deltaLPlusOne, float* z_l_or_a_l, int batch_size){
    //compute weights^T*delta^l+1 first, where M = n_in, K = n_out, N = samples
    cudaMultiply(weights, deltaLPlusOne, delta, n_in, batch_size, n_out, handle, CUBLAS_OP_T, CUBLAS_OP_N);
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_in*batch_size);

    if(activation_func == "RELU"){
        reluPrimeKernel<<<numBlocks, threadsPerBlock>>>(z_l_or_a_l, delta, n_in, batch_size);
    }
    else{
        tanhPrimeKernel<<<numBlocks, threadsPerBlock>>>(z_l_or_a_l, delta, n_in, batch_size); 
    }
    float* out;
    CUDA_CHECK(cudaMalloc((void**)&out, n_in*batch_size*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(out, delta, n_in*batch_size*sizeof(float), cudaMemcpyDeviceToDevice));
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
void Layer::getWeightGradients(float* delta_l, float* a_l_minus_one, int batch_size){
    //follow formula: dC/dW_l = delta_l * (a_l-1)^T
    //delta_l = n_out x samples, a_l-1 = n_in x samples
    //thus, M = n_out, K = samples, M = n_in
    cudaMultiply(delta_l, a_l_minus_one, weight_gradients, n_out, n_in, batch_size, handle, CUBLAS_OP_N, CUBLAS_OP_T);
    //average out by multiplying by 1/samples
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out*n_in);
    scalarMultiplyKernel<<<numBlocks, threadsPerBlock>>>(weight_gradients, 1.0f/batch_size, n_out*n_in);
}

void Layer::getBiasGradients(float* delta_l, int batch_size){
    //get rid of previous bias gradient so that the column sum doesn't add to this
    CUDA_CHECK(cudaMemset(bias_gradients, 0.0, n_out*sizeof(float)));
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out*batch_size);
    //column sum for each delta from each sample
    columnSumKernel<<<numBlocks, threadsPerBlock>>>(delta_l, bias_gradients, n_out, batch_size);
    //back to back kernels that are changing bias_gradients, so we should sync up before 
    //moving to the second kernel
    cudaDeviceSynchronize();
    //average out
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out);
    scalarMultiplyKernel<<<numBlocks, threadsPerBlock>>>(bias_gradients, 1.0f/batch_size, n_out);
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
void Layer::updateWeights(float learning_rate, int t){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out*n_in);
    //call the update kernel with the weight array and weight gradient
    updateParameterKernel<<<numBlocks, threadsPerBlock>>>(weights, weight_gradients, learning_rate, n_out*n_in, m_weights, v_weights, beta_1, beta_2, epsilon, t);
}

//updating biases after backprop
void Layer::updateBiases(float learning_rate, int t){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, n_out);
    //call the update kernel with the bias array and bias gradient
    updateParameterKernel<<<numBlocks, threadsPerBlock>>>(biases, bias_gradients, learning_rate, n_out, m_biases, v_biases, beta_1, beta_2, epsilon, t);
}

void Layer::logGradientStats(){
    std::vector<float> w_grads(n_out*n_in);
    CUDA_CHECK(cudaMemcpy(w_grads.data(), weight_gradients, n_out*n_in*sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> b_grads(n_out);
    CUDA_CHECK(cudaMemcpy(b_grads.data(), bias_gradients, n_out*sizeof(float), cudaMemcpyDeviceToHost));
    float w_max = 0.0f;
    float w_mean = 0.0f;
    long w_infinites = 0;
    for(float grad : w_grads){
        w_max = std::max(w_max, std::fabs(grad));
        w_mean += std::fabs(grad) / (n_out*n_in);
        if (!std::isfinite(grad)) {
            w_infinites++;
        }
    }
    float b_max = 0.0f;
    float b_mean = 0.0f;
    long b_infinites = 0;
    for(float grad : b_grads){
        b_max = std::max(b_max, std::fabs(grad));
        b_mean += std::fabs(grad) / n_out;
        if (!std::isfinite(grad)) {
            b_infinites++;
        }
    }
    std::cout << "Weight gradients\n";
    std::cout << "  max |dW| = " << w_max << "\n";
    std::cout << "  mean|dW| = " << w_mean << "\n";
    std::cout << "Num infinites: "<<w_infinites<<"\n";

    std::cout << "Bias gradients\n";
    std::cout << "  max |db| = " << b_max << "\n";
    std::cout << "  mean|db| = " << b_mean << "\n";
    std::cout << "Num infinites: "<<b_infinites<<"\n\n";
}

void Layer::logZStats(int batch_size){
    std::vector<float> logits(n_out*batch_size);
    CUDA_CHECK(cudaMemcpy(logits.data(), z, n_out*batch_size*sizeof(float), cudaMemcpyDeviceToHost));
    float z_max = -std::numeric_limits<float>::infinity();
    float z_min =  std::numeric_limits<float>::infinity();
    float mean = 0;
    for(float node : logits){
        z_max = std::max(node, z_max);
        z_min = std::min(node, z_min);
        mean += std::fabs(node) / (n_out * batch_size);
    }
    std::cout<<"Logit stats:\n"<<"Mean: "<<mean<<"\nMax: "<<z_max<<"\nMin: "<<z_min<<"\n\n";
}


Layer::~Layer(){
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(biases));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(z));
    CUDA_CHECK(cudaFree(delta));
    CUDA_CHECK(cudaFree(weight_gradients));
    CUDA_CHECK(cudaFree(bias_gradients));
    CUDA_CHECK(cudaFree(m_weights));
    CUDA_CHECK(cudaFree(v_weights));
    CUDA_CHECK(cudaFree(m_biases));
    CUDA_CHECK(cudaFree(v_biases));
}



