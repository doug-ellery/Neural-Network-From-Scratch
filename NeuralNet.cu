#include "NeuralNet.h"
#include "Layer.h"
#include "MyMatrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <climits>
#include <string>
#include <cublas_v2.h>
#include <numeric>
#include <random>


NeuralNet::NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>& data, std::vector<float>& outputs, std::string activation_func){
    this->numSamples = numSamples;
    this->outputSize = outputSize;
    this->inputSize = inputSize;
    if(activation_func != "RELU" && activation_func != "TANH"){
        std::cerr<<"Bad activation function param\n";
        exit(1);
    }
    this->activation_func = activation_func;
    learning_rate = 0.0006f;
    curr_cost = INT_MAX;
    CUBLAS_CHECK(cublasCreate(&handle));

    //Enable tensor cores to be used
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    Layer::handle = handle;
    batch_size = 64;
    //create input layer weights
    layers.reserve(numHiddenLayers + 1);
    layers.push_back(Layer(inputSize, nodesPerHiddenLayer, numSamples, false, activation_func, 0));
    int i = 0;
    //create hidden layer weights
    while(i < numHiddenLayers - 1){
        layers.push_back(Layer(nodesPerHiddenLayer, nodesPerHiddenLayer, numSamples, false, activation_func, i + 1));
        i++;
    }
    //create weights that take us to output layer
    layers.push_back(Layer(nodesPerHiddenLayer, outputSize, numSamples, true, activation_func, i + 1));
    //get values of first layer onto GPU
    CUDA_CHECK(cudaMalloc(&inputs, inputSize * numSamples * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(inputs, data.data(), inputSize * numSamples * sizeof(float), cudaMemcpyHostToDevice));
    //get predictions onto gpu (itll start as empty because we haven't done a forward pass
    CUDA_CHECK(cudaMalloc(&predictions, outputSize*numSamples*sizeof(float)));
    CUDA_CHECK(cudaMemset(predictions, 0, outputSize*numSamples*sizeof(float)));
    //get outputs onto GPU
    CUDA_CHECK(cudaMalloc(&correctOutputs, outputSize*numSamples*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(correctOutputs, outputs.data(), outputSize*numSamples*sizeof(float), cudaMemcpyHostToDevice));
    //get starting delta allocated
    CUDA_CHECK(cudaMalloc(&startingDelta, outputSize*numSamples*sizeof(float)));
    CUDA_CHECK(cudaMemset(startingDelta, 0, outputSize*numSamples*sizeof(float)));
    //make host side inputs and outputs, without making copies because these could be very large
    h_inputs = std::move(data);
    h_outputs = std::move(outputs);
    shuffled_inputs.resize(h_inputs.size());
    shuffled_outputs.resize(h_outputs.size());
    h_inputs_indices.resize(numSamples);
    //fill h_inputs_indices with values 0 -> numSamples - 1, each index maps to one sample in h_inputs that will be shuffled later
    std::iota(h_inputs_indices.begin(), h_inputs_indices.end(), 0);
    //random seed for std::shuffle
    rng = std::mt19937{std::random_device{}()};

}


std::vector<float> NeuralNet::forwardPass(float * batch_input){
    float* prev = layers[0].getNextLayer(batch_input, batch_size);
    float* curr = nullptr;
    for(int i = 1; i < layers.size(); i++){
        curr = layers[i].getNextLayer(prev, batch_size);
        CUDA_CHECK(cudaFree(prev));
        prev = curr;
    }
    CUDA_CHECK(cudaFree(predictions));
    predictions = prev;
    std::vector<float> outVec(batch_size * outputSize);
    CUDA_CHECK(cudaMemcpy(outVec.data(), predictions, batch_size * outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    return outVec;
}

void NeuralNet::getStartingDelta(float * batch_output){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, batch_size*outputSize);

    //call starting delta kernel
    startingDeltaKernel<<<numBlocks, threadsPerBlock>>>(predictions, batch_output, startingDelta, batch_size*outputSize);

}


void NeuralNet::getCost(float * batch_output){
    //init cost
    float * cost;
    CUDA_CHECK(cudaMalloc(&cost, sizeof(float)));
    CUDA_CHECK(cudaMemset(cost, 0, sizeof(float)));
    
    //get threads per block and number of blocks
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, batch_size*outputSize);

    //call cost kernel
    costKernel<<<numBlocks, threadsPerBlock>>>(predictions, batch_output, cost, batch_size, outputSize);
    CUDA_CHECK(cudaMemcpy(&curr_cost, cost, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cost));
    
}

void NeuralNet::getAllDeltas(){
    float* deltaLPlusOne = startingDelta;
    float* temp;
    for(int i = layers.size() - 1; i > 0; i--){
        temp = activation_func == "RELU" ? layers[i].getDelta(deltaLPlusOne, layers[i - 1].getZ(), batch_size) : layers[i].getDelta(deltaLPlusOne, layers[i - 1].getActivation(), batch_size);
        if(deltaLPlusOne != startingDelta){
            CUDA_CHECK(cudaFree(deltaLPlusOne));
        }
        deltaLPlusOne = temp;
    }
}

void NeuralNet::showAllDeltas(){
    std::vector<float> outputDelta(outputSize*numSamples);
    CUDA_CHECK(cudaMemcpy(outputDelta.data(), startingDelta, outputSize*numSamples*sizeof(float), cudaMemcpyDeviceToHost));
    printVec(outputDelta, outputSize, numSamples);
    for(int i = layers.size() - 1; i > 0; i--){
        std::cout<<"\n\n";
        layers[i].printDelta();
    }
}

void NeuralNet::backProp(int t, float * batch_input, float * batch_output){
    getCost(batch_output);
    getStartingDelta(batch_output);
    getAllDeltas();
    float * delta_l = startingDelta;
    float * a_l_minus_one = layers.size() == 1 ? batch_input : layers[layers.size() - 2].getActivation();
    for(int i = layers.size() - 1; i >= 0; i--){
        //calculate gradients
        layers[i].getWeightGradients(delta_l, a_l_minus_one, batch_size);
        layers[i].getBiasGradients(delta_l, batch_size);
        //update based on those gradients
        layers[i].updateWeights(learning_rate, t);
        layers[i].updateBiases(learning_rate, t);
        delta_l = layers[i].returnDelta();
        if(i != 0){
            a_l_minus_one = i == 1 ? batch_input : layers[i - 2].getActivation();
        }
    }
}

void NeuralNet::train(){
    //Adam needs this t variable that increments once per epoch
    int t = 1;
    int num_batches = numSamples / batch_size;
    for(int epoch = 0; epoch < 300000; epoch++){
        //Shuffle inputs and correctOutputs
        std::shuffle(h_inputs_indices.begin(), h_inputs_indices.end(), rng);
        //use these shuffled indices to shuffle the arrays using std::copy
        //we aren't necessarily just moving one number over, we are copying one input/output 
        //sample at a time, which is inputSize and outputSize number of elements
        for(int i = 0; i < numSamples; i++){
            //the index for the random samples that we want to copy into the i'th sample in
            //shuffled_inputs/outputs
            int src = h_inputs_indices[i];
            std::copy(h_inputs.begin() + src * inputSize, h_inputs.begin() + src * inputSize + inputSize, shuffled_inputs.begin() + i * inputSize);
            std::copy(h_outputs.begin() + src * outputSize, h_outputs.begin() + src * outputSize + outputSize, shuffled_outputs.begin() + i * outputSize);
        }
        //copy the shuffled versions of inputs and outputs onto the GPU
        CUDA_CHECK(cudaMemcpy(inputs, shuffled_inputs.data(), inputSize * numSamples * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(correctOutputs, shuffled_outputs.data(), outputSize * numSamples * sizeof(float), cudaMemcpyHostToDevice));
        for(int batch = 0; batch < num_batches; batch++){
            float * batch_input = inputs + batch * batch_size * inputSize;
            float * batch_output = correctOutputs + batch * batch_size * outputSize;
            forwardPass(batch_input);
            backProp(t, batch_input, batch_output);
            t++;
        }
        if(epoch % 10 == 0){
            std::cout<<"Epoch "<<epoch<<" | Cost "<<curr_cost<<"\n";
        }
    }
}

//predict an output by running a forward pass
std::vector<float> NeuralNet::predict(std::vector<float> prediction_inputs){
    //prediction_input = device version of prediction_inputs
    float * prediction_input = nullptr;
    CUDA_CHECK(cudaMalloc(&prediction_input, prediction_inputs.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(prediction_input, prediction_inputs.data(), prediction_inputs.size()*sizeof(float), cudaMemcpyHostToDevice));
    //same logic as forward pass but use getNextLayerPrediction
    float* prev = layers[0].getNextLayerPrediction(prediction_input);
    float* curr = nullptr;
    for(int i = 1; i < layers.size(); i++){
        curr = layers[i].getNextLayerPrediction(prev);
        CUDA_CHECK(cudaFree(prev));
        prev = curr;
    }
    std::vector<float> outVec(outputSize);
    CUDA_CHECK(cudaMemcpy(outVec.data(), prev, outputSize*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(prediction_input));
    return outVec;
}

NeuralNet::~NeuralNet(){
    //free all the GPU memory we allocated
    CUDA_CHECK(cudaFree(inputs));
    CUDA_CHECK(cudaFree(predictions));
    CUDA_CHECK(cudaFree(correctOutputs));
    CUDA_CHECK(cudaFree(startingDelta));
    CUBLAS_CHECK(cublasDestroy(handle));
}

