#include "NeuralNet.h"
#include "Layer.h"
#include "MyMatrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <climits>
#include <string>


NeuralNet::NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>& data, std::vector<float>& outputs, std::string activation_func){
    this->numSamples = numSamples;
    this->outputSize = outputSize;
    this->inputSize = inputSize;
    if(activation_func != "RELU" && activation_func != "TANH"){
        std::cerr<<"Bad activation function param\n";
        exit(1);
    }
    this->activation_func = activation_func;
    learning_rate = 0.03f;
    curr_cost = INT_MAX;
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

}


std::vector<float> NeuralNet::forwardPass(std::vector<float> prediction_inputs){
    //prediction_input = device version of prediction_inputs
    float * prediction_input = nullptr;
    if(prediction_inputs.size() != 0){
        CUDA_CHECK(cudaMalloc(&prediction_input, prediction_inputs.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(prediction_input, prediction_inputs.data(), prediction_inputs.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
    //we either want to start a forward pass with the training data (inputs), or the input for a prediction
    float* prev = prediction_input!= nullptr ? layers[0].getNextLayerPrediction(prediction_input) : layers[0].getNextLayer(inputs);
    float* curr = nullptr;
    for(int i = 1; i < layers.size(); i++){
        //curr = layers[i].getNextLayer(prev);
        curr = prediction_input != nullptr ? layers[i].getNextLayerPrediction(prev) : layers[i].getNextLayer(prev);
        CUDA_CHECK(cudaFree(prev));
        prev = curr;
    }
    if(prediction_input == nullptr){
        predictions = prev;
    }
    //training forward pass case
    if(prediction_input == nullptr){
        std::vector<float> outVec(numSamples * outputSize);
        CUDA_CHECK(cudaMemcpy(outVec.data(), predictions, numSamples * outputSize * sizeof(float), cudaMemcpyDeviceToHost));
        return outVec;
    }
    std::vector<float> outVec(outputSize);
    CUDA_CHECK(cudaMemcpy(outVec.data(), prev, outputSize*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(prediction_input));
    return outVec;
}

void NeuralNet::getStartingDelta(){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, numSamples*outputSize);

    //call starting delta kernel
    startingDeltaKernel<<<numBlocks, threadsPerBlock>>>(predictions, correctOutputs, startingDelta, numSamples*outputSize);

}


void NeuralNet::getCost(){
    //Get a cost variable onto the GPU
    float* cost;
    CUDA_CHECK(cudaMalloc(&cost, sizeof(float)));
    CUDA_CHECK(cudaMemset(cost, 0, sizeof(float)));
    
    //get threads per block and number of blocks
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, numSamples*outputSize);

    //call cost kernel
    costKernel<<<numBlocks, threadsPerBlock>>>(predictions, correctOutputs, cost, numSamples, outputSize);
    CUDA_CHECK(cudaMemcpy(&curr_cost, cost, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cost));
    std::cout<<"Cost: "<<curr_cost<<"\n";
    
}

void NeuralNet::getAllDeltas(){
    float* deltaLPlusOne = startingDelta;
    float* temp;
    for(int i = layers.size() - 1; i > 0; i--){
        temp = activation_func == "RELU" ? layers[i].getDelta(deltaLPlusOne, layers[i - 1].getZ()) : layers[i].getDelta(deltaLPlusOne, layers[i - 1].getActivation());
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

void NeuralNet::backProp(){
    getCost();
    getStartingDelta();
    getAllDeltas();
    float * delta_l = startingDelta;
    float * a_l_minus_one = layers.size() == 1 ? inputs : layers[layers.size() - 2].getActivation();
    for(int i = layers.size() - 1; i >= 0; i--){
        //calculate gradients
        layers[i].getWeightGradients(delta_l, a_l_minus_one);
        layers[i].getBiasGradients(delta_l);
        //update based on those gradients
        layers[i].updateWeights(learning_rate);
        layers[i].updateBiases(learning_rate);
        delta_l = layers[i].returnDelta();
        if(i != 0){
            a_l_minus_one = i == 1 ? inputs : layers[i - 2].getActivation();
        }
    }
}

void NeuralNet::train(){
    for(int epoch = 0; epoch < 5000; epoch++){
        forwardPass();
        backProp();
        if(curr_cost < 1e-6){
            break;
        }
    }
}

//predict an output by running a forward pass
std::vector<float> NeuralNet::predict(std::vector<float> prediction_inputs){
    return forwardPass(prediction_inputs);
}

NeuralNet::~NeuralNet(){
    //free all the GPU memory we allocated
    CUDA_CHECK(cudaFree(inputs));
    CUDA_CHECK(cudaFree(predictions));
    CUDA_CHECK(cudaFree(correctOutputs));
    CUDA_CHECK(cudaFree(startingDelta));
}

