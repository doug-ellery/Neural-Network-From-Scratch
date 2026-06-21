#include "NeuralNet.h"
#include "Layer.h"
#include "MyMatrix.h"
#include <cuda_runtime.h>
#include <iostream>


NeuralNet::NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>& data, std::vector<float>& outputs){
    this->numSamples = numSamples;
    this->outputSize = outputSize;
    this->inputSize = inputSize;
    learning_rate = 0.01;
    //create input layer weights
    layers.reserve(numHiddenLayers + 1);
    layers.push_back(Layer(inputSize, nodesPerHiddenLayer, numSamples, false));
    int i = 0;
    //create hidden layer weights
    while(i < numHiddenLayers - 1){
        layers.push_back(Layer(nodesPerHiddenLayer, nodesPerHiddenLayer, numSamples, false));
        i++;
    }
    //create weights that take us to output layer
    layers.push_back(Layer(nodesPerHiddenLayer, outputSize, numSamples, true));
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


std::vector<float> NeuralNet::forwardPass(){
    //std::cout<<"Layer 0 nodes: \n";
    //std::vector<float> layerZero(inputSize*numSamples, 0);
    //CUDA_CHECK(cudaMemcpy(layerZero.data(), inputs, inputSize*numSamples*sizeof(float), cudaMemcpyDeviceToHost));
    //printVec(layerZero, inputSize, numSamples);
    float* prev = layers[0].getNextLayer(inputs);
    //std::cout<<"Layer 1 nodes: \n";
    //layers[0].printActivation();
    float* curr = nullptr;
    for(int i = 1; i < layers.size(); i++){
        curr = layers[i].getNextLayer(prev);
        CUDA_CHECK(cudaFree(prev));
        //std::cout<<"Layer "<<i + 1<<" nodes: \n";
        //layers[i].printActivation();
        prev = curr;
    }
    predictions = prev;
    std::vector<float> outVec(numSamples * outputSize);
    CUDA_CHECK(cudaMemcpy(outVec.data(), predictions, numSamples * outputSize * sizeof(float), cudaMemcpyDeviceToHost));
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
    float result;
    CUDA_CHECK(cudaMemcpy(&result, cost, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cost));
    std::cout<<"Cost: "<<result<<"\n\n";
    
}

void NeuralNet::getAllDeltas(){
    float* deltaLPlusOne = startingDelta;
    float* temp;
    for(int i = layers.size() - 1; i > 0; i--){
        temp = layers[i].getDelta(deltaLPlusOne, layers[i - 1].getZ());
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
    for(int i = 0; i < 50; i++){
        forwardPass();
        backProp();
    }
}

NeuralNet::~NeuralNet(){
    //free all the GPU memory we allocated
    CUDA_CHECK(cudaFree(inputs));
    CUDA_CHECK(cudaFree(predictions));
    CUDA_CHECK(cudaFree(correctOutputs));
    CUDA_CHECK(cudaFree(startingDelta));
}

