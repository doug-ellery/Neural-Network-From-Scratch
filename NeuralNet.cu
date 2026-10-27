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


NeuralNet::NeuralNet(int numHiddenLayers, int nodesPerHiddenLayer, int inputSize, int outputSize, int numSamples, std::vector<float>& data, std::vector<float>& outputs, Activation_funcs activation_func){
    this->numSamples = numSamples;
    this->outputSize = outputSize;
    this->inputSize = inputSize;
    this->activation_func = activation_func;
    learning_rate = 0.001f;
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
    inputs = rmm::device_uvector<float>(inputSize*numSamples, rmm::cuda_stream_default);
    CUDA_CHECK(cudaMemcpy(inputs, data.data(), inputSize * numSamples * sizeof(float), cudaMemcpyHostToDevice));
    //get predictions onto gpu (itll start as empty because we haven't done a forward pass
    predictions = rmm::device_uvector<float>(outputSize*numSamples, rmm::cuda_stream_default);
    CUDA_CHECK(cudaMemset(predictions, 0, outputSize*numSamples*sizeof(float)));
    //get outputs onto GPU
    correctOutputs = rmm::device_uvector<float>(outputSize*numSamples, rmm::cuda_stream_default);
    CUDA_CHECK(cudaMemcpy(correctOutputs, outputs.data(), outputSize*numSamples*sizeof(float), cudaMemcpyHostToDevice));
    //get starting delta allocated
    startingDelta = rmm::device_uvector<float>(outputSize*numSamples, rmm::cuda_stream_default);
    CUDA_CHECK(cudaMemset(startingDelta, 0, outputSize*numSamples*sizeof(float)));
    //buffer for transposing each batch from (batch_size x inputSize) to (inputSize x batch_size)
    transposed_batch = rmm::device_uvector<float>(inputSize*batch_size, rmm::cuda_stream_default);
    //make host side inputs and outputs, without making copies because these could be very large
    h_inputs = std::move(data);
    h_outputs = std::move(outputs);
    shuffled_inputs.resize(h_inputs.size());
    shuffled_outputs.resize(h_outputs.size());
    h_inputs_indices.resize(numSamples);
    //fill h_inputs_indices with values 0 -> numSamples - 1, each index maps to one sample in h_inputs that will be shuffled later
    std::iota(h_inputs_indices.begin(), h_inputs_indices.end(), 0);
    //random seed for std::shuffle
    std::random_device rd;
    rng = std::mt19937{rd()};

}


void NeuralNet::forwardPass(const float* const batch_input){
    const float* prev = layers[0].getNextLayer(batch_input, batch_size);
    const float* curr = nullptr;
    for(int i = 1; i < layers.size(); i++){
        curr = layers[i].getNextLayer(prev, batch_size);
        prev = curr;
    }
    CUDA_CHECK(cudaMemcpy(predictions.data(), prev, batch_size*outputSize*sizeof(float)));
}

void NeuralNet::getStartingDelta(const float* const batch_output){
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, batch_size*outputSize);

    //call starting delta kernel
    startingDeltaKernel<<<numBlocks, threadsPerBlock>>>(predictions.data(), batch_output, startingDelta.data(), batch_size, outputSize);

}


void NeuralNet::getCost(const float* const batch_output){
    //init cost
    rmm::device_scalar<float> cost(0.0f, rmm::cuda_stream_default);
    
    //get threads per block and number of blocks
    int threadsPerBlock, numBlocks;
    getThreadsBlocks(threadsPerBlock, numBlocks, batch_size*outputSize);

    //call cost kernel
    costKernel<<<numBlocks, threadsPerBlock>>>(predictions.data(), batch_output, cost.data(), batch_size, outputSize);
    CUDA_CHECK(cudaMemcpy(&curr_cost, cost.data(), sizeof(float), cudaMemcpyDeviceToHost)); 
}

void NeuralNet::getAllDeltas(){
    const float* deltaLPlusOne = startingDelta.data();
    const float* temp = nullptr;
    for(int i = layers.size() - 1; i > 0; i--){
        if(activation_func == Activation_funcs::ReLU){
            temp = layers[i].getDelta(deltaLPlusOne, layers[i - 1].getZ(), batch_size);
        }
        else if(activation_func == Activation_funcs::Tanh){
            temp = layers[i].getDelta(deltaLPlusOne, layers[i - 1].getActivation(), batch_size);
        }
        else{
            std::cerr<<"Bad activation func"<<std::endl;
            exit(1);
        }
        deltaLPlusOne = temp;
    }
}


void NeuralNet::backProp(int t, const float* const batch_input, const float* const batch_output){
    getCost(batch_output);
    getStartingDelta(batch_output);
    getAllDeltas();
    const float* delta_l = startingDelta.data();
    const float* a_l_minus_one = layers.size() == 1 ? batch_input : layers[layers.size() - 2].getActivation();
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
    for(int epoch = 0; epoch < 30; epoch++){
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
        float epoch_cost = 0.0f;
        for(int batch = 0; batch < num_batches; batch++){
            float * batch_input = inputs + batch * batch_size * inputSize;
            float * batch_output = correctOutputs + batch * batch_size * outputSize;
            // MNIST data is sample-major (batch_size x inputSize); transpose to feature-major (inputSize x batch_size)
            transposeMatrix(batch_input, transposed_batch, batch_size, inputSize);
            forwardPass(transposed_batch);
            backProp(t, transposed_batch, batch_output);
            epoch_cost += curr_cost;
            t++;
        }
        if(epoch % 1 == 0){
            std::cout<<"Epoch "<<epoch<<" | Avg. Cost Over Mini Batches "<<epoch_cost / num_batches<<"\n";
        }
    }
}

//predict an output by running a forward pass
std::vector<float> NeuralNet::predict(std::vector<float> prediction_inputs, int num_predictions){
    float * raw_input = nullptr;
    CUDA_CHECK(cudaMalloc(&raw_input, prediction_inputs.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(raw_input, prediction_inputs.data(), prediction_inputs.size()*sizeof(float), cudaMemcpyHostToDevice));
    // Transpose from sample-major (num_predictions x inputSize) to feature-major (inputSize x num_predictions)
    float* transposed_pred = nullptr;
    CUDA_CHECK(cudaMalloc(&transposed_pred, inputSize * num_predictions * sizeof(float)));
    transposeMatrix(raw_input, transposed_pred, num_predictions, inputSize);
    CUDA_CHECK(cudaFree(raw_input));
    //same logic as forward pass
    float* prev = layers[0].getNextLayer(transposed_pred, num_predictions);
    float* curr = nullptr;
    for(int i = 1; i < layers.size(); i++){
        curr = layers[i].getNextLayer(prev, num_predictions);
        CUDA_CHECK(cudaFree(prev));
        prev = curr;
    }
    // Transpose result from feature-major (outputSize x num_predictions) to sample-major (num_predictions x outputSize)
    float* transposed_out = nullptr;
    CUDA_CHECK(cudaMalloc(&transposed_out, outputSize * num_predictions * sizeof(float)));
    transposeMatrix(prev, transposed_out, outputSize, num_predictions);
    std::vector<float> outVec(outputSize*num_predictions);
    CUDA_CHECK(cudaMemcpy(outVec.data(), transposed_out, outputSize*num_predictions*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(transposed_pred));
    CUDA_CHECK(cudaFree(prev));
    CUDA_CHECK(cudaFree(transposed_out));
    return outVec;
}

NeuralNet::~NeuralNet(){
    CUBLAS_CHECK(cublasDestroy(handle));
}

