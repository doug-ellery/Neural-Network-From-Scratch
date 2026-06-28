#include <iostream>
#include <vector>
#include <algorithm>
#include "NeuralNet.h"
#include "DataProcessing.h"
#include "MyMatrix.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
    std::vector<float> train_images = get_MNIST_Images("train-images-idx3-ubyte");
    std::vector<float> train_labels = get_MNIST_Labels("train-labels-idx1-ubyte");

    int numSamples = 60000;
    int inputSize = 784;
    int outputSize = 10;
    int numHiddenLayers = 3;
    int nodesPerHiddenLayer = 128;

    NeuralNet net(
        numHiddenLayers,
        nodesPerHiddenLayer,
        inputSize,
        outputSize,
        numSamples,
        train_images,
        train_labels,
        "RELU"
    );

    net.train();

    // Load test data and run accuracy check
    std::vector<float> test_images = get_MNIST_Images("t10k-images.idx3-ubyte");
    std::vector<float> test_labels = get_MNIST_Labels("t10k-labels.idx1-ubyte");

    int numTestSamples = 10000;
    std::vector<float> predictions = net.predict(test_images, numTestSamples);

    int correct = 0;
    for(int i = 0; i < numTestSamples; i++){
        auto pred_begin = predictions.begin() + i * outputSize;
        auto label_begin = test_labels.begin() + i * outputSize;
        int predicted = std::max_element(pred_begin, pred_begin + outputSize) - pred_begin;
        int actual    = std::max_element(label_begin, label_begin + outputSize) - label_begin;
        if(predicted == actual) correct++;
    }

    std::cout << "Test Accuracy: " << 100.0f * correct / numTestSamples << "%\n";

    return 0;
}
