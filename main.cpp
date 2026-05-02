#include <iostream>
#include "MyMatrix.h"
#include "NeuralNet.h"
#include <vector>

int main() {
    int numSamples = 5;
    int inputSize = 4;
    int numHiddenLayers = 2;
    int nodesPerLayer = 8;
    int outputSize = 3;

    std::vector<float> inputs;
    std::vector<float> outputs;

    //Small, normalized inputs
    // Each column = one sample
    // Values in range ~[-1, 1]
    float rawInputs[] = {
        0.1f,  0.2f,  0.3f,  0.4f,  0.5f,
       -0.2f, -0.1f,  0.0f,  0.1f,  0.2f,
        0.5f,  0.4f,  0.3f,  0.2f,  0.1f,
       -0.5f, -0.4f, -0.3f, -0.2f, -0.1f
    };

    inputs.assign(rawInputs, rawInputs + inputSize * numSamples);

    //Simple target outputs (also small scale)
    float rawOutputs[] = {
        0.0f, 0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.4f, 0.3f, 0.2f, 0.1f,
        0.2f, 0.2f, 0.2f, 0.2f, 0.2f
    };

    outputs.assign(rawOutputs, rawOutputs + outputSize * numSamples);

    NeuralNet testNet(
        numHiddenLayers,
        nodesPerLayer,
        inputSize,
        outputSize,
        numSamples,
        inputs,
        outputs
    );

    std::vector<float> predictedOutputs = testNet.forwardPass();

    std::cout << "Predicted outputs:\n";
    for (int r = 0; r < outputSize; r++) {
        for (int c = 0; c < numSamples; c++) {
            std::cout << predictedOutputs[r * numSamples + c] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nCorrect outputs:\n";
    for (int r = 0; r < outputSize; r++) {
        for (int c = 0; c < numSamples; c++) {
            std::cout << outputs[r * numSamples + c] << " ";
        }
        std::cout << "\n";
    }

    testNet.getCost();

    return 0;
}