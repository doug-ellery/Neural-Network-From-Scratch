#include <iostream>
#include "MyMatrix.h"
#include "NeuralNet.h"
#include <vector>

int main() {
    int numSamples = 3;
    int inputSize = 2;
    int numHiddenLayers = 1;
    int nodesPerLayer = 2;
    int outputSize = 1;

    std::vector<float> inputs;
    std::vector<float> outputs;

    //Small, normalized inputs
    // Each column = one sample
    // Values in range ~[-1, 1]
    float rawInputs[] = {
        1.0f, 3.0f, 5.0f,
        2.0f, 4.0f, 6.0f
    };

    inputs.assign(rawInputs, rawInputs + inputSize * numSamples);

    //Simple target outputs (also small scale)
    float rawOutputs[] = {
        18.0f, 38.0f, 58.0f
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
    testNet.layers[0].setWeights({1.0f, 2.0f, 3.0f, 4.0f});
    testNet.layers[1].setWeights({1.0f, 1.0f});
    testNet.layers[0].setBiases({1,-1});
    testNet.layers[1].setBiases({2});

    std::vector<float> predictedOutputs = testNet.forwardPass();

    std::cout << "Predicted outputs:\n";
    for (int r = 0; r < outputSize; r++) {
        for (int c = 0; c < numSamples; c++) {
            std::cout << predictedOutputs[r * numSamples + c] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nExpected:\n18 38 58\n";

    testNet.getCost();

    return 0;
}