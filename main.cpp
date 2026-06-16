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

    float rawInputs[] = {
        2.0f,  1.0f, -1.0f,
        1.0f, -2.0f,  3.0f
    };

    inputs.assign(rawInputs, rawInputs + inputSize * numSamples);

    float rawOutputs[] = {
        10.0f, -5.0f, 20.0f
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

    testNet.layers[0].setWeights({
        2.0f, -1.0f,
        1.0f,  3.0f
    });

    testNet.layers[1].setWeights({
        4.0f, -2.0f
    });

    testNet.layers[0].setBiases({1.0f, -2.0f});
    testNet.layers[1].setBiases({5.0f});

    std::vector<float> predictedOutputs = testNet.forwardPass();

    std::cout << "Predicted outputs:\n";
    for (int r = 0; r < outputSize; r++) {
        for (int c = 0; c < numSamples; c++) {
            std::cout << predictedOutputs[r * numSamples + c] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nExpected:\n";
    std::cout << "13 29 0\n";

    testNet.getCost();

    testNet.showAllDeltas();

    return 0;
}