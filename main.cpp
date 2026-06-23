#include <iostream>
#include <vector>
#include "NeuralNet.h"
#include "MyMatrix.h"

int main() {
    // -----------------------------
    // Network config
    // -----------------------------
    int inputSize = 1;
    int numHiddenLayers = 3;
    int nodesPerLayer = 64;
    int outputSize = 1;

    // -----------------------------
    // Normalization constants
    // -----------------------------
    const float xMin = -10.0f, xMax = 10.0f;
    const float yMin =   0.0f, yMax = 100.0f;

    auto normalizeX = [&](float x) {
        return x / xMax; // maps [-10, 10] -> [-1, 1]
    };
    auto normalizeY = [&](float y) {
        return (y - yMin) / (yMax - yMin); // maps [0, 100] -> [0, 1]
    };
    auto denormalizeY = [&](float y) {
        return y * (yMax - yMin) + yMin;   // maps [0, 1] -> [0, 100]
    };

    // -----------------------------
    // Training data: f(x) = x^2, normalized
    // -----------------------------
    std::vector<float> inputs;
    std::vector<float> outputs;

    for (float x = -10.0f; x <= 10.0f; x += 0.5f) {
        inputs.push_back(normalizeX(x));
        outputs.push_back(normalizeY(x * x));
    }
    int numSamples = inputs.size();

    NeuralNet net(
        numHiddenLayers,
        nodesPerLayer,
        inputSize,
        outputSize,
        numSamples,
        inputs,
        outputs,
        "TANH"
    );

    // -----------------------------
    // Train
    // -----------------------------
    net.train();

    // -----------------------------
    // Test (denormalize predictions)
    // -----------------------------
    std::vector<float> testInputs = {
        -12.0f, -10.0f, -7.0f, -3.5f,
         -2.0f,  -1.0f,  0.0f,
          1.0f,   2.0f,  3.5f,
          7.0f,  10.0f, 12.0f
    };

    std::cout << "\n--- Predictions (x^2) ---\n";

    for (float x : testInputs) {
        float expected = x * x;

        float xNorm = normalizeX(x);
        std::vector<float> pred = net.predict({xNorm});

        float predicted = denormalizeY(pred[0]);

        std::cout << "x = " << x
                  << " | predicted = " << predicted
                  << " | expected = " << expected
                  << "\n";
    }

    return 0;
}