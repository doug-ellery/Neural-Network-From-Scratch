#include <iostream>
#include "MyMatrix.h"
#include "NeuralNet.h"
#include <chrono>
#include <vector>

int main(){
    std::vector<float> inputs = {2.0,3.0,4.0,7.0,8.0,9.0,10.0,11.0,12.0};
    NeuralNet testNet(1, 4, 3, 1, 3, inputs);
    std::vector<float> outputs = testNet.forwardPass();
    for(int r = 0; r < 1; r++){
        for(int c = 0; c < 3; c++){
            std::cout<<outputs[r + c];
        }
        std::cout<<"\n";
    }
    std::cout << "\n\n";
    for(int i = 0; i < outputs.size(); i++){
        std::cout<<outputs[i];
    }
    
    return 0;

}