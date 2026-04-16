#include <iostream>
#include "MyMatrix.h"
#include "NeuralNet.h"
#include <chrono>
#include <vector>

int main(){
    NeuralNet testNet = NeuralNet(1, 4, 3, 1, 3, {2,3,4,7,8,9,10,11,12});
    vector<float> outputs = testNet.forwardPass();
    for(int r = 0; r < 1; r++){
        for(int c = 0; c < 3; c++){
            cout<<outputs[r + c];
        }
        cout<<"\n";
    }
    cout<<"\n\n"
    for(int i = 0; i < outputs.size(); i++){
        cout<<outputs[i];
    }
    
    return 0;

}