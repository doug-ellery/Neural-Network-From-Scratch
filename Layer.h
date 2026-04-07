

#ifndef LAYER_H
#define LAYER_H

class Layer(){
    float* weights, *biases;
    int n_in, n_out;
    public:
    Layer(int, int);

};

#endif