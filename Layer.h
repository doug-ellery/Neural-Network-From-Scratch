
#ifndef LAYER_H
#define LAYER_H

class Layer{
    float* weights, *biases;
    int n_in, n_out, samples;
    public:
        Layer(int, int, int);
        float* getNextLayer(float*);
        ~Layer();
};

#endif