#include <vector>
#include <cuda_fp16.h>

#ifndef MYMATRIX_H
#define MYMATRIX_H

class MyMatrix{
    public:    
        static std::vector<float> cudaMultiply(std::vector<half>& A, std::vector<half>& B, int M, int N, int K);
};

#endif