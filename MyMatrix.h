#include <vector>
#include <cuda_fp16.h>

#ifndef MYMATRIX_H
#define MYMATRIX_H

 
std::vector<float> cudaMultiply(std::vector<half>&, std::vector<half>&, int, int, int);
void cudaSigmoid(std::vector<float>&);
void cudaAdd(std::vector<float>&, std::vector<float>&, std::vector<float>&, int, int)


#endif