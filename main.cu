#include <iostream>
#include <cuda_fp16.h>
#include "MyMatrix.h"
#include <chrono>
#include <vector>

int main(){
    //testing matrix mutliplication A = M x K, B = K x N
    int M = 2000;
    int K = 4000;
    int N = 6000;
    std::vector<half> A(M*K), B(K*N);
    for(int i = 0; i < A.size(); i++){
        A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    for(int i = 0; i < B.size(); i++){
        B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    std::vector<float> correctAnswer(M*N,0.0f);
    std::cout<<"Starting non parallel multiplication...\n\n";
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                correctAnswer[i*N + j] += __half2float(A[i*K + k]) * __half2float(B[k*N + j]);
            }
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSecondsNonParallel = end1 - start1;
    //for(int r = 0; r < M; r++){
    //    for(int c = 0; c < N; c++){
    //        std::cout<<correctAnswer[r*N + c]<<" ";
     //   }
      //  std::cout<<"\n";
    //}
    std::cout<<"\n\nTime for multiplication without gpu: "<<elapsedSecondsNonParallel.count()<<"\n\n";
    
    std::cout<<"Starting parallel multiplication...\n\n";

    auto start2 = std::chrono::high_resolution_clock::now();
    std::vector<float> C = MyMatrix::cudaMultiply(A, B, M, N, K);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSecondsParallel = end2 - start2;
    //for(int r = 0; r < M; r++){
     //   for(int c = 0; c < N; c++){
       //     std::cout<<C[r*N + c]<<" ";
        //}
       // std::cout<<"\n";
    //}
    std::cout<<"\n\nTime for multiplication with gpu: "<<elapsedSecondsParallel.count()<<"\n\n";
    return 0;


}