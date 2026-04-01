#include <iostream>
#include <cuda_fp16.h>
#include "MyMatrix.h"

int main(){
    //testing matrix mutliplication A = M x K, B = K x N
    int M = 20;
    int K = 40;
    int N = 60;
    std::vector<half> A, B;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            A.push_back(__float2half(static_cast<float>(rand()) / RAND_MAX));
        }
    }
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            B.push_back(__float2half(static_cast<float>(rand()) / RAND_MAX));
        }
    }
    std::vector<float> correctAnswer(M*N,0.0f);
    cout<<"Starting non parallel multiplication...";
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                correctAnswer[i*N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSecondsNonParallel = end1 - start1;
    for(int r = 0; r < M; r++){
        for(int c = 0; c < N; c++){
            cout<<correctAnswer[r*N + c]<<" ";
        }
        cout<<"\n";
    }
    cout<<"\n\nTime for multiplication without gpu: "<<elapsedSecondsNonParallel.count()<<"\n\n";
    

    auto start2 = std::chrono::high_resolution_clock::now();
    std::vector<float> C = MyMatrix::cudaMultiply(A, B, M, N, K);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSecondsParallel = end1 - start1;
    for(int r = 0; r < M; r++){
        for(int c = 0; c < N; c++){
            cout<<C[r*N + c]<<" ";
        }
        cout<<"\n";
    }
    cout<<"\n\nTime for multiplication without gpu: "<<elapsedSecondsParallel.count()<<"\n\n";
    



}