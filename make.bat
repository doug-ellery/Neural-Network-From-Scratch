@echo off
nvcc main.cpp DataProcessing.cpp NeuralNet.cu Layer.cu MyMatrix.cu -lcublas -lcurand -o run.exe