@echo off
nvcc main.cpp NeuralNet.cu Layer.cu MyMatrix.cu -lcublas -o run.exe