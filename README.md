# Neural Network From Scratch

A high-performance neural network implementation built from the ground up using **CUDA** and **C++**, leveraging GPU acceleration for efficient training and inference on the MNIST dataset.

## Overview

This project implements a fully-connected feed-forward neural network with support for customizable architecture, multiple activation functions, and GPU-accelerated computations. The network is trained on the MNIST handwritten digit dataset and achieves 97-98% accuracy consistently through optimized CUDA kernels and cuBLAS operations.

### Key Features

- **GPU-Accelerated Computation**: Leverages NVIDIA CUDA for matrix operations and neural network computations
- **Customizable Architecture**: Easily configure number of hidden layers, nodes per layer, and activation functions
- **Efficient Linear Algebra**: Uses cuBLAS library for high-performance matrix operations
- **MNIST Dataset Support**: Built-in support for loading and processing MNIST training and test data
- **Training & Inference**: Complete pipeline for training the network and evaluating test accuracy
- **RELU Activation**: Implemented with optimized CUDA kernels

### Project Structure

```
├── main.cpp              # Entry point; loads MNIST data and runs training/testing
├── NeuralNet.cu          # Core neural network implementation (forward pass, backprop)
├── Layer.cu              # Individual layer computations and activations
├── MyMatrix.cu           # Custom matrix math with CUDA kernels
├── DataProcessing.cpp    # MNIST dataset loading and preprocessing
├── CMakeLists.txt        # Build configuration with CUDA setup
└── *.h                   # Corresponding header files
```

## Building the Project

### Prerequisites

- **CUDA Toolkit**: Version 11.0 or higher
- **CMake**: Version 3.24 or higher
- **C++ Compiler**: GCC, Clang, or MSVC with C++11 support
- **cuBLAS**: Included with CUDA Toolkit

### Build/Run Instructions

**IMPORTANT NOTICE FOR WINDOWS USER: Before you try and build, you need to run the devshell.bat script provided for you in the repo so that nvcc can find MSVC, to ensure devshell works, make sure you have Visual Studio + the desketop developement with C++ workload insatlled**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/doug-ellery/Neural-Network-From-Scratch.git
   cd Neural-Network-From-Scratch
   ```

2. **Configure with CMake**:
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
   ```

3. **Build the project**:
   ```bash
   cmake --build build
   ```

4. **Run the executable**:
   Windows:
   ```bash
   .\build\run.exe
   ```
   Linux:
   ```bash
   ./build/run
   ```
   Do not try to run from within build because then the program wont be able to open the MNIST files


### Prerequisites for Execution

You'll need the MNIST dataset files in your working directory:
- `train-images-idx3-ubyte` - Training images
- `train-labels-idx1-ubyte` - Training labels
- `t10k-images.idx3-ubyte` - Test images
- `t10k-labels.idx1-ubyte` - Test labels

This is already in the repo that you will clone.

### Execution

The program will:
1. Load MNIST training data (60,000 images)
2. Initialize a neural network with 3 hidden layers of 128 nodes each
3. Train the network on the training dataset
4. Load test data (10,000 images)
5. Evaluate accuracy on the test set
6. Output final test accuracy percentage

### Example Output
```
Epoch 0 | Avg. Cost Over Mini Batches 0.246144
Epoch 1 | Avg. Cost Over Mini Batches 0.104006
Epoch 2 | Avg. Cost Over Mini Batches 0.0701897
...
Epoch 29 | Avg. Cost Over Mini Batches 0.012921
Test Accuracy: 97.95%
```


### NN Configuration

Default configuration (customizable in `main.cpp`):
- **Input Layer**: 784 neurons (28×28 pixel MNIST images)
- **Hidden Layers**: 3 layers with 128 nodes each
- **Output Layer**: 10 neurons (digits 0-9)
- **Activation Function**: RELU
- **Training Samples**: 60,000
- **Test Samples**: 10,000

### GPU Memory Usage

The implementation efficiently manages GPU memory through custom matrix operations, with memory allocated for:
- Network weights and biases
- Intermediate activation values
- Gradient computations during backpropagation
- cuBLAS workspace

## Implementation Highlights

- **Custom CUDA Kernels**: Optimized kernels for activation functions (RELU, tanh, soft-max) and layer operations
- **cuBLAS Integration**: High-performance matrix multiplication via NVIDIA's optimized BLAS library
- **GPU Memory Management**: Efficient allocation and deallocation of GPU memory
- **Batch Processing**: Supports batch training for improved GPU utilization

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Computation | CUDA 11.0+ |
| Build System | CMake 3.24+ |
| Language | C++ 11 / CUDA C |
| Breakdown | 73.9% CUDA, 25.1% C++, 1% CMake |



