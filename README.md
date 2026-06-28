# Neural Network From Scratch

A high-performance neural network implementation built from the ground up using **CUDA** and **C++**, leveraging GPU acceleration for efficient training and inference on the MNIST dataset.

## Overview

This project implements a fully-connected feed-forward neural network with support for customizable architecture, multiple activation functions, and GPU-accelerated computations. The network is trained on the MNIST handwritten digit dataset and achieves state-of-the-art accuracy through optimized CUDA kernels and cuBLAS operations.

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
- **CMake**: Version 3.18 or higher
- **C++ Compiler**: GCC, Clang, or MSVC with C++11 support
- **cuBLAS**: Included with CUDA Toolkit

### Build Instructions

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

## GPU Configuration

The project is optimized for NVIDIA GPUs with specific compute capabilities. The GPU architecture is specified in `CMakeLists.txt` at **line 9**:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

### GPU Architecture Numbers

The number (89 in this case) specifies the CUDA compute capability of your GPU:

- **89**: NVIDIA RTX 4080 / RTX 4090 (Ada architecture)
- **86**: NVIDIA RTX 3060 / RTX 3070 / RTX 3080 (Ampere architecture)
- **80**: NVIDIA RTX 2080 / A100 (Ampere architecture)
- **75**: NVIDIA RTX 2060 / RTX 2070 / RTX 2080 Ti (Turing architecture)
- **70**: NVIDIA Tesla P100 (Pascal architecture)

### Important: Update for Your GPU

If you're using a different GPU, **you must update line 9 in `CMakeLists.txt`** before building:

```cmake
# Change this number depending on your GPU
set(CMAKE_CUDA_ARCHITECTURES XX)  # Replace XX with your GPU's compute capability
```

To find your GPU's compute capability, you can:
- Check NVIDIA's [GPU Compute Capability Chart](https://developer.nvidia.com/cuda-gpus)
- Run `nvidia-smi` and look up your GPU model
- Use the CUDA sample `deviceQuery` utility

## Running the Project

### Prerequisites for Execution

You'll need the MNIST dataset files in your working directory:
- `train-images-idx3-ubyte` - Training images
- `train-labels-idx1-ubyte` - Training labels
- `t10k-images.idx3-ubyte` - Test images
- `t10k-labels.idx1-ubyte` - Test labels

This is already in the repo that you will clone.

### Execution

From the build directory, run:
```bash
./run
```

The program will:
1. Load MNIST training data (60,000 images)
2. Initialize a neural network with 3 hidden layers of 128 nodes each
3. Train the network on the training dataset
4. Load test data (10,000 images)
5. Evaluate accuracy on the test set
6. Output final test accuracy percentage

### Example Output
```
Test Accuracy: 97.45%
```

## Architecture Details

### Network Configuration

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
| Linear Algebra | cuBLAS |
| Build System | CMake 3.18+ |
| Language | C++ 11 / CUDA C |
| Breakdown | 73.9% CUDA, 25.1% C++, 1% CMake |

## Troubleshooting

### Build Issues

- **CUDA not found**: Ensure CUDA Toolkit is installed and in your PATH
- **CMake version error**: Update to CMake 3.18 or higher
- **GPU architecture mismatch**: Verify line 9 of CMakeLists.txt matches your GPU

### Runtime Issues

- **Missing MNIST files**: Ensure dataset files are in the working directory before running
- **Out of memory**: Check GPU memory availability; reduce `nodesPerHiddenLayer` if needed


