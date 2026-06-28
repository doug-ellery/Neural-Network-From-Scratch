cmake_minimum_required(VERSION 3.18)

# Find CUDA toolkit (for cublas, curand)
find_package(CUDAToolkit REQUIRED)

project(run LANGUAGES CXX CUDA)

# Change this number depending on your GPU, this is for 4080
set(CMAKE_CUDA_ARCHITECTURES 89)

# Define the executable
add_executable(run
    main.cpp
    DataProcessing.cpp
    NeuralNet.cu
    Layer.cu
    MyMatrix.cu
)

# Link CUDA libraries
target_link_libraries(run
    PRIVATE
    CUDA::cublas
    CUDA::curand
)