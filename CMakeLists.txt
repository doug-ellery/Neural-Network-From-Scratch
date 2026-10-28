cmake_minimum_required(VERSION 3.24)

project(run LANGUAGES CXX CUDA)

# Find CUDA toolkit (for cublas, curand) & RMM
find_package(CUDAToolkit REQUIRED)
find_package(rmm REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# Define the executable
add_executable(run
    main.cpp
    DataProcessing.cpp
    NeuralNet.cu
    Layer.cu
    MyMatrix.cu 
)

# Link CUDA libraries & RMM
target_link_libraries(run
    PRIVATE
    CUDA::cublas
    CUDA::curand
    rmm::rmm
)
