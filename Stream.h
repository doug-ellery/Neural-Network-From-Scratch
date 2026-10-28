#pragma once
#include <cuda/stream_ref>
#include <cuda_runtime.h>

//constexpr for the default cuda stream I use for all of my uvectors
inline constexpr cuda::stream_ref default_strm{cudaStream_t{nullptr}};