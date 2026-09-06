#pragma once
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declaration of the C++/CUDA interface
torch::Tensor flash_attn_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool is_causal
);
