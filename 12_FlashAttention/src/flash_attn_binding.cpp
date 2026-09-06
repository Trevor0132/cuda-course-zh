#include <torch/extension.h>
#include "flash_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", &flash_attn_forward, "FlashAttention Forward Pass (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("is_causal") = false);
}
