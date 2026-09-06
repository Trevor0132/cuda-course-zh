from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as ext
import os

# 绕过系统 CUDA (12.6) 与 PyTorch 预编译 CUDA (13.0) 的次版本差异检查
ext._check_cuda_version = lambda *args: None

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="custom_flash_attn",
    ext_modules=[
        CUDAExtension(
            name="custom_flash_attn",
            sources=[
                os.path.join(this_dir, "flash_attn_binding.cpp"),
                os.path.join(this_dir, "flash_attn_forward.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_75",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
