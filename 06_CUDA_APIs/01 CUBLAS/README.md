> **注意**：在进行性能测试之前，先执行预热轮次（Warmup Runs）与基准测试轮次（Benchmark Runs）对于获得真实的执行时间至关重要。如果不进行预热，cuBLAS 在首次运行时会产生较大的初始化与上下文加载开销（约 45ms 延迟），从而严重扭曲测量结果。

# cuBLAS 基础与变体

- **NVIDIA cuBLAS（CUDA 基础线性代数子程序库）** 是一套 GPU 加速的工业级标准 BLAS 库，广泛用于加速 AI 和 HPC 任务。它提供了即插即用的标准 BLAS API，以及深度针对 NVIDIA GPU 优化的通用矩阵乘法（GEMM）接口与算子融合支持。
<br>

- 务必注意数据排布格式（Row-major 行优先 vs Column-major 列优先）：[cuBLAS SGEMM 行优先乘法排坑指南](https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication)。

## cuBLASLt (Lightweight)
- **cuBLASLt（轻量级 cuBLAS）** 是 cuBLAS 库的高性能扩展，提供了更加灵活的 API 设计，专为深度学习负载进行优化。绝大多数数据类型和 API 均围绕 GEMM 操作展开。
- 当一个庞大的计算任务无法在单个核函数中直接跑满时，cuBLASLt 能够将其自动拆解为多个子任务并在各自核函数中并发求解。
- 原生支持低精度加速：FP16、FP8、INT8 等张量核心（Tensor Core）计算。

## cuBLASXt (Multi-GPU / Extended)
- 支持多 GPU 分布式运算与 Host-GPU 混合计算（如果跨 PCIe/NVLink 通信繁重，会受到传输带宽瓶颈制约）。
- **核心特性**：
  - **多 GPU 支持**：将超大尺寸 BLAS 操作自动分发到多个 GPU 上并行计算，实现算力扩展并支持超出单卡显存容量的超大规模数据集。
  - **线程安全**：原生支持多线程并发向不同 GPU 发射任务。
- 适用场景：当线性代数矩阵规模超出单张 GPU 显存容量上限时使用。

- **cuBLAS 与 cuBLAS-Xt 性能对比**：
  - 计算矩阵规模：$(M, N) \times (N, K)$，其中 $M = N = K = 16384$
  - ![](../assets/cublas-vs-cublasxt.png)

## cuBLASDx (Device-side)

> **强调**：本课程**不使用** cuBLASDx。

- cuBLASDx 是一款在 GPU 设备端（Device-side）核函数内部直接执行 BLAS 运算的预览版扩展。它允许在 CUDA 核函数内部直接调用 BLAS 计算并与其它数学逻辑融合，消除多次访存。
- 文档请参考：[cuBLASDx 官方文档](https://docs.nvidia.com/cuda/cublasdx)。
- 该库需单独从官网下载：[cuBLASDx 下载中心](https://developer.nvidia.com/cublasdx-downloads)。

## CUTLASS

- cuBLAS 及其变体通常在 Host 端发起调用；而在深度学习中，矩阵乘法往往需要与前后激活函数、偏置加法进行算子融合（Operator Fusion）以避免显存往返。
- [CUTLASS](https://github.com/NVIDIA/cutlass)（CUDA 线性代数模板库）通过 C++ 模板元编程，允许开发者自由组合与深度融合各类矩阵乘法流水线与算子。
- *注：FlashAttention 并没有直接使用 CUTLASS，而是由作者手写的深度调优原生 CUDA 核函数。*
![](../assets/flashattn.png) -> 来源：https://arxiv.org/pdf/2205.14135