# CUTLASS 深度解析

- 官方博客介绍：[CUDA 线性代数子程序与求解器模板库 (CUTLASS)](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- CUTLASS 最广泛的应用场景是通用矩阵乘法（GEMM）。在 Transformer 架构中，矩阵乘法是算力消耗的核心，因此 `cutlass/gemm` 模块尤为关键。
- CUTLASS 依赖大量的 C++ 高级模板元编程，具有较高的复杂度与学习门槛。它专为高性能计算（HPC）工程师设计，用于针对特定的硬件架构（如 Tensor Core、异步数据搬运指令 TMA 等）实现极致的定制与微架构算子融合。
- 在示例测试脚本中，对 $1024 \times 1024 \times 1024$ 规模的矩阵乘法在 cuBLAS 与 CUTLASS 之间进行了对比（先执行 10 次预热，再统计实际耗时）：

```bash
cuBLAS 耗时: 0.202861 ms
CUTLASS 耗时: 0.227451 ms
```

- **编译说明**：使用 CUTLASS 时，需要通过 `-I` 标志指定 CUTLASS 头文件所在路径。建议在 `~/.bashrc` 或 `~/.zshrc` 中导出相应环境变量以便日常编译。
