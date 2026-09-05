# 矩阵乘法性能优化指南 (Optimizing Matmul)

![](assets/comparison.png)

> - **Naive（朴素实现）**：最直观易懂，但性能极差。
> - **Coalesced Memory Access（合并访存）**：确保访存模式契合 GPU 硬件吞吐特性。
> - **Shared Memory（共享内存缓存）**：大幅减少对高延迟全局内存的访问，提升有效内存带宽。
> - **1D/2D Blocktiling（线程块分块）**：在网格中的所有 SM / 线程块之间均衡分配计算负载。
> - **Vectorized Memory Access（向量化内存访问）**：利用 `float4` 等向量指令，单条指令搬运 128 位数据而不是 32 位。
> - **Autotuning（自动调优）**：针对特定的 GPU 微架构网格搜索最优的分块参数。
> - **cuBLAS**：NVIDIA 官方闭源线性代数加速库（工业级性能基准）。

**本模块参考了 Simon Boehm 的经典深度博文：[博客文章](https://siboehm.com/articles/22/CUDA-MMM) 与 [GitHub 仓库](https://github.com/siboehm/SGEMM_CUDA)**

---

## 行优先 (Row-Major) vs 列优先 (Column-Major)

- cuBLAS 默认假定矩阵以列优先（Column-Major）格式排布，因此通常需要提前进行转置或合理调整步长。
- **行优先**：元素 `A[i][j]` 存储在一维偏移量 `A[i * N + j]`
- **列优先**：元素 `A[i][j]` 存储在一维偏移量 `A[j * M + i]`

```python
# 行优先 (Row-Major)
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

# 内存中的实际物理扁平排布
A = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 列优先 (Column-Major)
A = [[1, 4, 7],
     [2, 5, 8],
     [3, 6, 9]]

# 内存中的实际物理扁平排布
A = [1, 4, 7, 2, 5, 8, 3, 6, 9]
```

---

## `#pragma unroll` 循环展开的作用

- 目标：在每个循环迭代内部包含更多实质性的计算，减少循环计数器自增、边界条件比较及跳转带来的额外开销。
- 在很多情况下，现代 `nvcc` 编译器会在特定启发式规则下自动展开小循环，即使你没有显式添加 `#pragma unroll`。
- 可以使用命令查看生成的 PTX 汇编代码确认循环是否被成功展开：
  ```bash
  nvcc -ptx v1.cu -o - | less
  ```
- **基准测试与对比**：编写未展开与展开版本的核函数，通过计时对比二者的平均耗时，验证展开是否切实带来了性能收益。同时务必进行数值正确性校验（逐元素校验）。

---

## 什么是 GPU 占用率 (Occupancy)？

> **占用率（Occupancy）** 定义为每个流式多处理器（SM）上当前活跃的线程束（Active Warps）数量与该 SM 理论最大支持活跃线程束数量的比值。

限制单个 SM 驻留更多活跃 Block 的三大主要硬件瓶颈包括：
1. **寄存器数量（Register Count）**
2. **活跃线程束总数（Warp Count）**
3. **共享内存容量（Shared Memory Capacity）**

官方指南：[CUDA C 最佳实践指南：Occupancy 分析](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)  
性能调优参考：[深度学习矩阵乘法性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

---

## 汇编指令深入 (Assembly)

- [PTX 指令集参考 (Parallel Thread Execution)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ptx-machine-model)
- [如何阅读 GPU 着色器汇编代码 (SASS)](https://interplayoflight.wordpress.com/2021/04/18/how-to-read-shader-assembly/)

### 为什么需要探究或编写汇编级代码？
- 深入洞察当前的性能瓶颈所在（例如：线程束分化 Warp Divergence、等待数据加载至寄存器的延迟停顿、高开销指令等）。
- 进行时钟周期级别的微调（最贴近硬件物理极限的优化手段）。

## 致敬与灵感来源

1. [Simon Boehm (Anthropic 研究员)](https://siboehm.com/articles/22/CUDA-MMM)
2. [Lei Mao (NVIDIA 专家)](https://github.com/leimao/CUDA-GEMM-Optimization)

## 进阶探索

想要进一步理解顶尖工业界（如 NVIDIA cuBLAS）如何实现峰值 TFLOPS 的矩阵乘法，请学习 CUTLASS：
- [CUTLASS GitHub 仓库](https://github.com/NVIDIA/cutlass)
- [CUTLASS 官方技术博客](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [CUTLASS 官方文档](https://nvidia.github.io/cutlass/)
