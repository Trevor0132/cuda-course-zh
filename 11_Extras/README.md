# 进阶与补充知识 (Extras)

## CUDA 编译器 (NVCC)
![](assets/nvcc.png)

## CUDA 如何处理条件分支 (if/else)？
- CUDA 硬件对分支逻辑的处理效率较低。如果在同一个 Warp 内部存在条件分支，且不同线程走入不同分支，硬件将不得不**串行执行每个分支路径**（使用谓词寄存器 Predicate 掩码来选择有效输出）。这意味着当某个分支在执行时，另一个分支的线程只能处于空闲等待状态。这种现象称为**线程束分化（Warp Divergence）**。
- 如果条件分支较为复杂，这种串行化会极大地拉低计算吞吐。因此在编写 CUDA 核函数时，应尽量消除分支判断，或确保同一个 Warp（32 个线程）内的所有线程总是走向相同的分支路径。
- 若必须使用分支，可以通过查看 PTX 汇编（`nvcc -ptx kernel.cu -o kernel`）来观察编译器生成的谓词指令，针对性调优指令级效率。
- 向量加法之所以极快，是因为全部线程执行完全相同的数学计算，不存在任何 Warp 分化。

## 统一内存 (Unified Memory) 的优缺点
- **统一内存（Unified Memory）** 是 CUDA 的一项核心特性，它允许分配由 CPU（主机内存）与 GPU（显存）共享访问的单一内存空间（通过 `cudaMallocManaged`）。开发者无需再显式编写 `cudaMemcpy` 在两端手动搬运数据，极大降低了心智负担。
- [对比：CUDA 显式内存 vs 统一内存实测](https://github.com/lintenn/cudaAddVectors-explicit-vs-unified-memory)
- [NVIDIA 博客：最大化统一内存性能](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)
- 统一内存可在后台借助 CUDA 流自动实现**页预取（Prefetching）**：
  - [CUDA 流详解 - Lei Mao](https://leimao.github.io/blog/CUDA-Stream/)
  - [NVIDIA 官方并发执行文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
  - 通过流机制，数据拷贝与核函数计算可以无缝重叠执行。当一个流在执行核函数时，另一个流可以在后台为下一个批次提前搬运数据。这种技术即常说的“双缓冲（Double Buffering）”或“多缓冲”。

![](assets/async.png)

## 硬件存储单元架构微探 (Memory Architectures)
- **DRAM / VRAM 单元**：计算机主存与显存的最小物理存储单元，由微型电容与晶体管组成。电容以电荷形式存储比特，晶体管负责读写控制门（需要周期性刷新）。
![](assets/dram-cell.png)
- **SRAM 单元（共享内存 / 片上缓存）**：比 DRAM 快得多但也更昂贵、占用更大芯片面积。CPU 与 GPU 内部的 L1/L2 高速缓存及 Shared Memory 均采用 SRAM。现代 NVIDIA GPU 片上 SRAM 通常采用 6T（六晶体管）或 8T（八晶体管）结构。
![](assets/sram-cell.png)
![](assets/8t-sram-cell.png)

## 进阶探索与前沿领域
- **模型量化 (Quantization)**：FP32 -> FP16 -> INT8 / FP8 / FP4
- **张量核心 (Tensor Cores / WMMA)**：专用于矩阵乘加的硬件加速单元
- **稀疏计算 (Sparsity)**：利用结构化稀疏性（如 2:4 稀疏）使吞吐翻倍
- 经典学习资源：
  - [《CUDA by Example》电子书 (PDF)](https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf)
  - [深度学习模型的数据并行分布式训练解析 - Simon Boehm](https://siboehm.com/articles/22/data-parallel-training)
  - [mnist-cudnn 源码实现](https://github.com/haanjack/mnist-cudnn)
  - [CUDA MODE 讲座合集](https://github.com/cuda-mode/lectures)
  - [micrograd-cuda 纯 C/CUDA 实现](https://github.com/mlecauchois/micrograd-cuda)
  - [Karpathy 的 micrograd 原版](https://github.com/karpathy/micrograd)
  - [Sasha Rush 的 GPU Puzzles 闯关解密](https://github.com/srush/GPU-Puzzles)