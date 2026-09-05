# 编写你的第一个 CUDA 核函数

> 官方文档起点 -> https://docs.nvidia.com/cuda/  
> 本课程主要参考 CUDA C 编程指南 -> https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html  
> 推荐阅读 NVIDIA 官方入门博客 -> https://developer.nvidia.com/blog/even-easier-introduction-cuda/  

- 通常的做法是：先在 CPU 上编写功能代码（易于调试验证），然后再移植到 GPU 上，确保你的算法逻辑能够映射到线程块（Block）和线程（Thread）层级。你可以设置一个输入向量 $x$，分别送入 CPU 函数和 GPU 核函数，检查输出是否一致。这能直接验证你的 GPU 代码是否按预期正确运行。

- 建议在纸上手动模拟向量加法和矩阵乘法的计算流程
- 深入理解线程（Threads）、线程块（Blocks）和网格（Grids）的核心概念

## 编译并运行向量加法核函数：

```bash
nvcc -o 01 01_vector_addition.cu
./01
```

## 硬件层级映射 (Hardware Mapping)

- **CUDA 核心 (Cores)**：负责执行具体的**线程 (Threads)**
- **流式多处理器 (SMs)**：负责调度管理**线程块 (Blocks)**（通常每个 SM 可以根据所需硬件资源同时驻留并调度多个 Block）
- **网格 (Grids)**：映射至**整个 GPU 设备**，是 CUDA 执行层级中的最高组织单位

## 内存层级模型 (Memory Model)

- **寄存器与局部内存 (Registers & Local Memory)**：每个线程独占，访问速度最快
- **共享内存 (Shared Memory)**：同一个线程块（Block）内的所有线程均可访问，用于线程间高效通信与数据协作
- **L2 高速缓存 (L2 Cache)**：作为计算核心/寄存器与全局显存之间的缓冲屏障，也是所有 SM 共享的片上缓存
- **L2 缓存与共享内存 / L1 缓存**：底层均采用 SRAM 电路实现，因此硬件周期速度非常接近。但 L2 缓存容量更大：
- **速度差异**：虽然同属 SRAM，但 L2 的访问延迟通常略高于 L1 / 共享内存。这并非制造技术差异所致，而是因为：
  - **容量**：L2 容量大得多，线路寻址周期更长。
  - **共享特性**：L2 由芯片上所有 SM 共享，需要更复杂的总线仲裁与访问调度机制。
  - **物理距离**：在硅片上，L2 距离计算核心通常比片内 L1 更加遥远。
- **全局显存 (Global Memory / VRAM)**：容量最大，存放从主机端（CPU）拷入或拷出的数据，GPU 上的所有线程和 SM 均可访问
- **主机内存 (Host)**：主板插槽上的 DDR 内存（16/32/64GB DRAM 等）
- **寄存器溢出 (Register Spilling)**：如果数组或局部变量过大超出寄存器容量，会溢出到局部内存（Local Memory，物理上位于高延迟的全局显存）。我们的性能优化目标之一就是尽量避免寄存器溢出，让程序全速运转。

![](assets/memhierarchy.png)

### 什么是随机存取内存（Random Access Memory）？

- 在磁带或录像带中，你必须按顺序快进快退遍历前面所有数据才能到达尾部内容；而“随机”指的是能够**以几乎恒定的时间即时访问任意给定索引地址的数据**（无需依次扫描前面的数据）。在软件抽象上，显存看起来像是一条连续的一维线性空间，但在芯片硅片内部实际上是呈网格状矩阵阵列排布的（由底层硬件电路完成行/列选通与映射）。

![](assets/memmodel.png)

> 延伸阅读：[NVIDIA 官方博客：高效矩阵转置与访存合并](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
