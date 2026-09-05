# GPU 浅显入门

> 本章旨在为你介绍 GPU 的发展简史、为什么我们将 GPU 用于深度学习任务，以及为什么在特定任务上 GPU 比 CPU 显著更快。

## 硬件架构对比

![](assets/cpu.png)
- **CPU：中央处理器（Central Processing Unit）**
    - 通用计算
    - 高主频（Clock Speed）
    - 少量核心（Few Cores）
    - 大容量高速缓存（High Cache）
    - 低延迟（Low Latency）
    - 低吞吐量（Low Throughput）

![](assets/gpu.png)
- **GPU：图形处理器（Graphics Processing Unit）**
    - 专用计算
    - 相对较低的主频（Low Clock Speed）
    - 海量核心（Many Cores）
    - 相对较小的缓存（Low Cache）
    - 较高延迟（High Latency）
    - 极高吞吐量（High Throughput）

![](assets/tpu.png)
- **TPU：张量处理器（Tensor Processing Unit）**
    - 专为深度学习算法（如矩阵乘法等）定制的加速芯片

![](assets/fpga.png)
- **FPGA：现场可编程门阵列（Field Programmable Gate Array）**
    - 可根据特定任务重新配置电路结构的专用硬件
    - 极低延迟
    - 极高吞吐量
    - 较高的功耗
    - 昂贵的开发与硬件成本

## NVIDIA GPU 简史

> NVIDIA GPU 发展简史推荐视频 -> https://www.youtube.com/watch?v=kUqkOAU84bA

![](assets/history01.png)
![](assets/history02.png)
![](assets/history03.png)

## 为什么 GPU 在深度学习任务中如此迅猛？

![](assets/cpu-vs-gpu.png)

- **CPU（主机端 / Host）**
    - 核心目标：最小化单一任务的执行时间
    - 评估指标：以秒为单位的延迟（Latency）

- **GPU（设备端 / Device）**
    - 核心目标：最大化并行任务的总吞吐量
    - 评估指标：每秒处理的任务数（例如：每毫秒处理的像素数或 FLOPS）

## 典型 CUDA 程序的执行流程

1. CPU 分配主机端内存并初始化数据
2. CPU 将数据从主机显式拷贝至 GPU 显存
3. CPU 启动并调度 GPU 上的核函数（实际计算在此处高并发完成）
4. CPU 将计算结果从 GPU 拷贝回主机内存，以供后续业务使用

**直观理解核函数：**
核函数的单线程代码看起来就像普通的串行程序，代码本身不显式包含并行循环。想象你在拼一幅拼图，已知每一块拼图碎片的唯一位置。高层算法的目标是将这些零散碎片放置到位，对每块碎片而言只需解决一个简单问题：“把这块拼图放到它的正确位置”。只要最终所有拼图都落在了正确的位置，拼图就算成功完成！你不需要从一个角落按部就班地拼到另一个角落——只要互不干扰，你可以同时由成千上万双手（线程）拼放多块碎片。

## 需要牢记的核心术语

- **核函数 (Kernels)**：不是爆米花，不是卷积核，也不是 Linux 内核，而是指在 GPU 上并发运行的函数。
- **线程 (Threads)、线程块 (Blocks) 与网格 (Grid)**：（下一章将详细展开）。
- **GEMM**：通用矩阵乘法 (**GE**neral **M**atrix **M**ultiplication)。
- **SGEMM**：单精度浮点数 (FP32) 通用矩阵乘法 (**S**ingle-precision **GE**neral **M**atrix **M**ultiplication)。
- **CPU / Host / 函数 vs GPU / Device / 核函数**：
  - CPU 称为**主机端（Host）**，负责执行普通的 C/C++ 主机函数。
  - GPU 称为**设备端（Device）**，负责并发执行专门的 GPU 函数，即**核函数（Kernels）**。
