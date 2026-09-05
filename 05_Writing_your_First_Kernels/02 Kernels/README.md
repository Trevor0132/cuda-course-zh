# 核函数深入 (Kernels)

## 核函数启动参数配置

- 类型 `dim3` 是一种三维结构体类型，用于配置 Grid 和 Block 的长宽高维度，传入核函数启动语法 `<<<...>>>` 中。
- 允许以一维向量、二维矩阵或三维张量（Volume）的形式为计算元素进行空间网格索引。

```cpp
dim3 gridDim(4, 4, 1);  // x 方向 4 个 block，y 方向 4 个 block，z 方向 1 个 block
dim3 blockDim(4, 2, 2); // 每个 block 内：x 方向 4 个线程，y 方向 2 个线程，z 方向 2 个线程
```

- 对于简单的一维情况，可以直接使用整型 `int` 进行配置：

```cpp
int gridDim = 16;  // 16 个 block
int blockDim = 32; // 每个 block 32 个线程
kernel<<<gridDim, blockDim>>>(...);
```

- `gridDim` ⇒ 总 Block 数量 = `gridDim.x * gridDim.y * gridDim.z`
- `blockDim` ⇒ 每个 Block 包含的线程数 = `blockDim.x * blockDim.y * blockDim.z`
- 总启动线程数 = `(每个 Block 线程数) * (Block 总数)`

- 启动语法 `<<<Dg, Db, Ns, S>>>` 的四个参数含义：
  - `Dg` (`dim3`)：指定 Grid 的维度和尺寸。
  - `Db` (`dim3`)：指定每个 Block 的维度和尺寸。
  - `Ns` (`size_t`)：指定每个 Block 额外动态分配的共享内存字节数（若无动态共享内存则可省略或填 0）。
  - `S` (`cudaStream_t`)：指定关联的 CUDA 流，可选参数，默认为 0（默认流 / 空流）。

> 详细讨论请参考 -> [StackOverflow: 理解 CUDA 核函数启动参数](https://stackoverflow.com/questions/26770123/understanding-this-cuda-kernels-launch-parameters)

## 线程同步机制 (Thread Synchronization)

- **`cudaDeviceSynchronize();`**：在主机端（CPU）调用，阻塞等待当前 GPU 设备上的所有任务执行完毕后才返回。它相当于主机与设备间的一道硬屏障（Barrier），确保前序核函数的所有计算完全落盘后，才安全启动下一个阶段。
- **`__syncthreads();`**：在**核函数内部**设置的块内同步屏障。同一个 Block 内的所有线程必须全部到达该行，才能继续向下执行。当不同线程需要协同读写同一块共享内存时至关重要。如果缺少同步，执行较快的线程可能会提前修改慢速线程仍需读取的内存数据，导致竞争冒险（Race Condition）与计算错误。
- **`__syncwarp();`**：仅在单个 Warp（32 个线程）内部执行锁步同步。

**为什么必须进行线程同步？**  
因为 GPU 上的各线程是完全异步、独立调度的，执行顺序无法预估。当不同线程间存在数据依赖时，必须显式设立同步屏障。例如在向量元素乘加操作中，必须确保所有乘法计算全部完成后，才能统一开始累加计算。

![](../assets/barrier.png)

## 线程安全性 (Thread Safety)

- 参考讨论：[CUDA 是否是线程安全的？](https://forums.developer.nvidia.com/t/is-cuda-thread-safe/2262/2)
- “线程安全”指多线程并发执行某段代码时，不会产生数据竞争（Race Conditions）或不符合预期的异常结果。
- 数据竞争常发生在多个线程同时向同一内存地址写入或读写顺序未受约束时。使用同步原语或硬件原子操作（Atomics）是确保线程安全的关键手段。

## SIMD 与 SIMT 架构

- 参考讨论：[CUDA 能否使用 SIMD 指令？](https://stackoverflow.com/questions/5238743/can-cuda-use-simd-extensions)
- 类似于 CPU 的 SIMD（单指令多数据流），GPU 采用更为灵活的 **SIMT（单指令多线程，Single Instruction, Multiple Threads）** 模型。
- 在 CPU 上通常用 `for` 循环依次串行迭代，而在 GPU 上可以将循环展开分配给海量线程同时执行——每一次循环迭代对应一个线程，在理想状态下使得整段循环耗时趋近于单次迭代的耗时。
- 相比 CPU，GPU 核心控制逻辑更简单：
  - 严格按序发射指令（In-order instruction issue）
  - 无分支预测器（No branch prediction）
  - 极大地精简了单个控制核心的晶体管开销，从而将硅片面积最大化留给**海量算术逻辑计算核心（ALU）**

> 在后续章节（矩阵乘法优化）中，我们将深入学习 Warp 级指令优化：[Warp 级原语使用指南](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

- 根据 [CUDA 官方线程层级说明](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)：由于同一个 Block 内的所有线程都必须驻留在同一个 SM 核心上共享硬件资源，目前架构中单个 Block 最大支持 **1024 个线程**（即每个 Block 最多容纳 32 个 Warp）。

## 硬件数学内置函数 (Math Intrinsics)

- GPU 硬件专用的底层数学指令 -> [CUDA Math API 参考手册](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
- 例如在核函数中调用标准主机库 `log()` 会比专用的设备端硬件内建指令 `logf()` 或 `__logf()` 慢很多。
- 向 `nvcc` 编译器传入 `-use_fast_math` 编译标志，编译器会自动将高耗时的数学函数替换为硬件原生快速单周期/低周期指令（以极其微小的精度折损换取成倍的吞吐加速）。
- 传入 `--fmad=true` 可以启用硬件乘加融合（Fused Multiply-Add, FMA），一条指令完成乘加且只舍入一次。
