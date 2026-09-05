# CUDA 基础概念

## 打印 GPU 硬件配置信息
![](../assets/gpustats.png)

## 基础术语
- **主机端 (Host)** ⇒ CPU ⇒ 使用主板上的物理内存条（RAM / DRAM）
- **设备端 (Device)** ⇒ GPU ⇒ 使用显卡板载的独立显存（VRAM）

CUDA 程序的宏观执行流程：
1. 将输入数据从主机端（CPU 内存）拷贝到设备端（GPU 显存）
2. 在 GPU 上加载并并发执行核函数，直接操作显存中的数据
3. 将计算结果从设备端拷贝回主机端，以便后续展示或进一步处理

## 主机端与设备端命名约定
- `h_A`：表示位于**主机端（Host / CPU）**的变量 “A”
- `d_A`：表示位于**设备端（Device / GPU）**的变量 “A”

## 函数执行空间限定符
- `__global__`：声明一个核函数（Kernel）。在全局可见，由 CPU（主机端）发起调用，但在 GPU（设备端）上并发执行。核函数通常返回 `void`，其计算结果通常直接写入作为指针传入的输出显存缓冲区。例如矩阵乘法 $A \times B$，我们将预先分配好的显存缓冲区指针 $C$ 传入核函数，并在核函数中将计算结果写回 $C$。
- `__device__`：声明一个设备函数。只能由 GPU 上的其他 `__global__` 或 `__device__` 函数调用，并在 GPU 上执行。例如，当你在 `__global__` 核函数中计算注意力机制的分数矩阵时，需要对矩阵进行标量掩码（Masking）或 Softmax，此时可以提取出一个专用的 `__device__` 函数来处理单元素计算，类似于普通编程中调用子函数。
- `__host__`：普通主机函数，只在 CPU 上编译和执行。如果不写限定符，默认就是 `__host__`。`__host__` 和 `__device__` 可以组合使用，让同一函数同时编译出 CPU 与 GPU 两个版本。

## 显存管理核心 API

- `cudaMalloc`：仅在 GPU 显存（全局内存 / Global Memory）中分配空间：

```cpp
float *d_a, *d_b, *d_c;

cudaMalloc(&d_a, N * N * sizeof(float));
cudaMalloc(&d_b, N * N * sizeof(float));
cudaMalloc(&d_c, N * N * sizeof(float));
```

- `cudaMemcpy`：在内存空间之间传输数据：
  - 主机端到设备端 ⇒ CPU 到 GPU (`cudaMemcpyHostToDevice`)
  - 设备端到主机端 ⇒ GPU 到 CPU (`cudaMemcpyDeviceToHost`)
  - 设备端到设备端 ⇒ GPU 显存不同地址之间 (`cudaMemcpyDeviceToDevice`)
- `cudaFree`：释放由 `cudaMalloc` 分配的 GPU 显存

# `nvcc` 编译器工作机制
- **主机端代码**：经由编译器分离并编译为标准的 x86/ARM 主机机器码。
- **设备端代码**：编译为 PTX（并行线程执行，Parallel Thread Execution）虚拟汇编指令集。
- PTX 在不同的 GPU 微架构世代间保持向后兼容。
- **JIT（即时编译）**：驱动程序在运行时可将通用的 PTX 编译为目标 GPU 原生的机器码（SASS），确保跨代向前兼容。

## CUDA 组织层级 (CUDA Hierarchy)
1. 核函数（Kernel）由具体的**线程 (Thread)** 并发执行
2. 多个线程组织为**线程块 (Thread Block，简称 Block)**
3. 多个线程块组织为一个**网格 (Grid)**
4. 启动一个核函数，即启动了一个由多个 Block 组成的 Grid

### 4 个核心内置变量：
- `gridDim` ⇒ 网格的尺寸（包含的 Block 数量）
- `blockIdx` ⇒ 当前 Block 在网格中的三维索引坐标
- `blockDim` ⇒ 线程块的尺寸（每个 Block 中包含的 Thread 数量）
- `threadIdx` ⇒ 当前 Thread 在所属 Block 中的三维索引坐标

## 线程 (Threads)
- 每个线程拥有自己私有的局部内存与寄存器（Registers）。
- 示例：若要计算向量加法 $a = [1, 2, 3, \dots, N]$ 与 $b = [2, 4, 6, \dots, N]$，每个线程独立计算单个元素：线程 0 计算 $a[0] + b[0]$，线程 1 计算 $a[1] + b[1]$，依此类推。

## 线程束 (Warps)
![](../assets/weft.png)
- [Warp 与 Weft 纺织术语维基](https://en.wikipedia.org/wiki/Warp_and_weft)：经线（Warp）是在织布机上平行张紧的基准纱线。
- 在 CUDA 中，**Warp（线程束）** 是 SM 调度和执行的基本硬件单元，固定由 **32 个连续线程** 组成。
- GPU 的指令是按 Warp 为单位发射的，同一个 Warp 内的 32 个线程以锁步（Lock-step / SIMT）方式执行相同的指令。
- 硬件底层调度必然围绕 Warp 展开，无法绕过。
- 每个 SM 通常拥有 4 个独立的 Warp 调度器（Warp Schedulers）。
![](../assets/schedulers.png)

## 线程块 (Blocks)
- 每个 Block 内部拥有一块高速的**共享内存 (Shared Memory)**，同一 Block 内的所有线程均可读写该共享内存。
- 同一 Block 内的线程可以执行高效的数据协同与线程同步（`__syncthreads()`）。

## 网格 (Grids)
- 核函数执行期间，Grid 内所有 Block 的所有线程均可访问底层的全局显存（Global Memory）。
- Grid 包含大量 Block。最直观的例子是批处理（Batch Processing）：Grid 中的每个 Block 处理 Batch 中的一个独立样本。

> **为什么需要 Block 和 Thread 两级组织，而不能只有单一层次的 Thread？**  
> 因为硬件中由 SM 独立调度 Block，且 Block 内部共享高速 Shared Memory。Block 之间的执行完全解耦，没有固定的先后依赖关系——Block 0 和 Block 1 可能同时被执行，也可能是 Block 3 和 Block 0 先被调度。这种无序性正是 CUDA 能够随着 GPU 核心数增加实现线性扩展（Scalability）的根本所在。每个 Block 就像拼图中的一块，彼此独立解题，最终拼装出完整结果。

> 推荐阅读：[CUDA 线程、线程束与线程块如何映射到硬件核心？](https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores)