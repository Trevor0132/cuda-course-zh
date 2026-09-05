# CUDA 流机制与并发实战 (CUDA Streams)

## 直观理解 (Intuition)
你可以把 CUDA 流（Stream）想象成**“河流”**：操作严格沿着时间线向前流淌执行。例如：拷贝数据（阶段 1）-> 执行核函数计算（阶段 2）-> 拷贝回结果（阶段 3）。这就是流的基本工作模型。

在 CUDA 中，我们可以同时创建多个独立的流，每个流都拥有自己的独立时间线。这使得我们能够在时间上将不同流的操作相互**重叠（Overlap）**，从而最大化 GPU 的利用率并隐藏数据传输延迟。

在训练大规模语言模型时，如果把大量时间浪费在将 Token 张量在 CPU 与 GPU 之间来回串行等待传输上是极其低效的。CUDA 流允许我们在 GPU 执行计算的同时，在后台异步预取下一批数据。这种“预取（Prefetching）”软件模式能够完美隐藏数据搬运的时间延迟。

本项目包含两个实战示例：
1. `01_stream_basics.cu`：展示基本的异步内存拷贝与核函数并发执行。
2. `02_stream_advanced.cu`：演示流优先级、事件依赖与主机端回调等高级特性。

---

## 核心代码与 API

- **默认流 (Default Stream)** = 流 0 = 空流 (Null Stream)
```cpp
// 该核函数默认在空流（流 0）中排队执行
myKernel<<<gridSize, blockSize>>>(args);

// 完全等价于显式传入流 0：
myKernel<<<gridSize, blockSize, 0, 0>>>(args);
```

还记得核函数启动配置中的第四个参数吗？
`<<<gridDim, blockDim, Ns, S>>>`
- `S` (`cudaStream_t`)：用于指定该核函数关联的 CUDA 流。

### 流优先级配置
可以为流赋予不同的执行优先级，具有更高优先级的流中的任务会被调度器优先发射，为多任务并发控制带来更细腻的把控力：

```cpp
// 创建具有不同优先级的流
int leastPriority, greatestPriority;
CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));
```

## 编译运行

```bash
nvcc -o 01 01_stream_basics.cu
nvcc -o 02 02_stream_advanced.cu
```

## 官方参考资料
- [NVIDIA Streams & Concurrency 研讨会文档 (PDF)](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)

---

## 页锁定内存 (Pinned / Page-Locked Memory)

- 直观理解：“操作系统，这块内存 GPU 马上要频繁异步访问，千万不要把它换页（Page-out）换到硬盘交换区！”
- 普通的 `malloc` 分配的是可分页内存（Pageable Memory）。在进行异步传输（`cudaMemcpyAsync`）时，系统必须先将数据拷贝到临时锁页缓冲区，这不仅增加额外耗时，而且无法真正实现计算与传输的并发重叠。
- 使用 `cudaMallocHost` 分配的页锁定内存可以直接参与异步传输，实现极致的高速 DMA 搬运：

```cpp
// 分配页锁定内存 (Pinned Host Memory)
float* h_data;
cudaMallocHost((void**)&h_data, size);

// 释放使用 cudaFreeHost(h_data);
```

---

## CUDA 事件 (CUDA Events)

- **精确计时**：在核函数启动前后记录事件戳，测量纯核函数的耗时。
- **流间依赖同步**：通过 `cudaStreamWaitEvent` 让某个流等待另一个流中的特定事件触发，优雅实现跨流任务依赖。
- **计算与传输重叠确认**：使用事件标记数据传输完成，触发后续相关核函数启动。

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 记录起始点
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
// 记录结束点
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop); // 阻塞等待事件发生
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop); // 计算耗时（毫秒）
```

---

## 流回调函数 (Stream Callbacks)

- 通过 `cudaStreamAddCallback`，可以在 GPU 流上某项任务顺利执行完成后，自动在 CPU 端触发执行一个主机端的回调函数，从而构建精细的 CPU-GPU 异步流水线调度（Pipeline）。

```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU 端异步任务已完成，CPU 回调触发！\n");
    // 可在此处触发下一阶段的主机端任务调度
}

kernel<<<grid, block, 0, stream>>>(args);
cudaStreamAddCallback(stream, MyCallback, nullptr, 0);
```
