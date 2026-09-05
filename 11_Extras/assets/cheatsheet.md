# CUDA 常用速查表 (Cheatsheet)

## 主机与设备端内存传输 (Memory Transfer)

1. **`cudaMalloc((void**)&d_array, size * sizeof(type));`**
   - 在 GPU 设备端全局显存中分配指定字节大小的内存。
   - `d_array` 是指向分配的设备端显存空间的指针。

2. **`cudaMemcpy(d_array, h_array, size * sizeof(type), cudaMemcpyHostToDevice);`**
   - 将数据从主机端数组 `h_array` 拷贝到设备端数组 `d_array`。
   - 传输方向：主机端（CPU）-> 设备端（GPU）。

3. **`cudaMemcpy(h_array, d_array, size * sizeof(type), cudaMemcpyDeviceToHost);`**
   - 将数据从设备端数组 `d_array` 拷贝回主机端数组 `h_array`。
   - 传输方向：设备端（GPU）-> 主机端（CPU）。

4. **`cudaFree(d_array);`**
   - 释放先前在 GPU 设备端分配的显存空间。
   - 必须及时调用以防止显存泄漏。

5. **`cudaMallocHost((void**)&h_array, size * sizeof(type));`**
   - 在主机端分配页锁定（Pinned / Page-Locked）物理内存。
   - 页锁定内存支持 GPU 进行高速直接内存访问（DMA），并允许进行异步内存传输。

6. **`cudaFreeHost(h_array);`**
   - 释放使用 `cudaMallocHost` 分配的页锁定主机内存。

7. **`cudaMemcpyAsync(dst, src, size, kind, stream);`**
   - 在指定的 CUDA 流中执行异步数据拷贝。
   - 允许数据传输与核函数计算在后台同时并发重叠执行。

---

## 核函数启动语法 (Kernel Launch Syntax)

8. **`__global__ void kernelName(parameters) { /* 核函数代码 */ }`**
   - 定义在 GPU 上执行的 CUDA 核函数。
   - `__global__` 限定符表示该函数在设备端并发执行，由主机端发起调用。

9. **`kernelName<<<gridDim, blockDim>>>(arguments);`**
   - 以指定的网格维度（`gridDim`）和线程块维度（`blockDim`）在 GPU 上启动核函数。
   - 分别定义了 Block 的总数量以及每个 Block 包含的线程数。

10. **`kernelName<<<gridDim, blockDim, sharedMemSize>>>(arguments);`**
    - 启动核函数，并为每个线程块动态分配额外的共享内存（字节数由 `sharedMemSize` 指定）。

11. **`kernelName<<<gridDim, blockDim, sharedMemSize, stream>>>(arguments);`**
    - 启动核函数，指定动态共享内存大小并将其绑定到特定的 CUDA 流（`stream`）中异步执行。

---

## 线程坐标与索引计算 (Thread Indexing)

### 一维索引 (1D)

12. **`int tid = threadIdx.x;`**
    - 获取当前线程在所属线程块内部的 x 轴相对索引（0 到 blockDim.x - 1）。

13. **`int bid = blockIdx.x;`**
    - 获取当前线程块在网格中的 x 轴索引（0 到 gridDim.x - 1）。

14. **`int idx = blockIdx.x * blockDim.x + threadIdx.x;`**
    - 计算一维网格中当前线程的全局唯一索引。常用于扁平数组的元素定位。

### 二维索引 (2D)

15. **`int row = blockIdx.y * blockDim.y + threadIdx.y;`**
    - 结合 y 维度的 Block 和 Thread 索引计算二维全局行索引。常用于图像或矩阵的行定位。

16. **`int col = blockIdx.x * blockDim.x + threadIdx.x;`**
    - 结合 x 维度的 Block 和 Thread 索引计算二维全局列索引。常用于图像或矩阵的列定位。

17. **`int idx_2d = row * width + col;`**
    - 将二维的 (行, 列) 坐标展平转换为一维行优先（Row-Major）连续内存偏移量。

### 三维索引 (3D)

18. **`int idx_3d = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;`**
    - 将三维网格中的坐标转换为一维线性索引，常用于处理三维体素（Volumetric）或高阶张量数据。

19. **多维索引注意事项：**
    - 必须仔细检查网格与线程块各轴向维度配置，严防越界；结合硬件特性优化访存合并。

---

## 函数限定符 (Function Qualifiers)

20. **`__global__ void functionName() { }`**
    - 声明 CUDA 核函数。由主机端（CPU）调用，在设备端（GPU）上由多个线程并发执行。

21. **`__device__ void functionName() { }`**
    - 声明设备端子函数。仅可在 GPU 上运行，且只能由其他 `__device__` 函数或 `__global__` 核函数内部调用。

22. **`__host__ void functionName() { }`**
    - 声明普通主机端函数。由 CPU 调用并在 CPU 上执行（函数默认即为 `__host__`）。

23. **`__noinline__ void functionName() { }`**
    - 提示编译器不要对该函数进行内联展开。

24. **`__forceinline__ void functionName() { }`**
    - 强制编译器在编译时将该函数内联展开，消除函数调用的栈帧开销。

---

## 变量存储限定符 (Variable Qualifiers)

25. **`__shared__ type variableName;`**
    - 声明共享内存变量。位于片上高速 SRAM 中，同一个 Block 内的所有线程均可读写该变量。

26. **`__device__ type variableName;`**
    - 声明静态设备端全局内存变量，驻留在全局显存中，生命周期贯穿整个应用程序。

27. **`__constant__ type variableName;`**
    - 声明常量内存变量（最大 64KB）。对所有线程只读，配有硬件专用高速常量缓存，支持单周期广播读取。

28. **`__managed__ type variableName;`**
    - 声明统一内存（Unified Memory）变量。CPU 和 GPU 均可直接读写访问，由底层驱动自动按需迁移数据。

---

## 线程同步与内存栅障 (Thread Synchronization & Fences)

29. **`__syncthreads();`**
    - 块内同步屏障。确保当前 Block 内的所有线程都执行到此行后，才允许继续向下推进。

30. **`__syncthreads_and(predicate);`**
    - 同步当前 Block 内所有线程并评估条件。当且仅当该 Block 内**所有**线程的断言均为真时返回非零值。

31. **`__syncthreads_or(predicate);`**
    - 同步当前 Block 内所有线程并评估条件。只要该 Block 内有**至少一个**线程的断言为真，即返回非零值。

32. **`__syncthreads_count(predicate);`**
    - 同步当前 Block 内所有线程，并统计其中断言评估结果为真的线程总数。

33. **`__threadfence();`**
    - 设备级内存栅障。确保调用线程之前的所有显存写入对该 GPU 设备上的**所有线程**均立即可见。

34. **`__threadfence_block();`**
    - 块级内存栅障。确保调用线程之前的所有写入对**同 Block 内的其他线程**立即可见。

35. **`__threadfence_system();`**
    - 系统级内存栅障。确保调用线程的写入对系统中的所有计算实体（包括 CPU、其他 GPU）全部立即可见。

---

## 流同步 (Stream Synchronization)

36. **`cudaStreamSynchronize(stream);`**
    - 阻塞主机端 CPU，直到指定 CUDA 流中的所有排队操作全部执行完毕。

37. **`cudaDeviceSynchronize();`**
    - 阻塞主机端 CPU，直到当前 GPU 设备上的全部流和所有排队操作彻底执行完毕。

38. **`cudaStreamWaitEvent(stream, event);`**
    - 让指定的 CUDA 流异步等待某个事件触发，实现轻量级、非阻塞的跨流依赖控制。

---

## 原子操作 (Atomic Operations)

39. **`atomicAdd(address, val);`**
    - 原子加法：将 `val` 加到 `address` 处的值上，返回原值。

40. **`atomicSub(address, val);`**
    - 原子减法：从 `address` 处的值减去 `val`，返回原值。

41. **`atomicExch(address, val);`**
    - 原子交换：将 `address` 处的值替换为 `val`，返回原值。

42. **`atomicMin(address, val);`**
    - 原子求最小值：将 `address` 处的值更新为其当前值与 `val` 的较小者，返回原值。

43. **`atomicMax(address, val);`**
    - 原子求最大值：将 `address` 处的值更新为其当前值与 `val` 的较大者，返回原值。

44. **`atomicInc(address, val);`**
    - 原子自增：将 `address` 处的值加 1；若超过 `val` 则循环回绕重置为 0，返回原值。

45. **`atomicDec(address, val);`**
    - 原子自减：将 `address` 处的值减 1；若小于等于 0 或原值大于 `val` 则循环回绕为 `val`，返回原值。

46. **`atomicCAS(address, compare, val);`**
    - 原子比较并交换（Compare-And-Swap）：若 `*address == compare` 则写入 `val`；无论是否写入，均返回 `address` 处的旧值。

---

## 错误处理 (Error Handling)

47. **`cudaError_t error = cudaGetLastError();`**
    - 获取最近一次 CUDA 调用的错误码，同时清空内部错误状态。常用于核函数发射后的错误检测。

48. **`const char* errorString = cudaGetErrorString(error);`**
    - 将 CUDA 错误枚举码转换为人类可读的字符串描述。

49. **通用错误检查宏：**
    ```cpp
    #define CUDA_CHECK(call) { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA 错误于 %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    }
    ```

---

## 设备管理 (Device Management)

50. **`cudaDeviceProp prop; cudaGetDeviceProperties(&prop, deviceId);`**
    - 查询指定 GPU 设备（`deviceId`）的硬件属性与参数配置（计算能力架构版本、显存大小、SM 数量等）。

51. **`cudaSetDevice(deviceId);`**
    - 设置当前线程后续所有 CUDA 操作的目标 GPU 设备。多卡编程时用于卡间切换。

52. **`int deviceId; cudaGetDevice(&deviceId);`**
    - 获取当前激活的 CUDA 设备编号。

53. **`int deviceCount; cudaGetDeviceCount(&deviceCount);`**
    - 查询系统中可用的支持 CUDA 的 GPU 设备总数。

---

## 流管理 (Stream Management)

54. **`cudaStream_t stream; cudaStreamCreate(&stream);`**
    - 创建一个用于异步执行任务的标准 CUDA 流。

55. **`cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);`**
    - 创建一个非阻塞流（Non-blocking Stream），该流中的核函数不会与默认流（流 0）产生隐式串行同步。

56. **`cudaStreamDestroy(stream);`**
    - 销毁指定的 CUDA 流并回收底层相关驱动资源。

---

## 事件与精确计时 (Events & Timing)

57. **`cudaEvent_t event; cudaEventCreate(&event);`**
    - 创建一个 CUDA 事件对象，用于流同步标记或精确时间测量。

58. **`cudaEventRecord(event, stream);`**
    - 在指定的流中插入/记录一个事件标记。

59. **`cudaEventSynchronize(event);`**
    - 阻塞主机端 CPU，直到指定的事件被 GPU 执行记录完成。

60. **`cudaEventElapsedTime(&ms, start, stop);`**
    - 精确计算两个事件记录点（`start` 和 `stop`）之间的 GPU 实际执行耗时（毫秒，精度约 0.5 微秒）。

61. **`cudaEventDestroy(event);`**
    - 销毁 CUDA 事件对象并释放资源。

---

## 常用实用宏与计算公式

```c
// 线程块大小
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// 向上取整计算所需 Grid 大小的宏
#define GRID_SIZE(n, b) (((n) + (b) - 1) / (b))

// 查询硬件支持的最大 Grid 与 Block 维度
dim3 maxGridSize(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
dim3 maxThreadsPerBlock(prop.maxThreadsPerBlock);
```

62. **`#define BLOCK_SIZE 256`**：通常将线程块尺寸设为 32 的整数倍（如 128、256），以便整除对齐 Warp。
63. **`#define WARP_SIZE 32`**：定义 Warp 大小，所有支持 CUDA 的硬件上 Warp 恒为 32 个线程。
64. **`#define GRID_SIZE(n, b) (((n) + (b) - 1) / (b))`**：向上取整计算处理 `n` 个元素所需的最少 Block 数量。

---

## 运行时常用环境配置 API

65. **`cudaDeviceReset();`**
    - 重置当前 CUDA 设备并清空当前进程在该设备上分配的所有资源（显存、流、事件等）。

66. **`cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);`**
    - 为指定核函数配置 L1/共享内存偏好，倾向于分配更大的片上共享内存。

67. **`cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);`**
    - 为指定核函数配置偏好，倾向于分配更大的片上 L1 缓存空间。

68. **`cudaDeviceSetLimit(cudaLimitStackSize, value);`**
    - 设置当前设备每个线程私有调用栈的大小上限（递归或大局部变量时使用）。

69. **`cudaDeviceSetLimit(cudaLimitMallocHeapSize, value);`**
    - 设置设备端动态内存分配（设备端 `malloc`/`new`）的堆内存大小上限。

---

## 模板核函数 (Template Kernels)

70. **模板核函数定义：**
    ```cpp
    template<typename T>
    __global__ void kernelName(T* data) {
        // 核函数实现逻辑
    }
    ```
    - 允许同一个核函数同时支持多种数据类型（如 `float`, `half`, `int`）。

71. **启动模板核函数：**
    ```cpp
    kernelName<float><<<grid, block>>>(data);
    ```

---

## 线程束级原语 (Warp-Level Primitives)

> 自 CUDA 9.0 起引入，通过 32 位掩码 `mask` 精确控制 Warp 内参与协同的活跃线程。

### 线程束投票指令 (Warp Vote)

72. **`__all_sync(mask, predicate);`**
    - 评估 Warp 内所有活跃线程的条件断言。当且仅当 `mask` 指定的**全部**线程断言均为真时返回 1。

73. **`__any_sync(mask, predicate);`**
    - 评估 Warp 内活跃线程的条件断言。只要 `mask` 指定的线程中**有任意一个**断言为真，即返回 1。

74. **`__ballot_sync(mask, predicate);`**
    - 收集 Warp 内所有线程的断言结果，返回一个 32 位整型掩码（第 $N$ 位对应第 $N$ 号线程的真假值）。

### 线程束洗牌指令 (Warp Shuffle - 寄存器直通交换)

> 允许同一个 Warp 内的线程直接跨寄存器交换数据，**无需经过共享内存**，延迟仅为单个时钟周期！

75. **`__shfl_sync(mask, var, srcLane);`**
    - 广播：将编号为 `srcLane`（0 到 31）的线程中的变量 `var` 直接广播给 Warp 内的其他活跃线程。

76. **`__shfl_up_sync(mask, var, delta);`**
    - 向上平移：当前线程获取编号为 `(current_lane - delta)` 的线程持有的变量 `var`。

77. **`__shfl_down_sync(mask, var, delta);`**
    - 向下平移：当前线程获取编号为 `(current_lane + delta)` 的线程持有的变量 `var`。广泛用于 Warp 级树状归约求和（Reduction）。

78. **`__shfl_xor_sync(mask, var, laneMask);`**
    - 蝶形交换：当前线程与编号为 `(current_lane ^ laneMask)` 的线程相互交换变量 `var`。常用于 FFT 与快速排序算法。
