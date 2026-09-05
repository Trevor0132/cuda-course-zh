## CUDA 底层运算机制详解

以下是 CUDA 编程中最核心的基础构件，它们将在后续代码中被频繁使用：

### 1. 核函数 (Kernel)
核函数是在 GPU（图形显卡）上并发运行的特殊函数，而不是在 CPU 上执行。可以把它想象成向一支庞大的工人团队（GPU 线程）分派任务，所有工人同时并行开工。核函数必须使用 `__global__` 关键字修饰，且返回值必须为 `void`。

示例：
```cpp
__global__ void addNumbers(int *a, int *b, int *result) {
    *result = *a + *b;
}
```

### 2. 网格 (Grid)
网格表示某一次核函数调用所启动的**全部线程集合**。它是整个核函数的宏观执行空间，由若干个线程块（Thread Block）组成。

当启动核函数时，你需要指定 Grid 的维度（即创建多少个 Block）。Grid 可以是 1 维、2 维或 3 维（形如一条线、一个平面或一个三维空间立体块）。它专门用于组织超大规模的并行计算任务。
- **示例**：处理一张大尺寸图像时，Grid 中的每个 Block 可以负责处理图像中的一个子区域切片。

### 3. 线程块 (Block)
线程块是一组可以互相协作并通过高速共享内存（Shared Memory）快速共享数据的线程集合。Block 同样可以被定义为 1 维、2 维或 3 维。
- 同一个 Block 内的线程可以：
  - 共享片上高速内存
  - 彼此执行线程同步
  - 协同完成复杂的局部计算
- **示例**：在图像处理中，一个 Block 可以负责一个 $16 \times 16$ 像素的小区域。

### 4. 线程 (Thread)
线程是 CUDA 中最小的执行单元。每个线程独立执行核函数的指令序列。在 Block 内部，每个线程由唯一的线程 ID 区分。这个 ID 允许线程根据自己在 Block 中的位置去读取对应的数据或执行特定逻辑。

> 每个线程都有属于自己的唯一编号，从而精准识别自己需要处理哪一部分数据。

---

### 理解 CUDA 线程索引机制 (Thread Indexing)

在 CUDA 中，每个线程都可以通过内置变量精确定位其在整个 Grid 和 Block 中的物理坐标：

1. **`threadIdx`**：
   - 三维向量结构体（`threadIdx.x`, `threadIdx.y`, `threadIdx.z`），表示当前线程在所属 Block 内部的相对坐标。
   - 示例：如果是一个包含 256 个线程的一维 Block，`threadIdx.x` 的范围是 `0` 到 `255`。

2. **`blockDim`**：
   - 三维向量结构体（`blockDim.x`, `blockDim.y`, `blockDim.z`），指定当前 Block 的维度大小（每个 Block 包含多少线程）。
   - 示例：如果 Block 在 x 方向有 256 个线程，则 `blockDim.x` 为 `256`。

3. **`blockIdx`**：
   - 三维向量结构体（`blockIdx.x`, `blockIdx.y`, `blockIdx.z`），表示当前 Block 在整个 Grid 中的位置坐标。
   - 示例：如果一维 Grid 包含 10 个 Block，则 `blockIdx.x` 的范围是 `0` 到 `9`。

4. **`gridDim`**：
   - 三维向量结构体（`gridDim.x`, `gridDim.y`, `gridDim.z`），表示整个 Grid 的维度大小（总共有多少个 Block）。
   - 示例：如果 Grid 在 x 方向有 10 个 Block，则 `gridDim.x` 为 `10`。

### 计算全局全局线程 ID (Global Thread ID)

对于一维数据（例如遍历一个一维大数组），计算全局唯一线程索引的黄金公式为：

```cpp
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx.x * blockDim.x`：算出当前 Block 之前所有 Block 包含的线程总数（当前 Block 的起始偏移量）。
- `threadIdx.x`：加上当前线程在当前 Block 内部的相对偏移量。

---

## 常用辅助类型与函数

#### `dim3`
- CUDA 用于定义 3D 尺寸的结构体类型
- 常用于指定 Grid 和 Block 的长宽高配置
- 示例：
```cpp
dim3 blockSize(16, 16, 1);  // 每个 Block 包含 16x16x1 个线程
dim3 gridSize(8, 8, 1);     // Grid 包含 8x8x1 个 Block
```

#### `<<<...>>>` 执行配置语法 (Execution Configuration)

用于配置并启动 GPU 核函数。它指定了核函数的网格维度、线程块维度、动态共享内存大小以及关联的 CUDA 流。合理配置这些参数对于释放 GPU 性能至关重要。

```cpp
addNumbers<<<gridSize, blockSize>>>(a, b, result);
```

其中 `addNumbers` 是核函数名，`gridSize` 和 `blockSize` 为启动配置参数，`a`、`b`、`result` 是传入核函数的参数。

---

## 显存管理机制

#### `cudaMalloc`
- 在 GPU 显存（全局内存）上分配指定字节大小的内存空间
- 类似于标准 C 的 `malloc`，但分配的是 GPU 物理显存
- 示例：
```cpp
int *device_array;
cudaMalloc(&device_array, size * sizeof(int));
```

#### `cudaMemcpy`
在 Host 与 Device 之间或 Device 内部执行数据拷贝：
- `cudaMemcpyHostToDevice`：主机端（CPU）内存拷贝至设备端（GPU）显存
- `cudaMemcpyDeviceToHost`：设备端（GPU）显存拷贝回主机端（CPU）内存
- `cudaMemcpyDeviceToDevice`：GPU 显存内部地址之间相互拷贝

释放显存调用 `cudaFree(device_array)`。

示例：
```cpp
cudaMemcpy(device_array, host_array, size * sizeof(int), cudaMemcpyHostToDevice);
```

#### `cudaDeviceSynchronize()`
默认情况下，GPU 核函数的启动是异步的（CPU 发起启动指令后会立即向下执行，无需等待 GPU 执行完毕）。在需要等待 GPU 任务彻底完成时，调用 `cudaDeviceSynchronize()` 会阻塞 CPU 线程，直到当前设备上的所有排队任务全部执行完毕。
- 适用场景：需要根据 GPU 输出结果在 CPU 上继续处理，或者进行精准性能基准测试（Benchmark）计时。
- 示例：
```cpp
kernel<<<gridSize, blockSize>>>(data);
cudaDeviceSynchronize();  // 等待 GPU 核函数完全执行完毕
printf("核函数执行完毕!\n");
```

---

## 内存层级理论深入

CUDA 架构提供了多种不同物理特性与速度的内存层次：

1. **全局内存 (Global Memory)**：
   - 显卡板载的主显存（VRAM），所有 Block 的所有线程均可访问。
   - **速度最慢**，但**容量最大**（GB 级别）。
   - 适合存放线程间共享的大规模输入/输出数据集。

2. **共享内存 (Shared Memory)**：
   - 位于芯片内部的高速片上 SRAM，由同一个 **Block** 内的全部线程共享。
   - **速度极快**，但**容量很小**（每个 SM 通常几十 KB 到上百 KB）。
   - 用于同一 Block 内线程间的频繁数据协作与中间变量暂存（如分块矩阵乘法）。

3. **寄存器 (Registers)**：
   - 访问速度最快，每个 **线程** 私有。
   - 用于存储局部标量变量。
   - 数量有限，应合理控制使用量以避免寄存器溢出（Spilling）。

4. **常量内存 (Constant Memory)**：
   - 对所有线程只读的常量显存空间（64KB 限制）。
   - 配有专用硬件缓存，支持统一广播读取。
   - 适合存放核函数执行期间全局不变的配置参数或常量系数。

5. **局部内存 (Local Memory)**：
   - 当线程使用的寄存器不足时，多出的变量溢出（Spill）至此。
   - **速度极慢**（物理上位于片外全局内存，延迟极高）。
   - 应尽量通过代码优化避免局部内存溢出。

---

## 融会贯通示例

以下示例将上述概念整合为一个完整的两数组并行相加程序：

```cpp
// 核函数定义
__global__ void addArrays(int *a, int *b, int *c, int size) {
    // 计算当前线程的全局唯一索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查：确保不发生数组越界访问
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

// 在 main 主机函数中：
dim3 blockSize(256);  // 每个 Block 分配 256 个线程
dim3 gridSize((size + blockSize.x - 1) / blockSize.x);  // 向上取整计算所需 Block 数量
addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
```

该示例展示了如何组织线程以并行计算两个数组的加法：每个线程负责一个元素的计算，并通过合理的 Grid 和 Block 划分覆盖整个数组范围。
