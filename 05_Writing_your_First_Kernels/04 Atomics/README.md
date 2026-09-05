# 什么是原子操作 (Atomic Operations)

“原子（Atomic）”在物理学中原意代表不可再被分割的基本粒子。在计算机并行计算中，**原子操作**表示某一个操作是不可分割、互不干扰且一气呵成的。

**原子操作**能够确保对某个内存地址的操作由单个线程完全独立执行完毕之后，其他线程才被允许访问或修改该相同的内存地址。这彻底杜绝了多线程并发读写时的竞争条件（Race Condition）。

因为原子操作在微观上序列化了针对同一内存地址的访问，限制了单位时间内对该显存位置的并行吞吐，所以会在一定程度上带来延迟开销。它是**以少许性能为代价，换取硬件级别的内存数据安全性与一致性**。

### **整型原子操作 API**

- **`atomicAdd(int* address, int val)`**：原子地将 `val` 加到 `address` 指向的值上，并返回修改前的旧值。
- **`atomicSub(int* address, int val)`**：原子地从 `address` 指向的值中减去 `val`，并返回旧值。
- **`atomicExch(int* address, int val)`**：原子地将 `address` 处的值替换为 `val`，并返回旧值。
- **`atomicMax(int* address, int val)`**：原子地将 `address` 处的值设为当前值与 `val` 的较大者。
- **`atomicMin(int* address, int val)`**：原子地将 `address` 处的值设为当前值与 `val` 的较小者。
- **`atomicAnd(int* address, int val)`**：原子地在 `address` 处执行按位与（Bitwise AND）。
- **`atomicOr(int* address, int val)`**：原子地在 `address` 处执行按位或（Bitwise OR）。
- **`atomicXor(int* address, int val)`**：原子地在 `address` 处执行按位异或（Bitwise XOR）。
- **`atomicCAS(int* address, int compare, int val)`**：比较并交换（Compare-And-Swap）。原子地比较 `address` 处的值是否等于 `compare`；若相等，则将其替换为 `val`。无论是否替换，始终返回 `address` 处的原始值。

### **浮点型原子操作 API**

- **`atomicAdd(float* address, float val)`**：单精度浮点数原子加法，从 CUDA 2.0 起支持。
- **注**：双精度浮点数（`double`）原子加法 `atomicAdd(double* address, double val)` 从 Compute Capability 6.0（Pascal 架构）起获得原生硬件支持。

---

### 从零理解原子操作的底层逻辑

现代 GPU 拥有专用的硬件指令来高效执行原子操作。它们在硬件电路层级广泛运用了 CAS（Compare-And-Swap）机制。

你可以将原子操作视作硬件级别极度高效的轻量级互斥锁（Mutex）。每个原子操作在概念上等价于以下执行序列：

1. `lock(memory_location)`：对该内存地址加锁
2. `old_value = *memory_location`：读取旧值
3. `*memory_location = old_value + increment`：写入新值
4. `unlock(memory_location)`：释放锁
5. `return old_value`：返回旧值

软件层面上利用 `atomicCAS` 模拟原子加法的原理示例：

```cpp
__device__ int softwareAtomicAdd(int* address, int increment) {
    __shared__ int lock;
    int old;
    
    if (threadIdx.x == 0) lock = 0;
    __syncthreads();
    
    // 自旋等待获取锁
    while (atomicCAS(&lock, 0, 1) != 0);
    
    old = *address;
    *address = old + increment;
    
    __threadfence();  // 确保写入对其他线程立即可见
    
    atomicExch(&lock, 0);  // 释放锁
    
    return old;
}
```

- 互斥（Mutual Exclusion）推荐视频：https://www.youtube.com/watch?v=MqnpIwN7dz0
  - “Mutual（互）”：指实体（线程/进程）之间的相互对等关系，规则平等约束所有参与方。
  - “Exclusion（斥）”：指防止对临界资源的并发同时访问。

---

### GPU 互斥锁（Mutex）实战完整代码

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// 自定义 Mutex 结构体
struct Mutex {
    int *lock;
};

// 在主机端初始化 Mutex
__host__ void initMutex(Mutex *m) {
    cudaMalloc((void**)&m->lock, sizeof(int));
    int initial = 0;
    cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice);
}

// 在设备端获取互斥锁（自旋锁）
__device__ void lock(Mutex *m) {
    while (atomicCAS(m->lock, 0, 1) != 0) {
        // 自旋等待
    }
}

// 在设备端释放互斥锁
__device__ void unlock(Mutex *m) {
    atomicExch(m->lock, 0);
}

// 演示 Mutex 保护临界区的核函数
__global__ void mutexKernel(int *counter, Mutex *m) {
    lock(m);
    // 临界区（Critical Section）
    int old = *counter;
    *counter = old + 1;
    unlock(m);
}

int main() {
    Mutex m;
    initMutex(&m);
    
    int *d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));
    int initial = 0;
    cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);
    
    // 启动包含 1000 个线程的核函数同时累加计数器
    mutexKernel<<<1, 1000>>>(d_counter, &m);
    
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("计数器最终结果: %d\n", result);
    
    cudaFree(m.lock);
    cudaFree(d_counter);
    
    return 0;
}
```