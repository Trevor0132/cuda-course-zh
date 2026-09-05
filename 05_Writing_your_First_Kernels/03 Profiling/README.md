# 如何对 CUDA 核函数进行性能分析 (Profiling)

## 实操练习步骤

1. 编译并使用 Nsight Systems 剖析 NVTX 矩阵乘法：
```bash
nvcc -o 00 00\ nvtx_matmul.cu -lnvToolsExt
nsys profile --stats=true ./00
```

> 生成的 `.nsys-rep` 文件可以直接拖拽到 Linux/Windows 上的 `nsys-ui` 图形界面左侧栏中打开。  
> 生成的 `.sqlite` 文件可直接导入 SQLite 数据库进行定制化的 SQL 查询与深入分析。

2. 朴素矩阵乘法分析：
```bash
nvcc -o 01 01_naive_matmul.cu
nsys profile --stats=true ./01
```

3. 共享内存分块（Tiled）矩阵乘法分析：
```bash
nvcc -o 02 02_tiled_matmul.cu
nsys profile --stats=true ./02
```

## 常用命令行工具 (CLI Tools)

- 监控 GPU 资源使用率与显存占用：
  - `nvitop`：交互式、现代化的 GPU 监控终端工具
  - `nvidia-smi` 或实时刷新：`watch -n 0.1 nvidia-smi`

# Nsight Systems 与 Nsight Compute

- 老旧的 `nvprof` 已被废弃，目前官方推荐使用 `nsys` (Nsight Systems) 与 `ncu` (Nsight Compute)。
- 基础性能剖析命令：`nsys profile --stats=true ./main`
![](../assets/nsight-ui.png)

### 推荐的性能调优策略（两步法）：
1. **第一步（宏观瓶颈定位）**：先使用 **Nsight Systems (nsys)** 分析系统全局瓶颈（如 CPU-GPU 通信、内存传输等待、算子排队空闲等），找出最耗时、对整体性能影响最大的关键核函数（Hotspot Kernels）。
2. **第二步（微观算子调优）**：针对定位出的瓶颈核函数，使用 **Nsight Compute (ncu)** 进行细粒度的硬件性能计数器分析（如访存合并率、SM 占用率 Occupancy、指令停顿原因等），寻找具体的代码级优化方案。

- 针对生成的分析报告：
  - 执行 `nsys stats file.nsys-rep` 获取定量统计概览
  - 执行 `nsys analyze file.sqlite` 提取结构化指标数据
- 打开图形界面分析：启动 `nsight-sys` (或在桌面环境打开 `nsys-ui`) ⇒ File ⇒ Open ⇒ 选择生成的 `.nsys-rep` 文件。
- 分析 Python / PyTorch 脚本示例：
  ```bash
  nsys profile --stats=true -o mlp python mlp.py
  ```
- **权限问题排查**：如果在运行 `ncu` 时遇到性能计数器权限被拒错误（Permission denied），可编辑配置文件 `/etc/modprobe.d/nvidia.conf`，添加配置项 `options nvidia NVreg_RestrictProfilingToAdminUsers=0`，随后重启机器生效。[参考 NVIDIA 开发者论坛说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)。
- 显存越界与泄漏检查：`compute-sanitizer ./main`（替代旧的 cuda-memcheck）
- 核函数性能分析图形界面：`ncu-ui`

## 核函数细粒度分析 (ncu)

- 官方文档：[Nsight Compute Kernel Profiling 指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- 命令行示例：
  ```bash
  ncu --kernel-name matrixMulKernelOptimized --launch-skip 0 --launch-count 1 --section Occupancy "./nvtx_matmul"
  ```

## 向量加法性能演进对比

对 3200 万（$2^{25}$）元素的一维向量加法在三种实现下进行分析：
1. 单线程 CPU / 基础版本（无 Block 与 Thread 级并行）：
![](../assets/prof1.png)
2. 引入 Thread 并行：
![](../assets/prof2.png)
3. 结合 Thread 与 Block 两级网格并发：
![](../assets/prof3.png)

## NVTX 标记与范围标注 (NVIDIA Tools Extension)

```bash
# 编译时链接 libnvToolsExt
nvcc -o matmul matmul.cu -lnvToolsExt

# 使用 Nsight Systems 采集范围标记与性能数据
nsys profile --stats=true ./matmul
```

## CUPTI 底层接口

- **CUPTI (CUDA Profiling Tools Interface)**：支持开发者构建自定义的性能分析与调用追踪工具。
- 提供了 Activity API、Callback API、Event API、Metric API、SASS Metric API 以及 Checkpoint API 等接口，能够在底层探查 CUDA 程序在 CPU 与 GPU 之间的行为交互。
- 官方文档：https://docs.nvidia.com/cupti/overview/overview.html
- CUPTI 学习曲线较为陡峭，本课程中我们优先掌握开箱即用的官方性能剖析工具（`nsys` 与 `ncu`）。