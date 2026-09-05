# 大规模多卡集群与数据中心计算

对比：**cuBLASMp** vs **NCCL** vs **MIG (多实例 GPU)**

## cuBLASMp
- **NVIDIA cuBLASMp** 是一款面向分布式环境的高性能、多进程、GPU 加速基础稠密线性代数库。
- 主要用于**单节点内多 GPU** 的稠密张量运算。当单个模型层或单次矩阵乘法超出了单张显卡的显存容量时，可以使用 cuBLASMp 将计算跨卡拆分。

## NCCL (NVIDIA 集体通信库)
- **NVIDIA Collective Communications Library (NCCL)**：专为跨 GPU、跨节点的多卡分布式集群通信而生。
- NCCL 负责在多卡/多机之间高速分发、同步与汇聚张量数据。如果在 8 卡 H100 服务器上由 cuBLAS 处理每张卡上的矩阵乘法，则由 NCCL 负责在卡间执行批次数据的跨卡通信。
- **核心集体通信原语**：
  - **All-Reduce**：全归约（计算梯度平均最常用）
  - **Broadcast**：广播（单卡向所有卡同步权重）
  - **Reduce-Scatter**：归约散射
  - **All-Gather**：全收集
- 在 PyTorch 中，这对应于高层的 `DistributedDataParallel` (DDP) 或 FSDP。而在最顶级的超大规模训练框架中，专家通常会针对 NCCL 通信拓扑与 CUDA 通信核函数进行深入调优。
- 视频教程：[CUDA MODE: NCCL 深度解析讲座](https://www.youtube.com/watch?v=T22e3fgit-A&ab_channel=CUDAMODE)
- [扩展 GPU 显存文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-gpu-memory)
- 并行范式对比：模型并行（Model Parallelism，切分模型权重） vs 数据并行（Data Parallelism，切分训练 Batch）
- 官方文档：[NCCL 用户指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html) 与 [NCCL API 参考](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)

## MIG (Multi-Instance GPU，多实例 GPU)
- MIG 允许将一张物理大算力 GPU（如 A100 / H100）硬件级切割成多达 7 个完全独立、互不干扰的微型 GPU 实例。
- 广泛应用于云计算与数据中心多租户场景，为吞吐要求不高或轻量级推理任务提供硬件级 QoS 隔离与最大化的设备利用率。
