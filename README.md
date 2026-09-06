# CUDA 课程

FreeCodeCamp 上的 CUDA 课程官方 GitHub 仓库

> 注意：本课程专为 Ubuntu Linux 设计。Windows 用户可以使用 WSL（Windows Subsystem for Linux）或 Docker 容器来模拟 Ubuntu Linux 环境。

## 目录

1. [深度学习生态系统](01_Deep_Learning_Ecosystem/README.md)
2. [环境配置与安装](02_Setup/README.md)
3. [C/C++ 复习](03_C_and_C++_Review/README.md)
4. [GPU 浅显入门](04_Gentle_Intro_to_GPUs/README.md)
5. [编写你的第一个核函数](05_Writing_your_First_Kernels/README.md)
6. [CUDA API (cuBLAS, cuDNN 等)](06_CUDA_APIs/README.md)
7. [矩阵乘法优化](07_Faster_Matmul/README.md)
8. [Triton](08_Triton/README.md)
9. [PyTorch 扩展 (CUDA)](09_PyTorch_Extensions/README.md)
10. [结课项目](10_Final_Project/README.md)
11. [附加内容](11_Extras/README.md)
12. [FlashAttention (闪电注意力机制)](12_FlashAttention/README.md)

## 课程理念

本课程旨在：

- 降低进入高性能计算（HPC）领域的门槛
- 为理解像 Karpathy 的 [llm.c](https://github.com/karpathy/llm.c) 这样的项目奠定基础
- 将零散的 CUDA 编程资源整合为一套全面、结构清晰的课程

## 课程概述

- 聚焦于通过 GPU 核函数优化来提升性能
- 涵盖 CUDA、PyTorch 和 Triton
- 强调编写更高效核函数的技术细节
- 专为 NVIDIA GPU 量身定制
- 以 CUDA 实现的简易 MNIST 多层感知机（MLP）项目作为结课成果

## 前置知识

- Python 编程（必备）
- 反向传播所需的基础微分与向量微积分（推荐）
- 线性代数基础（推荐）

## 核心收获

- 优化现有的算子实现
- 为前沿研究构建自定义 CUDA 核函数
- 深入理解 GPU 性能瓶颈，尤其是内存带宽限制

## 硬件要求

- 任意 NVIDIA GTX、RTX 或数据中心级别 GPU
- 无本地硬件的用户可选用云端 GPU

## CUDA / GPU 编程的应用场景

- 深度学习（本课程的核心重点）
- 计算机图形学与光线追踪
- 流体模拟
- 视频编辑
- 加密货币计算
- 3D 建模
- 任何需要针对大型数组进行并行处理的场景

## 参考资源

- GitHub 仓库（即本仓库）
- Stack Overflow
- NVIDIA 开发者论坛
- NVIDIA 与 PyTorch 官方文档
- 辅助探索与答疑的大语言模型（LLM）
- 速查表见 [此处](/11_Extras/assets/cheatsheet.md)

## 其他学习资料

- https://github.com/CoffeeBeforeArch/cuda_programming
- https://www.youtube.com/@GPUMODE
- https://discord.com/invite/gpumode

## 推荐 YouTube 视频

- [How do GPUs works? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU)
- [But how do GPUs actually work?](https://www.youtube.com/watch?v=58jtf24uijw&ab_channel=Graphicode)
- [Getting Started With CUDA for Python Programmers](https://www.youtube.com/watch?v=nOxKexn3iBo&ab_channel=JeremyHoward)
- [Transformers Explained From The Atom Up](https://www.youtube.com/watch?v=7lJZHbg0EQ4&ab_channel=JacobRintamaki)
- [How CUDA Programming Works - Stephen Jones, CUDA Architect, NVIDIA](https://www.youtube.com/watch?v=QQceTDjA4f4&ab_channel=ChristopherHollinworth)
- [Parallel Computing with Nvidia CUDA - NeuralNine](https://www.youtube.com/watch?v=zSCdTOKrnII&ab_channel=NeuralNine)
- [CPU vs GPU vs TPU vs DPU vs QPU](https://www.youtube.com/watch?v=r5NQecwZs1A&ab_channel=Fireship)
- [Nvidia CUDA in 100 Seconds](https://www.youtube.com/watch?v=pPStdjuYzSI&ab_channel=Fireship)
- [How AI Discovered a Faster Matrix Multiplication Algorithm](https://www.youtube.com/watch?v=fDAPJ7rvcUw&t=1s&ab_channel=QuantaMagazine)
- [The fastest matrix multiplication algorithm](https://www.youtube.com/watch?v=sZxjuT1kUd0&ab_channel=Dr.TreforBazett)
- [From Scratch: Cache Tiled Matrix Multiplication in CUDA](https://www.youtube.com/watch?v=ga2ML1uGr5o&ab_channel=CoffeeBeforeArch)
- [From Scratch: Matrix Multiplication in CUDA](https://www.youtube.com/watch?v=DpEgZe2bbU0&ab_channel=CoffeeBeforeArch)
- [Intro to GPU Programming](https://www.youtube.com/watch?v=G-EimI4q-TQ&ab_channel=TomNurkkala)
- [CUDA Programming](https://www.youtube.com/watch?v=xwbD6fL5qC8&ab_channel=TomNurkkala)
- [Intro to CUDA (part 1): High Level Concepts](https://www.youtube.com/watch?v=4APkMJdiudU&ab_channel=JoshHolloway)
- [Intro to GPU Hardware](https://www.youtube.com/watch?v=kUqkOAU84bA&ab_channel=TomNurkkala)

## 联系作者

- [Twitter/X](https://x.com/elliotarledge)
- [LinkedIn](https://www.linkedin.com/in/elliot-arledge-a392b7243/)
- [YouTube](https://www.youtube.com/channel/UCjlt_l6MIdxi4KoxuMjhYxg)
- [Discord](https://discord.gg/JTTcFe7Pw2)
