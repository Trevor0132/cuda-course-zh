# 第 01 章：当前深度学习生态系统

### **免责声明：** 

本部分不涉及过多高深的 CUDA 技术细节。在盲目钻研技术细节之前，先纵观整个生态全局是更好的学习方式。根据我的学习经验，对生态系统具备足够清晰的认知不仅能帮助你建立全局知识图谱，还能为你提供最初的学习动力。

随着我们后续不断深入细节，我鼓励大家主动探索并把玩自己感兴趣的技术（在本节中你会遇到很多酷炫的内容）。如果只是听别人讲 20 个小时，你的学习效果会非常有限。理清深度学习基础设施的广度与深度确实充满挑战，而跳出舒适区、亲自动手实践甚至“折腾出错”，正是最好的学习方式。

## 研究框架 (Research)

- **PyTorch** ([PyTorch - Fireship 视频](https://www.youtube.com/watch?v=ORMx45xqWkA&t=0s&ab_channel=Fireship))
    - 如果你正在学习本课程，我假定你已经掌握了 PyTorch 的基本知识。如果尚未了解，建议观看 [Daniel Bourke 的 PyTorch 入门视频](https://www.youtube.com/watch?v=Z_ikDlimN6A)。
    - PyTorch 分为 Nightly（预览版）和 Stable（稳定版）两种发布版本 ⇒ [参考讨论](https://discuss.pytorch.org/t/pytorch-nightly-vs-stable/105633)
        
        Nightly 版本可能不够稳定，但提供了最新的 PyTorch 特性以及前沿的框架级优化。
        
    - 开发者普遍更倾向于 PyTorch，很大程度上得益于其与 Hugging Face 极佳的易用性配合。
    - 你可以在 torchvision（`pip install torchvision`）和 `torch.hub` 中找到预训练模型。PyTorch 生态在获取预训练模型上采用了一种更为去中心化、但寻找起来稍显零散的模式：大家通常直接在 GitHub 仓库发布模型，而不是推送到统一的中央模型库。得益于开源社区的努力，目前 Hugging Face 是最主流的模型集散地。
    - 拥有良好的 ONNX 导出支持。

- **TensorFlow** ([TensorFlow - Fireship 视频](https://www.youtube.com/watch?v=i8NETqtGHms))
    - 文档齐全，社区支持庞大。曾经是使用最广泛的深度学习框架。
    - 相比之下是运行效率较慢的深度学习框架。
    - 由 Google 开发（最初针对 TPU 优化设计），兼具通用机器学习能力（SVM、决策树等）。
    - 预训练模型可直接在官方库中获取 ⇒ https://www.tensorflow.org/resources/models-datasets
    - 支持用 1-3 行代码直接下载并加载预训练模型。
    - 对 ONNX 支持相对有限（需借助 `tf2onnx`）。

- **Keras**
    - 类似于 TensorFlow 中的 `torch.nn`，但抽象层级更高。
    - 虽然是独立库，但与 TensorFlow 深度集成，作为其主要的高级 API。
    - 是一个用于构建和训练完整模块的完备框架，而不仅限于构建神经网络层。

- **JAX** ([JAX - Fireship 视频](https://www.youtube.com/watch?v=_0D5lXDjNpw))
    - 即时编译（JIT）、自动求导（Autograd）以及基于 XLA（加速线性代数）的高性能计算库。
    - 官方文档 ⇒ https://jax.readthedocs.io/en/latest/
    - API 风格与 NumPy 极其相似。
    - Reddit 上关于 JAX 的讨论与评价 ⇒ https://www.reddit.com/r/MachineLearning/comments/1b08qv6/d_is_it_worth_switching_to_jax_from/
    - JAX 和 TensorFlow 均由 Google 主导开发。
    - 底层采用 XLA 编译器。
    - 支持通过 `tf2onnx` 转换。

- **MLX**
    - 苹果公司专门针对 Apple Silicon（M 系列芯片）开发的开源机器学习框架。
    - 专注于在 Apple 设备上实现高能效、高性能的机器学习计算。
    - 同时兼顾训练与推理设计。
    - 针对苹果 Metal GPU 架构深度优化。
    - 支持动态计算图。
    - 非常适合在 Mac 上进行前沿机器学习模型的研究与开发。

- **PyTorch Lightning**
    - 社区讨论：[为什么使用或不使用 PyTorch Lightning](https://www.reddit.com/r/deeplearning/comments/t31ppy/for_what_reason_do_you_or_dont_you_use_pytorch/)
    - 核心优势在于消除繁琐的样板代码并提供开箱即用的分布式扩展能力。
    - 使用统一的 `Trainer()` 替代手动编写的训练循环。

## 生产部署 (Production)

- **纯推理引擎 (Inference-only)**
    - **vLLM**
        - 高吞吐量且低延迟的大语言模型（LLM）推理与部署服务库，采用 PagedAttention 等核心显存优化技术。
    - **TensorRT**
        - 能够与 PyTorch 深度集成用于高性能模型推理。
        - 支持加载 ONNX 模型。
        - 包含针对以下方向高度优化的 CUDA 核函数：
            - 利用网络稀疏性（Sparsity）加速计算
            - 推理量化（INT8/FP8/FP4 等）
            - 针对特定 GPU 硬件微架构深度调优
            - 优化全局显存（VRAM）与片上片内缓存之间的访存模式
        - 全称为 Tensor RunTime。
        - 由 NVIDIA 自研、设计并维护。
        - 专为极致推理性能而生（包括 TensorRT-LLM）。
        - 运用了本课程中涉及的大量底层技术，但在上层进行了易用性封装。
        - 推荐快速浏览学习路线：
            - https://nvidia.github.io/TensorRT-LLM/
            - https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
            - https://pytorch.org/TensorRT/getting_started/installation.html#installation

- **Triton (语言与编译器)**
    - 由 OpenAI 开发并维护 ⇒ https://openai.com/index/triton/
    - 类似 CUDA 的底层编程体验，但完全基于 Python。消除了使用传统 CUDA C/C++ 开发核函数时的大量样板与繁琐细节，在矩阵乘法等任务上能够媲美甚至超越顶尖专家的极致性能。
    - 快速入门 ⇒ https://triton-lang.org/main/index.html
    - 编写你的第一个 Triton 核函数 ⇒ https://triton-lang.org/main/getting-started/tutorials/index.html
    - Triton 开创性论文 ⇒ https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
    - `triton-viz` 是 Triton 主要的性能分析与可视化工具包。
    - 让你在 Python 中精细控制 GPU 底层行为，而无需面对 C/C++ 中诸多的意外陷阱与复杂性：
        - 免除显式手动内存管理（如 `cudaMalloc`、`cudaMemcpy`、`cudaFree`）
        - 无需反复编写错误检查宏（如 `CUDA_CHECK_ERROR`）
        - 大幅简化了核函数启动参数中 Grid / Block / Thread 级别的索引计算复杂度。

- **Triton Inference Server**
    - 由 NVIDIA 开发并维护 ⇒ https://developer.nvidia.com/triton-inference-server
    - 开源的模型推理服务框架，用于在生产环境中实现快速、可扩展的多模型 AI 推理部署。
    - *注意：与 OpenAI 的 Triton 语言没有任何直接技术关联（仅重名）。*

- **torch.compile**
    - 比老一代的 TorchScript 获得更多关注，并且通常具有更好的执行性能。
    - 将模型计算图编译为静态中间表示（IR），使得底层无需担心 PyTorch 动态图特性带来的动态变化开销。直接将模型编译为高度优化的机器码/内核二进制文件执行。
    - 深入探讨 ⇒ [TorchScript 与 torch.compile 的差异](https://discuss.pytorch.org/t/the-difference-between-torch-jit-script-and-torch-compile/188167)

- **TorchScript**
    - 在特定场景下能带来显著提速，尤其是在纯 C++ 环境下部署时。
    - 性能收益高度依赖于具体的神经网络架构设计。

- **ONNX Runtime**
    - 视频介绍 ⇒ https://youtu.be/M4o4YRVba4o
    - “**ONNX Runtime 训练**：只需在现有的 PyTorch 训练脚本中添加一行代码，即可在多节点 NVIDIA GPU 上显著加速 Transformer 模型的训练速度。”
    - 由微软开发并开源维护。

- **Detectron2**
    - 同时支持目标检测与分割模型的训练与推理。
    - 由 Facebook AI Research (Meta) 开源的计算机视觉研究平台。

## 底层技术 (Low-Level)

- **CUDA**
    - 全称为 Compute Unified Device Architecture（统一计算设备架构），可以理解为面向 NVIDIA GPU 的并行计算平台与底层编程语言扩展。
    - 核心加速库 ⇒ cuDNN（深度神经网络）、cuBLAS（基础线性代数）、CUTLASS（高性能线性代数模板库）、cuFFT（快速傅里叶变换，可用于快速卷积计算，本课程中会有所涉及）。
    - 开发者可以根据具体的 GPU 硬件微架构亲自手写核函数（NVIDIA 官方库在底层也是通过向编译器传递专属硬件优化标志来实现的）。

- **ROCm**
    - AMD 推出的开源 GPU 计算平台，等价于 AMD 版的 CUDA。

- **OpenCL**
    - 全称为 Open Computing Language（开放计算语言）。
    - 跨平台异构计算标准，支持 CPU、GPU、数字信号处理器（DSP）等多种硬件。
    - 由于 NVIDIA 专门为其硬件量身设计了 CUDA，因此在 NVIDIA 硬件上 CUDA 的性能通常显著优于 OpenCL。如果你从事嵌入式系统开发或电子/计算机工程（EE/CE），OpenCL 仍非常值得学习。

## 边缘计算与嵌入式系统的推理

- **边缘计算 (Edge Computing)** 是指在分布式真实场景（如车载智能车队、边缘设备）中进行低延迟、高能效的本地实时计算。特斯拉的 FSD（完全自动驾驶）就是边缘计算的典型代表：神经网络必须在车载计算芯片上实时运行，同时还需要将关键数据回传服务器以迭代改进模型。

- **CoreML**
    - 主要用于在苹果全系设备上部署预训练机器学习模型。
    - 专为移动端本地硬件推理高度优化。
    - 支持端侧增量训练。
    - 支持极其广泛的模型类型（计算机视觉、自然语言处理、语音处理等）。
    - 与 Apple 生态（iOS、macOS、watchOS、tvOS）深度无缝整合。
    - 重视用户隐私，数据完全保留在设备本地处理。
    - 支持从主流框架模型（PyTorch、TensorFlow 等）进行格式转换。
    - 极大降低了普通应用开发者在 App 中集成 ML 能力的门槛。

- **PyTorch Mobile**
- **TensorFlow Lite**

## 易用型工具库与中间件

- **FastAI**
    - 高阶 API：构建在 PyTorch 之上，为常见的深度学习任务提供了极为友好简洁的高层抽象接口。
    - 极速原型验证：专为快速实验和复现 SOTA（前沿顶尖）模型设计。
    - 最佳实践：默认集成了深度学习领域的诸多前沿技巧与工程最佳实践。
    - 精炼的代码量：相比原生 PyTorch，通常只需极少的代码即可完成复杂模型的构建与训练。
    - 迁移学习：开箱即用的一流迁移学习支持。

- **ONNX (Open Neural Network eXchange)**
    - 开放神经网络交换格式，旨在实现不同深度学习框架之间的模型互通。
    - PyTorch 导出示例：`torch.onnx.export(model, dummy_input, "resnet18.onnx")`
    
    ```python
    import tensorflow as tf
    import tf2onnx
    import onnx
    
    # 加载 TensorFlow 模型
    tf_model = tf.keras.models.load_model('path/to/your/model.h5')
    
    # 将模型转换为 ONNX 格式
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model)
    
    # 保存 ONNX 模型文件
    onnx.save(onnx_model, 'path/to/save/model.onnx')
    ```
    
    ![ONNX](assets/onnx.png)

- **Weights & Biases (WandB)**
    - 实验跟踪与可视化神器。
    - 只需几行代码即可轻松接入已有项目。
    - 优秀的团队协作支持。
    - 直观易用的 Web UI，方便对比各实验轮次的超参数与指标曲线。
    
    ![WandB](assets/wandb.png)

## 云算力提供商 (Cloud Providers)

- **AWS**
    - EC2 GPU 实例
    - SageMaker（集群 Jupyter 笔记本、数据标注、模型训练及云上一键部署）
- **Google Cloud (GCP)**
    - Vertex AI
    - 各种搭载 GPU/TPU 的虚拟机实例
- **Microsoft Azure**
    - DeepSpeed 算力支持与认知服务
- **OpenAI**
- **Vast.ai**
    - 高性价比的去中心化 GPU 算力租赁市场
- **Lambda Labs**
    - 提供极具性价比的数据中心级 GPU 算力云服务

## 编译器 (Compilers)

- **XLA (Accelerated Linear Algebra)**
    - 专用于线性代数的领域特定编译器（DSL Compiler），可深度优化 TensorFlow 计算图。
    - 为 JAX 提供底层的代码生成与硬件优化后端。
    - 执行全图级别优化（Whole-program optimization），突破单算子的局限，跨越整个计算图发掘优化空间。
    - 通过生成极致优化的机器码，实现跨多种异构硬件（CPU、GPU、TPU）的高效执行。
    - 实现算子融合（Operator Fusion）等高级优化，将多个细粒度算子合并为单个高效的计算核函数，大幅削减显存读写瓶颈。
    - 使 JAX 开发者无需手写特定硬件的底层汇编或 CUDA 代码，即可坐享极致计算性能。

- **LLVM**
- **MLIR**
- **NVCC**
    - NVIDIA CUDA 编译器（Nvidia CUDA Compiler）。
    - 编译处理 CUDA Toolkit 中的所有 C++/CUDA 代码。
    
    ![NVCC](../11_Extras/assets/nvcc.png)

## 其他资源

- **Hugging Face**：开源 AI 模型的最大中心枢纽，包含 Transformers、Datasets、Diffusers 等核心库。
