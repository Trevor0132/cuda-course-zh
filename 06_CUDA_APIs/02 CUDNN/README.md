# cuDNN 深度神经网络加速库

从技术上讲，要实现一个 GPT 模型的训练与推理，并不一定需要 cuFFT 或手写成百上千行自定义核函数。cuDNN 内部已经封装了极致优化的卷积运算，并且在高层抽象上包含了基于 cuBLAS 的矩阵乘法。然而，深入理解朴素卷积 vs 快速卷积、朴素矩阵乘 vs 快速矩阵乘的底层原理仍然是极具价值的功底。

NVIDIA cuDNN 为深度学习应用中高频出现的计算操作提供了高度优化的工业级实现：

- **正向与反向卷积**（包括互相关计算）
- **通用矩阵乘法 (GEMM)**
- **正向与反向池化 (Pooling)**
- **正向与反向 Softmax**
- **激活函数**：ReLU、Tanh、Sigmoid、ELU、GELU、Softplus、Swish 等点对点算术、数学与逻辑操作
- **张量变形与重排**（Reshape、Transpose、Concat 等）
- **归一化算子**：LRN、LCN、Batch Normalization、Instance Normalization 及 Layer Normalization 的前向与反向

除了为单一算子提供极致性能外，cuDNN 还支持灵活的多算子融合模式（Fusion Patterns），以消除不必要的全局显存往返读写，在 NVIDIA GPU 上压榨出顶尖性能。

---

## Legacy API vs Graph API

- 在 cuDNN v7 及更早版本中，API 围绕一组固定的预设操作与融合模式展开，我们称之为**传统 API（Legacy API）**。
- 从 cuDNN v8 开始，为了支持深度学习领域爆发式涌现的新型融合需求，NVIDIA 引入了 **[Graph API](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html#graph-api)**。用户可以通过定义“计算图（Operation Graph）”来表达多算子组合，而无需受限于固定的单一 API 接口。对于绝大多数现代用例，Graph API 是官方推荐的使用方式。
- **注意**：这里的“Graph API”指的是将一系列计算算子组织为有向无环计算图，与“图神经网络（GNN）”无直接关联。

---

## 核心描述符类型与概念

cuDNN 采用大量不透明结构体类型（Opaque Types）来管理计算上下文与张量元数据：

- `cudnnHandle_t`：cuDNN 上下文句柄
- `cudnnTensorDescriptor_t`：张量描述符（定义维度、步长、数据类型等）
- `cudnnConvolutionDescriptor_t`：卷积算子描述符（定义步长、填充、空洞等）
- `cudnnFilterDescriptor_t`：卷积核权重描述符
- `cudnnConvolutionFwdAlgo_t`：前向卷积算法类型枚举

以经典的前向卷积函数调用为例：

```cpp
cudnnConvolutionForward(
    cudnn,                  // cudnnHandle_t: 上下文句柄
    &alpha,                 // 指向标量系数 alpha 的指针
    inputDesc,              // 输入张量描述符
    d_input,                // 设备端显存中的输入数据指针
    filterDesc,             // 卷积核描述符
    d_kernel,               // 设备端显存中的卷积核权重指针
    convDesc,               // 卷积算法描述符
    algo,                   // 选择的具体前向算法枚举
    workspace,              // 算法执行所需的临时显存工作空间 (Workspace)
    workspaceSize,          // 工作空间字节大小
    &beta,                  // 指向标量系数 beta 的指针
    outputDesc,             // 输出张量描述符
    d_output_cudnn          // 设备端显存中的输出数据指针
);
```

### 内存布局与数据排布 (NCHW)
在 PyTorch 中，一个多维张量可能展示为一个三维或四维的嵌套结构：
```python
# 形状: (4, 2, 3) -> 批大小 Batch=4, 通道/高度=2, 宽度=3
tensor([[[-1.7182,  1.2014, -0.0144],
         [-0.6332, -0.5842, -0.7202]],
        ...])
```
但在 GPU 显存底层，数据始终是一维连续排布的浮点数组：
```python
[-1.7182, 1.2014, -0.0144, -0.6332, -0.5842, -0.7202, ...]
```
只要你在 `cudnnSetTensor4dDescriptor` 中准确指定了维度（例如 NCHW 格式：Batch Size、Channels、Height、Width）与内存步长（Strides），cuDNN 底层就能精准映射与解析连续显存。

---

## 四大计算引擎类型

1. **预编译单操作引擎 (Pre-compiled Single Operation Engines)**：
   - 针对特定单一算子预先编译并深度固化。执行效率极高，但灵活性较低。
2. **通用运行时融合引擎 (Generic Runtime Fusion Engines)**：
   - 在运行时动态将多个逐元素算子融合为单个核函数，避免中间结果写回全局显存。
3. **特化运行时融合引擎 (Specialized Runtime Fusion Engines)**：
   - 针对深度学习中常见的高频范式（如卷积后紧跟偏置加法与 ReLU 激活）进行了微架构级别的模式特化调优。
4. **特化预编译融合引擎 (Specialized Pre-compiled Fusion Engines)**：
   - 针对高频的算子链路（如 Conv + BatchNorm + ReLU）提供开箱即用、预编译好的极致性能核函数。

### 运行时算子融合的威力
假设需要计算：
```python
output = torch.sigmoid(tensor1 + tensor2 * tensor3)
```
- **无融合时**：乘法、加法、Sigmoid 分别启动 3 次核函数，产生 2 次冗余的全局显存中间写入与重新读取。
- **算子融合后**：单次核函数发射，全部计算在线程私有寄存器中连续完成，仅在最后一步将最终结果写入全局显存，速度大幅飙升。

![](../assets/knlfusion1.png)
![](../assets/knlfusion2.png)

---

## API 架构四层划分

1. **Graph API**：基于计算图构建的核函数融合引擎（节点为算子，边为张量数据流）
2. **Ops API**：单一核心算子引擎（Softmax、BatchNorm、Dropout 等）
3. **CNN API**：卷积与池化运算（Graph API 底层依赖的基础组件）
4. **Adversarial / Specialized API**：序列模型与专用网络（RNN、CTC Loss、Multi-Head Attention 等）

## 函数声明速查

```cpp
cudnnConvolutionForward(
    cudnnHandle_t                       handle,
    const void                         *alpha,
    const cudnnTensorDescriptor_t       xDesc,
    const void                         *x,
    const cudnnFilterDescriptor_t       wDesc,
    const void                         *w,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionFwdAlgo_t           algo,
    void                               *workSpace,
    size_t                              workSpaceSizeInBytes,
    const void                         *beta,
    const cudnnTensorDescriptor_t       yDesc,
    void                               *y
);
```