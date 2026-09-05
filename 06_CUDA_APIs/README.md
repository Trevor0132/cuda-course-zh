# CUDA API 编程指南
> 涵盖 cuBLAS、cuDNN、cuBLASMp 等

- 初学者可能会对“API”这个词产生困惑。在 CUDA 生态中，API 通常意味着我们使用的是预编译的闭源加速库（无法直接查看内部源码）。官方提供了完备的函数调用文档，但它们作为高度优化的机器二进制动态库分发。虽然底层实现细节被封装，但其计算性能极高。

## 不透明结构体类型 (Opaque Struct Types)
- 在 CUDA API 中，很多类型是“不透明”的，即开发者无法直接查看或修改结构体内部的具体字段，只能通过 API 接口传递指针（例如句柄 handle、描述符 descriptor）。这些 API 通常以 `.so`（Linux 动态链接库）分发。例如 `cublasLtHandle_t` 就是一个不透明句柄类型，用于保存 cuBLASLt 操作所需的上下文环境。

要快速查阅与掌握 CUDA API，推荐以下工具与方法：
1. [Perplexity.ai](http://perplexity.ai)（检索最新 API 变更与用法示例）
2. 搜索引擎查找技术博客与社区讨论
3. 大语言模型（如 ChatGPT / Claude / Gemini）辅助理解概念
4. 查阅 NVIDIA 官方文档中的关键字索引

---

## 错误检查宏 (API 专属)

在调用 CUDA API 时，必须编写状态检查宏以捕获错误：

- **cuBLAS 错误检查示例**：
```cpp
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS 错误于 %s:%d, 错误码: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- **cuDNN 错误检查示例**：
```cpp
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN 错误于 %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

**错误检查的必要性**：
在配置上下文并调用 CUDA API 后，将 API 函数包裹在宏的 `call` 参数中。如果调用成功，程序继续平稳执行；如果失败，宏会打印出错文件、行号以及人类可读的详细错误信息，避免静默失败或直接发生段错误（Segmentation Fault）。

> 推荐阅读：[规范的 CUDA 错误检查实践](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/)

---

## 矩阵乘法支持对比

- **cuDNN**：通过特定的卷积及深度学习算子在内部隐式支持矩阵乘法，但矩阵乘法并不是 cuDNN 的核心主打功能。
- **cuBLAS**：进行高吞吐量、生产级矩阵乘法运算的首选库，涵盖极其全面的 BLAS 操作，针对各类矩阵维度与数据类型均做了极致的微架构优化。

## 参考资源
- [NVIDIA 官方 CUDA 库示例代码集](https://github.com/NVIDIA/CUDALibrarySamples)