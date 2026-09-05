# cuBLASLt 矩阵乘法

- 在最初测试 cuBLASLt 时，如果为了快速验证而把矩阵尺寸设置得很小，可能会遭遇报错。
- 查阅 [cuBLASLt 文档](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul) 可以发现关键约束：**“Dimensions m and k must be multiples of 4”**（维度 $m$ 和 $k$ 必须是 4 的整数倍）。
- 这意味着不能使用 $3 \times 4$ 或 $2 \times 4$ 的矩阵，但可以使用 $4 \times 4$ 或 $4 \times 8$ 的矩阵。

## 编译与运行指南

- 使用 `nvcc` 编译器进行编译，需显式链接 `cublasLt`、`cublas` 和 `cuda`：
  ```sh
  nvcc -o matmul main.cu -lcublasLt -lcublas -lcuda
  ```
- 运行生成的可执行程序：
  ```sh
  ./matmul
  ```
  （其中 `main.cu` 为当前示例源码文件名）
