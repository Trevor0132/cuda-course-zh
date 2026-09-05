# cuFFT 快速笔记（选学内容）

- **cuFFT**：主要用于频域快速卷积以及音频/雷达信号预处理。
- FFTW（Fastest Fourier Transform in the West，[https://www.fftw.org](https://www.fftw.org/)）是 CPU 端最经典通用的 C 语言 FFT 库。cuFFT 在设计哲学上继承了 FFTW 的接口范式，但底层完全针对 NVIDIA GPU 的高度并行架构进行了重写与硬件级加速。

### 卷积与频域转换
- 常见的三种卷积模式：
  - `full`（完全卷积）：`output_size = input_size + kernel_size - 1`（常用于 1D 信号处理）
  - `valid`（有效卷积）：`output_size = input_size - kernel_size + 1`（深度学习 Conv2D 中最常用，无边界填充时）
  - `same`（同尺寸卷积）：`output_size = input_size`（通过补零保持输入输出尺寸一致）
- 利用卷积定理，时域卷积等价于频域逐元素乘积：
  $$\text{conv\_out} = \text{IFFT}(\text{FFT}(x) \odot \text{FFT}(w))$$
  对于超大卷积核，通过 cuFFT 转换到频域计算往往比直接在时域滑窗卷积更加高效。

- **离散傅里叶变换 (DFT)  직관理解**：
  - 例如序列 `[1, -1, 1, -1]`：该序列每 2 个采样点反转一次，具有明显的周期性。在进行 4 点 DFT 时，能量会完全集中在对应数字频率 $k = 2$ 的频槽（Frequency Bin）上，输出呈现为 `[0, 0, 4, 0]`（其中峰值为 $N=4$）。
  - 参考资料：[维基百科：离散傅里叶变换](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) 与 [快速傅里叶变换 (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform)。