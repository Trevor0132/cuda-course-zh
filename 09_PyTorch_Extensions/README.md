# 自定义 PyTorch C++/CUDA 扩展

## 安装自定义扩展

```bash
python setup.py install 
```

## 什么是 `scalar_t` 类型？
- 可以将其理解为 PyTorch CUDA 张量内部元素的标量数据类型。
- PyTorch 的调度分发宏（如 `AT_DISPATCH_FLOATING_TYPES`）会在编译时将 `scalar_t` 安全地实例化为具体的硬件类型（如 `float` / FP32 或 `double` / FP64）。

## 为什么在指针前使用 `__restrict__` 关键字？

`__restrict__` 是对编译器的承诺，表明通过该指针访问的内存区域，在当前生命周期内**绝不会与任何其他指针指向的内存区域发生重叠（No Pointer Aliasing）**。

如果不加 `__restrict__`，编译器必须考虑指针别名（指针可能指向同一块数组并存在重叠部分）的最坏情况，因而无法激进地优化指令排布：

```cpp
// 演示指针重叠导致的潜在数据依赖：
void add_arrays(int* a, int* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] + b[i];
    }
}

int main() {
    int data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 内存重叠调用：b 指针偏移了 3 个位置
    add_arrays(data, data + 3, 7);
    
    // 打印结果
    for (int i = 0; i < 10; i++) {
        printf("%d ", data[i]);
    }
    return 0;
}
```

```python
# 数组 'data' 初始状态:
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 内存布局可视化:
#  a (data)     b (data + 3)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#  ^        ^
#  |        |
#  a[0]     b[0]

# 当 i = 0 时: data[0] = data[0] + data[3] (1 + 4 = 5)
[5, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 当 i = 1 时: data[1] = data[1] + data[4] (2 + 5 = 7)
[5, 7, 3, 4, 5, 6, 7, 8, 9, 10]

# 当 i = 2 时: data[2] = data[2] + data[5] (3 + 6 = 9)
[5, 7, 9, 4, 5, 6, 7, 8, 9, 10]

# 当 i = 3 时: data[3] = data[3] + data[6] (4 + 7 = 11)
[5, 7, 9, 11, 5, 6, 7, 8, 9, 10]

# 当 i = 4 时: data[4] = data[4] + data[7] (此时 data[4] 已经被前面的迭代修改为 13！)
[5, 7, 9, 11, 13, 6, 7, 8, 9, 10]

# 当 i = 5 时: data[5] = data[5] + data[8]
[5, 7, 9, 11, 13, 15, 7, 8, 9, 10]

# 当 i = 6 时: data[6] = data[6] + data[9]
[5, 7, 9, 11, 13, 15, 17, 8, 9, 10]

# 最终输出结果:
data = [5, 7, 9, 11, 13, 15, 17, 8, 9, 10]
```

> 添加 `__restrict__` 告诉编译器传入的指针之间绝不存在重叠，使得编译器能够大胆地将数据缓存在寄存器中，并实施激进的向量化和指令乱序重排优化。

## PyTorch 绑定 (Pybind11)

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polynomial_activation", &polynomial_activation_cuda, "多项式激活函数 (CUDA)");
}
```

本部分使用 Pybind11 为 C++/CUDA 扩展构建 Python 导入模块：
- `PYBIND11_MODULE`：宏定义，声明 Python 扩展模块的入口。
- `TORCH_EXTENSION_NAME`：PyTorch 自动注入的宏，展开为扩展模块的名字（通常与 `setup.py` 中定义的名称对应）。
- `m`：正在构建的 Python 模块对象。
- `m.def()`：向模块注册导出函数：
  - 第 1 个参数 `"polynomial_activation"`：在 Python 中调用的函数名。
  - 第 2 个参数 `&polynomial_activation_cuda`：指向底层 C++ 包装函数的指针。
  - 第 3 个参数：该函数的 Python 文档字符串（Docstring）。

- **编译缓存**：即时编译生成的二进制文件通常缓存于 `~/.cache/torch_extensions/` 目录下（如需排查编译缓存问题可清空此目录）。

---

## 推荐学习资料
- [PyTorch 官方 C++ 扩展示例仓库](https://github.com/pytorch/extension-cpp)
- [PyTorch 官方教程：C++ 自定义算子](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [PyTorch 官方教程：定制 C++ 与 CUDA 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [PyTorch 官方文档：扩展 PyTorch 的权威笔记](https://pytorch.org/docs/stable/notes/extending.html)
