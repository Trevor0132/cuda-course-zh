## 用于 C/C++ 的 GDB 调试器（视频课程未涵盖内容）

> 安装命令 -> `sudo apt install gdb`

推荐观看教程：https://www.youtube.com/watch?v=Dq8l1_-QgAc

### 常用命令
- `run` 或 `r`：从头到尾执行程序。
- `break` 或 `b`：在指定行或函数处设置断点。
- `disable`：禁用指定断点。
- `enable`：重新启用已禁用的断点。

- `next` 或 `n`：单步执行下一行 C 代码（Step Over，不进入函数内部）。
- `nexti`：单步执行下一条汇编指令。
- `step` 或 `s`：单步执行下一行代码；如果下一行是函数调用，则进入该函数内部并停在第一行（Step Into）。
- `stepi`：单步执行下一条汇编指令；如果该指令是函数调用，则进入该函数并在第一条汇编指令处暂停。

- `list` 或 `l`：查看当前作用域周围的源代码。
- `print` 或 `p`：打印变量或表达式的值。
- `quit` 或 `q`：退出 GDB。
- `clear`：清除断点。
- `continue` 或 `c`：继续执行程序，直到遇到下一个断点。

> 值得注意的是：`gdb` 用于调试标准的 C/C++ 程序，而调试 CUDA 程序则需要使用专门的 `cuda-gdb`。
