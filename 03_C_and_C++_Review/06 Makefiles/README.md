### Makefiles 基础语法

```make
targets: prerequisites
    bash command
    possibly another bash command?
```

### CMakeLists.txt 的作用是什么？

CMake 是一个**生成 Makefile 的工具**。它本身是一个构建系统生成器（Build System Generator），用于构建、测试和打包软件。它跨平台运行，旨在通过简单、与平台和编译器无关的配置文件来控制软件的编译构建流程。

### `.PHONY` 的作用是什么？

假设我们在 Makefile 中定义了一个名为 `clean` 的目标：

```make
clean:
    rm -rf build/*
```

如果在 Makefile 所在的同级目录下恰好存在一个名为 `clean` 的目录或文件，当我们运行 `make clean` 时，make 会检测到该文件/目录已经存在且是最新的，从而**跳过**执行目标下的命令。

简而言之，Makefile 本质上是在建立目标名称与命令之间的映射关系。如果当前目录下有同名文件或目录，make 可能会误判而不执行命令。这正是引入 `.PHONY` 的原因——显式声明目标为“伪目标”：

```make
.PHONY: clean
clean:
    rm -rf build/*
```

### Makefiles 中的 `:=` 与 `=` 有何区别？

- `=` 用于变量定义，称为**递归展开赋值（Recursive Assignment）**。每次使用该变量时，都会重新计算求值其引用的表达式。
- `:=` 用于变量定义，称为**简单赋值 / 立即展开赋值（Simple Assignment / Immediate Assignment）**。仅在定义时立即计算求值一次，后续保持不变。

示例：
```make
A = $(B)
B = hello
C := $(B)
B = world

all:
    @echo A is $(A)  # 输出: A is world
    @echo C is $(C)  # 输出: C is hello
```

### Makefile 中 `@` 符号的作用是什么？

在命令前添加 `@` 符号，可以防止执行 Makefile 时在控制台中回显（打印）该命令本身。

示例：
```make
clean:
    rm -rf build/*
```

```bash
$ make clean
rm -rf build/*
```

使用 `@` 隐藏命令本身：
```make
clean:
    @rm -rf build/*
```

```bash
$ make clean
$
```
