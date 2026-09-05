# 01

# 02
## `#pragma once` 的作用是什么？
- 维基百科参考 ⇒ [Pragma once](https://en.wikipedia.org/wiki/Pragma_once#:~:text=In%20the%20C%20and%20C,once%20in%20a%20single%20compilation)
- 添加 `#pragma once` 可以确保头文件在单次编译中仅被包含（include）一次。否则在下面的示例中会出现 `error: redefinition of 'foo'`（重复定义错误）。

`grandparent.h`
```cpp
// #pragma once

struct foo 
{
    int member;
};
```

`parent.h`
```cpp
#include "grandparent.h"
```

`child.h`
```cpp
#include "grandparent.h"
#include "parent.h"

int main() {
    int member;
}
```