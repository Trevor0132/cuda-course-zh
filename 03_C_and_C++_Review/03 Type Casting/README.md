### a. `static_cast<new_type>(expression)`

示例：
```cpp
double pi = 3.14159;
int rounded_pi = static_cast<int>(pi);
```

**解析：**  
`static_cast` 用于类型之间的隐式转换。它是最常用的类型转换方式，并在编译时进行检查。在上面的示例中，它将 `double` 转换为 `int`，截断了小数部分。

---

### b. `dynamic_cast<new_type>(expression)`

示例：
```cpp
class Base { virtual void foo() {} };
class Derived : public Base { };

Base* base_ptr = new Derived;
Derived* derived_ptr = dynamic_cast<Derived*>(base_ptr);
```

**解析：**  
`dynamic_cast` 用于继承体系中安全的向下转型（Downcasting）。它会在运行时执行类型检查；如果转换不合法，则返回 `nullptr`（对于指针）或抛出异常（对于引用）。它要求基类中至少包含一个虚函数。

---

### c. `const_cast<new_type>(expression)`

示例：
```cpp
const int constant = 10;
int* non_const_ptr = const_cast<int*>(&constant);
*non_const_ptr = 20; // 修改只读常量变量（未定义行为）
```

**解析：**  
`const_cast` 用于添加或移除变量的 `const`（或 `volatile`）修饰符。它是唯一能够执行该操作的 C++ 风格类型转换。然而，修改原本定义为 `const` 的对象属于未定义行为（Undefined Behavior）。

---

### d. `reinterpret_cast<new_type>(expression)`

示例：
```cpp
int num = 42;
char* char_ptr = reinterpret_cast<char*>(&num);
```

**解析：**  
`reinterpret_cast` 是最危险的类型转换。它可以在完全不相关的类型之间进行位级别的重新解释转换（例如将指针转换为整型，反之亦然）。它常用于底层操作和硬件交互，使用时必须极其谨慎。
