# 此文档用来记录C++学习过程中关键点

# 0 感觉有用但没有仔细看的章节

* 所有C++11的变量初始化
* `4.5 & 4.6` : 共用体和枚举
* `4.10.2` : 模板类array(C++11)
* `6.8` : 简单文件I/O
* `7.10.3` : 深入探讨函数指针
* `8.2.4 & 8.2.5` : 类与对象使用引用
* `8.5.6` : C++ 11对于函数模板的新标准
* `9.2` : 存储连续性，作用域和链接性
* `11.5` : 为矢量重载运算符
* `11.6` ：`explicit`类的类型转换

***

# 1 一些基础知识

* C++在C的基础上添加了面向对象编程和泛型编程的支持

## 1.1 名空间

* 基础知识

之前学习C++时会见到形如这样的语句，当时不知道什么意思
```cpp
using namespace std;
```
。
一个比较简答的例子，当头文件使用而非`iostream.h`时，就需要指定std标准库名空间，才能找到标准库的对应功能和组件。名空间的封装便于进行不同版本功能的控制。
```cpp
#include <iostream>
```

若不使用`using namespace std`，那就必须使用如下形式调用：
```cpp
std::cout<<"Hello";
std::cout<<std::endl;
```

但在大型项目上直接加一句`using namespace std`显然是不合理的，更好的做法是只声名特定的组件：
```cpp
using std::cout;
using std::cin;
using std::endl;
```

* 定义一个名空间

头文件中的声明：
```cpp
namespace pers
{
    struct Person
    { 
        std::string fname;
        std::string lname;
     };
    void getPerson(Person &);
    void showPerson(const Person &);
}
```

源文件中示例化：
```cpp
namespace pers
{
    using std::cout;
    using std::cin;
    void getPerson(Person & rp)
    {
        cout << "Enter first name: ";
        cin >> rp.fname;
        cout << "Enter last name: ";
        cin >> rp.lname;
    }
    
    void showPerson(const Person & rp)
    {
        std::cout << rp.lname << ", " << rp.fname;
    }
}
```

* 名空间使用原则

1. 使用在已命名空间中声明的变量，而不是使用外部全局变量或静态全局变量
2. 如果开发了一个函数库或者类库，将其放在一个命空间中
3. 对于`using`声明，首选将其作用域设置为局部

示例：
```cpp
#include <iostream>
namespace GlobalVariables 
{
    int counter = 0; // 命名空间中的全局变量
}

void increment() 
{
    GlobalVariables::counter++; // 访问命名空间中的变量
    std::cout << "Counter: " << GlobalVariables::counter << std::endl;
}

int main() 
{
    increment(); // 输出: Counter: 1
    increment(); // 输出: Counter: 2
    return 0;
}
```
全局变量的使用是非常不稳定的，容易出现很多问题，而名空间的限定可以很好地解决问题

## 1.2 数据类型

### 1.2.1 变量声明

* 使用`__`和大写字母加下划线`_A`的变量声明在任何情况下禁止使用
* 使用`_`的变量声明禁止在全局（包括命名空间作用域、全局变量、全局函数、类/命名空间名）使用

## 1.2.2 连续赋值

与C语言不同，C++支持连续使用`=`进行赋值：
```cpp
int a;
int b;
a = b = 1;
```

### 1.2.3 整数类型

* `short`至少16位
* `int`至少和`short`一样长
* `long`至少32位，且至少和`int`一样场
* `long long`至少64位，且至少和`long`一样长

变量选择标准：
* 一般使用`int`,若超出16位则使用`long`，为了移植16位系统上时也能工作。
* `short`可以节省内存，一般在大型整形数组中使用

### 1.2.4 `const`限定符

使用`const`限定符限制变量不能更改：
```cpp
const int Month = 12;
```

### 1.2.5 浮点数类型

* `float`通常为32位
* `double`通常为64位
* `long double`为96位或128位

### 1.2.6 运算符重载

针对不同类型数据的除法，除法运算符会自动进行重载，如下图所示：
![divide_overload](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/divide_overload.png)

## 1.3 复合数据类型

### 1.3.1 数组

* 初始化
```cpp
int cards[4] = {1,2,3,4}; //可行
int hands[4]; //可行
int eggs[4] = {} //将全部设置为0
hands[4] = {5,6,7,8};//数组一旦被赋值，不能再使这种方法更改，需要逐个操作数组元素更改
```

### 1.3.2 字符串

```cpp
char XB[20] = "hello,XB"; 
```
C++中字符串的终止以'\0'控制；`strlen(XB)`返回字符串的有效长度8（不把'\0'计算在内），而使用`sizeof(XB)`则直接返回数组大小20

* `cin.get()`的使用

由于`cin`使用空白（空格，制表符，换行符）确定字符串的结束位置，因此在输入带空格的字符串时会出现问题。
这里引入`cin.get()`的`get()`成员函数，注意此函数存在函数重载：
```cpp
cin.get(string,size); // 将以换行符为控制符读取整行的字符串，但是不包括换行符
cin.get(); // 强制读取一个字符
```

因此可以使用`cin.get(string,size).get()`的方式读取一行字符串并丢弃换行符

* `string`类，C++98添加了`string`类，因此可以使用`string`类存储字符串而不是字符数组

```cpp
include <string>
int main()
{
    string XB = "Hello,XB";
}
```

`string`类重载了运算符，可以比较方便地进行如下操作：
```cpp
string str1;
string str2 = "XB";
str1 = str2; // 直接赋值是允许的，strcpy()的作用
str2 = ",hello"; // 直接修改值

string str3;
str3 = str1 + str2; // 方便直接将两个字符串进行合并，strcat()的作用
int len = str3.size(); // 相当于strlen()
```

这里注意，由于`cin`的设计只考虑了`double`，`int`等基本数据类型，没有处理`string`的类方法，因此，这里在读取时最好直接使用`getline(cin,string)`

### 1.3.3 结构体

* 一个结构体声明并实例化的例子：
```cpp
#include <iostream>
struct person   // structure declaration
{
    char name[20];
    float volume;
    double price;
};

int main()
{
    using namespace std;
    person guest =
    {
        "Glorious Gloria",  // name value
        1.88,               // volume value
        29.99               // price value
    };
    person pal =
    {
        "Audacious Arthur",
        3.12,
        32.99
    };
```

* 结构数组
```cpp
struct person
{
    char name[20];
    float volume;
    double price;
};
int main()
{
    using namespace std;
    person guests[2] =          // initializing an array of structs
    {
        {"Bambi", 0.5, 21.99},      // first structure in array
        {"Godzilla", 2000, 565.99}  // next structure in array
    };
    return 0; 
}
```

### 1.3.4 指针

* 初始化
```cpp
int a = 5;
int* ptr = &a;
```

* 野指针（何老师定义的）
```cpp
long* fellow;
*fellow = 123;
```
这种的一看就不行，指针得先指向地址才行，直接赋值都没地址肯定不行。

* `new`的使用

之前学习的指针都是直接指向了**编译阶段**已经分配好内存的变量，而指针真正的用武之地在于，在**运行阶段**分配未命名的内存以存储值。C语言中使用`malloc()`动态分配内存，而C++中的`new`运算符则更为好用：
```cpp
int* pn = new int;
*pn = 100;
delete pn;
```
`new int`告诉程序需要适合存储`int`的内存，`new`运算符则根据类型确定需要多少字节的内存，并返回其地址，赋值给`pn`，此后此内存位置的数据只能通过`pn`进行访问

若需要释放内存，使用`delete`对指针进行操作即可，注意`delete`只能处理`new`动态分配的内存而无法作用于已经指向确定变量的指针

使用`new`创建动态数组：
```cpp
int* psome = new int [10];
delete [] psome;
```
`new`运算符返回第一个元素的地址并赋个指针`psome`,此时`delete`的方括号代表应该释放整个数组

* 静态联编和动态联编

一般情况下C++将数组名解释为地址，声明的数组使用静态联编，长度和地址都是固定的，数组的静态联编体现在不能在允许过程中改变数组大小；而使用`new`分配的使用动态联编

* 指针算数

C语言学习期间已经滚瓜烂熟了，这里就说个`XB[3]`等同于`*(XB+3)`就行

* 指针和字符串

如果给`cout`提供一个地址，则它将从该地址向后打印直到遇到空字符为止；但是`cout`只封装了对字符数组这样的重载功能，对于其他如`int`数组，`cout`会直接打印其地址，下面例子：
```cpp
char a[20] = "hello";
cout << "a is: " << a << endl;
int b[10] = {1, 2, 3, 4, 5};
cout << "b is: " << * b << endl;
```
的输出为：
```bash
a is: hello
b is: 0x5ffe80
```

下面的例子中，`cin`封装了同样的操作效果对字符数组进行重新赋值
```cpp
#include <iostream>
#include <cstring>      // or string.h
using namespace std;
char * getname(void);   // function prototype
int main()
{
    char * name;        // create pointer but no storage

    name = getname();   // assign address of string to name
    cout << name << " at " << (int *) name << "\n";
    delete [] name;     // memory freed

    name = getname();   // reuse freed memory
    cout << name << " at " << (int *) name << "\n";
    delete [] name;     // memory freed again
    return 0;
}

char * getname()        // return pointer to new string
{
    char temp[80];      // temporary storage
    cout << "Enter last name: ";
    cin >> temp;
    char * pn = new char[strlen(temp) + 1];
    strcpy(pn, temp);   // copy string into smaller space

    return pn;          // temp lost when function ends
}
```
此例同时可以看出`new`的用武之地，以长度`strlen(temp) + 1`动态分配内存给pn，函数`getname()`中的临时字符数组`temp`将在每次使用后被释放掉，而只返回`pn`动态指针，这样可以大大节省内存空间

* 使用`new`创建动态结构

在运行时创建数组优于在编译时创建数组，对结构也是如此，将`new`用于结构分为两个步骤：创建结构和访问其成员，创建结构指针这么操作：
```cpp
struct person
{
    char name[20];
    float volume;
    double price;
};
int main()
{
    using namespace std;
    person* ps = new person;
}
```

比较棘手的是访问成员，因为创建的动态结构指针实际上代表地址，不能直接使用`.`访问其内容。这里要使用`->`访问其成员内容，如`ps->price`，或者使用`(*ps).price`进行访问

* 变量存储类型

1. 自动存储
在函数内部定义的常规变量即为自动存储，作用域仅仅为花括号包含的代码块，在函数运行结束时就会自动释放。自动变量通常存储在**栈**内，遵循后进先出的规则

2. 静态存储
静态存储是在整个程序执行期间都存在的存储方式，使变量成为静态的方式有两种：在函数外面定义，或在声明时使用关键字`static`
静态存储的变量类型有：
> 全局变量`extern`，没有限制，这个使用比较危险，不建议使用
> 静态全局变量`static`，不可跨文件访问，作用域是文件全局
> 静态局部变量`static`，作用域是函数内部
> 静态成员变量，后面讨论
> 名空间中定义的变量，实际上可以替代全局变量使用

1. 动态存储
`new`和`delete`提供了更为灵活的存储方式。它们管理了一个内存池称为**堆（heap）**，这个内存池和用于自动变量和静态变量的内存是分开的。`new`和`delete`允许在一个函数中分配内存，并在另一个函数中释放，这使得数据的声明和周期不完全受程序和函数的生存时间控制，使得程序员有对内存更大的控制权。缺点是内存管理也变得复杂了，在栈中的内存总是连续的，而动态存储会使内存空间不连续

* **有趣**的指针数组

很简单的，没大一时描述的那么难以理解
```cpp
const person* XB[3] = {&p1, &p2, &p3};
std::cout << XB[1] -> price;
```
### 1.3.5 `vector`容器

使用场景：
> 需要动态增长和缩小的数组
> 需要频繁在序列末尾添加或移除元素时
> 需要一个可以高效随机访问元素的容器时

那么使用`vector`吧：

```cpp
#include <vector>
```

* 创建`vector`

```cpp
std::vector<int> myVector; // 创建一个存储整数的空 vector
std::vector<int> myVector(5); // 创建一个包含 5 个整数的 vector，每个值都为默认值（0）
std::vector<int> myVector(5, 10); // 创建一个包含 5 个整数的 vector，每个值都为 10
std::vector<int> vec2 = {1, 2, 3, 4}; // 初始化一个包含元素的 vector
```

* 添加和删除元素

```cpp
myVector.push_back(7); // 将整数 7 添加到 vector 的末尾
myVector.insert(vec.begin() + 1, 10);// 在第二个位置插入元素 10
myVector.erase(myVector.begin() + 2); // 删除第三个元素
myVector.clear(); // 清空 vector
```

* 访问元素和大小

```cpp
int x = myVector[0]; // 获取第一个元素
int y = myVector.at(1); // 获取第二个元素
int size = myVector.size(); // 获取 vector 中的元素数量
```

* 循环遍历

下面是一般使用for循环的形式：
```cpp
for (auto it = myVector.begin(); it != myVector.end(); ++it) 
{
    std::cout << *it << " ";
}
```
对于容器类对象还可以使用范围循环`int element : myVector`，非常像python的`for i in range(container)`
```cpp
for (int element : myVector) 
{
    std::cout << element << " ";
}
```

## 1.4 循环语句

* 修改规则

C++允许以下形式的`for`语句的语法，在循环中定义`i`，可以节省不少事
```cpp
for(int i = 0;i < 5 :i++)
{
    cout << i;
}
cout << i; // 此时i已经被释放
```

* 逗号运算符

使用`,`可以将两个表达式合为一个：
```cpp
for(j = 0,i = word.size()-1; j < i; --i, ++j)
```
* 基于范围的循环（C++11）

对容器类的数据类型使用（数组，`vector`等）
```cpp
int XB = {1,2,3,4,5};
for(int xb : XB)
{
    std::cout << xb << std::endl;
}
for(int& xt : XB)
{
    xt *= 0.8;
}
```

## 1.5 分支语句

没有有用的，故跳过

## 1.6 单独编译

### 1.6.1 头文件

头文件中经常包含的内容：
* 函数原型，内联函数原型
* 使用`#define`或`const`定义的符号常量
* 结构，类声明
* 函数模板声明
* 名空间

以便在多个源文件中共享。这样可以提高代码的可读性和可维护性。

以下是一个头文件定义的示例：
```cpp
#ifndef COORDIN_H_
#define COORDIN_H_

struct polar
{
    double distance;    // distance from origin
    double angle;        // direction from origin
};
struct rect
{
    double x;        // horizontal distance from origin
    double y;        // vertical distance from origin
};

polar rect_to_polar(rect xypos);
void show_polar(polar dapos); 

#endif
```
其中`#ifndef`、`#define`和`#endif`是预处理指令，确保头文件只被包含一次，防止重复定义。

而`COORDIN_H_`是一种常见的约定，只需要保重宏定义的唯一性即可，最好兼顾可读性

### 1.6.2 多文件编译

* 编译命令

使用`g++`编译器对多文件进行编译（大一点的工程要考虑使用`cmake`），其中`output`为输出`.exe`文件的名称
```bash
g++ -o output file1.cpp file2.cpp
```

注意这里的编译不需要指定头文件，（如果头文件处于当前工作文件夹下的话）

对于`file1.cpp`和`file2.cpp`中指定头文件的方式应该为：
```cpp
#include <iostream>
#include "coordin.h"
```
使用`""`时编译器首先会在当前源文件所在的目录中查找该头文件。如果找不到，才会在系统标准库路径中查找。这通常用于包含项目自定义的头文件

使用`<>`时编译器会在系统的标准库路径中查找该头文件

* 使用`cmake`

确保文件目录为：
```
your_project/
├── CMakeLists.txt
├── file1.cpp
└── file2.cpp
```

其中`CMakeLists.txt`的内容为
```bash
cmake_minimum_required(VERSION 3.10)  # 设置所需的 CMake 版本
project(MyProject)                      # 项目名称

# 指定源文件
set(SOURCES
    file1.cpp
    file2.cpp)

# 如果有头文件目录，可以这样指定
include_directories(include)  # 假设你的头文件在 include 目录下

# 创建可执行文件
add_executable(output ${SOURCES})
```

接着执行如下命令:

```bash
mkdir build           # 创建一个构建目录
cd build              # 进入构建目录
cmake ..              # 配置项目
make                  # 编译项目
```

***

# 2 函数

## 2.1 函数基础知识

### 2.1.1 函数和数组

* 函数形参

在函数中的数组形参可以写为形如`int sum(int arr[], int n)`这种，它和`int sum(int* arr,int n)`代表的含义是相同的，这里不是很明白C++为什么要允许这种形式的形参存在，或许是为了直观性？

* `const`保护

为了避免传递给函数的数组被修改，可以使用`const`保护：`void show_array(const double arr[], int n)`，则后续不再能用arr修改数组

但注意下面几种操作是非法的：
```cpp
const int a = 1;
int* b = &a; //这样b可以修改啊的内容，显得前面的const毫无卵用

int* const c = &a;//只能这么做
int* d = c;//非法的，理由同上
```

区分不同；
```cpp
int a = 3;
const int* b = &a;
int* const c = &a;
```
这里第一种表达形式表明指针指向的内容是`const`的，禁止用指针修改`a`，但允许对指针进行重定向；第二种表明指针本身是`const`的，不允许对指针进行重定向，但允许使用指针修改内容

* 二维数组

形参和上面类似，可以写成`int sun(int arr[][4], int n)`，等同于`int sum(int** arr, int n)`

### 2.1.2 函数指针

与数据相似，函数也有地址，函数的地址是存储其机器语言代码的内存开始的地址（好家伙）。通常这些地址对用户来说没什么卵用，但是对程序来说比较有用

函数指针的使用需要获取函数地址，声明函数指针，并使用函数指针进行调用:

1. 获取函数地址是比较简答的，直接使用函数命不带参数即可
2. 对于函数`double XB(int n)`使用指针`double (*xt)(int)`即可
3. 直接调用即可`(*xt)(4)`等同于`XB(4)`

* `auto`的使用
  
C++11提供的`auto`可以在声明一些非常复杂的指针时进行使用；
```cpp
const double* (*p1)(const double*, int) = f1;
auto p2 = f1;
```

## 2.2 内联函数

内联函数是C++为了提高程序运行速度而所做的一项改进。在常规的函数中，在程序执行的过程中，会跳转到函数的地址所对应的机器码上，这会带来一定的开销；而内联函数的代码在调用处被直接插入，因此在执行时不需要进行上下文切换

直接在声明和定义时使用关键词`inline`即可

使用宏定义了类似函数的功能可以考虑使用更加稳定可靠的`inline`代替

## 2.3 引用变量

* 创建引用变量

```cpp
int rats;
int& cats = rats;
```
其中`&`并不是取地址运算符，而是类型标识符，`int&`指的是指向`int`的引用。乍一看引用和指针其实差不多，区别是指针可以先声明再使用，而引用必须声明的同时进行赋值。引用`int& cats = rats`更像是`int* const dogs = &rats`的伪装表示，一旦指向确定的对象后不能进行重定向：
```cpp
int rats = 100;
int& cats = rats;
int dogs = 50;
cats = dogs;
```
这里的`cats`再指向`rats`后不会进行重定向，`cats = dogs`等同于`rats = dogs`，不会修改`cats`与`rats`共享同一处地址

* 引用作为函数形参

使用引用作为形参是简单高效的，一方面，与使用值作为形参作为对比，函数无法修改传入参数的值，同时，在函数运行时，需要为实参分配临时空间，加大了内存消耗；另一方面，与使用指针作为形参作为对比，两者的效果似乎是相似的，但使用指针作为形参时，实参一般要取地址`&`，这使得函数调用时显得没有那么直观

其他高级语言如python等函数形参默认都是变量的引用

但如果传递的数据对象是数组时，对于C++而言，只能使用指针；这里与python作对比，python的`list`，`array`和`tensor`都能为其添加引用，这是因为python的这些数据结构全是对象，实际上是指向对象的引用

## 2.4 默认参数

这里和python的默认参数差不多
```cpp
int XB(int n, int m = 1; int j = 2);//可行的
int xt(int n, int m = 1; int j);//不可行
```

## 2.5 函数重载

函数的多态性的体现，重载的关键在于函数的参数列表，注意几种不被允许的重载：
```cpp
double cube(double x);
double cube(double& x);

double dog(const double* x);
double dog(double* x);
```
仔细一想就知道，上述函数在调用时`cube(1.0)`，编译器根本无法根据实参的数据类型选用函数，所以是不被允许的

## 2.6 函数模板

```cpp
template<typename Anytype>
void swap(Anytpe& a, Anytype &b)
{
    Anytype temp;
    temp = a;
    a = b;
    b = temp;
}
```
关键字`template`和`typename`是必需的，除非可以使用关键字`class`替代`typename`，这样创建的`swap`函数可以交换任意数据类型的数据

这样做的问题是任意的类型如果为数组或者结构，许多比如相加赋值将无法进行。因此C++提供了两种解决方案：一是重载运算符`+`，二是为特定类型提供具体化的模板

* 显式具体化

和模板化的函数作对比：
```cpp
template<typename T>
void swap(T& a, T& b);//模板类定义

struct job
{
    char name[20];
    double salary;
}
template<> void swap(job& a, job& b);//具体化函数定义
int main()
{
    job a,b;
    swap(a,b);
}
```

* 显式实例化

```cpp
template<typename T>
void swap(T& a, T& b);//模板类定义

int main()
{
    template void swap<char>(char& a, char& b);//显示实例化,相当于实例化了一个模板类
    char a,b;
    swap(a,b);
}
```

***

# 3 对象和类

## 3.1 类定义

下面是一个头文件中声明类的实例：
```cpp
#ifndef STOCK1_H_
#define STOCK1_H_
#include <string>
class Stock
{
private:
    std::string company;
    long shares;
    double share_val;
    double total_val;
    void set_tot() { total_val = shares * share_val; }
public:
    Stock();        // default constructor
    Stock(const std::string & co, long n = 0, double pr = 0.0);
    ~Stock();       // noisy destructor
    void buy(long num, double price);
    void sell(long num, double price);
    void update(double price);
    void show();
};
#endif
```
其中`set_tot()`定义在类声明内，将会自动成为内联函数

数据被封装在`private`中，方法则封装在`public`，这是常用的类声明方法

实例化其中一个函数，使用`Stock::`作用域符号
```cpp
void Stock::acquire(const std::string & co, long n, double pr)
{
    company = co;
    if (n < 0)
    {
        std::cout << "Number of shares can't be negative; "
                  << company << " shares set to 0.\n";
        shares = 0;
    }
    else
        shares = n;
    share_val = pr;
    set_tot();
}
```

## 3.2 构造函数和析构函数

### 3.2.1 构造函数

构造函数没有返回类型，属于类声明的公有部分
下面是构造函数的一种可能定义：
```cpp
Stock::Stock(const string& co, long n, double pr)
{
    company = co;
    shares = n;
    share_val = pr;
    set_hot();
}
```

默认构造函数相当于构造函数的一种重载，在未输入参数时使用，如果未定义则由编译器处理：
```cpp
Stock::Stock()        // default constructor
{
    std::cout << "Default constructor called\n";
    company = "no name";
    shares = 0;
    share_val = 0.0;
    total_val = 0.0;
}
```
使用方法：
```cpp
Stock XB("XB", 50, 2.5);
Stock xt = new Stock("xt", 25, 1.25); //创建动态指针
```

* 在类内创建常量

由于声明类只是描述了对象的形式，，并没有创建对象，因此，在创建对象前并没有用于存储的空间。要想在类内创建对象，使用关键字`static`创建静态成员变量：
```cpp
class XB
{
    private:
        static const int Months = 12;
}
```
这里的`Months`成员变量将被所以`XB`对象共享

### 3.2.2 析构函数

其他语言的OOP部分并没有考虑析构函数；析构函数主要用于使用`delete`释放掉`new`分配的内存
通常对于正常创建的对象，析构函数没有要完成的任务，做做样子就行了：
```cpp
Stock::~Stock()        // verbose class destructor
{
    std::cout << "Bye, " << company << "!\n";
}
```

### 3.2.3 `const`成员函数

以后约定如果类的成员函数不对私有部分进行修改时则声明为：
```cpp
void show() const; // 声明
void Stock::show() const; // 定义
```

## 3.3 `this`指针

假设需要一个比较两个对象功能的函数，则需要将其中一个对象的数据传递给另外一个对象。假设我们定义一个这样的成员函数，返回值是对象的引用：
```cpp
const Stock& top(const Stock& s) const;
```

使用的方法可以为下面的两条语句之一：
```cpp
top = stock1.top(stock2);
top = stock2.top(stock1);
```
`top`函数如何返回本身对象是个问题，这里要用到`this`指针：
```cpp
const Stock & Stock::topval(const Stock & s) const
{
    if (s.total_val > total_val)
        return s;
    else
        return *this; 
}
```

`this`指针指向调用成员函数的对象，上述函数中，其实`total_val`不过也是`this->total_val`的简写

## 3.4 使用类实现'栈'的数据结构

* 声明
```cpp
#ifndef STACK_H_
#define STACK_H_
typedef unsigned long Item;
class Stack
{
private:
    static const int MAX = 10;    
    Item items[MAX];    
    int top;         
public:
    Stack();
    bool isempty() const;
    bool isfull() const;
    bool push(const Item & item);   // add item to stack
    bool pop(Item & item);          // pop top into item
};
#endif
```

* 实现
```cpp
#include "stack.h"
Stack::Stack()    // create an empty stack
{
    top = 0;
}

bool Stack::isempty() const
{
    return top == 0;
}

bool Stack::isfull() const
{
    return top == MAX;
}

bool Stack::push(const Item & item) 
{
    if (top < MAX)
    {
        items[top++] = item;
        return true;
    }
    else
        return false;
}

bool Stack::pop(Item & item)
{
    if (top > 0)
    {
        item = items[--top];
        return true;
    }
    else
        return false; 
}
```

# 4 使用类

## 4.1 运算符重载

函数重载和运算符重载都是C++多态特性的体现；两种重载都是要求根据使用的数据类型而决定函数和运算符采取哪种行为，这很好地体现C++面向数据（对象）的特点，以数据（对象）为核心设计函数和运算符

* 下面从一个实例展现如何对运算符进行重载

头文件：
```cpp
#ifndef MYTIME1_H_
#define MYTIME1_H_
class Time
{
private:
    int hours;
    int minutes;
public:
    Time();
    Time(int h, int m = 0);
    void AddMin(int m);
    void AddHr(int h);
    void Reset(int h = 0, int m = 0);
    Time operator+(const Time & t) const;
    void Show() const;
};
#endif
```
`Time operator+(const Time & t) const`重载了`+`运算符，关键字`operator`

实现：
```cpp
Time Time::operator+(const Time & t) const
{
    Time sum;
    sum.minutes = minutes + t.minutes;
    sum.hours = hours + t.hours + sum.minutes / 60;
    sum.minutes %= 60;
    return sum;
}
```
上述代码可以实现一个将时间对象（包含小时和分钟）相加的运算符`+`重载

因为使用了关键字`operator`，可以直接进行如下操作：
```cpp
Time time1;
Time time2;
total = time1 + time2;
```
相当于`total = time1.operator+(time2)`

## 4.2 友元

C++对私有部分的访问控制严格，公有类方法提供唯一的访问途径；友元提供了另一种形式的访问

友元有3种：
> 友元函数
> 友元类
> 元成员函数

下面介绍友元函数，其他两种友元将在第15章介绍

* 友元的必要性

上面使用成员函数定义二元运算符时会出现一个小问题，比如`A = B * 2`可以解释为`A = B.operator*(2)`，但是`A = 2 * B`显然就不太行，因为`2`不是对象，不能调用成员函数计算该式

因此要将`*`重载为非成员函数，可以非成员函数并不能访问对象的私有部分，于是友元函数就很有必要了

* 创建友元函数

第一步是将其原型放在类声明中（我是你朋友的声明）
```cpp
friend Time operator*(double m,const Time& t);
```

第二步是编写函数定义,在类外实例化，由于不是成员函数，不需要作用域`Time::`限定符
```cpp
Time operator*(double m,const Time& t)
{
    Time result;
    long total = t.hours * m * 60 + t.minutes * m;
    result.hours = total / 60;
    reslut.minutes = total % 60;
    return result;
}
```

还有不使用友元函数的方法：
```cpp
Time operator*(double m,const Time& t)
{
    return t * m;
}
```

* 重载`<<`运算符

`std::cout`实际上是一个`ostream`的对象，而并非关键字，因为`ostream`重载了`<<`运算符，故可以使用`std::cout<<`来输出乱七八糟其他的东西

为了为我们自定义的类重载`<<`，可以如下定义：
```cpp
class Time
{
private:
    int hours;
    int minutes;
public:
    Time();
    Time(int h, int m = 0);
    void AddMin(int m);
    void AddHr(int h);
    void Reset(int h = 0, int m = 0);
    Time operator+(const Time & t) const;
    Time operator-(const Time & t) const;
    Time operator*(double n) const;
    friend Time operator*(double m, const Time & t)
        { return t * m; }   // inline definition
    friend std::ostream & operator<<(std::ostream & os, const Time & t);
};
```

如下定义重载`<<`的函数：
```cpp
std::ostream & operator<<(std::ostream & os, const Time & t)
{
    os << t.hours << " hours, " << t.minutes << " minutes";
    return os; 
}
```
注意重载运算函数的最后的返回值为`ostream`对象`os`，是为了形如`cout<<time1<<"hello"<<time2`的操作可以被支持

## 4.3 动态内存分配

