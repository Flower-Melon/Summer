# 此文档用来记录C++学习过程中关键点

***

# 1 一些基础知识

* C++在C的基础上添加了面向对象编程和泛型编程的支持

## 1.1 名空间

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

3. 动态存储
`new`和`delete`提供了更为灵活的存储方式。它们管理了一个内存池称为**堆（heap）**，这个内存池和用于自动变量和静态变量的内存是分开的。`new`和`delete`允许在一个函数中分配内存，并在另一个函数中释放，这使得数据的声明和周期不完全受程序和函数的生存时间控制，使得程序员有对内存更大的控制权。缺点是内存管理也变得复杂了，在栈中的内存总是连续的，而动态存储会使内存空间不连续

* **有趣**的指针数组

很简单的，没大一时描述的那么难以理解
```cpp
const person* XB[3] = {&p1, &p2, &p3};
std::cout << XB[1] -> price;
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

## 1.5 分支语句

没有有用的，故跳过

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