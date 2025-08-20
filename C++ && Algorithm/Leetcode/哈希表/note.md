# 哈希表

## 原理介绍

大二的数据结构课中学过哈希表,不过忘光了,这里需要自学一下[参考帖子](https://blog.csdn.net/Peealy/article/details/116895964)

哈希表本质上就是一个数组，只不过数组存放的是单一的数据，而哈希表中存放的是键值对,`key`通过哈希函数得到数组的索引，进而存取索引位置的值,所以它的查找和插入时间复杂度都是O(1)

* C++中使用哈希表

C++中使用哈希表需要包含头文件`#include<unordered_map>`,它是一个模板类,可以存放任意类型的键值对,常用的成员函数有:
> `insert()`: 插入键值对
> `erase()`: 删除键值对
> `find()`: 以`key`作为参数寻找哈希表中的元素，如果哈希表中存在该`key`值则返回该位置上的迭代器，否则返回哈希表最后一个元素下一位置上的迭代器
> `count()`: 统计某个`key`值对应的元素个数， 因为`unordered_map`不允许重复元素，所以返回值为0或1
> `size()`: 返回哈希表中键值对的个数
> `clear()`: 清空哈希表
> `[]`: 通过`key`访问值,如果`key`不存在,则插入键值对,值为默认值
> `at()`: 通过`key`访问值,如果`key`不存在,则抛出异常
> `begin()`: 返回哈希表的迭代器指向第一个元素
> `end()`: 返回哈希表的迭代器指向最后一个元素的下一个位置

* 声明方法:
```cpp
#include<unordered_map>
std::unordered_map<key_type, value_type> map_name;
std::unordered_map<int, int> map;
```

* 值的查找方法
```cpp
value = map_name[key]; // 如果key不存在，则插入键值对，值为默认值
value = map_name.at(key); // 如果key不存在，则抛出异常
value = map_name.find(key)->second; // 如果key不存在，则返回end()迭代器
```

* 下面是一个实例判断数组里是否有重复元素:
```cpp
#include <iostream>
#include <vector>
#include <unordered_set>

bool hasDuplicate(const std::vector<int>& A) 
{
    std::unordered_set<int> seen;  // 底层哈希表
    for (int x : A) 
    {
        // hash(x) -> 桶索引 -> 查找是否已存在
        if (seen.find(x) != seen.end()) 
        {
            // 已经见过，说明有重复
            return true;
        }
        // 否则把它插入哈希表
        seen.insert(x);
    }
    return false;
}

int main() 
{
    std::vector<int> v1 = {1, 3, 5, 7, 3, 9};
    std::vector<int> v2 = {2, 4, 6, 8};

    std::cout << std::boolalpha;
    std::cout << "v1 有重复吗？ " << hasDuplicate(v1) << "\n";  // 输出 true
    std::cout << "v2 有重复吗？ " << hasDuplicate(v2) << "\n";  // 输出 false

    return 0;
}
```

## 1.两数之和

原始暴力枚举做法:
```cpp
std::vector<int> twoSum(std::vector<int>& nums, int target) 
{
    std::vector<int> answer;
    for(int i = 0; i < (nums.size()-1); i++)
    {
        for(int j = i + 1; j < nums.size(); j++)
        {
            if((nums[i] + nums[j]) == target)
            {
                answer.push_back(i);
                answer.push_back(j);
                return answer;
            }
        }
    }
    return answer;
}
```

使用哈希表方法:
```cpp
using std::vector;
using std::unordered_map
vector<int> twoSum(vector<int>& nums, int target) 
{
    unordered_map<int, int> hashtable;
    for (int i = 0; i < nums.size(); ++i) 
    {
        auto it = hashtable.find(target - nums[i]);
        if (it != hashtable.end()) 
        {
            return {it->second, i};
        }
        hashtable[nums[i]] = i;
    }
    return {};
}
```