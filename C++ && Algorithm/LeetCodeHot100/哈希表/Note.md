# 哈希表

## 原理介绍

大二的数据结构课中学过哈希表,不过忘光了,这里需要自学一下[参考帖子](https://blog.csdn.net/Peealy/article/details/116895964)

哈希表本质上就是一个数组，只不过数组存放的是单一的数据，而哈希表中存放的是键值对,`key`通过哈希函数得到数组的索引，进而存取索引位置的值,所以它的查找和插入时间复杂度都是O(1)

* 哈希表的插入，删除，查找操作的平均时间复杂度都为$O(1)$

要理解这个需要了解哈希表的存储方式。哈希表这种数据借助哈希函数，根据输入`key`的值计算出一个位置，将`key`和`value`存储于哈希函数计算出的位置处，查找时只需要根据`key`调用一次哈希函数，查看计算出的位置处是否为空，就能完成查找。

可以想到哈希函数是有一定限制的，多个`key`可能经过哈希函数计算而指向同一片存储区域，这就是**哈希冲突**，可以在同一个位置上建立链表、树之类的数据结构缓解冲突，因为哈希冲突的存在，时间复杂度只能做到平均为$O(1)$

哈希冲突的解决和哈希函数的具体实现先不关系，要学会的是如何借助哈希表解决问题

* C++中使用哈希表

C++中使用哈希表需要包含头文件`#include<unordered_map>`或`#include<unordered_set>`,它是一个模板类,可以存放任意类型的键值对,常用的成员函数有:
> `insert()`: 插入键值对或键
> `erase()`: 删除键值对或键
> `find()`: 以`key`作为参数寻找哈希表中的元素，如果哈希表中存在该`key`值则返回该位置上的迭代器，否则返回哈希表最后一个元素下一位置上的迭代器
> `count()`: 统计某个`key`值对应的元素个数， 因为哈希表不允许重复元素，所以返回值为0或1
> `size()`: 返回哈希表中键值对的个数
> `clear()`: 清空哈希表
> `begin()`: 返回哈希表的迭代器指向第一个元素
> `end()`: 返回哈希表的迭代器指向最后一个元素的下一个位置

* 声明方法
```cpp
#include<unordered_map>
#include<unordered_set>
std::unordered_map<int, int> map; //存储键值对
std::unordered_set<int> set; // 只存储键
```

* 存储键值对时值的访问方法
```cpp
#include<unordered_map>
std::unordered_map<int, int> map; //存储键值对
value = map[key]; // 如果key不存在，则插入键值对，值为默认值
value = map.at(key); // 如果key不存在，则抛出异常
value = map.find(key)->second; // 如果key不存在，则返回end()迭代器
```

* 下面是一个实例判断数组里是否有重复元素
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

给定一个整数数组`nums`和一个整数目标值`target`，请你在该数组中找出 和为目标值`target`的那**两个**整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素

* 原始暴力枚举做法
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
时间复杂度为$O(n^2)$

* 使用哈希表方法
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
只需要一次遍历，时间复杂度为$O(n)$

## 49.字母异位词分组

给你一个字符串数组，请你将**字母异位词**组合在一起。可以按任意顺序返回结果列表。

示例：

输入：`strs = ["eat", "tea", "tan", "ate", "nat", "bat"]`
输出：`[["bat"],["nat","tan"],["ate","eat","tea"]]`

* 思考过程

本题中关键点是如何区分两个单词是否为**字母异位词**，这里我还是蠢了，想了一会全是废物办法，包括统计每个字符的出现次数并作比较，为每一个单词生成哈希表再做比较等等，归根到底全都是些力大砖飞的办法

官方给出的方法其实很简单，说白了就是**字母异位词**的`sort`排序结果一定是相同的。说起来简单，但我就是没想到，还是太废物了

* 灵活使用数据结构

虽然本题是哈希表专题下的题，但是不能拘泥于哈希表的一般形式

需要构建`unordered_map<string,vector<string>> XB`的数据结构，以`sort`后的单词作为`key`，所有**字母异位词**作为值存储于对应的`key`之后

这里哈希的`value`值实际上存储了一堆东西，也是思维僵化了，总觉得`value`只能是一个值，实际上可以是一个容器

到这里能想出**字母异位词**的等价条件，并构建出对应的数据结构，也基本上结束了

* 题解

```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) 
{
    unordered_map<string,vector<string>> XB;
    for(string& str : strs)
    {
        string sorted = str;
        sort(sorted.begin(),sorted.end());
        XB[sorted].push_back(str);
    }

    vector<vector<string>> ans;
    for(auto& it : XB)
    {
        ans.push_back(it.second);
    }
    return ans;
}
```