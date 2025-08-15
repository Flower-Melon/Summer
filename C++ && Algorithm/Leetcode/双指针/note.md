# 本文档用来做刷题记录

## 11.盛最多水的容器

给定一个长度为$n$的整数数组$height$。有$n$条垂线，第i条线的两个端点是$(i, 0)$和$(i, height[i])$。

找出其中的两条线，使得它们与$x$轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

![water_container](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/water_container.png)

* 一开始的暴力求解思路

```cpp
#include <vector>
class Solution {
public:
    int maxArea(std::vector<int>& height) 
    {
        int contain = 0;
        int H = 0;
        int W = 0;
        for(int i = 0; i < (height.size() - 1); i++)
        {
            for(int j = 0; j < height.size(); j++)
            {
                W = j - i;
                H = (height[i] < height[j]) ? height[i] : height[j];
                contain = (contain > H * W) ? contain : H * W;
            }
        }
        return contain;
    }
};
```
使用两层`for`循环暴力遍历，虽然能满足要求但是有着$O(n^2)$的时间复杂度，在输入序列为100000时直接超时了，要想办法优化。

* 双指针法

```cpp
#include <vector>
#include <algorithm>
class Solution {
public:
    int maxArea(std::vector<int>& height) 
    {
        int left = 0; // 指向最左端的木板
        int right = height.size() - 1; // 指向最右端的木板
        int contain = 0; // 容量
        int H = 0; // 短板
        int W = 0; // 两木板间的距离
        while(left < right)
        {
            W = right - left;
            if(height[left] >= height[right])
            {
                H = height[right];
                right--;
            }
            else
            {
                H = height[left];
                left++;
            }
            contain = std::max(contain, H * W);
        }
        return contain;
    }
};
```
这个方法我一开始是非常难以理解的，总觉得会少算了很多种情况。

我们从一开始进行考虑，暴力求解中需要两个`for`循环解决问题，这个过程是否能够优化？现在我们让一个循环从左向右遍历，一个循环从右向左遍历。

由于决定高度的是短木板，当内层循环（从右向左遍历的循环）的木板高度高于当前外层循环的木板高度时，此时内层循环再往后遍历是徒劳的，因为再往后遍历宽度会减少，高度不会高于当前外层循环的木板高度，所以此时直接终止内层循环，外层循环指向下一个木板。

停止遍历的部分就是我们优化掉的部分，减少了时间复杂度。

直观上来说，这就像一个头指针和一个尾指针交替向中间移动，直到汇合，由于两个指针加起来只移动了整个队列的长度，因此算法的时间复杂度为$O(n)$.

* 一点感想

这个题目是Leetcode上第一个拿下我的题目（其实只是我做到的第二道题），本题深刻展示了一个好的算法是多么有用。之前的编程思路全是暴力遍历，这种思想是行不通的，得多想优化办法才行。

## 26.删除有序数组中的重复项

给你一个**非严格递增排列**的数组`nums`，请你**原地**删除重复出现的元素，使每个元素只出现一次 ，返回删除后数组的新长度。元素的相对顺序应该保持一致 。然后返回`nums`中唯一元素的个数。

* 一开始的需要空间复杂度为$O(n)$的思路

```cpp
int removeDuplicates(std::vector<int>& nums) 
{
    int numsize = nums.size();
    int* temp = new int[numsize];    // 使用动态数组来存储不重复的元素
    temp[0] = nums[0];
    int cur = 0; // 指向临时数组的下标

    // 从 nums[1] 开始与临时数组的元素比较，不相同的直接放进去 temp
    for (int i = 1; i < numsize; i++) 
    {
        if (nums[i] != temp[cur]) 
        {
            temp[++cur] = nums[i]; 
        }
    }

    // 将临时数组拷回去
    for (int i = 0; i <= cur; i++) 
    {
        nums[i] = temp[i];
    }
    // 释放动态分配的内存
    delete[] temp;
    return cur + 1;
}
```
这种方法是之前不计后果的编程思想的一种反应，不计时间代价暴力遍历，不计空间代价随意开辟新空间。

* 双指针思路尽需要空间复杂度为$O(1)$

```cpp
int removeDuplicates(std::vector<int>& nums) 
{
    int numsize = nums.size();
    int XB = 0; // 存放不同的值
    int xt = 1; // 遍历数组
    while(xt < numsize)
    {
        if(nums[XB] != nums[xt])
        {
            XB++;
            nums[XB] = nums[xt];
        }
        xt++;
    }
    return XB + 1;
}
```
可以看出思路是没有多难的，但多动动脑子可以节省内存空间，在原数组的基础上进行操作即可。

* 一点感想

通过上来这两题，希望在以后的刷题过程中时刻牢记多动脑子，暴力遍历时能想想是否可以优化，分配内存给新变量时想想是否真的有必要。