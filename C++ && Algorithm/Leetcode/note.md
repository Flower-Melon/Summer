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

