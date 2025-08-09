/*
 * @lc app=leetcode.cn id=11 lang=cpp
 *
 * [11] 盛最多水的容器
 */

// @lc code=start
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxArea(std::vector<int>& height) 
    {
        int left = 0;
        int right = height.size() - 1;
        int contain = 0;
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
// @lc code=end

