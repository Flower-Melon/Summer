/*
 * @lc app=leetcode.cn id=11 lang=cpp
 *
 * [11] 盛最多水的容器
 */

// @lc code=start
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
// @lc code=end

