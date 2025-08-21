/*
 * @lc app=leetcode.cn id=283 lang=cpp
 *
 * [283] 移动零
 */

// @lc code=start
# include <vector>
# include <algorithm>
class Solution {
public:
    void moveZeroes(std::vector<int>& nums) 
    {
        int numsize = nums.size();
        int now = 0;
        int future = 0;
        while(future < numsize)
        {
            if(nums[future])
            {
                std::swap(nums[now++],nums[future]);
            }
            future++;
        }
    }
};
// @lc code=end

