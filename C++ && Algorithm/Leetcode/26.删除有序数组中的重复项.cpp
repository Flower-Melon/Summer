/*
 * @lc app=leetcode.cn id=26 lang=cpp
 *
 * [26] 删除有序数组中的重复项
 */

// @lc code=start
#include <vector>
class Solution {
public:
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
};
// @lc code=end

