/*
 * @lc app=leetcode.cn id=198 lang=cpp
 *
 * [198] 打家劫舍
 */

// @lc code=start
#include <vector>
#include <algorithm>
using std::vector;
using std::max;
class Solution 
{
public:
    int rob(vector<int>& nums) 
    {
        int n = nums.size();
        if(n == 1)
        {
            return nums[0];
        }
        else if(n == 2)
        {
            return max(nums[0],nums[1]);
        }
        else if(n == 3)
        {
            return max(nums[1],nums[0] + nums[2]);
        }
        vector<int> ans;
        ans.push_back(nums[0]);
        ans.push_back(max(nums[0],nums[1]));
        ans.push_back(max(nums[1],nums[0] + nums[2]));
        for(int i = 4; i <= n; i++)
        {
            ans.push_back(max(nums[i - 2] + ans[i - 4], nums[i - 1] + ans[i - 3])); // 递推公式，核心公式
        }
        return *(ans.end()-1);
    }
};
// @lc code=end

