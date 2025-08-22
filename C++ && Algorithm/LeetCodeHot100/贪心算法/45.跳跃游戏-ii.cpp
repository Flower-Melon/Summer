/*
 * @lc app=leetcode.cn id=45 lang=cpp
 *
 * [45] 跳跃游戏 II
 */

// @lc code=start
#include <vector>
using std::vector;
class Solution 
{
public:
    int jump(vector<int>& nums) 
    {
        int ans = 0;
        int i = 0; // 当前下标
        if(nums.size() == 1)
        {
            return 0;
        }
        while(i + nums[i] < nums.size() - 1)
        {
            int step = 1; // 这一次计划跳几步
            for(int j = 1; j <= nums[i]; j++)
            {
                if(j + nums[i + j] > step + nums[i + step])
                {
                    step = j;
                }
            }
            i = i + step;
            ans++;
        }
        return ans + 1;
    }
};
// @lc code=end

