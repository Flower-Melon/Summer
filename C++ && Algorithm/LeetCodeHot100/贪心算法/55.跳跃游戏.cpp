/*
 * @lc app=leetcode.cn id=55 lang=cpp
 *
 * [55] 跳跃游戏
 */

// @lc code=start
#include <vector>
#include <algorithm>
using std::vector;
using std::max;
class Solution {
public:
    bool canJump(vector<int>& nums) 
    {
        int reach = 0; // 可以到达的最远距离
        for(int i = 0; i < nums.size(); i++)
        {
            if(i > reach)
            {
                return false;
            }
            reach = max(reach,i + nums[i]);
        }
        return true;
    }
};
// @lc code=end

