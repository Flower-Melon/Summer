/*
 * @lc app=leetcode.cn id=1 lang=cpp
 *
 * [1] 两数之和
 */

// @lc code=start
#include <vector>

class Solution 
{
public:
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
};

// @lc code=end