/*
 * @lc app=leetcode.cn id=1 lang=cpp
 *
 * [1] 两数之和
 */

// @lc code=start
#include <vector>
#include <unordered_map>
using std::vector;
using std::unordered_map;
class Solution 
{
public:
    vector<int> twoSum(vector<int>& nums, int target) 
    {
        unordered_map<int, int> hashtable;
        for (int i = 0; i < nums.size(); ++i) 
        {
            auto it = hashtable.find(target - nums[i]);
            if (it != hashtable.end()) 
            {
                return {it->second, i};
            }
            hashtable[nums[i]] = i;
        }
        return {};
    }
};

// @lc code=end