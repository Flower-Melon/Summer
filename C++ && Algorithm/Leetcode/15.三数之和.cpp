/*
 * @lc app=leetcode.cn id=15 lang=cpp
 *
 * [15] 三数之和
 */

// @lc code=start
#include <vector>
#include <algorithm>
class Solution {
public:
    std::vector<std::vector<int>> threeSum(std::vector<int>& nums) 
    {
        std::vector<std::vector<int>> answer;
        std::sort(nums.begin(),nums.end());
        for(int i = 0; i < (nums.size() - 2); i++)
        {
            if((i > 0) && (nums[i] == nums[i - 1])) continue;;
            if(nums[i] > 0) break;
            int l = i + 1;
            int r = nums.size() - 1;
            while(l < r)
            {
                if((l > i + 1) && (nums[l] == nums[l - 1]))
                {
                    l++;
                }
                else if ((r < nums.size() - 1) && (nums[r] == nums [r + 1]))
                {
                    r--;
                }
                else if(nums[i] + nums[l] + nums[r] > 0)
                {
                    r--;
                }
                else if(nums[i] + nums[l] + nums[r] < 0)
                {
                    l++;
                }
                else
                {
                    answer.push_back({nums[i], nums[l], nums[r]});
                    l++; 
                    r--;
                }
            }
        }
        return answer;
    }
};
// @lc code=end

