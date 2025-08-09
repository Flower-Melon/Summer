/*
 * @lc app=leetcode.cn id=167 lang=cpp
 *
 * [167] 两数之和 II - 输入有序数组
 */

// @lc code=start
#include <vector>
class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& numbers, int target) 
    {
        int l = 0;
        int r = numbers.size() - 1;

        while (l < r) {
            int sum = numbers[l] + numbers[r];
            if (sum < target) 
            {
                l++;
            } 
            else if (sum == target) 
            {
                // 注意返回的下标从 1 开始
                return {l + 1, r + 1};
            } 
            else 
            {
                r--;
            }
        }
        return {}; // 这行不会执行，但需写上以通过编译。
    }
};
// @lc code=end

