/*
 * @lc app=leetcode.cn id=128 lang=cpp
 *
 * [128] 最长连续序列
 */

// @lc code=start
#include <vector>
#include <unordered_set>
#include <algorithm>
using std::max;
using std::vector;
using std::unordered_set;
class Solution 
{
public:
    int longestConsecutive(vector<int>& nums) 
    {
        unordered_set<int> num_set;
        for (int num : nums) 
        {
            num_set.insert(num);
        } // 将数组转换为哈希表存储

        int ans = 0;

        for(int num : num_set) // 这里遍历哈希表是因为哈希表可以去重
        {
            if(!num_set.count(num - 1)) // 这个条件判断非常关键，大大优化了时间复杂度
            {
                int current_num = num;
                int length = 1;
                while(num_set.count(current_num + 1))
                {
                    current_num++;
                    length++;
                }
                ans = max(ans, length); // 更新最长连续序列长度
            }
        }
        return ans;
    }
};
// @lc code=end

