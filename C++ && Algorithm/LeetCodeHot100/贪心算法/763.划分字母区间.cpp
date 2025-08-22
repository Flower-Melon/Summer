/*
 * @lc app=leetcode.cn id=763 lang=cpp
 *
 * [763] 划分字母区间
 */

// @lc code=start
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
using std::vector;
using std::string;
using std::unordered_map;
using std::max;
class Solution 
{
public:
    vector<int> partitionLabels(string s) 
    {
        unordered_map<char, int> lastIndex;
        vector<int> ans;
        for (int i = 0; i < s.size(); i++)
        {
            lastIndex[s[i]] = i; // 记录每个字符最后出现的位置
        }

        int reach = lastIndex[s[0]];
        int count = 0;
        for(int i = 0; i < s.size(); i++)
        {
            reach = max(reach,lastIndex[s[i]]);
            count++;
            if(i == reach)
            {
                ans.push_back(count);
                count = 0;
            }
        }
        return ans;
    }
};
// @lc code=end

