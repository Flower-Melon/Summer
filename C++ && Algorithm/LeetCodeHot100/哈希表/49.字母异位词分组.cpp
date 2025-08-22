/*
 * @lc app=leetcode.cn id=49 lang=cpp
 *
 * [49] 字母异位词分组
 */

// @lc code=start
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
using std::vector;
using std::string;
using std::unordered_map;
using std::sort;
class Solution 
{
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) 
    {
        unordered_map<string,vector<string>> XB;
        for(string& str : strs)
        {
            string sorted = str;
            sort(sorted.begin(),sorted.end());
            XB[sorted].push_back(str);
        }

        vector<vector<string>> ans;
        for(auto& it : XB)
        {
            ans.push_back(it.second);
        }
        return ans;
    }
};
// @lc code=end

