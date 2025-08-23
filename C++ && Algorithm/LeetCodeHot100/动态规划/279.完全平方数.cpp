/*
 * @lc app=leetcode.cn id=279 lang=cpp
 *
 * [279] 完全平方数
 */

// @lc code=start
#include <vector>
#include <algorithm>

class Solution 
{
public:
    int numSquares(int n) 
    {
        std::vector<int> f(n + 1);
        for (int i = 1; i <= n; i++) 
        {
            int minn = INT_MAX;
            for (int j = 1; j * j <= i; j++) 
            {
                minn = std::min(minn, f[i - j * j]);
            }
            f[i] = minn + 1;
        }
        return f[n];
    }
};
// @lc code=end

