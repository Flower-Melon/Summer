/*
 * @lc app=leetcode.cn id=121 lang=cpp
 *
 * [121] 买卖股票的最佳时机
 */

// @lc code=start
#include <vector>
#include <algorithm>
using std::vector;
using std::max;
class Solution {
public:
    int maxProfit(vector<int>& prices) 
    {
        int ans = 0;
        int size = prices.size();
        int minp = 0; // 最小指针
        int maxp = 1; //最大指针

        while(maxp < size)
        {
            ans = max(ans,prices[maxp] - prices[minp]);
            if(prices[minp] > prices[maxp])
            {
                minp++;
            }
            else
            {
                maxp++;
            }
        }
        return ans;
    }
};
// @lc code=end

