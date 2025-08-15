/*
 * @lc app=leetcode.cn id=42 lang=cpp
 *
 * [42] 接雨水
 */

// @lc code=start
#include <vector>
#include <algorithm>
class Solution {
public:
    int trap(std::vector<int>& height) 
    {
        int numsize = height.size();
        int C = 0; // 容量

        int H = 0;
        int key = 0;
        for(int i = 0; i < numsize; i++)
        {
            if(height[i] > H)
            {
                H = height[i];
                key = i;
            }
        }

        int XB = 0;
        int xt = 1;
        while(xt < key)
        {
            if(height[xt] >= height[XB])
            {
                XB = xt;
            }
            else
            {
                C += (height[XB] - height[xt]);
            }
            xt++;
        }    

        XB = numsize - 1;
        xt = numsize - 2;
        while(xt > key)
        {
            if(height[xt] >= height[XB])
            {
                XB = xt;
            }
            else
            {
                C += (height[XB] - height[xt]);
            }
            xt--;
        }
        return C;  
    }
};
// @lc code=end