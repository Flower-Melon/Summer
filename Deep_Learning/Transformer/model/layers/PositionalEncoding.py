from torch import nn
import torch as T

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout=0.0, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P矩阵
        self.P = T.zeros((1, max_len, num_hiddens))
        X = T.arange(max_len, dtype=T.float32).reshape(-1, 1) / T.pow(
            10000, T.arange(0, num_hiddens, 2, dtype=T.float32) / num_hiddens)
        
        self.P[:, :, 0::2] = T.sin(X)
        self.P[:, :, 1::2] = T.cos(X)
    
    def forward(self, X):
        # 输入X的形状为(批量大小, 序列长度, num_hiddens)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
if __name__ == "__main__":
    """测试PositionalEncoding类"""
    X = T.zeros((2, 100, 512))
    pos_encoding = PositionalEncoding(num_hiddens=512, dropout=0.5)
    Y = pos_encoding(X)
    print(Y.shape)  # 应该输出 (2, 100, 512)