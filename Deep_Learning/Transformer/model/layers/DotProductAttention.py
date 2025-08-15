import torch as T
from torch import nn
import math

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.shape[1]
    # unsqueeze(1)将valid_len的形状调整为(batch_size, 1)
    # unsqueeze(0)将mask的形状调整为(1, maxlen)
    # 使用广播机制，mask的形状为(batch_size, maxlen)
    mask = T.arange((maxlen), dtype=T.float32,
                        device=X.device).unsqueeze(0) < valid_len.unsqueeze(1)
    X[~mask] = value
    return X

class DotProductAttention(nn.Module):
    """点积注意力"""
    def __init__(self,dropout=0.0):
        super(DotProductAttention,self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self,queries,keys,values,valid_lens=None):
        """计算点积注意力  
        参数:
        queries: 查询张量，形状为 (batch_size, num_queries, d)
        keys: 键张量，形状为 (batch_size, num_keys, d)
        values: 值张量，形状为 (batch_size, num_keys, value_dim)
        valid_len: 有效长度张量，形状为 (batch_size,)
        返回:
        (加权后的值, 注意力权重)
        """
        d = queries.shape[-1]
        scores = T.bmm(queries,keys.permute(0,2,1)) / math.sqrt(d)
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        self.attention_weights = self.dropout(self.attention_weights)
        return T.bmm(self.attention_weights, values)
    
    def masked_softmax(self, X, valid_len):
        """遮蔽softmax"""
        if valid_len is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if len(valid_len.shape) == 1:
                valid_len = T.repeat_interleave(valid_len, shape[1]) 
                # valid_len的重复后的形状为(batch_size * num_steps1，)
            else:
                valid_len = valid_len.reshape(-1)
            # 这里输入的X的形状为(batch_size, num_steps1， num_steps2),调整为(batch_size * num_steps1, num_steps2)
            X = sequence_mask(X.reshape(-1, shape[-1]), valid_len, value=-1e6)
            
            # 重新调整形状为(batch_size, num_steps1, num_steps2)，并对最后一个维度进行softmax
            return nn.functional.softmax(X.reshape(shape), dim=-1)
    
    
if __name__== "__main__":
    """测试DotProductAttention类"""
    queries, keys = T.normal(0, 1, (2, 1, 2)), T.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = T.arange(40, dtype=T.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = T.tensor([2, 6])
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print("输出为：",attention(queries, keys, values, valid_lens))
    print("注意力输出的形状:", attention.attention_weights.shape)