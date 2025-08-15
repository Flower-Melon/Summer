from torch import nn
from model.layers.DotProductAttention import DotProductAttention

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, q_size, k_size, v_size, num_hiddens, 
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(q_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(k_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(v_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # 输入queries，keys，values的形状:(batch_size，查询或者“键－值”对的个数，值size)
        # 经过变换后，输出的queries，keys，values的形状:(batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            # valid_lens的复制后形状为(batch_size * num_heads,)
            valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)
        
        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
       
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

if __name__ == "__main__":
    """测试MultiHeadAttention"""
    import torch as T
    
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens =  6, T.tensor([3, 2])
    X = T.ones((batch_size, num_queries, num_hiddens))
    Y = T.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)
    