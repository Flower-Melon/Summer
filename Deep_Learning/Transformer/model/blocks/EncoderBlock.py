from model.layers.MultiHeadAttention import MultiHeadAttention
from model.layers.PositionWiseFFN import PositionWiseFFN
from model.layers.AddNorm import AddNorm
from torch import nn

class EncoderBlock(nn.Module):
    """编码器块"""
    def __init__(self, query_size, key_size, value_size, num_heads,
                 num_hiddens, ffn_num_hiddens, norm_shape, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(
            query_size, key_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens=None):
        # X的形状为(batch_size, num_steps, num_hiddens)
        # valid_lens的形状为(batch_size,)或(batch_size, num_steps)
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

if __name__ == "__main__":
    """测试EncoderBlock"""
    import torch as T
    
    X = T.ones((2, 100, 24))
    valid_lens = T.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 8, 24, 48, [24], 0.5)
    encoder_blk.eval()
    print(encoder_blk(X, valid_lens).shape)