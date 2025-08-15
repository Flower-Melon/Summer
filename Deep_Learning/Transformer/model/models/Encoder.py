from model.blocks.EncoderBlock import EncoderBlock
from model.layers.PositionalEncoding import PositionalEncoding
from torch import nn
import math

class Encoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, query_size, key_size, value_size, num_heads,
                 num_hiddens, ffn_num_hiddens, norm_shape, num_layers, dropout=0.0):
        super(Encoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"encoder_block_{i}",
                                 EncoderBlock(query_size, key_size, value_size,
                                              num_heads, num_hiddens, ffn_num_hiddens,
                                              norm_shape, dropout))
    def forward(self, X, valid_lens):
        # X的形状为(batch_size, num_steps)
        # valid_lens的形状为(batch_size,)或(batch_size, num_steps)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
        return X

if __name__ == "__main__":
    """测试Encoder类"""
    import torch as T
    
    encoder = Encoder(200, 24, 24, 24, 8, 24, 48, [24], 2, 0.5)
    valid_lens = T.tensor([3, 2])
    encoder.eval()
    print(encoder(T.ones((2, 100), dtype=T.long), valid_lens).shape)
    # 期望输出形状为(batch_size, num_steps, num_hiddens)，即(2, 100, 24)