from model.blocks.DecoderBlock import DecoderBlock
from model.layers.PositionalEncoding import PositionalEncoding
from torch import nn
import math

class Decoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, vocab_size, query_size, key_size, value_size, num_heads,
                 num_hiddens, ffn_num_hiddens, norm_shape, num_layers, dropout=0.0):
        super(Decoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"decoder_block_{i}",
                                 DecoderBlock(query_size, key_size, value_size,
                                              num_heads, num_hiddens, ffn_num_hiddens,
                                              norm_shape, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def initialize_state(self, enc_outputs, enc_valid_lens):
        """初始化解码器的状态"""
        # enc_outputs的形状为(batch_size, num_steps, num_hiddens)
        return [enc_outputs, enc_valid_lens, [None] * len(self.blks)]
    
    def forward(self, X, state):
        # X的形状为(batch_size, num_steps)
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X, state = blk(X, state)
        # X的形状为(batch_size, num_steps, num_hiddens)
        return self.dense(X)

if __name__ == "__main__":
    """测试Decoder"""
    import torch as T
    
    X = T.ones((2, 100), dtype=T.long)
    valid_lens = T.tensor([3, 2])
    decoder = Decoder(10, 24, 24, 24, 8, 24, 48, [24], 2, 0.5)
    decoder.eval()
    state = decoder.initialize_state(T.ones((2, 10, 24)), valid_lens)
    print(decoder(X, state).shape)  # 输出形状应为(batch_size, num_steps, vocab_size)