from model.layers.MultiHeadAttention import MultiHeadAttention
from model.layers.PositionWiseFFN import PositionWiseFFN
from model.layers.AddNorm import AddNorm
import torch as T
from torch import nn

class DecoderBlock(nn.Module):
    """解码器块"""
    def __init__(self, query_size, key_size, value_size, num_heads,
                 num_hiddens, ffn_num_hiddens, norm_shape, dropout, i):
        super(DecoderBlock, self).__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(
            query_size, key_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            query_size, key_size, value_size, num_hiddens, num_heads, dropout)  
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
        
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练时直接调用EncoderDecoder.forward，
        # 对于新的批次会重置状态，state[2][self.i]的值一直是none, key_values也一直是X
        if state[2][self.i] is None:
            key_values = X
        # 当推理时，循环调用decoder.forward，状态不会重置；推理时每个step只生成一个词元，
        # 所以需要把每一步的X的历史值记录下来，才能进行后面的注意力汇聚
        else:
            key_values = T.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        
        if self.training:
            batch_size, num_steps,  = X.shape[0], X.shape[1]
            # 这里的dec_valid_lens被称为掩膜注意力有效长度,
            # 它的意义在于要保证每个解码器处理目标序列时每个位置 i 只能看到 ≤ i 的历史
            dec_valid_lens = T.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
            
        # 自注意力
        X2 = self.addnorm1(X, self.attention1(X, key_values, key_values, dec_valid_lens))
        # 注意编码器输出
        X3 = self.addnorm2(X2, self.attention2(X2, enc_outputs, enc_outputs, enc_valid_lens))
        return self.addnorm3(X3, self.ffn(X3)), state

if __name__ == "__main__":
    """测试DecoderBlock"""
    X = T.ones((2, 100, 24))
    valid_lens = T.tensor([3, 2])
    state = [T.ones((2, 10, 24)), valid_lens, [None] * 6]
    decoder_blk = DecoderBlock(24, 24, 24, 8, 24, 48, [24], 0.5, 0)
    decoder_blk.eval()
    print(decoder_blk(X, state)[0].shape)
