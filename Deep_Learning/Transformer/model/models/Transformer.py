from model.models.Decoder import Decoder
from model.models.Encoder import Encoder
from torch import nn

class Transformer(nn.Module):
    """Transformer模型"""
    def  __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder  
    
    def forward(self, enc_X, dec_X, enc_valid_lens):
        # enc_X的形状为(batch_size, num_steps)
        # dec_X的形状为(batch_size, num_steps)
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        dec_state = self.decoder.initialize_state(enc_outputs, enc_valid_lens)
        return self.decoder(dec_X, dec_state)

if __name__ == "__main__":
    """测试Transformer"""
    import torch as T
    
    enc_X = T.ones((2, 100), dtype=T.long)
    dec_X = T.ones((2, 100), dtype=T.long)
    valid_lens = T.tensor([3, 2])
    
    encoder = Encoder(10, 24, 24, 24, 8, 24, 48, [24], 2, 0.5)
    decoder = Decoder(10, 24, 24, 24, 8, 24, 48, [24], 4, 0.5)
    
    transformer = Transformer(encoder, decoder)
    transformer.eval()
    
    print(transformer(enc_X, dec_X, valid_lens).shape)  # 输出形状应为(batch_size, num_steps, vocab_size)