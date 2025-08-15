import torch 

from utils.DataLoader import truncate_pad
from utils.DataLoader import load_data_nmt
from utils.bleu import bleu

from model.models.Transformer import Transformer
from model.models.Encoder import Encoder
from model.models.Decoder import Decoder

from config import *

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    
    # 增广，满足编码器的输入要求（批量大小为1）
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.initialize_state(enc_outputs, enc_valid_len)
    
    # 添加'批量'轴，以<bos>作为解码器的初始输入
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq = []
    
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        #print("解码器输出的形状:", Y.shape)
        dec_X = Y.argmax(dim=2)
        # pred的形状为(batch_size, 1)，我们需要将其转换为标量
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
        
    return ' '.join(tgt_vocab.to_tokens(output_seq))

if __name__ == "__main__":
    """测试预测函数"""
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .',]
    
    _, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Encoder(len(tgt_vocab), query_size, key_size, value_size, num_heads,
                      num_hiddens, ffn_num_hiddens, norm_shape, num_layers, dropout)
    decoder = Decoder(len(tgt_vocab), query_size, key_size, value_size, num_heads,
                      num_hiddens, ffn_num_hiddens, norm_shape, num_layers, dropout)
    net = Transformer(encoder, decoder)
    # 加载训练模型
    net.load_state_dict(torch.load(save_path))
    net.to(device)

    for eng, fra in zip(engs, fras):
        translation = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, ',f'bleu {bleu(translation, fra, k=2):.3f}')