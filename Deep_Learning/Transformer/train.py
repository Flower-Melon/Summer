from torch import nn
import torch as T
from matplotlib import pyplot as plt

from utils.MaskedSoftmaxCELoss import MaskedSoftmaxCELoss
from utils.DataLoader import load_data_nmt
from utils.GradClipping import grad_clipping

from model.models.Transformer import Transformer
from model.models.Encoder import Encoder
from model.models.Decoder import Decoder

from config import *

def train_seq2seq(net,data_dir,lr,num_epochs,tgt_vocab,device):
    """训练序列到序列模型"""
    
    # 权重初始化
    def xavier_init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 偏置初始化为零
        elif isinstance(m, nn.GRU):
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
                elif "bias" in param:
                    nn.init.zeros_(m._parameters[param])  # 偏置初始化为零
                    
    net.apply(xavier_init_weights)  # 初始化权重
    net.to(device)  # 将模型移动到指定设备
    optimizer = T.optim.Adam(net.parameters(), lr=lr)  # 优化器
    loss = MaskedSoftmaxCELoss()  # 损失函数
    epoch_losses = []    # 存放每个 epoch 的 loss
    
    net.train()  # 设置模型为训练模式
    epoch_losses = []  # 用于记录每个 epoch 的损失

    for epoch in range(num_epochs):
        total_loss = 0  # 初始化每个 epoch 的总损失
        num_batches = 0  # 记录当前 epoch 的批次数

        for batch in data_dir:
            optimizer.zero_grad()
            
            # 获取批量数据并将其移动到指定设备
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = T.tensor([tgt_vocab['<bos>']] * Y.shape[0], 
                            device=device).reshape(-1, 1)
            dec_X = T.cat((bos, Y[:, :-1]), dim=1)
            
            Y_hat = net(X, dec_X, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clipping(net, 1)
            optimizer.step()
            
            # 累加损失
            total_loss += l.sum().item()  # 使用 .item() 获取标量值
            num_batches += Y_valid_len.sum().item()  # 增加批次数

        # 计算当前 epoch 的平均损失
        average_loss = total_loss / num_batches
        epoch_losses.append(average_loss)

        # 每 20 个 epoch 打印一次损失
        if (epoch + 1) % 20 == 0:
            print("epoch:", epoch + 1, "loss:", average_loss)
         # 训练完毕后画图
    plt.figure(figsize=(6,4))
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    """训练 Transformer 模型"""
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Encoder(len(tgt_vocab), query_size, key_size, value_size, num_heads,
                      num_hiddens, ffn_num_hiddens, norm_shape, num_layers, dropout)
    decoder = Decoder(len(tgt_vocab), query_size, key_size, value_size, num_heads,
                      num_hiddens, ffn_num_hiddens, norm_shape, num_layers, dropout)
    net = Transformer(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)