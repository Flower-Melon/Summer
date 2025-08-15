from torch import nn
import torch as T
from model.layers.DotProductAttention import sequence_mask

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = T.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        
        # 设置为非聚合，方便后续舍弃没用的损失
        self.reduction='none'
        
        # permute(0, 2, 1) 将 pred 的维度重排为 (batch_size, vocab_size, num_steps)
        # label的形状为 (batch_size, num_steps)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # unweighted_loss 的形状为 (batch_size, num_steps)，
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        # 返回的 weighted_loss 的形状为 (batch_size,)
        return weighted_loss

if __name__ == "__main__":
    """测试MaskedSoftmaxCELoss"""
    pred = T.ones((2, 10, 5))  # 假设有2个样本，10个时间步，5个词汇
    label = T.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                      [1, 2, 3, 4, 0, 1, 2, 3, 4, 0]])
    valid_len = T.tensor([10, 8])  # 第一个样本有效长度为10，第二个样本有效长度为8

    loss_fn = MaskedSoftmaxCELoss()
    loss = loss_fn(pred, label, valid_len)
    print(loss)  # 输出每个样本的损失值