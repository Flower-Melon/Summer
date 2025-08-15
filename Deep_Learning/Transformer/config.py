import torch as T

# 模型超参数
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
num_hiddens  = 32
num_layers   = 2
dropout      = 0.1
batch_size   = 64
num_steps    = 10

# 训练超参数
lr           = 0.005
num_epochs   = 200

# Transformer 中前馈网络 & 多头自注意力超参数
ffn_num_hiddens = 64
num_heads       = 4

# Query/Key/Value 维度
key_size     = 32
query_size   = 32
value_size   = 32

# LayerNorm 形状
norm_shape   = 32

# 数据集路径
save_path    = 'saved/Transformer.pth'
