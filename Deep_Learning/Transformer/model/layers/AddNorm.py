from torch import nn

class AddNorm(nn.Module):
    """残差连接和层规范化"""
    def __init__(self,normalized_shape,dropout=0.0):
        super(AddNorm,self).__init__()
        self.ln = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout)
    def forward(self,X,Y):
        # X,Y的形状为(batch_size, num_steps, num_hiddens)
        return self.ln(X + self.dropout(Y))

if __name__ == "__main__":
    """测试Addnorm"""
    import torch as T
    add_norm = AddNorm(4, 0.5)
    add_norm.eval()
    print(add_norm(T.ones((2, 3, 4)), T.ones((2, 3, 4))).shape)